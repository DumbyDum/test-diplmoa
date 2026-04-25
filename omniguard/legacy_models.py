from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

import modules.Unet_common as common
from model_invert import Model, init_model

from .image_ops import clamp_scales, ensure_rgb_uint8, resize_image
from .settings import RuntimeSettings


def _strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key.replace("module.", "", 1) if key.startswith("module.") else key
        cleaned[new_key] = value
    return cleaned


def _ensure_local_checkpoint(path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Checkpoint not found: {resolved}")
    return resolved


@dataclass(slots=True)
class LegacyModelBundle:
    settings: RuntimeSettings
    device: torch.device = field(init=False)
    dwt: common.DWT = field(init=False)
    iwt: common.IWT = field(init=False)
    _model: Model | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dwt = common.DWT()
        self.iwt = common.IWT()

    @property
    def model(self) -> Model:
        if self._model is None:
            model = Model().to(self.device)
            init_model(model)
            checkpoint_path = _ensure_local_checkpoint(self.settings.invert_checkpoint_path)
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            network_state_dict = {
                key: value
                for key, value in checkpoint["net"].items()
                if ("tmp_var" not in key) and ("bm" not in key)
            }
            model.load_state_dict(_strip_module_prefix(network_state_dict), strict=False)
            model.eval()
            self._model = model
        return self._model

    def _to_tensor01(self, image: np.ndarray) -> torch.Tensor:
        array = ensure_rgb_uint8(image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    def _tensor_to_uint8(self, tensor: torch.Tensor) -> np.ndarray:
        image = tensor.detach().cpu().clamp(0.0, 1.0).squeeze(0).permute(1, 2, 0).numpy()
        return np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8)

    def _prepare_secret_image(self, width: int, height: int) -> torch.Tensor:
        secret_image = Image.open(self.settings.secret_image_path).convert("RGB").resize(
            (width, height),
            Image.Resampling.LANCZOS,
        )
        array = np.array(secret_image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    def embed_tamper_watermark(self, image: np.ndarray) -> np.ndarray:
        cover = self._to_tensor01(image)
        _, _, height, width = cover.shape
        secret = self._prepare_secret_image(width, height)
        cover_input = self.dwt(cover)
        secret_input = self.dwt(secret)
        with torch.inference_mode():
            steg_img, _, _, _ = self.model(cover_input, secret_input)
        return self._tensor_to_uint8(steg_img)

    def embed_payload_bits(self, image: np.ndarray, bits: list[int], strength: float) -> np.ndarray:
        if len(bits) != self.settings.payload_bit_length:
            raise ValueError(
                f"Payload bit stream must contain {self.settings.payload_bit_length} bits."
            )
        source = ensure_rgb_uint8(image)
        height, width = source.shape[:2]
        payload_image = resize_image(source, (self.settings.payload_resolution, self.settings.payload_resolution))
        source_small = self._to_tensor01(payload_image)
        source_full = self._to_tensor01(source)
        secret = torch.tensor(bits, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.inference_mode():
            stego_small = self.model.bm.encoder(source_small * 2.0 - 1.0, secret)
        stego_small = ((stego_small.clamp(-1.0, 1.0) + 1.0) / 2.0).clamp(0.0, 1.0)
        residual_small = stego_small - source_small
        residual_full = F.interpolate(
            residual_small,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        stego_full = (source_full + residual_full * strength).clamp(0.0, 1.0)
        return self._tensor_to_uint8(stego_full)

    def decode_payload_bits(self, image: np.ndarray) -> list[int]:
        source = ensure_rgb_uint8(image)
        payload_image = resize_image(source, (self.settings.payload_resolution, self.settings.payload_resolution))
        tensor = self._to_tensor01(payload_image) * 2.0 - 1.0
        with torch.inference_mode():
            logits = self.model.bm.decoder(tensor)
        return (logits > 0).int().detach().cpu().flatten().tolist()

    def _downsample_until_limit(self, tensor: torch.Tensor) -> torch.Tensor:
        _, _, height, width = tensor.shape
        while max(height, width) > self.settings.tamper_mask_input_size:
            tensor = F.interpolate(tensor, scale_factor=0.5, mode="bilinear", align_corners=False)
            _, _, height, width = tensor.shape
        return tensor

    def _predict_single_tamper_mask(self, image: np.ndarray) -> np.ndarray:
        source = self._to_tensor01(image)
        original_height, original_width = image.shape[:2]
        with torch.inference_mode():
            steg_input = self.dwt(source)
            output_image = self.model(steg_input, rev=True)
            secret_rev = output_image.narrow(1, 0, 12)
            secret_rev = self.iwt(secret_rev)
            artifact = self._downsample_until_limit(secret_rev)
            fuse = self._downsample_until_limit(source)
            reference_secret = self._downsample_until_limit(
                self._prepare_secret_image(original_width, original_height)
            )
            residual_mask = torch.mean(torch.abs(artifact - reference_secret), dim=1, keepdim=True)
            residual_mask = F.interpolate(
                residual_mask,
                size=(original_height, original_width),
                mode="bilinear",
                align_corners=False,
            )
        return residual_mask.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)

    def reveal_tamper_mask(self, image: np.ndarray, scales: tuple[float, ...]) -> np.ndarray:
        source = ensure_rgb_uint8(image)
        height, width = source.shape[:2]
        masks: list[np.ndarray] = []
        for scale in clamp_scales(scales):
            if scale == 1.0:
                scaled = source
            else:
                scaled = resize_image(
                    source,
                    (max(64, int(width * scale)), max(64, int(height * scale))),
                )
            mask = self._predict_single_tamper_mask(scaled)
            if mask.shape[:2] != (height, width):
                mask = np.array(
                    Image.fromarray(mask, mode="F").resize((width, height), Image.Resampling.BILINEAR)
                )
            masks.append(mask)
        if not masks:
            return np.zeros((height, width), dtype=np.float32)
        stacked = np.stack(masks, axis=0)
        return np.clip(np.mean(stacked, axis=0), 0.0, 1.0)
