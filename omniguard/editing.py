from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
from PIL import Image

from .image_ops import ensure_mask_uint8, ensure_rgb_uint8
from .settings import RuntimeSettings

try:
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionInpaintPipeline
except Exception:  # pragma: no cover - optional dependency
    DPMSolverMultistepScheduler = None
    StableDiffusionInpaintPipeline = None


@dataclass(slots=True)
class EditingResult:
    image: np.ndarray
    backend_name: str
    prompt: str


class BaseEditor:
    backend_name = "base"

    def edit(self, image: np.ndarray, mask: np.ndarray, prompt: str) -> EditingResult:
        raise NotImplementedError


class OpenCVInpaintEditor(BaseEditor):
    backend_name = "opencv-inpaint"

    def edit(self, image: np.ndarray, mask: np.ndarray, prompt: str) -> EditingResult:
        image = ensure_rgb_uint8(image)
        mask = ensure_mask_uint8(mask)
        edited = cv2.inpaint(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), mask, 3, cv2.INPAINT_TELEA)
        return EditingResult(
            image=cv2.cvtColor(edited, cv2.COLOR_BGR2RGB),
            backend_name=self.backend_name,
            prompt=prompt,
        )


class DiffusersInpaintEditor(BaseEditor):
    backend_name = "diffusers-inpaint"

    def __init__(self, settings: RuntimeSettings):
        if StableDiffusionInpaintPipeline is None or DPMSolverMultistepScheduler is None:
            raise RuntimeError("Diffusers is not available in the current environment.")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            settings.inpaint_model_id,
            torch_dtype=dtype,
            local_files_only=True,
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(device)
        if device == "cpu":
            pipe.enable_attention_slicing()
        self.device = device
        self.pipe = pipe

    def edit(self, image: np.ndarray, mask: np.ndarray, prompt: str) -> EditingResult:
        image = ensure_rgb_uint8(image)
        mask = ensure_mask_uint8(mask)
        height, width = image.shape[:2]
        max_side = 512
        scale = min(1.0, max_side / max(height, width))
        work_w = max(8, (int(width * scale) // 8) * 8)
        work_h = max(8, (int(height * scale) // 8) * 8)
        prompt = prompt.strip() or "remove the selected object and fill the area naturally"
        image_init = Image.fromarray(image, mode="RGB").resize((work_w, work_h), Image.Resampling.LANCZOS)
        mask_init = Image.fromarray(mask, mode="L").resize((work_w, work_h), Image.Resampling.NEAREST)
        result = self.pipe(
            prompt=prompt,
            image=image_init,
            mask_image=mask_init,
            num_inference_steps=10,
            guidance_scale=7.0,
        ).images[0]
        edited = np.array(result.resize((width, height), Image.Resampling.LANCZOS))
        return EditingResult(image=edited, backend_name=self.backend_name, prompt=prompt)


def build_editor(settings: RuntimeSettings) -> BaseEditor:
    if settings.use_diffusers:
        try:
            return DiffusersInpaintEditor(settings)
        except Exception:
            pass
    return OpenCVInpaintEditor()
