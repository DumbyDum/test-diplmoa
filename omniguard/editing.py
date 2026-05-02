from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
from PIL import Image

from .image_ops import ensure_mask_uint8, ensure_rgb_uint8
from .settings import RuntimeSettings

try:
    from diffusers import DPMSolverMultistepScheduler
except Exception:  # pragma: no cover - optional dependency
    DPMSolverMultistepScheduler = None

try:
    from diffusers import AutoPipelineForInpainting
except Exception:  # pragma: no cover - optional dependency
    AutoPipelineForInpainting = None

try:
    from diffusers import StableDiffusionInpaintPipeline
except Exception:  # pragma: no cover - optional dependency
    StableDiffusionInpaintPipeline = None


EDITOR_LABELS: dict[str, str] = {
    "auto": "Автовыбор",
    "opencv-telea": "OpenCV Telea",
    "opencv-ns": "OpenCV Navier-Stokes",
    "diffusers-fast": "Diffusers Fast",
    "diffusers-quality": "Diffusers Quality",
}


EDITOR_DESCRIPTIONS: dict[str, str] = {
    "auto": "Система сама выберет лучший доступный backend: сначала диффузионный, затем OpenCV fallback.",
    "opencv-telea": "Быстрый локальный редактор. Хорош для smoke-test и грубого удаления объекта.",
    "opencv-ns": "Локальный редактор с более мягким заполнением области. Иногда выглядит аккуратнее на плавных текстурах.",
    "diffusers-fast": "Генеративный inpainting с упором на скорость. Нужна доступная локально diffusers-модель или разрешение на скачивание.",
    "diffusers-quality": "Генеративный inpainting с упором на качество. Медленнее, но обычно дает более естественное заполнение выделенной области.",
}


@dataclass(slots=True)
class EditingResult:
    image: np.ndarray
    backend_name: str
    prompt: str
    backend_label: str
    model_id: str | None = None
    num_inference_steps: int | None = None
    guidance_scale: float | None = None
    allow_download: bool = False


class BaseEditor:
    backend_name = "base"
    backend_label = "Base editor"

    def edit(self, image: np.ndarray, mask: np.ndarray, prompt: str) -> EditingResult:
        raise NotImplementedError


class OpenCVInpaintEditor(BaseEditor):
    def __init__(self, *, backend_name: str, backend_label: str, method: int, radius: float) -> None:
        self.backend_name = backend_name
        self.backend_label = backend_label
        self.method = method
        self.radius = radius

    def edit(self, image: np.ndarray, mask: np.ndarray, prompt: str) -> EditingResult:
        image = ensure_rgb_uint8(image)
        mask = ensure_mask_uint8(mask)
        edited = cv2.inpaint(
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
            mask,
            self.radius,
            self.method,
        )
        return EditingResult(
            image=cv2.cvtColor(edited, cv2.COLOR_BGR2RGB),
            backend_name=self.backend_name,
            backend_label=self.backend_label,
            prompt=prompt,
        )


class DiffusersInpaintEditor(BaseEditor):
    def __init__(
        self,
        settings: RuntimeSettings,
        *,
        backend_name: str,
        backend_label: str,
        model_id: str,
        allow_download: bool,
        num_inference_steps: int,
        guidance_scale: float,
        max_side: int,
    ) -> None:
        if StableDiffusionInpaintPipeline is None and AutoPipelineForInpainting is None:
            raise RuntimeError("Diffusers недоступен в текущем окружении.")
        self.backend_name = backend_name
        self.backend_label = backend_label
        self.model_id = model_id
        self.allow_download = allow_download
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.max_side = max_side

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        pipe = self._load_pipeline(model_id, dtype=dtype, allow_download=allow_download)
        if DPMSolverMultistepScheduler is not None and getattr(pipe, "scheduler", None) is not None:
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(device)
        if device == "cpu" and hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
        self.device = device
        self.pipe = pipe

    def _load_pipeline(self, model_id: str, *, dtype: torch.dtype, allow_download: bool):
        local_files_only = not allow_download
        if AutoPipelineForInpainting is not None:
            return AutoPipelineForInpainting.from_pretrained(
                model_id,
                torch_dtype=dtype,
                local_files_only=local_files_only,
            )
        return StableDiffusionInpaintPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            local_files_only=local_files_only,
        )

    def edit(self, image: np.ndarray, mask: np.ndarray, prompt: str) -> EditingResult:
        image = ensure_rgb_uint8(image)
        mask = ensure_mask_uint8(mask)
        height, width = image.shape[:2]
        scale = min(1.0, self.max_side / max(height, width))
        work_w = max(8, (int(width * scale) // 8) * 8)
        work_h = max(8, (int(height * scale) // 8) * 8)
        prompt = prompt.strip() or "remove the selected object and fill the area naturally"
        image_init = Image.fromarray(image, mode="RGB").resize((work_w, work_h), Image.Resampling.LANCZOS)
        mask_init = Image.fromarray(mask, mode="L").resize((work_w, work_h), Image.Resampling.NEAREST)
        result = self.pipe(
            prompt=prompt,
            image=image_init,
            mask_image=mask_init,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
        ).images[0]
        edited = np.array(result.resize((width, height), Image.Resampling.LANCZOS))
        return EditingResult(
            image=edited,
            backend_name=self.backend_name,
            backend_label=self.backend_label,
            prompt=prompt,
            model_id=self.model_id,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            allow_download=self.allow_download,
        )


def editor_choices() -> list[tuple[str, str]]:
    return [
        ("Автовыбор: лучший доступный редактор", "auto"),
        ("OpenCV Telea: быстрый локальный", "opencv-telea"),
        ("OpenCV Navier-Stokes: более мягкое заполнение", "opencv-ns"),
        ("Diffusers Fast: генеративный, быстрее", "diffusers-fast"),
        ("Diffusers Quality: генеративный, качественнее", "diffusers-quality"),
    ]


def editor_help_markdown() -> str:
    lines = ["### Редакторы изображения", ""]
    for key in ("auto", "opencv-telea", "opencv-ns", "diffusers-fast", "diffusers-quality"):
        lines.append(f"- **{EDITOR_LABELS[key]}**: {EDITOR_DESCRIPTIONS[key]}")
    return "\n".join(lines)


def build_editor(
    settings: RuntimeSettings,
    editor_name: str = "auto",
    *,
    model_id: str | None = None,
    allow_download: bool | None = None,
) -> BaseEditor:
    selected_model_id = (model_id or settings.inpaint_model_id).strip()
    resolved_allow_download = settings.allow_inpaint_model_download if allow_download is None else allow_download

    if editor_name == "auto":
        if settings.use_diffusers:
            try:
                return DiffusersInpaintEditor(
                    settings,
                    backend_name="diffusers-quality",
                    backend_label=EDITOR_LABELS["diffusers-quality"],
                    model_id=selected_model_id,
                    allow_download=resolved_allow_download,
                    num_inference_steps=28,
                    guidance_scale=7.5,
                    max_side=768,
                )
            except Exception:
                pass
        return OpenCVInpaintEditor(
            backend_name="opencv-telea",
            backend_label=EDITOR_LABELS["opencv-telea"],
            method=cv2.INPAINT_TELEA,
            radius=3.0,
        )

    if editor_name == "opencv-telea":
        return OpenCVInpaintEditor(
            backend_name="opencv-telea",
            backend_label=EDITOR_LABELS["opencv-telea"],
            method=cv2.INPAINT_TELEA,
            radius=3.0,
        )

    if editor_name == "opencv-ns":
        return OpenCVInpaintEditor(
            backend_name="opencv-ns",
            backend_label=EDITOR_LABELS["opencv-ns"],
            method=cv2.INPAINT_NS,
            radius=4.0,
        )

    if editor_name == "diffusers-fast":
        return DiffusersInpaintEditor(
            settings,
            backend_name="diffusers-fast",
            backend_label=EDITOR_LABELS["diffusers-fast"],
            model_id=selected_model_id,
            allow_download=resolved_allow_download,
            num_inference_steps=14,
            guidance_scale=6.5,
            max_side=640,
        )

    if editor_name == "diffusers-quality":
        return DiffusersInpaintEditor(
            settings,
            backend_name="diffusers-quality",
            backend_label=EDITOR_LABELS["diffusers-quality"],
            model_id=selected_model_id,
            allow_download=resolved_allow_download,
            num_inference_steps=28,
            guidance_scale=7.5,
            max_side=768,
        )

    raise ValueError(f"Неизвестный редактор: {editor_name}")
