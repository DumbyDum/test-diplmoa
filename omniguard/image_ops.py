from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


def load_image(image: str | Path | Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, np.ndarray):
        array = image
    elif isinstance(image, Image.Image):
        array = np.array(image.convert("RGB"))
    else:
        array = np.array(Image.open(image).convert("RGB"))
    return ensure_rgb_uint8(array)


def ensure_rgb_uint8(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    if array.ndim == 3 and array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    if array.ndim == 3 and array.shape[2] == 4:
        # Gradio and PNG inputs can arrive in RGBA; the analysis pipeline expects RGB.
        array = array[..., :3]
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("Expected an RGB image with shape HxWx3.")
    if array.dtype != np.uint8:
        if array.max() <= 1.0:
            array = np.clip(array, 0.0, 1.0) * 255.0
        array = np.clip(array, 0.0, 255.0).astype(np.uint8)
    return array


def ensure_mask_uint8(mask: np.ndarray | Image.Image) -> np.ndarray:
    if isinstance(mask, Image.Image):
        array = np.array(mask.convert("L"))
    else:
        array = np.asarray(mask)
    if array.ndim == 3:
        array = array[..., 0]
    if array.dtype != np.uint8:
        if array.max() <= 1.0:
            array = np.clip(array, 0.0, 1.0) * 255.0
        array = np.clip(array, 0.0, 255.0).astype(np.uint8)
    return array


def save_image(image: np.ndarray, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(ensure_rgb_uint8(image)).save(path)
    return path


def save_mask(mask: np.ndarray, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(ensure_mask_uint8(mask)).save(path)
    return path


def resize_image(image: np.ndarray, size: tuple[int, int], resample: int = Image.Resampling.LANCZOS) -> np.ndarray:
    return np.array(Image.fromarray(ensure_rgb_uint8(image)).resize(size, resample))


def resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    pil_mask = Image.fromarray(ensure_mask_uint8(mask), mode="L")
    return np.array(pil_mask.resize(size, Image.Resampling.NEAREST))


def limit_image_side(image: np.ndarray, max_side: int) -> np.ndarray:
    height, width = image.shape[:2]
    if max(height, width) <= max_side:
        return image
    scale = max_side / max(height, width)
    resized = Image.fromarray(ensure_rgb_uint8(image)).resize(
        (max(1, int(width * scale)), max(1, int(height * scale))),
        Image.Resampling.LANCZOS,
    )
    return np.array(resized)


def normalize_mask(mask: np.ndarray) -> np.ndarray:
    mask_uint8 = ensure_mask_uint8(mask)
    return mask_uint8.astype(np.float32) / 255.0


def overlay_mask(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int] = (255, 64, 64)) -> np.ndarray:
    base = ensure_rgb_uint8(image).astype(np.float32)
    alpha = normalize_mask(mask)[..., None]
    tint = np.array(color, dtype=np.float32)[None, None, :]
    mixed = base * (1.0 - alpha * 0.5) + tint * (alpha * 0.5)
    return np.clip(mixed, 0.0, 255.0).astype(np.uint8)


def ensure_multiple_of(value: int, multiple: int) -> int:
    return max(multiple, (value // multiple) * multiple)


def clamp_scales(scales: Iterable[float]) -> tuple[float, ...]:
    unique = []
    for scale in scales:
        if scale <= 0:
            continue
        if scale not in unique:
            unique.append(scale)
    return tuple(unique)
