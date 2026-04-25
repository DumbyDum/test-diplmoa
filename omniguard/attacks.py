from __future__ import annotations

from dataclasses import dataclass
import random

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from .image_ops import ensure_mask_uint8, ensure_rgb_uint8


@dataclass(slots=True)
class AttackOutput:
    image: np.ndarray
    mask: np.ndarray | None = None


def identity(image: np.ndarray) -> AttackOutput:
    return AttackOutput(image=ensure_rgb_uint8(image))


def jpeg_roundtrip(image: np.ndarray, quality: int = 70) -> AttackOutput:
    image = ensure_rgb_uint8(image)
    encoded = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), quality])[1]
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return AttackOutput(image=cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB))


def gaussian_blur(image: np.ndarray, radius: float = 1.5) -> AttackOutput:
    image = Image.fromarray(ensure_rgb_uint8(image))
    return AttackOutput(image=np.array(image.filter(ImageFilter.GaussianBlur(radius=radius))))


def resize_roundtrip(image: np.ndarray, scale: float = 0.65) -> AttackOutput:
    image = Image.fromarray(ensure_rgb_uint8(image))
    width, height = image.size
    resized = image.resize(
        (max(1, int(width * scale)), max(1, int(height * scale))),
        Image.Resampling.BILINEAR,
    )
    restored = resized.resize((width, height), Image.Resampling.BILINEAR)
    return AttackOutput(image=np.array(restored))


def adjust_brightness(image: np.ndarray, factor: float = 1.15) -> AttackOutput:
    image = Image.fromarray(ensure_rgb_uint8(image))
    return AttackOutput(image=np.array(ImageEnhance.Brightness(image).enhance(factor)))


def copy_move(image: np.ndarray, seed: int = 13) -> AttackOutput:
    rng = random.Random(seed)
    source = ensure_rgb_uint8(image).copy()
    height, width = source.shape[:2]
    box_h = max(16, height // 6)
    box_w = max(16, width // 6)
    src_y = rng.randint(0, max(0, height - box_h))
    src_x = rng.randint(0, max(0, width - box_w))
    dst_y = rng.randint(0, max(0, height - box_h))
    dst_x = rng.randint(0, max(0, width - box_w))
    patch = source[src_y : src_y + box_h, src_x : src_x + box_w].copy()
    source[dst_y : dst_y + box_h, dst_x : dst_x + box_w] = patch
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[dst_y : dst_y + box_h, dst_x : dst_x + box_w] = 255
    return AttackOutput(image=source, mask=mask)


def rectangular_inpaint(image: np.ndarray, seed: int = 7) -> AttackOutput:
    rng = random.Random(seed)
    source = ensure_rgb_uint8(image)
    height, width = source.shape[:2]
    box_h = max(24, height // 5)
    box_w = max(24, width // 5)
    y = rng.randint(0, max(0, height - box_h))
    x = rng.randint(0, max(0, width - box_w))
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y : y + box_h, x : x + box_w] = 255
    edited = cv2.inpaint(cv2.cvtColor(source, cv2.COLOR_RGB2BGR), mask, 3, cv2.INPAINT_TELEA)
    return AttackOutput(image=cv2.cvtColor(edited, cv2.COLOR_BGR2RGB), mask=mask)


def masked_edit(image: np.ndarray, mask: np.ndarray, method: str = "inpaint") -> AttackOutput:
    source = ensure_rgb_uint8(image)
    mask_uint8 = ensure_mask_uint8(mask)
    if method == "zero":
        edited = source.copy()
        edited[mask_uint8 > 0] = 0
        return AttackOutput(image=edited, mask=mask_uint8)
    edited = cv2.inpaint(cv2.cvtColor(source, cv2.COLOR_RGB2BGR), mask_uint8, 3, cv2.INPAINT_TELEA)
    return AttackOutput(image=cv2.cvtColor(edited, cv2.COLOR_BGR2RGB), mask=mask_uint8)
