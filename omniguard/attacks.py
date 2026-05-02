from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Callable

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from .image_ops import ensure_mask_uint8, ensure_rgb_uint8


AttackCallable = Callable[[np.ndarray], "AttackOutput"]


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


def gaussian_noise(image: np.ndarray, mean: float = 0.0, sigma: float = 0.01, seed: int = 17) -> AttackOutput:
    source = ensure_rgb_uint8(image).astype(np.float32) / 255.0
    rng = np.random.default_rng(seed)
    noisy = source + rng.normal(mean, sigma, source.shape).astype(np.float32)
    return AttackOutput(image=np.clip(noisy * 255.0, 0.0, 255.0).astype(np.uint8))


def salt_and_pepper_noise(image: np.ndarray, amount: float = 0.02, seed: int = 23) -> AttackOutput:
    source = ensure_rgb_uint8(image).copy()
    rng = np.random.default_rng(seed)
    height, width = source.shape[:2]
    noise = rng.random((height, width))
    pepper = noise < (amount / 2.0)
    salt = noise > (1.0 - amount / 2.0)
    source[pepper] = 0
    source[salt] = 255
    return AttackOutput(image=source)


def random_crop_10(image: np.ndarray, seed: int = 29) -> AttackOutput:
    source = ensure_rgb_uint8(image)
    height, width = source.shape[:2]
    crop_height = max(1, int(height * 0.9))
    crop_width = max(1, int(width * 0.9))
    rng = random.Random(seed)
    top = rng.randint(0, max(0, height - crop_height))
    left = rng.randint(0, max(0, width - crop_width))
    cropped = source[top : top + crop_height, left : left + crop_width]
    restored = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)
    return AttackOutput(image=restored)


def add_brightness(image: np.ndarray, delta: int) -> AttackOutput:
    source = ensure_rgb_uint8(image).astype(np.int16)
    adjusted = np.clip(source + int(delta), 0, 255).astype(np.uint8)
    return AttackOutput(image=adjusted)


def median_filter(image: np.ndarray, kernel_size: int = 3) -> AttackOutput:
    source = ensure_rgb_uint8(image)
    filtered = cv2.medianBlur(source, kernel_size)
    return AttackOutput(image=filtered)


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


def downscale_upscale_2x(image: np.ndarray) -> AttackOutput:
    return resize_roundtrip(image, scale=0.5)


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


REQUIREMENT_ATTACKS: tuple[tuple[str, str, AttackCallable], ...] = (
    ("jpeg_q100", "JPEG-сжатие с качеством QF=100", lambda image: jpeg_roundtrip(image, quality=100)),
    ("jpeg_q90", "JPEG-сжатие с качеством QF=90", lambda image: jpeg_roundtrip(image, quality=90)),
    ("jpeg_q50", "JPEG-сжатие с качеством QF=50", lambda image: jpeg_roundtrip(image, quality=50)),
    ("gaussian_noise_0_01", "Gaussian noise: mean=0, sigma=0.01", gaussian_noise),
    ("salt_pepper", "Salt-and-Pepper: импульсный шум", salt_and_pepper_noise),
    ("random_crop_10", "Random crop: случайное кадрирование 10%", random_crop_10),
    ("brightness_plus_15", "Засветление: +15 к каждому каналу", lambda image: add_brightness(image, 15)),
    ("brightness_minus_15", "Затемнение: -15 от каждого канала", lambda image: add_brightness(image, -15)),
    ("brightness_mul_1_5", "Увеличение яркости: умножение на 1.5", lambda image: adjust_brightness(image, 1.5)),
    ("brightness_mul_0_8", "Уменьшение яркости: умножение на 0.8", lambda image: adjust_brightness(image, 0.8)),
    ("median_3x3", "Median filter: медианный фильтр 3x3", lambda image: median_filter(image, 3)),
    ("down_up_2x", "Уменьшение в 2 раза и обратное увеличение", downscale_upscale_2x),
)


ATTACK_REFERENCE_MARKDOWN = """
### Атаки из требований

1. **JPEG (QF=100, 90, 50).** Изображение сохраняется в JPEG и сразу читается обратно. Чем ниже QF, тем сильнее сжатие и тем больше потери высокочастотных деталей, где часто прячется ЦВЗ.
2. **Gaussian noise (mean=0, sigma=0.01).** К каждому пикселю добавляется случайный нормальный шум: `I' = clip(I + N(0, 0.01))`, если изображение нормировано в диапазон `[0, 1]`.
3. **Salt-and-Pepper.** Часть пикселей случайно заменяется на черный `0` или белый `255`. Это имитирует импульсные ошибки.
4. **Random crop (10%).** Случайно вырезается область размером `90% x 90%`, затем она растягивается обратно до исходного размера.
5. **Засветление/затемнение (+15/-15).** К каждому каналу добавляется `+15` или `-15`: `I' = clip(I +/- 15)`.
6. **Изменение яркости (*1.5, *0.8).** Каждый пиксель умножается на коэффициент: `I' = clip(I * alpha)`.
7. **Median filter (3x3).** Каждый пиксель заменяется медианой соседей в окне `3x3`, что подавляет шум и может разрушать слабый ЦВЗ.
8. **Уменьшение с возвращением.** Изображение уменьшается в 2 раза и увеличивается обратно; мелкие детали и часть скрытого сигнала теряются.
"""
