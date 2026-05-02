from __future__ import annotations

import math

import numpy as np

from .image_ops import ensure_mask_uint8, ensure_rgb_uint8

try:
    from skimage.metrics import structural_similarity as _skimage_ssim
except Exception:  # pragma: no cover - optional dependency
    _skimage_ssim = None


def mse(image_a: np.ndarray, image_b: np.ndarray) -> float:
    image_a = ensure_rgb_uint8(image_a).astype(np.float32)
    image_b = ensure_rgb_uint8(image_b).astype(np.float32)
    return float(np.mean((image_a - image_b) ** 2))


def mae(image_a: np.ndarray, image_b: np.ndarray) -> float:
    image_a = ensure_rgb_uint8(image_a).astype(np.float32)
    image_b = ensure_rgb_uint8(image_b).astype(np.float32)
    return float(np.mean(np.abs(image_a - image_b)))


def rmse(image_a: np.ndarray, image_b: np.ndarray) -> float:
    return float(math.sqrt(mse(image_a, image_b)))


def psnr(image_a: np.ndarray, image_b: np.ndarray) -> float:
    error = mse(image_a, image_b)
    if error <= 1.0e-10:
        return 100.0
    return float(10.0 * math.log10((255.0**2) / error))


def _global_ssim_fallback(image_a: np.ndarray, image_b: np.ndarray) -> float:
    image_a = image_a.astype(np.float64)
    image_b = image_b.astype(np.float64)
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    scores: list[float] = []
    for channel in range(3):
        x = image_a[..., channel]
        y = image_b[..., channel]
        mu_x = float(x.mean())
        mu_y = float(y.mean())
        sigma_x = float(((x - mu_x) ** 2).mean())
        sigma_y = float(((y - mu_y) ** 2).mean())
        sigma_xy = float(((x - mu_x) * (y - mu_y)).mean())
        numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
        denominator = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
        scores.append(numerator / denominator if denominator != 0 else 1.0)
    return float(np.clip(np.mean(scores), -1.0, 1.0))


def ssim(image_a: np.ndarray, image_b: np.ndarray) -> float | None:
    image_a = ensure_rgb_uint8(image_a)
    image_b = ensure_rgb_uint8(image_b)
    if _skimage_ssim is not None and min(image_a.shape[:2]) >= 7 and min(image_b.shape[:2]) >= 7:
        try:
            return float(_skimage_ssim(image_a, image_b, channel_axis=2, data_range=255))
        except ValueError:
            pass
    return _global_ssim_fallback(image_a, image_b)


def bpp(payload_bits: int, image: np.ndarray) -> float:
    """Bits per pixel: how many watermark bits are embedded per image pixel."""
    source = ensure_rgb_uint8(image)
    height, width = source.shape[:2]
    if height == 0 or width == 0:
        return 0.0
    return float(payload_bits / (height * width))


def ber(reference_bits: list[int], extracted_bits: list[int]) -> float | None:
    """Bit error rate: fraction of watermark bits changed after an attack."""
    if not reference_bits or not extracted_bits:
        return None
    shared = min(len(reference_bits), len(extracted_bits))
    errors = sum(
        int(reference_bits[index] != extracted_bits[index])
        for index in range(shared)
    )
    missing = abs(len(reference_bits) - len(extracted_bits))
    return float((errors + missing) / max(len(reference_bits), len(extracted_bits)))


def changed_pixel_ratio(image_a: np.ndarray, image_b: np.ndarray, threshold: int = 12) -> float:
    image_a = ensure_rgb_uint8(image_a).astype(np.int16)
    image_b = ensure_rgb_uint8(image_b).astype(np.int16)
    diff = np.abs(image_a - image_b).max(axis=2)
    return float((diff >= threshold).mean())


def _binary_masks(mask_a: np.ndarray, mask_b: np.ndarray, threshold: int = 127) -> tuple[np.ndarray, np.ndarray]:
    a = ensure_mask_uint8(mask_a) > threshold
    b = ensure_mask_uint8(mask_b) > threshold
    return a, b


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray, threshold: int = 127) -> float:
    a, b = _binary_masks(mask_a, mask_b, threshold=threshold)
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


def mask_dice(mask_a: np.ndarray, mask_b: np.ndarray, threshold: int = 127) -> float:
    a, b = _binary_masks(mask_a, mask_b, threshold=threshold)
    size = a.sum() + b.sum()
    if size == 0:
        return 1.0
    intersection = np.logical_and(a, b).sum()
    return float((2.0 * intersection) / size)


def mask_precision(mask_true: np.ndarray, mask_pred: np.ndarray, threshold: int = 127) -> float:
    true_mask, pred_mask = _binary_masks(mask_true, mask_pred, threshold=threshold)
    tp = np.logical_and(true_mask, pred_mask).sum()
    fp = np.logical_and(~true_mask, pred_mask).sum()
    denom = tp + fp
    if denom == 0:
        return 1.0
    return float(tp / denom)


def mask_recall(mask_true: np.ndarray, mask_pred: np.ndarray, threshold: int = 127) -> float:
    true_mask, pred_mask = _binary_masks(mask_true, mask_pred, threshold=threshold)
    tp = np.logical_and(true_mask, pred_mask).sum()
    fn = np.logical_and(true_mask, ~pred_mask).sum()
    denom = tp + fn
    if denom == 0:
        return 1.0
    return float(tp / denom)


def mask_f1(mask_true: np.ndarray, mask_pred: np.ndarray, threshold: int = 127) -> float:
    precision = mask_precision(mask_true, mask_pred, threshold=threshold)
    recall = mask_recall(mask_true, mask_pred, threshold=threshold)
    denom = precision + recall
    if denom <= 1.0e-10:
        return 0.0
    return float((2.0 * precision * recall) / denom)


def bit_accuracy(bits_a: list[int], bits_b: list[int]) -> float | None:
    if not bits_a or not bits_b:
        return None
    shared = min(len(bits_a), len(bits_b))
    matches = sum(int(bits_a[index] == bits_b[index]) for index in range(shared))
    return matches / shared
