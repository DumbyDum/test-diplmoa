from __future__ import annotations

import numpy as np

from omniguard.image_ops import ensure_rgb_uint8
from omniguard.legacy_models import LegacyModelBundle
from omniguard.settings import RuntimeSettings


_LEGACY_BUNDLE: LegacyModelBundle | None = None


def _bundle() -> LegacyModelBundle:
    global _LEGACY_BUNDLE
    if _LEGACY_BUNDLE is None:
        _LEGACY_BUNDLE = LegacyModelBundle(RuntimeSettings())
    return _LEGACY_BUNDLE


def EditGuard_Hide(net, cover):
    """Backward-compatible wrapper around the new tamper watermark encoder."""
    if hasattr(cover, "detach"):
        image = cover.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
        image = np.clip(image, 0.0, 1.0)
        image = (image * 255.0).astype(np.uint8)
    else:
        image = ensure_rgb_uint8(cover)
    return _bundle().embed_tamper_watermark(image)


def EditGuard_Reveal(net, steg):
    """Backward-compatible wrapper around the new tamper localization path."""
    if hasattr(steg, "detach"):
        image = steg.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
        image = np.clip(image, 0.0, 1.0)
        image = (image * 255.0).astype(np.uint8)
    else:
        image = ensure_rgb_uint8(steg)
    mask = _bundle().reveal_tamper_mask(image, _bundle().settings.tamper_mask_scales)
    return np.expand_dims(np.expand_dims(mask, 0), 0)
