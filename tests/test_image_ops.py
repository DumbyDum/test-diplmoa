from __future__ import annotations

import unittest

import numpy as np

from omniguard.image_ops import ensure_rgb_uint8


class ImageOpsTests(unittest.TestCase):
    def test_rgba_is_converted_to_rgb(self) -> None:
        rgba = np.zeros((4, 4, 4), dtype=np.uint8)
        rgba[..., 3] = 255
        rgb = ensure_rgb_uint8(rgba)
        self.assertEqual(rgb.shape, (4, 4, 3))

    def test_grayscale_is_converted_to_rgb(self) -> None:
        gray = np.zeros((4, 4), dtype=np.uint8)
        rgb = ensure_rgb_uint8(gray)
        self.assertEqual(rgb.shape, (4, 4, 3))


if __name__ == "__main__":
    unittest.main()
