from __future__ import annotations

import unittest

import numpy as np

from omniguard.metrics import (
    ber,
    bit_accuracy,
    bpp,
    changed_pixel_ratio,
    mae,
    mask_dice,
    mask_f1,
    mask_iou,
    mask_precision,
    mask_recall,
    mse,
    psnr,
    rmse,
    ssim,
)


class MetricsTests(unittest.TestCase):
    def test_basic_image_metrics(self) -> None:
        image_a = np.zeros((16, 16, 3), dtype=np.uint8)
        image_b = np.ones((16, 16, 3), dtype=np.uint8) * 10
        self.assertEqual(mse(image_a, image_a), 0.0)
        self.assertEqual(mae(image_a, image_a), 0.0)
        self.assertEqual(rmse(image_a, image_a), 0.0)
        self.assertEqual(psnr(image_a, image_a), 100.0)
        self.assertAlmostEqual(ssim(image_a, image_a), 1.0)
        self.assertGreater(mse(image_a, image_b), 0.0)
        self.assertLess(psnr(image_a, image_b), 100.0)
        self.assertGreater(changed_pixel_ratio(image_a, image_b, threshold=1), 0.0)
        self.assertEqual(bpp(128, image_a), 128 / (16 * 16))

    def test_mask_metrics(self) -> None:
        mask_a = np.zeros((8, 8), dtype=np.uint8)
        mask_b = np.zeros((8, 8), dtype=np.uint8)
        mask_a[2:4, 2:4] = 255
        mask_b[2:4, 2:4] = 255
        self.assertEqual(mask_iou(mask_a, mask_b), 1.0)
        self.assertEqual(mask_dice(mask_a, mask_b), 1.0)
        self.assertEqual(mask_precision(mask_a, mask_b), 1.0)
        self.assertEqual(mask_recall(mask_a, mask_b), 1.0)
        self.assertEqual(mask_f1(mask_a, mask_b), 1.0)

    def test_bit_accuracy(self) -> None:
        self.assertEqual(bit_accuracy([1, 0, 1], [1, 0, 1]), 1.0)
        self.assertEqual(bit_accuracy([1, 0, 1], [0, 0, 1]), 2 / 3)
        self.assertEqual(ber([1, 0, 1], [1, 0, 1]), 0.0)
        self.assertEqual(ber([1, 0, 1], [0, 0, 1]), 1 / 3)


if __name__ == "__main__":
    unittest.main()
