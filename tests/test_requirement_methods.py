from __future__ import annotations

import unittest

import numpy as np

from omniguard.attacks import REQUIREMENT_ATTACKS
from omniguard.basic_watermarking import DCTWatermarkMethod, LSBWatermarkMethod, bits_to_text
from omniguard.metrics import ber


class RequirementMethodsTests(unittest.TestCase):
    def test_lsb_roundtrip(self) -> None:
        image = np.zeros((32, 32, 3), dtype=np.uint8) + 128
        method = LSBWatermarkMethod()
        embedded = method.embed(image, "demo")
        extracted = method.extract(
            embedded.watermarked_image,
            len(embedded.payload_bits),
            embedded.metadata,
        )
        self.assertEqual(ber(embedded.payload_bits, extracted.bits), 0.0)
        self.assertEqual(bits_to_text(extracted.bits), "demo")

    def test_dct_roundtrip(self) -> None:
        image = np.zeros((64, 64, 3), dtype=np.uint8) + 128
        method = DCTWatermarkMethod()
        embedded = method.embed(image, "demo")
        extracted = method.extract(
            embedded.watermarked_image,
            len(embedded.payload_bits),
            embedded.metadata,
        )
        score = ber(embedded.payload_bits, extracted.bits)
        self.assertIsNotNone(score)
        self.assertLessEqual(score, 0.25)

    def test_requirement_attacks_keep_shape(self) -> None:
        image = np.zeros((48, 64, 3), dtype=np.uint8) + 120
        for attack_name, _, attack_fn in REQUIREMENT_ATTACKS:
            with self.subTest(attack=attack_name):
                attacked = attack_fn(image)
                self.assertEqual(attacked.image.shape, image.shape)
                self.assertEqual(attacked.image.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
