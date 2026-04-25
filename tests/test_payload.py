from __future__ import annotations

import unittest

from omniguard.payload import PAYLOAD_TOTAL_BITS, build_payload_bits, decode_payload_bits


class PayloadTests(unittest.TestCase):
    def test_payload_roundtrip_without_errors(self) -> None:
        payload = build_payload_bits("doc-001", "secret-key")
        self.assertEqual(len(payload.encoded_bits), PAYLOAD_TOTAL_BITS)
        decoded = decode_payload_bits(
            payload.encoded_bits,
            "secret-key",
            expected_document_id="doc-001",
            reference_bits=payload.encoded_bits,
        )
        self.assertTrue(decoded.auth_ok)
        self.assertTrue(decoded.document_match)
        self.assertEqual(decoded.bit_accuracy, 1.0)
        self.assertIsNotNone(decoded.record)
        self.assertEqual(decoded.record.document_hash_hex, payload.record.document_hash_hex)

    def test_payload_roundtrip_with_single_bit_errors(self) -> None:
        payload = build_payload_bits("doc-002", "secret-key")
        noisy = payload.encoded_bits[:]
        for index in range(0, 98, 7):
            noisy[index] ^= 1
        decoded = decode_payload_bits(
            noisy,
            "secret-key",
            expected_document_id="doc-002",
            reference_bits=payload.encoded_bits,
        )
        self.assertTrue(decoded.auth_ok)
        self.assertTrue(decoded.document_match)
        self.assertEqual(decoded.corrected_errors, 14)


if __name__ == "__main__":
    unittest.main()
