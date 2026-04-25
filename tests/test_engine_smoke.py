from __future__ import annotations

import os
from pathlib import Path
import unittest

from omniguard import OmniGuardEngine


@unittest.skipUnless(
    os.getenv("OMNIGUARD_RUN_SLOW_TESTS", "0") == "1",
    "Slow smoke test is disabled by default.",
)
class EngineSmokeTests(unittest.TestCase):
    def test_engine_smoke_roundtrip(self) -> None:
        engine = OmniGuardEngine()
        protection = engine.protect_image(Path("examples/0000.png"), "pytest-smoke-001")
        analysis = engine.analyze_image(
            protection.protected_image,
            expected_document_id="pytest-smoke-001",
            reference_bits=protection.payload.encoded_bits,
        )
        self.assertTrue(analysis.payload.auth_ok)
        self.assertTrue(analysis.payload.document_match)
        self.assertGreaterEqual(analysis.tamper_score_max, analysis.tamper_score_mean)


if __name__ == "__main__":
    unittest.main()
