from __future__ import annotations

from pathlib import Path

from omniguard import OmniGuardEngine


def main() -> int:
    engine = OmniGuardEngine()
    protection = engine.protect_image(Path("examples/0000.png"), "invert-smoke-001")
    analysis = engine.analyze_image(
        protection.protected_image,
        expected_document_id="invert-smoke-001",
        reference_bits=protection.payload.encoded_bits,
        output_dir=engine.settings.runtime_dir / "invert_smoke",
    )
    print("Payload auth ok:", analysis.payload.auth_ok)
    print("Payload document match:", analysis.payload.document_match)
    print("Tamper score max:", analysis.tamper_score_max)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
