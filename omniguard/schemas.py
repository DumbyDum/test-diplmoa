from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _serialize(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, tuple):
        return [_serialize(item) for item in value]
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize(item) for key, item in value.items()}
    return value


@dataclass(slots=True)
class PayloadRecord:
    version: int
    issued_at_utc: datetime
    document_hash_hex: str
    nonce: int
    auth_tag_hex: str

    def to_dict(self) -> dict[str, Any]:
        return _serialize(asdict(self))


@dataclass(slots=True)
class PayloadEncodeResult:
    record: PayloadRecord
    encoded_bits: list[int]
    raw_bits: list[int]

    def to_dict(self) -> dict[str, Any]:
        return _serialize(asdict(self))


@dataclass(slots=True)
class PayloadDecodeResult:
    record: PayloadRecord | None
    decoded_bits: list[int]
    corrected_errors: int
    bit_accuracy: float | None = None
    auth_ok: bool = False
    document_match: bool | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _serialize(asdict(self))


@dataclass(slots=True)
class ProtectionResult:
    protected_image_path: Path | None
    protected_image: Any
    payload: PayloadEncodeResult
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload_dict = self.payload.to_dict()
        data = {
            "protected_image_path": self.protected_image_path,
            "payload": payload_dict,
            "metadata": self.metadata,
        }
        return _serialize(data)


@dataclass(slots=True)
class AnalysisResult:
    payload: PayloadDecodeResult
    tamper_heatmap_path: Path | None
    binary_mask_path: Path | None
    tamper_heatmap: Any
    binary_mask: Any
    tamper_score_mean: float
    tamper_score_max: float
    tamper_ratio: float
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return _serialize(
            {
                "payload": self.payload.to_dict(),
                "tamper_heatmap_path": self.tamper_heatmap_path,
                "binary_mask_path": self.binary_mask_path,
                "tamper_score_mean": self.tamper_score_mean,
                "tamper_score_max": self.tamper_score_max,
                "tamper_ratio": self.tamper_ratio,
                "metadata": self.metadata,
            }
        )


@dataclass(slots=True)
class AttackResult:
    attack_name: str
    attacked_image_path: Path
    predicted_mask_path: Path
    metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return _serialize(asdict(self))


@dataclass(slots=True)
class DatasetSampleRecord:
    source_path: Path
    document_id: str
    attack_name: str
    original_path: Path
    protected_path: Path
    edited_path: Path
    mask_path: Path

    def to_dict(self) -> dict[str, Any]:
        return _serialize(asdict(self))
