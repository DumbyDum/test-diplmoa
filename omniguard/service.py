from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np

from .editing import EditingResult, build_editor
from .image_ops import (
    ensure_mask_uint8,
    load_image,
    limit_image_side,
    normalize_mask,
    overlay_mask,
    save_image,
    save_mask,
)
from .legacy_models import LegacyModelBundle
from .payload import build_payload_bits, decode_payload_bits
from .schemas import AnalysisResult, PayloadEncodeResult, ProtectionResult
from .settings import RuntimeSettings


class OmniGuardEngine:
    def __init__(self, settings: RuntimeSettings | None = None):
        self.settings = settings or RuntimeSettings()
        self.settings.ensure_runtime_dirs()
        self.models = LegacyModelBundle(self.settings)
        self._editor = None

    @property
    def editor(self):
        if self._editor is None:
            self._editor = build_editor(self.settings)
        return self._editor

    def protect_image(
        self,
        image: str | Path | np.ndarray,
        document_id: str,
        output_path: str | Path | None = None,
    ) -> ProtectionResult:
        source = limit_image_side(load_image(image), self.settings.max_input_side)
        payload = build_payload_bits(document_id, self.settings.hmac_secret)
        tamper_protected = self.models.embed_tamper_watermark(source)
        final_protected = self.models.embed_payload_bits(
            tamper_protected,
            payload.encoded_bits,
            strength=self.settings.payload_strength,
        )
        protected_path = None
        if output_path is not None:
            protected_path = save_image(final_protected, output_path)
        metadata = {
            "document_id": document_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "payload_bit_length": self.settings.payload_bit_length,
            "payload_strength": self.settings.payload_strength,
            "tamper_mask_scales": list(self.settings.tamper_mask_scales),
        }
        return ProtectionResult(
            protected_image_path=protected_path,
            protected_image=final_protected,
            payload=payload,
            metadata=metadata,
        )

    def edit_image(
        self,
        image: str | Path | np.ndarray,
        mask: np.ndarray,
        prompt: str = "",
    ) -> EditingResult:
        source = limit_image_side(load_image(image), self.settings.max_input_side)
        safe_mask = ensure_mask_uint8(mask)
        return self.editor.edit(source, safe_mask, prompt)

    def analyze_image(
        self,
        image: str | Path | np.ndarray,
        expected_document_id: str | None = None,
        reference_bits: list[int] | None = None,
        output_dir: str | Path | None = None,
    ) -> AnalysisResult:
        source = limit_image_side(load_image(image), self.settings.max_input_side)
        predicted_bits = self.models.decode_payload_bits(source)
        payload_result = decode_payload_bits(
            predicted_bits,
            self.settings.hmac_secret,
            expected_document_id=expected_document_id,
            reference_bits=reference_bits,
        )
        heatmap_float = self.models.reveal_tamper_mask(source, self.settings.tamper_mask_scales)
        heatmap_uint8 = np.clip(heatmap_float * 255.0, 0.0, 255.0).astype(np.uint8)
        binary_mask = (heatmap_float >= self.settings.tamper_mask_threshold).astype(np.uint8) * 255
        tamper_ratio = float(normalize_mask(binary_mask).mean())
        tamper_heatmap_path = None
        binary_mask_path = None
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            tamper_heatmap_path = save_image(overlay_mask(source, heatmap_uint8), output_dir / "tamper_heatmap.png")
            binary_mask_path = save_mask(binary_mask, output_dir / "tamper_mask.png")
        metadata = {
            "expected_document_id": expected_document_id,
            "threshold": self.settings.tamper_mask_threshold,
            "scales": list(self.settings.tamper_mask_scales),
        }
        return AnalysisResult(
            payload=payload_result,
            tamper_heatmap_path=tamper_heatmap_path,
            binary_mask_path=binary_mask_path,
            tamper_heatmap=heatmap_uint8,
            binary_mask=binary_mask,
            tamper_score_mean=float(heatmap_float.mean()),
            tamper_score_max=float(heatmap_float.max()),
            tamper_ratio=tamper_ratio,
            metadata=metadata,
        )

    def save_json(self, data: dict[str, Any], path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
        return path

    def save_protection_bundle(
        self,
        result: ProtectionResult,
        image_path: str | Path,
        metadata_path: str | Path,
    ) -> tuple[Path, Path]:
        saved_image = save_image(result.protected_image, image_path)
        saved_meta = self.save_json(result.to_dict(), metadata_path)
        result.protected_image_path = saved_image
        return saved_image, saved_meta

    def save_analysis_bundle(
        self,
        result: AnalysisResult,
        output_dir: str | Path,
    ) -> tuple[Path, Path, Path]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        heatmap_path = save_mask(result.tamper_heatmap, output_dir / "analysis_heatmap.png")
        mask_path = save_mask(result.binary_mask, output_dir / "analysis_mask.png")
        report_path = self.save_json(result.to_dict(), output_dir / "analysis_report.json")
        return heatmap_path, mask_path, report_path

    def protection_summary(self, result: ProtectionResult) -> dict[str, Any]:
        summary = result.to_dict()
        summary["payload"]["record"]["issued_at_utc"] = result.payload.record.issued_at_utc.isoformat()
        return summary

    def analysis_summary(self, result: AnalysisResult) -> dict[str, Any]:
        summary = result.to_dict()
        if result.payload.record is not None:
            summary["payload"]["record"]["issued_at_utc"] = result.payload.record.issued_at_utc.isoformat()
        return summary
