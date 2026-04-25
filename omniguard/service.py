from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .editing import EditingResult, build_editor
from .image_ops import (
    ensure_mask_uint8,
    load_image,
    limit_image_side,
    normalize_mask,
    overlay_mask,
    resize_image,
    save_image,
    save_mask,
)
from .legacy_models import LegacyModelBundle
from .metrics import changed_pixel_ratio, mae, mse, psnr, rmse, ssim
from .payload import build_payload_bits, decode_payload_bits
from .schemas import AnalysisResult, ProtectionResult
from .settings import RuntimeSettings


class OmniGuardEngine:
    def __init__(self, settings: RuntimeSettings | None = None):
        self.settings = settings or RuntimeSettings()
        self.settings.ensure_runtime_dirs()
        self.models = LegacyModelBundle(self.settings)
        self._editors: dict[tuple[str, str, bool], object] = {}

    def get_editor(
        self,
        editor_name: str = "auto",
        *,
        model_id: str | None = None,
        allow_download: bool | None = None,
    ):
        resolved_model_id = (model_id or self.settings.inpaint_model_id).strip()
        resolved_allow_download = (
            self.settings.allow_inpaint_model_download if allow_download is None else allow_download
        )
        cache_key = (editor_name, resolved_model_id, bool(resolved_allow_download))
        if cache_key not in self._editors:
            self._editors[cache_key] = build_editor(
                self.settings,
                editor_name=editor_name,
                model_id=resolved_model_id,
                allow_download=resolved_allow_download,
            )
        return self._editors[cache_key]

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
        *,
        editor_name: str = "auto",
        editor_model_id: str | None = None,
        allow_download: bool | None = None,
    ) -> EditingResult:
        source = limit_image_side(load_image(image), self.settings.max_input_side)
        safe_mask = ensure_mask_uint8(mask)
        editor = self.get_editor(
            editor_name=editor_name,
            model_id=editor_model_id,
            allow_download=allow_download,
        )
        return editor.edit(source, safe_mask, prompt)

    def _align_reference_image(
        self,
        reference_image: str | Path | np.ndarray | None,
        target_shape: tuple[int, int],
    ) -> np.ndarray | None:
        if reference_image is None:
            return None
        reference = limit_image_side(load_image(reference_image), self.settings.max_input_side)
        target_height, target_width = target_shape
        if reference.shape[:2] != (target_height, target_width):
            reference = resize_image(reference, (target_width, target_height))
        return reference

    def _normalize_score_map(
        self,
        score_map: np.ndarray,
        *,
        low_percentile: float,
        high_percentile: float,
        blur_sigma: float = 0.0,
    ) -> np.ndarray:
        score = np.nan_to_num(score_map.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        score = np.maximum(score, 0.0)
        if blur_sigma > 0:
            score = cv2.GaussianBlur(score, (0, 0), blur_sigma)
        low = float(np.percentile(score, low_percentile))
        high = float(np.percentile(score, high_percentile))
        if high <= low + 1.0e-6:
            low = float(score.min())
            high = float(score.max())
        if high <= low + 1.0e-6:
            return np.zeros_like(score, dtype=np.float32)
        normalized = (score - low) / (high - low)
        return np.clip(normalized, 0.0, 1.0).astype(np.float32)

    def _reference_difference_map(self, source: np.ndarray, reference: np.ndarray) -> np.ndarray:
        source_f = source.astype(np.float32) / 255.0
        reference_f = reference.astype(np.float32) / 255.0
        channel_diff = np.abs(source_f - reference_f)
        gray = cv2.cvtColor((channel_diff * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        max_channel = channel_diff.max(axis=2)
        blended = np.maximum(gray, max_channel)
        return self._normalize_score_map(
            blended,
            low_percentile=65.0,
            high_percentile=99.5,
            blur_sigma=1.2,
        )

    def _combined_heatmap(
        self,
        source: np.ndarray,
        *,
        reference_image: str | Path | np.ndarray | None,
        analysis_mode: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
        watermark_raw = self.models.reveal_tamper_mask(source, self.settings.tamper_mask_scales)
        watermark_map = self._normalize_score_map(
            watermark_raw,
            low_percentile=80.0,
            high_percentile=99.8,
            blur_sigma=1.0,
        )
        reference = self._align_reference_image(reference_image, source.shape[:2])
        reference_map = self._reference_difference_map(source, reference) if reference is not None else None

        if analysis_mode == "reference" and reference_map is not None:
            combined = reference_map
        elif analysis_mode == "hybrid" and reference_map is not None:
            combined = np.clip(0.35 * watermark_map + 0.65 * reference_map, 0.0, 1.0)
            combined = np.maximum(combined, reference_map * 0.95)
        else:
            combined = watermark_map

        combined = cv2.GaussianBlur(combined.astype(np.float32), (0, 0), 1.1)
        combined = np.clip(combined, 0.0, 1.0)
        return combined, watermark_map, reference_map, reference

    def _build_binary_mask(
        self,
        heatmap: np.ndarray,
        *,
        reference_map: np.ndarray | None,
        threshold: float,
    ) -> np.ndarray:
        if reference_map is None:
            binary = (heatmap >= threshold).astype(np.uint8) * 255
        else:
            reference_u8 = np.clip(reference_map * 255.0, 0.0, 255.0).astype(np.uint8)
            otsu_value, _ = cv2.threshold(reference_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            reference_threshold = max(threshold, float(otsu_value) / 255.0)
            direct_mask = reference_map >= max(0.08, reference_threshold * 0.7)
            combined_mask = heatmap >= max(threshold, reference_threshold * 0.85)
            binary = np.logical_and(direct_mask, np.logical_or(combined_mask, reference_map >= 0.22)).astype(np.uint8) * 255

        if binary.any():
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((7, 7), dtype=np.uint8))
            binary = cv2.dilate(binary, np.ones((3, 3), dtype=np.uint8), iterations=1)
        return binary

    def _comparison_metrics(self, source: np.ndarray, reference: np.ndarray | None) -> dict[str, Any]:
        if reference is None:
            return {}
        return {
            "mse_vs_reference": mse(reference, source),
            "mae_vs_reference": mae(reference, source),
            "rmse_vs_reference": rmse(reference, source),
            "psnr_vs_reference": psnr(reference, source),
            "ssim_vs_reference": ssim(reference, source),
            "changed_pixel_ratio_vs_reference": changed_pixel_ratio(reference, source),
        }

    def analyze_image(
        self,
        image: str | Path | np.ndarray,
        expected_document_id: str | None = None,
        reference_bits: list[int] | None = None,
        output_dir: str | Path | None = None,
        reference_image: str | Path | np.ndarray | None = None,
        analysis_mode: str = "hybrid",
        threshold_override: float | None = None,
    ) -> AnalysisResult:
        source = limit_image_side(load_image(image), self.settings.max_input_side)
        predicted_bits = self.models.decode_payload_bits(source)
        payload_result = decode_payload_bits(
            predicted_bits,
            self.settings.hmac_secret,
            expected_document_id=expected_document_id,
            reference_bits=reference_bits,
        )
        threshold = threshold_override if threshold_override is not None else self.settings.tamper_mask_threshold
        heatmap_float, watermark_map, reference_map, aligned_reference = self._combined_heatmap(
            source,
            reference_image=reference_image,
            analysis_mode=analysis_mode,
        )
        binary_mask = self._build_binary_mask(
            heatmap_float,
            reference_map=reference_map,
            threshold=threshold,
        )
        heatmap_uint8 = np.clip(heatmap_float * 255.0, 0.0, 255.0).astype(np.uint8)
        tamper_ratio = float(normalize_mask(binary_mask).mean())
        tamper_heatmap_path = None
        binary_mask_path = None
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            tamper_heatmap_path = save_image(overlay_mask(source, heatmap_uint8), output_dir / "tamper_heatmap.png")
            binary_mask_path = save_mask(binary_mask, output_dir / "tamper_mask.png")
        comparison_metrics = self._comparison_metrics(source, aligned_reference)
        metadata = {
            "expected_document_id": expected_document_id,
            "threshold": threshold,
            "scales": list(self.settings.tamper_mask_scales),
            "analysis_mode": analysis_mode,
            "reference_image_used": aligned_reference is not None,
            "watermark_score_mean": float(watermark_map.mean()),
            "watermark_score_max": float(watermark_map.max()),
            "reference_score_mean": float(reference_map.mean()) if reference_map is not None else None,
            "reference_score_max": float(reference_map.max()) if reference_map is not None else None,
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
            comparison_metrics=comparison_metrics,
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
