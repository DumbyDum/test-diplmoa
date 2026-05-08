from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from . import attacks
from .image_ops import limit_image_side, load_image, overlay_mask, save_image, save_mask
from .metrics import ber, bit_accuracy, bpp, mask_auc, mask_f1, psnr, ssim
from .schemas import AnalysisResult, ProtectionResult
from .service import OmniGuardEngine


PAPER_COMPARISON_HEADERS = [
    "Метод",
    "Capacity, bits",
    "PSNR",
    "SSIM",
    "Bit Accuracy, %",
    "BER",
    "F1",
    "AUC",
    "tamper_ratio",
    "payload_auth_ok",
    "document_match",
]


@dataclass(slots=True)
class PaperComparisonResult:
    protected_image: np.ndarray
    attacked_image: np.ndarray
    ground_truth_mask: np.ndarray
    baseline_overlay: np.ndarray
    baseline_mask: np.ndarray
    enhanced_overlay: np.ndarray
    enhanced_mask: np.ndarray
    rows: list[dict[str, Any]]
    report: dict[str, Any]
    report_path: Path


def paper_local_edit_choices() -> list[tuple[str, str]]:
    return [(description, name) for name, description, _ in attacks.PAPER_LOCAL_EDITS]


def paper_degradation_choices() -> list[tuple[str, str]]:
    return [(description, name) for name, description, _ in attacks.PAPER_DEGRADATIONS]


def _apply_paper_condition(
    image: np.ndarray,
    *,
    local_edit_id: str,
    degradation_id: str,
) -> tuple[np.ndarray, np.ndarray]:
    if local_edit_id not in attacks.PAPER_LOCAL_EDIT_MAP:
        raise ValueError(f"Unknown local edit scenario: {local_edit_id}")
    if degradation_id not in attacks.PAPER_DEGRADATION_MAP:
        raise ValueError(f"Unknown degradation scenario: {degradation_id}")

    local_edit = attacks.PAPER_LOCAL_EDIT_MAP[local_edit_id](image)
    if local_edit.mask is None:
        raise ValueError("Paper comparison requires a local edit with a ground-truth mask.")
    degraded = attacks.PAPER_DEGRADATION_MAP[degradation_id](local_edit.image)
    return degraded.image, local_edit.mask


def _metric_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "да" if value else "нет"
    if isinstance(value, float):
        return round(value, 6)
    return value


class PaperComparisonRunner:
    """Runs paper-style metrics for the baseline and enhanced localization modes.

    The "baseline" branch intentionally uses the watermark-only localization map.
    This is the closest executable analogue of the original residual-mask idea
    from the article, while the "enhanced" branch uses the upgraded hybrid logic:
    watermark map + protected-reference comparison + adaptive mask post-processing.
    Both branches receive the same protected image, attacked image, document_id,
    reference bits, threshold and ground-truth mask.
    """

    def __init__(self, engine: OmniGuardEngine):
        self.engine = engine

    def run_generated(
        self,
        image: str | Path | np.ndarray,
        document_id: str,
        *,
        local_edit_id: str = "opencv_inpaint_proxy",
        degradation_id: str = "clean",
        threshold: float | None = None,
        output_dir: str | Path | None = None,
    ) -> PaperComparisonResult:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = Path(output_dir) if output_dir is not None else (
            self.engine.settings.runtime_dir / "paper_comparisons" / f"comparison_{timestamp}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        original = limit_image_side(load_image(image), self.engine.settings.max_input_side)
        protection = self.engine.protect_image(original, document_id)
        protected = protection.protected_image
        attacked, ground_truth_mask = _apply_paper_condition(
            protected,
            local_edit_id=local_edit_id,
            degradation_id=degradation_id,
        )

        save_image(original, output_dir / "original.png")
        save_image(protected, output_dir / "protected.png")
        save_image(attacked, output_dir / "attacked.png")
        save_mask(ground_truth_mask, output_dir / "ground_truth_mask.png")

        baseline_dir = output_dir / "baseline_watermark_only"
        enhanced_dir = output_dir / "enhanced_hybrid"
        baseline = self.engine.analyze_image(
            attacked,
            expected_document_id=document_id,
            reference_bits=protection.payload.encoded_bits,
            reference_image=None,
            analysis_mode="watermark",
            threshold_override=threshold,
            output_dir=baseline_dir,
        )
        enhanced = self.engine.analyze_image(
            attacked,
            expected_document_id=document_id,
            reference_bits=protection.payload.encoded_bits,
            reference_image=protected,
            analysis_mode="hybrid",
            threshold_override=threshold,
            output_dir=enhanced_dir,
        )

        rows = [
            self._build_metric_row(
                method_name="Базовая версия: watermark-only residual",
                original=original,
                protection=protection,
                analysis=baseline,
                ground_truth_mask=ground_truth_mask,
            ),
            self._build_metric_row(
                method_name="Улучшенная версия: hybrid watermark + reference",
                original=original,
                protection=protection,
                analysis=enhanced,
                ground_truth_mask=ground_truth_mask,
            ),
        ]
        report = {
            "document_id": document_id,
            "paper_metric_mapping": {
                "Capacity": "число встроенных бит payload",
                "PSNR": "качество protected относительно original",
                "SSIM": "структурное сходство protected относительно original",
                "Bit Accuracy": "доля правильно извлеченных payload-бит после атаки",
                "F1": "качество бинарной маски локализации относительно ground_truth_mask",
                "AUC": "порогонезависимое качество heatmap относительно ground_truth_mask",
            },
            "same_conditions": {
                "local_edit_id": local_edit_id,
                "degradation_id": degradation_id,
                "threshold": threshold if threshold is not None else self.engine.settings.tamper_mask_threshold,
                "baseline_analysis_mode": "watermark",
                "enhanced_analysis_mode": "hybrid",
                "same_protected_image": True,
                "same_attacked_image": True,
                "same_ground_truth_mask": True,
                "same_document_id": True,
                "same_reference_bits": True,
            },
            "rows": rows,
            "protection": protection.to_dict(),
            "baseline": baseline.to_dict(),
            "enhanced": enhanced.to_dict(),
        }
        report_path = self.engine.save_json(report, output_dir / "paper_comparison_report.json")
        return PaperComparisonResult(
            protected_image=protected,
            attacked_image=attacked,
            ground_truth_mask=ground_truth_mask,
            baseline_overlay=overlay_mask(attacked, baseline.tamper_heatmap),
            baseline_mask=baseline.binary_mask,
            enhanced_overlay=overlay_mask(attacked, enhanced.tamper_heatmap),
            enhanced_mask=enhanced.binary_mask,
            rows=rows,
            report=report,
            report_path=report_path,
        )

    def _build_metric_row(
        self,
        *,
        method_name: str,
        original: np.ndarray,
        protection: ProtectionResult,
        analysis: AnalysisResult,
        ground_truth_mask: np.ndarray,
    ) -> dict[str, Any]:
        capacity_bits = len(protection.payload.encoded_bits)
        extracted_accuracy = analysis.payload.bit_accuracy
        if extracted_accuracy is None:
            extracted_accuracy = bit_accuracy(
                protection.payload.encoded_bits,
                analysis.payload.decoded_bits,
            )
        return {
            "Метод": method_name,
            "Capacity, bits": capacity_bits,
            "PSNR": psnr(original, protection.protected_image),
            "SSIM": ssim(original, protection.protected_image),
            "Bit Accuracy, %": None if extracted_accuracy is None else extracted_accuracy * 100.0,
            "BER": ber(protection.payload.encoded_bits, analysis.payload.decoded_bits),
            "F1": mask_f1(ground_truth_mask, analysis.binary_mask),
            "AUC": mask_auc(ground_truth_mask, analysis.tamper_heatmap),
            "tamper_ratio": analysis.tamper_ratio,
            "payload_auth_ok": analysis.payload.auth_ok,
            "document_match": analysis.payload.document_match,
            "payload_bpp": bpp(capacity_bits, protection.protected_image),
        }


def rows_for_table(rows: list[dict[str, Any]]) -> list[list[Any]]:
    table: list[list[Any]] = []
    for row in rows:
        table.append([_metric_value(row.get(header)) for header in PAPER_COMPARISON_HEADERS])
    return table
