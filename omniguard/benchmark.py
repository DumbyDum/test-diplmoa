from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

from . import attacks
from .metrics import bit_accuracy, mask_dice, mask_iou, psnr, ssim
from .schemas import AttackResult
from .service import OmniGuardEngine


AttackCallable = Callable[[np.ndarray], attacks.AttackOutput]


DEFAULT_ATTACKS: tuple[tuple[str, AttackCallable], ...] = (
    ("identity", attacks.identity),
    ("jpeg_q70", lambda image: attacks.jpeg_roundtrip(image, quality=70)),
    ("gaussian_blur", lambda image: attacks.gaussian_blur(image, radius=1.5)),
    ("resize_065", lambda image: attacks.resize_roundtrip(image, scale=0.65)),
    ("brightness_115", lambda image: attacks.adjust_brightness(image, factor=1.15)),
    ("copy_move", attacks.copy_move),
    ("rect_inpaint", attacks.rectangular_inpaint),
)


class BenchmarkRunner:
    def __init__(self, engine: OmniGuardEngine):
        self.engine = engine

    def run(
        self,
        image: str | Path | np.ndarray,
        document_id: str,
        output_dir: str | Path | None = None,
        attack_plan: tuple[tuple[str, AttackCallable], ...] = DEFAULT_ATTACKS,
    ) -> tuple[list[AttackResult], Path]:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = Path(output_dir) if output_dir is not None else (
            self.engine.settings.runtime_dir / "benchmarks" / f"benchmark_{timestamp}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        protection = self.engine.protect_image(image, document_id)
        self.engine.save_protection_bundle(
            protection,
            output_dir / "protected.png",
            output_dir / "protected.json",
        )
        protected = protection.protected_image

        results: list[AttackResult] = []
        rows: list[dict[str, object]] = []
        for attack_name, attack_fn in attack_plan:
            attack_dir = output_dir / attack_name
            attack_dir.mkdir(parents=True, exist_ok=True)
            attacked = attack_fn(protected)
            from .image_ops import save_image, save_mask

            attacked_path = save_image(attacked.image, attack_dir / "attacked.png")
            if attacked.mask is not None:
                save_mask(attacked.mask, attack_dir / "ground_truth_mask.png")
            self.engine.save_json({"attack_name": attack_name}, attack_dir / "attack.json")
            analysis = self.engine.analyze_image(
                attacked.image,
                expected_document_id=document_id,
                reference_bits=protection.payload.encoded_bits,
                output_dir=attack_dir,
            )
            metrics = {
                "psnr_protected_vs_attacked": psnr(protected, attacked.image),
                "ssim_protected_vs_attacked": ssim(protected, attacked.image),
                "payload_bit_accuracy": bit_accuracy(
                    protection.payload.encoded_bits,
                    analysis.payload.decoded_bits,
                ),
                "payload_auth_ok": analysis.payload.auth_ok,
                "payload_document_match": analysis.payload.document_match,
                "tamper_score_mean": analysis.tamper_score_mean,
                "tamper_score_max": analysis.tamper_score_max,
                "tamper_ratio": analysis.tamper_ratio,
            }
            if attacked.mask is not None:
                metrics["mask_iou"] = mask_iou(attacked.mask, analysis.binary_mask)
                metrics["mask_dice"] = mask_dice(attacked.mask, analysis.binary_mask)
            predicted_mask_path = attack_dir / "tamper_mask.png"
            attack_result = AttackResult(
                attack_name=attack_name,
                attacked_image_path=attacked_path,
                predicted_mask_path=predicted_mask_path,
                metrics=metrics,
            )
            results.append(attack_result)
            rows.append({"attack_name": attack_name, **metrics})

        report_path = output_dir / "benchmark_report.json"
        self.engine.save_json(
            {
                "document_id": document_id,
                "protected": protection.to_dict(),
                "attacks": [result.to_dict() for result in results],
            },
            report_path,
        )
        self._save_csv(rows, output_dir / "benchmark_report.csv")
        return results, report_path

    def _save_csv(self, rows: list[dict[str, object]], path: Path) -> None:
        if not rows:
            return
        fieldnames: list[str] = []
        for row in rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
