from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from . import attacks
from .image_ops import load_image, save_image, save_mask
from .schemas import DatasetSampleRecord
from .service import OmniGuardEngine


class SyntheticDatasetBuilder:
    def __init__(self, engine: OmniGuardEngine):
        self.engine = engine

    def build(
        self,
        input_dir: str | Path,
        output_dir: str | Path | None = None,
        limit: int | None = None,
    ) -> tuple[list[DatasetSampleRecord], Path]:
        input_dir = Path(input_dir)
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = Path(output_dir) if output_dir is not None else (
            self.engine.settings.runtime_dir / "datasets" / f"synthetic_{timestamp}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        image_paths = sorted(
            path
            for path in input_dir.rglob("*")
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        )
        if limit is not None:
            image_paths = image_paths[:limit]

        records: list[DatasetSampleRecord] = []
        attack_plan = (
            ("rect_inpaint", attacks.rectangular_inpaint),
            ("copy_move", attacks.copy_move),
        )
        for sample_index, image_path in enumerate(image_paths):
            document_id = image_path.stem
            sample_dir = output_dir / f"{sample_index:04d}_{document_id}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            protection = self.engine.protect_image(image_path, document_id)
            source = load_image(image_path)
            source_path = save_image(source, sample_dir / "original.png")
            protected_path = save_image(protection.protected_image, sample_dir / "protected.png")
            self.engine.save_json(protection.to_dict(), sample_dir / "protected.json")
            for attack_name, attack_fn in attack_plan:
                attack_output = attack_fn(protection.protected_image)
                edited_path = save_image(attack_output.image, sample_dir / f"{attack_name}_edited.png")
                mask_path = save_mask(
                    attack_output.mask if attack_output.mask is not None else source[..., 0] * 0,
                    sample_dir / f"{attack_name}_mask.png",
                )
                record = DatasetSampleRecord(
                    source_path=source_path,
                    document_id=document_id,
                    attack_name=attack_name,
                    original_path=source_path,
                    protected_path=protected_path,
                    edited_path=edited_path,
                    mask_path=mask_path,
                )
                records.append(record)

        manifest_path = output_dir / "manifest.jsonl"
        with manifest_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        self.engine.save_json(
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "samples": len(records),
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
            },
            output_dir / "summary.json",
        )
        return records, manifest_path
