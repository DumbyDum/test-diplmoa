from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .attacks import REQUIREMENT_ATTACKS
from .basic_watermarking import (
    WatermarkEmbedResult,
    bits_to_text,
    get_basic_method,
)
from .image_ops import load_image, save_image
from .metrics import ber, bpp, psnr, ssim
from .service import OmniGuardEngine


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


@dataclass(slots=True)
class RequirementEmbedBundle:
    method_id: str
    method_name: str
    watermarked_image: np.ndarray
    payload_bits: list[int]
    metrics: dict[str, float | None]
    metadata: dict[str, Any]


class RequirementExperimentRunner:
    """Runs the exact attacks and metrics listed in the diploma requirements file."""

    def __init__(self, engine: OmniGuardEngine):
        self.engine = engine

    def collect_images(self, uploaded_files: Iterable[Any] | None, folder_path: str | None) -> list[Path]:
        paths: list[Path] = []
        for file_item in uploaded_files or []:
            candidate = getattr(file_item, "name", file_item)
            if candidate:
                paths.append(Path(candidate))

        if folder_path and folder_path.strip():
            folder = Path(folder_path.strip()).expanduser()
            if folder.exists() and folder.is_dir():
                for path in sorted(folder.rglob("*")):
                    if path.suffix.lower() in IMAGE_EXTENSIONS:
                        paths.append(path)

        unique: list[Path] = []
        seen: set[str] = set()
        for path in paths:
            key = str(path.resolve()) if path.exists() else str(path)
            if key not in seen:
                seen.add(key)
                unique.append(path)
        return unique

    def embed(self, method_id: str, image: np.ndarray, payload_text: str) -> RequirementEmbedBundle:
        if method_id == "omniguard":
            protection = self.engine.protect_image(image, payload_text)
            payload_bits = protection.payload.encoded_bits
            metrics = {
                "psnr": psnr(image, protection.protected_image),
                "ssim": ssim(image, protection.protected_image),
                "bpp": bpp(len(payload_bits), image),
            }
            metadata = {
                "payload_type": "OmniGuard encoded payload",
                "embedded_bits": len(payload_bits),
                "formula_bpp": "bpp = embedded_bits / (height * width)",
            }
            return RequirementEmbedBundle(
                method_id="omniguard",
                method_name="OmniGuard: нейросетевой ЦВЗ + payload",
                watermarked_image=protection.protected_image,
                payload_bits=payload_bits,
                metrics=metrics,
                metadata=metadata,
            )

        method = get_basic_method(method_id)
        result: WatermarkEmbedResult = method.embed(image, payload_text)
        return RequirementEmbedBundle(
            method_id=result.method_id,
            method_name=result.method_name,
            watermarked_image=result.watermarked_image,
            payload_bits=result.payload_bits,
            metrics=result.metrics,
            metadata=result.metadata,
        )

    def extract_bits(
        self,
        method_id: str,
        image: np.ndarray,
        bit_count: int,
        metadata: dict[str, Any],
    ) -> list[int]:
        if method_id == "omniguard":
            return self.engine.models.decode_payload_bits(image)[:bit_count]
        method = get_basic_method(method_id)
        return method.extract(image, bit_count, metadata).bits

    def run_single(
        self,
        image: np.ndarray,
        payload_text: str,
        method_id: str,
    ) -> tuple[RequirementEmbedBundle, str]:
        bundle = self.embed(method_id, image, payload_text)
        extracted = self.extract_bits(
            method_id,
            bundle.watermarked_image,
            len(bundle.payload_bits),
            bundle.metadata,
        )
        extracted_text = bits_to_text(extracted) if method_id in {"lsb", "dct"} else ""
        bundle.metadata["clean_ber"] = ber(bundle.payload_bits, extracted)
        if extracted_text:
            bundle.metadata["extracted_text_without_attack"] = extracted_text
        explanation = self._single_explanation(bundle)
        return bundle, explanation

    def run_batch(
        self,
        image_paths: list[Path],
        payload_text: str,
        method_ids: list[str],
        output_dir: str | Path | None = None,
    ) -> tuple[list[dict[str, Any]], Path]:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_root = Path(output_dir) if output_dir is not None else (
            self.engine.settings.runtime_dir / "requirement_benchmarks" / f"run_{timestamp}"
        )
        output_root.mkdir(parents=True, exist_ok=True)

        rows: list[dict[str, Any]] = []
        for image_index, image_path in enumerate(image_paths, start=1):
            try:
                source = load_image(image_path)
            except Exception as exc:
                rows.append(
                    {
                        "image": str(image_path),
                        "method": "",
                        "attack": "",
                        "psnr": None,
                        "ssim": None,
                        "bpp": None,
                        "ber": None,
                        "status": f"image_load_error: {exc}",
                    }
                )
                continue

            per_image_payload = payload_text or f"omniguard-{image_index:04d}"
            for method_id in method_ids:
                try:
                    bundle = self.embed(method_id, source, per_image_payload)
                    method_dir = output_root / image_path.stem / method_id
                    save_image(bundle.watermarked_image, method_dir / "watermarked.png")

                    clean_bits = self.extract_bits(
                        method_id,
                        bundle.watermarked_image,
                        len(bundle.payload_bits),
                        bundle.metadata,
                    )
                    rows.append(
                        self._row(
                            image_path=image_path,
                            method_id=method_id,
                            method_name=bundle.method_name,
                            attack_name="no_attack",
                            attack_description="Без атаки: проверка извлечения сразу после встраивания",
                            embed_bundle=bundle,
                            extracted_bits=clean_bits,
                            status="ok",
                        )
                    )

                    for attack_name, attack_description, attack_fn in REQUIREMENT_ATTACKS:
                        attacked = attack_fn(bundle.watermarked_image)
                        attack_dir = method_dir / attack_name
                        save_image(attacked.image, attack_dir / "attacked.png")
                        extracted_bits = self.extract_bits(
                            method_id,
                            attacked.image,
                            len(bundle.payload_bits),
                            bundle.metadata,
                        )
                        rows.append(
                            self._row(
                                image_path=image_path,
                                method_id=method_id,
                                method_name=bundle.method_name,
                                attack_name=attack_name,
                                attack_description=attack_description,
                                embed_bundle=bundle,
                                extracted_bits=extracted_bits,
                                status="ok",
                            )
                        )
                except Exception as exc:
                    rows.append(
                        {
                            "image": image_path.name,
                            "method": method_id,
                            "attack": "",
                            "attack_description": "",
                            "psnr": None,
                            "ssim": None,
                            "bpp": None,
                            "ber": None,
                            "embedded_bits": None,
                            "extracted_bits": None,
                            "status": f"method_error: {exc}",
                        }
                    )

        csv_path = output_root / "requirement_benchmark.csv"
        json_path = output_root / "requirement_benchmark.json"
        self._save_csv(rows, csv_path)
        self.engine.save_json(
            {
                "payload_text": payload_text,
                "methods": method_ids,
                "images": [str(path) for path in image_paths],
                "metrics": ["PSNR", "SSIM", "bpp", "BER"],
                "attacks": [
                    {"name": name, "description": description}
                    for name, description, _ in REQUIREMENT_ATTACKS
                ],
                "rows": rows,
            },
            json_path,
        )
        return rows, csv_path

    def _row(
        self,
        *,
        image_path: Path,
        method_id: str,
        method_name: str,
        attack_name: str,
        attack_description: str,
        embed_bundle: RequirementEmbedBundle,
        extracted_bits: list[int],
        status: str,
    ) -> dict[str, Any]:
        return {
            "image": image_path.name,
            "method": method_id,
            "method_name": method_name,
            "attack": attack_name,
            "attack_description": attack_description,
            "psnr": embed_bundle.metrics["psnr"],
            "ssim": embed_bundle.metrics["ssim"],
            "bpp": embed_bundle.metrics["bpp"],
            "ber": ber(embed_bundle.payload_bits, extracted_bits),
            "embedded_bits": len(embed_bundle.payload_bits),
            "extracted_bits": len(extracted_bits),
            "status": status,
        }

    def _save_csv(self, rows: list[dict[str, Any]], path: Path) -> None:
        if not rows:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames: list[str] = []
        for row in rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _single_explanation(self, bundle: RequirementEmbedBundle) -> str:
        psnr_value = bundle.metrics["psnr"]
        ssim_value = bundle.metrics["ssim"]
        bpp_value = bundle.metrics["bpp"]
        clean_ber = bundle.metadata.get("clean_ber")
        return (
            f"**Метод:** {bundle.method_name}\n\n"
            f"**Сколько встроено:** {len(bundle.payload_bits)} бит.\n\n"
            f"**PSNR:** {psnr_value:.4f} dB. Чем выше значение, тем менее заметно отличие после встраивания.\n\n"
            f"**SSIM:** {ssim_value if ssim_value is not None else 'не рассчитан'}. "
            "Чем ближе к 1, тем больше структурное сходство с исходным изображением.\n\n"
            f"**bpp:** {bpp_value:.8f}. Формула: `bpp = embedded_bits / (height * width)`.\n\n"
            f"**BER без атаки:** {clean_ber}. Формула: `BER = number_of_wrong_bits / number_of_embedded_bits`."
        )
