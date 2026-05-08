from __future__ import annotations

import argparse
from pathlib import Path

from .benchmark import BenchmarkRunner
from .dataset_generation import SyntheticDatasetBuilder
from .image_ops import save_image
from .paper_comparison import PaperComparisonRunner, rows_for_table
from .attacks import PAPER_DEGRADATION_MAP, PAPER_LOCAL_EDIT_MAP
from .service import OmniGuardEngine
from .ui import launch_ui


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OmniGuard 2.0 CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    protect_parser = subparsers.add_parser("protect", help="Protect an image and embed payload metadata.")
    protect_parser.add_argument("--input", required=True, help="Path to input image.")
    protect_parser.add_argument("--output", required=True, help="Path to protected image.")
    protect_parser.add_argument("--metadata", required=True, help="Path to metadata JSON.")
    protect_parser.add_argument("--document-id", required=True, help="Document ID to embed.")

    analyze_parser = subparsers.add_parser("analyze", help="Analyze an image for payload and tampering.")
    analyze_parser.add_argument("--input", required=True, help="Path to image for analysis.")
    analyze_parser.add_argument("--output-dir", required=True, help="Directory for analysis artifacts.")
    analyze_parser.add_argument("--expected-document-id", default="", help="Expected document ID.")
    analyze_parser.add_argument("--reference", default="", help="Optional protected reference image.")
    analyze_parser.add_argument(
        "--analysis-mode",
        default="hybrid",
        choices=["hybrid", "watermark", "reference"],
        help="Localization mode: hybrid, watermark-only, or reference-only.",
    )
    analyze_parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional override for the binary tamper-mask threshold.",
    )

    benchmark_parser = subparsers.add_parser("benchmark", help="Run the built-in attack benchmark.")
    benchmark_parser.add_argument("--input", required=True, help="Path to input image.")
    benchmark_parser.add_argument("--output-dir", required=True, help="Directory for benchmark report.")
    benchmark_parser.add_argument("--document-id", required=True, help="Document ID to embed.")

    paper_parser = subparsers.add_parser("paper-compare", help="Run paper-style baseline vs enhanced comparison.")
    paper_parser.add_argument("--input", required=True, help="Path to input image.")
    paper_parser.add_argument("--output-dir", required=True, help="Directory for comparison artifacts.")
    paper_parser.add_argument("--document-id", required=True, help="Document ID to embed.")
    paper_parser.add_argument(
        "--local-edit",
        default="opencv_inpaint_proxy",
        choices=sorted(PAPER_LOCAL_EDIT_MAP.keys()),
        help="Local edit scenario.",
    )
    paper_parser.add_argument(
        "--degradation",
        default="clean",
        choices=sorted(PAPER_DEGRADATION_MAP.keys()),
        help="Global degradation after the local edit.",
    )
    paper_parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional override for the binary tamper-mask threshold.",
    )

    dataset_parser = subparsers.add_parser("generate-dataset", help="Generate a synthetic dataset.")
    dataset_parser.add_argument("--input-dir", required=True, help="Directory with source images.")
    dataset_parser.add_argument("--output-dir", required=True, help="Directory where the dataset will be written.")
    dataset_parser.add_argument("--limit", type=int, default=None, help="Optional limit on processed images.")

    ui_parser = subparsers.add_parser("launch-ui", help="Launch the local Gradio UI.")
    ui_parser.add_argument("--host", default="127.0.0.1", help="Bind host.")
    ui_parser.add_argument("--port", type=int, default=7860, help="Bind port.")
    ui_parser.add_argument("--share", action="store_true", help="Expose a public Gradio share link.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    engine = OmniGuardEngine()

    if args.command == "protect":
        result = engine.protect_image(args.input, args.document_id)
        save_image(result.protected_image, args.output)
        engine.save_json(result.to_dict(), args.metadata)
        print(f"Protected image written to: {Path(args.output).resolve()}")
        print(f"Metadata written to: {Path(args.metadata).resolve()}")
        return 0

    if args.command == "analyze":
        result = engine.analyze_image(
            args.input,
            expected_document_id=args.expected_document_id or None,
            output_dir=args.output_dir,
            reference_image=args.reference or None,
            analysis_mode=args.analysis_mode,
            threshold_override=args.threshold,
        )
        engine.save_json(result.to_dict(), Path(args.output_dir) / "analysis_report.json")
        print(f"Analysis artifacts written to: {Path(args.output_dir).resolve()}")
        print(f"Payload auth ok: {result.payload.auth_ok}")
        print(f"Tamper score max: {result.tamper_score_max:.6f}")
        return 0

    if args.command == "benchmark":
        runner = BenchmarkRunner(engine)
        _, report_path = runner.run(
            image=args.input,
            document_id=args.document_id,
            output_dir=args.output_dir,
        )
        print(f"Benchmark report written to: {report_path.resolve()}")
        return 0

    if args.command == "paper-compare":
        runner = PaperComparisonRunner(engine)
        result = runner.run_generated(
            image=args.input,
            document_id=args.document_id,
            local_edit_id=args.local_edit,
            degradation_id=args.degradation,
            threshold=args.threshold,
            output_dir=args.output_dir,
        )
        print(f"Paper comparison report written to: {result.report_path.resolve()}")
        for row in rows_for_table(result.rows):
            print(" | ".join(str(value) for value in row))
        return 0

    if args.command == "generate-dataset":
        builder = SyntheticDatasetBuilder(engine)
        records, manifest_path = builder.build(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            limit=args.limit,
        )
        print(f"Synthetic dataset samples created: {len(records)}")
        print(f"Manifest written to: {manifest_path.resolve()}")
        return 0

    if args.command == "launch-ui":
        launch_ui(host=args.host, port=args.port, share=args.share)
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
