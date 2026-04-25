from __future__ import annotations

import argparse

from omniguard.ui import launch_ui


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the OmniGuard 2.0 UI.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host.")
    parser.add_argument("--port", type=int, default=7860, help="Bind port.")
    parser.add_argument(
        "--share",
        action="store_true",
        help="Expose a public Gradio share link. Disabled by default for safety.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    launch_ui(host=args.host, port=args.port, share=args.share)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
