from __future__ import annotations

from pathlib import Path

from omniguard import OmniGuardEngine
from omniguard.benchmark import BenchmarkRunner


def main() -> int:
    engine = OmniGuardEngine()
    output_dir = engine.settings.runtime_dir / "smoke_test"
    benchmark_runner = BenchmarkRunner(engine)
    _, report_path = benchmark_runner.run(
        image=Path("examples/0000.png"),
        document_id="smoke-test-001",
        output_dir=output_dir,
    )
    print(f"Smoke benchmark completed: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
