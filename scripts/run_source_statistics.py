"""Run the power-law source-statistics extension."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from frozen_noise_cortex.report import (  # noqa: E402
    write_source_statistics_csv,
    write_source_statistics_svg,
)
from frozen_noise_cortex.source_statistics import (  # noqa: E402
    SourceStatisticsConfig,
    run_source_statistics_experiment,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare frozen-exemplar and noise-source discrimination."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "build" / "source-statistics",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    arguments = build_parser().parse_args(argv)
    results = run_source_statistics_experiment(SourceStatisticsConfig())
    csv_path = arguments.output_dir / "source-statistics.csv"
    svg_path = arguments.output_dir / "source-statistics.svg"
    write_source_statistics_csv(results, csv_path)
    write_source_statistics_svg(results, svg_path)
    print(f"Wrote {csv_path}")
    print(f"Wrote {svg_path}")


if __name__ == "__main__":
    main()
