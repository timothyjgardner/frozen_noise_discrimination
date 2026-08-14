"""Reproduce and verify every generated project artifact."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
from pypdf import PdfReader

from build_report import build_pdf
from frozen_noise_cortex.experiment import ExperimentConfig, run_experiment
from frozen_noise_cortex.model import CochleaConfig
from frozen_noise_cortex.report import (
    write_accuracy_svg,
    write_csv,
    write_source_statistics_csv,
    write_source_statistics_svg,
)
from frozen_noise_cortex.source_statistics import (
    SourceStatisticsConfig,
    run_source_statistics_experiment,
)


REFERENCE_CSV = ROOT / "results" / "accuracy.csv"
REFERENCE_SOURCE_CSV = ROOT / "results" / "source-statistics.csv"
REFERENCE_MANIFEST = ROOT / "ARTIFACTS.sha256"
DEFAULT_OUTPUT = ROOT / "build" / "reproduced"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_reference_manifest(path: Path = REFERENCE_MANIFEST) -> None:
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        expected, relative_path = line.split(maxsplit=1)
        artifact = ROOT / relative_path.strip()
        actual = sha256(artifact)
        if actual != expected:
            raise AssertionError(
                f"reference hash mismatch at {path.name}:{line_number}: "
                f"{relative_path.strip()}"
            )


def read_csv(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def compare_results(
    reference_path: Path,
    reproduced_path: Path,
    tolerance: float,
) -> float:
    reference = read_csv(reference_path)
    reproduced = read_csv(reproduced_path)
    if len(reference) != len(reproduced):
        raise AssertionError(
            f"row count differs: reference={len(reference)}, reproduced={len(reproduced)}"
        )

    maximum_delta = 0.0
    for row_index, (expected, actual) in enumerate(zip(reference, reproduced), start=2):
        if expected.keys() != actual.keys():
            raise AssertionError(f"columns differ at CSV row {row_index}")
        for field in expected:
            delta = abs(expected[field] - actual[field])
            maximum_delta = max(maximum_delta, delta)
            if delta > tolerance:
                raise AssertionError(
                    f"{field} differs at CSV row {row_index}: "
                    f"expected={expected[field]!r}, actual={actual[field]!r}, "
                    f"delta={delta:.3g}"
                )
    return maximum_delta


def compare_mixed_results(
    reference_path: Path,
    reproduced_path: Path,
    tolerance: float,
) -> float:
    """Compare a CSV containing both categorical and numeric columns."""

    with reference_path.open(newline="", encoding="utf-8") as handle:
        reference = list(csv.DictReader(handle))
    with reproduced_path.open(newline="", encoding="utf-8") as handle:
        reproduced = list(csv.DictReader(handle))
    if len(reference) != len(reproduced):
        raise AssertionError(
            f"row count differs: reference={len(reference)}, reproduced={len(reproduced)}"
        )

    numeric_fields = {
        "duration_ms",
        "readout_noise_sd",
        "accuracy_mean",
        "accuracy_sem",
    }
    maximum_delta = 0.0
    for row_index, (expected, actual) in enumerate(zip(reference, reproduced), start=2):
        if expected.keys() != actual.keys():
            raise AssertionError(f"columns differ at CSV row {row_index}")
        for field in expected:
            if field in numeric_fields:
                delta = abs(float(expected[field]) - float(actual[field]))
                maximum_delta = max(maximum_delta, delta)
                if delta > tolerance:
                    raise AssertionError(
                        f"{field} differs at CSV row {row_index}: "
                        f"expected={expected[field]!r}, actual={actual[field]!r}, "
                        f"delta={delta:.3g}"
                    )
            elif expected[field] != actual[field]:
                raise AssertionError(
                    f"{field} differs at CSV row {row_index}: "
                    f"expected={expected[field]!r}, actual={actual[field]!r}"
                )
    return maximum_delta


def verify_pdf(path: Path) -> None:
    reader = PdfReader(path)
    if len(reader.pages) != 10:
        raise AssertionError(f"expected 10 PDF pages, found {len(reader.pages)}")
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    normalized_text = " ".join(text.split())
    required = [
        "Frozen-Noise Discrimination",
        "1. Question and modeling hypothesis",
        "2. Model architecture",
        "3. Experiment and evaluation",
        "4. Initial results",
        "5. Mechanistic interpretation",
        "6. Source-statistics extension",
        "7. Limitations and next experiments",
        "Appendix A. Where variability enters",
        "Appendix A. Noise after pooling and duration effects",
        "Complete stochastic feature equation",
        "Var[r_k(T)] approx {rho_k p(1-p) + lambda} / T + sigma_read^2",
        "100.0% 88.0% 72.3% 59.8%",
        "99.8% 86.8% 99.6% 99.3%",
    ]
    missing = [value for value in required if value not in normalized_text]
    if missing:
        raise AssertionError(f"PDF is missing expected text: {missing}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Regenerate and verify the complete frozen-noise project."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="directory for reproduced artifacts",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-12,
        help="maximum permitted absolute numeric difference from reference CSV",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    arguments = build_parser().parse_args(argv)
    output_dir = arguments.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    verify_reference_manifest()
    results = run_experiment(ExperimentConfig(), CochleaConfig())
    reproduced_csv = output_dir / "accuracy.csv"
    reproduced_svg = output_dir / "accuracy.svg"
    reproduced_pdf = output_dir / "frozen-noise-cortex-writeup.pdf"
    reproduced_source_csv = output_dir / "source-statistics.csv"
    reproduced_source_svg = output_dir / "source-statistics.svg"
    metadata_path = output_dir / "reproduction-metadata.json"

    write_csv(results, reproduced_csv)
    write_accuracy_svg(results, reproduced_svg)
    maximum_delta = compare_results(
        REFERENCE_CSV,
        reproduced_csv,
        arguments.tolerance,
    )
    source_results = run_source_statistics_experiment(
        SourceStatisticsConfig(),
        CochleaConfig(),
    )
    write_source_statistics_csv(source_results, reproduced_source_csv)
    write_source_statistics_svg(source_results, reproduced_source_svg)
    source_maximum_delta = compare_mixed_results(
        REFERENCE_SOURCE_CSV,
        reproduced_source_csv,
        arguments.tolerance,
    )
    maximum_delta = max(maximum_delta, source_maximum_delta)
    build_pdf(reproduced_csv, reproduced_pdf, reproduced_source_csv)
    verify_pdf(reproduced_pdf)

    metadata = {
        "status": "verified",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "random_seed": ExperimentConfig.seed,
        "reference_csv": str(REFERENCE_CSV.relative_to(ROOT)),
        "maximum_absolute_csv_delta": maximum_delta,
        "numeric_tolerance": arguments.tolerance,
        "artifacts": {
            path.name: {
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in (
                reproduced_csv,
                reproduced_svg,
                reproduced_source_csv,
                reproduced_source_svg,
                reproduced_pdf,
            )
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print("Reproduction verified")
    print(f"Numeric rows: {len(results) + len(source_results)}")
    print(f"Maximum CSV delta: {maximum_delta:.3g}")
    print(f"Artifacts: {output_dir}")


if __name__ == "__main__":
    main()
