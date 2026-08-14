"""Command-line entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from .experiment import ExperimentConfig, run_experiment
from .model import CochleaConfig
from .report import write_accuracy_svg, write_csv


def _parse_number_list(value: str, converter):
    try:
        return tuple(converter(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep frozen-noise duration and neural readout noise."
    )
    parser.add_argument(
        "--durations-ms",
        type=lambda value: _parse_number_list(value, int),
        default=ExperimentConfig.durations_ms,
        help="comma-separated durations (default: 25,50,100,200,400,800,1600)",
    )
    parser.add_argument(
        "--noise-sd",
        type=lambda value: _parse_number_list(value, float),
        default=ExperimentConfig.readout_noise_sd,
        help="comma-separated post-pooling noise SDs in spikes/s (default: 0,10,20,40)",
    )
    parser.add_argument("--pairs", type=int, default=ExperimentConfig.n_frozen_pairs)
    parser.add_argument("--seed", type=int, default=ExperimentConfig.seed)
    parser.add_argument("--channels", type=int, default=CochleaConfig.n_channels)
    parser.add_argument("--output", type=Path, default=Path("results"))
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    experiment = ExperimentConfig(
        durations_ms=args.durations_ms,
        readout_noise_sd=args.noise_sd,
        n_frozen_pairs=args.pairs,
        seed=args.seed,
    )
    cochlea = CochleaConfig(n_channels=args.channels)
    results = run_experiment(experiment, cochlea)
    write_csv(results, args.output / "accuracy.csv")
    write_accuracy_svg(results, args.output / "accuracy.svg")
    print(f"Wrote {args.output / 'accuracy.csv'}")
    print(f"Wrote {args.output / 'accuracy.svg'}")


if __name__ == "__main__":
    main()
