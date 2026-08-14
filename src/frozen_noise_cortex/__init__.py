"""Frozen-noise discrimination model."""

from .experiment import ExperimentConfig, run_experiment
from .model import CochleaConfig, FrozenNoiseEncoder

__all__ = [
    "CochleaConfig",
    "ExperimentConfig",
    "FrozenNoiseEncoder",
    "run_experiment",
]

