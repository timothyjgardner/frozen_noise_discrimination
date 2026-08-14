"""Duration experiment for exemplar identity versus noise-source identity.

The original experiment trains a decoder to distinguish two fixed frozen
waveforms.  This extension adds a complementary task: train on independent
exemplars from two stationary noise families and test generalization to new
exemplars.  Time averaging should remove the first task's accidental waveform
fingerprints while improving estimation of the second task's stable spectrum.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .experiment import (
    _classification_accuracy,
    _noisy_rate_trials,
    _summarize,
)
from .model import CochleaConfig, FrozenNoiseEncoder


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class NoiseFamily:
    """A Gaussian noise family with power spectral density proportional to 1/f^a."""

    name: str
    spectral_exponent: float


DEFAULT_FAMILIES: tuple[NoiseFamily, ...] = (
    NoiseFamily("white", 0.0),
    NoiseFamily("light-pink", 0.5),
    NoiseFamily("pink", 1.0),
    NoiseFamily("brown", 2.0),
)

DEFAULT_COMPARISONS: tuple[tuple[str, str], ...] = (
    ("white", "light-pink"),
    ("white", "pink"),
    ("pink", "brown"),
)


@dataclass(frozen=True)
class SourceStatisticsConfig:
    durations_ms: tuple[int, ...] = (25, 50, 100, 200, 400, 800, 1_600)
    readout_noise_sd: tuple[float, ...] = (10.0, 20.0, 40.0)
    families: tuple[NoiseFamily, ...] = DEFAULT_FAMILIES
    comparisons: tuple[tuple[str, str], ...] = DEFAULT_COMPARISONS
    n_source_exemplars: int = 160
    source_split_repeats: int = 12
    source_train_exemplars: int = 32
    source_test_exemplars: int = 96
    frozen_pairs_per_family: int = 8
    frozen_train_trials_per_class: int = 32
    frozen_test_trials_per_class: int = 96
    spike_dropout_probability: float = 0.03
    spontaneous_rate_hz: float = 0.20
    classifier_ridge: float = 0.20
    seed: int = 29

    def validate(self) -> None:
        if not self.durations_ms or min(self.durations_ms) <= 0:
            raise ValueError("durations_ms must contain positive values")
        if not self.readout_noise_sd or min(self.readout_noise_sd) < 0.0:
            raise ValueError("readout_noise_sd cannot contain negative values")
        names = [family.name for family in self.families]
        if not names or len(names) != len(set(names)):
            raise ValueError("families must have unique names")
        for first, second in self.comparisons:
            if first not in names or second not in names or first == second:
                raise ValueError(f"invalid source comparison: {(first, second)!r}")
        required_exemplars = self.source_train_exemplars + self.source_test_exemplars
        if required_exemplars > self.n_source_exemplars:
            raise ValueError("source train and test sets exceed n_source_exemplars")
        if self.source_split_repeats < 2:
            raise ValueError("source_split_repeats must be at least 2")
        if self.frozen_pairs_per_family < 2:
            raise ValueError("frozen_pairs_per_family must be at least 2")
        if 2 * self.frozen_pairs_per_family > self.n_source_exemplars:
            raise ValueError("not enough exemplars for frozen pairs")
        if self.frozen_train_trials_per_class < 2:
            raise ValueError("frozen_train_trials_per_class must be at least 2")
        if self.frozen_test_trials_per_class < 1:
            raise ValueError("frozen_test_trials_per_class must be positive")
        if not 0.0 <= self.spike_dropout_probability < 1.0:
            raise ValueError("spike_dropout_probability must lie in [0, 1)")
        if self.spontaneous_rate_hz < 0.0:
            raise ValueError("spontaneous_rate_hz cannot be negative")
        if self.classifier_ridge <= 0.0:
            raise ValueError("classifier_ridge must be positive")


def generate_powerlaw_noise(
    n_samples: int,
    spectral_exponent: float,
    rng: np.random.Generator,
    sample_rate_hz: int = 16_000,
    low_frequency_hz: float = 200.0,
    high_frequency_hz: float = 7_000.0,
) -> FloatArray:
    """Generate band-limited Gaussian noise with PSD proportional to 1/f^a.

    ``spectral_exponent`` is 0 for white, 1 for pink, and 2 for brown noise.
    The Fourier amplitude is therefore scaled by f^(-a/2).  All families are
    limited to the modeled cochlear range before the encoder equalizes RMS.
    """

    if n_samples < 4:
        raise ValueError("n_samples must be at least 4")
    if not 0.0 < low_frequency_hz < high_frequency_hz < sample_rate_hz / 2.0:
        raise ValueError("noise band must lie between 0 and Nyquist")
    waveform = rng.normal(size=n_samples)
    spectrum = np.fft.rfft(waveform)
    frequencies = np.fft.rfftfreq(n_samples, 1.0 / sample_rate_hz)
    scale = np.zeros_like(frequencies)
    in_band = (frequencies >= low_frequency_hz) & (
        frequencies <= high_frequency_hz
    )
    scale[in_band] = (frequencies[in_band] / 1_000.0) ** (
        -spectral_exponent / 2.0
    )
    return np.fft.irfft(spectrum * scale, n=n_samples)


def _noisy_rate_observations(
    clean_counts: IntArray,
    duration_s: float,
    rng: np.random.Generator,
    dropout_probability: float,
    spontaneous_rate_hz: float,
    readout_noise_sd: float,
) -> FloatArray:
    """Make one noisy rate observation for every independent waveform."""

    detected = rng.binomial(clean_counts, 1.0 - dropout_probability)
    spontaneous = rng.poisson(
        spontaneous_rate_hz * duration_s,
        size=clean_counts.shape,
    )
    rates = (detected + spontaneous) / duration_s
    if readout_noise_sd > 0.0:
        rates = rates + rng.normal(0.0, readout_noise_sd, size=rates.shape)
    return np.maximum(rates, 0.0)


def _source_splits(
    family_names: tuple[str, ...],
    config: SourceStatisticsConfig,
    rng: np.random.Generator,
) -> dict[tuple[str, int], tuple[NDArray[np.int64], NDArray[np.int64]]]:
    """Precompute splits so all readout-noise levels use identical exemplars."""

    splits = {}
    n_train = config.source_train_exemplars
    n_test = config.source_test_exemplars
    for repeat in range(config.source_split_repeats):
        for family_name in family_names:
            permutation = rng.permutation(config.n_source_exemplars)
            splits[(family_name, repeat)] = (
                permutation[:n_train],
                permutation[n_train : n_train + n_test],
            )
    return splits


def run_source_statistics_experiment(
    experiment: SourceStatisticsConfig | None = None,
    cochlea: CochleaConfig | None = None,
) -> list[dict[str, str | float]]:
    """Compare frozen-exemplar and source-family discrimination across duration."""

    experiment = experiment or SourceStatisticsConfig()
    experiment.validate()
    encoder = FrozenNoiseEncoder(cochlea)
    stimulus_rng = np.random.default_rng(experiment.seed)
    split_rng = np.random.default_rng(experiment.seed + 1)
    trial_rng = np.random.default_rng(experiment.seed + 2)
    family_names = tuple(family.name for family in experiment.families)

    results: list[dict[str, str | float]] = []
    for duration_ms in experiment.durations_ms:
        duration_s = duration_ms / 1_000.0
        n_samples = max(
            4,
            int(round(duration_s * encoder.config.sample_rate_hz)),
        )
        counts_by_family: dict[str, IntArray] = {}
        for family in experiment.families:
            counts_by_family[family.name] = np.stack(
                [
                    encoder.spike_counts(
                        generate_powerlaw_noise(
                            n_samples,
                            family.spectral_exponent,
                            stimulus_rng,
                            encoder.config.sample_rate_hz,
                            encoder.config.min_frequency_hz,
                            encoder.config.max_frequency_hz,
                        )
                    )
                    for _ in range(experiment.n_source_exemplars)
                ]
            )

        splits = _source_splits(family_names, experiment, split_rng)
        for noise_sd in experiment.readout_noise_sd:
            all_exemplar_scores: list[float] = []
            for family_name in family_names:
                family_scores = []
                family_counts = counts_by_family[family_name]
                for pair_index in range(experiment.frozen_pairs_per_family):
                    first_count = family_counts[2 * pair_index]
                    second_count = family_counts[2 * pair_index + 1]
                    n_train = experiment.frozen_train_trials_per_class
                    n_test = experiment.frozen_test_trials_per_class
                    first_trials = _noisy_rate_trials(
                        first_count,
                        duration_s,
                        n_train + n_test,
                        trial_rng,
                        experiment.spike_dropout_probability,
                        experiment.spontaneous_rate_hz,
                        noise_sd,
                    )
                    second_trials = _noisy_rate_trials(
                        second_count,
                        duration_s,
                        n_train + n_test,
                        trial_rng,
                        experiment.spike_dropout_probability,
                        experiment.spontaneous_rate_hz,
                        noise_sd,
                    )
                    family_scores.append(
                        _classification_accuracy(
                            first_trials[:n_train],
                            second_trials[:n_train],
                            first_trials[n_train:],
                            second_trials[n_train:],
                            experiment.classifier_ridge,
                        )
                    )
                all_exemplar_scores.extend(family_scores)
                mean, sem = _summarize(family_scores)
                results.append(
                    {
                        "task": "same_source_exemplar",
                        "source_a": family_name,
                        "source_b": "",
                        "duration_ms": float(duration_ms),
                        "readout_noise_sd": float(noise_sd),
                        "accuracy_mean": mean,
                        "accuracy_sem": sem,
                    }
                )

            mean, sem = _summarize(all_exemplar_scores)
            results.append(
                {
                    "task": "same_source_exemplar_mean",
                    "source_a": "all",
                    "source_b": "",
                    "duration_ms": float(duration_ms),
                    "readout_noise_sd": float(noise_sd),
                    "accuracy_mean": mean,
                    "accuracy_sem": sem,
                }
            )

            for first_name, second_name in experiment.comparisons:
                repeat_scores = []
                for repeat in range(experiment.source_split_repeats):
                    first_train, first_test = splits[(first_name, repeat)]
                    second_train, second_test = splits[(second_name, repeat)]
                    first_counts = counts_by_family[first_name]
                    second_counts = counts_by_family[second_name]
                    first_train_trials = _noisy_rate_observations(
                        first_counts[first_train],
                        duration_s,
                        trial_rng,
                        experiment.spike_dropout_probability,
                        experiment.spontaneous_rate_hz,
                        noise_sd,
                    )
                    second_train_trials = _noisy_rate_observations(
                        second_counts[second_train],
                        duration_s,
                        trial_rng,
                        experiment.spike_dropout_probability,
                        experiment.spontaneous_rate_hz,
                        noise_sd,
                    )
                    first_test_trials = _noisy_rate_observations(
                        first_counts[first_test],
                        duration_s,
                        trial_rng,
                        experiment.spike_dropout_probability,
                        experiment.spontaneous_rate_hz,
                        noise_sd,
                    )
                    second_test_trials = _noisy_rate_observations(
                        second_counts[second_test],
                        duration_s,
                        trial_rng,
                        experiment.spike_dropout_probability,
                        experiment.spontaneous_rate_hz,
                        noise_sd,
                    )
                    repeat_scores.append(
                        _classification_accuracy(
                            first_train_trials,
                            second_train_trials,
                            first_test_trials,
                            second_test_trials,
                            experiment.classifier_ridge,
                        )
                    )
                mean, sem = _summarize(repeat_scores)
                results.append(
                    {
                        "task": "different_source",
                        "source_a": first_name,
                        "source_b": second_name,
                        "duration_ms": float(duration_ms),
                        "readout_noise_sd": float(noise_sd),
                        "accuracy_mean": mean,
                        "accuracy_sem": sem,
                    }
                )
    return results
