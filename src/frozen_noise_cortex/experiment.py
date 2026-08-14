"""Monte Carlo experiment for two-alternative frozen-noise classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from .model import CochleaConfig, FrozenNoiseEncoder


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ExperimentConfig:
    durations_ms: tuple[int, ...] = (25, 50, 100, 200, 400, 800, 1_600)
    readout_noise_sd: tuple[float, ...] = (0.0, 10.0, 20.0, 40.0)
    n_frozen_pairs: int = 12
    train_trials_per_class: int = 32
    test_trials_per_class: int = 96
    spike_dropout_probability: float = 0.03
    spontaneous_rate_hz: float = 0.20
    classifier_ridge: float = 0.20
    seed: int = 7

    def validate(self) -> None:
        if not self.durations_ms or min(self.durations_ms) <= 0:
            raise ValueError("durations_ms must contain positive values")
        if not self.readout_noise_sd or min(self.readout_noise_sd) < 0.0:
            raise ValueError("readout_noise_sd cannot contain negative values")
        if self.n_frozen_pairs < 1:
            raise ValueError("n_frozen_pairs must be positive")
        if self.train_trials_per_class < 2 or self.test_trials_per_class < 1:
            raise ValueError("not enough train or test trials")
        if not 0.0 <= self.spike_dropout_probability < 1.0:
            raise ValueError("spike_dropout_probability must lie in [0, 1)")
        if self.spontaneous_rate_hz < 0.0:
            raise ValueError("spontaneous_rate_hz cannot be negative")
        if self.classifier_ridge <= 0.0:
            raise ValueError("classifier_ridge must be positive")


def _noisy_rate_trials(
    clean_counts: NDArray[np.int64],
    duration_s: float,
    n_trials: int,
    rng: np.random.Generator,
    dropout_probability: float,
    spontaneous_rate_hz: float,
    readout_noise_sd: float,
) -> FloatArray:
    """Corrupt spikes, pool over the trace, then add rate/readout noise."""

    detected = rng.binomial(
        clean_counts[None, :],
        1.0 - dropout_probability,
        size=(n_trials, clean_counts.size),
    )
    spontaneous = rng.poisson(
        spontaneous_rate_hz * duration_s,
        size=(n_trials, clean_counts.size),
    )
    rates = (detected + spontaneous) / duration_s
    if readout_noise_sd > 0.0:
        rates = rates + rng.normal(0.0, readout_noise_sd, size=rates.shape)
    return np.maximum(rates, 0.0)


def _fit_regularized_lda(
    class_zero: FloatArray,
    class_one: FloatArray,
    ridge_fraction: float,
) -> tuple[FloatArray, float]:
    """Fit a two-class linear discriminant with isotropic covariance shrinkage."""

    mean_zero = np.mean(class_zero, axis=0)
    mean_one = np.mean(class_one, axis=0)
    centered = np.vstack((class_zero - mean_zero, class_one - mean_one))
    covariance = centered.T @ centered / max(centered.shape[0] - 2, 1)
    mean_variance = max(float(np.trace(covariance) / covariance.shape[0]), 1e-9)
    covariance = covariance + ridge_fraction * mean_variance * np.eye(
        covariance.shape[0]
    )
    difference = mean_one - mean_zero
    weights = np.linalg.solve(covariance, difference)
    bias = -0.5 * float(np.dot(mean_zero + mean_one, weights))
    return weights, bias


def _classification_accuracy(
    train_zero: FloatArray,
    train_one: FloatArray,
    test_zero: FloatArray,
    test_one: FloatArray,
    ridge_fraction: float,
) -> float:
    weights, bias = _fit_regularized_lda(train_zero, train_one, ridge_fraction)
    predicted_zero = (test_zero @ weights + bias) >= 0.0
    predicted_one = (test_one @ weights + bias) >= 0.0
    correct = np.count_nonzero(~predicted_zero) + np.count_nonzero(predicted_one)
    return correct / (predicted_zero.size + predicted_one.size)


def _summarize(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(tuple(values), dtype=float)
    mean = float(np.mean(array))
    sem = float(np.std(array, ddof=1) / np.sqrt(array.size)) if array.size > 1 else 0.0
    return mean, sem


def run_experiment(
    experiment: ExperimentConfig | None = None,
    cochlea: CochleaConfig | None = None,
) -> list[dict[str, float]]:
    """Sweep duration and post-pooling noise; return tidy result dictionaries."""

    experiment = experiment or ExperimentConfig()
    experiment.validate()
    encoder = FrozenNoiseEncoder(cochlea)
    # Keep stimulus generation independent of the requested noise sweep. This
    # makes curves directly comparable when noise levels are added or removed.
    stimulus_rng = np.random.default_rng(experiment.seed)
    trial_rng = np.random.default_rng(experiment.seed + 1)

    accuracies: dict[tuple[int, float], list[float]] = {
        (duration, noise): []
        for duration in experiment.durations_ms
        for noise in experiment.readout_noise_sd
    }
    distances: dict[int, list[float]] = {
        duration: [] for duration in experiment.durations_ms
    }

    for duration_ms in experiment.durations_ms:
        duration_s = duration_ms / 1_000.0
        n_samples = max(4, int(round(duration_s * encoder.config.sample_rate_hz)))
        for _ in range(experiment.n_frozen_pairs):
            waveform_zero = stimulus_rng.normal(size=n_samples)
            waveform_one = stimulus_rng.normal(size=n_samples)
            counts_zero = encoder.spike_counts(waveform_zero)
            counts_one = encoder.spike_counts(waveform_one)
            clean_zero = counts_zero / duration_s
            clean_one = counts_one / duration_s
            distances[duration_ms].append(
                float(np.linalg.norm(clean_one - clean_zero) / np.sqrt(counts_zero.size))
            )

            for noise_sd in experiment.readout_noise_sd:
                n_train = experiment.train_trials_per_class
                n_test = experiment.test_trials_per_class
                trials_zero = _noisy_rate_trials(
                    counts_zero,
                    duration_s,
                    n_train + n_test,
                    trial_rng,
                    experiment.spike_dropout_probability,
                    experiment.spontaneous_rate_hz,
                    noise_sd,
                )
                trials_one = _noisy_rate_trials(
                    counts_one,
                    duration_s,
                    n_train + n_test,
                    trial_rng,
                    experiment.spike_dropout_probability,
                    experiment.spontaneous_rate_hz,
                    noise_sd,
                )
                accuracy = _classification_accuracy(
                    trials_zero[:n_train],
                    trials_one[:n_train],
                    trials_zero[n_train:],
                    trials_one[n_train:],
                    experiment.classifier_ridge,
                )
                accuracies[(duration_ms, noise_sd)].append(accuracy)

    results: list[dict[str, float]] = []
    for duration_ms in experiment.durations_ms:
        distance_mean, distance_sem = _summarize(distances[duration_ms])
        for noise_sd in experiment.readout_noise_sd:
            accuracy_mean, accuracy_sem = _summarize(
                accuracies[(duration_ms, noise_sd)]
            )
            results.append(
                {
                    "duration_ms": float(duration_ms),
                    "readout_noise_sd": float(noise_sd),
                    "accuracy_mean": accuracy_mean,
                    "accuracy_sem": accuracy_sem,
                    "clean_distance_mean": distance_mean,
                    "clean_distance_sem": distance_sem,
                }
            )
    return results
