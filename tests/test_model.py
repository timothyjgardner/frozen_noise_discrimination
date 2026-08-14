from __future__ import annotations

import unittest

import numpy as np

from frozen_noise_cortex.experiment import ExperimentConfig, run_experiment
from frozen_noise_cortex.model import CochleaConfig, FrozenNoiseEncoder
from frozen_noise_cortex.source_statistics import (
    NoiseFamily,
    SourceStatisticsConfig,
    generate_powerlaw_noise,
    run_source_statistics_experiment,
)


class EncoderTests(unittest.TestCase):
    def test_encoder_shape_and_reproducibility(self) -> None:
        config = CochleaConfig(sample_rate_hz=8_000, n_channels=8, max_frequency_hz=3_500)
        encoder = FrozenNoiseEncoder(config)
        waveform = np.random.default_rng(2).normal(size=800)
        first = encoder.encode(waveform)
        second = encoder.encode(waveform)
        self.assertEqual(first.shape, (8, 800))
        np.testing.assert_array_equal(first, second)

    def test_rms_normalization_removes_level_cue(self) -> None:
        config = CochleaConfig(sample_rate_hz=8_000, n_channels=8, max_frequency_hz=3_500)
        encoder = FrozenNoiseEncoder(config)
        waveform = np.random.default_rng(3).normal(size=1_600)
        np.testing.assert_array_equal(
            encoder.spike_counts(waveform), encoder.spike_counts(4.0 * waveform)
        )


class ExperimentTests(unittest.TestCase):
    def test_experiment_returns_tidy_rows(self) -> None:
        experiment = ExperimentConfig(
            durations_ms=(25, 100),
            readout_noise_sd=(0.0, 2.0),
            n_frozen_pairs=2,
            train_trials_per_class=4,
            test_trials_per_class=6,
            seed=11,
        )
        cochlea = CochleaConfig(
            sample_rate_hz=8_000,
            n_channels=6,
            max_frequency_hz=3_500,
        )
        results = run_experiment(experiment, cochlea)
        self.assertEqual(len(results), 4)
        for row in results:
            self.assertGreaterEqual(row["accuracy_mean"], 0.0)
            self.assertLessEqual(row["accuracy_mean"], 1.0)


class SourceStatisticsTests(unittest.TestCase):
    def test_powerlaw_noise_is_reproducible_and_finite(self) -> None:
        first = generate_powerlaw_noise(
            1_600,
            1.0,
            np.random.default_rng(17),
            sample_rate_hz=8_000,
            low_frequency_hz=200.0,
            high_frequency_hz=3_500.0,
        )
        second = generate_powerlaw_noise(
            1_600,
            1.0,
            np.random.default_rng(17),
            sample_rate_hz=8_000,
            low_frequency_hz=200.0,
            high_frequency_hz=3_500.0,
        )
        self.assertEqual(first.shape, (1_600,))
        self.assertTrue(np.all(np.isfinite(first)))
        np.testing.assert_array_equal(first, second)

    def test_source_experiment_returns_both_tasks(self) -> None:
        experiment = SourceStatisticsConfig(
            durations_ms=(25, 100),
            readout_noise_sd=(20.0,),
            families=(
                NoiseFamily("white", 0.0),
                NoiseFamily("pink", 1.0),
            ),
            comparisons=(("white", "pink"),),
            n_source_exemplars=16,
            source_split_repeats=2,
            source_train_exemplars=4,
            source_test_exemplars=8,
            frozen_pairs_per_family=2,
            frozen_train_trials_per_class=4,
            frozen_test_trials_per_class=6,
            seed=19,
        )
        cochlea = CochleaConfig(
            sample_rate_hz=8_000,
            n_channels=6,
            max_frequency_hz=3_500,
        )
        results = run_source_statistics_experiment(experiment, cochlea)
        self.assertEqual(len(results), 8)
        tasks = {row["task"] for row in results}
        self.assertEqual(
            tasks,
            {
                "same_source_exemplar",
                "same_source_exemplar_mean",
                "different_source",
            },
        )
        for row in results:
            self.assertGreaterEqual(float(row["accuracy_mean"]), 0.0)
            self.assertLessEqual(float(row["accuracy_mean"]), 1.0)


if __name__ == "__main__":
    unittest.main()
