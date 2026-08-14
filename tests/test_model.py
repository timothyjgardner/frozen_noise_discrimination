from __future__ import annotations

import unittest

import numpy as np

from frozen_noise_cortex.experiment import ExperimentConfig, run_experiment
from frozen_noise_cortex.model import CochleaConfig, FrozenNoiseEncoder


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


if __name__ == "__main__":
    unittest.main()

