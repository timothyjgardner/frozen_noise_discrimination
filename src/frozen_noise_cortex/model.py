"""Cochlea-like encoding and threshold-crossing spike generation.

The filterbank is deliberately lightweight: ERB-spaced, overlapping frequency
responses are applied in the Fourier domain, so NumPy is the only dependency.
It is intended as a transparent hypothesis-testing model, not a detailed model
of auditory-nerve physiology.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


def hz_to_erb_number(frequency_hz: FloatArray) -> FloatArray:
    """Convert frequency in hertz to the Glasberg-Moore ERB-number scale."""

    return 21.4 * np.log10(1.0 + 0.00437 * frequency_hz)


def erb_number_to_hz(erb_number: FloatArray) -> FloatArray:
    """Convert Glasberg-Moore ERB number to frequency in hertz."""

    return (10.0 ** (erb_number / 21.4) - 1.0) / 0.00437


def erb_bandwidth(frequency_hz: FloatArray) -> FloatArray:
    """Equivalent rectangular bandwidth in hertz."""

    return 24.7 * (1.0 + 0.00437 * frequency_hz)


@dataclass(frozen=True)
class CochleaConfig:
    sample_rate_hz: int = 16_000
    n_channels: int = 16
    min_frequency_hz: float = 200.0
    max_frequency_hz: float = 7_000.0
    bandwidth_scale: float = 1.5
    envelope_threshold: float = 1.45
    refractory_ms: float = 1.0
    normalize_stimulus_rms: bool = True

    def validate(self) -> None:
        nyquist = self.sample_rate_hz / 2.0
        if self.n_channels < 2:
            raise ValueError("n_channels must be at least 2")
        if not 0.0 < self.min_frequency_hz < self.max_frequency_hz < nyquist:
            raise ValueError("filter frequencies must lie between 0 and Nyquist")
        if self.bandwidth_scale <= 0.0:
            raise ValueError("bandwidth_scale must be positive")
        if self.envelope_threshold <= 0.0:
            raise ValueError("envelope_threshold must be positive")
        if self.refractory_ms < 0.0:
            raise ValueError("refractory_ms cannot be negative")


class FrozenNoiseEncoder:
    """Encode a waveform as threshold-crossing spikes in cochlear channels."""

    def __init__(self, config: CochleaConfig | None = None):
        self.config = config or CochleaConfig()
        self.config.validate()
        erb_limits = hz_to_erb_number(
            np.array(
                [self.config.min_frequency_hz, self.config.max_frequency_hz],
                dtype=float,
            )
        )
        self.center_frequencies_hz = erb_number_to_hz(
            np.linspace(erb_limits[0], erb_limits[1], self.config.n_channels)
        )

    def _filter_responses(self, n_samples: int) -> FloatArray:
        frequencies = np.fft.rfftfreq(n_samples, 1.0 / self.config.sample_rate_hz)
        centers = self.center_frequencies_hz[:, None]
        bandwidths = (
            self.config.bandwidth_scale * erb_bandwidth(centers)
        )

        # A smooth fourth-order auditory-band shape. Normalizing each row by
        # its mean-square gain gives equal expected RMS for ideal white noise.
        distance = (frequencies[None, :] - centers) / (0.5 * bandwidths)
        responses = 1.0 / np.sqrt(1.0 + distance**8)
        responses[:, 0] = 0.0
        gains = np.sqrt(np.mean(responses**2, axis=1, keepdims=True))
        return responses / np.maximum(gains, np.finfo(float).eps)

    @staticmethod
    def _analytic_envelope(signals: FloatArray) -> FloatArray:
        """Return Hilbert envelopes without requiring SciPy."""

        n_samples = signals.shape[-1]
        spectrum = np.fft.fft(signals, axis=-1)
        multiplier = np.zeros(n_samples, dtype=float)
        multiplier[0] = 1.0
        if n_samples % 2 == 0:
            multiplier[1 : n_samples // 2] = 2.0
            multiplier[n_samples // 2] = 1.0
        else:
            multiplier[1 : (n_samples + 1) // 2] = 2.0
        analytic = np.fft.ifft(spectrum * multiplier[None, :], axis=-1)
        return np.abs(analytic)

    def filter(self, waveform: FloatArray) -> FloatArray:
        """Return one approximately equal-white-noise-gain waveform per channel."""

        waveform = np.asarray(waveform, dtype=float)
        if waveform.ndim != 1 or waveform.size < 4:
            raise ValueError("waveform must be a one-dimensional array of length >= 4")
        waveform = waveform - np.mean(waveform)
        if self.config.normalize_stimulus_rms:
            rms = np.sqrt(np.mean(waveform**2))
            waveform = waveform / max(rms, np.finfo(float).eps)
        spectrum = np.fft.rfft(waveform)
        filtered = np.fft.irfft(
            self._filter_responses(waveform.size) * spectrum[None, :],
            n=waveform.size,
            axis=-1,
        )
        return filtered

    def encode(self, waveform: FloatArray) -> BoolArray:
        """Return a channels-by-time Boolean threshold-crossing spike train."""

        envelope = self._analytic_envelope(self.filter(waveform))
        above = envelope >= self.config.envelope_threshold
        crossings = np.zeros_like(above, dtype=bool)
        crossings[:, 0] = above[:, 0]
        crossings[:, 1:] = above[:, 1:] & ~above[:, :-1]

        refractory_samples = int(
            round(self.config.refractory_ms * self.config.sample_rate_hz / 1_000.0)
        )
        if refractory_samples <= 1:
            return crossings

        spikes = np.zeros_like(crossings, dtype=bool)
        for channel in range(crossings.shape[0]):
            last_spike = -refractory_samples
            for event in np.flatnonzero(crossings[channel]):
                if event - last_spike >= refractory_samples:
                    spikes[channel, event] = True
                    last_spike = int(event)
        return spikes

    def spike_counts(self, waveform: FloatArray) -> NDArray[np.int64]:
        """Return the number of threshold crossings in each channel."""

        return np.sum(self.encode(waveform), axis=1, dtype=np.int64)

    def rate_code(self, waveform: FloatArray) -> FloatArray:
        """Return the time-averaged spike rate in every channel, in spikes/s."""

        duration_s = np.asarray(waveform).size / self.config.sample_rate_hz
        return self.spike_counts(waveform).astype(float) / duration_s

