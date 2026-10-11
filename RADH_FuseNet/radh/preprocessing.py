from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import numpy as np
from scipy import signal


@dataclass
class RobustNormalizer:
    """A training-partition-only per-channel median / interquartile range."""

    center: np.ndarray
    scale: np.ndarray
    clip: float | None = 20.0

    @classmethod
    def fit(cls, x: np.ndarray, mask: np.ndarray, clip: float | None = 20.0) -> "RobustNormalizer":
        if x.ndim != 3 or mask.shape != x.shape:
            raise ValueError("Expected x and mask shaped [windows, channels, samples].")
        centers, scales = [], []
        for channel in range(x.shape[1]):
            values = x[:, channel][mask[:, channel] & np.isfinite(x[:, channel])]
            if not len(values):
                centers.append(0.0)
                scales.append(1.0)
                continue
            q25, median, q75 = np.percentile(values, [25, 50, 75])
            centers.append(float(median))
            scales.append(max(float(q75 - q25), 1e-6))
        return cls(np.asarray(centers, dtype=np.float32), np.asarray(scales, dtype=np.float32), clip)

    def transform(self, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        valid = mask & np.isfinite(x)
        result = (np.where(valid, x, self.center[None, :, None]) - self.center[None, :, None]) / self.scale[None, :, None]
        if self.clip is not None:
            result = np.clip(result, -self.clip, self.clip)
        return np.where(valid, result, 0).astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {"center": self.center.tolist(), "scale": self.scale.tolist(), "clip": self.clip}

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> "RobustNormalizer":
        return cls(np.asarray(state["center"], np.float32), np.asarray(state["scale"], np.float32), state.get("clip"))


def contiguous_runs(valid: np.ndarray) -> list[tuple[int, int]]:
    """Return half-open contiguous valid intervals."""
    edges = np.diff(np.pad(np.asarray(valid, dtype=np.int8), (1, 1)))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def resample_masked(
    x: np.ndarray, mask: np.ndarray, source_hz: float, target_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Anti-alias each observed run separately; never interpolate across long gaps.

    Masked intervals remain missing. Each run uses scipy's polyphase FIR, and
    boundary support is conservative: a target point outside a run is masked.
    This is an offline window preparation function, not a causal streaming filter.
    """
    if source_hz <= 0 or target_hz <= 0:
        raise ValueError("Sampling rates must be positive.")
    x = np.asarray(x, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool) & np.isfinite(x)
    if x.ndim != 2 or mask.shape != x.shape:
        raise ValueError("Expected [channels, samples] arrays.")
    if np.isclose(source_hz, target_hz):
        return np.where(mask, x, 0).astype(np.float32), mask.copy()
    ratio = Fraction(target_hz / source_hz).limit_denominator(10000)
    up, down = ratio.numerator, ratio.denominator
    n_out = max(1, int(np.ceil(x.shape[-1] * up / down)))
    output = np.zeros((x.shape[0], n_out), dtype=np.float32)
    observed = np.zeros_like(output, dtype=bool)
    for channel in range(x.shape[0]):
        for start, stop in contiguous_runs(mask[channel]):
            if stop - start < 2:
                continue
            segment = signal.resample_poly(x[channel, start:stop], up, down, padtype="line")
            grid = np.arange(n_out) / target_hz
            local_time = start / source_hz + np.arange(len(segment)) / target_hz
            selected = (grid >= start / source_hz) & (grid <= (stop - 1) / source_hz)
            output[channel, selected] = np.interp(grid[selected], local_time, segment)
            observed[channel, selected] = True
    return output, observed


def filter_observed_runs(
    x: np.ndarray, mask: np.ndarray, sample_hz: float,
    pulse_band: tuple[float, float] | None = None,
    humidity_detrend_seconds: float | None = None,
    wavelet: bool = False,
) -> np.ndarray:
    """Optional morphology-preserving offline filtering, restricted to valid runs.

    Pulse bandpass is applied to channels 0/1 with zero-phase SOS filtering.
    A slow median baseline is optionally subtracted from humidity channels 3/4.
    Wavelet denoising requires PyWavelets and uses soft universal thresholding.
    All options default off: the manuscript does not specify their cutoffs.
    """
    result = np.where(mask, x, 0).astype(np.float32)
    for channel in range(x.shape[0]):
        for start, stop in contiguous_runs(mask[channel]):
            values = result[channel, start:stop].copy()
            if channel < 2 and pulse_band is not None:
                low, high = pulse_band
                if not (0 < low < high < sample_hz / 2):
                    raise ValueError("Pulse band must be strictly inside (0, Nyquist).")
                sos = signal.butter(3, [low, high], btype="bandpass", fs=sample_hz, output="sos")
                if len(values) > 24:
                    values = signal.sosfiltfilt(sos, values)
            if wavelet and channel < 2 and len(values) >= 32:
                try:
                    import pywt
                except ImportError as error:
                    raise ImportError("Install PyWavelets to request wavelet denoising.") from error
                coeffs = pywt.wavedec(values, "sym4", mode="symmetric")
                sigma = np.median(np.abs(coeffs[-1])) / 0.6745
                threshold = sigma * np.sqrt(2 * np.log(len(values)))
                values = pywt.waverec([coeffs[0]] + [pywt.threshold(c, threshold, mode="soft") for c in coeffs[1:]], "sym4", mode="symmetric")[:len(values)]
            if channel >= 3 and humidity_detrend_seconds is not None:
                from scipy.ndimage import median_filter
                width = max(3, int(humidity_detrend_seconds * sample_hz) | 1)
                values = values - median_filter(values, size=width, mode="nearest")
            result[channel, start:stop] = values
    return np.where(mask, result, 0).astype(np.float32)


def align_pair_by_correlation(
    reference: np.ndarray, source: np.ndarray, reference_mask: np.ndarray,
    source_mask: np.ndarray, maximum_lag: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Optional bounded whole-window refinement, without circular wrapping.

    Only enable for signals that actually share landmarks (e.g. pulse sources).
    This generic correlation implementation is an explicit non-causal policy;
    it is not claimed to reproduce an unspecified landmark detector in the paper.
    """
    if maximum_lag < 0:
        raise ValueError("maximum_lag must be nonnegative.")
    length = len(source)
    best_lag, best_score = 0, -np.inf
    for lag in range(-maximum_lag, maximum_lag + 1):
        dest = np.arange(max(0, lag), min(length, length + lag))
        src = dest - lag
        valid = reference_mask[dest] & source_mask[src]
        if valid.sum() < 8:
            continue
        a, b = reference[dest[valid]], source[src[valid]]
        a, b = a - a.mean(), b - b.mean()
        score = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
        if score > best_score:
            best_lag, best_score = lag, score
    dest = np.arange(max(0, best_lag), min(length, length + best_lag))
    src = dest - best_lag
    output, output_mask = np.zeros_like(source), np.zeros_like(source_mask)
    output[dest] = source[src]
    output_mask[dest] = source_mask[src]
    return output, output_mask, best_lag
