"""Cross-correlation-based alignment for repeated pulse traces.

Repeated ultrasound bursts fired by one trigger can have small (<<1
sample) inter-pulse timing jitter that smears the naive average.
This module provides a NumPy-only helper to estimate per-pulse lag
via cross-correlation (with 3-point parabolic sub-sample refinement)
and produce time-aligned traces suitable for coherent averaging.
"""
from __future__ import annotations

import numpy as np


def _parabolic_refine(y_m1: float, y_0: float, y_p1: float) -> float:
    """Sub-sample offset (in [-0.5, 0.5]) of the max of a 3-point parabola."""
    denom = y_m1 - 2.0 * y_0 + y_p1
    if denom == 0.0:
        return 0.0
    return 0.5 * (y_m1 - y_p1) / denom


def estimate_lag_samples(x, ref, *, max_shift=None, refine=True):
    """Lag (in samples) of ``x`` relative to ``ref``.

    A positive lag means ``x`` is delayed relative to ``ref``: ``x[t]``
    corresponds roughly to ``ref[t - lag]``. To align ``x`` onto
    ``ref``'s time base you sample ``x`` at ``t + lag`` (see
    :func:`shift_trace`).

    Both inputs are mean-subtracted before correlation so any DC offset
    is ignored.
    """
    x = np.asarray(x, dtype=float) - float(np.mean(x))
    ref = np.asarray(ref, dtype=float) - float(np.mean(ref))
    n = x.size
    if n == 0 or ref.size == 0:
        return 0.0
    corr = np.correlate(x, ref, mode="full")
    lags = np.arange(corr.size) - (ref.size - 1)
    if max_shift is not None:
        mask = np.abs(lags) <= int(max_shift)
        if not mask.any():
            return 0.0
        corr = corr[mask]
        lags = lags[mask]
    k = int(np.argmax(corr))
    lag_int = int(lags[k])
    if not refine or k == 0 or k == corr.size - 1:
        return float(lag_int)
    frac = _parabolic_refine(float(corr[k - 1]), float(corr[k]), float(corr[k + 1]))
    return float(lag_int + frac)


def shift_trace(x, lag_samples):
    """Return ``x`` advanced by ``lag_samples`` using linear interpolation.

    ``lag_samples`` may be fractional. Samples that would map to indices
    outside the original trace are filled with the trace mean, which
    keeps DC unchanged and avoids creating spurious edges.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n == 0:
        return x.copy()
    idx = np.arange(n) + float(lag_samples)
    left = np.floor(idx).astype(int)
    frac = idx - left
    fill = float(np.mean(x))
    out = np.full(n, fill, dtype=float)
    valid = (left >= 0) & (left + 1 < n)
    out[valid] = (1.0 - frac[valid]) * x[left[valid]] + frac[valid] * x[left[valid] + 1]
    # Handle the exact-endpoint case where left == n-1 and frac == 0.
    exact = (left == n - 1) & (frac == 0.0)
    out[exact] = x[n - 1]
    return out


def align_pulse_traces(traces, *, dt_s, max_shift_samples=None,
                       reference="first", refine=True):
    """Align repeated pulse traces by cross-correlation.

    Parameters
    ----------
    traces : array_like, shape ``(n_pulses, n_samples)``
        Each row is one captured pulse on a common time base.
    dt_s : float
        Sample interval in seconds.
    max_shift_samples : int, optional
        Bound the lag search to ``+/-`` this many samples. ``None``
        (default) searches the full correlation range.
    reference : {"first", "mean"} or int
        Row to align onto. ``"mean"`` uses the naive average as the
        reference, which is more robust when pulse 0 is atypical but
        assumes jitter is small compared to the pulse duration.
    refine : bool
        If ``True`` (default), apply 3-point parabolic sub-sample
        refinement to each lag estimate.

    Returns
    -------
    aligned : ndarray, shape ``(n_pulses, n_samples)``
        Traces resampled onto the reference time base.
    lags_s : ndarray, shape ``(n_pulses,)``
        Estimated per-pulse lag in seconds. Positive means the pulse
        arrived late relative to the reference.
    """
    traces = np.asarray(traces, dtype=float)
    if traces.ndim != 2:
        raise ValueError("traces must be 2-D (n_pulses, n_samples)")
    n_pulses, _ = traces.shape
    if n_pulses == 0:
        return traces.copy(), np.zeros(0, dtype=float)

    if reference == "first":
        ref = traces[0]
    elif reference == "mean":
        ref = traces.mean(axis=0)
    elif isinstance(reference, (int, np.integer)):
        ref = traces[int(reference)]
    else:
        raise ValueError(f"unknown reference: {reference!r}")

    lags = np.zeros(n_pulses, dtype=float)
    aligned = np.empty_like(traces)
    for i in range(n_pulses):
        lag = estimate_lag_samples(
            traces[i], ref,
            max_shift=max_shift_samples,
            refine=refine,
        )
        lags[i] = lag
        aligned[i] = shift_trace(traces[i], lag)
    return aligned, lags * float(dt_s)


__all__ = [
    "estimate_lag_samples",
    "shift_trace",
    "align_pulse_traces",
]
