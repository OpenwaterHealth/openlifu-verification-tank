"""Drive frequency the TX hardware actually produces.

The TX7332 pattern generator builds one drive period out of whole
beamformer-clock samples. ``openlifu_sdk.io.LIFUTXDevice.calc_pulse_pattern``
rounds the period *down* (``int(clk / f)``), so the emitted tone is
``bf_clk / 2**clk_div_n / period_samples``, which is generally not the
commanded frequency. With the default 10 MHz clock:

* 400 kHz module, +/-25 kHz sweep: only four tones are emitted --
  384.6 kHz (375, 380 commanded), 400.0 (385-400), 416.7 (405-415) and
  434.8 (420, 425).
* 155 kHz module: every point lands 0.1-2.4 kHz high (155 -> 156.25,
  170 -> 172.4).

:func:`realized_frequency_hz` predicts the emitted tone from the SDK
pattern (deterministic, what the tank uses when configured to). It does
not account for any error in the TX clock itself.
:func:`estimate_tone_frequency` measures it from a captured waveform; it
is recorded for bench validation only, because it is unreliable on
short off-resonance bursts (the received waveform then rings at the
transducer's own resonance rather than settling to the drive tone).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

#: Beamformer clock used by the SDK unless the TX registers say otherwise.
DEFAULT_BF_CLK_HZ = 10e6


def realized_frequency_hz(commanded_hz, *, bf_clk_hz: float = DEFAULT_BF_CLK_HZ):
    """Tone (Hz) the TX pattern generator emits when ``commanded_hz`` is set.

    Uses the SDK's own ``calc_pulse_pattern`` so the prediction follows
    whatever the installed SDK will program. Accepts a scalar or an
    array (returned with the same shape); non-finite or non-positive
    entries are passed through unchanged.
    """
    from openlifu_sdk.io.LIFUTXDevice import calc_pulse_pattern

    arr = np.asarray(commanded_hz, dtype=float)
    out = arr.copy()
    for idx, f in np.ndenumerate(arr):
        if not np.isfinite(f) or f <= 0:
            continue
        pattern = calc_pulse_pattern(float(f), bf_clk=float(bf_clk_hz))
        period_samples = len(pattern["y"])
        out[idx] = float(bf_clk_hz) / (2 ** int(pattern["clk_div_n"])) / period_samples
    return float(out) if out.ndim == 0 else out


def estimate_tone_frequency(t_s, y, f_guess_hz: float, *,
                            burst_s: Optional[float] = None,
                            skip_s: float = 8e-6,
                            tail_s: float = 2e-6,
                            span: float = 0.15,
                            step_hz: float = 50.0) -> float:
    """Least-squares tone frequency (Hz) of the steady part of a burst.

    The burst onset is the first sample above 20 % of the peak (after
    removing the pre-burst baseline). The fit window starts ``skip_s``
    after onset and ends ``tail_s`` before ``onset + burst_s`` (or at
    the end of the capture). Within ``f_guess_hz * (1 +/- span)`` the
    frequency minimising the residual of a fundamental + 2nd/3rd
    harmonic + linear-trend model is found on a ``step_hz`` grid and
    refined parabolically. Returns ``nan`` if the window is too short.
    """
    t = np.asarray(t_s, dtype=float)
    y = np.asarray(y, dtype=float)
    if t.size < 16 or t.size != y.size or not np.all(np.isfinite(y)):
        return float("nan")
    pre = t < t[0] + 0.1 * (t[-1] - t[0])
    y = y - (np.median(y[pre]) if pre.sum() >= 4 else np.median(y))
    a = np.abs(y)
    if a.max() <= 0:
        return float("nan")
    onset = t[int(np.argmax(a > 0.2 * a.max()))]
    stop = t[-1] if burst_s is None else min(t[-1], onset + burst_s - tail_s)
    m = (t >= onset + skip_s) & (t <= stop)
    if m.sum() < 16:
        return float("nan")
    tt = t[m] - t[m][0]
    yy = y[m]
    if (tt[-1] * f_guess_hz) < 2.0:  # need a couple of cycles
        return float("nan")
    grid = np.arange(f_guess_hz * (1 - span), f_guess_hz * (1 + span), step_hz)
    res = np.empty(grid.size)
    ones = np.ones_like(tt)
    for i, f in enumerate(grid):
        w = 2 * np.pi * f * tt
        X = np.column_stack([np.sin(w), np.cos(w), np.sin(2 * w), np.cos(2 * w),
                             np.sin(3 * w), np.cos(3 * w), ones, tt])
        coef, *_ = np.linalg.lstsq(X, yy, rcond=None)
        r = yy - X @ coef
        res[i] = r @ r
    i = int(np.argmin(res))
    if 0 < i < grid.size - 1:
        den = res[i - 1] - 2 * res[i] + res[i + 1]
        if den > 0:
            return float(grid[i] + 0.5 * (res[i - 1] - res[i + 1]) / den * step_hz)
    return float(grid[i])
