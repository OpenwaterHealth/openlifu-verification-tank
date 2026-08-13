"""Robust first-arrival-time detection for hydrophone traces.

The naive "first sample above X% of peak" arrival detector locks onto
whichever edge of the envelope happens to rise first, which is
noise-sensitive and biased low by the slow leading edge of the burst
envelope. This module implements a more physical picker that
leverages the predictable sinusoidal shape of the RF signal:

1. Aggregate (average) many identical pulses so the SNR is high.
2. Find the first *local maximum of the raw trace* (a positive
   carrier crest) that exceeds ``min_prominence_frac`` of the trace's
   global positive peak. Latching onto a peak rather than an envelope
   threshold uses the tight sinusoidal structure of the burst as its
   own signature.
3. That first positive crest sits a quarter-cycle after the
   sinusoid's leading zero-crossing (the moment the wavefront
   "actually arrived"), so subtract ``250 / frequency_kHz`` \u00b5s.

Combined with a plane-wave excitation (all elements fire
simultaneously, so ``max(tof) == min(tof) == 0``), this gives a direct
distance-of-flight measurement: ``distance = arrival_us * SoS``.

For focused excitations the elements are staggered; the last element
to fire is the one directly above the hydrophone, so its time-of-flight
determines the range. The reported arrival time is biased *earlier* by
the outer elements firing first along a longer path, so callers should
subtract the max element delay before multiplying by the speed of
sound.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ArrivalResult:
    """Output of :func:`find_first_arrival_us`.

    Attributes:
        first_peak_us: Time (\u00b5s) of the first RF local maximum
            after ``skip_us`` that exceeded the prominence threshold.
        first_peak_value: Trace value at that peak, in the trace's
            native units.
        arrival_us: ``first_peak_us`` minus a quarter carrier cycle
            \u2014 the estimated moment the sinusoidal wavefront's
            leading zero-crossing arrived.
        quarter_cycle_us: The ``250 / frequency_kHz`` correction
            applied.
        threshold: The prominence threshold that was applied.
        skip_us: The skip window that was actually used.
    """
    first_peak_us: float
    first_peak_value: float
    arrival_us: float
    quarter_cycle_us: float
    threshold: float
    skip_us: float


def find_first_arrival_us(t_us: np.ndarray,
                          trace: np.ndarray,
                          *,
                          frequency_kHz: float,
                          skip_us: float = 12.0,
                          min_prominence_frac: float = 0.125,
                          ) -> Optional[ArrivalResult]:
    """Locate the first arrival of a narrow-band pulse.

    Finds the first positive local maximum of the raw trace after
    ``skip_us`` whose amplitude exceeds ``min_prominence_frac`` times
    the global positive-peak of the (post-skip) trace, then subtracts
    a quarter carrier cycle to map that crest back to the sinusoid's
    leading zero-crossing.

    Args:
        t_us: Sample-time axis in \u00b5s (monotonically increasing).
        trace: 1-D trace (native units \u2014 mV or Pa, doesn't
            matter, the picker is amplitude-independent).
        frequency_kHz: Carrier frequency in kHz; used for the
            quarter-cycle correction.
        skip_us: Ignore everything before this time (\u00b5s). Defaults
            to 12 \u00b5s to skip the trigger flash / cross-talk.
        min_prominence_frac: Local maxima whose value is below
            ``min_prominence_frac`` * global-positive-peak are rejected
            as noise. Default 0.125 (12.5% of the burst peak) \u2014 well
            above typical noise floors for a 32-pulse coherent average,
            and comfortably below the leading crest of a boxcar-onset
            tone-burst.

    Returns:
        :class:`ArrivalResult`, or ``None`` if no qualifying peak is
        found in the window.
    """
    t_us = np.asarray(t_us, dtype=float)
    trace = np.asarray(trace, dtype=float)
    if t_us.size != trace.size or t_us.size < 3:
        return None

    mask = t_us >= float(skip_us)
    if mask.sum() < 3:
        return None
    t_win = t_us[mask]
    x_win = trace[mask]

    global_peak = float(np.max(x_win))
    if global_peak <= 0:
        return None
    threshold = float(min_prominence_frac) * global_peak

    # First interior positive local maximum above the prominence
    # threshold. Using the raw RF signal (not the envelope) latches
    # onto the sharp carrier crest, which has a much tighter
    # localization than the slowly-rising envelope leading edge.
    idx_peak: Optional[int] = None
    for i in range(1, x_win.size - 1):
        if x_win[i] < threshold:
            continue
        if x_win[i] >= x_win[i - 1] and x_win[i] >= x_win[i + 1]:
            idx_peak = i
            break
    if idx_peak is None:
        return None

    # Parabolic refinement of the RF peak location (sub-sample).
    y_m, y_0, y_p = x_win[idx_peak - 1], x_win[idx_peak], x_win[idx_peak + 1]
    denom = y_m - 2.0 * y_0 + y_p
    if denom != 0.0:
        frac = 0.5 * (y_m - y_p) / denom
        # Clip to a physically reasonable +/- 0.5 sample shift.
        frac = float(np.clip(frac, -0.5, 0.5))
    else:
        frac = 0.0
    dt_us = float(t_win[1] - t_win[0])
    first_peak_us = float(t_win[idx_peak]) + frac * dt_us

    # 1 / freq_kHz = period in ms; * 1000 => \u00b5s; / 4 => quarter cycle
    # => quarter_cycle_us = 250 / freq_kHz.
    quarter_cycle_us = 250.0 / float(frequency_kHz)
    arrival_us = first_peak_us - quarter_cycle_us

    return ArrivalResult(
        first_peak_us=first_peak_us,
        first_peak_value=float(y_0),
        arrival_us=arrival_us,
        quarter_cycle_us=quarter_cycle_us,
        threshold=threshold,
        skip_us=float(skip_us),
    )

