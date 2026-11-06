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
3. Walk back from that crest to the onset: the zero crossing at the
   start of its lobe, extended back over any immediately preceding
   lobes (either polarity) that still rise clearly above the noise
   floor. The onset is measured rather than assumed to be a fixed
   quarter-cycle before the crest, because short bursts and the
   transducer's ring-up distort the first cycle (the first crest of
   a ~1-cycle 155 kHz burst lands well under T/4 after the onset, and
   a sub-threshold first cycle puts the first qualifying crest a
   whole period late).
4. Guard against a flat foot: a small residual baseline offset under
   a many-pulse median can keep the sign from flipping before the
   burst, stretching the first lobe back over a flat, near-zero
   stretch. When the lobe's zero crossing sits more than T/8 before
   its leading edge extrapolated to zero (through the 10 % and 50 %
   levels), the extrapolated edge is the onset.

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
        first_peak_us: Time (us) of the first RF local maximum
            after ``skip_us`` that exceeded the prominence threshold.
        first_peak_value: Trace value at that peak, in the trace's
            native units.
        arrival_us: The onset -- zero crossing at the start of the
            earliest above-noise lobe leading into the first peak.
        onset_to_peak_us: ``first_peak_us - arrival_us``. For a clean
            hard-onset sinusoid this is a quarter cycle
            (``250 / frequency_kHz``); a different value shows how far
            the real waveform departs from that.
        threshold: The prominence threshold that was applied.
        noise_gate: Lobes preceding the first peak are counted as
            signal only if they exceed this (``noise_k`` x robust
            noise RMS of the pre-arrival window).
        skip_us: The skip window that was actually used.
    """
    first_peak_us: float
    first_peak_value: float
    arrival_us: float
    onset_to_peak_us: float
    threshold: float
    noise_gate: float
    skip_us: float


def find_first_arrival_us(t_us: np.ndarray,
                          trace: np.ndarray,
                          *,
                          frequency_kHz: float,
                          skip_us: Optional[float] = None,
                          min_prominence_frac: float = 0.05,
                          noise_k: float = 4.0,
                          ) -> Optional[ArrivalResult]:
    """Locate the first arrival of a narrow-band pulse.

    Finds the first positive local maximum of the raw trace after
    ``skip_us`` whose amplitude exceeds ``min_prominence_frac`` times
    the global positive-peak of the (post-skip) trace, then walks back
    to the onset: the zero crossing at the start of that crest's lobe,
    extended over preceding lobes (either polarity) whose peak exceeds
    ``noise_k`` x the noise RMS. Preceding lobes wider than one carrier
    period (baseline drift, not carrier) end the walk, and it never
    reaches more than two periods before the first peak's lobe. A lobe
    whose zero crossing precedes its extrapolated leading edge by more
    than T/8 (a flat foot) uses the extrapolated edge and ends the walk.

    Args:
        t_us: Sample-time axis in us (monotonically increasing).
        trace: 1-D trace (native units -- mV or Pa, doesn't
            matter, the picker is amplitude-independent).
        frequency_kHz: Carrier frequency in kHz; bounds the lobe
            width and walk-back distance.
        skip_us: Ignore everything before this time (us). ``None``
            (default) searches from the start of the trace; callers
            that know the pulse geometry (e.g.
            :meth:`VerificationTank.calibrate_hydrophone_depth`)
            should pass a window that clears the transmit EM pickup.
        min_prominence_frac: Local maxima whose value is below
            ``min_prominence_frac`` * global-positive-peak are rejected
            as noise. Default 0.05 (5% of the burst peak) -- well
            above typical noise floors for a 32-pulse median,
            and comfortably below the leading crest of a boxcar-onset
            tone-burst.
        noise_k: Walk-back gate in units of the robust (MAD) noise
            RMS of the window between ``skip_us`` and the first
            peak's lobe.

    Returns:
        :class:`ArrivalResult`, or ``None`` if no qualifying peak is
        found in the window.
    """
    t_us = np.asarray(t_us, dtype=float)
    trace = np.asarray(trace, dtype=float)
    if t_us.size != trace.size or t_us.size < 3:
        return None

    if skip_us is None:
        skip_us = float(t_us[0])
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

    sign = np.sign(x_win)

    def lobe_start(i: int) -> int:
        """First index of the same-sign run containing ``i``."""
        while i > 0 and sign[i - 1] == sign[i]:
            i -= 1
        return i

    period_us = 1000.0 / float(frequency_kHz)
    period_samples = period_us / dt_us

    def level_crossing_us(k: int, lo: int, level: float) -> Optional[float]:
        """Time where ``|x|`` last rises through ``level`` before ``k``,
        searching back no further than ``lo``."""
        i = k
        while i > lo and abs(x_win[i - 1]) >= level:
            i -= 1
        if i == lo:
            return None
        a, b = abs(float(x_win[i - 1])), abs(float(x_win[i]))
        return float(t_win[i - 1]) + dt_us * (level - a) / (b - a)

    def lobe_onset(lo: int, hi: int) -> tuple[float, bool]:
        """Onset of the lobe ``x_win[lo:hi]`` and whether it has a flat foot.

        The onset is normally the zero crossing at the lobe start. A
        small residual baseline offset (left after DC removal) keeps
        the sign from flipping, so the same-sign run can creep back
        across a flat, near-zero foot well before the signal really
        starts. The leading edge is therefore also extrapolated to zero
        through its 10 % and 50 % levels (within 0.01 rad of the true
        zero crossing for a sinusoid, ~0.05 T late for a slow ring-up
        lobe); when the zero crossing sits more than T/8 before that,
        the lobe has a flat foot and the extrapolation is used.
        """
        if lo == 0:
            t_zc = float(t_win[0])
        else:
            x0, x1 = float(x_win[lo - 1]), float(x_win[lo])
            t_zc = float(t_win[lo - 1]) + dt_us * x0 / (x0 - x1)
        k = lo + int(np.argmax(np.abs(x_win[lo:hi])))
        amp = abs(float(x_win[k]))
        t50 = level_crossing_us(k, lo, 0.5 * amp)
        t10 = level_crossing_us(k, lo, 0.1 * amp)
        if t50 is None or t10 is None or t50 <= t10:
            return t_zc, False
        t_edge = t10 - 0.1 * (t50 - t10) / 0.4
        if t_edge - t_zc > period_us / 8.0:
            return t_edge, True
        return t_zc, False

    start = lobe_start(idx_peak)
    pre = x_win[:start]
    noise_rms = (1.4826 * float(np.median(np.abs(pre - np.median(pre))))
                 if pre.size >= 8 else 0.0)
    noise_gate = float(noise_k) * noise_rms
    earliest = start - 2.0 * period_samples
    arrival_us, flat_foot = lobe_onset(start, idx_peak + 1)
    # A flat foot means nothing signal-like precedes this lobe, so
    # don't walk back past it.
    while not flat_foot and start > 0 and sign[start - 1] != 0:
        prev = lobe_start(start - 1)
        if (start - prev > period_samples or prev < earliest
                or np.max(np.abs(x_win[prev:start])) <= noise_gate):
            break
        arrival_us, flat_foot = lobe_onset(prev, start)
        start = prev

    return ArrivalResult(
        first_peak_us=first_peak_us,
        first_peak_value=float(y_0),
        arrival_us=arrival_us,
        onset_to_peak_us=first_peak_us - arrival_us,
        threshold=threshold,
        noise_gate=noise_gate,
        skip_us=float(skip_us),
    )

