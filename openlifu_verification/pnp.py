"""Peak-negative-pressure statistics from the interference-free part of a capture.

Long transmit pulses overlap the acoustic arrival with transmit RF
pickup. Once transmission stops, the tail of the acoustic pulse (about
one time-of-flight long) is clean and fully rung up; these helpers
locate that window and summarise the negative peaks inside it.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from scipy.signal import find_peaks


def emitted_duration_us(duration_us: float, frequency_kHz: float) -> float:
    """Burst length actually emitted: the LIFU fires whole cycles only."""
    f = float(frequency_kHz)
    return int(float(duration_us) * f / 1000.0) * 1000.0 / f


def clean_pnp_window_us(*, duration_us: float, arrival_us: float,
                        max_delay_us: float = 0.0,
                        last_arrival_us: Optional[float] = None,
                        start_margin_us: float = 3.0,
                        end_margin_us: float = 3.0) -> tuple[float, float]:
    """Window (us, emission-referenced) free of transmit RF pickup.

    The last element stops driving at ``max_delay_us + duration_us``.
    Element wavefronts reach the hydrophone between ``arrival_us`` and
    ``last_arrival_us`` (equal when the hydrophone sits at the focus),
    and the first stops arriving at ``arrival_us + duration_us``. The
    window covers the part where transmission has stopped and every
    element is contributing, i.e. the fully rung-up, interference-free
    tail. For long pulses at the focus its length is ~ the one-way time
    of flight (``arrival_us - max_delay_us``) minus the margins. The
    end margin keeps the window clear of the acoustic pulse's
    falling edge (at 400 kHz a 0 us margin admits its first falling
    peak).

    Raises:
        ValueError: If the margins leave an empty window.
    """
    last = arrival_us if last_arrival_us is None else last_arrival_us
    start = max(duration_us + max_delay_us, last) + start_margin_us
    stop = arrival_us + duration_us - end_margin_us
    if stop <= start:
        raise ValueError(
            f"Empty clean PNP window: start={start:.2f} us >= "
            f"stop={stop:.2f} us (reduce the margins or lengthen the pulse)")
    return float(start), float(stop)


def windowed_pnp(t_us: np.ndarray, trace: np.ndarray,
                 window_us: tuple[float, float], *,
                 frequency_kHz: Optional[float] = None) -> dict:
    """Negative-peak statistics of ``trace`` inside ``window_us``.

    Finds every local negative peak in the window (at most one per
    acoustic cycle when ``frequency_kHz`` is given) and returns their
    median / mean / sample std along with the window's absolute
    minimum. The median is the reported PNP: a partially rung-up or
    falling peak at either end of the window can't pull it. All
    magnitudes are positive and in the trace's units.

    Returns:
        Dict with ``window_us``, ``pnp`` (max negative excursion in
        the window), ``peak_t_us`` / ``peak_values`` (each local
        negative peak), ``n_peaks``, ``median``, ``mean`` and ``std``
        (NaN when there are too few peaks).
    """
    t_us = np.asarray(t_us, dtype=float)
    trace = np.asarray(trace, dtype=float)
    lo, hi = float(window_us[0]), float(window_us[1])
    mask = (t_us >= lo) & (t_us <= hi)
    if mask.sum() < 3:
        raise ValueError(
            f"PNP window [{lo:.2f}, {hi:.2f}] us holds fewer than 3 samples "
            f"(capture spans {t_us[0]:.2f}..{t_us[-1]:.2f} us)")
    t_win = t_us[mask]
    neg = -trace[mask]
    distance = None
    if frequency_kHz:
        dt_us = float(t_win[1] - t_win[0])
        # 0.75 period rejects noise ripples while never skipping a real cycle.
        distance = max(1, int(0.75 * 1e3 / float(frequency_kHz) / dt_us))
    idx, _ = find_peaks(neg, height=0.0, distance=distance)
    peaks = neg[idx]
    return {
        "window_us": (lo, hi),
        "pnp": float(neg.max()),
        "peak_t_us": t_win[idx],
        "peak_values": peaks,
        "n_peaks": int(idx.size),
        "median": float(np.median(peaks)) if idx.size else float("nan"),
        "mean": float(peaks.mean()) if idx.size else float("nan"),
        "std": float(peaks.std(ddof=1)) if idx.size > 1 else float("nan"),
    }


def scan_window_pnp(scan, focus_timing_us: Callable, *, duration_us: float,
                    frequency_kHz: float, start_margin_us: float = 3.0,
                    end_margin_us: float = 3.0,
                    traces: Optional[np.ndarray] = None) -> np.ndarray:
    """Per-point clean-window median PNP of a focus scan, in the trace units.

    The hydrophone stays at ``metadata["hydrophone_position_mm"]`` while
    the focus moves through ``metadata["focus_points_mm"]``, so each point
    gets its own window from ``focus_timing_us(x, y, z, hydrophone_mm=...)``.
    ``traces`` defaults to ``scan.traces``; pass e.g.
    ``metadata["repeat_traces"]`` (shape ``(n_points, n_repeats,
    samples)``) for one PNP per repeat. Returns an array shaped like
    ``traces`` without the time axis; NaN where a trace is missing or
    no window or negative peaks could be found.
    """
    traces = np.asarray(scan.traces if traces is None else traces, dtype=float)
    foci = np.asarray(scan.metadata["focus_points_mm"], dtype=float).reshape(-1, 3)
    n_traces = traces.size // traces.shape[-1]
    if n_traces % foci.shape[0]:
        raise ValueError(f"{foci.shape[0]} focus points for {n_traces} traces")
    flat = traces.reshape(foci.shape[0], -1, traces.shape[-1])
    hydrophone = np.asarray(scan.metadata["hydrophone_position_mm"], dtype=float)
    out = np.full(flat.shape[:2], np.nan)
    for i, (focus, point_traces) in enumerate(zip(foci, flat)):
        first_us, last_us, max_delay_us = focus_timing_us(*focus, hydrophone_mm=hydrophone)
        try:
            window = clean_pnp_window_us(
                duration_us=duration_us, arrival_us=first_us,
                last_arrival_us=last_us, max_delay_us=max_delay_us,
                start_margin_us=start_margin_us, end_margin_us=end_margin_us)
        except ValueError:
            continue
        for j, trace in enumerate(point_traces):
            if not np.isfinite(trace).all():
                continue
            try:
                out[i, j] = windowed_pnp(scan.t, trace, window,
                                         frequency_kHz=frequency_kHz)["median"]
            except ValueError:
                pass
    return out.reshape(traces.shape[:-1])
