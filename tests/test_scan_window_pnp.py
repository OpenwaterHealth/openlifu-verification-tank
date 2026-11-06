"""1-D scan PNP: one clean-window PNP per repeat, median over repeats."""
from types import SimpleNamespace

import numpy as np
import pytest

from openlifu_verification.characterization import Characterization
from openlifu_verification.pnp import scan_window_pnp
from openlifu_verification.scan_results import ScanResult

FREQ_KHZ = 400.0
DURATION_US = 60.0


def _timing(x, y, z, hydrophone_mm=None):
    """(first arrival, last arrival, max delay) in us."""
    return 37.0, 37.0, 4.0


def _scan(n_points=5, n_repeats=5, spike=None):
    t = np.arange(-10.0, 120.0, 0.1)
    rel = t - 37.0
    burst = np.where((rel >= 0) & (rel <= DURATION_US),
                     -np.sin(2 * np.pi * FREQ_KHZ * 1e-3 * rel), 0.0)
    amps = np.linspace(0.6, 1.0, n_points)
    repeats = amps[:, None, None] * np.broadcast_to(
        burst, (n_points, n_repeats, t.size)).copy()
    if spike is not None:
        i, j = spike
        # A noise burst in one repeat inside the clean window.
        repeats[i, j, (t > 70.0) & (t < 85.0)] *= 3.0
    traces = np.median(repeats, axis=1)
    repeats[0, -1] = np.nan  # a repeat that wasn't captured
    return ScanResult(
        scan_type="1d", t=t, traces=traces,
        coords={"yfoci": np.arange(n_points, dtype=float)},
        metadata={"focus_points_mm": np.zeros((n_points, 3)),
                  "hydrophone_position_mm": np.zeros(3),
                  "repeat_traces": repeats},
    ), amps


def _kw():
    return dict(duration_us=DURATION_US, frequency_kHz=FREQ_KHZ)


def test_scan_window_pnp_per_repeat_shape():
    scan, amps = _scan()
    per_point = scan_window_pnp(scan, _timing, **_kw())
    per_repeat = scan_window_pnp(scan, _timing, traces=scan.metadata["repeat_traces"],
                                 **_kw())
    assert per_point.shape == (5,)
    assert per_repeat.shape == (5, 5)
    assert np.isnan(per_repeat[0, -1])
    assert per_point == pytest.approx(amps, rel=5e-3)


def test_attach_window_pnp_takes_median_over_repeats():
    scan, amps = _scan(spike=(2, 1))
    char = SimpleNamespace(
        ver=SimpleNamespace(focus_timing_us=_timing),
        frequency_kHz=FREQ_KHZ,
        _pulse_duration_us=lambda: DURATION_US,
        PNP_WINDOW_START_MARGIN_US=Characterization.PNP_WINDOW_START_MARGIN_US,
        PNP_WINDOW_END_MARGIN_US=Characterization.PNP_WINDOW_END_MARGIN_US,
    )
    Characterization._attach_window_pnp(char, scan)
    pnp = scan.metadata["window_pnp"]
    std = scan.metadata["window_pnp_std"]
    assert pnp == pytest.approx(amps, rel=5e-3)
    # The spiked repeat shows up as spread, not in the reported PNP.
    assert std[2] > 10 * np.nanmax(np.delete(std, 2))
    assert scan.metadata["window_pnp_repeats"].shape == (5, 5)
