"""Clean-window PNP: window margins and the median-of-peaks statistic."""
import numpy as np
import pytest

from openlifu_verification.characterization import Characterization
from openlifu_verification.pnp import clean_pnp_window_us, windowed_pnp


def test_default_trailing_margin_is_3us():
    lo, hi = clean_pnp_window_us(duration_us=60.0, arrival_us=37.0,
                                 max_delay_us=4.0)
    assert hi == pytest.approx(37.0 + 60.0 - 3.0)
    assert Characterization.PNP_WINDOW_END_MARGIN_US == 3.0


def test_median_ignores_low_edge_peaks():
    """Partially rung-up / falling peaks at the window edges pull the
    mean down but not the median."""
    freq_kHz = 400.0
    t_us = np.arange(0.0, 40.0, 0.01)
    T = 1000.0 / freq_kHz
    env = np.ones_like(t_us)
    env[t_us < 2 * T] = 0.6          # first two cycles still ringing up
    env[t_us > 40.0 - T] = 0.5       # last cycle falling
    trace = -env * np.sin(2 * np.pi * t_us / T)
    res = windowed_pnp(t_us, trace, (0.0, 40.0), frequency_kHz=freq_kHz)
    assert res["median"] == pytest.approx(1.0, abs=1e-3)
    assert res["mean"] < 0.97
