"""Focal-gain LUT used to turn frequency-sweep PNP into element sensitivity."""
from __future__ import annotations

import numpy as np
import pytest

from openlifu_verification import report_io as rio
from openlifu_verification.scan_config import ScanConfig


def _sweep_freqs_kHz(nominal: float) -> np.ndarray:
    offsets = ScanConfig().plan_for(nominal).frequency_sweep.offsets_kHz
    return nominal + np.asarray(offsets, dtype=float)


def test_lut_shape_and_axes():
    assert rio._FOCAL_GAIN_DATA.shape == (len(rio._FOCAL_GAIN_F0_HZ),
                                         len(rio._FOCAL_GAIN_CROSSTALK))
    assert np.all(np.diff(rio._FOCAL_GAIN_F0_HZ) > 0)
    assert np.all(np.diff(rio._FOCAL_GAIN_CROSSTALK) > 0)
    assert np.all(rio._FOCAL_GAIN_DATA > 0)
    assert rio.FOCAL_GAIN_DEPTH_MM == 50.0


@pytest.mark.parametrize("nominal", [155.0, 400.0])
def test_lut_covers_every_standard_sweep_frequency(nominal):
    """No sweep point may fall back to extrapolation."""
    for f_kHz in _sweep_freqs_kHz(nominal):
        assert f_kHz * 1e3 in rio._FOCAL_GAIN_F0_HZ
        g = rio._focal_gain(f_kHz, 0.12)
        assert np.isfinite(g) and g > 0


def test_grid_points_are_exact():
    i_f, i_xt = 5, 3  # 155 kHz, 12 % crosstalk
    f_kHz = rio._FOCAL_GAIN_F0_HZ[i_f] / 1e3
    xt = rio._FOCAL_GAIN_CROSSTALK[i_xt]
    assert rio._focal_gain(f_kHz, xt) == pytest.approx(rio._FOCAL_GAIN_DATA[i_f, i_xt])


def test_interpolates_between_grid_points():
    lo, hi = rio._focal_gain(155.0, 0.12), rio._focal_gain(160.0, 0.12)
    assert rio._focal_gain(157.5, 0.12) == pytest.approx(0.5 * (lo + hi))


@pytest.mark.parametrize("f_kHz", [100.0, 270.0, 450.0])
def test_outside_simulated_bands_is_nan(f_kHz):
    """Never interpolate across the 180-375 kHz gap or extrapolate."""
    assert np.isnan(rio._focal_gain(f_kHz, 0.12))


def test_gain_increases_with_frequency_and_crosstalk_at_nominal():
    # Physical sanity at 50 mm: larger aperture in wavelengths -> more gain,
    # and in-phase neighbour crosstalk adds to the focal pressure.
    assert rio._focal_gain(400.0, 0.12) > rio._focal_gain(155.0, 0.12)
    for f in (155.0, 400.0):
        assert rio._focal_gain(f, 0.12) > rio._focal_gain(f, 0.0)
