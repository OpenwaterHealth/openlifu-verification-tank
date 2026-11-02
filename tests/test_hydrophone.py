"""Tests for :class:`openlifu_verification.Hydrophone`.

Uses the calibration files shipped in the package's setup data
template so tests don't need network / hardware access.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from openlifu_verification import Hydrophone, paths
from openlifu_verification.setup_data import SetupData


CAL_ID = "2246"


@pytest.fixture()
def cal_file() -> Path:
    """Absolute path to the shipped 2246 calibration file."""
    hits = list(paths.BUNDLED_CALIBRATIONS_DIR.glob(f"HNR0500-{CAL_ID}*.txt"))
    assert hits, "shipped 2246 calibration missing"
    return hits[0]


def test_load_by_absolute_path(cal_file):
    hydro = Hydrophone(cal_file)
    assert hydro.metadata.get("HYD_SN") == CAL_ID
    assert not hydro.calibration_data.empty


def test_load_by_bare_id_finds_shipped_file():
    """Passing just the ID should discover the shipped calibration."""
    hydro = Hydrophone(CAL_ID)
    assert hydro.metadata.get("HYD_SN") == CAL_ID


def test_load_by_bare_id_prefers_active_setup_data(cal_file, tmp_path):
    """Calibrations in the active setup data folder win over the
    shipped fallback."""
    setup = SetupData(tmp_path / "lab_setup")
    setup.calibrations_dir.mkdir(parents=True)
    local_copy = setup.calibrations_dir / cal_file.name
    shutil.copy2(cal_file, local_copy)
    setup.make_active()

    assert Hydrophone._resolve_calibration_path(CAL_ID) == local_copy


def test_sensitivity_is_finite_within_calibrated_range(cal_file):
    """Sanity check: sensitivity should be finite and positive across
    the calibration table."""
    hydro = Hydrophone(cal_file)
    for freq_hz in (500e3, 1e6, 2e6, 5e6):
        s = hydro.get_sensitivity_pa_per_v(freq_hz)
        assert np.isfinite(s) and s > 0


def test_mv_to_pa_scales_linearly(cal_file):
    """``mv_to_pa`` on a sine wave at a calibrated frequency should
    scale linearly with the input amplitude."""
    hydro = Hydrophone(cal_file)
    freq_hz = 1e6
    fs = 50e6
    t = np.linspace(0, 2e-6, int(fs * 2e-6), endpoint=False)
    trace_mv = 100.0 * np.sin(2 * np.pi * freq_hz * t)
    trace_pa = hydro.mv_to_pa(trace_mv, freq_hz)
    # Scale input by 2, output should scale by 2 (up to numerical noise).
    trace_pa_2x = hydro.mv_to_pa(2.0 * trace_mv, freq_hz)
    ratio = np.max(np.abs(trace_pa_2x)) / max(np.max(np.abs(trace_pa)), 1e-12)
    assert 1.99 < ratio < 2.01


def test_unresolvable_id_raises():
    """A garbage ID that doesn't match any file should error clearly."""
    with pytest.raises(Exception):
        Hydrophone("__definitely_not_a_real_hydrophone__")
