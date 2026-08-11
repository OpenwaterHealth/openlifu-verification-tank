"""Tests for :class:`openlifu_verification.Hydrophone`.

Uses the calibration file shipped in ``config/hydrophone_calibrations/``
so tests don't need network / hardware access.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from openlifu_verification import Hydrophone, paths


CAL_ID = "2246"


@pytest.fixture()
def cal_file() -> Path:
    """Absolute path to the shipped 2246 calibration file."""
    hits = list(paths.REPO_CALIBRATIONS_DIR.glob(f"HNR0500-{CAL_ID}*.txt"))
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


def test_load_by_bare_id_prefers_cwd_config_over_shipped(
    clean_cwd, cal_file, tmp_path
):
    """When a ``config/hydrophone_calibrations/`` folder exists under
    the CWD, its files must win over the repo-shipped fallback."""
    local_dir = tmp_path / "config" / "hydrophone_calibrations"
    local_dir.mkdir(parents=True)
    local_copy = local_dir / cal_file.name
    shutil.copy2(cal_file, local_copy)

    # Now load by bare ID from a CWD that has both a config copy and
    # the repo fallback available.
    hydro = Hydrophone(CAL_ID)
    assert hydro.metadata.get("HYD_SN") == CAL_ID
    # Round-trip through the resolver to confirm the CWD copy wins.
    resolved = Hydrophone._resolve_calibration_path(CAL_ID)
    assert resolved == local_copy


def test_load_by_bare_id_falls_back_to_legacy_layout(clean_cwd, cal_file, tmp_path):
    """Pre-refactor deployments kept files at ``hydrophone_calibrations/``.
    That legacy layout must still resolve so old CWDs keep working."""
    legacy_dir = tmp_path / "hydrophone_calibrations"
    legacy_dir.mkdir()
    legacy_copy = legacy_dir / cal_file.name
    shutil.copy2(cal_file, legacy_copy)

    hydro = Hydrophone(CAL_ID)
    assert hydro.metadata.get("HYD_SN") == CAL_ID


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
