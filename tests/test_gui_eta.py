"""The GUI's phase timing model reproduces measured 155 kHz run times.

Run totals are from console logs of 2026-10-15 ("fully connected" ->
"Run directory", plus ~1 s connect). Those runs predate skipping the
HV re-send in configure_lifu, so each is reduced by the measured
saving: 6 same-voltage reconfigures x (7.94 - 2.32) s
(scripts/time_hv_reconfigure.py, 40 V).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_report_gui as gui  # noqa: E402

from openlifu_verification import ScanConfig  # noqa: E402

_HALF_POINTS = {
    "lateral_1d": {"points": 19},
    "elevation_1d": {"points": 19},
    "axial_1d": {"points": 11},
    "scan_2d": {"points": 5},
    "frequency_sweep": {"offsets_kHz": [-20, -10, 0, 10, 20]},
    "voltage_sweep": {"voltages_V": [10, 20, 30]},
}


def _config(tmp_path, overrides):
    if not overrides:
        return ScanConfig()
    path = tmp_path / "override.json"
    path.write_text(json.dumps(overrides))
    return ScanConfig.with_overrides(path)


_HV_RESEND_SAVING_S = 6 * (7.94 - 2.32)


@pytest.mark.parametrize("overrides, measured_s", [
    ({}, 222.0),
    ({"by_freq_kHz": {"155": _HALF_POINTS, "400": _HALF_POINTS}}, 153.7),
    ({"scope": {"n_averages": 1, "pnp_n_averages": 1}}, 176.5),
])
def test_model_matches_measured_155_runs(tmp_path, overrides, measured_s):
    phases = gui._estimate_phases(_config(tmp_path, overrides), 155, 40)
    total = sum(p.seconds for p in phases)
    assert total == pytest.approx(measured_s - _HV_RESEND_SAVING_S, rel=0.03)


def test_lower_rail_voltage_is_faster():
    cfg = ScanConfig()
    at_40 = sum(p.seconds for p in gui._estimate_phases(cfg, 400, 40))
    at_20 = sum(p.seconds for p in gui._estimate_phases(cfg, 400, 20))
    # HV on at startup + the rail restore in waveform-at-peak.
    assert at_40 - at_20 == pytest.approx(2 * 20 * gui._HV_SETTLE_S_PER_V)


def test_skipped_phases_are_dropped():
    labels = [p.label for p in gui._estimate_phases(
        ScanConfig(), 155, 40,
        skip_2d=True, skip_frequency=True, skip_voltage=True)]
    assert labels == [
        "Connecting to hardware", "Configuring transducer",
        "Plane-wave depth calibration", "Peak search", "Waveform at peak",
        "Writing report", "Complete",
    ]
