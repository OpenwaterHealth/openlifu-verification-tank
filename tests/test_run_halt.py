"""Characterization.run stops when the hydrophone depth or the peak
location is unusable, since every later measurement depends on them."""
from __future__ import annotations

import pytest

from openlifu_verification import AcceptanceCriteria, RunHalted, ScanConfig
from openlifu_verification.characterization import Characterization
from openlifu_verification.dry_run import DryRunTank
from openlifu_verification.operator_prefs import OperatorPrefs


def _chz(tank: DryRunTank) -> Characterization:
    prefs = OperatorPrefs(tester_name="t", test_app_version="t",
                          hydrophone_sn="t", txm_sn="t", console_sn="t")
    return Characterization(tank, prefs=prefs,
                            criteria=AcceptanceCriteria(),
                            scan_config=ScanConfig(),
                            frequency_kHz=400.0)


def _fail_if_called(*_a, **_k):
    raise AssertionError("downstream phase ran after a halt")


def _run(chz: Characterization) -> None:
    chz.run_beam_scans = _fail_if_called
    chz.run(skip_frequency=True, skip_voltage=True)


def test_depth_calibration_error_halts():
    tank = DryRunTank(rng_seed=0)

    def boom(**_k):
        raise RuntimeError("start_trigger returned device error")
    tank.calibrate_hydrophone_depth = boom
    tank.find_peak = _fail_if_called
    with pytest.raises(RunHalted, match="depth calibration failed"):
        _run(_chz(tank))


def test_depth_out_of_range_halts():
    tank = DryRunTank(rng_seed=0, hydrophone_position=(0.0, 0.0, 40.0))
    tank.find_peak = _fail_if_called
    with pytest.raises(RunHalted, match="depth is out of range"):
        _run(_chz(tank))


def test_find_peak_error_halts():
    tank = DryRunTank(rng_seed=0)

    def boom(**_k):
        raise RuntimeError("start_trigger returned device error")
    tank.find_peak = boom
    with pytest.raises(RunHalted, match="find_peak"):
        _run(_chz(tank))


def test_off_center_peak_halts():
    tank = DryRunTank(rng_seed=0, peak_xy=(2.5, 0.0))
    with pytest.raises(RunHalted, match="not centered"):
        _run(_chz(tank))


def test_good_setup_does_not_halt():
    chz = _chz(DryRunTank(rng_seed=0))
    chz.run(skip_2d=True, skip_frequency=True, skip_voltage=True)


def test_console_hwid_mismatch_halts():
    tank = DryRunTank(rng_seed=0)
    tank.calibrate_hydrophone_depth = _fail_if_called
    chz = _chz(tank)
    chz.expected_console_hwid = "NotThisConsole"
    with pytest.raises(RunHalted, match="does not match NotThisConsole"):
        _run(chz)


def test_console_hwid_match_runs():
    chz = _chz(DryRunTank(rng_seed=0))
    expected = chz.collect_test_info().console_hwid
    assert expected
    chz.expected_console_hwid = expected
    chz.run(skip_2d=True, skip_frequency=True, skip_voltage=True)
