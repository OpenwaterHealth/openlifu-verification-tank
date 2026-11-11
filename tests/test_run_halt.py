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


def test_console_registered_with_legacy_hwid_runs_with_warning(caplog):
    # Consoles registered from older reports carry base58 of the first
    # 6 UID bytes ("2naX97Jp7" for the dry-run console) instead of the
    # full form ("4vfAkNk3axRSP1x4R").
    chz = _chz(DryRunTank(rng_seed=0))
    chz.expected_console_hwid = "2naX97Jp7"
    chz.run(skip_2d=True, skip_frequency=True, skip_voltage=True)
    assert "legacy 6-byte hardware ID 2naX97Jp7" in caplog.text
    assert "full ID 4vfAkNk3axRSP1x4R" in caplog.text


# ----------------------------------------------------------------------
# Depth calibration failure: ask, then continue at nominal or stop
# ----------------------------------------------------------------------
def _boom(**_k):
    raise RuntimeError("no arrival found")


class _Ask:
    def __init__(self, answer: bool):
        self.answer = answer
        self.calls = []

    def __call__(self, problem, nominal_mm):
        self.calls.append((problem, nominal_mm))
        return self.answer


def test_depth_error_continue_runs_at_nominal_and_fails_depth():
    tank = DryRunTank(rng_seed=0)
    tank.calibrate_hydrophone_depth = _boom
    chz = _chz(tank)
    ask = _Ask(True)
    report = chz.run(skip_2d=True, skip_frequency=True, skip_voltage=True,
                     on_depth_failure=ask)
    assert ask.calls == [
        ("Plane-wave depth calibration failed: no arrival found", 50.0)]
    from openlifu_verification.characterization import ROW
    row = report.rows[ROW["plane_wave_depth_mm"]]
    assert row.status == "FAIL"
    assert row.value is None
    assert "continued at nominal 50.0 mm" in row.note
    assert "no arrival found" in row.note
    assert report.overall_pass is False
    # The rest of the run went ahead (peak search graded).
    assert report.rows[ROW["peak_x_mm"]].status in ("PASS", "FAIL")


def test_depth_out_of_range_continue_moves_to_nominal():
    tank = DryRunTank(rng_seed=0, hydrophone_position=(0.0, 0.0, 40.0))
    chz = _chz(tank)
    ask = _Ask(True)
    report = chz.run(skip_2d=True, skip_frequency=True, skip_voltage=True,
                     on_depth_failure=ask)
    (problem, nominal), = ask.calls
    assert problem.startswith("Hydrophone depth is out of range: depth = ")
    assert nominal == 50.0
    from openlifu_verification.characterization import ROW
    row = report.rows[ROW["plane_wave_depth_mm"]]
    assert row.status == "FAIL"
    assert abs(row.value - 40.0) < 2.0           # the measured depth is kept
    assert row.note.endswith("; continued at nominal 50.0 mm")
    assert tank.hydrophone_position[2] == 50.0
    assert report.overall_pass is False


def test_depth_failure_stop_raises_operator_stopped():
    from openlifu_verification import OperatorStoppedRun, RunAborted
    tank = DryRunTank(rng_seed=0)
    tank.calibrate_hydrophone_depth = _boom
    tank.find_peak = _fail_if_called
    with pytest.raises(OperatorStoppedRun) as info:
        _chz(tank).run(skip_2d=True, skip_frequency=True, skip_voltage=True,
                       on_depth_failure=_Ask(False))
    assert isinstance(info.value, RunAborted)


def test_good_depth_does_not_ask():
    ask = _Ask(False)
    _chz(DryRunTank(rng_seed=0)).run(
        skip_2d=True, skip_frequency=True, skip_voltage=True,
        on_depth_failure=ask)
    assert ask.calls == []
