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
    # 6 UID bytes ("2ezGGCzTN" for the dry-run console) instead of the
    # full form ("4dyWcpSyLJHurUomU").
    chz = _chz(DryRunTank(rng_seed=0))
    chz.expected_console_hwid = "2ezGGCzTN"
    chz.run(skip_2d=True, skip_frequency=True, skip_voltage=True)
    assert "legacy 6-byte hardware ID 2ezGGCzTN" in caplog.text
    assert "full ID 4dyWcpSyLJHurUomU" in caplog.text


# ----------------------------------------------------------------------
# Firmware older than the acceptance minimum stops the run up front
# ----------------------------------------------------------------------
def test_minimum_firmware_versions():
    fw = AcceptanceCriteria().firmware
    assert (fw.min_txm_version, fw.min_console_version) == ("2.0.8", "1.2.7")


@pytest.mark.parametrize("txm, console, outdated", [
    ("2.0.7", "1.2.7", [("TXM firmware", "2.0.7", "2.0.8")]),
    ("v2.0.8", "1.2.6", [("Console firmware", "1.2.6", "1.2.7")]),
    ("1.9.9", "", [("TXM firmware", "1.9.9", "2.0.8"),
                   ("Console firmware", "", "1.2.7")]),
])
def test_outdated_firmware_halts_before_measuring(txm, console, outdated):
    from openlifu_verification import FirmwareOutOfDate
    tank = DryRunTank(rng_seed=0, txm_fw_version=txm,
                      console_fw_version=console)
    tank.calibrate_hydrophone_depth = _fail_if_called
    with pytest.raises(FirmwareOutOfDate) as info:
        _run(_chz(tank))
    assert info.value.devices == outdated
    assert isinstance(info.value, RunHalted)
    message = str(info.value)
    assert message.startswith("Firmware update required.\n")
    for label, reported, minimum in outdated:
        assert f"{label}: " in message
        assert f"(required: {minimum} or newer)" in message
    if not console:
        assert "Console firmware: version could not be read" in message


@pytest.mark.parametrize("txm, console", [
    ("2.0.8", "1.2.7"), ("v2.1.0", "1.3.0"), ("2.0.10", "1.2.7-rc1")])
def test_current_firmware_runs(txm, console):
    tank = DryRunTank(rng_seed=0, txm_fw_version=txm,
                      console_fw_version=console)
    _chz(tank).run(skip_2d=True, skip_frequency=True, skip_voltage=True)


def test_firmware_check_disabled_by_empty_minimum():
    tank = DryRunTank(rng_seed=0, txm_fw_version="0.0.1")
    chz = _chz(tank)
    chz.criteria.firmware.min_txm_version = ""
    chz.run(skip_2d=True, skip_frequency=True, skip_voltage=True)


# ----------------------------------------------------------------------
# Depth calibration failure: ask, then continue at nominal or stop
# ----------------------------------------------------------------------
def _boom(**_k):
    raise RuntimeError("no arrival found")


class _Ask:
    def __init__(self, answer: bool):
        self.answer = answer
        self.calls = []

    def __call__(self, failure):
        self.calls.append(failure)
        return self.answer


def test_depth_error_continue_runs_at_nominal_and_fails_depth():
    tank = DryRunTank(rng_seed=0)
    tank.calibrate_hydrophone_depth = _boom
    chz = _chz(tank)
    ask = _Ask(True)
    report = chz.run(skip_2d=True, skip_frequency=True, skip_voltage=True,
                     on_setup_failure=ask)
    (failure,) = ask.calls
    assert failure.step == "depth"
    assert failure.problem == "Plane-wave depth calibration failed: no arrival found"
    assert failure.fallback == "the nominal depth of 50.0 mm"
    assert failure.continue_label == "Continue at 50 mm"
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
                     on_setup_failure=ask)
    (failure,) = ask.calls
    assert failure.problem.startswith("Hydrophone depth is out of range: depth = ")
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
                       on_setup_failure=_Ask(False))
    assert isinstance(info.value, RunAborted)


def test_good_depth_does_not_ask():
    ask = _Ask(False)
    _chz(DryRunTank(rng_seed=0)).run(
        skip_2d=True, skip_frequency=True, skip_voltage=True,
        on_setup_failure=ask)
    assert ask.calls == []


def test_peak_error_continue_runs_at_center_and_fails_peak():
    from openlifu_verification.characterization import ROW
    tank = DryRunTank(rng_seed=0)

    def boom(**_k):
        raise RuntimeError("no signal above noise")
    tank.find_peak = boom
    ask = _Ask(True)
    report = _chz(tank).run(skip_2d=True, skip_frequency=True,
                            skip_voltage=True, on_setup_failure=ask)
    (failure,) = ask.calls
    assert failure.step == "peak"
    assert failure.title == "Peak search failed"
    assert failure.problem == "Peak search (find_peak) failed: no signal above noise"
    assert failure.continue_label == "Continue at center"
    for key in ("peak_x_mm", "peak_y_mm"):
        row = report.rows[ROW[key]]
        assert row.status == "FAIL" and row.value is None
        assert row.note.endswith("; continued at (0, 0)")
    assert tuple(tank.hydrophone_position[:2]) == (0.0, 0.0)
    assert report.overall_pass is False
    assert report.waveform_at_peak  # the rest of the run went ahead


def test_off_center_peak_continue_stays_at_found_peak():
    from openlifu_verification.characterization import ROW
    tank = DryRunTank(rng_seed=0, peak_xy=(2.5, 0.0))
    ask = _Ask(True)
    report = _chz(tank).run(skip_2d=True, skip_frequency=True,
                            skip_voltage=True, on_setup_failure=ask)
    (failure,) = ask.calls
    assert failure.problem.startswith(
        "Hydrophone is not centered on the focus: |x| = ")
    assert failure.continue_label == "Continue at found peak"
    x, y = report.peak_xy_mm
    assert failure.fallback == (
        f"the peak it found (x = {x:.2f} mm, y = {y:.2f} mm)")
    x_row = report.rows[ROW["peak_x_mm"]]
    assert x_row.status == "FAIL" and abs(x_row.value - 2.5) < 0.5
    assert x_row.note.endswith("; continued at the found peak")
    # The rest of the run is measured around the found peak.
    assert abs(tank.hydrophone_position[0] - x) < 1e-9
    assert abs(tank.hydrophone_position[1] - y) < 1e-9
    assert report.overall_pass is False


def test_peak_failure_stop_raises_operator_stopped():
    from openlifu_verification import OperatorStoppedRun
    tank = DryRunTank(rng_seed=0, peak_xy=(2.5, 0.0))
    chz = _chz(tank)
    chz.run_beam_scans = _fail_if_called
    chz.measure_waveform_at_peak = _fail_if_called
    with pytest.raises(OperatorStoppedRun, match="peak search failed"):
        chz.run(skip_frequency=True, skip_voltage=True,
                on_setup_failure=_Ask(False))


def test_depth_and_peak_both_fail_asks_twice():
    tank = DryRunTank(rng_seed=0)
    tank.calibrate_hydrophone_depth = _boom
    tank.find_peak = _boom
    ask = _Ask(True)
    report = _chz(tank).run(skip_2d=True, skip_frequency=True,
                            skip_voltage=True, on_setup_failure=ask)
    assert [f.step for f in ask.calls] == ["depth", "peak"]
    assert report.overall_pass is False


# ----------------------------------------------------------------------
# Plane wave not detected -> focused-pulse fallback, D.3 WARN
# ----------------------------------------------------------------------
def _weak_plane_wave(tank, *, focused_ok=True, focused_depth=None):
    """Plane-wave calls raise DepthNotFound; focused calls run the
    dry-run calibration (optionally at a different true depth) or raise
    DepthNotFound too. Records each call's kwargs."""
    from openlifu_verification import DepthNotFound
    real = tank.calibrate_hydrophone_depth
    calls = []

    def cal(**kw):
        calls.append(kw)
        if kw.get("z_focus_mm") is None:
            raise DepthNotFound("No hydrophone signal above the noise near 50 mm "
                                "(signal-to-noise 2.1, need 6)")
        if not focused_ok:
            raise DepthNotFound("Could not locate a first-arrival RF peak")
        if focused_depth is not None:
            tank.hydrophone_position[2] = focused_depth
        return real(**kw)
    tank.calibrate_hydrophone_depth = cal
    return calls


def test_weak_plane_wave_falls_back_to_focused_and_warns():
    from openlifu_verification.characterization import ROW
    tank = DryRunTank(rng_seed=0)
    calls = _weak_plane_wave(tank)
    ask = _Ask(False)
    report = _chz(tank).run(skip_2d=True, skip_frequency=True,
                            skip_voltage=True, on_setup_failure=ask)
    assert ask.calls == []                        # no operator prompt
    plane, focused = calls[0], calls[1]
    assert plane["z_focus_mm"] is None and focused["z_focus_mm"] == 50.0
    for kw in (plane, focused):                   # detection is on
        assert kw["expected_mm"] == 50.0 and kw["min_snr"] == 6.0
    d3 = report.rows[ROW["plane_wave_depth_mm"]]
    assert d3.status == "WARN"
    assert abs(d3.value - 50.0) < 1.0
    assert "focused at 50 mm" in d3.note and "signal-to-noise 2.1" in d3.note
    assert report.rows[ROW["plane_wave_arrival_us"]].value is None
    assert report.depth_cal_plane_wave["z_focus_mm"] == 50.0
    assert report.overall_pass is True            # WARN doesn't fail


def test_focused_fallback_out_of_range_still_fails():
    from openlifu_verification.characterization import ROW
    tank = DryRunTank(rng_seed=0)
    _weak_plane_wave(tank, focused_depth=44.0)
    ask = _Ask(True)
    report = _chz(tank).run(skip_2d=True, skip_frequency=True,
                            skip_voltage=True, on_setup_failure=ask)
    (failure,) = ask.calls
    assert failure.problem.startswith("Hydrophone depth is out of range")
    assert report.rows[ROW["plane_wave_depth_mm"]].status == "FAIL"
    assert report.overall_pass is False


def test_plane_and_focused_not_found_asks_operator():
    tank = DryRunTank(rng_seed=0)
    _weak_plane_wave(tank, focused_ok=False)
    tank.find_peak = _fail_if_called
    ask = _Ask(False)
    from openlifu_verification import OperatorStoppedRun
    with pytest.raises(OperatorStoppedRun):
        _chz(tank).run(skip_2d=True, skip_frequency=True, skip_voltage=True,
                       on_setup_failure=ask)
    (failure,) = ask.calls
    assert "plane wave and a pulse focused at 50 mm both failed" in failure.problem


def test_plane_and_focused_not_found_halts_without_handler():
    tank = DryRunTank(rng_seed=0)
    _weak_plane_wave(tank, focused_ok=False)
    tank.find_peak = _fail_if_called
    with pytest.raises(RunHalted, match="both failed to detect"):
        _run(_chz(tank))


def test_other_errors_do_not_try_the_fallback():
    """A hardware / capture error is not "no signal": no focused retry."""
    tank = DryRunTank(rng_seed=0)
    calls = []

    def boom(**kw):
        calls.append(kw)
        raise RuntimeError("Rapid-block capture timed out")
    tank.calibrate_hydrophone_depth = boom
    tank.find_peak = _fail_if_called
    with pytest.raises(RunHalted, match="timed out"):
        _run(_chz(tank))
    assert len(calls) == 1
