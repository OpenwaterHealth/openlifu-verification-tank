"""LauncherDialog: setup data validation and the Enter-to-advance chain."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_report_gui as gui  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialog, QLabel, QPushButton  # noqa: E402

from openlifu_verification.setup_data import (  # noqa: E402
    CONSOLES,
    TESTERS,
    SetupData,
)


@pytest.fixture(scope="module", autouse=True)
def _app():
    yield QApplication.instance() or QApplication([])


@pytest.fixture
def setup(tmp_path) -> SetupData:
    s = SetupData(tmp_path / "setup")
    s.initialize()
    s.add(TESTERS, "T1", "Jane Doe")
    s.add(CONSOLES, "WL134Q00007", "4ZGd1kX")
    return s


def _launcher(setup: SetupData, tmp_path, *flags: str,
              hardware_ok: bool = True) -> gui.LauncherDialog:
    """Launcher (not shown, so not polling); ``hardware_ok`` marks
    every hardware LED green so Start depends on the fields only."""
    args = gui.build_gui_parser().parse_args(
        ["--reports-root", str(tmp_path / "reports"), *flags])
    dlg = gui.LauncherDialog(args, app_version="9.9",
                             setup_root=setup.root)
    if hardware_ok:
        dlg._show_hardware_status(dict.fromkeys(dlg.hw_leds, True))
        if dlg._console_detector is not None:
            dlg._console_detector.wait()  # finish the console read
            QApplication.processEvents()
    return dlg


def _scan(dlg: gui.LauncherDialog, edit, text: str) -> None:
    edit.setFocus()
    edit.setText(text)
    QTest.keyClick(edit, Qt.Key.Key_Return)


def test_enter_chain_validates_and_enables_start(setup, tmp_path):
    dlg = _launcher(setup, tmp_path)
    assert dlg.freq_combo.currentData() is None
    assert not dlg.freq_combo.isEnabled()
    assert not dlg.start_button.isEnabled()

    _scan(dlg, dlg.txm_edit, "WLM44Q00012")
    assert dlg.txm_edit.property("valid") == "yes"
    assert dlg.freq_combo.currentData() == 400.0
    assert dlg.focusWidget() is dlg.tester_edit

    _scan(dlg, dlg.tester_edit, "T1")
    assert dlg._tester.detail.toolTip() == "Jane Doe"
    assert dlg.focusWidget() is dlg.console_edit
    _scan(dlg, dlg.console_edit, "WL134Q00007")
    assert not dlg.start_button.isEnabled()
    _scan(dlg, dlg.hydrophone_edit, "HYDRO2246")

    assert dlg.start_button.isEnabled()
    assert dlg.focusWidget() is dlg.start_button
    assert dlg.result() == 0  # Enter never clicks Start by itself
    prefs = dlg.collected_prefs
    assert prefs.tester_name == "Jane Doe (T1)"
    assert (prefs.txm_sn, prefs.console_sn, prefs.hydrophone_sn,
            prefs.test_app_version) == ("WLM44Q00012", "WL134Q00007",
                                        "HYDRO2246", "9.9")
    assert dlg.selected_console_hwid == "4ZGd1kX"
    assert dlg.selected_hydrophone_calibration.is_file()
    assert dlg.selected_frequency_kHz == 400.0


def test_txm_sn_sets_frequency(setup, tmp_path):
    dlg = _launcher(setup, tmp_path)
    _scan(dlg, dlg.txm_edit, "WLM24Q00012")
    assert dlg.freq_combo.currentData() == 155.0
    dlg.txm_edit.setText("WLM24Q0001")
    assert dlg.freq_combo.currentData() is None
    _scan(dlg, dlg.txm_edit, "WLM44Q00012")
    assert dlg.freq_combo.currentData() == 400.0


@pytest.mark.parametrize("field, value, reason", [
    ("txm_edit", "WLM34Q00012", "Expected WLM24Q/WLM44Q#####"),
    ("console_edit", "WL134Q00008", "Not in setup data"),
    ("console_edit", "WL134W00007", "Expected WL114Q/WL134Q#####"),
    ("hydrophone_edit", "HYDRO22", "Expected HYDRO####"),
    ("tester_edit", "T2", "Not in setup data"),
])
def test_invalid_entry_is_flagged_and_keeps_focus(setup, tmp_path, field,
                                                 value, reason):
    dlg = _launcher(setup, tmp_path)
    edit = getattr(dlg, field)
    _scan(dlg, edit, value)
    assert edit.property("valid") == "no"
    assert dlg._fields[edit].detail.toolTip() == reason
    assert dlg.focusWidget() is edit
    edit.setText("")
    assert edit.property("valid") == ""


def test_missing_calibration_file_is_invalid(setup, tmp_path):
    setup.calibration_path("HYDRO2581").unlink()
    dlg = _launcher(setup, tmp_path)
    _scan(dlg, dlg.hydrophone_edit, "HYDRO2581")
    assert dlg.hydrophone_edit.property("valid") == "no"
    assert dlg._hydrophone.detail.toolTip().startswith("Calibration file missing")


def test_no_validate_accepts_anything_but_hydrophone(setup, tmp_path):
    dlg = _launcher(setup, tmp_path, "--no-validate")
    assert dlg.freq_combo.isEnabled()
    _scan(dlg, dlg.txm_edit, "proto-7")
    assert dlg.txm_edit.property("valid") == "yes"
    assert dlg.focusWidget() is dlg.freq_combo
    dlg.freq_combo.setCurrentIndex(dlg.freq_combo.findData(155.0))
    QTest.keyClick(dlg.freq_combo, Qt.Key.Key_Return)
    assert dlg.focusWidget() is dlg.tester_edit
    _scan(dlg, dlg.tester_edit, "Visitor")
    _scan(dlg, dlg.console_edit, "bench console")
    _scan(dlg, dlg.hydrophone_edit, "HYDRO9999")
    assert dlg.hydrophone_edit.property("valid") == "no"
    assert not dlg.start_button.isEnabled()
    _scan(dlg, dlg.hydrophone_edit, "HYDRO2585")
    assert dlg.start_button.isEnabled()

    prefs = dlg.collected_prefs
    assert (prefs.tester_name, prefs.txm_sn, prefs.console_sn) \
        == ("Visitor", "proto-7", "bench console")
    assert dlg.selected_console_hwid is None
    assert dlg.selected_frequency_kHz == 155.0


def test_no_validate_still_names_registered_tester(setup, tmp_path):
    dlg = _launcher(setup, tmp_path, "--no-validate")
    _scan(dlg, dlg.tester_edit, "T1")
    assert dlg.collected_prefs.tester_name == "Jane Doe (T1)"
    _scan(dlg, dlg.txm_edit, "WLM24Q00012")
    assert dlg.freq_combo.currentData() == 155.0


@pytest.mark.parametrize("flags, visible", [((), False),
                                            (("--add-items",), True)])
def test_folder_buttons_only_with_add_items(setup, tmp_path, flags, visible):
    dlg = _launcher(setup, tmp_path, *flags)
    buttons = [b for b in dlg.findChildren(QPushButton)
               if b.text() == "\u2026"]
    # Reports row always; setup-data row only with --add-items.
    assert len(buttons) == (2 if visible else 1)
    assert all(b.isHidden() is not visible for b in buttons)
    labels = [lbl.text() for lbl in dlg.findChildren(QLabel)]
    assert ("Setup data:" in labels) is visible


def test_add_items_registers_missing_tester(setup, tmp_path, monkeypatch):
    class _FakeAdd:
        value = "New Tester"

        def __init__(self, registry, key, parent, **_kw):
            assert (registry, key) == (TESTERS, "T2")

        def exec(self):
            return QDialog.DialogCode.Accepted

    monkeypatch.setattr(gui, "_AddItemDialog", _FakeAdd)
    dlg = _launcher(setup, tmp_path, "--add-items")
    _scan(dlg, dlg.tester_edit, "T2")
    assert dlg.tester_edit.property("valid") == "yes"
    assert setup.lookup(TESTERS, "T2") == (True, {"Name": "New Tester"})


def test_last_setup_only_with_flag(setup, tmp_path):
    ids = {"tester": "T1", "console": "WL134Q00007", "hydrophone": "HYDRO2246"}
    setup.save_last_setup(ids)
    assert _launcher(setup, tmp_path).tester_edit.text() == ""

    dlg = _launcher(setup, tmp_path, "--use-last-setup")
    assert [f.valid for f in (dlg._tester, dlg._console, dlg._hydrophone)] \
        == [True, True, True]
    assert dlg.txm_edit.text() == ""
    assert dlg.freq_combo.currentData() is None


def test_accept_saves_last_setup(setup, tmp_path):
    dlg = _launcher(setup, tmp_path)
    dlg.tester_edit.setText("T1")
    dlg.console_edit.setText("WL134Q00007")
    dlg.hydrophone_edit.setText("HYDRO2585")
    dlg.accept()
    assert setup.load_last_setup() == {
        "tester": "T1", "console": "WL134Q00007", "hydrophone": "HYDRO2585"}


def _states(dlg: gui.LauncherDialog) -> dict:
    return {name: led.property("state") for name, led in dlg.hw_leds.items()}


def _wait_until(predicate, timeout_s: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while not predicate() and time.monotonic() < deadline:
        QTest.qWait(10)
    return predicate()


@pytest.mark.parametrize("flags, hv", [((), "Console"),
                                       (("--ext-power",), "HV supply")])
def test_hardware_leds_show_probe_result(setup, tmp_path, monkeypatch,
                                         flags, hv):
    calls = []

    def _probe(*, ext_power_supply):
        calls.append(ext_power_supply)
        return {"Scope": True, "Transducer": False, hv: True}

    monkeypatch.setattr(gui, "probe_hardware", _probe)
    dlg = _launcher(setup, tmp_path, *flags, hardware_ok=False)
    assert _states(dlg) == dict.fromkeys(["Scope", "Transducer", hv],
                                         "checking")
    assert not calls  # polling starts only once the launcher is shown

    dlg.show()
    assert _wait_until(lambda: _states(dlg)["Scope"] == "ok")
    assert _states(dlg) == {"Scope": "ok", "Transducer": "missing", hv: "ok"}
    assert calls[0] is bool(flags)
    assert "not detected" in dlg.hw_leds["Transducer"].toolTip()

    dlg.reject()  # the run must not start while a probe is in flight
    assert not dlg._hw_poller.isRunning()


def test_hardware_leds_unknown_when_probe_fails(setup, tmp_path,
                                                monkeypatch):
    def _probe(*, ext_power_supply):
        raise OSError("driver missing")

    monkeypatch.setattr(gui, "probe_hardware", _probe)
    dlg = _launcher(setup, tmp_path, hardware_ok=False)
    dlg.show()
    assert _wait_until(lambda: _states(dlg)["Scope"] == "")
    assert set(_states(dlg).values()) == {""}
    dlg.reject()


def test_start_blocked_until_hardware_detected(setup, tmp_path):
    setup.save_last_setup({"tester": "T1", "console": "WL134Q00007",
                           "hydrophone": "HYDRO2246"})
    dlg = _launcher(setup, tmp_path, "--use-last-setup", hardware_ok=False)
    _scan(dlg, dlg.txm_edit, "WLM44Q00012")
    assert not dlg.start_button.isEnabled()
    assert dlg.start_button.toolTip() == "Checking hardware\u2026"

    dlg._show_hardware_status(
        {"Scope": True, "Transducer": False, "Console": False})
    assert not dlg.start_button.isEnabled()
    assert dlg.start_button.toolTip().startswith(
        "Not detected: Transducer, Console.")

    dlg._show_hardware_status(
        {"Scope": True, "Transducer": True, "Console": True})
    assert dlg.start_button.isEnabled()
    assert dlg.start_button.toolTip() == ""

    # A failed check must not lock the operator out; the run's own
    # connection checks still apply.
    dlg._show_hardware_status(None)
    assert dlg.start_button.isEnabled()


# ----------------------------------------------------------------------
# Console S/N auto-detection
# ----------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _no_console(monkeypatch):
    """No test talks to a real console; tests that need one patch
    ``read_console_identity`` themselves."""
    def _read():
        raise RuntimeError("Console is not connected.")
    monkeypatch.setattr(gui, "read_console_identity", _read)


def _console_reads(monkeypatch, sn: str, hwid: str = "4ZGd1kX") -> list:
    calls = []

    def _read():
        calls.append(1)
        return sn, hwid
    monkeypatch.setattr(gui, "read_console_identity", _read)
    return calls


def _plug_console(dlg: gui.LauncherDialog, present: bool = True) -> None:
    dlg._show_hardware_status(
        {"Scope": True, "Transducer": True, "Console": present})
    if dlg._console_detector is not None:
        dlg._console_detector.wait()
    QApplication.processEvents()


def test_console_sn_read_from_user_config(setup, tmp_path, monkeypatch):
    _console_reads(monkeypatch, "WL134Q00007")
    dlg = _launcher(setup, tmp_path, hardware_ok=False)
    _plug_console(dlg)
    assert dlg.console_edit.text() == "WL134Q00007"
    assert dlg._console.valid
    assert dlg._console.detail.toolTip() == "4ZGd1kX"


def test_console_sn_found_by_hwid_when_config_has_none(setup, tmp_path,
                                                      monkeypatch):
    _console_reads(monkeypatch, "", hwid="4ZGd1kX")
    dlg = _launcher(setup, tmp_path, hardware_ok=False)
    _plug_console(dlg)
    assert dlg.console_edit.text() == "WL134Q00007"
    assert dlg._console.valid


def test_console_sn_unknown_leaves_field_for_operator(setup, tmp_path,
                                                     monkeypatch):
    _console_reads(monkeypatch, "", hwid="unregistered")
    dlg = _launcher(setup, tmp_path, hardware_ok=False)
    _plug_console(dlg)
    assert dlg.console_edit.text() == ""
    assert not dlg._console.valid
    assert "not set" in dlg._console.detail.toolTip()


def test_console_read_error_is_shown(setup, tmp_path):
    dlg = _launcher(setup, tmp_path, hardware_ok=False)
    _plug_console(dlg)
    assert dlg.console_edit.text() == ""
    assert dlg._console.detail.text() == "Could not read console"
    assert dlg._console.detail.toolTip() == "Console is not connected."


def test_typed_console_sn_is_not_overwritten(setup, tmp_path, monkeypatch):
    calls = _console_reads(monkeypatch, "WL134Q00099")
    dlg = _launcher(setup, tmp_path, hardware_ok=False)
    dlg.console_edit.setText("WL134Q00007")
    _plug_console(dlg)
    assert calls == []
    assert dlg.console_edit.text() == "WL134Q00007"


def test_console_detected_replaces_last_setup(setup, tmp_path, monkeypatch):
    setup.add(CONSOLES, "WL134Q00008", "other")
    setup.save_last_setup({"console": "WL134Q00008"})
    _console_reads(monkeypatch, "WL134Q00007")
    dlg = _launcher(setup, tmp_path, "--use-last-setup", hardware_ok=False)
    assert dlg.console_edit.text() == "WL134Q00008"
    _plug_console(dlg)
    assert dlg.console_edit.text() == "WL134Q00007"


def test_console_read_once_per_connection(setup, tmp_path, monkeypatch):
    calls = _console_reads(monkeypatch, "WL134Q00007")
    dlg = _launcher(setup, tmp_path, hardware_ok=False)
    _plug_console(dlg)
    _plug_console(dlg)
    assert len(calls) == 1
    _plug_console(dlg, present=False)
    _plug_console(dlg)
    assert len(calls) == 2


def test_no_console_read_with_external_supply(setup, tmp_path, monkeypatch):
    calls = _console_reads(monkeypatch, "WL134Q00007")
    dlg = _launcher(setup, tmp_path, "--ext-power", hardware_ok=False)
    _plug_console(dlg)
    assert calls == []


def test_add_items_prefills_detected_console_hwid(setup, tmp_path,
                                                  monkeypatch):
    _console_reads(monkeypatch, "WL134Q00042", hwid="NewHwid")
    seen = {}

    class _FakeAdd:
        def __init__(self, registry, key, parent=None, *, value="", **_):
            seen.update(key=key, value=value)
            self.value = value

        def exec(self):
            return QDialog.DialogCode.Accepted

    monkeypatch.setattr(gui, "_AddItemDialog", _FakeAdd)
    dlg = _launcher(setup, tmp_path, "--add-items", hardware_ok=False)
    _plug_console(dlg)
    assert dlg._console.missing
    _scan(dlg, dlg.console_edit, "WL134Q00042")
    assert seen == {"key": "WL134Q00042", "value": "NewHwid"}
    assert setup.lookup(CONSOLES, "WL134Q00042") == (True, {"HWID": "NewHwid"})
    assert dlg._console.valid


# ----------------------------------------------------------------------
# --dry-run: simulated hardware
# ----------------------------------------------------------------------
@pytest.fixture
def real_hardware_calls(monkeypatch) -> list:
    """Records any attempt to reach real hardware: the hardware probe
    (PicoSDK enumeration, serial ports) or a real ``LIFUInterface``.
    Console reads go through the real :func:`read_console_identity`,
    as the launcher makes them."""
    import openlifu_sdk.io
    import serial.tools.list_ports

    from openlifu_verification import device_info, hardware_status

    calls = []

    def _recorder(name):
        def _call(*_a, **_k):
            calls.append(name)
            raise RuntimeError(f"{name} called in a dry run")
        return _call

    monkeypatch.setattr(gui, "probe_hardware", _recorder("probe_hardware"))
    monkeypatch.setattr(hardware_status, "picoscope_present",
                        _recorder("picoscope_present"))
    monkeypatch.setattr(serial.tools.list_ports, "comports",
                        _recorder("comports"))
    monkeypatch.setattr(openlifu_sdk.io, "LIFUInterface",
                        _recorder("LIFUInterface"))
    monkeypatch.setattr(gui, "read_console_identity",
                        device_info.read_console_identity)
    return calls


def _sim_console_hwid() -> str:
    from openlifu_verification.simulated_hardware import (
        simulated_console_identity)
    return simulated_console_identity()[1]


def _click(widget) -> None:
    QTest.mouseClick(widget, Qt.MouseButton.LeftButton)


@pytest.mark.parametrize("flags, hv", [((), "Console"),
                                       (("--ext-power",), "HV supply")])
def test_dry_run_shows_simulated_lights(setup, tmp_path, real_hardware_calls,
                                        flags, hv):
    dlg = _launcher(setup, tmp_path, "--dry-run", *flags, hardware_ok=False)
    assert list(dlg.hw_leds) == ["Scope", "Transducer", hv]
    assert set(_states(dlg).values()) == {"checking"}
    dlg.show()
    assert _wait_until(lambda: set(_states(dlg).values()) == {"ok"})
    assert dlg.hw_leds["Scope"].toolTip().startswith("Scope: simulated")
    dlg.reject()
    assert not dlg._hw_poller.isRunning()
    assert real_hardware_calls == []


def test_dry_run_detects_simulated_console(setup, tmp_path,
                                           real_hardware_calls):
    from openlifu_verification.simulated_hardware import SIMULATED_CONSOLE_SN

    dlg = _launcher(setup, tmp_path, "--dry-run", hardware_ok=False)
    dlg.show()
    assert _wait_until(
        lambda: dlg.console_edit.text() == SIMULATED_CONSOLE_SN)
    assert dlg._console.missing  # not registered: Enter offers to add it
    assert dlg._detected_console_hwid == _sim_console_hwid()
    dlg.reject()

    setup.add(CONSOLES, SIMULATED_CONSOLE_SN, _sim_console_hwid())
    dlg = _launcher(setup, tmp_path, "--dry-run")
    assert dlg.console_edit.text() == SIMULATED_CONSOLE_SN
    assert dlg._console.valid
    assert dlg._console.detail.toolTip() == _sim_console_hwid()
    assert real_hardware_calls == []


def test_dry_run_add_console_reads_simulated_hwid(setup, tmp_path,
                                                  real_hardware_calls):
    dlg = _launcher(setup, tmp_path, "--dry-run", "--add-items")
    add = gui._AddItemDialog(CONSOLES, "WL134Q00001", dlg,
                             read_console_hwid=dlg._read_console_hwid)
    add._read_console_hwid()  # the "Read from console" button
    assert add.value == _sim_console_hwid()
    assert real_hardware_calls == []


def test_dry_run_unplugging_blocks_start(setup, tmp_path,
                                         real_hardware_calls):
    from openlifu_verification.simulated_hardware import (
        SIMULATED_CONSOLE_SN, SimulatedHardware)

    setup.add(CONSOLES, SIMULATED_CONSOLE_SN, _sim_console_hwid())
    sim = SimulatedHardware()
    args = gui.build_gui_parser().parse_args(
        ["--reports-root", str(tmp_path / "reports"), "--dry-run"])
    dlg = gui.LauncherDialog(args, app_version="9.9", setup_root=setup.root,
                             simulated_hardware=sim)
    reads = []
    real_read = dlg._read_console_identity
    dlg._read_console_identity = lambda: reads.append(1) or real_read()
    dlg.show()
    assert _wait_until(lambda: dlg._console.valid)
    _scan(dlg, dlg.txm_edit, "WLM44Q00012")
    _scan(dlg, dlg.tester_edit, "T1")
    _scan(dlg, dlg.hydrophone_edit, "HYDRO2246")
    assert dlg.start_button.isEnabled()
    assert reads == [1]

    _click(dlg.hw_leds["Transducer"])
    assert _wait_until(lambda: _states(dlg)["Transducer"] == "missing")
    assert not sim.is_present("Transducer")
    assert not dlg.start_button.isEnabled()
    assert dlg.start_button.toolTip().startswith("Not detected: Transducer.")
    _click(dlg.hw_leds["Transducer"])
    assert _wait_until(lambda: dlg.start_button.isEnabled())

    # Unplugging and replugging the console reads it again.
    _click(dlg.hw_leds["Console"])
    assert _wait_until(lambda: _states(dlg)["Console"] == "missing")
    assert not dlg.start_button.isEnabled()
    _click(dlg.hw_leds["Console"])
    assert _wait_until(lambda: len(reads) == 2)
    assert _wait_until(lambda: dlg.start_button.isEnabled())
    assert dlg.console_edit.text() == SIMULATED_CONSOLE_SN
    dlg.reject()
    assert real_hardware_calls == []


def test_dry_run_console_matches_dry_run_device_info():
    from openlifu_verification.device_info import DeviceInfo
    from openlifu_verification.dry_run import DryRunTank
    info = DeviceInfo.collect(DryRunTank(rng_seed=0))
    assert info.console_hwid == _sim_console_hwid()


# ----------------------------------------------------------------------
# Hydrophone calibration preview / replace
# ----------------------------------------------------------------------
def _new_calibration(setup: SetupData, tmp_path, *, sn="2246",
                     date="01-Nov-2026") -> Path:
    """A copy of HYDRO2246's calibration with a new S/N / date."""
    text = setup.calibration_path("HYDRO2246").read_text(encoding="utf-8")
    text = (text.replace("HYD_SN\t2246", f"HYD_SN\t{sn}")
                .replace("19-Dec-2022", date))
    path = tmp_path / f"HNR0500-{sn}_new_cal.txt"
    path.write_text(text, encoding="utf-8")
    return path


def test_hydrophone_calibration_is_a_link(setup, tmp_path, monkeypatch):
    opened = []

    class _FakeCal:
        def __init__(self, setup_, hydrophone_id, *, can_replace, parent):
            opened.append((hydrophone_id, can_replace))

        def exec(self):
            return QDialog.DialogCode.Rejected

    monkeypatch.setattr(gui, "_CalibrationDialog", _FakeCal)
    dlg = _launcher(setup, tmp_path)
    detail = dlg._hydrophone.detail
    _scan(dlg, dlg.tester_edit, "nope")  # non-hydrophone stays plain
    assert dlg._tester.detail.textFormat() == Qt.TextFormat.PlainText
    _scan(dlg, dlg.hydrophone_edit, "HYDRO2246")
    assert detail.textFormat() == Qt.TextFormat.RichText
    assert 'href="calibration"' in detail.text()
    assert "HNR0500-2246" in detail.text()
    detail.linkActivated.emit("calibration")
    assert opened == [("HYDRO2246", False)]

    _scan(dlg, dlg.hydrophone_edit, "HYDRO0000")  # not registered
    assert detail.textFormat() == Qt.TextFormat.PlainText


def test_calibration_dialog_shows_metadata_and_plot(setup):
    dlg = gui._CalibrationDialog(setup, "HYDRO2246")
    texts = [w.text() for w in dlg.findChildren(QLabel)]
    for expected in ("HYDRO2246", "2246", "Onda HNR0500", "19-Dec-2022",
                     "Sensitivity @ 155 kHz", "Sensitivity @ 400 kHz"):
        assert expected in texts
    assert any(t.endswith(" nV/Pa") for t in texts)
    assert not dlg.plot_label.pixmap().isNull()
    assert dlg.replace_button.isHidden()
    assert dlg.save_button.isHidden()


def test_calibration_replace_previews_then_saves(setup, tmp_path):
    new = _new_calibration(setup, tmp_path)
    dlg = gui._CalibrationDialog(setup, "HYDRO2246", can_replace=True)
    assert not dlg.replace_button.isHidden()
    assert dlg.load_replacement(new)
    assert dlg.replacement_warnings() == []
    assert dlg.warning_label.isHidden()
    assert not dlg.save_button.isHidden()
    texts = [w.text() for w in dlg.findChildren(QLabel)]
    assert "01-Nov-2026" in texts and "19-Dec-2022" in texts
    # Nothing changes until Save.
    assert setup.calibration_path("HYDRO2246").name.startswith(
        "HNR0500-2246_xxxxxx")
    assert dlg.save_replacement()
    assert dlg.result() == QDialog.DialogCode.Accepted
    assert setup.calibration_path("HYDRO2246").name == new.name


def test_calibration_replace_warns_on_suspect_file(setup, tmp_path):
    new = _new_calibration(setup, tmp_path, sn="2538", date="01-Jan-2020")
    dlg = gui._CalibrationDialog(setup, "HYDRO2246", can_replace=True)
    assert dlg.load_replacement(new)
    warnings = dlg.replacement_warnings()
    assert len(warnings) == 2
    assert "S/N 2538, not 2246" in warnings[0]
    assert "not newer" in warnings[1]
    assert not dlg.warning_label.isHidden()


def test_calibration_replace_rejects_unparseable_file(setup, tmp_path,
                                                      monkeypatch):
    shown = []
    monkeypatch.setattr(gui.QMessageBox, "warning",
                        lambda *a: shown.append(a[2]))
    bogus = tmp_path / "notes.txt"
    bogus.write_text("not a calibration")
    dlg = gui._CalibrationDialog(setup, "HYDRO2246", can_replace=True)
    assert not dlg.load_replacement(bogus)
    assert dlg.new is None and dlg.save_button.isHidden()
    assert "Could not read notes.txt" in shown[0]


def test_saved_calibration_updates_launcher(setup, tmp_path, monkeypatch):
    new = _new_calibration(setup, tmp_path)

    class _SavingCal:
        def __init__(self, setup_, hydrophone_id, *, can_replace, parent):
            assert can_replace
            setup_.replace_hydrophone_calibration(hydrophone_id, new)

        def exec(self):
            return QDialog.DialogCode.Accepted

    monkeypatch.setattr(gui, "_CalibrationDialog", _SavingCal)
    dlg = _launcher(setup, tmp_path, "--add-items")
    _scan(dlg, dlg.hydrophone_edit, "HYDRO2246")
    dlg._hydrophone.detail.linkActivated.emit("calibration")
    assert dlg._hydrophone.valid
    assert dlg._hydrophone.detail.toolTip().startswith(new.name)
    assert dlg.selected_hydrophone_calibration.name == new.name


# ----------------------------------------------------------------------
# --add-items default from the setup data's gui_settings.json
# ----------------------------------------------------------------------
@pytest.mark.parametrize("setting, flags, expected", [
    (None, [], False),                  # template default
    (True, [], True),
    (True, ["--no-add-items"], False),
    (False, ["--add-items"], True),
])
def test_add_items_default_from_setup_data(setup, tmp_path, monkeypatch,
                                           setting, flags, expected):
    import json
    if setting is not None:
        setup.gui_settings_path.write_text(json.dumps({"add_items": setting}))
    seen = []

    class _FakeLauncher:
        def __init__(self, args, **_kw):
            seen.append(args.add_items)

        def exec(self):
            return QDialog.DialogCode.Rejected

    monkeypatch.setattr(gui, "LauncherDialog", _FakeLauncher)
    monkeypatch.setattr(gui, "apply_theme", lambda *a: None)
    gui.main(["--reports-root", str(tmp_path / "reports"),
              "--setup-data", str(setup.root), *flags])
    assert seen == [expected]


def test_add_items_follows_setup_folder_change(setup, tmp_path, monkeypatch):
    """Switching setup data folders in the launcher applies the new
    folder's default to the next launcher."""
    other = SetupData(tmp_path / "other")
    other.initialize()
    other.gui_settings_path.write_text('{"add_items": true}')
    seen = []

    class _FakeLauncher:
        def __init__(self, args, **_kw):
            seen.append(args.add_items)

        def exec(self):
            return (QDialog.DialogCode.Accepted if len(seen) == 1
                    else QDialog.DialogCode.Rejected)

        collected_prefs = None
        carried_ids: dict = {}
        selected_frequency_kHz = 400.0
        selected_reports_root = tmp_path / "reports"
        selected_setup_root = other.root
        selected_hydrophone_calibration = tmp_path / "cal.txt"
        selected_console_hwid = None

    class _Settings:
        def value(self, *_a, **_k):
            return ""

        def setValue(self, *_a):
            pass

    monkeypatch.setattr(gui, "LauncherDialog", _FakeLauncher)
    monkeypatch.setattr(gui, "_run_once", lambda *a: ("another", 0))
    monkeypatch.setattr(gui, "apply_theme", lambda *a: None)
    monkeypatch.setattr(gui, "QSettings", lambda *a: _Settings())
    gui.main(["--reports-root", str(tmp_path / "reports"),
              "--setup-data", str(setup.root)])
    assert seen == [False, True]


# ----------------------------------------------------------------------
# "Scan Another" keeps the tester and hydrophone
# ----------------------------------------------------------------------
def test_prefill_fills_and_validates(setup, tmp_path):
    args = gui.build_gui_parser().parse_args(
        ["--reports-root", str(tmp_path / "reports")])
    dlg = gui.LauncherDialog(args, app_version="9.9", setup_root=setup.root,
                             prefill={"tester": "T1",
                                      "hydrophone": "HYDRO2246"})
    assert dlg.tester_edit.text() == "T1"
    assert dlg.hydrophone_edit.text() == "HYDRO2246"
    assert dlg._tester.valid and dlg._hydrophone.valid
    assert dlg.console_edit.text() == ""
    assert dlg.txm_edit.text() == ""
    assert dlg.carried_ids == {"tester": "T1", "hydrophone": "HYDRO2246"}


def test_prefill_overrides_last_setup(setup, tmp_path):
    setup.add(TESTERS, "T2", "John Roe")
    setup.save_last_setup({"tester": "T1", "console": "WL134Q00007",
                           "hydrophone": "HYDRO2246"})
    args = gui.build_gui_parser().parse_args(
        ["--reports-root", str(tmp_path / "reports"), "--use-last-setup"])
    dlg = gui.LauncherDialog(args, app_version="9.9", setup_root=setup.root,
                             prefill={"tester": "T2",
                                      "hydrophone": "HYDRO2538"})
    assert dlg.tester_edit.text() == "T2"
    assert dlg.console_edit.text() == "WL134Q00007"
    assert dlg.hydrophone_edit.text() == "HYDRO2538"


def test_scan_another_prefills_previous_tester_and_hydrophone(
        setup, tmp_path, monkeypatch):
    prefills = []

    class _FakeLauncher:
        def __init__(self, args, *, prefill, **_kw):
            prefills.append(dict(prefill))
            self._first = len(prefills) == 1

        def exec(self):
            return (QDialog.DialogCode.Accepted if self._first
                    else QDialog.DialogCode.Rejected)

        collected_prefs = None
        carried_ids = {"tester": "T1", "hydrophone": "HYDRO2246"}
        selected_frequency_kHz = 400.0
        selected_reports_root = tmp_path / "reports"
        selected_setup_root = setup.root
        selected_hydrophone_calibration = tmp_path / "cal.txt"
        selected_console_hwid = "4ZGd1kX"

    monkeypatch.setattr(gui, "LauncherDialog", _FakeLauncher)
    monkeypatch.setattr(gui, "_run_once", lambda *a: ("another", 0))
    monkeypatch.setattr(gui, "apply_theme", lambda *a: None)
    monkeypatch.setattr(gui, "QSettings", lambda *a: _FakeSettings())
    gui.main(["--reports-root", str(tmp_path / "reports"),
              "--setup-data", str(setup.root)])
    assert prefills == [{}, {"tester": "T1", "hydrophone": "HYDRO2246"}]


class _FakeSettings:
    def value(self, *_a, **_k):
        return ""

    def setValue(self, *_a):
        pass
