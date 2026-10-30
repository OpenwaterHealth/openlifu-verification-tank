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
from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtTest import QTest  # noqa: E402
from PyQt6.QtWidgets import QApplication, QDialog, QLabel, QPushButton  # noqa: E402

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
    ("console_edit", "WL134W00007", "Expected WL134Q#####"),
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

        def __init__(self, registry, key, parent):
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


def test_no_hardware_leds_in_dry_run(setup, tmp_path, monkeypatch):
    monkeypatch.setattr(gui, "probe_hardware", pytest.fail)
    dlg = _launcher(setup, tmp_path, "--dry-run")
    dlg.show()
    assert dlg.hw_leds == {}
    assert dlg._hw_poller is None
    dlg.reject()
