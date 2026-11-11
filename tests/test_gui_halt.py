"""Run halts reach the GUI with their kind; outdated firmware gets its
own explanation instead of the hydrophone-placement advice."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_report  # noqa: E402
import run_report_gui as gui  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from openlifu_verification import FirmwareOutOfDate  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _app():
    yield QApplication.instance() or QApplication([])


@pytest.mark.parametrize("kind", ["firmware", "setup", None])
def test_worker_emits_halt_kind(monkeypatch, kind):
    result = run_report.RunResult(halted_reason="stopped", halted_kind=kind)
    monkeypatch.setattr(run_report, "run", lambda *a, **k: result)
    signals = gui._RunSignals()
    seen = []
    signals.halted.connect(lambda reason, k: seen.append((reason, k)))
    gui._WorkerThread(None, None, signals).run()  # run inline
    assert seen == [("stopped", kind or "setup")]


def test_firmware_popup_lists_devices(monkeypatch):
    shown = {}

    def _exec(box):
        shown.update(title=box.windowTitle(), text=box.text(),
                     info=box.informativeText())
        return 0
    monkeypatch.setattr(QMessageBox, "exec", _exec)
    err = FirmwareOutOfDate([("TXM firmware", "2.0.5", "2.0.8"),
                             ("Console firmware", "", "1.2.7")])
    gui._show_firmware_out_of_date(str(err))
    assert shown["title"] == "Firmware update required"
    assert "test was not run" in shown["text"]
    assert shown["info"].startswith(
        "TXM firmware: version 2.0.5 (required: 2.0.8 or newer)\n"
        "Console firmware: version could not be read "
        "(required: 1.2.7 or newer)\n\nUpdate the firmware")
    assert "hydrophone" not in shown["info"]
