"""ResultDialog: test-summary squares and the unwritten-data prompt."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_report  # noqa: E402
import run_report_gui as gui  # noqa: E402
from PySide6.QtWidgets import QApplication, QLabel, QMessageBox  # noqa: E402

from openlifu_verification.characterization import ReportRow  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _app():
    yield QApplication.instance() or QApplication([])


def _result() -> run_report.RunResult:
    return run_report.RunResult(
        passed=False,
        graded_rows=[
            ReportRow("D.3", "Plane-Wave Hydrophone Depth", 50.0063, "mm",
                      "PASS", "50.0 \u00b1 2.00 mm", "depth = 50.006 mm"),
            ReportRow("D.14", "Focused Pulse PNP", 0.1, "MPa", "FAIL", 0.225),
        ],
        elapsed_s=186.4,
    )


def test_summary_squares_and_count():
    dlg = gui.ResultDialog(_result(), can_write=True)
    squares = dlg.findChildren(QLabel, "gradeSquare")
    assert [s.property("status") for s in squares] == ["PASS", "FAIL"]
    assert squares[0].toolTip().splitlines() == [
        "D.3 Plane-Wave Hydrophone Depth: PASS",
        "Value: 50.01 mm",
        "Limit: 50.0 \u00b1 2.00 mm",
        "depth = 50.006 mm",
    ]
    texts = [lbl.text() for lbl in dlg.findChildren(QLabel)]
    assert "1/2 tests passed in 3:06" in texts


@pytest.mark.parametrize("choice, action", [("exit", "Exit"),
                                            ("another", "Continue")])
def test_unwritten_prompt_names_the_action(monkeypatch, choice, action):
    shown = []

    def fake_warning(parent, title, text, *args):
        shown.append(text)
        return QMessageBox.StandardButton.No

    monkeypatch.setattr(QMessageBox, "warning", fake_warning)
    dlg = gui.ResultDialog(run_report.RunResult(passed=True), can_write=True)
    dlg._finish_with(choice)
    assert shown and shown[0].endswith(f"Are you sure you want to {action}?")


# ----------------------------------------------------------------------
# Hydrophone depth calibration failure: Continue / Abort
# ----------------------------------------------------------------------
@pytest.mark.parametrize("button, expected", [
    ("Continue at 50 mm", True), ("Abort Scan", False)])
def test_depth_failure_modal(monkeypatch, button, expected):
    seen = {}

    def _exec(box):
        seen.update(title=box.windowTitle(), text=box.text(),
                    info=box.informativeText(),
                    default=box.defaultButton().text())
        chosen = next(b for b in box.buttons() if b.text() == button)
        monkeypatch.setattr(box, "clickedButton", lambda: chosen)
        return 0
    monkeypatch.setattr(QMessageBox, "exec", _exec)
    assert gui._ask_continue_after_depth_failure(
        None, "Plane-wave depth calibration failed: no arrival", 50.0) is expected
    assert seen["title"] == "Hydrophone depth calibration failed"
    assert seen["text"] == "Plane-wave depth calibration failed: no arrival."
    assert "free of bubbles" in seen["info"]
    assert "nominal depth of 50.0 mm" in seen["info"]
    assert seen["default"] == "Abort Scan"


@pytest.mark.parametrize("answer", [True, False])
def test_worker_waits_for_depth_answer(answer):
    import threading
    signals = gui._RunSignals()
    worker = gui._WorkerThread(None, None, signals)
    asked = []
    signals.depth_failed.connect(lambda p, n: asked.append((p, n)))
    got = []
    t = threading.Thread(
        target=lambda: got.append(worker._on_depth_failure("bad", 50.0)))
    t.start()
    t.join(0.2)
    assert t.is_alive()          # blocked until the operator answers
    worker.answer_depth_failure(answer)
    t.join(2)
    QApplication.processEvents()  # deliver the cross-thread signal
    assert got == [answer] and asked == [("bad", 50.0)]


@pytest.mark.parametrize("acknowledged", [True, False])
def test_worker_forwards_acknowledged_abort(monkeypatch, acknowledged):
    result = run_report.RunResult(aborted=True,
                                  abort_acknowledged=acknowledged)
    monkeypatch.setattr(run_report, "run", lambda *a, **k: result)
    signals = gui._RunSignals()
    seen = []
    signals.aborted.connect(seen.append)
    gui._WorkerThread(None, None, signals).run()
    assert seen == [acknowledged]
