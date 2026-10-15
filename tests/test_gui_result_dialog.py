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
from PyQt6.QtWidgets import QApplication, QLabel, QMessageBox  # noqa: E402

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
