"""Regenerate the screenshots in docs/images/manual/ for docs/user_manual.md.

Drives the real report GUI (``scripts/run_report_gui.py``) in
``--dry-run`` (simulated hardware), the way an operator would: a timer
fills in fields, opens dialogs and presses buttons, and saves each
screen as a PNG. Everything runs in a temporary setup data / reports
folder, and the GUI's per-user state (active setup data folder, saved
reports folder, operator prefs) is redirected there, so running this
never changes the machine's real settings.

Usage::

    python docs/make_manual_screenshots.py            # offscreen
    python docs/make_manual_screenshots.py --out DIR  # elsewhere

Needs the GUI's dependencies (PySide6, openlifu-sdk) and, for the
report page image, ``pdftoppm`` (Poppler) on PATH (skipped if
missing).
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path, PureWindowsPath

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "docs" / "images" / "manual"

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_SCALE_FACTOR", "2")   # crisp images
sys.path.insert(0, str(ROOT / "scripts"))

import run_report  # noqa: E402
import run_report_gui as gui  # noqa: E402
from PySide6.QtCore import QObject, QPoint, QTimer  # noqa: E402
from PySide6.QtGui import QPainter, QPixmap  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QDialog,
    QMessageBox,
    QPushButton,
)

from openlifu_verification import paths  # noqa: E402
from openlifu_verification.setup_data import (  # noqa: E402
    CONSOLES,
    TESTERS,
    SetupData,
)
from openlifu_verification.simulated_hardware import (  # noqa: E402
    SIMULATED_CONSOLE_SN,
    simulated_console_identity,
)

TXM_SN = "WLM44Q00012"       # 400 kHz module
TESTER = ("JD01", "Jane Doe")
HYDROPHONE = "HYDRO2585"


# ----------------------------------------------------------------------
# Driver: run a list of (wait-for, action) steps on a timer
# ----------------------------------------------------------------------
class Driver(QObject):
    """Polls every 50 ms; when the first step's ``wait`` returns a
    widget, pops the step and calls ``action(widget)``. Actions may
    open modal dialogs: the timer keeps firing in their event loop."""

    def __init__(self, steps, out: Path):
        super().__init__()
        self.steps = list(steps)
        self.out = out
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(50)
        self._busy = False

    def _tick(self) -> None:
        if self._busy or not self.steps:
            return
        wait, action = self.steps[0]
        widget = wait()
        if widget is None:
            return
        self.steps.pop(0)
        if os.environ.get("SCREENSHOT_DEBUG"):
            print("step:", getattr(action, "__name__", action), flush=True)
        self._busy = True
        try:
            action(widget)
        finally:
            self._busy = False

    def save(self, widget, name: str) -> None:
        if isinstance(widget, gui.LauncherDialog):
            dress(widget)
        QApplication.processEvents()
        path = self.out / f"{name}.png"
        widget.grab().save(str(path))
        print("saved", path.relative_to(ROOT) if path.is_relative_to(ROOT)
              else path)


#: Folders shown in the screenshots (the real ones are temporary).
SHOWN_SETUP_DATA = PureWindowsPath(
    r"C:\Users\operator\.openlifu_verification\setup_data")
SHOWN_REPORTS = PureWindowsPath(r"C:\Users\operator\Documents\test_reports")


def dress(launcher) -> None:
    """Cosmetic only: show typical Windows folders instead of the
    temporary ones, and hide the dry run's "Simulated:" label, so the
    screenshots look like a station's."""
    launcher._set_link(launcher.setup_link, SHOWN_SETUP_DATA)
    launcher._set_link(launcher.reports_link, SHOWN_REPORTS)
    for label in launcher.findChildren(gui.QLabel):
        if label.text() == "Simulated:":
            label.hide()


def modal(cls=None, title: str | None = None):
    """Wait-for: the active modal widget, if it matches."""
    def wait():
        w = QApplication.activeModalWidget()
        if w is None or not w.isVisible():
            return None
        if cls is not None and not isinstance(w, cls):
            return None
        if title is not None and title not in w.windowTitle():
            return None
        return w
    return wait


def launcher_ready(console_detected: bool = True):
    """Wait-for: the launcher, with every hardware light settled (and
    the console S/N detected)."""
    def wait():
        w = modal(gui.LauncherDialog)()
        if w is None or w._hw_blocker:
            return None
        if console_detected and not w._console.valid:
            return None
        return w
    return wait


def after(ms: int, wait):
    """Wait-for: ``wait``'s widget, but only ``ms`` after first seen."""
    state = {}

    def inner():
        w = wait()
        if w is None:
            return None
        import time
        t0 = state.setdefault("t0", time.monotonic())
        return w if time.monotonic() - t0 >= ms / 1000 else None
    return inner


def later(fn) -> None:
    """Run ``fn`` from the event loop, so an action that opens a modal
    dialog returns at once and the driver keeps ticking inside it."""
    QTimer.singleShot(0, fn)


def button(dialog, text: str) -> QPushButton:
    return next(b for b in dialog.findChildren(QPushButton)
                if b.text() == text)


def scan(launcher, edit, text: str) -> None:
    """Type ``text`` and press Enter, as a barcode reader does."""
    edit.setFocus()
    edit.setText(text)
    launcher._on_enter(edit)


def with_menu(driver: Driver, launcher, field, name: str) -> None:
    """Screenshot the launcher with ``field``'s dropdown list open
    (menus are separate windows, so composite them)."""
    menu = launcher._registry_menu(field)
    menu.setMinimumWidth(field.edit.width())
    menu.adjustSize()
    QApplication.processEvents()
    base, popup = launcher.grab(), menu.grab()
    pos = field.edit.mapTo(launcher, field.edit.rect().bottomLeft())
    out = QPixmap(base.size())
    out.setDevicePixelRatio(base.devicePixelRatio())
    painter = QPainter(out)
    dress(launcher)
    QApplication.processEvents()
    base = launcher.grab()
    painter.drawPixmap(0, 0, base)
    painter.drawPixmap(pos + QPoint(0, 2), popup)
    painter.end()
    out.save(str(driver.out / f"{name}.png"))
    print("saved", name)


def fill_ready(launcher) -> None:
    scan(launcher, launcher.txm_edit, TXM_SN)
    scan(launcher, launcher.tester_edit, TESTER[0])
    scan(launcher, launcher.hydrophone_edit, HYDROPHONE)


# ----------------------------------------------------------------------
# Scenarios
# ----------------------------------------------------------------------
def scenario_main(d: Driver) -> list:
    """Launcher, calibration preview, add-item, hardware light, run,
    result, unwritten-data warning, write, Scan Another."""
    def unregistered_tester(w):
        later(lambda: scan(w, w.tester_edit, "JD02"))  # opens Add dialog

    def add_tester_dialog(dlg):
        dlg.value_edit.setText("John Smith")
        d.save(dlg, "add_tester")
        dlg.reject()

    def launcher_start(w):
        w.tester_edit.clear()
        w._show(w._tester, None)
        w.txm_edit.setFocus()
        d.save(w, "launcher_start")
        scan(w, w.txm_edit, TXM_SN)
        d.save(w, "launcher_txm")
        with_menu(d, w, w._tester, "launcher_tester_dropdown")
        w._pick_registered(w._tester, TESTER[0])
        # Not registered (validated as on leaving the field; Enter
        # would offer to add it).
        w.hydrophone_edit.setText("HYDRO9999")
        w._validate(w._hydrophone)
        d.save(w, "launcher_hydrophone_unknown")
        scan(w, w.hydrophone_edit, HYDROPHONE)
        d.save(w, "launcher_ready")
        w._toggle_simulated("Scope")

    def scope_unplugged(w):
        d.save(w, "launcher_scope_missing")
        w._toggle_simulated("Scope")

    def open_calibration(w):
        later(lambda: w._hydrophone.detail.linkActivated.emit("calibration"))

    def calibration(dlg):
        d.save(dlg, "calibration_preview")
        dlg.reject()

    def start(w):
        w.start_button.click()

    def running(w):
        d.save(w, "running")

    def result_pass(w):
        d.save(w, "result_pass")
        later(button(w, "Scan Another").click)  # not written: warning

    def unwritten_warning(box):
        d.save(box, "unwritten_warning")
        box.done(QMessageBox.StandardButton.No)

    def write(w):
        button(w, "Write Data to Device").click()

    def written(w):
        d.save(w, "result_written")
        d.state["run_dir"] = run_dir_of(w)
        button(w, "Scan Another").click()

    def scan_another(w):
        d.save(w, "launcher_scan_another")
        w.reject()

    def result_written(w):
        return w if w.write_line.isVisible() else None

    d.state = {}
    return [
        (launcher_ready(), unregistered_tester),
        (modal(gui._AddItemDialog), add_tester_dialog),
        (launcher_ready(), launcher_start),
        (lambda: (lambda w: w if w and w._hw_blocker.startswith("Not")
                  else None)(modal(gui.LauncherDialog)()), scope_unplugged),
        (launcher_ready(), open_calibration),
        (modal(gui._CalibrationDialog), calibration),
        (launcher_ready(), start),
        (lambda: (lambda w: w if w and running_far(w) else None)(
            modal(gui.RunningDialog)()), running),
        (after(400, modal(gui.ResultDialog)), result_pass),
        (modal(QMessageBox, "not written"), unwritten_warning),
        (modal(gui.ResultDialog), write),
        (lambda: (lambda w: result_written(w) if w else None)(
            modal(gui.ResultDialog)()), written),
        (launcher_ready(), scan_another),
    ]


def running_far(w) -> bool:
    """Enough acceptance gates resolved for a representative shot."""
    rows = w.findChildren(gui._GateRow)
    done = sum(r.state in (r.PASS, r.FAIL) for r in rows)
    return done >= 6


def run_dir_of(result_dialog) -> Path | None:
    for label in result_dialog.findChildren(gui.QLabel):
        text = label.text()
        if "Open report folder" in text and 'href="' in text:
            from PySide6.QtCore import QUrl
            href = text.split('href="', 1)[1].split('"', 1)[0]
            return Path(QUrl(href).toLocalFile())
    return None


def scenario_depth_failure(d: Driver) -> list:
    """Depth calibration fails -> Continue -> FAIL result."""
    real = run_report.DryRunTank

    def tank(**kw):
        t = real(**kw)
        calibrate = t.calibrate_hydrophone_depth

        def no_plane_wave_arrival(**k):
            if k.get("z_focus_mm") is None:      # plane wave only
                raise RuntimeError(
                    "no pulse arrival found in the capture window")
            return calibrate(**k)
        t.calibrate_hydrophone_depth = no_plane_wave_arrival
        return t
    run_report.DryRunTank = tank

    def go(w):
        fill_ready(w)
        w.start_button.click()

    def failure(box):
        d.save(box, "depth_failure")
        button(box, "Continue at 50 mm").click()

    def result(w):
        d.save(w, "result_fail")
        button(w, "Exit").click()

    return [
        (launcher_ready(), go),
        (modal(QMessageBox, "depth calibration failed"), failure),
        (after(400, modal(gui.ResultDialog)), result),
    ]


def scenario_peak_failure(d: Driver) -> list:
    """Peak found off center -> the Continue / Abort question."""
    real = run_report.DryRunTank
    run_report.DryRunTank = lambda **kw: real(peak_xy=(2.6, 0.2), **kw)

    def go(w):
        fill_ready(w)
        w.start_button.click()

    def failure(box):
        d.save(box, "peak_failure")
        button(box, "Abort Scan").click()

    return [
        (launcher_ready(), go),
        (modal(QMessageBox, "Peak search failed"), failure),
        (launcher_ready(), lambda w: w.reject()),
    ]


def scenario_firmware(d: Driver) -> list:
    """TXM firmware older than the minimum."""
    real = run_report.DryRunTank
    run_report.DryRunTank = lambda **kw: real(txm_fw_version="2.0.5", **kw)

    def go(w):
        fill_ready(w)
        w.start_button.click()

    def popup(box):
        d.save(box, "firmware_update_required")
        box.accept()

    return [
        (launcher_ready(), go),
        (modal(QMessageBox, "Firmware update required"), popup),
        (launcher_ready(), lambda w: w.reject()),
    ]


SCENARIOS = {
    "main": scenario_main,
    "depth": scenario_depth_failure,
    "peak": scenario_peak_failure,
    "firmware": scenario_firmware,
}


# ----------------------------------------------------------------------
def run_scenario(name: str, out: Path, work: Path) -> dict:
    setup = SetupData(work / "setup_data")
    setup.initialize()
    setup.add(TESTERS, *TESTER)
    setup.add(CONSOLES, SIMULATED_CONSOLE_SN, simulated_console_identity()[1])
    # Keep the GUI's per-user state inside ``work``.
    paths.SETUP_DATA_LOCATION_FILE = work / "setup_data_location.txt"

    class _Settings:
        def value(self, *_a, **_k):
            return ""

        def setValue(self, *_a):
            pass
    gui.QSettings = lambda *a: _Settings()

    app = QApplication.instance() or QApplication([])
    driver = Driver([], out)
    driver.steps = SCENARIOS[name](driver)
    gui.main(["--dry-run", "--add-items",
              "--setup-data", str(setup.root),
              "--reports-root", str(work / "reports"),
              "--prefs", str(work / "operator_prefs.json")])
    if driver.steps:
        raise SystemExit(f"scenario {name!r} stopped early; "
                         f"{len(driver.steps)} step(s) left")
    del app
    return getattr(driver, "state", {})


def save_report_page(run_dir: Path | None, out: Path) -> None:
    if run_dir is None or shutil.which("pdftoppm") is None:
        print("skipped report_pdf_page1 (no report or no pdftoppm)")
        return
    pdf = next(run_dir.glob("*_Report.pdf"))
    stem = out / "report_pdf_page1"
    subprocess.run(["pdftoppm", "-png", "-r", "110", "-f", "1", "-l", "1",
                    "-singlefile", str(pdf), str(stem)], check=True)
    print("saved", stem.with_suffix(".png"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--scenario", choices=list(SCENARIOS),
                        help="Run one scenario in this process (internal).")
    parser.add_argument("--work", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.scenario:
        state = run_scenario(args.scenario, args.out, args.work)
        if args.scenario == "main":
            save_report_page(state.get("run_dir"), args.out)
        return 0

    # Each scenario patches the dry run differently: run each in its
    # own process.
    with tempfile.TemporaryDirectory() as tmp:
        for name in SCENARIOS:
            work = Path(tmp) / name
            work.mkdir()
            subprocess.run([sys.executable, __file__, "--scenario", name,
                            "--out", str(args.out), "--work", str(work)],
                           check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
