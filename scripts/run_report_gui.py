"""Qt launcher for the TXM characterization report.

Presents a small dialog that collects the operator's tester name,
the three serial numbers (TX, console, hydrophone), and a frequency
choice. Clicking Start runs the same pipeline as :mod:`run_report`
on a background thread while a Running dialog shows a live progress
bar, the current phase label, and a mini terminal that mirrors the
INFO log stream. The Running dialog's Abort button stops the run
after the step in progress and returns to the launcher.

When the run completes, the result dialog offers "Write Data to
Device" (PASS only), "Scan Another" (back to the launcher), and
"Exit". The hardware connection stays open until one of the latter
two is chosen so the write can happen from the result dialog.

The launcher deliberately reuses ``run_report.run()`` in-process so
the two entry points share every behavior (arg parsing, artifact
layout, device-write pass-gate, etc.).

Example::

    python scripts/run_report_gui.py               # real hardware
    python scripts/run_report_gui.py --dry-run     # smoke test
    python scripts/run_report_gui.py --hydrophone 2246 --frequency-khz 400
    python scripts/run_report_gui.py --theme Dark  # dark mode
"""
from __future__ import annotations

import argparse
import html
import logging
import queue
import re
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from pathlib import Path
from typing import Optional

# Vendored subset of the CLI arg parser. Import first so a --help
# invocation on a machine without PyQt6 still works.
import run_report

from PyQt6.QtCore import Qt, QObject, QSettings, QThread, QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices, QFont, QFontMetrics, QTextCursor
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from openlifu_verification import (
    HardwareNotConnectedError,
    Hydrophone,
    OperatorPrefs,
    ScanConfig,
    VerificationTank,
)
from openlifu_verification.acceptance import SUPPORTED_FREQUENCIES_KHZ
from openlifu_verification.report_io import NONSTANDARD_BANNER


logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Theme system
# ----------------------------------------------------------------------
# Two hand-tuned palettes (Light / Dark) are applied application-wide
# via ``QApplication.setStyleSheet``. Widgets that need state-specific
# colors (e.g. the gate glyph) advertise a dynamic ``state`` property
# that the QSS keys on, so a theme swap re-colors everything without
# any per-widget bookkeeping.

@dataclass(frozen=True)
class Palette:
    """Named color slots consumed by :func:`_build_qss` and by widgets
    that build rich-text (``ResultDialog``) at construction time.

    Keeping colors here (as opposed to buried inline) means adding a
    third theme is just another :class:`Palette` instance."""
    name: str
    window_bg: str        # dialog / app background
    panel_bg: str         # inputs, group boxes, terminal
    panel_alt_bg: str     # subtle stripe (progress trough, hover)
    text: str             # primary foreground
    subtle_text: str      # form-label style hint text
    muted: str            # PENDING glyph, SKIP glyph
    border: str           # 1px separators / input outline
    accent: str           # CHECKING glyph, focus ring, progress fill
    accent_hover: str     # button hover
    success: str          # PASS glyph, verdict PASS
    danger: str           # FAIL glyph, verdict FAIL
    warning: str          # (reserved) yellow highlights


_LIGHT = Palette(
    name="Light",
    window_bg="#f4f6f9",
    panel_bg="#ffffff",
    panel_alt_bg="#eef1f5",
    text="#1c2126",
    subtle_text="#4d5865",
    muted="#8a939e",
    border="#d5dbe3",
    accent="#1f6feb",
    accent_hover="#155bcb",
    success="#1a7a1a",
    danger="#b32020",
    warning="#c68a00",
)

_DARK = Palette(
    name="Dark",
    window_bg="#1e2229",
    panel_bg="#262b33",
    panel_alt_bg="#2f353e",
    text="#e6edf3",
    subtle_text="#c0c8d1",
    muted="#8b95a1",
    border="#3a4149",
    accent="#58a6ff",
    accent_hover="#7cb9ff",
    success="#4caf50",
    danger="#ff6b6b",
    warning="#e0b341",
)

_PALETTES: dict[str, Palette] = {p.name: p for p in (_LIGHT, _DARK)}

# Module-level "currently applied" palette so widgets that build
# rich-text at construction time (verdict label, write-line) can
# read the right colors without needing a back-reference to the
# QApplication.
_current_palette: Palette = _LIGHT


def current_palette() -> Palette:
    """Return the palette in force. Widgets that need to embed a
    color into rich text (e.g. HTML ``<span style='color:...'>``)
    read from here at build time."""
    return _current_palette


def _build_qss(p: Palette) -> str:
    """Assemble the QSS stylesheet for palette ``p``.

    Kept as a plain string with ``{p.field}`` interpolation for
    readability. Selectors intentionally target base widget types
    (rather than every subclass) so downstream custom widgets
    inherit the look automatically."""
    return f"""
    QWidget {{
        background-color: {p.window_bg};
        color: {p.text};
        font-size: 10pt;
    }}
    QDialog {{
        background-color: {p.window_bg};
    }}
    QLabel {{
        background: transparent;
        color: {p.text};
    }}
    QLabel[role="subtle"] {{
        color: {p.subtle_text};
    }}
    QLabel[role="section"] {{
        color: {p.subtle_text};
        font-weight: bold;
        padding-top: 2px;
    }}
    QLabel[role="alert"] {{
        color: {p.danger};
        font-weight: bold;
    }}

    /* --- Group boxes --- */
    QGroupBox {{
        background-color: {p.panel_bg};
        border: 1px solid {p.border};
        border-radius: 6px;
        margin-top: 14px;
        padding: 12px 10px 10px 10px;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 6px;
        color: {p.subtle_text};
        font-weight: bold;
    }}

    /* --- Inputs --- */
    QLineEdit, QComboBox, QPlainTextEdit {{
        background-color: {p.panel_bg};
        color: {p.text};
        border: 1px solid {p.border};
        border-radius: 4px;
        padding: 4px 6px;
        selection-background-color: {p.accent};
        selection-color: {p.panel_bg};
    }}
    QLineEdit:focus, QComboBox:focus, QPlainTextEdit:focus {{
        border: 1px solid {p.accent};
    }}
    QLineEdit:disabled, QComboBox:disabled {{
        color: {p.muted};
        background-color: {p.panel_alt_bg};
    }}
    QComboBox::drop-down {{
        border: none;
        width: 20px;
    }}
    QComboBox QAbstractItemView {{
        background-color: {p.panel_bg};
        color: {p.text};
        border: 1px solid {p.border};
        selection-background-color: {p.accent};
        selection-color: {p.panel_bg};
        outline: 0;
    }}

    /* --- Checkbox --- */
    QCheckBox {{
        spacing: 8px;
        color: {p.text};
    }}
    QCheckBox::indicator {{
        width: 16px;
        height: 16px;
        border: 1px solid {p.border};
        border-radius: 3px;
        background-color: {p.panel_bg};
    }}
    QCheckBox::indicator:hover {{
        border: 1px solid {p.accent};
    }}
    QCheckBox::indicator:checked {{
        background-color: {p.accent};
        border: 1px solid {p.accent};
        image: none;
    }}

    /* --- Buttons --- */
    QPushButton {{
        background-color: {p.panel_bg};
        color: {p.text};
        border: 1px solid {p.border};
        border-radius: 4px;
        padding: 6px 16px;
        min-width: 72px;
    }}
    QPushButton:hover {{
        border: 1px solid {p.accent};
    }}
    QPushButton:pressed {{
        background-color: {p.panel_alt_bg};
    }}
    QPushButton:default {{
        background-color: {p.accent};
        color: {p.panel_bg};
        border: 1px solid {p.accent};
        font-weight: bold;
    }}
    QPushButton:default:hover {{
        background-color: {p.accent_hover};
        border: 1px solid {p.accent_hover};
    }}
    QPushButton:disabled {{
        color: {p.muted};
        background-color: {p.panel_alt_bg};
        border: 1px solid {p.border};
    }}

    /* --- Progress bar --- */
    QProgressBar {{
        background-color: {p.panel_alt_bg};
        color: {p.text};
        border: 1px solid {p.border};
        border-radius: 4px;
        text-align: center;
        min-height: 20px;
    }}
    QProgressBar::chunk {{
        background-color: {p.accent};
        border-radius: 3px;
    }}

    /* --- Terminal (mini log) --- */
    QPlainTextEdit#miniTerminal {{
        background-color: {p.panel_alt_bg};
        color: {p.text};
        border: 1px solid {p.border};
        border-radius: 4px;
    }}

    /* --- Frame separator --- */
    QFrame#hLine {{
        color: {p.border};
        background-color: {p.border};
        max-height: 1px;
    }}

    /* --- Gate glyph (state-driven color) --- */
    QLabel#gateGlyph {{
        color: {p.muted};
        font-weight: bold;
    }}
    QLabel#gateGlyph[state="pending"] {{
        color: {p.muted};
    }}
    QLabel#gateGlyph[state="checking"] {{
        color: {p.accent};
    }}
    QLabel#gateGlyph[state="pass"] {{
        color: {p.success};
    }}
    QLabel#gateGlyph[state="fail"] {{
        color: {p.danger};
    }}
    QLabel#gateGlyph[state="skip"] {{
        color: {p.muted};
    }}
    QLabel#gateLabel[state="pass"] {{
        color: {p.text};
    }}
    QLabel#gateLabel[state="fail"] {{
        color: {p.danger};
    }}
    QLabel#gateLabel[state="skip"] {{
        color: {p.muted};
    }}

    /* --- Verdict header (ResultDialog) --- */
    QLabel#verdictBanner {{
        border-radius: 6px;
        padding: 10px 14px;
        font-size: 14pt;
        font-weight: bold;
    }}
    QLabel#verdictBanner[verdict="pass"] {{
        background-color: {p.panel_alt_bg};
        color: {p.success};
        border: 1px solid {p.success};
    }}
    QLabel#verdictBanner[verdict="fail"] {{
        background-color: {p.panel_alt_bg};
        color: {p.danger};
        border: 1px solid {p.danger};
    }}
    """


def _refresh_polish(widget: QWidget) -> None:
    """Force Qt to re-evaluate QSS on ``widget`` after a dynamic
    property change. Called from state-transition code."""
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
    widget.update()


def apply_theme(app: QApplication, name: str) -> None:
    """Apply the palette named ``name`` to ``app`` (and to every
    top-level widget already realised).

    Unknown names fall back to Light with a warning so a stale
    CLI value can't hard-crash the launcher."""
    global _current_palette
    if name not in _PALETTES:
        logger.warning("Unknown theme %r; falling back to Light.", name)
        name = "Light"
    _current_palette = _PALETTES[name]
    app.setStyleSheet(_build_qss(_current_palette))
    # Any widgets that use dynamic-property selectors need a repolish
    # to pick up the new stylesheet immediately.
    for w in app.allWidgets():
        _refresh_polish(w)


# List of valid theme names for argparse ``choices=``.
THEME_NAMES: tuple[str, ...] = tuple(_PALETTES.keys())


# ----------------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------------
def build_gui_parser() -> argparse.ArgumentParser:
    """CLI parser for :mod:`run_report_gui`.

    Reuses every option from :mod:`run_report` so behavioral flags
    like ``--dry-run``, ``--frequency-khz``, ``--skip-voltage``,
    ``--hydrophone`` etc. keep working. The prefs prompt is always
    suppressed (the dialog collects the values instead) so we force
    ``--no-prompt`` and ``--no-start-prompt`` at run time regardless
    of what the caller passed.
    """
    parent = run_report.build_parser()
    parser = argparse.ArgumentParser(
        description="Qt launcher for the TXM characterization report.",
        parents=[parent],
        conflict_handler="resolve",
    )
    parser.add_argument(
        "--theme",
        choices=THEME_NAMES,
        default="Light",
        help=(
            "GUI color theme. One of: "
            + ", ".join(THEME_NAMES)
            + " (default: Light)."
        ),
    )
    return parser


# ----------------------------------------------------------------------
# Progress phase model
# ----------------------------------------------------------------------
# Each phase starts when its marker substring appears in an INFO log
# record and has an expected duration in seconds. The RunningDialog
# advances the bar through each phase's share of the total and
# estimates remaining time as the rest of the current phase plus every
# later phase.
#
# Coefficients fitted to three 155 kHz runs (standard, half points,
# single average) on 2026-10-15; the model reproduced each total within
# ~2%. HV settle (set_voltage + wait) is linear in the target voltage,
# not the step size. Since configure_lifu stopped re-sending the HV
# setpoint on every reconfigure, only real voltage changes pay it; a
# reconfigure alone is register writes (measured 2.3 s).
_HV_SETTLE_S = 2.35
_HV_SETTLE_S_PER_V = 0.0894
_CONFIGURE_S = 2.3
_SCAN_POINT_S = 0.22          # stage move per 1-D / 2-D point
_CAPTURE_PER_AVG_S = 0.108    # one trigger + capture per average
_FREQ_POINT_S = 0.384         # set_pulse + armed train per frequency
_FREQ_PER_AVG_S = 0.021       # extra pulse in the same train
_WAVEFORM_PER_AVG_S = 0.18


@dataclass(frozen=True)
class _Phase:
    marker: Optional[str]  # None: starts with the dialog
    label: str
    seconds: float


def _hv_settle_s(voltage_V: float) -> float:
    return _HV_SETTLE_S + _HV_SETTLE_S_PER_V * float(voltage_V)


def _estimate_phases(scan_config: ScanConfig, frequency_kHz: float,
                     voltage_V: float, *, skip_2d: bool = False,
                     skip_frequency: bool = False,
                     skip_voltage: bool = False) -> list[_Phase]:
    """Expected phases of one run, in log order."""
    plan = scan_config.plan_for(frequency_kHz)
    n_avg = int(scan_config.scope.n_averages)
    n_pnp = int(scan_config.scope.pnp_n_averages)
    configure = _CONFIGURE_S
    point = _SCAN_POINT_S + _CAPTURE_PER_AVG_S * n_avg

    phases = [
        _Phase(None, "Connecting to hardware", 1.0),
        # First pulse setup + HV on from 0 V.
        _Phase("LIFU Device fully connected", "Configuring transducer",
               configure + _hv_settle_s(voltage_V) + 0.7),
        _Phase("Plane-wave depth calibration:",
               "Plane-wave depth calibration", 2 * configure + 6.8),
        _Phase("Running fresh find_peak", "Peak search", 7.0),
    ]
    if not skip_2d:
        phases += [
            _Phase("Running 1-D lateral", "1-D lateral scan",
                   0.3 + plan.lateral_1d.points * point),
            _Phase("Running 1-D elevation", "1-D elevation scan",
                   0.3 + plan.elevation_1d.points * point),
            _Phase("Running 1-D axial", "1-D axial scan",
                   0.3 + plan.axial_1d.points * point),
            _Phase("Running 2-D scan", "2-D scan",
                   0.3 + plan.scan_2d.points ** 2 * point),
        ]
    pnp_voltage_V = scan_config.acceptance.pnp_at_peak.reference_voltage_V
    phases.append(_Phase(
        "Measuring waveform at peak", "Waveform at peak",
        7.55 + _WAVEFORM_PER_AVG_S * n_pnp + _hv_settle_s(pnp_voltage_V)
        + _hv_settle_s(voltage_V) + 2 * configure,
    ))
    if not skip_frequency:
        n_freq = len(plan.frequency_sweep.offsets_kHz)
        phases.append(_Phase(
            "Frequency sweep across", "Frequency sweep",
            0.2 + 2 * configure
            + n_freq * (_FREQ_POINT_S + _FREQ_PER_AVG_S * n_pnp),
        ))
    if not skip_voltage:
        volts = plan.voltage_sweep.voltages_V
        phases.append(_Phase(
            "Voltage sweep across", "Voltage sweep",
            0.4 + sum(_hv_settle_s(v) + _CAPTURE_PER_AVG_S * n_avg
                      for v in volts),
        ))
    phases += [
        _Phase("Overall verdict:", "Writing report", 2.2),
        _Phase("Run directory:", "Complete", 0.0),
    ]
    return phases


# ----------------------------------------------------------------------
# Log bridge
# ----------------------------------------------------------------------
class _LogBridge(QObject):
    """Marshals ``logging.Handler`` records onto the Qt event loop.

    :class:`_QtLogHandler` emits on whatever thread produced the log
    record (usually the worker QThread); the ``message`` signal is
    delivered via a queued connection so the connected slots always
    run on the GUI thread.
    """

    # (formatted_line, raw_message)
    message = pyqtSignal(str, str)


class _QtLogHandler(logging.Handler):
    """Standard ``logging`` handler that pushes records through a
    :class:`_LogBridge` signal.

    Kept intentionally minimal: it does the formatting (so the mini
    terminal doesn't need to know about ``LogRecord`` internals) and
    forwards the *raw* message alongside so the progress-phase table
    can key off the unadorned text.
    """

    def __init__(self, bridge: _LogBridge, level: int = logging.INFO):
        super().__init__(level)
        self._bridge = bridge
        self.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(message)s",
            datefmt="%H:%M:%S",
        ))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._bridge.message.emit(
                self.format(record), record.getMessage()
            )
        except Exception:  # noqa: BLE001 - never let logging crash the app
            self.handleError(record)


# ----------------------------------------------------------------------
# Background worker
# ----------------------------------------------------------------------
class _RunSignals(QObject):
    """Qt signal bundle so the worker can post progress + result back
    to the GUI thread without touching widgets directly."""

    finished = pyqtSignal(object, bool)  # (RunResult, can_write)
    failed = pyqtSignal(str)       # emits traceback string
    #: Emitted when the run stopped because the operator pressed Abort.
    aborted = pyqtSignal()
    #: Emitted when the run stopped because depth calibration or the
    #: peak search failed. Payload is the operator-facing reason.
    halted = pyqtSignal(str)
    #: Emitted after a device write requested from the result dialog.
    write_done = pyqtSignal(bool)
    #: Emitted when the run aborted because a required piece of
    #: hardware is disconnected. Payload is a short, user-facing
    #: message (no traceback).
    hardware_disconnected = pyqtSignal(str, str)  # (hardware, message)


class _WorkerThread(QThread):
    """Executes :func:`run_report.run` on a background QThread.

    After the report is written the worker parks inside
    ``run_report.run``'s ``on_complete`` hook (hardware still
    connected, HV off) and services commands from the GUI:
    :meth:`request_write` pushes the device config, :meth:`release`
    lets the run finish and disconnect.
    """

    def __init__(self, args: argparse.Namespace,
                 signals: _RunSignals, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._args = args
        self._signals = signals
        self._abort_event = threading.Event()
        self._commands: queue.Queue[str] = queue.Queue()
        self._completed = False

    # --- Called from the GUI thread ---
    def request_abort(self) -> None:
        self._abort_event.set()

    def request_write(self) -> None:
        self._commands.put("write")

    def release(self) -> None:
        self._commands.put("close")

    # --- Worker thread ---
    def _on_complete(self, result, write_fn) -> None:
        self._completed = True
        self._signals.finished.emit(result, write_fn is not None)
        while True:
            cmd = self._commands.get()
            if cmd == "close":
                return
            if cmd == "write" and write_fn is not None:
                try:
                    ok = bool(write_fn())
                except Exception:  # noqa: BLE001 - report to GUI
                    logger.exception("Device write failed")
                    ok = False
                self._signals.write_done.emit(ok)

    def run(self) -> None:  # noqa: D401 - QThread entry point
        try:
            result = run_report.run(
                self._args,
                abort_event=self._abort_event,
                on_complete=self._on_complete,
            )
            if result.aborted:
                self._signals.aborted.emit()
            elif result.halted_reason:
                self._signals.halted.emit(result.halted_reason)
            elif not self._completed:
                self._signals.finished.emit(result, False)
        except HardwareNotConnectedError as e:
            logger.error("Hardware not connected (%s): %s", e.hardware, e)
            self._signals.hardware_disconnected.emit(str(e.hardware), str(e))
        except Exception:  # noqa: BLE001 - report + swallow to GUI
            tb = traceback.format_exc()
            logger.error("Run failed:\n%s", tb)
            self._signals.failed.emit(tb)


# ----------------------------------------------------------------------
# Launcher dialog
# ----------------------------------------------------------------------
# Restrict the frequency dropdown to the two nominal drive
# frequencies supported by the acceptance criteria. Both are the same
# values the CLI's --frequency-khz flag accepts.
_ALLOWED_FREQUENCIES_KHZ: tuple[float, ...] = SUPPORTED_FREQUENCIES_KHZ


class LauncherDialog(QDialog):
    """Modal form that collects operator inputs and kicks off a run.

    The Start button is only enabled when every required text field
    is non-empty. On accept, the dialog persists an ``OperatorPrefs``
    snapshot (so ``run_report.run`` picks up the values through its
    normal load path) and the caller reads
    :attr:`selected_frequency_kHz` to fold into the arg namespace
    before starting the worker.
    """

    def __init__(self, args: argparse.Namespace,
                 prefs: OperatorPrefs,
                 parent: Optional[QWidget] = None, *,
                 nonstandard_reasons: tuple[str, ...] | list[str] = ()):
        super().__init__(parent)
        title = "OpenLIFU Calibration Tool"
        if getattr(args, "dry_run", False):
            title += "  [DRY RUN]"
        if nonstandard_reasons:
            title += "  [NON-STANDARD]"
        self.setWindowTitle(title)
        self.setModal(True)
        self._args = args
        self._prefs = prefs

        # --- Header banner ---
        header = QLabel("OpenLIFU Calibration")
        hfont = QFont()
        hfont.setPointSize(hfont.pointSize() + 6)
        hfont.setBold(True)
        header.setFont(hfont)
        subtitle = QLabel(
            "Configure the run below, then press "
            "<b>Start</b> to begin characterization."
        )
        subtitle.setTextFormat(Qt.TextFormat.RichText)
        subtitle.setProperty("role", "subtle")
        subtitle.setWordWrap(True)

        nonstandard_label = None
        if nonstandard_reasons:
            nonstandard_label = QLabel(
                f"{NONSTANDARD_BANNER}. Calibration data cannot be "
                "written to the device."
            )
            nonstandard_label.setProperty("role", "alert")
            nonstandard_label.setWordWrap(True)
            nonstandard_label.setToolTip("\n".join(nonstandard_reasons))

        # --- Operator group ---
        self.tester_edit = QLineEdit(prefs.tester_name)
        self.tester_edit.setPlaceholderText("e.g. Jane Doe")
        self.txm_edit = QLineEdit(prefs.txm_sn)
        self.txm_edit.setPlaceholderText("TXM serial number")
        self.console_edit = QLineEdit(prefs.console_sn)
        self.console_edit.setPlaceholderText("Console serial number")
        self.hydrophone_edit = QLineEdit(prefs.hydrophone_sn)
        self.hydrophone_edit.setPlaceholderText("Hydrophone serial number")

        op_group = QGroupBox("Operator && device")
        op_form = QFormLayout(op_group)
        op_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        op_form.setFormAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        op_form.setHorizontalSpacing(12)
        op_form.setVerticalSpacing(8)
        op_form.addRow("Tester:", self.tester_edit)
        op_form.addRow("Assigned TXM S/N:", self.txm_edit)
        op_form.addRow("Test Console S/N:", self.console_edit)
        op_form.addRow("Hydrophone S/N:", self.hydrophone_edit)

        # --- Run configuration group ---
        # Frequency dropdown seeded from the CLI's --frequency-khz.
        # Only nominal drive frequencies (155 / 400 kHz) are offered
        # because the acceptance thresholds are only defined there.
        self.freq_combo = QComboBox()
        for f in _ALLOWED_FREQUENCIES_KHZ:
            self.freq_combo.addItem(f"{f:g} kHz", userData=float(f))
        idx = self.freq_combo.findData(float(args.frequency_khz))
        if idx >= 0:
            self.freq_combo.setCurrentIndex(idx)

        run_group = QGroupBox("Run configuration")
        run_form = QFormLayout(run_group)
        run_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        run_form.setHorizontalSpacing(12)
        run_form.setVerticalSpacing(8)
        run_form.addRow("Frequency:", self.freq_combo)

        # Reports folder: clickable path (opens Explorer) + a "…"
        # button to change it. Runs are written to
        # <folder>/<TXM S/N>/<date_time>/.
        self.reports_link = QLabel()
        self.reports_link.setTextFormat(Qt.TextFormat.RichText)
        self.reports_link.setTextInteractionFlags(
            Qt.TextInteractionFlag.LinksAccessibleByMouse
        )
        self.reports_link.linkActivated.connect(self._open_reports_root)
        change_button = QPushButton("\u2026")
        change_button.setToolTip("Change Reports Directory")
        change_button.setAutoDefault(False)
        change_button.setStyleSheet("min-width: 0px; padding: 2px 10px;")
        change_button.clicked.connect(self._browse_reports_root)
        reports_row = QHBoxLayout()
        reports_row.setSpacing(8)
        reports_row.addWidget(self.reports_link, 1)
        reports_row.addWidget(change_button)
        run_form.addRow("Reports saved to:", reports_row)
        self._set_reports_root(Path(args.reports_root))

        # --- Buttons ---
        self.start_button = QPushButton("Start")
        self.start_button.setDefault(True)
        exit_button = QPushButton("Exit")
        button_box = QDialogButtonBox(Qt.Orientation.Horizontal)
        button_box.addButton(self.start_button,
                             QDialogButtonBox.ButtonRole.AcceptRole)
        button_box.addButton(exit_button,
                             QDialogButtonBox.ButtonRole.RejectRole)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        # --- Assemble ---
        sep = QFrame()
        sep.setObjectName("hLine")
        sep.setFrameShape(QFrame.Shape.HLine)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(18, 16, 18, 14)
        outer.setSpacing(12)
        outer.addWidget(header)
        outer.addWidget(subtitle)
        if nonstandard_label is not None:
            outer.addWidget(nonstandard_label)
        outer.addWidget(sep)
        outer.addWidget(op_group)
        outer.addWidget(run_group)
        outer.addStretch(1)
        outer.addWidget(button_box)

        # Wire up validation last so the initial state is applied.
        for edit in (self.tester_edit, self.txm_edit,
                     self.console_edit, self.hydrophone_edit):
            edit.textChanged.connect(self._refresh_start_enabled)
        self._refresh_start_enabled()

        self.setMinimumWidth(460)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _refresh_start_enabled(self) -> None:
        """Enable Start only when every text field is non-blank."""
        ok = all(edit.text().strip() for edit in (
            self.tester_edit, self.txm_edit,
            self.console_edit, self.hydrophone_edit,
        ))
        self.start_button.setEnabled(ok)

    # ------------------------------------------------------------------
    # Reports folder
    # ------------------------------------------------------------------
    def _set_reports_root(self, root: Path) -> None:
        """Show ``root`` as a link, middle-elided to fit the form."""
        self._reports_root = root
        shown = QFontMetrics(self.reports_link.font()).elidedText(
            str(root), Qt.TextElideMode.ElideMiddle, 300
        )
        self.reports_link.setText(
            f'<a href="open" style="color:{current_palette().accent};'
            f' text-decoration:none;">{html.escape(shown)}</a>'
        )
        self.reports_link.setToolTip(f"{root}\nClick to open in Explorer")

    def _browse_reports_root(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Change Reports Directory", str(self._reports_root)
        )
        if path:
            self._set_reports_root(Path(path))

    def _open_reports_root(self) -> None:
        root = self._reports_root
        try:
            root.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            QMessageBox.warning(self, "Reports folder",
                                f"Could not create {root}:\n{e}")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(root)))

    # ------------------------------------------------------------------
    # Accessors used after ``exec()`` returns Accepted.
    # ------------------------------------------------------------------
    @property
    def collected_prefs(self) -> OperatorPrefs:
        """Return an ``OperatorPrefs`` populated from the form."""
        p = OperatorPrefs(
            tester_name=self.tester_edit.text().strip(),
            test_app_version=self._prefs.test_app_version,
            hydrophone_sn=self.hydrophone_edit.text().strip(),
            txm_sn=self.txm_edit.text().strip(),
            console_sn=self.console_edit.text().strip(),
        )
        return p

    @property
    def selected_frequency_kHz(self) -> float:
        """Return the drive frequency selected in the combo box."""
        data = self.freq_combo.currentData()
        return float(data) if data is not None else float(
            self._args.frequency_khz
        )

    @property
    def selected_reports_root(self) -> Path:
        """Return the reports folder shown in the form."""
        return self._reports_root


# ----------------------------------------------------------------------
# Running dialog
# ----------------------------------------------------------------------
# Strip ANSI colour codes (openlifu-sdk sometimes emits them via
# console formatters) before appending to the mini terminal.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[mK]")


# The acceptance gates the tracker surfaces. Each entry describes:
#
# * ``key``          - stable id, must match a summary key produced
#                      by ``Characterization.grade()``.
# * ``label``        - human-readable name shown in the tracker.
# * ``start_marker`` - substring in a log message that means the
#                      underlying measurement / evaluation has begun
#                      (moves the gate to CHECKING). Use ``None`` for
#                      instantaneous static checks (firmware / cal
#                      dates) that jump straight from PENDING to
#                      PASS/FAIL when their verdict line arrives.
# * ``verdict_res``  - tuple of regexes whose match yields a
#                      ``"PASS"``/``"FAIL"`` capture group. Applied
#                      to the log message; the first hit stamps the
#                      gate.
# * ``combine``      - ``"first"`` (single verdict) or ``"all"``
#                      (wait for every regex in ``verdict_res`` and
#                      AND them).
#
# Order here is the display order in the tracker. It roughly mirrors
# the wall-clock order the gates resolve during a real run so the
# operator sees them light up top-to-bottom.
#
# NB: the ``start_marker`` values were chosen to fire well BEFORE
# the corresponding grade line, so the animated CHECKING glyph has
# time to actually spin. Using a marker whose log record is emitted
# right next to the verdict yields a "jump-to-end" appearance.
_GATE_VERDICT_RE_TEMPLATE = (
    r"\[grade\]\s+{name}\s+(?:->|\u2192)\s+(PASS|FAIL|SKIP)"
)


def _grade_re(name: str) -> re.Pattern:
    return re.compile(_GATE_VERDICT_RE_TEMPLATE.format(name=re.escape(name)))


_GATES: list[dict] = [
    # --- Static checks (graded during collect_test_info) ---
    {
        "key": "firmware",
        "label": "Firmware versions",
        "start_marker": "Checking firmware + calibration",
        "verdict_res": (_grade_re("TXM firmware"),
                        _grade_re("Console firmware")),
        "combine": "all",
    },
    {
        "key": "calibration_dates",
        "label": "Calibration dates",
        "start_marker": "Checking firmware + calibration",
        "verdict_res": (_grade_re("Hydrophone cal date"),
                        _grade_re("PicoScope cal date")),
        "combine": "all",
    },
    # --- Plane-wave depth calibration (D.3) ---
    {
        "key": "hydrophone_depth",
        "label": "Hydrophone depth (D.3)",
        "start_marker": "Plane-wave depth calibration:",
        "verdict_res": (_grade_re("Hydrophone depth"),),
        "combine": "first",
    },
    # --- Peak search + 2-D scan (D.5, D.6, offset) ---
    {
        "key": "peak_xy",
        "label": "Peak X/Y position (D.5, D.6)",
        # find_peak begins with "Running fresh find_peak from..."
        # which fires well before ``_grade_peak_xy`` stamps its
        # verdict, so the spinner has time to actually animate.
        "start_marker": "Running fresh find_peak",
        "verdict_res": (_grade_re("Peak X"), _grade_re("Peak Y")),
        "combine": "all",
    },
    {
        "key": "peak_offset",
        "label": "Peak offset from centre",
        "start_marker": "Running fresh find_peak",
        "verdict_res": (_grade_re("Peak offset"),),
        "combine": "first",
    },
    # --- 1-D lateral / elevation peak offset (D.8, D.10) ---
    {
        "key": "lateral_peak_offset",
        "label": "Lateral 1-D peak offset (D.8)",
        "start_marker": "Running 1-D lateral",
        "verdict_res": (_grade_re("Lateral peak offset"),),
        "combine": "first",
    },
    {
        "key": "elevation_peak_offset",
        "label": "Elevation 1-D peak offset (D.10)",
        "start_marker": "Running 1-D elevation",
        "verdict_res": (_grade_re("Elevation peak offset"),),
        "combine": "first",
    },
    # --- Focused-pulse measurements (D.12, D.14, D.16) ---
    {
        "key": "peak_z_focus",
        "label": "Peak Z focus vs D.3 (D.12)",
        # measure_waveform_at_peak runs after the 1-D axial + 2-D
        # scans, so "Running 2-D scan" gives the spinner ample time.
        "start_marker": "Running 2-D scan",
        "verdict_res": (_grade_re("Peak Z focus"),),
        "combine": "first",
    },
    {
        "key": "pnp_at_peak",
        "label": "Focused pulse PNP (D.14)",
        "start_marker": "Running 2-D scan",
        "verdict_res": (_grade_re("PNP at peak"),),
        "combine": "first",
    },
    {
        "key": "peak_depth",
        "label": "Focused hydrophone depth (D.16)",
        "start_marker": "Running 2-D scan",
        "verdict_res": (_grade_re("Peak depth"),),
        "combine": "first",
    },
    # --- Sweeps (E, F) ---
    {
        "key": "freq_response",
        "label": "Frequency response (E)",
        "start_marker": "Frequency sweep across",
        "verdict_res": (_grade_re("Frequency response"),),
        "combine": "first",
    },
    {
        "key": "voltage_linearity",
        "label": "Voltage linearity (F.8)",
        "start_marker": "Voltage sweep across",
        "verdict_res": (_grade_re("Voltage linearity"),),
        "combine": "first",
    },
]


class _GateRow(QWidget):
    """Single acceptance-gate row: status glyph + label.

    The glyph is a :class:`QLabel` whose text is one of:

    * ``\u25cb``  empty circle   - PENDING (grey)
    * ``\u25d0``/animated dots   - CHECKING (blue, ``QTimer``-driven)
    * ``\u25cf``                 - PASS (green) or FAIL (red)
    * ``\u2013``  en-dash        - SKIP (grey, for disabled checks)

    Using text glyphs keeps the widget rendering-independent (no
    QSS assets required).
    """

    PENDING = "pending"
    CHECKING = "checking"
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"

    _CHECK_FRAMES = ("\u25d0", "\u25d3", "\u25d1", "\u25d2")

    def __init__(self, label: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._state = self.PENDING
        self._frame = 0

        # Glyph and label carry a dynamic ``state`` property that
        # the app-wide QSS keys on; swapping themes therefore
        # re-colors every gate automatically.
        self.glyph = QLabel("\u25cb")
        self.glyph.setObjectName("gateGlyph")
        self.glyph.setProperty("state", self.PENDING)
        gfont = QFont()
        gfont.setPointSize(gfont.pointSize() + 3)
        self.glyph.setFont(gfont)
        self.glyph.setFixedWidth(22)
        self.glyph.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.text = QLabel(label)
        self.text.setObjectName("gateLabel")
        self.text.setProperty("state", self.PENDING)

        row = QHBoxLayout(self)
        row.setContentsMargins(4, 2, 4, 2)
        row.setSpacing(8)
        row.addWidget(self.glyph)
        row.addWidget(self.text, 1)

        # Timer that animates the CHECKING glyph. Owned per row so
        # rows can independently sit in CHECKING; only started when
        # the row enters the CHECKING state.
        from PyQt6.QtCore import QTimer
        self._timer = QTimer(self)
        self._timer.setInterval(180)
        self._timer.timeout.connect(self._tick)

    def _tick(self) -> None:
        self._frame = (self._frame + 1) % len(self._CHECK_FRAMES)
        self.glyph.setText(self._CHECK_FRAMES[self._frame])

    def set_state(self, state: str) -> None:
        """Transition to ``state``. Illegal transitions (e.g.
        PASS -> PENDING) are ignored so a late-arriving log line
        can't undo a verdict."""
        rank = {self.PENDING: 0, self.CHECKING: 1,
                self.PASS: 2, self.FAIL: 2, self.SKIP: 2}
        if rank[state] < rank[self._state]:
            return
        self._state = state
        # Update the glyph character (color comes from QSS).
        if state == self.PENDING:
            self._timer.stop()
            self.glyph.setText("\u25cb")
        elif state == self.CHECKING:
            self._frame = 0
            self.glyph.setText(self._CHECK_FRAMES[0])
            self._timer.start()
        elif state in (self.PASS, self.FAIL):
            self._timer.stop()
            self.glyph.setText("\u25cf")
        elif state == self.SKIP:
            self._timer.stop()
            self.glyph.setText("\u2013")
        # Push the new state into the dynamic property and re-polish
        # so the QSS attribute selector picks up the change.
        self.glyph.setProperty("state", state)
        self.text.setProperty("state", state)
        _refresh_polish(self.glyph)
        _refresh_polish(self.text)

    @property
    def state(self) -> str:
        return self._state


class AcceptanceTracker(QWidget):
    """Column of :class:`_GateRow` widgets driven by log messages."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._rows: dict[str, _GateRow] = {}
        # Whether a per-gate verdict has been stamped. Prevents a
        # later log echo (e.g. a re-run) from flipping the state.
        self._verdict_set: dict[str, bool] = {}
        # For "combine=all" gates (peak_xy), collect verdicts as
        # they arrive and stamp the overall state only once every
        # sub-verdict is in.
        self._pending_verdicts: dict[str, list[bool]] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)
        for gate in _GATES:
            row = _GateRow(gate["label"])
            self._rows[gate["key"]] = row
            self._verdict_set[gate["key"]] = False
            self._pending_verdicts[gate["key"]] = []
            layout.addWidget(row)

    def on_log(self, raw: str) -> None:
        """Advance gate states based on ``raw`` log message."""
        for gate in _GATES:
            key = gate["key"]
            row = self._rows[key]
            # Start marker moves PENDING -> CHECKING (unless the gate
            # is already resolved). ``None`` opts out entirely.
            marker = gate.get("start_marker")
            if (marker is not None
                    and not self._verdict_set[key]
                    and row.state == _GateRow.PENDING
                    and marker in raw):
                row.set_state(_GateRow.CHECKING)
            # Verdict regex(es) stamp PASS / FAIL / SKIP.
            if self._verdict_set[key]:
                continue
            for pat in gate["verdict_res"]:
                m = pat.search(raw)
                if not m:
                    continue
                verdict = m.group(1)
                # SKIP counts as a "passing" outcome so a gate whose
                # sub-checks are all disabled still resolves (grey
                # en-dash). If we're in a ``combine="all"`` gate we
                # let a single SKIP contribute a True; a subsequent
                # sub-check that FAILs will still drive the gate to
                # FAIL because ``all([True, False]) == False``.
                as_bool = verdict != "FAIL"
                if gate["combine"] == "first":
                    if verdict == "SKIP":
                        row.set_state(_GateRow.SKIP)
                    else:
                        row.set_state(
                            _GateRow.PASS if as_bool else _GateRow.FAIL
                        )
                    self._verdict_set[key] = True
                else:  # "all"
                    self._pending_verdicts[key].append(as_bool)
                    n_expected = len(gate["verdict_res"])
                    if len(self._pending_verdicts[key]) >= n_expected:
                        combined = all(self._pending_verdicts[key])
                        row.set_state(
                            _GateRow.PASS if combined else _GateRow.FAIL
                        )
                        self._verdict_set[key] = True
                # Only match one pattern per record.
                break


class RunningDialog(QDialog):
    """Modal progress dialog shown while the pipeline runs.

    The dialog can't be dismissed by the operator (Esc / close are
    ignored); the Abort button asks the worker to stop after the step
    in progress. The caller closes the dialog via ``done()`` once the
    worker signals ``finished``, ``aborted``, or ``failed``.
    """

    #: Emitted once when the operator presses Abort.
    abort_requested = pyqtSignal()

    def __init__(self, *, phases: list[_Phase], dry_run: bool = False,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        title = "OpenLIFU Verification -- Running"
        if dry_run:
            title += "  [DRY RUN]"
        self.setWindowTitle(title)
        self.setModal(True)
        # No close button; the dialog closes itself when the worker
        # signals completion.
        self.setWindowFlag(Qt.WindowType.WindowCloseButtonHint, False)

        # --- Header ---
        header = QLabel("Characterization in progress")
        hfont = QFont()
        hfont.setPointSize(hfont.pointSize() + 4)
        hfont.setBold(True)
        header.setFont(hfont)

        self.section_label = QLabel("Current phase: preparing\u2026")
        self.section_label.setProperty("role", "subtle")

        # Chunky progress bar so it reads at a glance from across
        # the room.
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.progress.setMinimumHeight(24)

        # Progress and remaining time follow the phase model: the bar
        # moves through the current phase's expected share (capped at
        # its end until the next marker arrives).
        self._phases = phases
        self._phase_starts = [0.0]
        for p in phases[:-1]:
            self._phase_starts.append(self._phase_starts[-1] + p.seconds)
        self._total_s = self._phase_starts[-1] + phases[-1].seconds
        self._phase_idx = 0
        self._t_start = time.monotonic()
        self._phase_t0 = self._t_start
        self._clock = QTimer(self)
        self._clock.timeout.connect(self._refresh_time_text)
        self._clock.start(1000)
        self._refresh_time_text()

        # --- Acceptance-gate group ---
        gates_group = QGroupBox("Acceptance gates")
        gates_layout = QVBoxLayout(gates_group)
        gates_layout.setContentsMargins(6, 8, 6, 6)
        gates_layout.setSpacing(0)
        self.tracker = AcceptanceTracker()
        gates_layout.addWidget(self.tracker)

        # --- Log group ---
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)
        log_layout.setContentsMargins(6, 8, 6, 6)
        self.terminal = QPlainTextEdit()
        self.terminal.setObjectName("miniTerminal")
        self.terminal.setReadOnly(True)
        self.terminal.setMaximumBlockCount(4000)
        mono = QFont("Consolas")
        mono.setStyleHint(QFont.StyleHint.Monospace)
        if not mono.exactMatch():
            mono = QFont("Menlo")
            mono.setStyleHint(QFont.StyleHint.Monospace)
        mono.setPointSize(8)
        self.terminal.setFont(mono)
        self.terminal.setFixedHeight(150)
        log_layout.addWidget(self.terminal)

        # --- Abort ---
        self._aborting = False
        self.abort_button = QPushButton("Abort")
        self.abort_button.setToolTip(
            "Stop the scan after the current step and return to the "
            "start screen."
        )
        self.abort_button.clicked.connect(self._on_abort_clicked)
        abort_row = QHBoxLayout()
        abort_row.addStretch(1)
        abort_row.addWidget(self.abort_button)

        # --- Assemble ---
        sep = QFrame()
        sep.setObjectName("hLine")
        sep.setFrameShape(QFrame.Shape.HLine)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 14)
        layout.setSpacing(10)
        layout.addWidget(header)
        layout.addWidget(self.section_label)
        layout.addWidget(sep)
        layout.addWidget(self.progress)
        layout.addWidget(gates_group)
        layout.addWidget(log_group)
        layout.addLayout(abort_row)
        self.resize(640, 720)

    def reject(self) -> None:  # noqa: D401 - Qt override
        """Ignore Esc; the run can only be stopped via Abort."""

    def _on_abort_clicked(self) -> None:
        self._aborting = True
        self.abort_button.setEnabled(False)
        self.abort_button.setText("Aborting\u2026")
        self.section_label.setText(
            "Abort requested \u2013 stopping after the current step\u2026"
        )
        self.abort_requested.emit()

    # ------------------------------------------------------------------
    # Slot
    # ------------------------------------------------------------------
    def on_log(self, formatted: str, raw: str) -> None:
        """Append ``formatted`` to the mini terminal and, if ``raw``
        matches a known phase marker, advance the progress bar.

        Called on the GUI thread via a queued signal connection.
        """
        clean = _ANSI_RE.sub("", formatted)
        self.terminal.appendPlainText(clean)
        # Autoscroll to bottom.
        cursor = self.terminal.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.terminal.setTextCursor(cursor)

        # Advance the acceptance-gate tracker.
        self.tracker.on_log(raw)

        # Only later phases can match, so the bar never rewinds;
        # skipped phases are jumped over.
        for idx in range(self._phase_idx + 1, len(self._phases)):
            phase = self._phases[idx]
            if phase.marker in raw:
                self._phase_idx = idx
                self._phase_t0 = time.monotonic()
                self._refresh_time_text()
                if not self._aborting:
                    self.section_label.setText(
                        f"Current phase: {phase.label}")
                break

    def _refresh_time_text(self) -> None:
        now = time.monotonic()
        phase = self._phases[self._phase_idx]
        in_phase = now - self._phase_t0
        done_s = (self._phase_starts[self._phase_idx]
                  + min(in_phase, phase.seconds))
        self.progress.setValue(int(100 * done_s / self._total_s))
        text = f"%p%   \u00b7   elapsed {_fmt_duration(now - self._t_start)}"
        if self._phase_idx < len(self._phases) - 1:
            remaining = max(phase.seconds - in_phase, 0.0) + (
                self._total_s - self._phase_starts[self._phase_idx]
                - phase.seconds)
            text += f"   \u00b7   ~{_fmt_duration(remaining)} remaining"
        self.progress.setFormat(text)


def _fmt_duration(seconds: float) -> str:
    """Format seconds as ``m:ss`` (or ``h:mm:ss`` past an hour)."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


# ----------------------------------------------------------------------
# Result dialog
# ----------------------------------------------------------------------
class ResultDialog(QDialog):
    """Modal completion dialog: verdict, artifact links, and the
    Write Data to Device / Scan Another / Exit choices.

    The device write itself runs on the worker thread: clicking Write
    emits :attr:`write_requested`, and the caller routes the worker's
    reply into :meth:`on_write_done`. After ``exec()`` returns,
    :attr:`choice` is ``"another"`` or ``"exit"``.
    """

    write_requested = pyqtSignal()

    _TIP_NOT_PASSING = ("Calibration Data can only be written from a "
                        "successful report")
    _TIP_NO_DATA = ("No calibration data was generated for this run "
                    "(frequency sweep skipped)")
    _TIP_NONSTANDARD = ("Calibration Data cannot be written from a "
                        "non-standard configuration")
    _TIP_WRITTEN = "Calibration Data already Written"

    def __init__(self, result: run_report.RunResult, *,
                 can_write: bool,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("OpenLIFU Calibration -- Complete")
        self.setModal(True)
        self.choice = "exit"

        pal = current_palette()
        verdict = "PASS" if result.passed else "FAIL"
        verdict_label = QLabel(
            f"Overall verdict: {verdict}"
        )
        verdict_label.setObjectName("verdictBanner")
        verdict_label.setProperty("verdict", "pass" if result.passed else "fail")
        verdict_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        nonstandard_label = None
        if result.nonstandard_reasons:
            nonstandard_label = QLabel(
                NONSTANDARD_BANNER + "\n" + "\n".join(
                    f"  \u2022 {r}" for r in result.nonstandard_reasons)
            )
            nonstandard_label.setProperty("role", "alert")
            nonstandard_label.setWordWrap(True)

        # Hyperlinks to the report directory and (when present) the
        # PDF file. QLabel with ``openExternalLinks=False`` and a
        # ``linkActivated`` signal routes clicks through
        # ``QDesktopServices.openUrl`` so the user's default browser
        # / file explorer handles the click.
        link_labels: list[QLabel] = []
        if result.run_dir is not None:
            link_labels.append(self._link_label(
                "Open report folder", Path(result.run_dir), pal
            ))
        pdf = result.files.get("pdf") if result.files else None
        if pdf is not None and Path(pdf).is_file():
            link_labels.append(self._link_label(
                "Open report PDF", Path(pdf), pal
            ))
        xlsx = result.files.get("xlsx") if result.files else None
        if xlsx is not None and Path(xlsx).is_file():
            link_labels.append(self._link_label(
                "Open report XLSX", Path(xlsx), pal
            ))

        # Device-write status line; filled in by ``on_write_done``.
        self.write_line = QLabel()
        self.write_line.setTextFormat(Qt.TextFormat.RichText)
        self.write_line.setVisible(False)

        self.write_button = QPushButton("Write Data to Device")
        self.write_button.clicked.connect(self._on_write_clicked)
        if not result.passed:
            self.write_button.setEnabled(False)
            self.write_button.setToolTip(self._TIP_NOT_PASSING)
        elif result.nonstandard_reasons:
            self.write_button.setEnabled(False)
            self.write_button.setToolTip(self._TIP_NONSTANDARD)
        elif not can_write:
            self.write_button.setEnabled(False)
            self.write_button.setToolTip(self._TIP_NO_DATA)
        else:
            self.write_button.setDefault(True)

        self.another_button = QPushButton("Scan Another")
        self.another_button.clicked.connect(
            lambda: self._finish_with("another"))
        self.exit_button = QPushButton("Exit")
        self.exit_button.clicked.connect(lambda: self._finish_with("exit"))

        # Keep the blue "default" highlight on Write only; auto-default
        # buttons would otherwise take it over whenever they get focus.
        for b in (self.write_button, self.another_button, self.exit_button):
            b.setAutoDefault(False)
        # The ``:default`` stylesheet rule renders the label bold, but
        # Qt sizes the button for the regular font. Reserve room for
        # the bold text via a per-widget ``min-width`` (content box),
        # which takes precedence over the app-wide ``min-width: 72px``.
        # Polish first so ``font()`` reflects the stylesheet font-size.
        self.write_button.ensurePolished()
        bold = QFont(self.write_button.font())
        bold.setBold(True)
        text_w = QFontMetrics(bold).horizontalAdvance(self.write_button.text())
        self.write_button.setStyleSheet(f"min-width: {text_w + 4}px;")

        button_row = QHBoxLayout()
        button_row.addWidget(self.write_button)
        button_row.addStretch(1)
        button_row.addWidget(self.another_button)
        button_row.addWidget(self.exit_button)

        artifacts_label = QLabel("Artifacts")
        artifacts_label.setProperty("role", "section")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 14)
        layout.setSpacing(10)
        layout.addWidget(verdict_label)
        if nonstandard_label is not None:
            layout.addWidget(nonstandard_label)
        if link_labels:
            layout.addWidget(artifacts_label)
            for lbl in link_labels:
                layout.addWidget(lbl)
        layout.addSpacing(4)
        layout.addWidget(self.write_line)
        layout.addStretch(1)
        layout.addLayout(button_row)

        self.setMinimumWidth(460)

    # ------------------------------------------------------------------
    # Write flow
    # ------------------------------------------------------------------
    def _on_write_clicked(self) -> None:
        for b in (self.write_button, self.another_button, self.exit_button):
            b.setEnabled(False)
        self.write_button.setText("Writing\u2026")
        self.write_requested.emit()

    def on_write_done(self, ok: bool) -> None:
        """Slot for the worker's ``write_done`` signal."""
        pal = current_palette()
        self.write_button.setText("Write Data to Device")
        self.another_button.setEnabled(True)
        self.exit_button.setEnabled(True)
        if ok:
            self.write_button.setEnabled(False)
            self.write_button.setToolTip(self._TIP_WRITTEN)
            self.write_line.setText(
                f"Calibration data written to device: "
                f"<b style='color:{pal.success};'>OK</b>."
            )
        else:
            # Leave the button enabled so the operator can retry.
            self.write_button.setEnabled(True)
            self.write_line.setText(
                f"Calibration data write to device: "
                f"<b style='color:{pal.danger};'>FAILED</b>. "
                f"See the log for details."
            )
        self.write_line.setVisible(True)

    # ------------------------------------------------------------------
    # Leaving the dialog
    # ------------------------------------------------------------------
    def _finish_with(self, choice: str) -> None:
        if self.write_button.isEnabled():
            answer = QMessageBox.warning(
                self,
                "Calibration data not written",
                "The Calibration Data has not been written to the "
                "Device. Are you sure you want to Continue?",
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        self.choice = choice
        self.done(QDialog.DialogCode.Accepted)

    def reject(self) -> None:  # noqa: D401 - Qt override
        """Treat Esc / window close as Exit (same unwritten-data
        warning), and ignore it while a write is in flight."""
        if not self.exit_button.isEnabled():
            return
        self._finish_with("exit")

    @staticmethod
    def _link_label(text: str, target: Path, pal: Palette) -> QLabel:
        """Build a QLabel that opens ``target`` in the system's
        default handler when clicked."""
        url = QUrl.fromLocalFile(str(target.resolve()))
        label = QLabel(
            f'<a href="{url.toString()}" '
            f'style="color:{pal.accent}; text-decoration:none;">'
            f'&#128194; {text}</a>'
        )
        label.setTextFormat(Qt.TextFormat.RichText)
        label.setOpenExternalLinks(False)
        label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextBrowserInteraction
        )
        label.linkActivated.connect(lambda href: QDesktopServices.openUrl(
            QUrl(href)
        ))
        return label


# ----------------------------------------------------------------------
# Application entry point
# ----------------------------------------------------------------------
def _prefill_hydrophone_sn(prefs: OperatorPrefs, hydrophone_arg: str) -> None:
    """Populate ``prefs.hydrophone_sn`` from the calibration file
    metadata (best-effort). Mirrors :func:`run_report.run`."""
    if not hydrophone_arg:
        return
    try:
        prefs.hydrophone_sn = str(
            Hydrophone(hydrophone_arg).metadata.get("HYD_SN", "")
        ) or prefs.hydrophone_sn
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not read hydrophone S/N from %r: %s",
                       hydrophone_arg, e)


def _attach_qt_log_handler(bridge: _LogBridge) -> _QtLogHandler:
    """Attach a :class:`_QtLogHandler` to the root logger and return it
    so the caller can detach it after the run finishes."""
    root = logging.getLogger()
    handler = _QtLogHandler(bridge)
    root.addHandler(handler)
    # Root level: keep whatever the CLI configured. Ensure at least
    # INFO so the mini terminal shows the pipeline chatter.
    if root.level > logging.INFO or root.level == 0:
        root.setLevel(logging.INFO)
    return handler


def main(argv=None) -> int:
    args = build_gui_parser().parse_args(argv)

    # NB: don't call run_report._configure_root_logger here — the
    # worker calls it inside run_report.run() and its FileHandler
    # branch is not idempotent (would attach duplicate file
    # handlers). The Qt log handler we install below captures every
    # record from the moment it's attached, which is what the mini
    # terminal actually needs.

    # Force the two "no interactive prompt" flags on so ``run()``
    # doesn't try to read from stdin while the GUI is driving.
    args.no_prompt = True
    args.no_start_prompt = True

    app = QApplication.instance() or QApplication(sys.argv)
    # Fusion is Qt's platform-neutral style; it plays nicely with
    # heavy QSS overrides and looks identical on Windows / macOS /
    # Linux so the theme renders the same everywhere.
    try:
        from PyQt6.QtWidgets import QStyleFactory
        fusion = QStyleFactory.create("Fusion")
        if fusion is not None:
            app.setStyle(fusion)
    except Exception:  # noqa: BLE001 - style is cosmetic only
        pass
    apply_theme(app, args.theme)

    # Fail fast on a bad --scan-config before the operator fills in
    # the form.
    try:
        scan_config = (ScanConfig.with_overrides(args.scan_config)
                       if args.scan_config is not None else ScanConfig())
    except (OSError, ValueError) as e:
        QMessageBox.critical(None, "Scan configuration",
                             f"Could not load --scan-config:\n{e}")
        return 2
    nonstandard = run_report.nonstandard_reasons(args, scan_config)

    # Prefs seeded from the on-disk cache and, optionally, the
    # hydrophone metadata (so the S/N field is pre-filled).
    prefs = OperatorPrefs.load(args.prefs)
    try:
        prefs.test_app_version = _pkg_version("openlifu-verification")
    except PackageNotFoundError:
        prefs.test_app_version = ""
    _prefill_hydrophone_sn(prefs, args.hydrophone)

    # GUI writes from the result dialog, never inside run_report.run.
    args.write_config = False
    args.confirm_write_config = False

    # Reports folder: --reports-root, else the last folder picked in
    # the launcher, else ./test_reports.
    settings = QSettings("OpenLIFU", "VerificationGUI")
    if args.reports_root is None:
        saved = settings.value("reports_root", "", type=str)
        args.reports_root = Path(saved) if saved else run_report.DEFAULT_OUTPUT_DIR
    args.reports_root = Path(args.reports_root).resolve()

    exit_code = 130
    while True:
        launcher = LauncherDialog(args, prefs, nonstandard_reasons=nonstandard)
        if launcher.exec() != QDialog.DialogCode.Accepted:
            logger.info("Launcher closed.")
            return exit_code

        # Persist the collected prefs so :func:`run_report.run` (and
        # the next launcher) pick them up through the normal load path.
        prefs = launcher.collected_prefs
        prefs.save(args.prefs)
        args.frequency_khz = launcher.selected_frequency_kHz
        args.reports_root = launcher.selected_reports_root
        settings.setValue("reports_root", str(args.reports_root))

        choice, exit_code = _run_once(args, scan_config)
        if choice == "exit":
            return exit_code


def _run_once(args: argparse.Namespace,
              scan_config: ScanConfig) -> tuple[str, int]:
    """Run one characterization with the Running + Result dialogs.

    Returns ``(choice, exit_code)`` where ``choice`` is ``"another"``
    (go back to the launcher) or ``"exit"``.
    """
    # Attach the Qt log handler + build the RunningDialog. The bridge
    # signal is queued so slots always run on the GUI thread. Attach
    # BEFORE emitting the launch-config log line so the mini terminal
    # captures it.
    bridge = _LogBridge()
    handler = _attach_qt_log_handler(bridge)
    voltage_V = (args.voltage if args.voltage is not None
                 else VerificationTank.default_voltage_for(args.frequency_khz))
    phases = _estimate_phases(
        scan_config, args.frequency_khz, voltage_V,
        skip_2d=args.skip_2d, skip_frequency=args.skip_frequency,
        skip_voltage=args.skip_voltage,
    )
    running = RunningDialog(phases=phases, dry_run=bool(args.dry_run))
    bridge.message.connect(
        running.on_log, Qt.ConnectionType.QueuedConnection,
    )

    logger.info(
        "GUI launching run: freq=%g kHz dry_run=%s",
        args.frequency_khz, bool(args.dry_run),
    )

    signals = _RunSignals()
    outcome: dict = {"kind": None, "result": None, "can_write": False,
                     "tb": None, "hw": None, "reason": None}
    done = QDialog.DialogCode.Accepted

    def _on_finished(res, can_write):
        outcome.update(kind="finished", result=res, can_write=can_write)
        running.done(done)

    def _on_aborted():
        outcome["kind"] = "aborted"
        running.done(done)

    def _on_halted(reason):
        outcome.update(kind="halted", reason=reason)
        running.done(done)

    def _on_failed(tb):
        outcome.update(kind="failed", tb=tb)
        running.done(done)

    def _on_hardware_disconnected(hardware, message):
        outcome.update(kind="hw", hw=(hardware, message))
        running.done(done)

    queued = Qt.ConnectionType.QueuedConnection
    signals.finished.connect(_on_finished, queued)
    signals.aborted.connect(_on_aborted, queued)
    signals.halted.connect(_on_halted, queued)
    signals.failed.connect(_on_failed, queued)
    signals.hardware_disconnected.connect(_on_hardware_disconnected, queued)

    worker = _WorkerThread(args, signals)
    running.abort_requested.connect(worker.request_abort)
    worker.start()
    running.exec()

    try:
        if outcome["kind"] == "finished":
            result = outcome["result"]
            dlg = ResultDialog(result, can_write=outcome["can_write"])
            dlg.write_requested.connect(worker.request_write)
            signals.write_done.connect(dlg.on_write_done, queued)
            dlg.exec()
            # Let the worker leave on_complete and disconnect.
            worker.release()
            worker.wait()
            return dlg.choice, result.exit_code

        worker.wait()

        if outcome["kind"] == "aborted":
            QMessageBox.information(
                None,
                "Scan aborted",
                "The scan was aborted. No report was generated.\n\n"
                "Returning to the start screen.",
            )
            return "another", 130

        if outcome["kind"] == "halted":
            QMessageBox.critical(
                None,
                "Scan halted",
                outcome["reason"] + "\n\n"
                "The remaining measurements depend on this step, so the "
                "scan was stopped and no report was generated. Check the "
                "hydrophone placement and the hardware, then try again.",
            )
            return "another", 1

        if outcome["kind"] == "hw":
            hardware, message = outcome["hw"]
            QMessageBox.warning(None, f"{hardware} not connected", message)
            return "another", 2

        if outcome["kind"] == "failed":
            QMessageBox.critical(
                None,
                "Run failed",
                "The characterization pipeline raised an unexpected "
                "exception. Details:\n\n" + outcome["tb"],
            )
        return "exit", 2
    finally:
        # Detach the log handler so the next run doesn't stack
        # duplicates on the root logger.
        logging.getLogger().removeHandler(handler)


if __name__ == "__main__":
    sys.exit(main())
