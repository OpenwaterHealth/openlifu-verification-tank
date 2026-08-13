"""Qt launcher for the TXM characterization report.

Presents a small dialog that collects the operator's tester name,
the three serial numbers (TX, console, hydrophone), and a frequency
choice, plus a "save calibration data to device" checkbox. Clicking
Start runs the same pipeline as :mod:`run_report` on a background
thread while a Running dialog shows a live progress bar, the current
phase label, and a mini terminal that mirrors the INFO log stream.

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
import logging
import re
import sys
import traceback
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from pathlib import Path
from typing import Optional

# Vendored subset of the CLI arg parser. Import first so a --help
# invocation on a machine without PyQt6 still works.
import run_report

from PyQt6.QtCore import Qt, QObject, QThread, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices, QFont, QTextCursor
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
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

from openlifu_verification import Hydrophone, OperatorPrefs


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
# Progress phase table
# ----------------------------------------------------------------------
# Each entry maps a substring seen in a log record to a ``(pct, label)``
# tuple. When the RunningDialog sees the substring in an INFO record,
# it jumps the progress bar to ``pct`` (never rewinding) and updates
# the "current phase" label. Values were calibrated against the
# characterization pipeline in
# :meth:`openlifu_verification.characterization.Characterization.run`
# so the bar advances at roughly wall-clock rate on a real run.
_PHASES: list[tuple[str, int, str]] = [
    # Substring, progress %, human label.
    ("LIFU Device fully connected",         5,   "Connecting to hardware"),
    ("Arrival check:",                       10,  "Arrival check"),
    ("Plane-wave depth calibration:",        15,  "Plane-wave depth calibration"),
    ("Running fresh find_peak",              25,  "Peak search"),
    ("Running 1-D lateral",                  35,  "1-D lateral scan"),
    ("Running 1-D elevation",                45,  "1-D elevation scan"),
    ("Running 1-D axial",                    55,  "1-D axial scan"),
    ("Running 2-D scan",                     65,  "2-D scan"),
    ("Waveform at peak:",                    75,  "Waveform at peak"),
    ("Frequency sweep across",               80,  "Frequency sweep"),
    ("Voltage sweep across",                 90,  "Voltage sweep"),
    ("Overall verdict:",                     97,  "Writing report"),
    ("Run directory:",                       100, "Complete"),
]


def _phase_for_message(msg: str) -> Optional[tuple[int, str]]:
    """Return the ``(progress_pct, label)`` for the first phase whose
    marker substring appears in ``msg``, or ``None`` if no marker
    matches."""
    for marker, pct, label in _PHASES:
        if marker in msg:
            return pct, label
    return None


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

    finished = pyqtSignal(object)  # emits RunResult (or None on error)
    failed = pyqtSignal(str)       # emits traceback string


class _WorkerThread(QThread):
    """Executes :func:`run_report.run` on a background QThread."""

    def __init__(self, args: argparse.Namespace,
                 signals: _RunSignals, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._args = args
        self._signals = signals

    def run(self) -> None:  # noqa: D401 - QThread entry point
        try:
            result = run_report.run(self._args)
            self._signals.finished.emit(result)
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
_ALLOWED_FREQUENCIES_KHZ: tuple[float, ...] = (155.0, 400.0)


class LauncherDialog(QDialog):
    """Modal form that collects operator inputs and kicks off a run.

    The Start button is only enabled when every required text field
    is non-empty. On accept, the dialog persists an ``OperatorPrefs``
    snapshot (so ``run_report.run`` picks up the values through its
    normal load path) and the caller reads
    :attr:`selected_frequency_kHz` and :attr:`write_config_requested`
    to fold into the arg namespace before starting the worker.
    """

    def __init__(self, args: argparse.Namespace,
                 prefs: OperatorPrefs,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        title = "OpenLIFU Verification \u2014 Run Report"
        if getattr(args, "dry_run", False):
            title += "  [DRY RUN]"
        self.setWindowTitle(title)
        self.setModal(True)
        self._args = args
        self._prefs = prefs

        # --- Header banner ---
        header = QLabel("OpenLIFU Verification")
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
        op_form.addRow("Tester name:", self.tester_edit)
        op_form.addRow("TXM S/N:", self.txm_edit)
        op_form.addRow("Console S/N:", self.console_edit)
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

        # Checkbox default is CHECKED per spec; the pass-gate + actual
        # write logic lives in run_report.run() so the GUI only decides
        # whether to set --write-config.
        self.save_cal_checkbox = QCheckBox(
            "Save calibration data to device on PASS"
        )
        self.save_cal_checkbox.setChecked(True)

        run_group = QGroupBox("Run configuration")
        run_form = QFormLayout(run_group)
        run_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        run_form.setHorizontalSpacing(12)
        run_form.setVerticalSpacing(8)
        run_form.addRow("Frequency:", self.freq_combo)
        run_form.addRow("", self.save_cal_checkbox)

        # --- Buttons ---
        self.start_button = QPushButton("Start")
        self.start_button.setDefault(True)
        cancel_button = QPushButton("Cancel")
        button_box = QDialogButtonBox(Qt.Orientation.Horizontal)
        button_box.addButton(self.start_button,
                             QDialogButtonBox.ButtonRole.AcceptRole)
        button_box.addButton(cancel_button,
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
    def write_config_requested(self) -> bool:
        return self.save_cal_checkbox.isChecked()

    @property
    def selected_frequency_kHz(self) -> float:
        """Return the drive frequency selected in the combo box."""
        data = self.freq_combo.currentData()
        return float(data) if data is not None else float(
            self._args.frequency_khz
        )


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

    Cancelling this dialog is a no-op on the underlying thread (the
    characterization has hardware side effects and can't be safely
    torn down mid-scan). The dialog is dismissed automatically once
    the worker signals ``finished`` or ``failed``.
    """

    def __init__(self, *, dry_run: bool = False,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        title = "OpenLIFU Verification \u2014 Running"
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
        self.resize(640, 720)

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

        phase = _phase_for_message(raw)
        if phase is not None:
            pct, label = phase
            # Never rewind, in case markers arrive out of order or a
            # log message from earlier in the pipeline is emitted
            # after the run has moved on.
            if pct > self.progress.value():
                self.progress.setValue(pct)
            self.section_label.setText(f"Current phase: {label}")


# ----------------------------------------------------------------------
# Result dialog
# ----------------------------------------------------------------------
class ResultDialog(QDialog):
    """Modal completion dialog: verdict + hyperlink + Finish."""

    def __init__(self, result: run_report.RunResult,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("OpenLIFU Verification \u2014 Complete")
        self.setModal(True)

        pal = current_palette()
        verdict = "PASS" if result.passed else "FAIL"
        verdict_label = QLabel(
            f"Overall verdict: {verdict}"
        )
        verdict_label.setObjectName("verdictBanner")
        verdict_label.setProperty("verdict", "pass" if result.passed else "fail")
        verdict_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

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

        # Device-write status line (only shown when write was attempted).
        write_line: Optional[QLabel] = None
        if result.device_write_ok is True:
            write_line = QLabel(
                f"Calibration data written to device: "
                f"<b style='color:{pal.success};'>OK</b>."
            )
        elif result.device_write_ok is False:
            write_line = QLabel(
                f"Calibration data write to device: "
                f"<b style='color:{pal.danger};'>FAILED</b>."
            )
        if write_line is not None:
            write_line.setTextFormat(Qt.TextFormat.RichText)

        finish_button = QPushButton("Finish")
        finish_button.setDefault(True)
        finish_button.clicked.connect(self.accept)
        button_row = QHBoxLayout()
        button_row.addStretch(1)
        button_row.addWidget(finish_button)

        artifacts_label = QLabel("Artifacts")
        artifacts_label.setProperty("role", "section")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 14)
        layout.setSpacing(10)
        layout.addWidget(verdict_label)
        if link_labels:
            layout.addWidget(artifacts_label)
            for lbl in link_labels:
                layout.addWidget(lbl)
        if write_line is not None:
            layout.addSpacing(4)
            layout.addWidget(write_line)
        layout.addStretch(1)
        layout.addLayout(button_row)

        self.setMinimumWidth(380)

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

    # Prefs seeded from the on-disk cache and, optionally, the
    # hydrophone metadata (so the S/N field is pre-filled).
    prefs = OperatorPrefs.load(args.prefs)
    try:
        prefs.test_app_version = _pkg_version("openlifu-verification")
    except PackageNotFoundError:
        prefs.test_app_version = ""
    _prefill_hydrophone_sn(prefs, args.hydrophone)

    launcher = LauncherDialog(args, prefs)
    if launcher.exec() != QDialog.DialogCode.Accepted:
        logger.info("Run cancelled at launcher dialog.")
        return 130

    # Persist the collected prefs so :func:`run_report.run` (and any
    # future runs) can pick them up through its normal load path.
    collected = launcher.collected_prefs
    collected.save(args.prefs)

    # Reflect the launcher's choices back into the arg namespace
    # before handing off to the worker. The pass gate for the device
    # write lives inside ``run_report.run``.
    args.frequency_khz = launcher.selected_frequency_kHz
    args.write_config = launcher.write_config_requested
    args.confirm_write_config = False  # GUI never re-prompts

    # Attach the Qt log handler + build the RunningDialog. The bridge
    # signal is queued so slots always run on the GUI thread. Attach
    # BEFORE emitting the launch-config log line so the mini terminal
    # captures it.
    bridge = _LogBridge()
    handler = _attach_qt_log_handler(bridge)
    running = RunningDialog(dry_run=bool(args.dry_run))
    bridge.message.connect(
        running.on_log, Qt.ConnectionType.QueuedConnection,
    )

    logger.info(
        "GUI launching run: freq=%g kHz write_config=%s dry_run=%s",
        args.frequency_khz, args.write_config, bool(args.dry_run),
    )

    signals = _RunSignals()
    result_holder: dict[str, Optional[run_report.RunResult]] = {"r": None}
    error_holder: dict[str, Optional[str]] = {"tb": None}

    def _on_finished(res):
        result_holder["r"] = res
        running.accept()

    def _on_failed(tb):
        error_holder["tb"] = tb
        running.reject()

    signals.finished.connect(_on_finished, Qt.ConnectionType.QueuedConnection)
    signals.failed.connect(_on_failed, Qt.ConnectionType.QueuedConnection)

    worker = _WorkerThread(args, signals)
    worker.start()
    running.exec()
    worker.wait()

    # Detach the log handler so subsequent invocations of ``main()``
    # (e.g. under a test harness) don't stack duplicates on the root
    # logger.
    logging.getLogger().removeHandler(handler)

    if error_holder["tb"] is not None:
        QMessageBox.critical(
            None,
            "Run failed",
            "The characterization pipeline raised an unexpected "
            "exception. Details:\n\n" + error_holder["tb"],
        )
        return 2

    result = result_holder["r"]
    if result is None:
        # Shouldn't happen (either finished or failed always fires).
        return 2

    ResultDialog(result).exec()
    return result.exit_code


if __name__ == "__main__":
    sys.exit(main())
