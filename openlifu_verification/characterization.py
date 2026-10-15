"""End-to-end TXM characterization workflow.

Wraps a :class:`VerificationTank` (real or :class:`DryRunTank`) with
a fixed sequence of measurement phases matching the sections of
``TXM_Testreport_Template.xlsx``:

    A. Test information       (:meth:`Characterization.collect_test_info`)
    B. Transmit Module        (:meth:`Characterization.collect_txm_info`)
    C. Console                (:meth:`Characterization.collect_console_info`)
    -- Plane-wave depth cal --
                              (:meth:`Characterization.calibrate_depth_plane_wave`)
    -- Peak search --         (:meth:`Characterization.find_peak_xy`)
    D. 1-D + 2-D peak scans   (:meth:`Characterization.run_beam_scans`)
    D.10-D.14 Waveform at peak (:meth:`Characterization.measure_waveform_at_peak`)
    E. Frequency sweep        (:meth:`Characterization.sweep_frequency`)
    F. Voltage sweep          (:meth:`Characterization.sweep_voltage`)
    -- Acceptance grading --  (:meth:`Characterization.grade`)

The report layout intentionally drops the ``D.1 Axial Scan`` row from
the SONIQ protocol (this rig has no motorized axial stage) and uses
the 2-D XY scan in its place.
"""
from __future__ import annotations

import datetime
import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
from scipy.signal import hilbert

from .acceptance import AcceptanceCriteria
from .device_info import DeviceInfo
from .operator_prefs import OperatorPrefs
from .scan_config import PICOSCOPE_RANGES_MV, ScanConfig, choose_range_mv
from .scan_results import ScanResult
from .verificationtank import VerificationTank

logger = logging.getLogger(__name__)


class RunAborted(Exception):
    """Raised by :meth:`Characterization.run` when the caller's abort
    event is set. Checked between phases, so the phase in progress
    always completes before the run stops."""


class RunHalted(Exception):
    """Raised by :meth:`Characterization.run` when the hydrophone depth
    calibration or the peak search fails or lands out of tolerance.
    Every later measurement depends on those, so the run stops. The
    message is operator-facing."""


# ----------------------------------------------------------------------
# Report row IDs (match the XLSX template).
# ----------------------------------------------------------------------
ROW = {
    # A. Test Information
    "test_date":         "A.1",
    "tester_name":       "A.2",
    "test_app_version":  "A.3",
    "sdk_version":       "A.4",
    "hydrophone_sn":     "A.5",
    "hydrophone_model":  "A.6",
    "hydrophone_cal_date":"A.7",
    "hydrophone_cal_file":"A.8",
    "picoscope_variant": "A.9",
    "picoscope_sn":      "A.10",
    "picoscope_cal_date":"A.11",
    # B. Transmit Module
    "txm_sn":            "B.1",
    "txm_freq_kHz":      "B.2",
    "txm_hwid":          "B.3",
    "txm_fw_version":    "B.4",
    # C. Console
    "console_sn":        "C.1",
    "console_hwid":      "C.2",
    "console_fw_version":"C.3",
    # D. Peak Scans
    "voltage_rail":          "D.1",
    # Plane-wave depth calibration lives at the top of the D section
    # because it precedes the 2-D scan (its result seeds the focus
    # depth for every downstream focused measurement).
    "plane_wave_arrival_us": "D.2",
    "plane_wave_depth_mm":   "D.3",
    "scan_2d_image":         "D.4",
    # Hydrophone X / Y are derived from the 2-D scan, so they sit
    # immediately after D.4.
    "peak_x_mm":             "D.5",
    "peak_y_mm":             "D.6",
    "lateral_image":         "D.7",
    # 1-D lateral / elevation peak-offset rows sit right below their
    # respective figure rows (D.7, D.9). They grade the separation
    # between the 1-D scan peak and the 2-D-derived hydrophone
    # position against ``criteria.peak_offset.max_1d_axis_mm_for(freq)``.
    "lateral_peak_offset_mm":   "D.8",
    "elevation_image":       "D.9",
    "elevation_peak_offset_mm": "D.10",
    "axial_image":           "D.11",
    # Peak Z Focus is the commanded focus depth (from the plane-wave
    # depth cal / 1-D axial scan) that seeds the focused-pulse
    # measurement. Graded against the plane-wave depth (D.3) within
    # ``criteria.peak_depth.peak_z_focus_tol_pct``.
    "peak_z_focus_mm":       "D.12",
    "waveform_image":        "D.13",
    # Focused-pulse block: the focused-pulse PNP (D.14), the raw
    # focused arrival time (D.15, informational only), and the
    # focused-arrival hydrophone depth (D.16, graded against
    # ``criteria.peak_depth.focused_tol_pct``).
    "pnp_at_peak_MPa":       "D.14",
    "focused_arrival_us":    "D.15",
    "focused_depth_mm":      "D.16",
}

# Voltage-linearity R**2 lives at F.8 (the row on which the F section's
# PASS/FAIL is graded). Individual F.2 - F.7 PNP rows are informational
# only.
ROW["voltage_r2"] = "F.8"

# Freq-response deviation at nominal freq lives at the LAST row of
# the E section (the row on which the E section's PASS/FAIL is
# graded). E.1 is the voltage rail setting; E.2.. are the
# informational per-frequency PNP values; the trailing deviation row
# ID is set dynamically inside :meth:`Characterization.sweep_frequency`
# because its index depends on the number of sweep frequencies.
ROW["freq_deviation_pct"] = "E.2"  # placeholder, overwritten at runtime


# ----------------------------------------------------------------------
# Dataclasses
# ----------------------------------------------------------------------
@dataclass
class ReportRow:
    """One row in the Report sheet."""
    id: str
    label: str
    value: Any = None
    unit: str = ""
    status: str = "NA"           # "PASS" / "FAIL" / "NA"
    threshold: Any = None
    note: str = ""


@dataclass
class TestReport:
    """Everything gathered during a characterization run."""
    rows: dict[str, ReportRow] = field(default_factory=dict)
    scans: dict[str, ScanResult] = field(default_factory=dict)
    waveform_at_peak: dict = field(default_factory=dict)
    peak_xy_mm: tuple = (0.0, 0.0)
    freq_response: dict = field(default_factory=dict)
    voltage_response: dict = field(default_factory=dict)
    device_info: Optional[DeviceInfo] = None
    prefs: Optional[OperatorPrefs] = None
    criteria: Optional[AcceptanceCriteria] = None
    scan_config: Optional[ScanConfig] = None
    #: Why this run deviates from the standard configuration (empty
    #: for a standard run). Non-empty marks the report non-standard.
    nonstandard_reasons: list[str] = field(default_factory=list)
    frequency_kHz: float = 400.0
    voltage_V: float = 20.0
    started_at: str = ""
    finished_at: str = ""
    overall_pass: bool = False

    def set_row(self, id_: str, label: str, value: Any, *,
                unit: str = "", status: str = "NA",
                threshold: Any = None, note: str = "") -> ReportRow:
        row = ReportRow(id=id_, label=label, value=value, unit=unit,
                        status=status, threshold=threshold, note=note)
        self.rows[id_] = row
        return row

    def grade_row(self, id_: str, *, passed: bool,
                  threshold: Any = None, note: str = "") -> None:
        if id_ not in self.rows:
            return
        self.rows[id_].status = "PASS" if passed else "FAIL"
        if threshold is not None:
            self.rows[id_].threshold = threshold
        if note:
            self.rows[id_].note = note


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _find_arrival_us(t_us: np.ndarray, trace: np.ndarray,
                     *, envelope_frac: float = 0.15,
                     skip_us: float = 0.0) -> Optional[float]:
    """First-arrival time (us) via Hilbert-envelope threshold crossing.

    Returns ``None`` if no sample of the envelope exceeds ``envelope_frac``
    of the peak envelope (i.e. no clear signal).
    """
    t_us = np.asarray(t_us, dtype=float)
    if skip_us > 0:
        mask = t_us >= skip_us
        if not mask.any():
            return None
        t_us = t_us[mask]
        trace = np.asarray(trace)[mask]
    env = np.abs(hilbert(trace))
    peak = env.max()
    if peak <= 0:
        return None
    idx = int(np.argmax(env > peak * envelope_frac))
    if env[idx] <= peak * envelope_frac:
        return None
    return float(t_us[idx])


def _pnp_MPa(trace_Pa: np.ndarray) -> float:
    """Peak negative pressure in MPa (assumes trace is in Pa)."""
    return float(-np.min(trace_Pa)) / 1e6


def _1d_scan_pnp_peak(scan) -> tuple[np.ndarray, np.ndarray, int]:
    """Return ``(coord, pnp_MPa, argmax)`` for a 1-D scan.

    Single source of truth for the 1-D peak used both by the grader
    and by the report figure, so the plotted maximum and the graded
    offset always agree.
    """
    coord = np.ravel(np.asarray(next(iter(scan.coords.values())), dtype=float))
    traces = np.asarray(scan.traces, dtype=float)
    traces = traces.reshape(-1, traces.shape[-1])
    pnp = -traces.min(axis=-1) / 1e6
    if pnp.size != coord.size or pnp.size == 0:
        raise ValueError(
            f"1-D scan has {pnp.size} traces but {coord.size} coords")
    return coord, pnp, int(np.argmax(pnp))


def _linear_r2(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Return ``(slope, intercept, r2)`` for the best-fit line."""
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(intercept), float(r2)


# --- Firmware version comparison -------------------------------------
_FW_INT_RE = None  # lazy-compiled below


def _fw_version_tuple(v: str) -> Optional[tuple[int, ...]]:
    """Parse a version string to a tuple of ints for ordering.

    Accepts ``"1.2.3"``, ``"v1.2.3"``, ``"1.2.3-beta"``,
    ``"1.2"``; strips a leading ``"v"`` / ``"V"`` and takes only
    the first three (or fewer) integer dotted components. Returns
    ``None`` if no integer component is found (empty / stub strings
    like ``""`` or ``"unknown"``)."""
    global _FW_INT_RE
    if _FW_INT_RE is None:
        import re as _re
        _FW_INT_RE = _re.compile(r"\d+")
    if not v:
        return None
    v = v.strip()
    if v[:1] in ("v", "V"):
        v = v[1:]
    parts = _FW_INT_RE.findall(v)
    if not parts:
        return None
    return tuple(int(p) for p in parts[:3])


def _fw_version_ge(reported: str, minimum: str) -> bool:
    """Return True iff ``reported >= minimum`` under tuple ordering.

    A missing or unparseable ``reported`` fails (we can't confirm
    the device meets the minimum). A missing / unparseable
    ``minimum`` short-circuits to True (the caller should have
    guarded this)."""
    m = _fw_version_tuple(minimum)
    if m is None:
        return True
    r = _fw_version_tuple(reported)
    if r is None:
        return False
    return r >= m


# --- Calibration date parsing ----------------------------------------
def _parse_cal_date(raw: str) -> Optional["datetime.date"]:
    """Best-effort parse of a calibration date string.

    Handles the common shapes we see in device metadata: ISO
    ``"YYYY-MM-DD"``, ``"DD-Mon-YYYY"`` (Onda hydrophone files),
    ``"DD/MM/YYYY"``, and ``"YYYYMMDD"`` (PicoScope driver strings).
    Returns ``None`` for empty or unrecognized inputs so the caller
    can skip the check without raising."""
    if not raw:
        return None
    s = str(raw).strip()
    if not s:
        return None
    fmts = (
        "%Y-%m-%d",
        "%d-%b-%Y", "%d-%B-%Y",
        "%b-%d-%Y", "%B-%d-%Y",
        "%d/%m/%Y", "%m/%d/%Y",
        "%Y%m%d",
    )
    for fmt in fmts:
        try:
            return datetime.datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


# ----------------------------------------------------------------------
# Characterization workflow
# ----------------------------------------------------------------------
class Characterization:
    """Orchestrator for the full TXM characterization sequence.

    The individual ``*`` phase methods can be called in any order after
    ``__init__`` (they only write into ``self.report``); :meth:`run`
    invokes them in the canonical order and returns the completed
    :class:`TestReport`.

    Args:
        ver: A live :class:`VerificationTank` (or :class:`DryRunTank`).
        prefs: :class:`OperatorPrefs` -- supplies tester name, serial
            numbers, hydrophone S/N.
        criteria: :class:`AcceptanceCriteria`. If ``None``, defaults
            are used and every graded row will still get a threshold
            attached.
        output_dir: Where artifacts (figures, npz, JSON, xlsx) will
            eventually be written by Stage 3. Only stored here; not
            created in Stage 2.
        frequency_kHz: Nominal center frequency of the device under
            test (155 or 400). Must match ``ver.frequency``.
        voltage_V: HV rail voltage for the peak scans + freq sweep.
            Defaults to ``VerificationTank.default_voltage_for(frequency_kHz)``.
        plot: Passed through to :meth:`VerificationTank.find_peak`.

    Attributes:
        report: The :class:`TestReport` being populated.
    """

    def __init__(self, ver, *,
                 prefs: OperatorPrefs,
                 criteria: Optional[AcceptanceCriteria] = None,
                 scan_config: Optional[ScanConfig] = None,
                 output_dir: Optional[Path] = None,
                 frequency_kHz: float = 400.0,
                 voltage_V: Optional[float] = None,
                 plot: bool = False):
        self.ver = ver
        self.prefs = prefs
        self.criteria = criteria or AcceptanceCriteria()
        self.scan_config = scan_config or ScanConfig()
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.frequency_kHz = float(frequency_kHz)
        if voltage_V is None:
            voltage_V = VerificationTank.default_voltage_for(self.frequency_kHz)
        self.voltage_V = float(voltage_V)
        self.plot = bool(plot)
        self._abort_event: Optional[threading.Event] = None

        self.report = TestReport(
            prefs=prefs,
            criteria=self.criteria,
            scan_config=self.scan_config,
            frequency_kHz=self.frequency_kHz,
            voltage_V=self.voltage_V,
            started_at=datetime.datetime.now().isoformat(timespec="seconds"),
        )

    # ------------------------------------------------------------------
    # Phases
    # ------------------------------------------------------------------
    def collect_test_info(self) -> DeviceInfo:
        """Populate sections A + B + C from prefs + SDK auto-extract."""
        info = DeviceInfo.collect(self.ver)
        self.report.device_info = info

        p = self.prefs
        r = self.report
        r.set_row(ROW["test_date"],        "Test Date",           info.test_date)
        r.set_row(ROW["tester_name"],      "Tester Name",         p.tester_name)
        r.set_row(ROW["test_app_version"], "Test App Version",    p.test_app_version)
        r.set_row(ROW["sdk_version"],      "SDK Version",         info.sdk_version)
        r.set_row(ROW["hydrophone_sn"],    "Hydrophone S/N",      p.hydrophone_sn)
        r.set_row(ROW["hydrophone_model"],   "Hydrophone Model",    info.hydrophone_model)
        r.set_row(ROW["hydrophone_cal_date"],"Hydrophone Cal Date", info.hydrophone_cal_date)
        r.set_row(ROW["hydrophone_cal_file"],"Hydrophone Cal File", info.hydrophone_cal_file)
        r.set_row(ROW["picoscope_variant"], "PicoScope Variant",  info.picoscope_variant)
        r.set_row(ROW["picoscope_sn"],      "PicoScope S/N",      info.picoscope_sn)
        r.set_row(ROW["picoscope_cal_date"],"PicoScope Cal Date", info.picoscope_cal_date)

        r.set_row(ROW["txm_sn"],           "Serial Number",       p.txm_sn)
        r.set_row(ROW["txm_freq_kHz"],     "Frequency",           self.frequency_kHz,   unit="kHz")
        r.set_row(ROW["txm_hwid"],         "Hardware ID",         info.txm_hwid)
        r.set_row(ROW["txm_fw_version"],   "Firmware Version",    info.txm_fw_version)

        r.set_row(ROW["console_sn"],       "Serial Number",       p.console_sn)
        r.set_row(ROW["console_hwid"],     "Hardware ID",         info.console_hwid)
        r.set_row(ROW["console_fw_version"],"Firmware Version",   info.console_fw_version)

        r.set_row(ROW["voltage_rail"],     "Voltage Rail Setting", self.voltage_V,      unit="V (+/-)")

        # Grade the static (info-only) checks eagerly so the operator
        # sees firmware / cal-date verdicts before any hardware
        # measurement kicks off. Idempotent - grade() re-runs them at
        # the end anyway.
        logger.info("Checking firmware + calibration acceptance...")
        self._grade_firmware()
        self._grade_calibration_dates()
        return info

    def calibrate_depth_plane_wave(self, *,
                                    voltage_V: Optional[float] = None,
                                    n_pulses: int = 32,
                                    duration_usec: float = 8.0,
                                    skip_us: Optional[float] = None,
                                    ) -> dict:
        """Run the plane-wave depth calibration.

        Fires ``n_pulses`` short plane-wave bursts through
        :meth:`VerificationTank.calibrate_hydrophone_depth`, records
        the arrival time and inferred array-to-hydrophone distance,
        and updates ``ver.hydrophone_position[2]`` in place. Populates
        the D.12 (plane-wave arrival) and D.13 (plane-wave depth)
        report rows.

        Args:
            voltage_V: HV rail for the calibration pulses. Defaults
                to ``self.voltage_V`` (i.e. reuses the acceptance
                rail).
            n_pulses: Number of pulses in the averaged burst.
            duration_usec: Per-pulse duration (us).
            skip_us: Ignore samples before this time when hunting
                for the first arrival. ``None`` (default) lets the
                primitive auto-compute it from pulse geometry.

        Returns:
            The full result dict from
            :meth:`VerificationTank.calibrate_hydrophone_depth`.
        """
        v = float(self.voltage_V if voltage_V is None else voltage_V)
        cfg = self.scan_config
        scope_kw = cfg.scope_kwargs()
        logger.info(
            "Plane-wave depth calibration: %d pulses @ %.1f V, %.1f us each, "
            "skip_us=%s",
            n_pulses, v, duration_usec,
            "auto" if skip_us is None else f"{float(skip_us):.1f}",
        )
        result = self.ver.calibrate_hydrophone_depth(
            voltage_V=v,
            n_pulses=n_pulses,
            duration_usec=duration_usec,
            skip_us=None if skip_us is None else float(skip_us),
            time_start_us=scope_kw.get("time_start_s", 0.0) * 1e6
                if scope_kw.get("time_start_s") is not None else 0.0,
            time_stop_us=scope_kw.get("time_stop_s", 100e-6) * 1e6
                if scope_kw.get("time_stop_s") is not None else 100.0,
            sampling_interval_ns=float(scope_kw.get(
                "sampling_interval_ns", 100.0)),
            hydrophone_range_mv=int(cfg.scope.hydrophone_range_mv),
            z_focus_mm=None,  # plane wave
            store=True,
            save=False,
        )
        self.report.set_row(
            ROW["plane_wave_arrival_us"],
            "Plane-Wave Arrival Time",
            float(result["arrival_us"]),
            unit="us",
        )
        self.report.set_row(
            ROW["plane_wave_depth_mm"],
            "Plane-Wave Hydrophone Depth",
            float(result["distance_mm"]),
            unit="mm",
        )
        logger.info(
            "Plane-wave depth: arrival=%.3f us, distance=%.3f mm "
            "(hydrophone_position[2] updated)",
            float(result["arrival_us"]), float(result["distance_mm"]),
        )
        # Grade D.3 inline so the operator sees the plane-wave depth
        # verdict alongside the raw measurement.
        self._grade_hydrophone_depth()
        # calibrate_hydrophone_depth leaves the LIFU armed with 8 us
        # / 30 V / 32-pulse / single-trigger settings. Restore the
        # acceptance rail + default 20-cycle burst so find_peak and
        # the beam scans that follow don't fire the wrong pulse.
        self._apply_baseline_pulse()
        return result

    def find_peak_xy(self, *, x0: float = 0.0, y0: float = 0.0,
                     **kw) -> tuple[float, float]:
        """Locate the true (x, y) peak and update ``hydrophone_position``.

        Defaults ``x0``/``y0`` to the origin so the verification
        pipeline always seeds the search from a known reference,
        independent of any prior calibration or leftover position.
        """
        for k, v in self._burst_scope_kwargs().items():
            kw.setdefault(k, v)
        x, y = self.ver.find_peak(x0=x0, y0=y0,
                                  plot=self.plot, store=True, save=False,
                                  keep_plot_open=False, **kw)
        self.report.peak_xy_mm = (float(x), float(y))
        # Populate the D.13 / D.14 rows with the located peak so the
        # xlsx / PDF report carries the raw (x, y) alongside the
        # PASS/FAIL verdicts stamped by ``_grade_peak_xy``.
        self.report.set_row(ROW["peak_x_mm"], "Hydrophone X Position",
                            float(x), unit="mm")
        self.report.set_row(ROW["peak_y_mm"], "Hydrophone Y Position",
                            float(y), unit="mm")
        logger.info("Peak located at (%.4f, %.4f) mm", x, y)
        # Grade inline so the offset PASS/FAIL is visible immediately.
        self._grade_peak_offset()
        self._grade_peak_xy()
        return float(x), float(y)

    def run_beam_scans(self) -> dict:
        """Run 1-D lateral, 1-D elevation, 1-D axial, and 2-D scans.

        Geometry (extents / point counts) comes from
        :attr:`scan_config`; results are stored under
        ``"lateral_1d"``, ``"elevation_1d"``, ``"axial_1d"``,
        ``"scan_2d"``.
        """
        cfg = self.scan_config
        plan = cfg.plan_for(self.frequency_kHz)
        scope_kw = self._burst_scope_kwargs()
        n_avg = int(cfg.scope.n_averages)
        ext = plan.lateral_1d.extent_mm
        pts = plan.lateral_1d.points

        # The lateral / elevation peak-offset rows depend only on their
        # own sweep, so each is graded right after it completes.
        logger.info("Running 1-D lateral (x) scan (\u00b1%.1f mm, %d pts)...", ext, pts)
        lat = self.ver.scan_1d(
            dim="x", scan_range=(-ext, ext), num=pts,
            absolute=False, n_averages=n_avg, **scope_kw,
        )
        self.report.scans["lateral_1d"] = lat
        self._set_1d_peak_offset_row(lat, ROW["lateral_peak_offset_mm"],
                                     "1-D Lateral Peak Offset")
        self._grade_lateral_peak_offset()
        self._check_abort()

        ext_e = plan.elevation_1d.extent_mm
        pts_e = plan.elevation_1d.points
        logger.info("Running 1-D elevation (y) scan (\u00b1%.1f mm, %d pts)...",
                    ext_e, pts_e)
        elev = self.ver.scan_1d(
            dim="y", scan_range=(-ext_e, ext_e), num=pts_e,
            absolute=False, n_averages=n_avg, **scope_kw,
        )
        self.report.scans["elevation_1d"] = elev
        self._set_1d_peak_offset_row(elev, ROW["elevation_peak_offset_mm"],
                                     "1-D Elevation Peak Offset")
        self._grade_elevation_peak_offset()
        self._check_abort()

        ext_a = plan.axial_1d.extent_mm
        pts_a = plan.axial_1d.points
        logger.info("Running 1-D axial (z) scan (\u00b1%.1f mm, %d pts)...",
                    ext_a, pts_a)
        axial = self.ver.scan_1d(
            dim="z", scan_range=(-ext_a, ext_a), num=pts_a,
            absolute=False, n_averages=n_avg, **scope_kw,
        )
        self.report.scans["axial_1d"] = axial
        self._check_abort()

        ext2 = plan.scan_2d.extent_mm
        pts2 = plan.scan_2d.points
        logger.info("Running 2-D scan (\u00b1%.1f mm, %d\u00d7%d pts)...", ext2, pts2, pts2)
        two_d = self.ver.scan_2d(
            x_range=(-ext2, ext2), num_x=pts2,
            y_range=(-ext2, ext2), num_y=pts2,
            absolute=False, n_averages=n_avg, **scope_kw,
        )
        self.report.scans["scan_2d"] = two_d
        return {"lateral_1d": lat, "elevation_1d": elev,
                "axial_1d": axial, "scan_2d": two_d}

    def _set_1d_peak_offset_row(self, scan: ScanResult, row_key: str,
                                label: str) -> None:
        """Create a 1-D peak-offset row holding the sweep's argmax
        coordinate (NaN if it can't be extracted), ready for
        ``_grade_1d_peak_offset`` to stamp PASS/FAIL on."""
        try:
            coord_arr, _pnp, idx = _1d_scan_pnp_peak(scan)
            offset_mm = float(coord_arr[idx])
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not extract 1-D peak for %s: %s",
                           row_key, e)
            offset_mm = float("nan")
        self.report.set_row(row_key, label, offset_mm, unit="mm")

    def measure_waveform_at_peak(self) -> dict:
        """Fire at the peak; compute PNP + focused arrival depth.

        The waveform and PNP (D.14) are captured at
        ``criteria.pnp_at_peak.reference_voltage_V`` (50 V / 100 Vpp by
        default) rather than the scan voltage, so the reported PNP is
        directly comparable to the specified threshold. A single
        probe pulse at the scan voltage sizes the scope range; the
        capture steps to a wider range if it still clips.

        Also fills the five "image" rows D.4-D.8 with
        ``"see Figure N"`` cross-references. The reported focused
        arrival time (D.11) and focused-arrival hydrophone depth
        (D.12) come from a focused-arrival pulse train
        (``calibrate_hydrophone_depth`` steered at the commanded
        z-focus), which averages more pulses and uses the first-RF-peak
        picker with quarter-cycle correction. The focused depth is
        graded against ``criteria.peak_depth`` (within ``tol_pct`` of
        ``nominal_mm``); the focused arrival time itself is
        informational.
        """
        logger.info("Measuring waveform at peak...")
        pos = self.ver.hydrophone_position
        xyz = (float(pos[0]), float(pos[1]), float(pos[2]))
        scope_kw = self._burst_scope_kwargs()
        self._apply_baseline_range()
        probe = self.ver.measure_pressure(*xyz, **scope_kw)
        if probe is None:
            raise RuntimeError("Scope timeout while measuring waveform at peak.")
        pnp_voltage_V = float(self.criteria.pnp_at_peak.reference_voltage_V)
        probe_mV = self._trace_peak_mV(probe)
        start_range = (
            choose_range_mv(probe_mV * pnp_voltage_V / self.voltage_V,
                            headroom_pct=self.scan_config.scope.voltage_scan_headroom_pct)
            if probe_mV is not None
            else int(self.scan_config.scope.hydrophone_range_mv)
        )
        can_range = hasattr(self.ver, "set_hydrophone_range")
        meas = None
        self.ver.set_voltage(pnp_voltage_V, wait=True)
        try:
            for rng in (r for r in PICOSCOPE_RANGES_MV if r >= start_range):
                if can_range:
                    self.ver.set_hydrophone_range(int(rng))
                meas = self.ver.measure_pressure(
                    *xyz, n_averages=int(self.scan_config.scope.pnp_n_averages),
                    **scope_kw,
                )
                if meas is None:
                    raise RuntimeError(
                        "Scope timeout while measuring waveform at peak.")
                peak_mV = self._trace_peak_mV(meas)
                if not can_range or peak_mV is None or peak_mV < 0.95 * rng:
                    break
                logger.warning(
                    "Waveform at peak clipped on \u00b1%d mV (%.1f mV); "
                    "retrying on a wider range", rng, peak_mV,
                )
        finally:
            self.ver.set_voltage(self.voltage_V, wait=True)
            self._apply_baseline_range()
        pnp_MPa = _pnp_MPa(meas["trace"]) if meas["units"] == "Pa" else float("nan")

        # Focused arrival + depth via the plane-wave-style calibrator
        # steered at the commanded z-focus. The picker subtracts the
        # transducer's max element delay, so ``distance_mm`` remains
        # the one-way array-to-hydrophone normal distance.
        z_focus_mm = float(pos[2])
        cfg = self.scan_config
        scope_kw = cfg.scope_kwargs()
        try:
            depth_result = self.ver.calibrate_hydrophone_depth(
                voltage_V=float(self.voltage_V),
                n_pulses=32,
                duration_usec=8.0,
                skip_us=None,  # auto-compute from pulse geometry
                time_start_us=(scope_kw.get("time_start_s", 0.0) or 0.0)
                              * 1e6,
                time_stop_us=(scope_kw.get("time_stop_s", 100e-6) or 100e-6)
                             * 1e6,
                sampling_interval_ns=float(scope_kw.get(
                    "sampling_interval_ns", 100.0)),
                hydrophone_range_mv=int(cfg.scope.hydrophone_range_mv),
                z_focus_mm=z_focus_mm,
                store=False,  # keep the plane-wave depth as the truth
                save=False,
            )
            arrival_us = float(depth_result["arrival_us"])
            axial_depth_mm = float(depth_result["distance_mm"])
            max_delay_us = float(depth_result["max_delay_us"])
        except Exception as e:  # noqa: BLE001 - dry-run safety
            logger.warning(
                "Focused depth calibration failed (%s); "
                "falling back to Hilbert-envelope arrival on the "
                "single-pulse trace.", e,
            )
            arrival_us = _find_arrival_us(meas["t"], meas["trace"],
                                          skip_us=0.0)
            sos_m_per_s = float(cfg.sos_water_m_per_s)
            axial_depth_mm = (arrival_us * sos_m_per_s / 1000.0
                              if arrival_us is not None else float("nan"))
            max_delay_us = 0.0
        finally:
            # calibrate_hydrophone_depth leaves the LIFU armed with the
            # short-burst / high-voltage / 32-pulse settings. Restore
            # the acceptance state so the frequency / voltage sweeps
            # that follow fire the intended waveform.
            self._apply_baseline_pulse()

        sos_m_per_s = float(cfg.sos_water_m_per_s)
        result = {
            **meas,
            "pnp_MPa": pnp_MPa,
            "pnp_voltage_V": pnp_voltage_V,
            "peak_z_focus_mm": z_focus_mm,
            "focused_depth_mm": axial_depth_mm,
            "focused_arrival_us": arrival_us,
            # Legacy aliases kept so any external consumer (or older
            # notebook / script) that reads the ``waveform_at_peak``
            # dict still finds the same values under their previous
            # keys.
            "axial_depth_mm": axial_depth_mm,
            "arrival_us": arrival_us,
            "focused_max_delay_us": max_delay_us,
            "sos_water_m_per_s": sos_m_per_s,
        }
        self.report.waveform_at_peak = result

        # D.4 / D.7 - D.9 / D.11 are figure cross-references so the
        # row grid isn't sparse in the PDF/XLSX. The figures
        # themselves are still embedded on the "Figures" sheet / PDF
        # pages.
        self.report.set_row(ROW["scan_2d_image"],   "2-D XY Scan",       "see Figure 1")
        self.report.set_row(ROW["lateral_image"],   "1-D Lateral Scan",  "see Figure 2")
        self.report.set_row(ROW["elevation_image"], "1-D Elevation Scan","see Figure 3")
        self.report.set_row(ROW["axial_image"],     "1-D Axial Scan",    "see Figure 4")
        self.report.set_row(ROW["waveform_image"],  "Waveform at Peak",  "see Figure 5")

        # D.10 shows the commanded focus depth (Peak Z Focus, =
        # the plane-wave-derived hydrophone depth). D.12 is the
        # focused-pulse PNP (graded against ``criteria.pnp_at_peak``).
        # D.13 is the raw focused-pulse arrival time (informational
        # only; the focused pulse arrives at
        # ``distance / SOS + max_delay_us`` so a meaningful acceptance
        # threshold would have to reference the geometric max delay).
        # D.14 is the arrival-derived depth (max-delay corrected)
        # which IS graded against ``criteria.peak_depth``.
        self.report.set_row(ROW["peak_z_focus_mm"], "Peak Z Focus",
                            z_focus_mm, unit="mm")
        self.report.set_row(ROW["pnp_at_peak_MPa"],
                            f"Focused Pulse PNP ({pnp_voltage_V:g}V/"
                            f"{2 * pnp_voltage_V:g}Vpp)",
                            pnp_MPa, unit="MPa")
        self.report.set_row(ROW["focused_arrival_us"],
                            "Focused Pulse Arrival Time",
                            arrival_us if arrival_us is not None else float("nan"),
                            unit="us")
        self.report.set_row(ROW["focused_depth_mm"],
                            "Focused Pulse Hydrophone Depth",
                            axial_depth_mm, unit="mm")
        logger.info(
            "Waveform at peak: PNP=%.3f MPa at %g V, arrival=%s, depth=%.2f mm "
            "(sos=%.0f m/s)",
            pnp_MPa, pnp_voltage_V,
            f"{arrival_us:.2f}us" if arrival_us is not None else "n/a",
            axial_depth_mm, sos_m_per_s,
        )
        # Grade inline so the operator sees the PASS/FAIL verdict
        # immediately after the measurement, not retroactively at
        # the end of the run.
        self._grade_pnp_at_peak()
        self._grade_peak_depth()
        self._grade_peak_z_focus()
        return result

    def sweep_frequency(self, *, duration_usec: Optional[float] = None) -> ScanResult:
        """Sweep pulse frequency around nominal; fill E.2.. with per-freq PNPs.

        The section's PASS/FAIL is graded on the final row (deviation
        of the nominal-frequency PNP from the peak PNP in the sweep),
        which sits after the per-frequency PNP rows because it is
        only computable once the sweep has completed. The individual
        E.2.. per-frequency rows are informational only.
        """
        cfg = self.scan_config
        offsets = cfg.plan_for(self.frequency_kHz).frequency_sweep.offsets_kHz
        freqs = self.frequency_kHz + np.asarray(offsets, dtype=float)
        if duration_usec is None:
            duration_usec = self._pulse_duration_us()
        self._apply_baseline_range()
        logger.info("Frequency sweep across %s kHz...", freqs.tolist())
        result = self.ver.scan_frequency(
            frequencies_kHz=freqs,
            duration_usec=duration_usec,
            n_averages=int(cfg.scope.pnp_n_averages),
            **self._burst_scope_kwargs(duration_usec),
        )
        # Per-freq PNP.
        traces = np.atleast_2d(result.traces)
        pnp = np.array([_pnp_MPa(row) for row in traces])
        self.report.scans["freq_sweep"] = result
        self.report.freq_response = {
            "frequencies_kHz": freqs,
            "pnp_MPa": pnp,
            "voltage_V": self.voltage_V,
        }
        self.report.set_row("E.1", "Voltage Rail Setting", self.voltage_V, unit="V (+/-)")
        # E.2.. : informational per-frequency PNP values (no grading).
        for i, (f, p) in enumerate(zip(freqs, pnp), start=2):
            row_id = f"E.{i}"
            label = f"PNP ({int(round(f))} kHz)"
            self.report.set_row(row_id, label, float(p), unit="MPa")
        # PNP deviation at nominal freq is the graded row and sits at
        # the end of the E section because it is only computed after
        # the sweep completes.
        ROW["freq_deviation_pct"] = f"E.{len(freqs) + 2}"
        # Grade inline so the PASS/FAIL is logged as soon as
        # the sweep finishes.
        self._grade_freq_response()
        return result

    def sweep_voltage(self, *,
                      voltages_V: Optional[np.ndarray] = None) -> ScanResult:
        """Sweep HV rail; fill F.2 - F.7 and compute linearity R^2.

        The scope's vertical range is picked once as the widest
        predicted across all voltages, so every point is captured in a
        single rapid-block pass. The trade-off is coarser vertical
        resolution on the low-voltage points, but the linearity R**2
        computation is unaffected because peak amplitude is what
        matters.

        Args:
            voltages_V: Override the voltages in :attr:`scan_config`.
        """
        cfg = self.scan_config
        vsweep = cfg.plan_for(self.frequency_kHz).voltage_sweep
        volts = np.asarray(
            vsweep.voltages_V if voltages_V is None else voltages_V,
            dtype=float,
        )
        volts = np.sort(volts)
        logger.info("Voltage sweep across %s V...", volts.tolist())

        # Pick the widest predicted scope range so every point fits in
        # one rapid-block pass. Fall back to leaving the scope alone
        # if we can't predict amplitudes yet (no baseline waveform,
        # no hydrophone).
        base_peak_mV = self._predict_peak_mV_at_ref()
        baseline_range_mV = int(cfg.scope.hydrophone_range_mv)
        # If the baseline waveform_at_peak capture saturated the scope,
        # the predicted peak equals the range and every downstream
        # scaling estimate would come out too small (the classic
        # "linear R^2=1.0 straight line clipped to the range ceiling"
        # symptom). Re-probe on progressively wider ranges until the
        # capture is comfortably unclipped, and use that peak instead.
        SATURATION_FRAC = 0.95
        if (base_peak_mV is not None
                and base_peak_mV >= SATURATION_FRAC * baseline_range_mV):
            logger.warning(
                "waveform_at_peak was clipped (%.1f mV >= %.0f%% of "
                "+/-%d mV baseline range); re-probing on wider range "
                "before sizing voltage sweep",
                base_peak_mV, SATURATION_FRAC * 100.0, baseline_range_mV,
            )
            reprobed = self._probe_unclipped_peak_mV(
                saturation_frac=SATURATION_FRAC,
            )
            if reprobed is not None:
                logger.info(
                    "Re-probed unclipped peak = %.1f mV "
                    "(was %.1f mV clipped at +/-%d mV)",
                    reprobed, base_peak_mV, baseline_range_mV,
                )
                base_peak_mV = reprobed
            else:
                logger.warning(
                    "Could not obtain an unclipped peak on any wider "
                    "range; sweep range will fall back to voltage_sweep."
                    "range_mv floor.",
                )
                base_peak_mV = None
        widest_range = None
        if base_peak_mV is not None and self.voltage_V > 0:
            headroom = cfg.scope.voltage_scan_headroom_pct
            widest_range = max(
                choose_range_mv(base_peak_mV * float(v) / float(self.voltage_V),
                                headroom_pct=headroom)
                for v in volts
            )
        # Apply the configured voltage-sweep floor (quantized up to a
        # valid PicoScope range). Guarantees the sweep uses at least
        # this range even when the auto prediction is missing or too
        # small (e.g. clipped baseline waveform, no hydrophone).
        sweep_floor_mV = int(vsweep.range_mv)
        sweep_floor = next((r for r in PICOSCOPE_RANGES_MV
                            if r >= sweep_floor_mV),
                           PICOSCOPE_RANGES_MV[-1])
        widest_range = (sweep_floor if widest_range is None
                        else max(widest_range, sweep_floor))
        if hasattr(self.ver, "set_hydrophone_range"):
            try:
                self.ver.set_hydrophone_range(int(widest_range))
                logger.info(
                    "Voltage sweep scope range \u00b1%d mV",
                    int(widest_range),
                )
            except Exception as e:
                logger.warning("Could not set scope range %s: %s",
                               widest_range, e)

        result = self.ver.scan_voltage(
            voltages_V=volts,
            n_averages=int(cfg.scope.n_averages),
            **self._burst_scope_kwargs(),
        )
        # Reset to baseline for anything downstream.
        self._apply_baseline_range()

        traces = np.atleast_2d(result.traces)
        pnp = np.array([_pnp_MPa(row) for row in traces])
        slope, intercept, r2 = _linear_r2(volts, pnp)
        self.report.scans["voltage_sweep"] = result
        self.report.voltage_response = {
            "voltages_V": volts,
            "pnp_MPa": pnp,
            "slope_MPa_per_V": slope,
            "intercept_MPa": intercept,
            "r2": r2,
        }
        self.report.set_row("F.1", "Frequency Setting", self.frequency_kHz, unit="kHz")
        for i, (v, p) in enumerate(zip(volts, pnp), start=2):
            row_id = f"F.{i}"
            label = f"PNP ({int(round(v))}V/{int(round(2*v))}Vpp)"
            self.report.set_row(row_id, label, float(p), unit="MPa")
        # F.8 carries the linearity R**2 (the metric the section's
        # PASS/FAIL is graded on). The individual F.2 - F.7 PNP rows
        # stay ungraded (status="NA") so their values aren't
        # misinterpreted as per-voltage acceptance results.
        self.report.set_row(ROW["voltage_r2"], "Voltage Linearity R**2",
                            float(r2))
        logger.info("Voltage linearity: slope=%.4f MPa/V  R**2=%.4f",
                    slope, r2)
        # Grade inline so R**2 PASS/FAIL is logged as soon as the
        # sweep finishes.
        self._grade_voltage_linearity()
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    #: Extra time (us) kept after the end of the burst's arrival.
    BURST_WINDOW_MARGIN_US = 10.0

    def _pulse_duration_us(self) -> float:
        """Acceptance pulse duration (us) at the nominal frequency,
        from ``VerificationTank.DEFAULT_DURATION_US_BY_FREQ_KHZ``."""
        return VerificationTank.default_duration_us_for(self.frequency_kHz)

    def _burst_scope_kwargs(self, duration_usec: Optional[float] = None) -> dict:
        """Scope kwargs with ``time_stop`` extended to cover the full burst.

        The window must reach ``arrival + burst duration + margin``,
        where arrival is at the deepest axial-scan point. The burst
        defaults to :meth:`_pulse_duration_us`. The configured
        ``time_stop_us`` is kept when it is already long enough.
        """
        cfg = self.scan_config
        if duration_usec is None:
            duration_usec = self._pulse_duration_us()
        axial_ext = cfg.plan_for(self.frequency_kHz).axial_1d.extent_mm
        z_mm = float(self.ver.hydrophone_position[2]) + float(axial_ext)
        arrival_us = z_mm / (float(cfg.sos_water_m_per_s) / 1000.0)
        min_stop_us = arrival_us + float(duration_usec) + self.BURST_WINDOW_MARGIN_US
        return cfg.scope_kwargs(min_time_stop_us=min_stop_us)

    def _apply_baseline_range(self) -> None:
        """Reset the scope's hydrophone channel to the configured baseline."""
        rng = int(self.scan_config.scope.hydrophone_range_mv)
        if hasattr(self.ver, "set_hydrophone_range"):
            try:
                self.ver.set_hydrophone_range(rng)
            except Exception as e:
                logger.debug("set_hydrophone_range(%d) failed: %s", rng, e)

    def _apply_baseline_pulse(self) -> None:
        """Re-arm the LIFU with the acceptance pulse settings and
        re-steer to the current ``hydrophone_position``.

        :meth:`VerificationTank.calibrate_hydrophone_depth` mutates the
        LIFU (voltage=30 V, duration=8 us, pulse_count=32,
        trigger_mode="single") AND calls ``set_focus(0, 0, z_focus)``,
        never restoring either. Any phase that fires pulses afterwards
        -- :meth:`find_peak`, the beam scans, the frequency /
        voltage sweeps -- would inherit those stale settings and see
        a transient short-burst field at the wrong rail steered at
        the array center (a sidelobe when the peak is off-axis)
        instead of the steady-state 20-cycle burst focused at the
        empirical peak. This method re-applies the acceptance
        defaults and re-steers to
        ``self.ver.hydrophone_position`` so downstream phases start
        from a clean, peak-centered state.
        """
        if hasattr(self.ver, "apply_pulse"):
            try:
                self.ver.apply_pulse(
                    frequency_kHz=self.frequency_kHz,
                    voltage=self.voltage_V,
                    duration_usec=self._pulse_duration_us(),
                )
            except Exception as e:
                logger.warning(
                    "apply_pulse(freq=%g kHz, voltage=%g V) failed while "
                    "restoring baseline: %s",
                    self.frequency_kHz, self.voltage_V, e,
                )
        if hasattr(self.ver, "set_focus"):
            pos = self.ver.hydrophone_position
            try:
                self.ver.set_focus(float(pos[0]), float(pos[1]),
                                   float(pos[2]))
            except Exception as e:
                logger.warning(
                    "set_focus(%.3f, %.3f, %.3f) failed while restoring "
                    "baseline: %s",
                    float(pos[0]), float(pos[1]), float(pos[2]), e,
                )

    def _trace_peak_mV(self, meas: dict) -> Optional[float]:
        """Single-sided peak of a capture in scope mV, or ``None`` if a
        Pa trace can't be converted back (no hydrophone calibration)."""
        peak = float(np.max(np.abs(np.asarray(meas["trace"], dtype=float))))
        if meas.get("units") != "Pa":
            return peak
        hyd = getattr(self.ver, "hydrophone", None)
        if hyd is None:
            return None
        try:
            pa_per_v = float(hyd.get_frequency_response(self.frequency_kHz * 1e3))
        except Exception:
            return None
        if pa_per_v <= 0:
            return None
        return peak / pa_per_v * 1000.0

    def _predict_peak_mV_at_ref(self) -> Optional[float]:
        """Estimate the single-sided peak hydrophone signal in mV at ``self.voltage_V``.

        Scales the :attr:`report.waveform_at_peak` capture (taken at
        ``pnp_voltage_V``) linearly to ``self.voltage_V``. Returns
        ``None`` if we can't derive one (no waveform captured yet, no
        hydrophone attached, etc.).
        """
        wf = self.report.waveform_at_peak
        if not wf or np.asarray(wf.get("trace")).size == 0:
            return None
        peak_mV = self._trace_peak_mV(wf)
        if peak_mV is None:
            return None
        return peak_mV * self.voltage_V / float(wf["pnp_voltage_V"])

    def _probe_unclipped_peak_mV(self, *,
                                 max_range_mv: int = 5000,
                                 saturation_frac: float = 0.95,
                                 ) -> Optional[float]:
        """Fire a single pulse at the calibrated peak on progressively
        wider scope ranges until the capture is unclipped.

        Used by :meth:`sweep_voltage` when the baseline
        ``waveform_at_peak`` capture appears to have saturated the
        PicoScope. Iterates :data:`PICOSCOPE_RANGES_MV` starting above
        the current baseline range; on each range fires
        :meth:`VerificationTank.measure_pressure` at the calibrated
        hydrophone position (same pulse settings as the acceptance
        run) and returns the peak amplitude in mV as soon as the
        capture stays below ``saturation_frac`` of that range.

        Restores the baseline scope range on return.

        Returns:
            Peak amplitude in mV at the current ``self.voltage_V``, or
            ``None`` if no wider range yielded an unclipped capture
            (e.g. no hydrophone, no scope, or truly saturating at
            ``max_range_mv``).
        """
        if not hasattr(self.ver, "set_hydrophone_range"):
            return None
        pos = self.ver.hydrophone_position
        cfg = self.scan_config
        baseline_range = int(cfg.scope.hydrophone_range_mv)
        scope_kw = self._burst_scope_kwargs()
        hyd = getattr(self.ver, "hydrophone", None)
        pa_per_v: Optional[float] = None
        if hyd is not None:
            try:
                pa_per_v = float(hyd.get_frequency_response(
                    self.frequency_kHz * 1e3))
                if pa_per_v <= 0:
                    pa_per_v = None
            except Exception:
                pa_per_v = None
        try:
            for r in PICOSCOPE_RANGES_MV:
                if r <= baseline_range:
                    continue
                if r > max_range_mv:
                    break
                try:
                    self.ver.set_hydrophone_range(int(r))
                except Exception as e:
                    logger.debug("set_hydrophone_range(%d) failed: %s", r, e)
                    continue
                meas = self.ver.measure_pressure(
                    float(pos[0]), float(pos[1]), float(pos[2]),
                    **scope_kw,
                )
                if meas is None:
                    logger.debug("Probe at +/-%d mV: scope timeout", r)
                    continue
                trace = np.asarray(meas["trace"], dtype=float)
                if trace.size == 0:
                    continue
                if meas.get("units") == "Pa":
                    if pa_per_v is None:
                        continue
                    peak_mV = float(np.max(np.abs(trace))) / pa_per_v * 1e3
                else:
                    peak_mV = float(np.max(np.abs(trace)))
                logger.info(
                    "Probe at +/-%d mV: peak=%.1f mV (%.0f%% of range)",
                    r, peak_mV, 100.0 * peak_mV / float(r),
                )
                if peak_mV < saturation_frac * float(r):
                    return peak_mV
            return None
        finally:
            self._apply_baseline_range()

    # ------------------------------------------------------------------
    # Grading
    # ------------------------------------------------------------------
    # The per-section ``_grade_*`` helpers are called from within the
    # measurement methods themselves so the operator sees a PASS/FAIL
    # verdict for each phase as soon as the underlying data lands.
    # ``grade()`` at the end of the run just aggregates whatever
    # verdicts are already recorded (plus a couple of section-less
    # checks like peak offset).

    def _grade_pnp_at_peak(self) -> Optional[bool]:
        """Stamp the D.7 row with PASS/FAIL and log the verdict."""
        wf = self.report.waveform_at_peak
        if not wf:
            return None
        pnp_MPa = float(wf.get("pnp_MPa", float("nan")))
        ref_V = float(self.criteria.pnp_at_peak.reference_voltage_V)
        meas_V = float(wf["pnp_voltage_V"])
        thr_ref_MPa = self.criteria.pnp_min_MPa_for(self.frequency_kHz)
        thr_MPa = thr_ref_MPa * meas_V / ref_V
        passed = pnp_MPa >= thr_MPa
        note = f"threshold >= {thr_ref_MPa:g} MPa at {ref_V:g} V"
        if meas_V != ref_V:
            note += f" (scaled to {thr_MPa:.3f} MPa at {meas_V:g} V)"
        self.report.grade_row(ROW["pnp_at_peak_MPa"], passed=passed,
                              threshold=thr_MPa, note=note)
        logger.info("[grade] PNP at peak \u2192 %s (measured=%.3f MPa at %g V, %s)",
                    "PASS" if passed else "FAIL", pnp_MPa, meas_V, note)
        return passed

    def _grade_freq_response(self) -> Optional[bool]:
        """Grade the trailing E-section row: how far below the sweep
        peak the PNP at nominal frequency sits, as a percentage of
        the peak. Passes if the deviation is within
        ``criteria.freq_response.max_deviation_pct``. The individual
        per-frequency PNP rows (E.2..) are left informational
        (``status="NA"``)."""
        fr = self.report.freq_response
        pnp = np.asarray(fr.get("pnp_MPa", []), dtype=float)
        freqs = np.asarray(fr.get("frequencies_kHz", []), dtype=float)
        if pnp.size == 0 or freqs.size == 0 or pnp.size != freqs.size:
            return None
        peak = float(np.max(pnp))
        if peak <= 0:
            return None
        # PNP at (or nearest to) the nominal frequency.
        idx_nom = int(np.argmin(np.abs(freqs - self.frequency_kHz)))
        pnp_nom = float(pnp[idx_nom])
        deviation_pct = (peak - pnp_nom) / peak * 100.0
        max_dev = float(self.criteria.freq_response.max_deviation_pct)
        passed = deviation_pct <= max_dev
        note = (f"nominal={pnp_nom:.3f} MPa, peak={peak:.3f} MPa "
                f"@ {freqs[int(np.argmax(pnp))]:.0f} kHz "
                f"(max dev {max_dev:g}%)")
        self.report.set_row(
            ROW["freq_deviation_pct"],
            "PNP Deviation at Nominal Freq",
            float(deviation_pct),
            unit="%",
        )
        self.report.grade_row(ROW["freq_deviation_pct"], passed=passed,
                              threshold=max_dev, note=note)
        logger.info("[grade] Frequency response \u2192 %s "
                    "(deviation=%.2f%%, %s)",
                    "PASS" if passed else "FAIL", deviation_pct, note)
        return passed

    def _grade_voltage_linearity(self) -> Optional[bool]:
        """Stamp the F.8 R**2 row with PASS/FAIL and log the verdict.
        The individual F.2 - F.7 PNP rows are left with ``status="NA"``
        so their values aren't misinterpreted as per-voltage
        acceptance results."""
        vr = self.report.voltage_response
        if vr.get("r2") is None:
            return None
        r2 = float(vr["r2"])
        r2_min = float(self.criteria.voltage_linearity.r2_min)
        passed = r2 >= r2_min
        note = f"R**2 = {r2:.4f} (min {r2_min})"
        if ROW["voltage_r2"] in self.report.rows:
            self.report.grade_row(ROW["voltage_r2"], passed=passed,
                                  threshold=r2_min, note=note)
        logger.info("[grade] Voltage linearity \u2192 %s (%s)",
                    "PASS" if passed else "FAIL", note)
        return passed

    def _grade_peak_offset(self) -> Optional[bool]:
        """Peak offset from nominal (0, 0). No dedicated report row --
        log-only for now."""
        if not self.report.peak_xy_mm:
            return None
        off = float(np.hypot(*self.report.peak_xy_mm))
        max_mm = self.criteria.peak_offset.max_mm
        passed = off <= max_mm
        note = f"|peak - (0,0)| = {off:.3f} mm (max {max_mm} mm)"
        logger.info("[grade] Peak offset \u2192 %s (%s)",
                    "PASS" if passed else "FAIL", note)
        return passed

    def _grade_peak_xy(self) -> Optional[bool]:
        """Grade D.10 / D.11 on per-axis displacement from (0, 0).

        Passes when both ``|x|`` and ``|y|`` are within
        ``criteria.peak_offset.max_axis_mm``. Returns the combined
        verdict (both must pass) or ``None`` if the peak has not
        been located yet.
        """
        if not self.report.peak_xy_mm:
            return None
        x, y = float(self.report.peak_xy_mm[0]), float(self.report.peak_xy_mm[1])
        max_axis = float(self.criteria.peak_offset.max_axis_mm)
        x_pass = abs(x) <= max_axis
        y_pass = abs(y) <= max_axis
        threshold = f"|axis| \u2264 {max_axis} mm"
        if ROW["peak_x_mm"] in self.report.rows:
            self.report.grade_row(ROW["peak_x_mm"], passed=x_pass,
                                  threshold=threshold,
                                  note=f"|x| = {abs(x):.3f} mm")
        if ROW["peak_y_mm"] in self.report.rows:
            self.report.grade_row(ROW["peak_y_mm"], passed=y_pass,
                                  threshold=threshold,
                                  note=f"|y| = {abs(y):.3f} mm")
        logger.info(
            "[grade] Peak X \u2192 %s (|x|=%.3f mm, max %.2f mm)",
            "PASS" if x_pass else "FAIL", abs(x), max_axis,
        )
        logger.info(
            "[grade] Peak Y \u2192 %s (|y|=%.3f mm, max %.2f mm)",
            "PASS" if y_pass else "FAIL", abs(y), max_axis,
        )
        return x_pass and y_pass

    def _grade_peak_depth(self) -> Optional[bool]:
        """Grade D.16 on focused-arrival hydrophone depth vs. nominal.

        Passes when
        ``|depth - nominal_mm(freq)| <= focused_tol_pct% * nominal_mm(freq)``.
        Returns the boolean verdict (or ``None`` if
        ``measure_waveform_at_peak`` has not been called yet or the
        arrival could not be picked).
        """
        wf = self.report.waveform_at_peak
        if not wf:
            return None
        depth = wf.get("focused_depth_mm", wf.get("axial_depth_mm"))
        if depth is None or not np.isfinite(depth):
            return None
        crit = self.criteria.peak_depth
        nominal = float(crit.nominal_for(self.frequency_kHz))
        tol_mm = nominal * float(crit.focused_tol_pct) / 100.0
        passed = abs(float(depth) - nominal) <= tol_mm
        threshold = f"{nominal:.1f} \u00b1 {tol_mm:.2f} mm"
        note = (f"depth = {float(depth):.3f} mm "
                f"(nominal {nominal:.1f} @ {self.frequency_kHz:g} kHz, "
                f"tol \u00b1{crit.focused_tol_pct:g}%)")
        if ROW["focused_depth_mm"] in self.report.rows:
            self.report.grade_row(ROW["focused_depth_mm"], passed=passed,
                                  threshold=threshold, note=note)
        logger.info("[grade] Peak depth \u2192 %s (%s)",
                    "PASS" if passed else "FAIL", note)
        return passed

    # ------------------------------------------------------------------
    # Static (data-independent) graders
    # ------------------------------------------------------------------
    # These grade info already collected during ``collect_test_info`` /
    # ``calibrate_depth_plane_wave`` / ``run_beam_scans`` /
    # ``measure_waveform_at_peak``; each is called inline from the
    # producing method so verdicts land in the log next to the raw
    # measurement.

    def _grade_firmware(self) -> Optional[bool]:
        """Grade B.4 (TXM firmware) and C.3 (console firmware).

        Empty version strings on either side skip that specific
        check (empty ``min_*_version`` disables the criterion; an
        empty *reported* version fails it so a mis-connected device
        can't sneak through). Returns the combined AND of the two
        sub-checks, or ``None`` if collect_test_info hasn't run yet.
        """
        info = self.report.device_info
        if info is None:
            return None
        crit = self.criteria.firmware
        results: list[bool] = []
        for row_key, reported, minimum, label in (
            (ROW["txm_fw_version"],     info.txm_fw_version,
             crit.min_txm_version,     "TXM firmware"),
            (ROW["console_fw_version"], info.console_fw_version,
             crit.min_console_version, "Console firmware"),
        ):
            if not minimum:
                # Check disabled - leave row NA.
                logger.info("[grade] %s \u2192 SKIP (no minimum set)", label)
                continue
            ok = _fw_version_ge(reported, minimum)
            note = f"reported={reported or 'n/a'!s}, minimum={minimum}"
            threshold = f">= {minimum}"
            if row_key in self.report.rows:
                self.report.grade_row(row_key, passed=ok,
                                      threshold=threshold, note=note)
            logger.info("[grade] %s \u2192 %s (%s)",
                        label, "PASS" if ok else "FAIL", note)
            results.append(ok)
        return all(results) if results else None

    def _grade_calibration_dates(self) -> Optional[bool]:
        """Grade A.7 (hydrophone) and A.11 (picoscope) cal dates.

        Ages older than ``criteria.calibration.max_age_years`` fail.
        Unparseable / empty dates are logged as SKIP (row stays NA)
        so we don't punish a bring-up run whose stub metadata lacks
        a date. Returns the AND of the parseable sub-checks, or
        ``None`` if collect_test_info hasn't run yet.
        """
        info = self.report.device_info
        if info is None:
            return None
        max_years = float(self.criteria.calibration.max_age_years)
        today = datetime.date.today()
        results: list[bool] = []
        for row_key, raw, label in (
            (ROW["hydrophone_cal_date"], info.hydrophone_cal_date,
             "Hydrophone cal date"),
            (ROW["picoscope_cal_date"],  info.picoscope_cal_date,
             "PicoScope cal date"),
        ):
            parsed = _parse_cal_date(raw)
            if parsed is None:
                logger.info("[grade] %s \u2192 SKIP (unparseable %r)",
                            label, raw)
                continue
            age_years = (today - parsed).days / 365.25
            ok = age_years <= max_years
            note = (f"cal={parsed.isoformat()}, "
                    f"age={age_years:.2f} y (max {max_years:g} y)")
            threshold = f"age <= {max_years:g} y"
            if row_key in self.report.rows:
                self.report.grade_row(row_key, passed=ok,
                                      threshold=threshold, note=note)
            logger.info("[grade] %s \u2192 %s (%s)",
                        label, "PASS" if ok else "FAIL", note)
            results.append(ok)
        return all(results) if results else None

    def _grade_hydrophone_depth(self) -> Optional[bool]:
        """Grade D.3 (plane-wave hydrophone depth) against LUT nominal.

        Uses ``criteria.peak_depth.nominal_for(freq)`` and the
        absolute ``hydrophone_tol_mm``. Returns the boolean verdict
        (or ``None`` if plane-wave depth calibration hasn't landed
        yet)."""
        row = self.report.rows.get(ROW["plane_wave_depth_mm"])
        if row is None or row.value is None:
            return None
        depth = float(row.value)
        crit = self.criteria.peak_depth
        nominal = float(crit.nominal_for(self.frequency_kHz))
        tol_mm = float(crit.hydrophone_tol_mm)
        passed = abs(depth - nominal) <= tol_mm
        threshold = f"{nominal:.1f} \u00b1 {tol_mm:.2f} mm"
        note = (f"depth = {depth:.3f} mm (nominal {nominal:.1f} @ "
                f"{self.frequency_kHz:g} kHz)")
        self.report.grade_row(ROW["plane_wave_depth_mm"], passed=passed,
                              threshold=threshold, note=note)
        logger.info("[grade] Hydrophone depth \u2192 %s (%s)",
                    "PASS" if passed else "FAIL", note)
        return passed

    def _grade_peak_z_focus(self) -> Optional[bool]:
        """Grade D.12 (Peak Z Focus) against D.3 (hydrophone depth).

        Passes when
        ``|peak_z - hydrophone_depth| <= peak_z_focus_tol_pct% *
        hydrophone_depth``. Because Peak Z Focus is seeded from the
        plane-wave depth calibration these should agree to within
        rounding under normal operation; the tolerance catches
        pathological cases where the calibration silently fell back
        to the initial estimate."""
        depth_row = self.report.rows.get(ROW["plane_wave_depth_mm"])
        focus_row = self.report.rows.get(ROW["peak_z_focus_mm"])
        if depth_row is None or focus_row is None:
            return None
        if depth_row.value is None or focus_row.value is None:
            return None
        depth = float(depth_row.value)
        focus = float(focus_row.value)
        if depth <= 0:
            return None
        crit = self.criteria.peak_depth
        tol_pct = float(crit.peak_z_focus_tol_pct)
        tol_mm = depth * tol_pct / 100.0
        passed = abs(focus - depth) <= tol_mm
        threshold = f"|focus - {depth:.2f}| \u2264 {tol_mm:.2f} mm"
        note = (f"focus = {focus:.3f} mm, hydrophone = {depth:.3f} mm "
                f"(tol \u00b1{tol_pct:g}%)")
        self.report.grade_row(ROW["peak_z_focus_mm"], passed=passed,
                              threshold=threshold, note=note)
        logger.info("[grade] Peak Z focus \u2192 %s (%s)",
                    "PASS" if passed else "FAIL", note)
        return passed

    def _grade_1d_peak_offset(self, *, scan_key: str, axis: str,
                               row_key: str, label: str) -> Optional[bool]:
        """Grade the D.8 / D.10 rows (1-D lateral / elevation peak offset).

        ``scan_key`` is the :attr:`TestReport.scans` key
        (``"lateral_1d"`` or ``"elevation_1d"``). ``axis`` is
        ``"x"`` or ``"y"`` and selects which component of the
        ``(x, y)`` peak located by :meth:`find_peak_xy` we compare
        against. The 1-D scan is stored with coordinates relative
        to the 2-D peak (``absolute=False``), so the reference is
        zero and the offset is simply the argmax coordinate of the
        1-D sweep's PNP (the same quantity plotted in the report)."""
        scan = self.report.scans.get(scan_key)
        if scan is None:
            return None
        try:
            coord_arr, _pnp, idx = _1d_scan_pnp_peak(scan)
            peak_coord = float(coord_arr[idx])
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not extract 1-D peak from %s: %s",
                           scan_key, e)
            return None
        # 1-D scans are ``absolute=False`` so the coordinate origin
        # is the 2-D peak. The offset is therefore the peak coord
        # itself.
        offset_mm = float(peak_coord)
        max_mm = self.criteria.peak_offset.max_1d_axis_mm_for(self.frequency_kHz)
        passed = abs(offset_mm) <= max_mm
        threshold = f"|{axis}| \u2264 {max_mm} mm"
        note = f"1-D peak {axis} = {offset_mm:+.3f} mm from 2-D peak"
        # The row was created by ``run_beam_scans`` right after the
        # scan completed; grade it in-place.
        if row_key in self.report.rows:
            self.report.grade_row(row_key, passed=passed,
                                  threshold=threshold, note=note)
        logger.info("[grade] %s peak offset \u2192 %s (%s)",
                    label, "PASS" if passed else "FAIL", note)
        return passed

    def _grade_lateral_peak_offset(self) -> Optional[bool]:
        return self._grade_1d_peak_offset(
            scan_key="lateral_1d", axis="x",
            row_key=ROW["lateral_peak_offset_mm"], label="Lateral",
        )

    def _grade_elevation_peak_offset(self) -> Optional[bool]:
        return self._grade_1d_peak_offset(
            scan_key="elevation_1d", axis="y",
            row_key=ROW["elevation_peak_offset_mm"], label="Elevation",
        )

    def grade(self) -> dict:
        """Aggregate the per-section verdicts into an overall PASS/FAIL.

        Each section is graded eagerly by its own ``_grade_*`` helper
        as soon as the underlying data is available (PNP,
        frequency deviation, voltage linearity), so this method is a
        thin aggregator. It also runs the couple of checks that
        don't have a natural attachment point in a measurement
        method (currently: peak offset).
        """
        summary: dict[str, Optional[bool]] = {}
        summary["firmware"] = self._grade_firmware()
        summary["calibration_dates"] = self._grade_calibration_dates()
        summary["hydrophone_depth"] = self._grade_hydrophone_depth()
        summary["peak_offset"] = self._grade_peak_offset()
        summary["peak_xy"] = self._grade_peak_xy()
        summary["lateral_peak_offset"] = self._grade_lateral_peak_offset()
        summary["elevation_peak_offset"] = self._grade_elevation_peak_offset()
        summary["peak_z_focus"] = self._grade_peak_z_focus()
        summary["pnp_at_peak"] = self._grade_pnp_at_peak()
        summary["peak_depth"] = self._grade_peak_depth()
        summary["freq_response"] = self._grade_freq_response()
        summary["voltage_linearity"] = self._grade_voltage_linearity()

        # Overall pass = all non-None entries pass.
        booleans = [v for v in summary.values() if v is not None]
        self.report.overall_pass = all(booleans) if booleans else False
        logger.info("Overall verdict: %s (%s)",
                    "PASS" if self.report.overall_pass else "FAIL",
                    ", ".join(f"{k}={v}" for k, v in summary.items()))
        return summary

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------
    def _check_abort(self) -> None:
        """Raise :class:`RunAborted` if the run's abort event is set."""
        if self._abort_event is not None and self._abort_event.is_set():
            logger.warning("Abort requested; stopping characterization.")
            raise RunAborted("Characterization aborted by operator.")

    def _require_pass(self, row_ids: list[str], problem: str) -> None:
        """Raise :class:`RunHalted` unless every row in ``row_ids`` passed."""
        failed = [self.report.rows.get(r) for r in row_ids
                  if getattr(self.report.rows.get(r), "status", None) != "PASS"]
        if not failed:
            return
        details = "; ".join(
            f"{row.note} (allowed {row.threshold})" if row is not None
            else "not measured" for row in failed)
        logger.error("%s: %s. Stopping the run.", problem, details)
        raise RunHalted(f"{problem}: {details}.")

    def run(self, *,
            skip_2d: bool = False,
            skip_frequency: bool = False,
            skip_voltage: bool = False,
            abort_event: Optional[threading.Event] = None) -> TestReport:
        """Run every phase in order and grade.

        If ``abort_event`` is given, it is checked between phases (and
        between the individual beam scans); once set, the current
        phase finishes and :class:`RunAborted` is raised.

        Raises :class:`RunHalted` if the plane-wave depth calibration
        or the peak search fails or is out of tolerance, since every
        later measurement depends on them.
        """
        self._abort_event = abort_event
        self._check_abort()
        self.collect_test_info()
        self._check_abort()
        # Refine hydrophone depth via plane-wave arrival BEFORE the
        # peak search so find_peak fires focused pulses at the true
        # array-to-hydrophone distance rather than the operator's
        # rough initial estimate.
        try:
            self.calibrate_depth_plane_wave()
        except Exception as e:  # noqa: BLE001 - surfaced to operator
            logger.exception("Plane-wave depth calibration failed")
            raise RunHalted(f"Plane-wave depth calibration failed: {e}") from e
        self._require_pass([ROW["plane_wave_depth_mm"]],
                           "Hydrophone depth is out of range")
        self._check_abort()
        # Wipe any XY drift from a prior calibration / leftover run
        # so the peak search is guaranteed to start from a known,
        # transducer-centered reference. Axial (z) is left intact
        # because it now encodes the freshly-measured focus depth.
        self.ver.hydrophone_position[0] = 0.0
        self.ver.hydrophone_position[1] = 0.0
        logger.info("Running fresh find_peak from (x=0, y=0, z=%.2f mm).",
                    float(self.ver.hydrophone_position[2]))
        try:
            self.find_peak_xy(x0=0.0, y0=0.0)
        except Exception as e:  # noqa: BLE001 - surfaced to operator
            logger.exception("Peak search failed")
            raise RunHalted(f"Peak search (find_peak) failed: {e}") from e
        self._require_pass([ROW["peak_x_mm"], ROW["peak_y_mm"]],
                           "Hydrophone is not centered on the focus")
        self._check_abort()
        if not skip_2d:
            self.run_beam_scans()
            self._check_abort()
        else:
            logger.info("Skipping beam scans (--skip-2d).")
        self.measure_waveform_at_peak()
        self._check_abort()
        if not skip_frequency:
            self.sweep_frequency()
            self._check_abort()
        else:
            logger.info("Skipping frequency sweep.")
        if not skip_voltage:
            self.sweep_voltage()
            self._check_abort()
        else:
            logger.info("Skipping voltage sweep.")
        self.grade()
        self.report.finished_at = datetime.datetime.now().isoformat(timespec="seconds")
        return self.report
