"""End-to-end TXM characterization workflow.

Wraps a :class:`VerificationTank` (real or :class:`DryRunTank`) with
a fixed sequence of measurement phases matching the sections of
``TXM_Testreport_Template.xlsx``:

    A. Test information       (:meth:`Characterization.collect_test_info`)
    B. Transmit Module        (:meth:`Characterization.collect_txm_info`)
    C. Console                (:meth:`Characterization.collect_console_info`)
    -- Arrival-time sanity check --
                              (:meth:`Characterization.warmup_and_arrival_check`)
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
from scipy.signal import hilbert

from .acceptance import AcceptanceCriteria
from .device_info import DeviceInfo
from .operator_prefs import OperatorPrefs
from .scan_config import ScanConfig, choose_range_mv
from .scan_results import ScanResult

logger = logging.getLogger(__name__)


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
    "elevation_image":       "D.8",
    "axial_image":           "D.9",
    # Peak Z Focus is the commanded focus depth (from the 1-D axial
    # scan / plane-wave depth cal) that seeds the focused-pulse
    # measurement.
    "peak_z_focus_mm":       "D.10",
    "waveform_image":        "D.11",
    # Focused-pulse block: the focused-pulse PNP (D.12), the raw
    # focused arrival time (D.13, informational only), and the
    # focused-arrival hydrophone depth (D.14, graded against
    # ``criteria.peak_depth``).
    "pnp_at_peak_MPa":       "D.12",
    "focused_arrival_us":    "D.13",
    "focused_depth_mm":      "D.14",
}

# Voltage-linearity R\u00b2 lives at F.8 (the row on which the F section's
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

# Scan geometry defaults kept as module constants for backward compat;
# the live values are pulled from :class:`ScanConfig` at run time.
LATERAL_1D_EXTENT_MM = 5.0
LATERAL_1D_POINTS = 21
SCAN_2D_EXTENT_MM = 3.0
SCAN_2D_POINTS = 13

# Freq sweep default: 8 points, -25 kHz .. +10 kHz around nominal @ 5 kHz
# spacing (overridable via ``scan_config.json``). Populates E.3..E.N.
FREQ_SWEEP_OFFSETS_KHZ = np.array([-25, -20, -15, -10, -5, 0, +5, +10], dtype=float)

# Voltage sweep: 6 points, 5..30 V (mirrors template F.2 - F.7).
VOLTAGE_SWEEP_V = np.array([5.0, 10.0, 15.0, 20.0, 25.0, 30.0])


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
    arrival_check: dict = field(default_factory=dict)
    freq_response: dict = field(default_factory=dict)
    voltage_response: dict = field(default_factory=dict)
    device_info: Optional[DeviceInfo] = None
    prefs: Optional[OperatorPrefs] = None
    criteria: Optional[AcceptanceCriteria] = None
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
    """First-arrival time (\u00b5s) via Hilbert-envelope threshold crossing.

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


def _linear_r2(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Return ``(slope, intercept, r2)`` for the best-fit line."""
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(intercept), float(r2)


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
        prefs: :class:`OperatorPrefs` \u2014 supplies tester name, serial
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
                 voltage_V: float = 20.0,
                 plot: bool = False):
        self.ver = ver
        self.prefs = prefs
        self.criteria = criteria or AcceptanceCriteria()
        self.scan_config = scan_config or ScanConfig()
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.frequency_kHz = float(frequency_kHz)
        self.voltage_V = float(voltage_V)
        self.plot = bool(plot)

        self.report = TestReport(
            prefs=prefs,
            criteria=self.criteria,
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
        return info

    def warmup_and_arrival_check(self) -> dict:
        """Fire a single pulse at the nominal focus and verify arrival."""
        pos = self.ver.hydrophone_position
        # Ensure the scope range is at the config's baseline for the
        # "warm-up" and every subsequent low-voltage phase.
        self._apply_baseline_range()
        meas = self.ver.measure_pressure(
            float(pos[0]), float(pos[1]), float(pos[2]),
            **self.scan_config.scope_kwargs(),
        )
        if meas is None:
            result = {"passed": False, "reason": "scope timeout"}
            self.report.arrival_check = result
            logger.error("Arrival check: scope timed out")
            return result

        # In the emission-referenced time frame the pulse arrives at
        # ``t = z / SOS`` (>= 0 for any real target), so no skip is
        # required to reject the electrical crosstalk (which now sits
        # at t = -system_transmit_delay_us and is outside the window).
        skip_us = 0.0
        arrival_us = _find_arrival_us(meas["t"], meas["trace"], skip_us=skip_us)
        # sos in mm/µs = m/s / 1000.
        sos_mm_per_us = float(self.scan_config.sos_water_m_per_s) / 1000.0
        expected_us = float(pos[2]) / sos_mm_per_us
        tol_us = expected_us * self.criteria.arrival_time.tol_pct / 100.0

        passed = (arrival_us is not None
                  and abs(arrival_us - expected_us) <= tol_us)

        result = {
            "passed": bool(passed),
            "arrival_us": arrival_us,
            "expected_us": expected_us,
            "tol_us": tol_us,
            "meas": meas,
        }
        self.report.arrival_check = result
        logger.info(
            "Arrival check: measured=%s expected=%.2f\u00b5s tol=\u00b1%.2f\u00b5s \u2192 %s",
            f"{arrival_us:.2f}\u00b5s" if arrival_us is not None else "n/a",
            expected_us, tol_us, "PASS" if passed else "FAIL",
        )
        return result

    def calibrate_depth_plane_wave(self, *,
                                    voltage_V: Optional[float] = None,
                                    n_pulses: int = 32,
                                    duration_usec: float = 8.0,
                                    skip_us: float = 12.0,
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
            duration_usec: Per-pulse duration (\u00b5s).
            skip_us: Ignore samples before this time when hunting
                for the first arrival.

        Returns:
            The full result dict from
            :meth:`VerificationTank.calibrate_hydrophone_depth`.
        """
        v = float(self.voltage_V if voltage_V is None else voltage_V)
        cfg = self.scan_config
        scope_kw = cfg.scope_kwargs()
        logger.info(
            "Plane-wave depth calibration: %d pulses @ %.1f V, %.1f \u00b5s each",
            n_pulses, v, duration_usec,
        )
        result = self.ver.calibrate_hydrophone_depth(
            voltage_V=v,
            n_pulses=n_pulses,
            duration_usec=duration_usec,
            skip_us=skip_us,
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
            unit="\u00b5s",
        )
        self.report.set_row(
            ROW["plane_wave_depth_mm"],
            "Plane-Wave Hydrophone Depth",
            float(result["distance_mm"]),
            unit="mm",
        )
        logger.info(
            "Plane-wave depth: arrival=%.3f \u00b5s, distance=%.3f mm "
            "(hydrophone_position[2] updated)",
            float(result["arrival_us"]), float(result["distance_mm"]),
        )
        # calibrate_hydrophone_depth leaves the LIFU armed with 8 \u00b5s
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
        scope_kw = cfg.scope_kwargs()
        ext = cfg.lateral_1d.extent_mm
        pts = cfg.lateral_1d.points

        logger.info("Running 1-D lateral (x) scan (\u00b1%.1f mm, %d pts)...", ext, pts)
        lat = self.ver.scan_1d(
            dim="x", scan_range=(-ext, ext), num=pts,
            absolute=False, **scope_kw,
        )

        ext_e = cfg.elevation_1d.extent_mm
        pts_e = cfg.elevation_1d.points
        logger.info("Running 1-D elevation (y) scan (\u00b1%.1f mm, %d pts)...",
                    ext_e, pts_e)
        elev = self.ver.scan_1d(
            dim="y", scan_range=(-ext_e, ext_e), num=pts_e,
            absolute=False, **scope_kw,
        )

        ext_a = cfg.axial_1d.extent_mm
        pts_a = cfg.axial_1d.points
        logger.info("Running 1-D axial (z) scan (\u00b1%.1f mm, %d pts)...",
                    ext_a, pts_a)
        axial = self.ver.scan_1d(
            dim="z", scan_range=(-ext_a, ext_a), num=pts_a,
            absolute=False, **scope_kw,
        )

        ext2 = cfg.scan_2d.extent_mm
        pts2 = cfg.scan_2d.points
        logger.info("Running 2-D scan (\u00b1%.1f mm, %d\u00d7%d pts)...", ext2, pts2, pts2)
        two_d = self.ver.scan_2d(
            x_range=(-ext2, ext2), num_x=pts2,
            y_range=(-ext2, ext2), num_y=pts2,
            absolute=False, **scope_kw,
        )

        self.report.scans["lateral_1d"] = lat
        self.report.scans["elevation_1d"] = elev
        self.report.scans["axial_1d"] = axial
        self.report.scans["scan_2d"] = two_d
        return {"lateral_1d": lat, "elevation_1d": elev,
                "axial_1d": axial, "scan_2d": two_d}

    def measure_waveform_at_peak(self) -> dict:
        """Fire one pulse at the peak; compute PNP + focused arrival depth.

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
        pos = self.ver.hydrophone_position
        self._apply_baseline_range()
        meas = self.ver.measure_pressure(
            float(pos[0]), float(pos[1]), float(pos[2]),
            **self.scan_config.scope_kwargs(),
        )
        if meas is None:
            raise RuntimeError("Scope timeout while measuring waveform at peak.")
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
                skip_us=12.0,
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
        self.report.set_row(ROW["pnp_at_peak_MPa"], "Focused Pulse PNP",
                            pnp_MPa, unit="MPa")
        self.report.set_row(ROW["focused_arrival_us"],
                            "Focused Pulse Arrival Time",
                            arrival_us if arrival_us is not None else float("nan"),
                            unit="\u00b5s")
        self.report.set_row(ROW["focused_depth_mm"],
                            "Focused Pulse Hydrophone Depth",
                            axial_depth_mm, unit="mm")
        logger.info(
            "Waveform at peak: PNP=%.3f MPa, arrival=%s, depth=%.2f mm "
            "(sos=%.0f m/s)",
            pnp_MPa,
            f"{arrival_us:.2f}\u00b5s" if arrival_us is not None else "n/a",
            axial_depth_mm, sos_m_per_s,
        )
        # Grade inline so the operator sees the PASS/FAIL verdict
        # immediately after the measurement, not retroactively at
        # the end of the run.
        self._grade_pnp_at_peak()
        self._grade_peak_depth()
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
        freqs = self.frequency_kHz + np.asarray(cfg.frequency_sweep.offsets_kHz,
                                                dtype=float)
        if duration_usec is None:
            # cycles / freq_kHz => ms; ×1000 => µs.
            duration_usec = (float(cfg.frequency_sweep.cycles_per_burst)
                             / self.frequency_kHz) * 1000.0
        self._apply_baseline_range()
        logger.info("Frequency sweep across %s kHz...", freqs.tolist())
        result = self.ver.scan_frequency(
            frequencies_kHz=freqs,
            duration_usec=duration_usec,
            **cfg.scope_kwargs(),
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
        resolution on the low-voltage points, but the linearity R\u00b2
        computation is unaffected because peak amplitude is what
        matters.

        Args:
            voltages_V: Override the voltages in :attr:`scan_config`.
        """
        cfg = self.scan_config
        volts = np.asarray(
            cfg.voltage_sweep.voltages_V if voltages_V is None else voltages_V,
            dtype=float,
        )
        volts = np.sort(volts)
        logger.info("Voltage sweep across %s V...", volts.tolist())

        # Pick the widest predicted scope range so every point fits in
        # one rapid-block pass. Fall back to leaving the scope alone
        # if we can't predict amplitudes yet (no baseline waveform,
        # no hydrophone).
        base_peak_mV = self._predict_peak_mV_at_ref()
        widest_range = None
        if base_peak_mV is not None and self.voltage_V > 0:
            headroom = cfg.scope.voltage_scan_headroom_pct
            widest_range = max(
                choose_range_mv(base_peak_mV * float(v) / float(self.voltage_V),
                                headroom_pct=headroom)
                for v in volts
            )
        if widest_range is not None and hasattr(self.ver, "set_hydrophone_range"):
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
            **cfg.scope_kwargs(),
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
        # F.8 carries the linearity R\u00b2 (the metric the section's
        # PASS/FAIL is graded on). The individual F.2 - F.7 PNP rows
        # stay ungraded (status="NA") so their values aren't
        # misinterpreted as per-voltage acceptance results.
        self.report.set_row(ROW["voltage_r2"], "Voltage Linearity R\u00b2",
                            float(r2))
        logger.info("Voltage linearity: slope=%.4f MPa/V  R\u00b2=%.4f",
                    slope, r2)
        # Grade inline so R\u00b2 PASS/FAIL is logged as soon as the
        # sweep finishes.
        self._grade_voltage_linearity()
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
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
        LIFU (voltage=30 V, duration=8 \u00b5s, pulse_count=32,
        trigger_mode="single") AND calls ``set_focus(0, 0, z_focus)``,
        never restoring either. Any phase that fires pulses afterwards
        \u2014 :meth:`find_peak`, the beam scans, the frequency /
        voltage sweeps \u2014 would inherit those stale settings and see
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

    def _predict_peak_mV_at_ref(self) -> Optional[float]:
        """Estimate the single-sided peak hydrophone signal in mV at ``self.voltage_V``.

        Uses :attr:`report.waveform_at_peak` (which was captured at
        ``self.voltage_V``): peak Pa \u2192 peak V via hydrophone
        sensitivity \u2192 peak mV. Returns ``None`` if we can't derive
        one (no waveform captured yet, no hydrophone attached, etc.).
        """
        wf = self.report.waveform_at_peak
        if not wf:
            return None
        trace = np.asarray(wf.get("trace"), dtype=float)
        if trace.size == 0:
            return None
        peak = float(np.max(np.abs(trace)))
        if wf.get("units") != "Pa":
            # Already in mV.
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
        peak_v = peak / pa_per_v
        return peak_v * 1000.0  # mV

    # ------------------------------------------------------------------
    # Grading
    # ------------------------------------------------------------------
    # The per-section ``_grade_*`` helpers are called from within the
    # measurement methods themselves so the operator sees a PASS/FAIL
    # verdict for each phase as soon as the underlying data lands.
    # ``grade()`` at the end of the run just aggregates whatever
    # verdicts are already recorded (plus a couple of section-less
    # checks like peak offset).

    def _grade_arrival(self) -> Optional[bool]:
        """Log the plane-wave arrival-check verdict.

        The arrival check is retained as a sanity log message but
        is no longer stamped on any graded report row: the raw
        focused arrival time (D.11) is informational only, and any
        acceptance threshold on it would have to include the
        transducer's per-element ``max_delay_us`` shift. The
        plane-wave hydrophone depth (D.3) and the focused-arrival
        depth (D.12) already provide graded, geometry-corrected
        checks on the arrival timing.

        Returns the boolean verdict (or ``None`` if the arrival
        check has not been run yet) so callers can still consult it
        without it feeding the overall pass/fail.
        """
        arr = self.report.arrival_check
        if not arr:
            return None
        passed = bool(arr.get("passed", False))
        expected_us = arr.get("expected_us")
        tol_us = arr.get("tol_us")
        if expected_us is not None and tol_us is not None:
            note = (f"measured={arr.get('arrival_us', float('nan')):.2f} \u00b5s, "
                    f"expected={expected_us:.2f} \u00b1 {tol_us:.2f} \u00b5s "
                    f"(tol \u00b1{self.criteria.arrival_time.tol_pct:g}%)")
        else:
            note = arr.get("reason", "")
        logger.info("[grade] Arrival time \u2192 %s (%s, informational)",
                    "PASS" if passed else "FAIL", note)
        return passed

    def _grade_pnp_at_peak(self) -> Optional[bool]:
        """Stamp the D.7 row with PASS/FAIL and log the verdict."""
        wf = self.report.waveform_at_peak
        if not wf:
            return None
        pnp = wf.get("pnp_MPa", float("nan"))
        thr = self.criteria.pnp_min_for(self.frequency_kHz)
        if thr is None:
            self.report.grade_row(
                ROW["pnp_at_peak_MPa"], passed=True,
                note=f"no threshold for {self.frequency_kHz} kHz",
            )
            logger.info("[grade] PNP at peak \u2192 SKIP "
                        "(no threshold for %g kHz; measured=%.3f MPa)",
                        self.frequency_kHz, float(pnp))
            return None
        passed = float(pnp) >= thr
        self.report.grade_row(ROW["pnp_at_peak_MPa"], passed=passed,
                              threshold=thr,
                              note=f"threshold >= {thr} MPa")
        logger.info("[grade] PNP at peak \u2192 %s "
                    "(measured=%.3f MPa, threshold \u2265 %.3f MPa)",
                    "PASS" if passed else "FAIL", float(pnp), float(thr))
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
        """Stamp the F.8 R\u00b2 row with PASS/FAIL and log the verdict.
        The individual F.2 - F.7 PNP rows are left with ``status="NA"``
        so their values aren't misinterpreted as per-voltage
        acceptance results."""
        vr = self.report.voltage_response
        if vr.get("r2") is None:
            return None
        r2 = float(vr["r2"])
        r2_min = float(self.criteria.voltage_linearity.r2_min)
        passed = r2 >= r2_min
        note = f"R\u00b2 = {r2:.4f} (min {r2_min})"
        if ROW["voltage_r2"] in self.report.rows:
            self.report.grade_row(ROW["voltage_r2"], passed=passed,
                                  threshold=r2_min, note=note)
        logger.info("[grade] Voltage linearity \u2192 %s (%s)",
                    "PASS" if passed else "FAIL", note)
        return passed

    def _grade_peak_offset(self) -> Optional[bool]:
        """Peak offset from nominal (0, 0). No dedicated report row \u2014
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
        """Grade D.12 on focused-arrival hydrophone depth vs. nominal.

        Passes when ``|depth - nominal_mm| <= tol_pct% * nominal_mm``.
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
        nominal = float(crit.nominal_mm)
        tol_mm = nominal * float(crit.tol_pct) / 100.0
        passed = abs(float(depth) - nominal) <= tol_mm
        threshold = f"{nominal:.1f} \u00b1 {tol_mm:.2f} mm"
        note = (f"depth = {float(depth):.3f} mm "
                f"(nominal {nominal:.1f}, tol \u00b1{crit.tol_pct:g}%)")
        if ROW["focused_depth_mm"] in self.report.rows:
            self.report.grade_row(ROW["focused_depth_mm"], passed=passed,
                                  threshold=threshold, note=note)
        logger.info("[grade] Peak depth \u2192 %s (%s)",
                    "PASS" if passed else "FAIL", note)
        return passed

    def grade(self) -> dict:
        """Aggregate the per-section verdicts into an overall PASS/FAIL.

        Each section is graded eagerly by its own ``_grade_*`` helper
        as soon as the underlying data is available (arrival, PNP,
        frequency deviation, voltage linearity), so this method is a
        thin aggregator. It also runs the couple of checks that
        don't have a natural attachment point in a measurement
        method (currently: peak offset).
        """
        summary: dict[str, Optional[bool]] = {}
        # Arrival-time sanity check is intentionally logged but NOT
        # aggregated into the overall verdict: the raw arrival time
        # is informational (the geometry-corrected plane-wave and
        # focused-arrival depths are the graded quantities).
        self._grade_arrival()
        summary["peak_offset"] = self._grade_peak_offset()
        summary["peak_xy"] = self._grade_peak_xy()
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
    def run(self, *,
            skip_2d: bool = False,
            skip_frequency: bool = False,
            skip_voltage: bool = False) -> TestReport:
        """Run every phase in order and grade."""
        self.collect_test_info()
        self.warmup_and_arrival_check()
        # Refine hydrophone depth via plane-wave arrival BEFORE the
        # peak search so find_peak fires focused pulses at the true
        # array-to-hydrophone distance rather than the operator's
        # rough initial estimate.
        try:
            self.calibrate_depth_plane_wave()
        except Exception as e:  # noqa: BLE001 - dry-run safety
            logger.warning(
                "Plane-wave depth calibration failed (%s); "
                "continuing with the seeded hydrophone_position[2]=%.2f mm.",
                e, float(self.ver.hydrophone_position[2]),
            )
        # Wipe any XY drift from a prior calibration / leftover run
        # so the peak search is guaranteed to start from a known,
        # transducer-centered reference. Axial (z) is left intact
        # because it now encodes the freshly-measured focus depth.
        self.ver.hydrophone_position[0] = 0.0
        self.ver.hydrophone_position[1] = 0.0
        logger.info("Running fresh find_peak from (x=0, y=0, z=%.2f mm).",
                    float(self.ver.hydrophone_position[2]))
        self.find_peak_xy(x0=0.0, y0=0.0)
        self.run_beam_scans() if not skip_2d else logger.info("Skipping beam scans (--skip-2d).")
        self.measure_waveform_at_peak()
        if not skip_frequency:
            self.sweep_frequency()
        else:
            logger.info("Skipping frequency sweep.")
        if not skip_voltage:
            self.sweep_voltage()
        else:
            logger.info("Skipping voltage sweep.")
        self.grade()
        self.report.finished_at = datetime.datetime.now().isoformat(timespec="seconds")
        return self.report
