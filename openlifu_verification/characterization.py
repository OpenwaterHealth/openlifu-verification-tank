"""End-to-end TXM characterization workflow.

Wraps a :class:`VerificationTank` (real or :class:`DryRunTank`) with
a fixed sequence of measurement phases matching the sections of
``TXM_Testreport_Template.xlsx``:

    A. Test information       (:meth:`Characterization.collect_test_info`)
    B. Transmit Module        (:meth:`Characterization.collect_txm_info`)
    C. Console                (:meth:`Characterization.collect_console_info`)
    -- Arrival-time sanity check --
                              (:meth:`Characterization.warmup_and_arrival_check`)
    -- Peak search --         (:meth:`Characterization.find_peak_xy`)
    D. 1-D + 2-D peak scans   (:meth:`Characterization.run_beam_scans`)
    D.5-D.7 Waveform at peak  (:meth:`Characterization.measure_waveform_at_peak`)
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
    # B. Transmit Module
    "txm_sn":            "B.1",
    "txm_freq_kHz":      "B.2",
    "txm_hw_rev":        "B.3",
    "txm_hwid":          "B.4",
    "txm_fw_version":    "B.5",
    # C. Console
    "console_sn":        "C.1",
    "console_hw_rev":    "C.2",
    "console_hwid":      "C.3",
    "console_fw_version":"C.4",
    # D. Peak Scans
    "voltage_rail":      "D.1",
    "scan_2d_image":     "D.2",   # (repurposed: was Axial Scan)
    "lateral_image":     "D.3",
    "elevation_image":   "D.4",
    "waveform_image":    "D.5",
    "pnp_at_peak_MPa":   "D.6",
    "axial_depth_mm":    "D.7",
}

# Scan geometry defaults kept as module constants for backward compat;
# the live values are pulled from :class:`ScanConfig` at run time.
LATERAL_1D_EXTENT_MM = 5.0
LATERAL_1D_POINTS = 21
SCAN_2D_EXTENT_MM = 3.0
SCAN_2D_POINTS = 13

# Freq sweep: 8 points, -25 kHz .. +10 kHz around nominal @ 5 kHz spacing
# (mirrors template rows E.2 - E.9).
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
def _find_arrival_us(t_ns: np.ndarray, trace: np.ndarray,
                     *, envelope_frac: float = 0.15,
                     skip_us: float = 0.0) -> Optional[float]:
    """First-arrival time (\u00b5s) via Hilbert-envelope threshold crossing.

    Returns ``None`` if no sample of the envelope exceeds ``envelope_frac``
    of the peak envelope (i.e. no clear signal).
    """
    t_us = np.asarray(t_ns, dtype=float) * 1e-3
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


def _ripple_dB(values: np.ndarray) -> float:
    """Peak-to-peak ripple of a positive-valued sequence, in dB."""
    v = np.asarray(values, dtype=float)
    v = v[v > 0]
    if v.size < 2:
        return float("nan")
    return float(20.0 * np.log10(v.max() / v.min()))


def _concat_voltage_results(results: list[ScanResult]) -> ScanResult:
    """Concatenate a list of voltage-sweep :class:`ScanResult` objects.

    All inputs must share the same ``t`` axis and units; each contributes
    its own slice of the ``voltage_V`` coord and its own rows of
    ``traces``. If only one result is given, it's returned as-is.
    """
    if len(results) == 1:
        return results[0]
    if not results:
        raise ValueError("no voltage-sweep results to concatenate")
    base = results[0]
    t = base.t
    for r in results[1:]:
        if r.t.shape != t.shape or not np.allclose(r.t, t):
            raise ValueError("voltage-sweep results have mismatched time axes")
    traces = np.concatenate([np.atleast_2d(r.traces) for r in results], axis=0)
    coords = {"voltage_V": np.concatenate(
        [np.asarray(r.coords["voltage_V"], dtype=float) for r in results]
    )}
    meta = dict(base.metadata)
    meta["range_groups_mv"] = [r.metadata.get("hydrophone_range_mv") for r in results]
    return ScanResult(
        scan_type=base.scan_type,
        t=t,
        traces=traces,
        coords=coords,
        hydrophone_channel=base.hydrophone_channel,
        chunk_size=sum(r.chunk_size for r in results),
        units=base.units,
        metadata=meta,
    )


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
            numbers, hydrophone S/N, hardware revs.
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

        r.set_row(ROW["txm_sn"],           "Serial Number",       p.txm_sn)
        r.set_row(ROW["txm_freq_kHz"],     "Frequency",           self.frequency_kHz,   unit="kHz")
        r.set_row(ROW["txm_hw_rev"],       "Hardware Rev",        p.txm_hw_rev)
        r.set_row(ROW["txm_hwid"],         "Hardware ID",         info.txm_hwid)
        r.set_row(ROW["txm_fw_version"],   "Firmware Version",    info.txm_fw_version)

        r.set_row(ROW["console_sn"],       "Serial Number",       p.console_sn)
        r.set_row(ROW["console_hw_rev"],   "Hardware Rev",        p.console_hw_rev)
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

        # Skip past the electrical transient so we detect the acoustic
        # arrival, not the pickup at t=0.
        skip_us = float(self.ver.system_transmit_delay_us) - 5.0
        arrival_us = _find_arrival_us(meas["t"], meas["trace"], skip_us=skip_us)
        expected_us = (float(self.ver.system_transmit_delay_us)
                       + float(pos[2]) / 1.5)
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

    def find_peak_xy(self, **kw) -> tuple[float, float]:
        """Locate the true (x, y) peak and update ``hydrophone_position``."""
        x, y = self.ver.find_peak(plot=self.plot, store=True, save=False,
                                  keep_plot_open=False, **kw)
        self.report.peak_xy_mm = (float(x), float(y))
        logger.info("Peak located at (%.4f, %.4f) mm", x, y)
        return float(x), float(y)

    def run_beam_scans(self) -> dict:
        """Run 1-D lateral, 1-D elevation, and the 2-D grid scan.

        Geometry (extents / point counts) comes from
        :attr:`scan_config`; results are stored under
        ``"lateral_1d"``, ``"elevation_1d"``, ``"scan_2d"``.
        """
        cfg = self.scan_config
        scope_kw = cfg.scope_kwargs()
        ext = cfg.lateral_1d.extent_mm
        pts = cfg.lateral_1d.points

        logger.info("Running 1-D lateral scan (\u00b1%.1f mm, %d pts)...", ext, pts)
        lat = self.ver.scan_lateral(
            x_range=(-ext, ext), num_x=pts, num_y=1, y=0.0,
            absolute=False, **scope_kw,
        )

        ext_e = cfg.elevation_1d.extent_mm
        pts_e = cfg.elevation_1d.points
        logger.info("Running 1-D elevation scan (\u00b1%.1f mm, %d pts)...", ext_e, pts_e)
        # Elevation scan = single-x, multiple-y "lateral" call.
        elev = self.ver.scan_lateral(
            x_range=(0.0, 0.0), num_x=1,
            y_range=(-ext_e, ext_e), num_y=pts_e,
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
        self.report.scans["scan_2d"] = two_d
        return {"lateral_1d": lat, "elevation_1d": elev, "scan_2d": two_d}

    def measure_waveform_at_peak(self) -> dict:
        """Fire one pulse at the peak; compute PNP + axial depth."""
        pos = self.ver.hydrophone_position
        self._apply_baseline_range()
        meas = self.ver.measure_pressure(
            float(pos[0]), float(pos[1]), float(pos[2]),
            **self.scan_config.scope_kwargs(),
        )
        if meas is None:
            raise RuntimeError("Scope timeout while measuring waveform at peak.")
        pnp_MPa = _pnp_MPa(meas["trace"]) if meas["units"] == "Pa" else float("nan")
        axial_depth_mm = float(pos[2])
        result = {
            **meas,
            "pnp_MPa": pnp_MPa,
            "axial_depth_mm": axial_depth_mm,
        }
        self.report.waveform_at_peak = result
        self.report.set_row(ROW["pnp_at_peak_MPa"], "PNP at Peak", pnp_MPa, unit="MPa")
        self.report.set_row(ROW["axial_depth_mm"], "Axial Depth of Peak",
                            axial_depth_mm, unit="mm")
        # D.2/D.3/D.4/D.5 image rows get their paths from the report writer.
        logger.info("Waveform at peak: PNP=%.3f MPa, depth=%.2f mm",
                    pnp_MPa, axial_depth_mm)
        return result

    def sweep_frequency(self, *, duration_msec: Optional[float] = None) -> ScanResult:
        """Sweep pulse frequency around nominal; fill E.2 - E.9."""
        cfg = self.scan_config
        freqs = self.frequency_kHz + np.asarray(cfg.frequency_sweep.offsets_kHz,
                                                dtype=float)
        if duration_msec is None:
            duration_msec = float(cfg.frequency_sweep.cycles_per_burst) / self.frequency_kHz
        self._apply_baseline_range()
        logger.info("Frequency sweep across %s kHz...", freqs.tolist())
        result = self.ver.scan_frequency(
            frequencies_kHz=freqs,
            duration_msec=duration_msec,
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
        # Fill E.2 - E.9 (up to 8 entries).
        for i, (f, p) in enumerate(zip(freqs, pnp), start=2):
            row_id = f"E.{i}"
            label = f"PNP ({int(round(f))} kHz)"
            self.report.set_row(row_id, label, float(p), unit="MPa")
        self.report.set_row("E.1", "Voltage Rail Setting", self.voltage_V, unit="V (+/-)")
        return result

    def sweep_voltage(self, *,
                      voltages_V: Optional[np.ndarray] = None) -> ScanResult:
        """Sweep HV rail; fill F.2 - F.7 and compute linearity R^2.

        The scope's vertical range is auto-scaled per voltage: voltages
        that would clip the current range are grouped together and run
        in a separate rapid-block pass with a larger range. This means
        one call may produce several underlying ``scan_voltage``
        captures which are concatenated back into a single
        :class:`ScanResult`.
        """
        cfg = self.scan_config
        volts = np.asarray(
            cfg.voltage_sweep.voltages_V if voltages_V is None else voltages_V,
            dtype=float,
        )
        logger.info("Voltage sweep across %s V...", volts.tolist())

        groups = self._plan_voltage_range_groups(volts)
        results: list[ScanResult] = []
        for group_range, group_volts in groups:
            if group_range is not None and hasattr(self.ver, "set_hydrophone_range"):
                try:
                    self.ver.set_hydrophone_range(int(group_range))
                    logger.info(
                        "Voltage sweep group %s V \u2192 scope range \u00b1%d mV",
                        group_volts.tolist(), int(group_range),
                    )
                except Exception as e:
                    logger.warning("Could not set scope range %s: %s", group_range, e)
            group_result = self.ver.scan_voltage(
                voltages_V=group_volts,
                **cfg.scope_kwargs(),
            )
            results.append(group_result)
        # Reset to baseline for anything downstream.
        self._apply_baseline_range()

        result = _concat_voltage_results(results)
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
        logger.info("Voltage linearity: slope=%.4f MPa/V  R\u00b2=%.4f",
                    slope, r2)
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

    def _plan_voltage_range_groups(
        self, volts: np.ndarray
    ) -> list[tuple[Optional[int], np.ndarray]]:
        """Group ``volts`` by required scope range.

        Returns a list of ``(range_mv, voltages_array)`` in the same
        order as ``volts``. If we can't predict amplitudes (no
        baseline waveform yet, no hydrophone attached), returns a
        single group with ``range_mv=None`` (meaning: leave the scope
        alone).
        """
        cfg = self.scan_config
        base_peak_mV = self._predict_peak_mV_at_ref()
        if base_peak_mV is None or self.voltage_V <= 0:
            return [(None, volts)]
        headroom = cfg.scope.voltage_scan_headroom_pct
        planned = []
        for v in volts:
            expected_peak_mV = base_peak_mV * float(v) / float(self.voltage_V)
            planned.append(choose_range_mv(expected_peak_mV, headroom_pct=headroom))
        # Group consecutive equal ranges together to minimize the
        # number of separate rapid-block passes.
        groups: list[tuple[Optional[int], list[float]]] = []
        for rng, v in zip(planned, volts):
            if groups and groups[-1][0] == rng:
                groups[-1][1].append(float(v))
            else:
                groups.append((rng, [float(v)]))
        return [(rng, np.asarray(vs, dtype=float)) for rng, vs in groups]

    # ------------------------------------------------------------------
    # Grading
    # ------------------------------------------------------------------
    def grade(self) -> dict:
        """Compare every measurement to acceptance criteria; set statuses."""
        c = self.criteria
        r = self.report
        summary = {}

        # Arrival time.
        arr = r.arrival_check
        if arr:
            summary["arrival_time"] = arr.get("passed", False)

        # Peak offset from nominal (0, 0).
        if r.peak_xy_mm:
            off = float(np.hypot(*r.peak_xy_mm))
            passed = off <= c.peak_offset.max_mm
            summary["peak_offset"] = passed
            note = f"|peak - (0,0)| = {off:.3f} mm"
            # There's no dedicated report row, so log-only for now.
            logger.info("Peak offset from nominal: %s  \u2192 %s",
                        note, "PASS" if passed else "FAIL")

        # PNP at peak.
        if r.waveform_at_peak:
            pnp = r.waveform_at_peak.get("pnp_MPa", float("nan"))
            thr = c.pnp_min_for(self.frequency_kHz)
            if thr is None:
                r.grade_row(ROW["pnp_at_peak_MPa"], passed=True,
                            note=f"no threshold for {self.frequency_kHz} kHz")
                summary["pnp_at_peak"] = None
            else:
                passed = float(pnp) >= thr
                r.grade_row(ROW["pnp_at_peak_MPa"], passed=passed,
                            threshold=thr,
                            note=f"threshold >= {thr} MPa")
                summary["pnp_at_peak"] = passed

        # Frequency response ripple.
        if r.freq_response.get("pnp_MPa") is not None:
            ripple = _ripple_dB(np.asarray(r.freq_response["pnp_MPa"]))
            passed = ripple <= c.freq_response.max_ripple_dB
            summary["freq_response"] = passed
            # No single row for it; tag every E row's note with the ripple.
            note = f"ripple = {ripple:.2f} dB (max {c.freq_response.max_ripple_dB} dB)"
            for i in range(2, 10):
                rid = f"E.{i}"
                if rid in r.rows:
                    r.grade_row(rid, passed=passed, threshold=c.freq_response.max_ripple_dB,
                                note=note)

        # Voltage linearity R^2.
        if r.voltage_response.get("r2") is not None:
            r2 = r.voltage_response["r2"]
            passed = r2 >= c.voltage_linearity.r2_min
            summary["voltage_linearity"] = passed
            note = f"R\u00b2 = {r2:.4f} (min {c.voltage_linearity.r2_min})"
            for i in range(2, 8):
                rid = f"F.{i}"
                if rid in r.rows:
                    r.grade_row(rid, passed=passed,
                                threshold=c.voltage_linearity.r2_min, note=note)

        # Overall pass = all non-None entries pass.
        booleans = [v for v in summary.values() if v is not None]
        r.overall_pass = all(booleans) if booleans else False
        logger.info("Overall verdict: %s (%s)",
                    "PASS" if r.overall_pass else "FAIL",
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
        self.find_peak_xy(x0=0,y0=0)
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
