"""Persistence layer for characterization results.

Writes a completed :class:`TestReport` to disk in four representations:

- **XLSX** -- an app-compatible "Report" sheet (columns
  ``Index / Item / Value``) plus a "Figures" sheet with embedded PNGs
  and a "Grading" sheet with pass/fail annotations.
- **PDF** -- one figure per page via
  :class:`matplotlib.backends.backend_pdf.PdfPages`.
- **CSV bundle** -- one CSV per scan under ``raw/`` plus a flat
  ``summary.csv`` of every report row.
- **device_config.json** -- matches the schema consumed by
  ``openlifu_sdk.io.LIFUConfig`` (as used by the openlifu-test-app),
  with the ``module.sensitivity`` matrix computed from the freq
  sweep + a vendored ``FOCAL_GAIN_LUT``.

A single top-level helper :func:`write_report` bundles all of the
above into a timestamped output directory.
"""
from __future__ import annotations

import csv
import datetime
import json
import logging
import re
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np

from .characterization import (
    ROW,
    TestReport,
    _1d_scan_pnp_peak,
    _pnp_MPa,
)
from .scan_results import ScanResult

logger = logging.getLogger(__name__)

#: Shown on reports (and in the GUI) for runs that used any override.
NONSTANDARD_BANNER = "NON-STANDARD CONFIGURATION - not valid for release"


# ----------------------------------------------------------------------
# Vendored FOCAL_GAIN_LUT from openlifu-test-app so we don't need to
# import that package. Values / axes verbatim from
# ``openlifu-test-app/test_reports/test_reports.py``.
# ----------------------------------------------------------------------
_FOCAL_GAIN_F0_HZ = np.array([
    130000.0, 135000.0, 140000.0, 145000.0, 150000.0,
    155000.0, 160000.0, 165000.0,
    375000.0, 380000.0, 385000.0, 390000.0, 395000.0,
    400000.0, 405000.0, 410000.0,
])

_FOCAL_GAIN_CROSSTALK = np.array([
    0.0, 0.05, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
])

_FOCAL_GAIN_DATA = np.array([
    [2.807589054, 3.228639126, 3.649686813, 3.818109274, 4.070739746, 4.491786957,
     4.912837505, 5.333885670, 5.754938126, 6.175983429, 6.597033501, 7.018085003],
    [2.904339314, 3.332431316, 3.760524273, 3.931760550, 4.188616753, 4.616710663,
     5.044803143, 5.472893715, 5.900986671, 6.329079151, 6.757172108, 7.185482502],
    [2.990927696, 3.428293467, 3.865659714, 4.040605068, 4.303024292, 4.740390778,
     5.177754402, 5.615119934, 6.052487373, 6.489851952, 6.927217484, 7.364583969],
    [3.077177286, 3.520135403, 3.964356184, 4.142045498, 4.408576965, 4.852799416,
     5.297021866, 5.741242409, 6.185462475, 6.629685879, 7.073910713, 7.518129349],
    [3.170368195, 3.617199183, 4.064029694, 4.242762566, 4.511042118, 4.961826801,
     5.412615299, 5.863399506, 6.314184666, 6.764969826, 7.215755463, 7.666543961],
    [3.242729664, 3.697956562, 4.153182507, 4.335274696, 4.608408928, 5.064931870,
     5.521984100, 5.979033947, 6.436083317, 6.893132687, 7.350184441, 7.807235718],
    [3.327850103, 3.783803701, 4.245124340, 4.429731846, 4.706640720, 5.168155193,
     5.629670620, 6.091186047, 6.552703381, 7.014214993, 7.475728989, 7.937244415],
    [3.415055990, 3.878386736, 4.341715336, 4.527048111, 4.805045605, 5.268375397,
     5.731706142, 6.195037842, 6.658364296, 7.121839523, 7.587955475, 8.054073334],
    [5.705652714, 5.966911793, 6.233223438, 6.343565941, 6.509076118, 6.784928799,
     7.060781479, 7.336633682, 7.612486362, 7.888339520, 8.164192200, 8.447998047],
    [5.738934040, 5.998416424, 6.260423183, 6.365228653, 6.522432327, 6.784440994,
     7.046449184, 7.310236931, 7.583878517, 7.857521534, 8.131163597, 8.404805183],
    [5.780664921, 6.028132915, 6.275601387, 6.374588013, 6.523068905, 6.777853966,
     7.039001465, 7.300150394, 7.561298370, 7.822445869, 8.083595276, 8.350235939],
    [5.814091206, 6.046409130, 6.284290314, 6.383488178, 6.532283306, 6.780277252,
     7.028269291, 7.276263714, 7.524255753, 7.772250175, 8.040055275, 8.318523407],
    [5.836524487, 6.067535400, 6.301788807, 6.395490170, 6.536042213, 6.770293713,
     7.004545689, 7.238796711, 7.482921600, 7.740076542, 8.005291939, 8.270505905],
    [5.868018150, 6.088025570, 6.308032990, 6.396038055, 6.528041840, 6.749823093,
     6.984026432, 7.218224049, 7.452858448, 7.704096317, 7.955334663, 8.206572533],
    [5.892360210, 6.097702980, 6.303044319, 6.390904903, 6.523708344, 6.745048046,
     6.966385841, 7.187726021, 7.417387962, 7.661881924, 7.913189411, 8.164498329],
    [5.906170368, 6.104805946, 6.312875748, 6.396102428, 6.520944595, 6.729012966,
     6.937082291, 7.157343388, 7.395428181, 7.633512974, 7.871599674, 8.109683037],
])


def _focal_gain(freq_kHz: float, crosstalk_frac: float) -> float:
    """Bilinear-interpolate FOCAL_GAIN_LUT at (freq_Hz, crosstalk_frac)."""
    f_Hz = float(freq_kHz) * 1e3
    xs = _FOCAL_GAIN_F0_HZ
    ys = _FOCAL_GAIN_CROSSTALK
    # bilinear interp
    ix = int(np.clip(np.searchsorted(xs, f_Hz) - 1, 0, len(xs) - 2))
    iy = int(np.clip(np.searchsorted(ys, crosstalk_frac) - 1, 0, len(ys) - 2))
    x0, x1 = xs[ix], xs[ix + 1]
    y0, y1 = ys[iy], ys[iy + 1]
    tx = 0.0 if x1 == x0 else (f_Hz - x0) / (x1 - x0)
    ty = 0.0 if y1 == y0 else (crosstalk_frac - y0) / (y1 - y0)
    q = _FOCAL_GAIN_DATA
    v = ((1 - tx) * (1 - ty) * q[ix, iy]
         + tx * (1 - ty) * q[ix + 1, iy]
         + (1 - tx) * ty * q[ix, iy + 1]
         + tx * ty * q[ix + 1, iy + 1])
    return float(v)


# ----------------------------------------------------------------------
# LIFU module templates (mirrors openlifu-test-app / SDK LIFUConfig).
# ----------------------------------------------------------------------
LIFU_MODULES = {
    400: {"id": "txm_400_{sn}", "name": "TXM 400kHz (S/N {sn})",
          "nx": 8, "ny": 8, "pitch": 5, "frequency": 400e3, "kerf": 0.3,
          "crosstalk_frac": 0.12, "crosstalk_dist": 5.05e-3},
    155: {"id": "txm_155_{sn}", "name": "TXM 155kHz (S/N {sn})",
          "nx": 8, "ny": 8, "pitch": 5, "frequency": 155e3, "kerf": 0.3,
          "crosstalk_frac": 0.12, "crosstalk_dist": 5.05e-3},
}


# ----------------------------------------------------------------------
# Section metadata for the "Report" sheet.
# ----------------------------------------------------------------------
SECTION_HEADERS = [
    ("A", "Test Information"),
    ("B", "Transmit Module"),
    ("C", "Console"),
    ("D", "Peak Scans"),
    ("E", "Frequency Sweep"),
    ("F", "Voltage Sweep"),
]


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------
def _pnp_MPa_grid(traces: np.ndarray) -> np.ndarray:
    """Reduce a 2-D or higher trace array to per-point PNP in MPa.

    ``traces`` last axis is time. Returns array with that axis removed.
    """
    arr = np.asarray(traces, dtype=float)
    return -arr.min(axis=-1) / 1e6


def _sanitize_stem(name: str) -> str:
    """Filesystem-safe stem for a serial number / path fragment."""
    return re.sub(r"[^A-Za-z0-9\-_]+", "_", str(name)).strip("_") or "unknown"


def _ts_for_filename(iso: str) -> str:
    """Format an ISO timestamp as ``YYYYMMDD_HHMMSS`` for filenames.

    Falls back to the current wall-clock time if ``iso`` is empty or
    unparseable so callers can always build a stem without extra
    guarding.
    """
    if iso:
        try:
            return datetime.datetime.fromisoformat(iso).strftime("%Y%m%d_%H%M%S")
        except ValueError:
            pass
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def report_file_stem(report: "TestReport") -> str:
    """Return the ``<SN>_<YYYYMMDD_HHMMSS>`` prefix used for report files.

    The timestamp comes from ``report.started_at`` so the on-disk
    filename matches the test start time (not the write time), and so
    the top-level files inside a run directory are self-describing
    even if moved elsewhere.
    """
    sn_row = report.rows.get(ROW["txm_sn"])
    sn = _sanitize_stem(sn_row.value if sn_row else "unknown")
    ts = _ts_for_filename(report.started_at)
    return f"{sn}_{ts}"


# ----------------------------------------------------------------------
# Figure builders
# ----------------------------------------------------------------------
def build_figures(report: TestReport) -> dict:
    """Build a dict of matplotlib figures keyed by artifact name.

    The insertion order (scan_2d \u2192 lateral \u2192 elevation \u2192
    waveform \u2192 freq \u2192 voltage) matches the "see Figure N"
    cross-references written into the D-section rows by
    :meth:`Characterization.measure_waveform_at_peak`. Figure
    titles carry the same "Figure N:" prefix so the PDF page order
    reads the way the row grid says it should. The plane-wave and
    focused depth calibrations follow as Figures 8 and 9.

    Where a figure backs a graded row, it draws the acceptance limit
    from ``report.criteria`` and marks the graded point. The title's
    second line shows the row's PASS/FAIL and the margin to the limit.
    """
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    figs: dict = {}
    scans = report.scans
    crit = report.criteria
    freq_kHz = float(report.frequency_kHz)

    def _status(*row_keys: str) -> str:
        """Combined verdict of the given rows ("" if ungraded)."""
        rows = [report.rows.get(ROW[k]) for k in row_keys]
        statuses = [r.status for r in rows if r is not None]
        if "FAIL" in statuses:
            return "FAIL"
        if statuses and all(s == "PASS" for s in statuses):
            return "PASS"
        return ""

    def _title(ax, text: str, status: str = "", detail: str = "") -> None:
        second = " \u2014 ".join(p for p in (status, detail) if p)
        color = {"PASS": "darkgreen", "FAIL": "firebrick"}.get(status, "black")
        ax.set_title(f"{text}\n{second}" if second else text,
                     fontsize=9, color=color)

    def _plot_pulses(ax, t_us, pulses, scale: float = 1.0) -> None:
        """Grey overlay of the aligned per-pulse traces (if > 1)."""
        if pulses is None:
            return
        pulses = np.asarray(pulses, dtype=float)
        if pulses.ndim != 2 or pulses.shape[0] < 2:
            return
        for i, p in enumerate(pulses * scale):
            ax.plot(t_us, p, color="0.6", lw=0.5, alpha=0.35,
                    label=f"aligned pulses (n={pulses.shape[0]})"
                    if i == 0 else None)

    # Helper: figure out the origin for a 1-D scan's Δ-axis label. In
    # relative mode (Characterization always runs relative) the origin
    # is the calibrated hydrophone position; in absolute mode it's 0.
    def _scan_origin(s, axis_index: int) -> float:
        meta = getattr(s, "metadata", {}) or {}
        if meta.get("absolute", False):
            return 0.0
        pos = meta.get("hydrophone_position_mm")
        if pos is None:
            return 0.0
        return float(np.asarray(pos)[axis_index])

    # Figure 1: 2-D XY heatmap.
    s2d = scans.get("scan_2d")
    if s2d is not None:
        xs = s2d.coords.get("xfoci")
        ys = s2d.coords.get("yfoci")
        pnp_kPa = _pnp_MPa_grid(s2d.traces) * 1000.0
        extent = (float(xs[0]), float(xs[-1]), float(ys[0]), float(ys[-1]))
        fig, ax = plt.subplots(figsize=(5, 4.6))
        im = ax.imshow(pnp_kPa, extent=extent,
                       origin="lower", aspect="equal", cmap="viridis")
        ax.plot(0, 0, "r+", ms=14, mew=2, label="located peak")
        detail = ""
        if crit is not None:
            # The scan is relative to the located peak, so the nominal
            # center (0, 0) sits at (-px, -py) in these coordinates.
            px, py = (float(v) for v in report.peak_xy_mm)
            lim = float(crit.peak_offset.max_axis_mm)
            ax.add_patch(Rectangle((-px - lim, -py - lim), 2 * lim, 2 * lim,
                                   fill=False, ec="w", ls="--", lw=1.2,
                                   label=f"pass |x|, |y| \u2264 {lim:g} mm"))
            ax.plot(-px, -py, "wx", ms=8, mew=1.5, label="nominal center")
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
            detail = (f"peak ({px:+.2f}, {py:+.2f}) mm, "
                      f"margin {lim - max(abs(px), abs(py)):+.2f} mm")
        ax.legend(loc="upper right", fontsize=7)
        ax.set_xlabel("X offset (mm)")
        ax.set_ylabel("Y offset (mm)")
        _title(ax, f"Figure 1: 2-D XY PNP map @ {report.frequency_kHz:.0f} kHz, "
                   f"{report.voltage_V:.0f} V",
               _status("peak_x_mm", "peak_y_mm"), detail)
        fig.colorbar(im, ax=ax, label="PNP (kPa)")
        fig.tight_layout()
        figs["scan_2d"] = fig

    # Figures 2 / 3: 1-D lateral (x) / elevation (y) sweeps.
    def _1d_offset_fig(scan, axis: str, axis_index: int, color: str,
                       title: str, row_key: str):
        # Same reduction the grader uses, so the marked peak is the
        # point that was actually graded.
        cs, pnp_MPa, ipk = _1d_scan_pnp_peak(scan)
        pnp_kPa = pnp_MPa * 1000.0
        c0 = _scan_origin(scan, axis_index)
        fig, ax = plt.subplots(figsize=(6, 3.4))
        detail = ""
        if crit is not None:
            lim = float(crit.peak_offset.max_1d_axis_mm_for(freq_kHz))
            ax.axvspan(-lim, lim, color="tab:green", alpha=0.12,
                       label=f"pass |\u0394{axis}| \u2264 {lim:g} mm")
            detail = (f"offset {cs[ipk]:+.2f} mm, "
                      f"margin {lim - abs(cs[ipk]):+.2f} mm")
        ax.plot(cs, pnp_kPa, "o-", lw=1.5, color=color)
        ax.plot(cs[ipk], pnp_kPa[ipk], "r*", ms=12,
                label=f"peak \u0394{axis} = {cs[ipk]:+.2f} mm")
        ax.legend(loc="upper right", fontsize=8)
        ax.axvline(0.0, color="0.7", lw=0.7)
        ax.set_xlabel(f"\u0394{axis} ({axis}\u2080 = {c0:.2f} mm, 2-D peak)")
        ax.set_ylabel("PNP (kPa)")
        _title(ax, f"{title} @ {report.frequency_kHz:.0f} kHz, "
                   f"{report.voltage_V:.0f} V", _status(row_key), detail)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    lat = scans.get("lateral_1d")
    if lat is not None:
        figs["lateral_1d"] = _1d_offset_fig(
            lat, "x", 0, "tab:blue", "Figure 2: 1-D Lateral scan",
            "lateral_peak_offset_mm")

    elev = scans.get("elevation_1d")
    if elev is not None:
        figs["elevation_1d"] = _1d_offset_fig(
            elev, "y", 1, "tab:orange", "Figure 3: 1-D Elevation scan",
            "elevation_peak_offset_mm")

    # Figure 4: 1-D axial (z sweep).
    axi = scans.get("axial_1d")
    if axi is not None:
        zs = axi.coords.get("zfoci")
        traces = axi.traces
        traces_2d = np.atleast_2d(traces)
        if traces_2d.ndim == 3 and traces_2d.shape[1] == 1:
            traces_2d = traces_2d[:, 0, :]
        pnp_kPa = _pnp_MPa_grid(traces_2d) * 1000.0
        zs = np.ravel(np.asarray(zs, dtype=float))
        z0 = _scan_origin(axi, 2)
        fig, ax = plt.subplots(figsize=(6, 3.4))
        detail = ""
        depth_row = report.rows.get(ROW["plane_wave_depth_mm"])
        focus_row = report.rows.get(ROW["peak_z_focus_mm"])
        depth = None if depth_row is None else depth_row.value
        focus = None if focus_row is None else focus_row.value
        if depth is not None and np.isfinite(depth):
            dz = float(depth) - z0
            if crit is not None:
                tol = float(depth) * float(crit.peak_depth.peak_z_focus_tol_pct) / 100.0
                ax.axvspan(dz - tol, dz + tol, color="tab:green", alpha=0.12,
                           label=f"focus pass \u00b1{tol:.2f} mm")
                if focus is not None and np.isfinite(focus):
                    detail = (f"focus {float(focus):.2f} mm vs depth {float(depth):.2f} mm, "
                              f"margin {tol - abs(float(focus) - float(depth)):+.2f} mm")
            ax.axvline(dz, color="tab:red", lw=0.9,
                       label=f"plane-wave depth {float(depth):.2f} mm")
        if focus is not None and np.isfinite(focus):
            ax.axvline(float(focus) - z0, color="k", ls="--", lw=0.9,
                       label=f"commanded focus {float(focus):.2f} mm")
        ax.plot(zs, pnp_kPa, "o-", lw=1.5, color="tab:green")
        if pnp_kPa.size == zs.size and pnp_kPa.size:
            ipk = int(np.argmax(pnp_kPa))
            ax.plot(zs[ipk], pnp_kPa[ipk], "r*", ms=12,
                    label=f"axial peak \u0394z = {zs[ipk]:+.2f} mm")
        ax.legend(loc="lower right", fontsize=7)
        ax.set_xlabel(f"\u0394z (z\u2080 = {z0:.2f} mm)")
        ax.set_ylabel("PNP (kPa)")
        _title(ax, f"Figure 4: 1-D Axial scan @ {report.frequency_kHz:.0f} kHz, "
                   f"{report.voltage_V:.0f} V", _status("peak_z_focus_mm"), detail)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        figs["axial_1d"] = fig

    # Figure 5: Waveform at peak.
    wf = report.waveform_at_peak
    if wf:
        t_us = np.asarray(wf["t"], dtype=float)
        trace = np.asarray(wf["trace"], dtype=float)
        in_pa = wf.get("units", "Pa") == "Pa"
        if in_pa:
            scale = 1e-3
            ylabel = "Pressure (kPa)"
        else:
            scale = 1.0
            ylabel = f"Amplitude ({wf.get('units', 'a.u.')})"
        y = trace * scale
        pnp_V = float(wf.get("pnp_voltage_V", report.voltage_V))
        pnp_MPa = float(wf.get("pnp_MPa", float("nan")))
        fig, ax = plt.subplots(figsize=(6.5, 3.6))
        _plot_pulses(ax, t_us, wf.get("pulse_traces"), scale)
        ax.plot(t_us, y, lw=0.9, color="tab:blue", label="median")
        detail = ""
        if in_pa and y.size:
            i_min = int(np.argmin(y))
            ax.plot(t_us[i_min], y[i_min], "rv", ms=7,
                    label=f"PNP {pnp_MPa:.3f} MPa")
            if crit is not None:
                ref_V = float(crit.pnp_at_peak.reference_voltage_V)
                thr_MPa = crit.pnp_min_MPa_for(freq_kHz) * pnp_V / ref_V
                ax.axhline(-thr_MPa * 1e3, color="firebrick", ls="--", lw=0.9,
                           label=f"pass: PNP \u2265 {thr_MPa:.3f} MPa")
                detail = (f"PNP {pnp_MPa:.3f} vs \u2265 {thr_MPa:.3f} MPa, "
                          f"margin {(pnp_MPa / thr_MPa - 1) * 100:+.1f}%")
        arrival_us = wf.get("arrival_us")
        if arrival_us is not None and np.isfinite(arrival_us):
            ax.axvline(float(arrival_us), color="tab:red", lw=0.8,
                       label=f"focused arrival {float(arrival_us):.2f} us")
        ax.legend(loc="upper right", fontsize=7)
        ax.set_xlabel("Time (us)")
        ax.set_ylabel(ylabel)
        _title(ax, f"Figure 5: Waveform at peak, {pnp_V:g} V",
               _status("pnp_at_peak_MPa"), detail)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        figs["waveform_at_peak"] = fig

    # Figure 6: Frequency response.
    fr = report.freq_response
    if fr:
        freqs = np.asarray(fr["frequencies_kHz"], dtype=float)
        pnp_kPa = np.asarray(fr["pnp_MPa"], dtype=float) * 1000.0
        fig, ax = plt.subplots(figsize=(6, 3.4))
        ax.plot(freqs, pnp_kPa, "o-", lw=1.5)
        detail = ""
        if pnp_kPa.size and pnp_kPa.size == freqs.size:
            # Same selection as ``_grade_freq_response``.
            ipk = int(np.argmax(pnp_kPa))
            inom = int(np.argmin(np.abs(freqs - freq_kHz)))
            peak, nom = float(pnp_kPa[ipk]), float(pnp_kPa[inom])
            ax.plot(freqs[ipk], peak, "r*", ms=12,
                    label=f"peak {freqs[ipk]:.0f} kHz")
            ax.plot(freqs[inom], nom, "s", mfc="none", mec="k", ms=10,
                    label=f"nominal {freqs[inom]:.0f} kHz")
            if crit is not None and peak > 0:
                max_dev = float(crit.freq_response.max_deviation_pct)
                ax.axhline(peak * (1 - max_dev / 100.0), color="firebrick",
                           ls="--", lw=0.9,
                           label=f"pass: nominal \u2265 {100 - max_dev:g}% of peak")
                dev = (peak - nom) / peak * 100.0
                detail = (f"deviation {dev:.1f}% vs \u2264 {max_dev:g}%, "
                          f"margin {max_dev - dev:+.1f} pts")
        ax.set_xlabel("Frequency (kHz)")
        ax.set_ylabel("PNP (kPa)")
        _title(ax, f"Figure 6: Frequency response @ {report.voltage_V:.0f} V",
               _status("freq_deviation_pct"), detail)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
        fig.tight_layout()
        figs["freq_response"] = fig

    # Figure 7: Voltage linearity.
    vr = report.voltage_response
    if vr:
        volts = np.asarray(vr["voltages_V"], dtype=float)
        pnp_kPa = np.asarray(vr["pnp_MPa"], dtype=float) * 1000.0
        # Report stores slope/intercept in MPa; scale to kPa for display.
        slope_kPa_per_V = float(vr.get("slope_MPa_per_V", 0.0)) * 1000.0
        intercept_kPa = float(vr.get("intercept_MPa", 0.0)) * 1000.0
        r2 = float(vr.get("r2", float("nan")))
        fit = slope_kPa_per_V * volts + intercept_kPa
        fig, (ax, ax_res) = plt.subplots(
            2, 1, figsize=(6, 4.2), sharex=True,
            gridspec_kw={"height_ratios": (3, 1)})
        ax.plot(volts, pnp_kPa, "o", label="measured")
        ax.plot(volts, fit, "-", lw=1, label=f"fit (R\u00b2 = {r2:.4f})")
        detail = ""
        if crit is not None:
            r2_min = float(crit.voltage_linearity.r2_min)
            detail = (f"R\u00b2 {r2:.4f} vs \u2265 {r2_min:g}, "
                      f"margin {r2 - r2_min:+.4f}")
        ax.set_ylabel("PNP (kPa)")
        _title(ax, f"Figure 7: Voltage linearity @ {report.frequency_kHz:.0f} kHz",
               _status("voltage_r2"), detail)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
        ax_res.axhline(0.0, color="0.6", lw=0.7)
        ax_res.plot(volts, pnp_kPa - fit, "o-", ms=4, lw=0.8)
        ax_res.set_xlabel("HV rail (V)")
        ax_res.set_ylabel("residual (kPa)")
        ax_res.grid(True, alpha=0.3)
        fig.tight_layout()
        figs["voltage_linearity"] = fig

    # Figures 8 / 9: plane-wave and focused depth calibrations.
    def _depth_cal_fig(dc: dict, title: str, row_key: str, tol_mm):
        t_us = np.asarray(dc["t_us"], dtype=float)
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        skip_us = dc.get("skip_us")
        if skip_us is not None:
            ax.axvspan(t_us[0], skip_us, color="0.85", alpha=0.6,
                       label="skipped (TX pickup)")
            bw = dc.get("baseline_window_us")
            if bw:
                ax.axvspan(skip_us, skip_us + bw, color="gold", alpha=0.2,
                           label="baseline window")
        detail = ""
        if crit is not None and tol_mm is not None:
            # Depth window mapped to arrival time:
            # arrival = depth / SoS + max_delay.
            nominal = float(crit.peak_depth.nominal_for(freq_kHz))
            sos = float(dc["sos_m_per_s"])
            max_delay = float(dc.get("max_delay_us", 0.0))
            ax.axvspan((nominal - tol_mm) * 1e3 / sos + max_delay,
                       (nominal + tol_mm) * 1e3 / sos + max_delay,
                       color="tab:green", alpha=0.15,
                       label=f"pass {nominal:g} \u00b1 {tol_mm:.2f} mm")
            depth = float(dc["distance_mm"])
            detail = (f"depth {depth:.2f} mm vs {nominal:g} \u00b1 {tol_mm:.2f} mm, "
                      f"margin {tol_mm - abs(depth - nominal):+.2f} mm")
        _plot_pulses(ax, t_us, dc.get("pulse_traces"))
        ax.plot(t_us, np.asarray(dc["mean_trace"], dtype=float), lw=0.9,
                color="tab:blue",
                label=f"median ({dc.get('n_pulses_used', '?')} pulses)")
        if dc.get("threshold") is not None:
            ax.axhline(float(dc["threshold"]), color="tab:orange", ls=":",
                       lw=0.9, label="pick threshold")
        if dc.get("first_peak_us") is not None:
            ax.plot(dc["first_peak_us"], dc.get("first_peak_value", 0.0),
                    "ro", ms=5, label=f"first peak {dc['first_peak_us']:.2f} us")
        ax.axvline(float(dc["arrival_us"]), color="tab:red", lw=0.9,
                   label=f"arrival {float(dc['arrival_us']):.2f} us")
        ax.legend(loc="upper right", fontsize=6.5, ncol=2)
        ax.set_xlabel("Time from emission (us)")
        ax.set_ylabel("Hydrophone (mV)")
        _title(ax, f"{title} @ {report.frequency_kHz:.0f} kHz, "
                   f"{report.voltage_V:.0f} V", _status(row_key), detail)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    depth_crit = crit.peak_depth if crit is not None else None
    if report.depth_cal_plane_wave:
        figs["depth_cal_plane_wave"] = _depth_cal_fig(
            report.depth_cal_plane_wave,
            "Figure 8: Plane-wave depth calibration", "plane_wave_depth_mm",
            None if depth_crit is None else float(depth_crit.hydrophone_tol_mm))
    if report.depth_cal_focused:
        tol = None
        if depth_crit is not None:
            tol = (float(depth_crit.nominal_for(freq_kHz))
                   * float(depth_crit.focused_tol_pct) / 100.0)
        figs["depth_cal_focused"] = _depth_cal_fig(
            report.depth_cal_focused,
            "Figure 9: Focused depth calibration", "focused_depth_mm", tol)

    return figs


# ----------------------------------------------------------------------
# Row-ordering utility used by both the XLSX and CSV writers.
# ----------------------------------------------------------------------
def _iter_rows_in_order(report: TestReport):
    """Yield ``(id, ReportRow)`` sorted by section (A..F) then subrow."""
    def _key(id_: str):
        if "." not in id_:
            return (id_, 0)
        letter, num = id_.split(".", 1)
        try:
            return (letter, int(num))
        except ValueError:
            return (letter, 999)
    for id_ in sorted(report.rows.keys(), key=_key):
        yield id_, report.rows[id_]


# ----------------------------------------------------------------------
# XLSX writer (app-compatible "Report" sheet)
# ----------------------------------------------------------------------
def write_xlsx(report: TestReport, path: Path, figures: Optional[dict] = None) -> Path:
    """Write ``report`` to ``path`` (openpyxl workbook)."""
    from openpyxl import Workbook
    from openpyxl.drawing.image import Image as XLImage
    from openpyxl.styles import Alignment, Font, PatternFill

    path = Path(path)
    wb = Workbook()

    # -- Report sheet -------------------------------------------------
    ws = wb.active
    ws.title = "Report"
    ws.append(["Index", "Item", "Value"])
    for c in ws[1]:
        c.font = Font(bold=True)

    hdr_fill = PatternFill(start_color="D9E1F2", end_color="D9E1F2",
                           fill_type="solid")
    subhdr_font = Font(bold=True, italic=True)
    section_rows: dict[str, int] = {}
    rows_by_section: dict[str, list] = {}
    for id_, row in _iter_rows_in_order(report):
        letter = id_.split(".", 1)[0]
        rows_by_section.setdefault(letter, []).append((id_, row))

    for letter, title in SECTION_HEADERS:
        ws.append([letter, title, ""])
        section_rows[letter] = ws.max_row
        for c in ws[ws.max_row]:
            c.fill = hdr_fill
            c.font = Font(bold=True)
        # Inner header row — the app-side reader
        # (openlifu-test-app/test_reports/test_reports.py) uses this as
        # the pandas ``header`` row for each section, so it MUST read
        # "Item" / "Value" in columns B / C.
        ws.append(["", "Item", "Value"])
        for c in ws[ws.max_row]:
            c.font = subhdr_font
        for id_, row in rows_by_section.get(letter, []):
            value = row.value
            if isinstance(value, (float, np.floating)) and not np.isnan(value):
                value = float(value)
            elif isinstance(value, (np.integer,)):
                value = int(value)
            elif value is None or (isinstance(value, float) and np.isnan(value)):
                value = ""
            ws.append([id_, row.label, value])
        ws.append(["", "", ""])  # blank spacer

    ws.column_dimensions["A"].width = 8
    ws.column_dimensions["B"].width = 30
    ws.column_dimensions["C"].width = 30

    # -- Grading sheet ----------------------------------------------
    gws = wb.create_sheet("Grading")
    gws.append(["Index", "Item", "Value", "Unit", "Status",
                "Threshold", "Note"])
    for c in gws[1]:
        c.font = Font(bold=True)
    for id_, row in _iter_rows_in_order(report):
        val = row.value
        if isinstance(val, (float, np.floating)) and not np.isnan(val):
            val = float(val)
        elif val is None or (isinstance(val, float) and np.isnan(val)):
            val = ""
        elif isinstance(val, (list, tuple, dict, np.ndarray)):
            val = str(val)
        thr = row.threshold
        if isinstance(thr, (list, tuple, dict, np.ndarray)):
            thr = str(thr)
        gws.append([id_, row.label, val, row.unit, row.status,
                    "" if thr is None else thr, row.note])
    # Status column formatting.
    status_col = 5
    green = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
    red = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
    grey = PatternFill(start_color="EEEEEE", end_color="EEEEEE", fill_type="solid")
    for r_i in range(2, gws.max_row + 1):
        cell = gws.cell(row=r_i, column=status_col)
        if cell.value == "PASS":
            cell.fill = green
        elif cell.value == "FAIL":
            cell.fill = red
        elif cell.value == "NA":
            cell.fill = grey
    gws.column_dimensions["A"].width = 8
    gws.column_dimensions["B"].width = 30
    for col in ("C", "D", "E", "F", "G"):
        gws.column_dimensions[col].width = 18

    # -- Summary sheet ---------------------------------------------
    sws = wb.create_sheet("Summary")
    sws.append(["Key", "Value"])
    for c in sws[1]:
        c.font = Font(bold=True)
    sws.append(["Overall verdict", "PASS" if report.overall_pass else "FAIL"])
    sws.append(["Configuration", NONSTANDARD_BANNER
                if report.nonstandard_reasons else "Standard"])
    if report.nonstandard_reasons:
        cfg_cell = sws.cell(row=sws.max_row, column=2)
        cfg_cell.fill = red
        cfg_cell.font = Font(bold=True)
        for reason in report.nonstandard_reasons:
            sws.append(["Non-standard setting", reason])
    sws.append(["Started at", report.started_at])
    sws.append(["Finished at", report.finished_at])
    sws.append(["Peak XY (mm)", f"({report.peak_xy_mm[0]:.4f}, "
                                f"{report.peak_xy_mm[1]:.4f})"])
    if report.voltage_response:
        sws.append(["Voltage linearity R**2",
                    f"{report.voltage_response.get('r2', float('nan')):.4f}"])
    verdict_cell = sws.cell(row=2, column=2)
    verdict_cell.fill = green if report.overall_pass else red
    verdict_cell.font = Font(bold=True)
    verdict_cell.alignment = Alignment(horizontal="center")
    sws.column_dimensions["A"].width = 28
    sws.column_dimensions["B"].width = 40

    # -- Figures sheet ---------------------------------------------
    if figures:
        fws = wb.create_sheet("Figures")
        anchor_row = 1
        # Save PNGs to a temp directory next to the xlsx.
        img_dir = path.parent / "figures"
        img_dir.mkdir(parents=True, exist_ok=True)
        for name, fig in figures.items():
            png_path = img_dir / f"{name}.png"
            fig.savefig(png_path, dpi=150)
            img = XLImage(str(png_path))
            fws.cell(row=anchor_row, column=1, value=name).font = Font(bold=True)
            fws.add_image(img, f"A{anchor_row + 1}")
            # Advance ~ (image height / 15px per row) rows plus a gap.
            anchor_row += 28

    wb.save(path)
    logger.info("XLSX written to %s", path)
    return path


# ----------------------------------------------------------------------
# PDF writer
# ----------------------------------------------------------------------
def write_pdf(report: TestReport, path: Path,
              figures: Optional[dict] = None) -> Path:
    """Write a multi-page PDF: cover page + one page per figure.

    Every page is rendered on 8.5\u00d711 in (US letter) so a print
    stack has consistent paper. Figures keep their original aspect
    ratio; each source figure is rasterised at 200 dpi and centered
    on the letter page under its title.
    """
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt

    path = Path(path)
    figures = figures or {}
    # Build a letter -> section title map so we can insert a header
    # line every time the section changes in the cover-page listing.
    section_titles = {letter: title for letter, title in SECTION_HEADERS}
    with PdfPages(path) as pdf:
        # Cover page (already letter-sized).
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        lines = [
            "TXM Characterization Report",
            "",
            f"Overall verdict: {'PASS' if report.overall_pass else 'FAIL'}",
            f"Date: {report.started_at}   Finished: {report.finished_at}",
            "",
        ]
        if report.nonstandard_reasons:
            lines[2:2] = [f"*** {NONSTANDARD_BANNER} ***",
                          *(f"    {r}" for r in report.nonstandard_reasons),
                          ""]
        current_letter: Optional[str] = None
        for id_, row in _iter_rows_in_order(report):
            letter = id_.split(".", 1)[0]
            if letter != current_letter:
                if current_letter is not None:
                    lines.append("")
                title = section_titles.get(letter, "")
                lines.append(f"-- Section {letter}: {title} --")
                current_letter = letter
            val = row.value
            if isinstance(val, float):
                val = f"{val:.4g}"
            unit_str = f" {row.unit}" if row.unit else ""
            status = f"  [{row.status}]" if row.status in ("PASS", "FAIL") else ""
            lines.append(f"  {id_:<6} {row.label:<28} {val}{unit_str}{status}")
        # Draw as monospace.
        ax.text(0.02, 0.98, "\n".join(lines), family="monospace",
                fontsize=8, va="top", ha="left")
        pdf.savefig(fig)
        plt.close(fig)

        # Figure pages: render each source figure onto its own
        # letter-sized page so the printout is uniform.
        for name, src_fig in figures.items():
            page = _render_figure_on_letter_page(src_fig, name)
            pdf.savefig(page)
            plt.close(page)
    logger.info("PDF written to %s", path)
    return path


def _render_figure_on_letter_page(src_fig, name: str):
    """Return a fresh 8.5\u00d711 figure with ``src_fig`` embedded.

    Rasterises ``src_fig`` (at 200 dpi with a ``bbox_inches='tight'``
    crop) and centers it on a letter page under the figure name.
    Rasterising decouples the source's aspect ratio from the page
    aspect, so slim landscape scan plots and taller waveform plots
    both fit cleanly.
    """
    import io
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    src_fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    img = plt.imread(buf)

    page = plt.figure(figsize=(8.5, 11))
    # Leave ~0.5" top for the caption, 0.5" side margins, 0.5" bottom.
    ax = page.add_axes((0.05, 0.05, 0.9, 0.88))
    ax.imshow(img)
    ax.axis("off")
    page.suptitle(name, fontsize=10, y=0.97)
    return page


# ----------------------------------------------------------------------
# CSV bundle writer
# ----------------------------------------------------------------------
def write_csv_bundle(report: TestReport, out_dir: Path,
                     *, summary_name: str = "summary.csv") -> Path:
    """Write flat summary CSV + per-scan CSVs into ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(exist_ok=True)

    # Flat summary CSV (name is caller-supplied so top-level artifacts
    # can share a ``<TXM-SN>_...`` prefix).
    summary_path = out_dir / summary_name
    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["Index", "Item", "Value", "Unit", "Status",
                    "Threshold", "Note"])
        for id_, row in _iter_rows_in_order(report):
            val = row.value
            if isinstance(val, (list, tuple, dict, np.ndarray)):
                val = str(val)
            thr = row.threshold
            if isinstance(thr, (list, tuple, dict, np.ndarray)):
                thr = str(thr)
            w.writerow([id_, row.label, val, row.unit, row.status,
                        "" if thr is None else thr, row.note])
    logger.info("%s written to %s", summary_path.name, summary_path)

    # Per-scan CSVs (reduced to PNP-per-point + coords) and raw NPZs.
    for name, scan in report.scans.items():
        if not isinstance(scan, ScanResult):
            continue
        pnp = _pnp_MPa_grid(scan.traces)
        csv_path = out_dir / f"{name}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            coord_names = list(scan.coords.keys())
            w.writerow(coord_names + ["pnp_MPa"])
            # Flatten grid.
            grids = np.meshgrid(*scan.coords.values(), indexing="ij")
            flat = [g.ravel() for g in grids] + [pnp.ravel()]
            for values in zip(*flat):
                w.writerow(values)
        # Raw NPZ.
        try:
            scan.save(raw_dir / f"{name}.npz")
        except Exception as e:
            logger.warning("Could not save %s raw NPZ: %s", name, e)

    # Waveform at peak (single trace).
    wf = report.waveform_at_peak
    if wf:
        wf_path = out_dir / "waveform_at_peak.csv"
        with wf_path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["t_us", f"trace_{wf.get('units', 'a.u.')}"])
            for t_i, v_i in zip(wf["t"], wf["trace"]):
                w.writerow([float(t_i), float(v_i)])
        np.savez_compressed(
            raw_dir / "waveform_at_peak.npz",
            t=np.asarray(wf["t"]), trace=np.asarray(wf["trace"]),
            units=str(wf.get("units", "")),
            pnp_MPa=float(wf.get("pnp_MPa", float("nan"))),
            axial_depth_mm=float(wf.get("axial_depth_mm", float("nan"))),
        )

    return out_dir


# ----------------------------------------------------------------------
# device_config.json writer
# ----------------------------------------------------------------------
def build_device_config(report: TestReport,
                        *, module_id: int = 0) -> dict:
    """Assemble the JSON dict written to ``device_config.json``.

    Mirrors ``openlifu-test-app.test_reports.test_report_to_config`` /
    ``openlifu_sdk.io.LIFUConfig`` schema so the device app can consume
    it directly.
    """
    fr = report.freq_response
    if not fr:
        raise ValueError("Cannot build device_config: freq response missing.")
    freqs_kHz = np.asarray(fr["frequencies_kHz"], dtype=float)
    pnp_MPa = np.asarray(fr["pnp_MPa"], dtype=float)
    voltage = float(fr.get("voltage_V", report.voltage_V))

    # Pick module template.
    nominal = int(round(report.frequency_kHz))
    if nominal not in LIFU_MODULES:
        raise ValueError(f"No LIFU module template for {nominal} kHz. "
                         f"Supported: {sorted(LIFU_MODULES)}")
    module = dict(LIFU_MODULES[nominal])  # shallow copy

    # Compute sensitivity matrix.
    sensitivity: list[tuple[int, int]] = []
    crosstalk = float(module["crosstalk_frac"])
    for f_kHz, pnp in zip(freqs_kHz, pnp_MPa):
        gain = _focal_gain(float(f_kHz), crosstalk)
        if gain <= 0:
            continue
        sens = pnp * 1e6 / gain / voltage  # Pa per V
        sensitivity.append((int(round(f_kHz * 1e3)), int(round(sens))))
    module["sensitivity"] = sensitivity

    sn = report.rows.get(ROW["txm_sn"])
    sn_str = str(sn.value) if sn is not None else ""
    sn_clean = re.sub(r"[^A-Za-z0-9\-_]+", "", sn_str)
    module["id"] = module["id"].format(sn=sn_clean.lower() or "unknown")
    module["name"] = module["name"].format(sn=sn_clean or "unknown")

    info = report.device_info
    config = {
        "sn": sn_str,
        "hwid": info.txm_hwid if info else "",
        "freq": nominal,
        "fw_ver": info.txm_fw_version if info else "",
        "sdk_ver": info.sdk_version if info else "",
        "updated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "module": module,
        "device": {},
    }
    return config


def write_device_config(report: TestReport, path: Path,
                        *, module_id: int = 0) -> Path:
    """Serialize :func:`build_device_config` to ``path``."""
    path = Path(path)
    config = build_device_config(report, module_id=module_id)
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    logger.info("device_config.json written to %s", path)
    return path


# ----------------------------------------------------------------------
# Orchestrator
# ----------------------------------------------------------------------
def write_report(report: TestReport, output_dir: Optional[Path] = None,
                 *, reports_root: Optional[Path] = None,
                 run_dir_name: Optional[str] = None,
                 write_device_config_json: bool = True) -> Path:
    """Serialize a :class:`TestReport` into a timestamped subdirectory.

    Layout when ``output_dir`` is omitted::

        test_reports/<TXM-SN>/<YYYYMMDD>_<HHMMSS>/
            <TXM-SN>_Report.xlsx, <TXM-SN>_Report.pdf,
            <TXM-SN>_device_config.json, <TXM-SN>_summary.csv,
            <TXM-SN>_scan_config.json,
            *.csv, figures/*.png, raw/*.npz, operator_prefs_snapshot.json

    Args:
        report: The completed report.
        output_dir: Parent directory. Defaults to
            ``<reports_root>/<TXM-SN>/`` so per-device runs live
            side-by-side. Created if missing.
        reports_root: Root folder used when ``output_dir`` is
            omitted. Defaults to ``test_reports/`` in the current
            working directory.
        run_dir_name: Override the auto-generated subdirectory name
            (defaults to a ``YYYYMMDD_HHMMSS`` timestamp).
        write_device_config_json: Set ``False`` to skip the config
            JSON (e.g. if the freq sweep was skipped).

    Returns:
        The path to the run directory.
    """
    sn_row = report.rows.get(ROW["txm_sn"])
    sn = _sanitize_stem(sn_row.value if sn_row else "unknown")
    # Stem used for every top-level file in the run directory.
    # Baked in the test start timestamp so a report copied out of its
    # folder still identifies its device + run uniquely.
    file_stem = report_file_stem(report)

    if output_dir is None:
        root = Path(reports_root) if reports_root else Path.cwd() / "test_reports"
        output_dir = root / sn
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if run_dir_name is None:
        # Match the file stem's timestamp so the folder and files
        # agree on when the test started.
        run_dir_name = _ts_for_filename(report.started_at)
    run_dir = output_dir / run_dir_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Figures first (both xlsx + pdf reuse them).
    figures = build_figures(report)

    # All top-level artifacts share the ``<SN>_<TS>_...`` prefix so a
    # single file picked out of the folder is self-describing.
    report_stem = f"{file_stem}_Report"
    write_xlsx(report, run_dir / f"{report_stem}.xlsx", figures=figures)
    write_pdf(report, run_dir / f"{report_stem}.pdf", figures=figures)
    write_csv_bundle(report, run_dir, summary_name=f"{file_stem}_summary.csv")

    if write_device_config_json:
        try:
            write_device_config(report, run_dir / f"{file_stem}_device_config.json")
        except Exception as e:
            logger.warning("Skipping device_config.json: %s", e)

    # Snapshot operator prefs.
    if report.prefs is not None:
        try:
            (run_dir / "operator_prefs_snapshot.json").write_text(
                json.dumps(asdict(report.prefs), indent=2), encoding="utf-8"
            )
        except Exception as e:
            logger.warning("Could not snapshot operator prefs: %s", e)

    # Effective scan configuration (standard or overridden).
    if report.scan_config is not None:
        report.scan_config.save(run_dir / f"{file_stem}_scan_config.json")

    # Close figures we opened so they don't linger.
    try:
        import matplotlib.pyplot as plt
        for fig in figures.values():
            plt.close(fig)
    except Exception:
        pass

    logger.info("Run directory: %s", run_dir)
    return run_dir
