"""Persistence layer for characterization results.

Writes a completed :class:`TestReport` to disk in four representations:

- **XLSX** \u2014 an app-compatible "Report" sheet (columns
  ``Index / Item / Value``) plus a "Figures" sheet with embedded PNGs
  and a "Grading" sheet with pass/fail annotations.
- **PDF** \u2014 one figure per page via
  :class:`matplotlib.backends.backend_pdf.PdfPages`.
- **CSV bundle** \u2014 one CSV per scan under ``raw/`` plus a flat
  ``summary.csv`` of every report row.
- **device_config.json** \u2014 matches the schema consumed by
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
    FREQ_SWEEP_OFFSETS_KHZ,
    ROW,
    TestReport,
    _pnp_MPa,
)
from .scan_results import ScanResult

logger = logging.getLogger(__name__)


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
    reads the way the row grid says it should.
    """
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    figs: dict = {}
    scans = report.scans

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
        fig, ax = plt.subplots(figsize=(5, 4.2))
        im = ax.imshow(pnp_kPa,
                       extent=(float(xs[0]), float(xs[-1]),
                               float(ys[0]), float(ys[-1])),
                       origin="lower", aspect="equal", cmap="viridis")
        ax.plot(0, 0, "r+", ms=14, mew=2, label="peak (relative)")
        ax.set_xlabel("X offset (mm)")
        ax.set_ylabel("Y offset (mm)")
        ax.set_title(f"Figure 1: 2-D XY PNP map @ {report.frequency_kHz:.0f} kHz, "
                     f"{report.voltage_V:.0f} V")
        fig.colorbar(im, ax=ax, label="PNP (kPa)")
        fig.tight_layout()
        figs["scan_2d"] = fig

    # Figure 2: 1-D lateral (x sweep).
    lat = scans.get("lateral_1d")
    if lat is not None:
        xs = lat.coords.get("xfoci")
        traces = lat.traces
        pnp_kPa = _pnp_MPa_grid(traces) * 1000.0
        x0 = _scan_origin(lat, 0)
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.plot(xs, pnp_kPa, "o-", lw=1.5)
        ax.axvline(0.0, color="0.7", lw=0.7)
        ax.set_xlabel(f"\u0394x (x\u2080 = {x0:.2f} mm)")
        ax.set_ylabel("PNP (kPa)")
        ax.set_title(f"Figure 2: 1-D Lateral scan @ {report.frequency_kHz:.0f} kHz, "
                     f"{report.voltage_V:.0f} V")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        figs["lateral_1d"] = fig

    # Figure 3: 1-D elevation (y sweep).
    elev = scans.get("elevation_1d")
    if elev is not None:
        ys = elev.coords.get("yfoci")
        traces = elev.traces
        # scan_1d returns (N, T); older scan_lateral (num_x=1, num_y=N)
        # returned (N, 1, T). Handle both.
        traces_2d = np.atleast_2d(traces)
        if traces_2d.ndim == 3 and traces_2d.shape[1] == 1:
            traces_2d = traces_2d[:, 0, :]
        pnp_kPa = _pnp_MPa_grid(traces_2d) * 1000.0
        y0 = _scan_origin(elev, 1)
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.plot(ys, pnp_kPa, "o-", lw=1.5, color="tab:orange")
        ax.axvline(0.0, color="0.7", lw=0.7)
        ax.set_xlabel(f"\u0394y (y\u2080 = {y0:.2f} mm)")
        ax.set_ylabel("PNP (kPa)")
        ax.set_title(f"Figure 3: 1-D Elevation scan @ {report.frequency_kHz:.0f} kHz, "
                     f"{report.voltage_V:.0f} V")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        figs["elevation_1d"] = fig

    # Figure 4: 1-D axial (z sweep).
    axi = scans.get("axial_1d")
    if axi is not None:
        zs = axi.coords.get("zfoci")
        traces = axi.traces
        traces_2d = np.atleast_2d(traces)
        if traces_2d.ndim == 3 and traces_2d.shape[1] == 1:
            traces_2d = traces_2d[:, 0, :]
        pnp_kPa = _pnp_MPa_grid(traces_2d) * 1000.0
        z0 = _scan_origin(axi, 2)
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.plot(zs, pnp_kPa, "o-", lw=1.5, color="tab:green")
        ax.axvline(0.0, color="0.7", lw=0.7)
        ax.set_xlabel(f"\u0394z (z\u2080 = {z0:.2f} mm)")
        ax.set_ylabel("PNP (kPa)")
        ax.set_title(f"Figure 4: 1-D Axial scan @ {report.frequency_kHz:.0f} kHz, "
                     f"{report.voltage_V:.0f} V")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        figs["axial_1d"] = fig

    # Figure 5: Waveform at peak.
    wf = report.waveform_at_peak
    if wf:
        t_us = np.asarray(wf["t"], dtype=float)
        trace = np.asarray(wf["trace"], dtype=float)
        if wf.get("units", "Pa") == "Pa":
            y = trace / 1e3
            ylabel = "Pressure (kPa)"
        else:
            y = trace
            ylabel = f"Amplitude ({wf.get('units', 'a.u.')})"
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.plot(t_us, y, lw=0.9)
        arrival_us = wf.get("arrival_us")
        if arrival_us is not None and np.isfinite(arrival_us):
            ax.axvline(float(arrival_us), color="tab:red", lw=0.8,
                       label=f"arrival {float(arrival_us):.2f} \u00b5s")
            ax.legend(loc="upper right", fontsize=8)
        ax.set_xlabel("Time (\u00b5s)")
        ax.set_ylabel(ylabel)
        pnp_kPa = float(wf.get("pnp_MPa", float("nan"))) * 1000.0
        ax.set_title(
            f"Figure 5: Waveform at peak "
            f"(PNP = {pnp_kPa:.1f} kPa)"
        )
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        figs["waveform_at_peak"] = fig

    # Figure 6: Frequency response.
    fr = report.freq_response
    if fr:
        freqs = np.asarray(fr["frequencies_kHz"], dtype=float)
        pnp_kPa = np.asarray(fr["pnp_MPa"], dtype=float) * 1000.0
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.plot(freqs, pnp_kPa, "o-", lw=1.5)
        ax.axvline(report.frequency_kHz, color="0.6", lw=0.7,
                   label=f"nominal {report.frequency_kHz:.0f} kHz")
        ax.set_xlabel("Frequency (kHz)")
        ax.set_ylabel("PNP (kPa)")
        ax.set_title(f"Figure 6: Frequency response @ {report.voltage_V:.0f} V")
        ax.grid(True, alpha=0.3)
        ax.legend()
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
        r2 = vr.get("r2", float("nan"))
        fit = slope_kPa_per_V * volts + intercept_kPa
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.plot(volts, pnp_kPa, "o", label="measured")
        ax.plot(volts, fit, "-", lw=1, label=f"fit (R\u00b2={r2:.4f})")
        ax.set_xlabel("HV rail (V)")
        ax.set_ylabel("PNP (kPa)")
        ax.set_title(f"Figure 7: Voltage linearity @ {report.frequency_kHz:.0f} kHz")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        figs["voltage_linearity"] = fig

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
    sws.append(["Started at", report.started_at])
    sws.append(["Finished at", report.finished_at])
    sws.append(["Peak XY (mm)", f"({report.peak_xy_mm[0]:.4f}, "
                                f"{report.peak_xy_mm[1]:.4f})"])
    arr = report.arrival_check
    if arr:
        exp = arr.get("expected_us")
        got = arr.get("arrival_us")
        sws.append(["Arrival check",
                    f"got {got:.2f}\u00b5s vs expected {exp:.2f}\u00b5s "
                    f"({'PASS' if arr.get('passed') else 'FAIL'})"
                    if got is not None else "no signal"])
    if report.voltage_response:
        sws.append(["Voltage linearity R\u00b2",
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
    """Write a multi-page PDF: cover page + one page per figure."""
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt

    path = Path(path)
    figures = figures or {}
    # Build a letter -> section title map so we can insert a header
    # line every time the section changes in the cover-page listing.
    section_titles = {letter: title for letter, title in SECTION_HEADERS}
    with PdfPages(path) as pdf:
        # Cover page
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        lines = [
            "TXM Characterization Report",
            "",
            f"Overall verdict: {'PASS' if report.overall_pass else 'FAIL'}",
            f"Date: {report.started_at}   Finished: {report.finished_at}",
            "",
        ]
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

        # Figure pages
        for name, fig in figures.items():
            pdf.savefig(fig)
    logger.info("PDF written to %s", path)
    return path


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
                 *, run_dir_name: Optional[str] = None,
                 write_device_config_json: bool = True) -> Path:
    """Serialize a :class:`TestReport` into a timestamped subdirectory.

    Layout when ``output_dir`` is omitted::

        test_reports/<TXM-SN>/<YYYYMMDD>_<HHMMSS>/
            <TXM-SN>_Report.xlsx, <TXM-SN>_Report.pdf,
            <TXM-SN>_device_config.json, <TXM-SN>_summary.csv,
            *.csv, figures/*.png, raw/*.npz, operator_prefs_snapshot.json

    Args:
        report: The completed report.
        output_dir: Parent directory. Defaults to
            ``test_reports/<TXM-SN>/`` in the current working
            directory so per-device runs live side-by-side. Created
            if missing.
        run_dir_name: Override the auto-generated subdirectory name
            (defaults to a ``YYYYMMDD_HHMMSS`` timestamp).
        write_device_config_json: Set ``False`` to skip the config
            JSON (e.g. if the freq sweep was skipped).

    Returns:
        The path to the run directory.
    """
    sn_row = report.rows.get(ROW["txm_sn"])
    sn = _sanitize_stem(sn_row.value if sn_row else "unknown")

    if output_dir is None:
        output_dir = Path.cwd() / "test_reports" / sn
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if run_dir_name is None:
        run_dir_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / run_dir_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Figures first (both xlsx + pdf reuse them).
    figures = build_figures(report)

    # All top-level artifacts share the ``<TXM-SN>_...`` prefix so a
    # single file picked out of the folder is self-describing.
    report_stem = f"{sn}_Report"
    write_xlsx(report, run_dir / f"{report_stem}.xlsx", figures=figures)
    write_pdf(report, run_dir / f"{report_stem}.pdf", figures=figures)
    write_csv_bundle(report, run_dir, summary_name=f"{sn}_summary.csv")

    if write_device_config_json:
        try:
            write_device_config(report, run_dir / f"{sn}_device_config.json")
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

    # Close figures we opened so they don't linger.
    try:
        import matplotlib.pyplot as plt
        for fig in figures.values():
            plt.close(fig)
    except Exception:
        pass

    logger.info("Run directory: %s", run_dir)
    return run_dir
