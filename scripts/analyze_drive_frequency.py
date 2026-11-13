"""Check commanded vs. realized TX drive frequency on saved runs.

For every run folder under ``ROOT`` (anything containing
``raw/freq_sweep.npz``) this

1. measures the tone in each frequency-sweep waveform
   (:func:`openlifu_verification.drive_frequency.estimate_tone_frequency`)
   and compares it with the frequency predicted from the SDK pulse
   pattern (:func:`~openlifu_verification.drive_frequency.realized_frequency_hz`);
2. reads the report's E-rows ("Mean/Median PNP (<f> kHz)"), which were
   converted with the hydrophone calibration at the *commanded*
   frequency, and rescales them to the calibration at the *realized*
   frequency, then reports how well points that share one emitted tone
   (400 kHz band) and different hydrophones on one module agree under
   each convention.

Usage::

    python scripts/analyze_drive_frequency.py ROOT [--out DIR]
        [--since YYYYMMDD_HHMMSS] [--exclude HYDRO2246:155]

Writes ``drive_freq_measured.csv`` and ``drive_freq_pnp_rescaled.csv``
to ``--out`` (default: current directory) and prints summaries.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from openlifu_verification.drive_frequency import (estimate_tone_frequency,
                                                   realized_frequency_hz)
from openlifu_verification.hydrophone import Hydrophone

_ROW = re.compile(r"(?:Mean|Median) PNP \((\d+) kHz\)")


def _hydrophone_sn(run: Path) -> str:
    try:
        prefs = json.loads((run / "operator_prefs_snapshot.json").read_text(encoding="utf-8"))
        return str(prefs.get("hydrophone_sn", "?"))
    except (OSError, ValueError):
        return "?"


def measured_frequencies(runs: list[Path]) -> pd.DataFrame:
    rows = []
    for run in runs:
        z = np.load(run / "raw" / "freq_sweep.npz", allow_pickle=True)
        t_s = np.asarray(z["t"], dtype=float) * 1e-6
        dur_s = float(z["meta_duration_usec"]) * 1e-6 if "meta_duration_usec" in z.files else None
        for cmd, y in zip(z["freq_kHz"], z["outputs"]):
            pred = realized_frequency_hz(float(cmd) * 1e3) / 1e3
            meas = estimate_tone_frequency(t_s, y, pred * 1e3, span=0.15,
                                           burst_s=dur_s) / 1e3
            rows.append({"run": str(run), "hydrophone": _hydrophone_sn(run),
                         "commanded_kHz": float(cmd), "predicted_kHz": pred,
                         "measured_kHz": meas})
    return pd.DataFrame(rows)


def rescaled_pnp(runs: list[Path], cal_dir: Path) -> pd.DataFrame:
    rows = []
    for run in runs:
        reports = sorted(run.glob("*_Report.xlsx"))
        if not reports:
            continue
        sheet = pd.read_excel(reports[0], sheet_name="Report", header=None)
        for _, r in sheet.iterrows():
            m = _ROW.fullmatch(str(r[1]).strip())
            if m and str(r[0]).startswith("E."):
                rows.append({"module": run.parent.name, "run": run.name,
                             "hydrophone": _hydrophone_sn(run),
                             "commanded_kHz": float(m.group(1)),
                             "pnp_cmd_MPa": float(r[2])})
    d = pd.DataFrame(rows)
    if d.empty:
        return d
    hyds = {}
    for sn in d.hydrophone.unique():
        hits = sorted(cal_dir.glob(f"HNR0500-{sn.replace('HYDRO', '')}*.txt"))
        hyds[sn] = Hydrophone(hits[0]) if hits else None
    d = d[d.hydrophone.map(lambda s: hyds.get(s) is not None)].copy()
    d["realized_kHz"] = realized_frequency_hz(d.commanded_kHz.to_numpy() * 1e3) / 1e3
    d["cal_ratio"] = [
        hyds[s].get_frequency_response(r * 1e3) / hyds[s].get_frequency_response(c * 1e3)
        for s, c, r in zip(d.hydrophone, d.commanded_kHz, d.realized_kHz)]
    d["pnp_real_MPa"] = d.pnp_cmd_MPa * d.cal_ratio
    return d


def _cv(v) -> float:
    v = np.asarray(v, dtype=float)
    return 100 * v.std(ddof=1) / v.mean() if v.size > 1 else np.nan


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root", type=Path)
    ap.add_argument("--out", type=Path, default=Path("."))
    ap.add_argument("--since", default="", help="skip runs whose folder name sorts before this")
    ap.add_argument("--exclude", action="append", default=[],
                    help="HYDROxxxx:<nominal kHz> pairs to drop from the PNP analysis")
    ap.add_argument("--cal-dir", type=Path, default=Path(__file__).resolve().parents[1]
                    / "openlifu_verification" / "setup_data_template" / "hydrophone_calibrations")
    a = ap.parse_args(argv)
    runs = sorted(p.parent.parent for p in a.root.rglob("raw/freq_sweep.npz")
                  if p.parent.parent.name >= a.since)
    a.out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)

    meas = measured_frequencies(runs)
    meas.to_csv(a.out / "drive_freq_measured.csv", index=False)
    meas["ratio"] = meas.measured_kHz / meas.predicted_kHz
    print("Measured vs predicted tone (mean over runs):")
    print(meas.groupby("commanded_kHz").agg(
        predicted=("predicted_kHz", "first"), measured=("measured_kHz", "mean"),
        sd=("measured_kHz", "std"), n=("run", "count")).round(3))

    pnp = rescaled_pnp(runs, a.cal_dir)
    for ex in a.exclude:
        sn, band = ex.split(":")
        nominal = 155.0 if float(band) < 300 else 400.0
        pnp = pnp[~((pnp.hydrophone == sn)
                    & ((pnp.commanded_kHz < 300) == (nominal < 300)))]
    pnp.to_csv(a.out / "drive_freq_pnp_rescaled.csv", index=False)
    if pnp.empty:
        return
    print("\nPNP change from reading the cal at the realized frequency (%):")
    print((100 * (pnp.pivot_table(index="commanded_kHz", columns="hydrophone",
                                  values="cal_ratio") - 1)).round(1))
    tone = pnp.groupby(["module", "run", "realized_kHz"]).agg(
        n=("pnp_cmd_MPa", "size"), cv_cmd=("pnp_cmd_MPa", _cv), cv_real=("pnp_real_MPa", _cv))
    tone = tone[tone.n > 1]
    if not tone.empty:
        print("\nWithin-run CV (%) across commanded points that emit the same tone:")
        print(tone.groupby("realized_kHz")[["cv_cmd", "cv_real"]].mean().round(2))
    per = pnp.groupby(["module", "commanded_kHz", "hydrophone"])[
        ["pnp_cmd_MPa", "pnp_real_MPa"]].mean().reset_index()
    across = per.groupby(["module", "commanded_kHz"]).agg(
        n_hyd=("hydrophone", "nunique"), cv_cmd=("pnp_cmd_MPa", _cv),
        cv_real=("pnp_real_MPa", _cv))
    across = across[across.n_hyd > 1]
    if not across.empty:
        print("\nAcross-hydrophone CV (%) of run-mean PNP on the same module:")
        print(across.round(2))


if __name__ == "__main__":
    main()
