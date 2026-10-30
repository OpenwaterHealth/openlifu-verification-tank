"""Measure the trigger-to-drive delay vs TX frequency on one device.

Answers whether ``SYSTEM_TRANSMIT_DELAY_US_BY_FREQ_KHZ`` depends on
frequency, on the device (TX module / console variant), or both.

For each frequency, fires a plane-wave pulse train and finds the first
edge of the drive's EM pickup on the hydrophone channel (first
crossing of ``--threshold-mv`` above the pre-trigger baseline). The
time axis uses a fixed ``--reference-delay-us`` for every frequency,
so ``transmit_delay_us = reference_delay_us + drive_start_us`` is the
absolute trigger-to-drive delay. Frequencies are interleaved across
``--repeats`` so session drift is not mistaken for a frequency effect.

Each run appends one row per (repeat, frequency) to ``--csv`` along
with the TX hardware ID, firmware version and ``--label``. Run it on
each device into the same CSV; the plot overlays every device:

- curves overlap, delay changes with frequency -> frequency dependent
- curves flat but offset per device             -> device dependent

Example::

    python scripts/transmit_delay_diagnostic.py --label "1x400 TXM #3" \\
        --frequencies 155 200 250 300 350 400 --repeats 3
"""
import argparse
import csv
import datetime
import logging
from pathlib import Path

import numpy as np

from openlifu_verification import VerificationTank, set_log_level

logger = logging.getLogger(__name__)

CSV_FIELDS = [
    "timestamp", "label", "tx_hardware_id", "tx_firmware", "num_modules",
    "repeat", "frequency_kHz", "voltage_V", "duration_usec", "n_pulses",
    "reference_delay_us", "drive_start_us", "drive_start_std_us",
    "n_detected", "pickup_peak_mv", "transmit_delay_us",
]


def drive_start_us(t_us, traces_mv, *, baseline_us, search_stop_us,
                   threshold_mv):
    """Per-pulse first threshold crossing (us), linearly interpolated.

    Each pulse is referenced to the median of its samples before
    ``baseline_us``. Returns an array with NaN where no crossing was
    found in ``[baseline_us, search_stop_us)``.
    """
    base_mask = t_us < baseline_us
    search = np.flatnonzero((t_us >= baseline_us) & (t_us < search_stop_us))
    starts = np.full(traces_mv.shape[0], np.nan)
    for i, trace in enumerate(traces_mv):
        v = np.abs(trace - np.median(trace[base_mask]))
        hits = search[v[search] > threshold_mv]
        if hits.size == 0:
            continue
        k = int(hits[0])
        frac = (threshold_mv - v[k - 1]) / (v[k] - v[k - 1])
        starts[i] = t_us[k - 1] + frac * (t_us[k] - t_us[k - 1])
    return starts


def _device_info(ver):
    info = {}
    for key, getter in (("tx_hardware_id", "get_hardware_id"),
                        ("tx_firmware", "get_version")):
        try:
            info[key] = str(getattr(ver.lifu.txdevice, getter)())
        except Exception as e:  # informational only
            logger.warning("Could not read %s: %s", key, e)
            info[key] = "unknown"
    return info


def plot_csv(csv_path):
    import matplotlib.pyplot as plt

    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    fig, ax = plt.subplots(figsize=(8, 5))
    devices = sorted({(r["label"], r["tx_hardware_id"]) for r in rows})
    for label, hwid in devices:
        sel = [r for r in rows
               if r["label"] == label and r["tx_hardware_id"] == hwid]
        f_khz = np.array([float(r["frequency_kHz"]) for r in sel])
        delay = np.array([float(r["transmit_delay_us"]) for r in sel])
        err = np.array([float(r["drive_start_std_us"]) for r in sel])
        ax.errorbar(f_khz, delay, yerr=err, fmt="o", capsize=3, alpha=0.7,
                    label=f"{label or '(no label)'} [{hwid}]")
    ax.set_xlabel("TX frequency (kHz)")
    ax.set_ylabel("trigger -> drive delay (us)")
    ax.set_title(f"Transmit delay ({Path(csv_path).name})")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", type=str, default="",
                        help="Device / variant description stored with "
                             "each row (e.g. '2x400 console B').")
    parser.add_argument("--frequencies", type=float, nargs="+",
                        default=[155.0, 200.0, 250.0, 300.0, 350.0, 400.0],
                        help="TX frequencies to test (kHz).")
    parser.add_argument("--repeats", type=int, default=3,
                        help="Passes over the frequency list (interleaved).")
    parser.add_argument("--voltage", type=float, default=30.0)
    parser.add_argument("--duration-usec", type=float, default=8.0)
    parser.add_argument("--pulse-count", "-n", type=int, default=32)
    parser.add_argument("--reference-delay-us", type=float, default=110.0,
                        help="Fixed time-axis delay for every frequency. "
                             "Must be below the true delay so the "
                             "baseline window precedes the drive.")
    parser.add_argument("--baseline-us", type=float, default=1.0,
                        help="Samples before this time (relative to the "
                             "reference delay) form the baseline.")
    parser.add_argument("--search-stop-us", type=float, default=20.0,
                        help="Ignore crossings after this time (keeps the "
                             "acoustic arrival out of the search).")
    parser.add_argument("--threshold-mv", type=float, default=1.0)
    parser.add_argument("--time-stop-us", type=float, default=30.0)
    parser.add_argument("--sampling-interval-ns", type=float, default=100.0)
    parser.add_argument("--hydro-range-mv", type=int, default=100)
    parser.add_argument("--num-modules", type=int, default=1)
    parser.add_argument("--csv", type=str, default="transmit_delay.csv",
                        help="CSV to append results to.")
    parser.add_argument("--plot", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Plot every device in --csv when done.")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip hardware; just plot --csv.")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    if args.verbose:
        set_log_level("DEBUG", sdk_level="DEBUG")

    if args.plot_only:
        plot_csv(args.csv)
        return

    csv_path = Path(args.csv)
    write_header = not csv_path.is_file()
    rows = []
    with VerificationTank(frequency=args.frequencies[0],
                          num_modules=args.num_modules,
                          hydrophone_range_mv=args.hydro_range_mv,
                          use_calibration=False,
                          calibration_path=None,
                          system_transmit_delay_us=args.reference_delay_us,
                          ) as ver, \
            open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        device = _device_info(ver)
        logger.info("Device: %s, label=%r", device, args.label)
        input("Press Enter to start the transmit-delay diagnostic")
        for rep in range(args.repeats):
            for freq in args.frequencies:
                resolved = ver.apply_pulse(
                    frequency_kHz=freq, voltage=args.voltage,
                    duration_usec=args.duration_usec,
                    pulse_count=args.pulse_count, trigger_mode="single",
                )
                ver.set_focus(0.0, 0.0, 1_000_000.0)  # plane wave
                ver.enable_hv_output(wait=True)
                interval_s = resolved["interval_msec"] * 1e-3
                bulk = ver.capture_pulse_train(
                    n_pulses=args.pulse_count,
                    time_start_s=0.0,
                    time_stop_s=args.time_stop_us * 1e-6,
                    sampling_interval_ns=args.sampling_interval_ns,
                    timeout_s=args.pulse_count * interval_s + 2.0,
                )
                if bulk is None:
                    logger.error("%g kHz: capture timed out", freq)
                    continue
                t_us = np.asarray(bulk["time"], dtype=float)
                traces = np.asarray(bulk[ver.hydrophone_channel], dtype=float)
                starts = drive_start_us(
                    t_us, traces, baseline_us=args.baseline_us,
                    search_stop_us=args.search_stop_us,
                    threshold_mv=args.threshold_mv,
                )
                ok = starts[np.isfinite(starts)]
                if ok.size == 0:
                    logger.error("%g kHz: no pickup above %g mV",
                                 freq, args.threshold_mv)
                    continue
                search = (t_us >= args.baseline_us) & (t_us < args.search_stop_us)
                start = float(np.median(ok))
                row = dict(
                    timestamp=datetime.datetime.now().isoformat(timespec="seconds"),
                    label=args.label, num_modules=args.num_modules,
                    repeat=rep, frequency_kHz=freq, voltage_V=args.voltage,
                    duration_usec=resolved["duration_usec"],
                    n_pulses=args.pulse_count,
                    reference_delay_us=args.reference_delay_us,
                    drive_start_us=round(start, 4),
                    drive_start_std_us=round(float(np.std(ok)), 4),
                    n_detected=int(ok.size),
                    pickup_peak_mv=round(float(np.median(
                        np.max(np.abs(traces[:, search]), axis=1))), 3),
                    transmit_delay_us=round(args.reference_delay_us + start, 4),
                    **device,
                )
                writer.writerow(row)
                f.flush()
                rows.append(row)
                logger.info(
                    "rep %d, %g kHz: delay = %.3f us (sigma %.3f, %d/%d pulses)",
                    rep, freq, row["transmit_delay_us"],
                    row["drive_start_std_us"], ok.size, args.pulse_count,
                )

    print(f"\n  {'kHz':>6}  {'delay us (median)':>18}  {'spread us':>9}")
    for freq in args.frequencies:
        d = [r["transmit_delay_us"] for r in rows if r["frequency_kHz"] == freq]
        if d:
            print(f"  {freq:6g}  {np.median(d):18.3f}  {np.ptp(d):9.3f}")
    print(f"\nAppended {len(rows)} rows to {csv_path}")

    if args.plot:
        plot_csv(csv_path)


if __name__ == "__main__":
    main()
