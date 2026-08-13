"""Dial-in demo for the arrival-time / hydrophone-depth calibration.

Fires a train of short plane-wave pulses at the hydrophone, coherently
averages them, locates the first RF-carrier peak above a prominence
threshold, applies a quarter-cycle correction, and reports the
inferred array-to-hydrophone distance. Also plots the aggregated
waveform with the picked first peak, the prominence threshold, the
quarter-cycle-corrected arrival, and a shaded pre-``skip_us``
exclusion region so you can iterate on ``--skip-us``,
``--duration-usec``, and ``--voltage`` before wiring this into
``find_peak.py``.

Example::

    python scripts/arrival_time_demo.py --hydrophone 2246 \\
        --voltage 30 --duration-usec 8 --pulse-count 32 --skip-us 12

The reported ``distance_mm`` is the one-way distance from the array
face to the hydrophone (plane-wave excitation, all elements fire
simultaneously). When you steer to a real focus (see ``find_peak``),
the last element to fire is the one directly above the hydrophone, so
subtracting the max element delay gives the same one-way normal
distance.
"""
import argparse
import logging

import numpy as np

from openlifu_verification import VerificationTank, paths, set_log_level

logger = logging.getLogger(__name__)


def _configure_root_logger():
    root = logging.getLogger()
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        root.addHandler(h)
    root.setLevel(logging.INFO)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    # --- Excitation ---
    parser.add_argument("--frequency-khz", type=float, default=None,
                        help="TX frequency in kHz (default: VerificationTank default).")
    parser.add_argument("--voltage", type=float, default=30.0,
                        help="HV rail (V). Plane-wave excitation spreads "
                             "energy so we default to 30 V to keep SNR up.")
    parser.add_argument("--duration-usec", type=float, default=8.0,
                        help="Per-pulse duration in \u00b5s (default 8).")
    parser.add_argument("--pulse-count", "-n", type=int, default=32,
                        help="Pulses fired per trigger; averaged coherently. "
                             "Default 32.")
    parser.add_argument("--interval-msec", type=float, default=None,
                        help="Interval between pulses in the train (ms).")
    # --- Capture ---
    parser.add_argument("--time-start-us", type=float, default=0.0)
    parser.add_argument("--time-stop-us", type=float, default=100.0)
    parser.add_argument("--sampling-interval-ns", type=float, default=100.0)
    parser.add_argument("--hydro-range-mv", type=int, default=100)
    parser.add_argument("--skip-us", type=float, default=12.0,
                        help="Ignore samples before this time when hunting "
                             "for the first arrival (default 12 \u00b5s to skip "
                             "the trigger flash / cross-talk).")
    parser.add_argument("--align", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Cross-correlate pulses before averaging.")
    # --- Hydrophone / calibration ---
    parser.add_argument("--hydrophone", type=str, default="",
                        help="Hydrophone calibration file or bare ID.")
    parser.add_argument("--calibration-path", type=str,
                        default=str(paths.HYDROPHONE_STATE_PATH))
    parser.add_argument("--save-calibration", action="store_true",
                        help="Persist the found depth into "
                             "hydrophone_position[2].")
    parser.add_argument("--num-modules", type=int, default=1)
    # --- Output ---
    parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-plot", type=str, default="")
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument("--verbose", "-v", action="store_true")
    verbosity.add_argument("--quiet", "-q", action="store_true")
    args = parser.parse_args()

    _configure_root_logger()
    if args.verbose:
        set_log_level("DEBUG", sdk_level="DEBUG")
    elif args.quiet:
        set_log_level("WARNING")

    tank_frequency = int(args.frequency_khz
                         if args.frequency_khz is not None
                         else VerificationTank.DEFAULT_FREQUENCY_KHZ)

    with VerificationTank(frequency=tank_frequency,
                          num_modules=args.num_modules,
                          hydrophone=args.hydrophone or None,
                          calibration_path=args.calibration_path or None,
                          hydrophone_range_mv=args.hydro_range_mv,
                          use_calibration=False,  # amplitude-independent
                          ext_power_supply=False) as ver:
        input("Press Enter to start plane-wave arrival-time calibration")
        result = ver.calibrate_hydrophone_depth(
            voltage_V=args.voltage,
            n_pulses=args.pulse_count,
            duration_usec=args.duration_usec,
            interval_msec=args.interval_msec,
            skip_us=args.skip_us,
            time_start_us=args.time_start_us,
            time_stop_us=args.time_stop_us,
            sampling_interval_ns=args.sampling_interval_ns,
            hydrophone_range_mv=args.hydro_range_mv,
            align=args.align,
            store=args.save_calibration,
            save=args.save_calibration,
        )

    print()
    print(f"  n_pulses averaged : {result['n_pulses_used']}")
    print(f"  first RF peak     : {result['first_peak_us']:.3f} \u00b5s")
    print(f"  peak value        : {result['first_peak_value']:.4g} mV")
    print(f"  threshold         : {result['threshold']:.4g} mV")
    print(f"  quarter-cycle corr: {result['quarter_cycle_us']:.3f} \u00b5s")
    print(f"  arrival           : {result['arrival_us']:.3f} \u00b5s")
    print(f"  SoS               : {result['sos_m_per_s']:.0f} m/s")
    print(f"  distance          : {result['distance_mm']:.3f} mm")

    if not (args.plot or args.save_plot):
        return

    import matplotlib.pyplot as plt

    t_us = np.asarray(result["t_us"])
    trace = np.asarray(result["mean_trace"])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t_us, trace, lw=0.9, color="tab:blue",
            label=f"mean RF ({result['n_pulses_used']} pulses)")
    ax.axhline(result["threshold"], color="tab:orange", lw=0.8, ls=":",
               alpha=0.8,
               label=f"prominence threshold ({result['threshold']:.3g} mV)")
    ax.axvspan(t_us[0], args.skip_us, color="grey", alpha=0.15,
               label=f"skip < {args.skip_us:g} \u00b5s")
    ax.axvline(result["first_peak_us"], color="tab:orange",
               ls="--", lw=1.2, label="1st RF peak (above threshold)")
    ax.axvline(result["arrival_us"], color="tab:red",
               ls="--", lw=1.5,
               label=(f"arrival = peak \u2212 \u00bc cycle "
                      f"({result['quarter_cycle_us']:.2f} \u00b5s)"))
    ax.set_xlabel("time (\u00b5s, relative to emission)")
    ax.set_ylabel("hydrophone (mV, averaged)")
    ax.set_title(
        f"Plane-wave arrival-time calibration  \u2192  "
        f"distance = {result['distance_mm']:.3f} mm"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()

    if args.save_plot:
        fig.savefig(args.save_plot, dpi=120)
        logger.info("Saved plot to %s", args.save_plot)
    if args.plot:
        plt.show()


if __name__ == "__main__":
    main()
