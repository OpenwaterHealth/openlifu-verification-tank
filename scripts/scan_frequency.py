"""CLI wrapper: TX pulse frequency sweep at a fixed focus.

Delegates to
:meth:`openlifu_verification.VerificationTank.scan_frequency`.
"""
import argparse
import logging
from pathlib import Path

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
    # --- Pulse / drive (None => VerificationTank class defaults) ---
    parser.add_argument("--center-frequency-khz", type=float, default=None,
                        help="Nominal TX frequency (kHz) used to pick the "
                             "pinmap and configure the base pulse. Sweep is "
                             "independent (--f-start/--f-stop/--f-step).")
    parser.add_argument("--voltage", type=float, default=None)
    parser.add_argument("--duration-usec", type=float, default=None,
                        help="Base pulse duration in µs. Also used as the "
                             "per-point pulse duration during the sweep.")
    parser.add_argument("--interval-msec", type=float, default=None)
    # --- Sweep grid ---
    parser.add_argument("--f-start", type=float, default=370.0,
                        help="Start frequency in kHz.")
    parser.add_argument("--f-stop", type=float, default=430.0,
                        help="Stop frequency in kHz.")
    parser.add_argument("--f-step", type=float, default=5.0,
                        help="Frequency step in kHz.")
    parser.add_argument("--focus", type=float, nargs=3, default=None,
                        help="x y z in mm (default: calibrated hydrophone_position).")
    # --- Hydrophone / calibration ---
    parser.add_argument("--hydrophone", type=str, default="")
    parser.add_argument("--calibration-path", type=str,
                        default=str(paths.HYDROPHONE_STATE_PATH),
                        help="JSON file storing hydrophone position + last-used ID.")
    parser.add_argument("--raw-mv", action="store_true",
                        help="Skip the mV->Pa calibration even if a "
                             "hydrophone ID is saved. Position calibration "
                             "and hydrophone ID still load normally; only "
                             "the trace/rms/vpp reporting stays in mV.")
    # --- Misc ---
    parser.add_argument("--chunk-size", type=int, default=0)
    parser.add_argument("--n-averages", type=int, default=1,
                        help="Repeat every frequency this many times and "
                             "coherently average.")
    parser.add_argument("--align", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Cross-correlate repeats before averaging.")
    parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-data", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-plot", type=str, default="")
    parser.add_argument("--log-file", type=str, default="")
    parser.add_argument("--progress", type=str, default="bar",
                        choices=["bar", "log", "both", "none"])
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument("--verbose", "-v", action="store_true",
                           help="Enable DEBUG logging from openlifu_verification and openlifu_sdk.")
    verbosity.add_argument("--quiet", "-q", action="store_true",
                           help="Suppress INFO logging (WARNING and above only).")
    args = parser.parse_args()

    _configure_root_logger()
    if args.verbose:
        set_log_level("DEBUG", sdk_level="DEBUG")
    elif args.quiet:
        set_log_level("WARNING")

    if not args.plot and not args.save_plot and not args.save_data:
        resp = input("Nothing will be plotted or saved. Continue anyway? [y/N] ")
        if resp.strip().lower() not in ("y", "yes"):
            logger.info("Aborted.")
            return

    frequencies = np.arange(args.f_start,
                            args.f_stop + 0.5 * args.f_step,
                            args.f_step)

    progress = None if args.progress == "none" else args.progress
    tank_frequency = int(args.center_frequency_khz
                         if args.center_frequency_khz is not None
                         else VerificationTank.DEFAULT_FREQUENCY_KHZ)

    try:
        with VerificationTank(frequency=tank_frequency,
                              num_modules=1,
                              ext_power_supply=False,
                              hydrophone=args.hydrophone or None,
                              use_calibration=not args.raw_mv,
                              calibration_path=args.calibration_path or None) as ver:
            if args.log_file:
                ver.add_log_file(args.log_file)
            resolved = ver.apply_pulse(
                frequency_kHz=args.center_frequency_khz,
                voltage=args.voltage,
                duration_usec=args.duration_usec,
                interval_msec=args.interval_msec,
            )
            focus = args.focus if args.focus is not None else ver.hydrophone_position.tolist()
            ver.set_focus(*focus)
            ver.enable_hv_output(wait=True)
            input("Press Enter to start")

            result = ver.scan_frequency(
                frequencies_kHz=frequencies,
                duration_usec=resolved["duration_usec"],
                chunk_size=args.chunk_size,
                n_averages=args.n_averages,
                align=args.align,
                progress=progress,
            )
    except (ConnectionError, ValueError, Exception) as e:
        logger.error("Scan aborted: %s", e)
        raise

    if args.save_data:
        out = Path(__file__).parent.resolve() / "data" / "scan_freq_data.npz"
        result.save(out)

    if args.plot or args.save_plot:
        result.plot(show=args.plot, save_as=args.save_plot)


if __name__ == "__main__":
    main()
