"""CLI wrapper: HV drive-voltage sweep at a fixed focus.

Delegates to
:meth:`openlifu_verification.VerificationTank.scan_voltage`.
"""
import argparse
import logging
from pathlib import Path

import numpy as np

from openlifu_verification import VerificationTank, set_log_level

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
    parser.add_argument("--v-start", type=float, default=5.0)
    parser.add_argument("--v-stop", type=float, default=60.0)
    parser.add_argument("--v-step", type=float, default=5.0)
    parser.add_argument("--focus", type=float, nargs=3, default=None,
                        help="x y z in mm (default: calibrated hydrophone_position).")
    parser.add_argument("--hydrophone", type=str, default="",
                        help="Path to a hydrophone calibration .txt to convert scan traces from mV to Pa.")
    parser.add_argument("--calibration-path", type=str, default="hydrophone_position.json",
                        help="JSON file storing the calibrated hydrophone_position (auto-loaded if present).")
    parser.add_argument("--hydro-range-mv", type=int, default=5000,
                        help="Scope full-scale on the hydrophone channel (default 5000 mV).")
    parser.add_argument("--chunk-size", type=int, default=0)
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

    frequency_kHz = 400
    duration_msec = 20 / 400
    interval_msec = 20
    voltages = np.arange(args.v_start,
                         args.v_stop + 0.5 * args.v_step,
                         args.v_step)

    progress = None if args.progress == "none" else args.progress

    try:
        with VerificationTank(frequency=frequency_kHz,
                              num_modules=1,
                              ext_power_supply=False,
                              hydrophone_range_mv=args.hydro_range_mv,
                              hydrophone=args.hydrophone or None,
                              calibration_path=args.calibration_path or None) as ver:
            if args.log_file:
                ver.add_log_file(args.log_file)
            ver.configure_lifu(
                frequency_kHz=frequency_kHz,
                voltage=float(args.v_start),
                duration_msec=duration_msec,
                interval_msec=interval_msec,
                pulse_count=1,
                trigger_mode="single",
            )
            focus = args.focus if args.focus is not None else ver.hydrophone_position.tolist()
            ver.set_focus(*focus)
            ver.enable_hv_output(wait=True)
            input("Press Enter to start")

            result = ver.scan_voltage(
                voltages_V=voltages,
                chunk_size=args.chunk_size,
                progress=progress,
            )
    except (ConnectionError, ValueError, Exception) as e:
        logger.error("Scan aborted: %s", e)
        raise

    if args.save_data:
        out = Path(__file__).parent.resolve() / "data" / "scan_voltage_data.npz"
        result.save(out)

    if args.plot or args.save_plot:
        result.plot(show=args.plot, save_as=args.save_plot)


if __name__ == "__main__":
    main()
