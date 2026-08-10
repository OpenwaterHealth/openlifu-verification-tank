"""CLI wrapper: 2-D (x, y) focus scan at a fixed z.

Delegates to
:meth:`openlifu_verification.VerificationTank.scan_2d`.

Examples::

    python scripts/scan_2d.py
    python scripts/scan_2d.py --num-x 21 --num-y 21 --save-plot scan_2d.png
"""
import argparse
import logging
from pathlib import Path

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
    parser.add_argument("--voltage", type=float, default=5.0)
    parser.add_argument("--num-x", type=int, default=9)
    parser.add_argument("--num-y", type=int, default=9)
    parser.add_argument("--x-range", type=float, nargs=2, default=[-4.0, 4.0])
    parser.add_argument("--y-range", type=float, nargs=2, default=[-4.0, 4.0])
    parser.add_argument("--z", type=float, default=None,
                        help="Depth in mm. Defaults to the calibrated hydrophone_position[2].")
    parser.add_argument("--absolute", action="store_true",
                        help="Treat x/y as absolute coords (skip offset by the hydrophone position).")
    parser.add_argument("--hydrophone", type=str, default="",
                        help="Path to a hydrophone calibration .txt to convert scan traces from mV to Pa.")
    parser.add_argument("--calibration-path", type=str, default="hydrophone_position.json",
                        help="JSON file storing the calibrated hydrophone_position (auto-loaded if present).")
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

    progress = None if args.progress == "none" else args.progress

    try:
        with VerificationTank(frequency=frequency_kHz,
                              num_modules=1,
                              ext_power_supply=False,
                              hydrophone=args.hydrophone or None,
                              calibration_path=args.calibration_path or None) as ver:
            if args.log_file:
                ver.add_log_file(args.log_file)
            ver.configure_lifu(
                frequency_kHz=frequency_kHz,
                voltage=args.voltage,
                duration_msec=duration_msec,
                interval_msec=interval_msec,
                pulse_count=1,
                trigger_mode="single",
            )
            ver.enable_hv_output(wait=True)
            input("Press Enter to start")

            scan_kwargs = dict(
                x_range=tuple(args.x_range),
                num_x=args.num_x,
                y_range=tuple(args.y_range),
                num_y=args.num_y,
                absolute=args.absolute,
                chunk_size=args.chunk_size,
                progress=progress,
            )
            if args.z is not None:
                scan_kwargs["z"] = args.z
            result = ver.scan_2d(**scan_kwargs)
    except (ConnectionError, ValueError, Exception) as e:
        logger.error("Scan aborted: %s", e)
        raise

    if args.save_data:
        # Historical location: workspace root (parent of scripts/).
        out = Path(__file__).parent.parent.resolve() / "scan_2d_data.npz"
        result.save(out, save_txt=False)

    if args.plot or args.save_plot:
        result.plot(show=args.plot, save_as=args.save_plot)


if __name__ == "__main__":
    main()
