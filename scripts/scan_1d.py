"""CLI wrapper: generic 1-D focus scan.

Delegates all the heavy lifting to
:meth:`openlifu_verification.VerificationTank.scan_1d`. Sweeps the
focus along a single axis (``--dim x``, ``y``, or ``z``) with the other
two coords held fixed at either the calibrated hydrophone position
(default) or absolute values.

If you want a 2-D grid, use ``scan_2d.py``.

Examples::

    python scripts/scan_1d.py --dim x --num 41
    python scripts/scan_1d.py --dim y --num 41 --range -5 5
    python scripts/scan_1d.py --dim z --num 21 --range -10 10 --save-plot axial.png
    python scripts/scan_1d.py --dim x --no-plot            # data-only headless run
"""
import argparse
import logging
from pathlib import Path

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
    # --- Pulse / drive (defaults live on VerificationTank; None = use default) ---
    parser.add_argument("--frequency-khz", type=float, default=None,
                        help="TX frequency in kHz "
                             "(default: VerificationTank.DEFAULT_FREQUENCY_KHZ).")
    parser.add_argument("--voltage", type=float, default=None,
                        help="HV rail in V "
                             "(default: VerificationTank.DEFAULT_VOLTAGE_V).")
    parser.add_argument("--duration-usec", type=float, default=None,
                        help="Pulse duration in µs "
                             "(default: cycles / frequency).")
    parser.add_argument("--interval-msec", type=float, default=None,
                        help="Pulse interval in ms "
                             "(default: VerificationTank.DEFAULT_INTERVAL_MSEC).")
    # --- Scan geometry ---
    parser.add_argument("--dim", type=str, choices=["x", "y", "z"], default="x",
                        help="Which axis to sweep.")
    parser.add_argument("--range", type=float, nargs=2, default=[-10.0, 10.0],
                        dest="scan_range",
                        help="(min, max) in mm along the swept axis.")
    parser.add_argument("--num", type=int, default=41,
                        help="Number of samples across --range.")
    parser.add_argument("--x", type=float, default=0.0,
                        help="Fixed x offset when --dim is y or z. "
                             "Ignored when --dim x.")
    parser.add_argument("--y", type=float, default=0.0,
                        help="Fixed y offset when --dim is x or z. "
                             "Ignored when --dim y.")
    parser.add_argument("--z", type=float, default=None,
                        help="Fixed depth in mm when --dim is x or y. "
                             "Defaults to the calibrated hydrophone_position[2]. "
                             "Ignored when --dim z.")
    parser.add_argument("--absolute", action="store_true",
                        help="Treat x/y/z as absolute coords (skip offset by the hydrophone position).")
    # --- Hydrophone / calibration ---
    parser.add_argument("--hydrophone", type=str, default="",
                        help="Hydrophone calibration file or bare ID. "
                             "Falls back to the last-used ID stored in --calibration-path.")
    parser.add_argument("--calibration-path", type=str,
                        default=str(paths.HYDROPHONE_STATE_PATH),
                        help="JSON file storing hydrophone position + last-used ID "
                             "(auto-loaded if present).")
    parser.add_argument("--raw-mv", action="store_true",
                        help="Skip the mV->Pa calibration even if a "
                             "hydrophone ID is saved. Position calibration "
                             "and hydrophone ID still load normally; only "
                             "the trace/rms/vpp reporting stays in mV.")
    # --- Misc ---
    parser.add_argument("--chunk-size", type=int, default=0,
                        help="Rapid-block chunk size (0 = whole sweep).")
    parser.add_argument("--n-averages", type=int, default=1,
                        help="Repeat every scan point this many times and "
                             "coherently average the traces.")
    parser.add_argument("--align", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Cross-correlate repeats against the first "
                             "before averaging (default on).")
    parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-data", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-plot", type=str, default="")
    parser.add_argument("--log-file", type=str, default="",
                        help="Optional path for a copy of the log.")
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

    progress = None if args.progress == "none" else args.progress

    # VerificationTank(frequency=...) picks the pinmap; if the operator
    # didn't override, fall back to the class default.
    tank_frequency = int(args.frequency_khz
                         if args.frequency_khz is not None
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
            ver.apply_pulse(
                frequency_kHz=args.frequency_khz,
                voltage=args.voltage,
                duration_usec=args.duration_usec,
                interval_msec=args.interval_msec,
            )
            ver.enable_hv_output(wait=True)
            input("Press Enter to start")

            scan_kwargs = dict(
                dim=args.dim,
                scan_range=tuple(args.scan_range),
                num=args.num,
                x=args.x,
                y=args.y,
                absolute=args.absolute,
                chunk_size=args.chunk_size,
                n_averages=args.n_averages,
                align=args.align,
                progress=progress,
            )
            if args.z is not None:
                scan_kwargs["z"] = args.z
            result = ver.scan_1d(**scan_kwargs)
    except (ConnectionError, ValueError, Exception) as e:
        logger.error("Scan aborted: %s", e)
        raise

    if args.save_data:
        out = Path(__file__).parent.resolve() / "data" / f"scan_1d_{args.dim}_data.npz"
        result.save(out)

    if args.plot or args.save_plot:
        result.plot(show=args.plot, save_as=args.save_plot)


if __name__ == "__main__":
    main()
