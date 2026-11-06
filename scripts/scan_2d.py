"""CLI wrapper: 2-D (x, y) focus scan at a fixed z.

Delegates to
:meth:`openlifu_verification.VerificationTank.scan_2d`.

Defaults match the characterization report's 2-D scan (pulse length,
voltage, extent / points, averaging, scope range and capture window)
and the heatmap shows PNP like Figure 1. The grid is centered on the
saved hydrophone position, so run ``find_peak.py --save-calibration``
first.

Examples::

    python scripts/scan_2d.py
    python scripts/scan_2d.py --num-x 21 --num-y 21 --save-plot scan_2d.png
"""
import argparse
import logging
from pathlib import Path

from openlifu_verification import (Characterization, ScanConfig, VerificationTank,
                                   set_log_level, setup_data)
from openlifu_verification.pnp import emitted_duration_us, scan_window_pnp

logger = logging.getLogger(__name__)


def _configure_root_logger():
    root = logging.getLogger()
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        root.addHandler(h)
    root.setLevel(logging.INFO)


def main():
    cfg = ScanConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    # --- Pulse / drive (None => VerificationTank class defaults) ---
    parser.add_argument("--frequency-khz", type=float, default=None)
    parser.add_argument("--voltage", type=float, default=None,
                        help="HV rail in V (default: VerificationTank.DEFAULT_VOLTAGE_BY_FREQ_KHZ).")
    parser.add_argument("--duration-usec", type=float, default=None,
                        help="Pulse duration in µs (default: the report's pulse for --frequency-khz).")
    parser.add_argument("--interval-msec", type=float, default=None)
    # --- Scan geometry (None => the report's 2-D scan plan) ---
    parser.add_argument("--num-x", type=int, default=None)
    parser.add_argument("--num-y", type=int, default=None)
    parser.add_argument("--x-range", type=float, nargs=2, default=None)
    parser.add_argument("--y-range", type=float, nargs=2, default=None)
    parser.add_argument("--z", type=float, default=None,
                        help="Depth in mm. Defaults to the calibrated hydrophone_position[2].")
    parser.add_argument("--absolute", action="store_true",
                        help="Treat x/y as absolute coords (skip offset by the hydrophone position).")
    # --- Hydrophone / calibration ---
    parser.add_argument("--hydrophone", type=str, default="",
                        help="Hydrophone calibration file or bare ID.")
    parser.add_argument("--calibration-path", type=str,
                        default=str(setup_data.SetupData.active().hydrophone_state_path),
                        help="JSON file storing hydrophone position + last-used ID.")
    parser.add_argument("--raw-mv", action="store_true",
                        help="Skip the mV->Pa calibration even if a "
                             "hydrophone ID is saved. Position calibration "
                             "and hydrophone ID still load normally; only "
                             "the trace/rms/vpp reporting stays in mV.")
    # --- Capture ---
    parser.add_argument("--time-start-us", type=float, default=cfg.scope.time_start_us)
    parser.add_argument("--time-stop-us", type=float, default=None,
                        help="Capture stop (µs). Default: like the report, "
                             f"max({cfg.scope.time_stop_us:g}, arrival + pulse + margin).")
    parser.add_argument("--sampling-interval-ns", type=float,
                        default=cfg.scope.sampling_interval_ns)
    parser.add_argument("--hydro-range-mv", type=int,
                        default=cfg.scope.hydrophone_range_mv)
    # --- Misc ---
    parser.add_argument("--chunk-size", type=int, default=0)
    parser.add_argument("--n-averages", type=int, default=cfg.scope.n_averages,
                        help="Repeat every grid point this many times and "
                             "coherently average (default: %(default)s).")
    parser.add_argument("--align", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Cross-correlate repeats before averaging.")
    parser.add_argument("--reduce", choices=["pnp", "vpp", "vmin", "vmax"],
                        default="pnp",
                        help="Per-point value to plot (default: PNP, as in the report).")
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

    progress = None if args.progress == "none" else args.progress
    tank_frequency = int(args.frequency_khz
                         if args.frequency_khz is not None
                         else VerificationTank.DEFAULT_FREQUENCY_KHZ)
    if args.duration_usec is None:
        args.duration_usec = Characterization.pulse_duration_us_for(tank_frequency)
    if None in (args.num_x, args.num_y, args.x_range, args.y_range):
        grid = cfg.plan_for(tank_frequency).scan_2d
        ext = [-grid.extent_mm, grid.extent_mm]
        args.num_x = grid.points if args.num_x is None else args.num_x
        args.num_y = grid.points if args.num_y is None else args.num_y
        args.x_range = ext if args.x_range is None else args.x_range
        args.y_range = ext if args.y_range is None else args.y_range

    try:
        with VerificationTank(frequency=tank_frequency,
                              num_modules=1,
                              ext_power_supply=False,
                              hydrophone=args.hydrophone or None,
                              hydrophone_range_mv=args.hydro_range_mv,
                              use_calibration=not args.raw_mv,
                              calibration_path=args.calibration_path or None) as ver:
            if args.log_file:
                ver.add_log_file(args.log_file)
            resolved = ver.apply_pulse(
                frequency_kHz=args.frequency_khz,
                voltage=args.voltage,
                duration_usec=args.duration_usec,
                interval_msec=args.interval_msec,
            )
            ver.enable_hv_output(wait=True)

            time_stop_us = args.time_stop_us
            if time_stop_us is None:
                depth_mm = args.z if args.z is not None else float(ver.hydrophone_position[2])
                time_stop_us = max(
                    cfg.scope.time_stop_us,
                    depth_mm / (cfg.sos_water_m_per_s / 1000.0)
                    + float(resolved["duration_usec"])
                    + Characterization.BURST_WINDOW_MARGIN_US,
                )
            input("Press Enter to start")

            scan_kwargs = dict(
                x_range=tuple(args.x_range),
                num_x=args.num_x,
                y_range=tuple(args.y_range),
                num_y=args.num_y,
                absolute=args.absolute,
                time_start_s=args.time_start_us * 1e-6,
                time_stop_s=time_stop_us * 1e-6,
                sampling_interval_ns=args.sampling_interval_ns,
                chunk_size=args.chunk_size,
                n_averages=args.n_averages,
                align=args.align,
                progress=progress,
            )
            if args.z is not None:
                scan_kwargs["z"] = args.z
            result = ver.scan_2d(**scan_kwargs)
            result.metadata["window_pnp"] = scan_window_pnp(
                result, ver.focus_timing_us,
                duration_us=emitted_duration_us(resolved["duration_usec"],
                                                resolved["frequency_kHz"]),
                frequency_kHz=resolved["frequency_kHz"],
                start_margin_us=Characterization.PNP_WINDOW_START_MARGIN_US,
                end_margin_us=Characterization.PNP_WINDOW_END_MARGIN_US,
            )
    except (ConnectionError, ValueError, Exception) as e:
        logger.error("Scan aborted: %s", e)
        raise

    if args.save_data:
        # Historical location: workspace root (parent of scripts/).
        out = Path(__file__).parent.parent.resolve() / "scan_2d_data.npz"
        result.save(out, save_txt=False)

    if args.plot or args.save_plot:
        result.plot(reduce=args.reduce, show=args.plot, save_as=args.save_plot)


if __name__ == "__main__":
    main()
