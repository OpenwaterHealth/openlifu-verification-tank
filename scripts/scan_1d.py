"""CLI wrapper: generic 1-D focus scan.

Delegates all the heavy lifting to
:meth:`openlifu_verification.VerificationTank.scan_1d`. Sweeps the
focus along a single axis (``--dim x``, ``y``, or ``z``) with the other
two coords held fixed at either the calibrated hydrophone position
(default) or absolute values.

If you want a 2-D grid, use ``scan_2d.py``.

Defaults match the characterization report's beam scans (pulse length,
voltage, per-axis extent / points, averaging, scope range and capture
window) and the plot shows PNP like the report figures. The scan is
centered on the saved hydrophone position, so run
``find_peak.py --save-calibration`` first.

Examples::

    python scripts/scan_1d.py --dim z                      # report's axial scan
    python scripts/scan_1d.py --dim x                      # report's lateral scan
    python scripts/scan_1d.py --dim y --num 41 --range -5 5
    python scripts/scan_1d.py --dim x --no-plot            # data-only headless run
"""
import argparse
import logging
from pathlib import Path

import numpy as np

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
    # --- Pulse / drive (defaults live on VerificationTank; None = use default) ---
    parser.add_argument("--frequency-khz", type=float, default=None,
                        help="TX frequency in kHz "
                             "(default: VerificationTank.DEFAULT_FREQUENCY_KHZ).")
    parser.add_argument("--voltage", type=float, default=None,
                        help="HV rail in V "
                             "(default: VerificationTank.DEFAULT_VOLTAGE_BY_FREQ_KHZ).")
    parser.add_argument("--duration-usec", type=float, default=None,
                        help="Pulse duration in µs (default: the report's "
                             "pulse for --frequency-khz).")
    parser.add_argument("--interval-msec", type=float, default=None,
                        help="Pulse interval in ms "
                             "(default: VerificationTank.DEFAULT_INTERVAL_MSEC).")
    # --- Scan geometry ---
    parser.add_argument("--dim", type=str, choices=["x", "y", "z"], default="x",
                        help="Which axis to sweep.")
    parser.add_argument("--range", type=float, nargs=2, default=None,
                        dest="scan_range",
                        help="(min, max) in mm along the swept axis "
                             "(default: the report's +/- extent for --dim).")
    parser.add_argument("--num", type=int, default=None,
                        help="Number of samples across --range "
                             "(default: the report's point count for --dim).")
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
                        default=str(setup_data.SetupData.active().hydrophone_state_path),
                        help="JSON file storing hydrophone position + last-used ID "
                             "(auto-loaded if present).")
    parser.add_argument("--raw-mv", action="store_true",
                        help="Skip the mV->Pa calibration even if a "
                             "hydrophone ID is saved. Position calibration "
                             "and hydrophone ID still load normally; only "
                             "the trace/rms/vpp reporting stays in mV.")
    # --- Capture ---
    parser.add_argument("--time-start-us", type=float, default=cfg.scope.time_start_us)
    parser.add_argument("--time-stop-us", type=float, default=None,
                        help="Capture stop (µs). Default: like the report, "
                             f"max({cfg.scope.time_stop_us:g}, deepest arrival + "
                             "pulse + margin).")
    parser.add_argument("--sampling-interval-ns", type=float,
                        default=cfg.scope.sampling_interval_ns)
    parser.add_argument("--hydro-range-mv", type=int,
                        default=cfg.scope.hydrophone_range_mv)
    # --- Misc ---
    parser.add_argument("--chunk-size", type=int, default=0,
                        help="Rapid-block chunk size (0 = whole sweep).")
    parser.add_argument("--n-averages", type=int, default=cfg.scope.n_averages,
                        help="Repeat every scan point this many times and "
                             "coherently average the traces (default: %(default)s).")
    parser.add_argument("--align", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Cross-correlate repeats against the first "
                             "before averaging (default on).")
    parser.add_argument("--reduce", choices=["pnp", "vpp", "vmin", "vmax"],
                        default="pnp",
                        help="Per-point value to plot (default: PNP, as in the report).")
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
    if args.duration_usec is None:
        args.duration_usec = Characterization.pulse_duration_us_for(tank_frequency)
    if args.scan_range is None or args.num is None:
        plan = cfg.plan_for(tank_frequency)
        axis = {"x": plan.lateral_1d, "y": plan.elevation_1d, "z": plan.axial_1d}[args.dim]
        if args.scan_range is None:
            args.scan_range = [-axis.extent_mm, axis.extent_mm]
        if args.num is None:
            args.num = axis.points

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
                origin_z = 0.0 if args.absolute else float(ver.hydrophone_position[2])
                if args.dim == "z":
                    deepest_mm = origin_z + max(args.scan_range)
                else:
                    deepest_mm = args.z if args.z is not None else float(ver.hydrophone_position[2])
                time_stop_us = max(
                    cfg.scope.time_stop_us,
                    deepest_mm / (cfg.sos_water_m_per_s / 1000.0)
                    + float(resolved["duration_usec"])
                    + Characterization.BURST_WINDOW_MARGIN_US,
                )
            input("Press Enter to start")

            scan_kwargs = dict(
                dim=args.dim,
                scan_range=tuple(args.scan_range),
                num=args.num,
                x=args.x,
                y=args.y,
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
            result = ver.scan_1d(**scan_kwargs)
            # One PNP per repeat; the point's PNP is their median.
            pnp = scan_window_pnp(
                result, ver.focus_timing_us,
                duration_us=emitted_duration_us(resolved["duration_usec"],
                                                resolved["frequency_kHz"]),
                frequency_kHz=resolved["frequency_kHz"],
                start_margin_us=Characterization.PNP_WINDOW_START_MARGIN_US,
                end_margin_us=Characterization.PNP_WINDOW_END_MARGIN_US,
                traces=result.metadata["repeat_traces"],
            )
            result.metadata["window_pnp"] = np.nanmedian(pnp, axis=-1)
    except (ConnectionError, ValueError, Exception) as e:
        logger.error("Scan aborted: %s", e)
        raise

    if args.save_data:
        out = Path(__file__).parent.resolve() / "data" / f"scan_1d_{args.dim}_data.npz"
        result.save(out)

    if args.plot or args.save_plot:
        result.plot(reduce=args.reduce, show=args.plot, save_as=args.save_plot)


if __name__ == "__main__":
    main()
