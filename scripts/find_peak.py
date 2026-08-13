"""CLI wrapper for :meth:`VerificationTank.find_peak`.

Runs a 2-D gradient-ascent search for the hydrophone peak with a
live matplotlib figure and (optionally) persists the located
``(x, y, z)`` to the calibration file so that subsequent scans are
dead-centered on the empirical peak.

Examples::

    python scripts/find_peak.py --hydrophone 2246
    python scripts/find_peak.py --initial-step 0.25 --tol 0.02 --save-calibration
"""
import argparse
import logging
import time

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
    parser.add_argument("--frequency-khz", type=float, default=None,
                        help="TX frequency in kHz.")
    parser.add_argument("--voltage", type=float, default=12.0,
                        help="TX voltage during peak search (V). Peak search "
                             "typically wants a lower voltage than the full "
                             "sweep, so this defaults to 12 V rather than "
                             "VerificationTank.DEFAULT_VOLTAGE_V.")
    parser.add_argument("--duration-usec", type=float, default=None,
                        help="Pulse duration (µs). Defaults to 20 cycles.")
    parser.add_argument("--interval-msec", type=float, default=None)
    # --- Search geometry ---
    parser.add_argument("--z", type=float, default=None,
                        help="Depth (mm). Defaults to ver.hydrophone_position[2].")
    parser.add_argument("--x0", type=float, default=0,
                        help="Starting x (mm). Defaults to 0.")
    parser.add_argument("--y0", type=float, default=0,
                        help="Starting y (mm). Defaults to 0.")
    parser.add_argument("--method", choices=("grid", "gradient"),
                        default="grid",
                        help="Search algorithm. 'grid' (default) walks a "
                             "fixed x/y grid, caches all samples, and "
                             "least-squares-fits a 2-D paraboloid over the "
                             "3x3 window around the bracketed peak. "
                             "'gradient' is the older direct-search + "
                             "subsample-refinement algorithm.")
    # --- grid_walk_search parameters ---
    parser.add_argument("--grid-step", type=float, default=0.2,
                        help="[grid] Grid spacing (mm). Also the side "
                             "length of the paraboloid fit cell. Default "
                             "0.2 mm.")
    parser.add_argument("--max-evaluations", type=int, default=50,
                        help="[grid] Cap on total new measurements. Cached "
                             "grid-node re-visits are free. Default 50.")
    parser.add_argument("--fit-window", type=int, default=1,
                        help="[grid] Radius (in grid nodes) around the "
                             "converged best used for the paraboloid fit. "
                             "1 -> 3x3, 2 -> 5x5. Default 1.")
    # --- gradient_search parameters ---
    parser.add_argument("--initial-step", type=float, default=0.25,
                        help="[gradient] Initial trial step length (mm).")
    parser.add_argument("--tol", type=float, default=0.02,
                        help="[gradient] Convergence tolerance (mm).")
    parser.add_argument("--max-iter", type=int, default=40)
    parser.add_argument("--hysteresis", type=float, default=0.01,
                        help="Required fractional RMS improvement to accept a move.")
    parser.add_argument("--probe-scale", type=float, default=0.5,
                        help="Probe (finite-difference) distance as a fraction "
                             "of the current step. <1 gives a local gradient "
                             "estimate; ~0.5 is a good default.")
    parser.add_argument("--min-line-step-scale", type=float, default=0.05,
                        help="Smallest backtracking line-search step, as a "
                             "fraction of the current step.")
    parser.add_argument("--min-step", type=float, default=0.2,
                        help="Minimum probe spacing (mm) below which the "
                             "grid stops shrinking. This is the roll-off "
                             "scale used for symmetry-centering. Set "
                             "~equal to the transducer spot radius "
                             "(default 0.2 mm = 200 µm). Setting this to "
                             "a very small value restores the old "
                             "micro-peak-hunting behavior.")
    parser.add_argument("--max-polish-iter", type=int, default=6,
                        help="Cap on symmetry-polish iterations (once step "
                             "has reached --min-step) before declaring "
                             "convergence. Prevents endless jitter around "
                             "a noisy top.")
    parser.add_argument("--no-rotate-basis", action="store_true",
                        help="Keep probes axis-aligned each iteration instead of "
                             "rotating along the accepted gradient direction.")
    parser.add_argument("--time-start-us", type=float, default=0.0)
    parser.add_argument("--time-stop-us", type=float, default=200.0)
    parser.add_argument("--sampling-interval-ns", type=float, default=100.0)
    parser.add_argument("-n", "--n-averages", type=int, default=1,
                        help="Number of pulses to fire and coherently "
                             "average at every probe point (default: 1). "
                             "Larger values reduce noise at the cost of "
                             "proportionally more wall-time per iteration.")
    parser.add_argument("--align", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Cross-correlate repeats before averaging "
                             "(only meaningful when --n-averages > 1).")
    parser.add_argument("--pause", action="store_true",
                        help="Pause for a keypress at the end of every "
                             "iteration (after each set of 5 samples: 4 "
                             "probes + 1 refinement). Handy for debugging "
                             "the convergence path.")
    # --- Hydrophone / calibration ---
    parser.add_argument("--hydrophone", type=str, default="",
                        help="Hydrophone calibration file or bare ID (e.g. '2246').")
    parser.add_argument("--calibration-path", type=str,
                        default=str(paths.HYDROPHONE_STATE_PATH),
                        help="JSON file storing hydrophone position + last-used ID.")
    parser.add_argument("--save-calibration", action="store_true",
                        help="After convergence, persist the found (x, y, z) + "
                             "hydrophone ID to --calibration-path.")
    parser.add_argument("--hydro-range-mv", type=int, default=100)
    parser.add_argument("--raw-mv", action="store_true",
                        help="Skip the mV->Pa calibration even if a "
                             "hydrophone ID is saved. Position calibration "
                             "and hydrophone ID still load normally; only "
                             "the reported trace amplitudes stay in mV.")
    parser.add_argument("--log-file", type=str, default="")
    # --- Depth calibration (plane-wave arrival time) ---
    parser.add_argument("--calibrate-depth",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="Run a plane-wave arrival-time measurement "
                             "before the 2-D search to set "
                             "hydrophone_position[2]. Default on. See "
                             "scripts/arrival_time_demo.py for tuning.")
    parser.add_argument("--depth-voltage", type=float, default=30.0,
                        help="[depth-cal] HV rail for the plane-wave "
                             "pulses (default 30 V).")
    parser.add_argument("--depth-pulse-count", type=int, default=32,
                        help="[depth-cal] Pulses averaged (default 32).")
    parser.add_argument("--depth-duration-usec", type=float, default=8.0,
                        help="[depth-cal] Per-pulse duration (default 8 \u00b5s).")
    parser.add_argument("--depth-skip-us", type=float, default=12.0,
                        help="[depth-cal] Ignore samples before this time "
                             "when searching for the first arrival "
                             "(default 12 \u00b5s).")
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

    try:
        with VerificationTank(frequency=tank_frequency,
                              num_modules=1,
                              ext_power_supply=False,
                              hydrophone_range_mv=args.hydro_range_mv,
                              hydrophone=args.hydrophone or None,
                              use_calibration=not args.raw_mv,
                              calibration_path=args.calibration_path or None) as ver:
            if args.log_file:
                ver.add_log_file(args.log_file)

            if args.calibrate_depth:
                logger.info("Running plane-wave depth calibration before 2-D search...")
                depth_result = ver.calibrate_hydrophone_depth(
                    voltage_V=args.depth_voltage,
                    n_pulses=args.depth_pulse_count,
                    duration_usec=args.depth_duration_usec,
                    skip_us=args.depth_skip_us,
                    hydrophone_range_mv=args.hydro_range_mv,
                    store=True,
                    save=False,
                )
                logger.info(
                    "Depth calibration \u2192 z = %.3f mm "
                    "(arrival=%.3f \u00b5s, %d pulses)",
                    depth_result["distance_mm"],
                    depth_result["arrival_us"],
                    depth_result["n_pulses_used"],
                )

            ver.apply_pulse(
                frequency_kHz=args.frequency_khz,
                voltage=args.voltage,
                duration_usec=args.duration_usec,
                interval_msec=args.interval_msec,
            )
            ver.enable_hv_output(wait=True)

            input("Press Enter to start")
            t_start = time.perf_counter()
            x_peak, y_peak = ver.find_peak(
                x0=args.x0, y0=args.y0, z=args.z,
                method=args.method,
                grid_step=args.grid_step,
                max_evaluations=args.max_evaluations,
                fit_window=args.fit_window,
                initial_step=args.initial_step,
                tol=args.tol,
                max_iter=args.max_iter,
                hysteresis=args.hysteresis,
                probe_scale=args.probe_scale,
                min_line_step_scale=args.min_line_step_scale,
                min_step=args.min_step,
                max_polish_iter=args.max_polish_iter,
                rotate_basis=not args.no_rotate_basis,
                time_start_s=args.time_start_us * 1e-6,
                time_stop_s=args.time_stop_us * 1e-6,
                sampling_interval_ns=args.sampling_interval_ns,
                n_averages=args.n_averages,
                align=args.align,
                plot=True,
                store=True,
                save=args.save_calibration,
                keep_plot_open=True,
                pause=args.pause,
            )
            elapsed = time.perf_counter() - t_start
            logger.info(
                "find_peak done in %.2fs. hydrophone_position = %s",
                elapsed, ver.hydrophone_position.tolist(),
            )
            print(f"Peak: x={x_peak:.4f} mm, y={y_peak:.4f} mm")

    except (ConnectionError, ValueError, Exception) as e:
        logger.error("Peak search aborted: %s", e)
        raise


if __name__ == "__main__":
    main()
