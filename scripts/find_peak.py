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
    parser.add_argument("--duration-msec", type=float, default=None,
                        help="Pulse duration (ms). Defaults to 20 cycles.")
    parser.add_argument("--interval-msec", type=float, default=None)
    # --- Search geometry ---
    parser.add_argument("--z", type=float, default=None,
                        help="Depth (mm). Defaults to ver.hydrophone_position[2].")
    parser.add_argument("--x0", type=float, default=None,
                        help="Starting x (mm). Defaults to ver.hydrophone_position[0].")
    parser.add_argument("--y0", type=float, default=None,
                        help="Starting y (mm). Defaults to ver.hydrophone_position[1].")
    parser.add_argument("--initial-step", type=float, default=0.5,
                        help="Initial trial step length (mm).")
    parser.add_argument("--tol", type=float, default=0.02,
                        help="Convergence tolerance (mm).")
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
    parser.add_argument("--no-rotate-basis", action="store_true",
                        help="Keep probes axis-aligned each iteration instead of "
                             "rotating along the accepted gradient direction.")
    parser.add_argument("--time-start-us", type=float, default=-14.0)
    parser.add_argument("--time-stop-us", type=float, default=86.0)
    parser.add_argument("--sampling-interval-ns", type=float, default=100.0)
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
    parser.add_argument("--log-file", type=str, default="")
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
                              calibration_path=args.calibration_path or None) as ver:
            if args.log_file:
                ver.add_log_file(args.log_file)
            ver.apply_pulse(
                frequency_kHz=args.frequency_khz,
                voltage=args.voltage,
                duration_msec=args.duration_msec,
                interval_msec=args.interval_msec,
            )
            ver.enable_hv_output(wait=True)

            input("Press Enter to start")
            t_start = time.perf_counter()
            x_peak, y_peak = ver.find_peak(
                x0=args.x0, y0=args.y0, z=args.z,
                initial_step=args.initial_step,
                tol=args.tol,
                max_iter=args.max_iter,
                hysteresis=args.hysteresis,
                probe_scale=args.probe_scale,
                min_line_step_scale=args.min_line_step_scale,
                rotate_basis=not args.no_rotate_basis,
                time_start_s=args.time_start_us * 1e-6,
                time_stop_s=args.time_stop_us * 1e-6,
                sampling_interval_ns=args.sampling_interval_ns,
                plot=True,
                store=True,
                save=args.save_calibration,
                keep_plot_open=True,
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
