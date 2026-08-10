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
    parser.add_argument("--voltage", type=float, default=12.0,
                        help="TX voltage during peak search.")
    parser.add_argument("--frequency-khz", type=float, default=400.0)
    parser.add_argument("--duration-msec", type=float, default=None,
                        help="Pulse duration (ms). Defaults to 20 cycles.")
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
    parser.add_argument("--no-rotate-basis", action="store_true",
                        help="Keep probes axis-aligned each iteration instead of "
                             "rotating along the accepted gradient direction.")
    parser.add_argument("--time-start-us", type=float, default=100.0)
    parser.add_argument("--time-stop-us", type=float, default=200.0)
    parser.add_argument("--sampling-interval-ns", type=float, default=100.0)
    parser.add_argument("--hydrophone", type=str, default="",
                        help="Hydrophone calibration file or ID (e.g. '2246').")
    parser.add_argument("--calibration-path", type=str,
                        default="hydrophone_position.json",
                        help="Auto-load / save destination for hydrophone_position.")
    parser.add_argument("--save-calibration", action="store_true",
                        help="After convergence, persist the found (x, y, z) "
                             "to --calibration-path.")
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

    frequency_kHz = args.frequency_khz
    duration_msec = (args.duration_msec
                     if args.duration_msec is not None
                     else 20 / frequency_kHz)
    interval_msec = 20

    try:
        with VerificationTank(frequency=int(frequency_kHz),
                              num_modules=1,
                              ext_power_supply=False,
                              hydrophone_range_mv=args.hydro_range_mv,
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
            t_start = time.perf_counter()
            x_peak, y_peak = ver.find_peak(
                x0=args.x0, y0=args.y0, z=args.z,
                initial_step=args.initial_step,
                tol=args.tol,
                max_iter=args.max_iter,
                hysteresis=args.hysteresis,
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
