"""Fire a single TX pulse and (optionally) capture it on the PicoScope.

Thin wrapper around :class:`VerificationTank`. All pulse / capture
defaults live on :class:`VerificationTank` — anything unspecified on
the command line falls back to those.
"""
import argparse
import logging

import numpy as np
import matplotlib.pyplot as plt

from openlifu_verification import ScanResult, VerificationTank, paths, set_log_level

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
    parser.add_argument("--frequency-khz", type=float, default=None)
    parser.add_argument("--voltage", type=float, default=12.0,
                        help="HV rail (V). Single-pulse defaults to 12 V for "
                             "safety; override to hit the full drive rail.")
    parser.add_argument("--duration-msec", type=float, default=None)
    parser.add_argument("--interval-msec", type=float, default=None)
    # --- Focus ---
    parser.add_argument("--x", type=float, default=0.0)
    parser.add_argument("--y", type=float, default=0.0)
    parser.add_argument("--z", type=float, default=50.0)
    # --- Capture ---
    parser.add_argument("--no-scope", action="store_true",
                        help="Do not open the Picoscope. Fire the TX pulse only "
                             "so an external scope application can capture it.")
    parser.add_argument("--time-start-us", type=float, default=100.0)
    parser.add_argument("--time-stop-us", type=float, default=220.0)
    parser.add_argument("--sampling-interval-ns", type=float, default=100.0)
    parser.add_argument("--hydro-range-mv", type=int, default=100)
    parser.add_argument("--timeout-s", type=float, default=3.0)
    # --- Hydrophone ---
    parser.add_argument("--hydrophone", type=str, default="",
                        help="Hydrophone calibration file or bare ID. Falls "
                             "back to the last-used ID stored in --calibration-path.")
    parser.add_argument("--calibration-path", type=str,
                        default=str(paths.HYDROPHONE_STATE_PATH))
    parser.add_argument("--num-modules", type=int, default=1)
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

    logger.info("Starting Single Pulse Script...")
    result = None
    scan_result = None
    try:
        with VerificationTank(frequency=tank_frequency,
                              num_modules=args.num_modules,
                              use_picoscope=not args.no_scope,
                              hydrophone=args.hydrophone or None,
                              calibration_path=args.calibration_path or None,
                              hydrophone_range_mv=args.hydro_range_mv,
                              ext_power_supply=False) as ver:
            resolved = ver.apply_pulse(
                frequency_kHz=args.frequency_khz,
                voltage=args.voltage,
                duration_msec=args.duration_msec,
                interval_msec=args.interval_msec,
            )
            ver.set_focus(args.x, args.y, args.z)
            ver.enable_hv_output(wait=True)

            input("Press Enter to start")

            if args.no_scope:
                ver.run_trigger()
            else:
                result = ver.run_capture(
                    time_start_s=args.time_start_us * 1e-6,
                    time_stop_s=args.time_stop_us * 1e-6,
                    sampling_interval_ns=args.sampling_interval_ns,
                    timeout_s=args.timeout_s,
                )
                if result is not None:
                    trace_mv = np.asarray(result[ver.hydrophone_channel])
                    if ver.hydrophone is not None:
                        trace = ver.hydrophone.mv_to_pa(trace_mv, ver.frequency * 1e3)
                        units = "Pa"
                    else:
                        trace = trace_mv
                        units = "mV"
                    scan_result = ScanResult(
                        scan_type="single_pulse",
                        t=result["time"],
                        traces=trace[None, :],
                        coords={"pulse": np.array([0])},
                        hydrophone_channel=ver.hydrophone_channel,
                        units=units,
                        metadata={
                            "voltage_V": float(resolved["voltage"]),
                            "frequency_kHz": float(resolved["frequency_kHz"]),
                            "focus_mm": np.array([args.x, args.y, args.z], dtype=float),
                        },
                    )
    except (ConnectionError, ValueError, Exception) as e:
        logger.error(f"An error occurred: {e}")
        return

    logger.info("Finished Single Pulse.")

    if args.no_scope:
        return
    if result is None or scan_result is None:
        logger.warning("No pulse captured within the timeout window.")
        return
    scan_result.plot(kind="trace", show=True)


if __name__ == "__main__":
    main()
