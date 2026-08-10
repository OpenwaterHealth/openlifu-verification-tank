import argparse
import logging
from pathlib import Path
import numpy as np
from openlifu_verification import VerificationTank, ScanResult
import matplotlib.pyplot as plt

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Prevent duplicate handlers and cluttered terminal output
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False

def main():
    parser = argparse.ArgumentParser(description="Fire a single TX pulse.")
    parser.add_argument(
        "--no-scope",
        action="store_true",
        help="Do not open the Picoscope. Fire the TX pulse only so an "
             "external scope application can capture it.",
    )
    args = parser.parse_args()

    # Parameters
    xInput = 0
    yInput = 0
    zInput = 50

    frequency_kHz = 400
    voltage = 36.0
    duration_msec = 5 / frequency_kHz
    interval_msec = 10
    num_modules = 1

    result = None
    scan_result = None

    logger.info("Starting Single Pulse Script...")
    try:
        with VerificationTank(frequency=frequency_kHz,
                              num_modules=num_modules,
                              use_picoscope=not args.no_scope,
                              hydrophone='2246',
                              ext_power_supply=False) as ver:
            # Configure LIFU and HVPS
            ver.configure_lifu(
                frequency_kHz=frequency_kHz,
                voltage=voltage,
                duration_msec=duration_msec,
                interval_msec=interval_msec,
                pulse_count=1,
                trigger_mode="single",
            )
            ver.set_focus(xInput, yInput, zInput)

            if not args.no_scope:
                # Scope channels + trigger are auto-configured with
                # sensible defaults during VerificationTank.__enter__.
                # Override the hydrophone vertical range here if needed.
                ver.set_hydrophone_range(range_mv=100)

            # Enable power supply
            ver.enable_hv_output(wait=True)

            input("Press Enter to start")

            if args.no_scope:
                ver.run_trigger()
            else:
                result = ver.run_capture(
                    time_start_s=50e-6,
                    time_stop_s=200e-6,
                    sampling_interval_ns=100,
                    timeout_s=3.0,
                )
                if result is not None:
                    # Wrap the single capture as a ScanResult so the
                    # units-aware .plot() picks Pa when a hydrophone
                    # calibration is attached and mV otherwise.
                    trace_mv = np.asarray(result[ver.hydrophone_channel])
                    if ver.hydrophone is not None:
                        trace = ver.hydrophone.mv_to_pa(
                            trace_mv, ver.frequency * 1e3
                        )
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
                            "voltage_V": float(voltage),
                            "frequency_kHz": float(frequency_kHz),
                            "focus_mm": np.array([xInput, yInput, zInput], dtype=float),
                        },
                    )

    except (ConnectionError, ValueError, Exception) as e:
        logger.error(f"An error occurred: {e}")
        return  # Exit gracefully

    logger.info("Finished Single Pulse.")

    if args.no_scope:
        return

    if result is None or scan_result is None:
        logger.warning("No pulse captured within the timeout window.")
        return

    scan_result.plot(kind="trace", show=True)


if __name__ == "__main__":
    main()
