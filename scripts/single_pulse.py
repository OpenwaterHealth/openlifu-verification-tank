import argparse
import logging
from pathlib import Path
import numpy as np
from openlifu_verification import VerificationTank
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
    voltage = 12.0
    duration_msec = 10 / frequency_kHz
    interval_msec = 10
    num_modules = 1

    result = None

    logger.info("Starting Single Pulse Script...")
    try:
        with VerificationTank(frequency=frequency_kHz,
                              num_modules=num_modules,
                              use_picoscope=not args.no_scope,
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
                ver.lifu.start_sonication()
            else:
                result = ver.run_capture(
                    time_start_s=100e-6,
                    time_stop_s=200e-6,
                    sampling_interval_ns=100,
                    timeout_s=3.0,
                )

    except (ConnectionError, ValueError, Exception) as e:
        logger.error(f"An error occurred: {e}")
        return  # Exit gracefully

    logger.info("Finished Single Pulse.")

    if args.no_scope:
        return

    if result is None:
        logger.warning("No pulse captured within the timeout window.")
        return
    if result:
        # Plot data. `time` is in ns relative to the trigger event.
        plt.plot(result["time"]*1e-3, result[ver.hydrophone_channel])
        plt.xlabel('Time (us)')
        plt.ylabel('Voltage (mV)')
        plt.show()
    else:
        logger.warning("No data was collected.")


if __name__ == "__main__":
    main()
