import logging
from pathlib import Path
import numpy as np
from openlifu_verification import VerificationTank

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
    # Parameters
    xInput = 0
    yInput = 0
    zInput = 50
    voltage = 20
    frequencies = np.arange(370, 431, 5)
    frequency_kHz = 400
    initial_voltage = 10.0
    duration_msec = 20 / 400
    interval_msec = 20
    num_modules = 1

    logger.info("Starting Frequency Scan Script...")
    try:
        with VerificationTank(frequency=frequency_kHz, num_modules=num_modules) as ver:
            # Configure LIFU and HVPS
            ver.configure_lifu(
                frequency_kHz=frequency_kHz,
                voltage=initial_voltage,
                duration_msec=duration_msec,
                interval_msec=interval_msec
            )
            ver.set_focus(xInput, yInput, zInput)

            # Configure Picoscope
            ver.scope.set_channel('A', range_mv=100, coupling='DC')
            ver.scope.set_channel('B', range_mv=5000, coupling='DC')
            ver.scope.set_trigger(channel='A', threshold_mv=-2, direction='falling')

            # Enable power supply
            ver.set_voltage(voltage)
            ver.hv.set_all_outputs(True)
            ver.hv.wait_ready()

            s = input("Press any key to start")

            outputs = []
            for frequency in frequencies:
                print(f"{frequency=}")
                ver.set_pulse(frequency_kHz=frequency, duration_msec=duration_msec)                
                print("Capturing...")
                data = ver.run_capture(pre_trigger_samples=100, post_trigger_samples=1500)
                print("Complete")
                outputs.append(data)

            # Stop the trigger manually after the scan is complete
            ver.lifu.txdevice.stop_trigger()

    except (ConnectionError, ValueError, Exception) as e:
        logger.error(f"An error occurred: {e}")
        return # Exit gracefully

    logger.info("Finished Frequency Scan.")
    if outputs:
        # Process and save data
        t = outputs[0]["time"]
        a_channel_outputs = np.array([output["A"] for output in outputs]).reshape([len(frequencies), -1])
        voltages_vpp = np.ptp(a_channel_outputs, axis=1)
        out_path = Path(__file__).parent.resolve() / 'data'
        savedata = {'t': t, "outputs": a_channel_outputs, "freq": frequencies}
        np.savez(out_path / "scan_freq_data.npz", **savedata)
        logger.info("Data saved to scan_freq_data.npz")

        txt_path = out_path / "Scsn_freq_voltsge.txt"
        np.savetxt(txt_path, np.column_stack((frequencies, voltages_vpp)), header="Frequency_kHz\tVpp_mV", fmt="%.3f")
        logger.info(f"Raw frequency-Vpp data saved to {txt_path}")
    else:
        logger.warning("No data was collected.")

if __name__ == "__main__":
    main()
