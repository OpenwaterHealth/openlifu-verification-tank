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
    zInput = 50
    xfoci =  np.linspace(-10, 10, 41)
    yfoci = [0]

    frequency_kHz = 400
    voltage = 20.0
    duration_msec = 20 / 400
    interval_msec = 20
    num_modules = 1

    logger.info("Starting Lateral Scan Script...")
    try:
        with VerificationTank(frequency=frequency_kHz, num_modules=num_modules) as ver:
            # Configure LIFU and HVPS
            ver.configure_lifu(
                frequency_kHz=frequency_kHz,
                voltage=voltage,
                duration_msec=duration_msec,
                interval_msec=interval_msec
            )

            # Configure Picoscope
            ver.scope.set_channel('A', range_mv=100, coupling='DC')
            ver.scope.set_channel('B', range_mv=5000, coupling='DC')
            ver.scope.set_trigger(channel='A', threshold_mv=-2, direction='falling')

            # Enable power supply
            ver.hv.set_all_outputs(True)

            s = input("Press any key to start")

            outputs = []
            for yfocus in yfoci:
                for xfocus in xfoci:
                    logger.info(f"{xfocus=}")
                    ver.set_focus(xfocus, yfocus, zInput)
                    data = ver.run_capture()
                    outputs.append(data)

            # Stop the trigger manually after the scan is complete
            ver.lifu.txdevice.stop_trigger()

    except (ConnectionError, ValueError, Exception) as e:
        logger.error(f"An error occurred: {e}")
        return # Exit gracefully

    logger.info("Finished Lateral Scan.")
    if outputs:
        # Process and save data
        t = outputs[0]["time"]
        a_channel_outputs = np.array([output["A"] for output in outputs]).reshape([len(yfoci), len(xfoci), -1])
        savedata = {'t': t, "outputs": a_channel_outputs, "xfoci": xfoci, "yfoci": yfoci}
        out_path = Path(__file__).parent.resolve() / 'data'
        np.savez(out_path / "scan_lat_data.npz", **savedata)
        logger.info("Data saved to scan_lat_data.npz")

        voltages_vpp = [np.ptp(outputs["A"]) for outputs in outputs]
        positions = [(xfocus, yfocus) for yfocus in yfoci for xfocus in xfoci]
        positions = np.array(positions, dtype=float)
        txt_path = out_path / "Scan_lat_voltage.txt"



        data_txt = np.column_stack((np.array(positions), np.array(voltages_vpp)))
        header = "x_focus(mm)\tyfocus(mm)\tVpp(mV)"
        np.savetxt(txt_path, data_txt, header = header, fmt="%.6e", delimiter="\t")
        logger.info(f"Peak-to_peak voltage data saved to {txt_path}")

    else:
        logger.warning("No data was collected.")


if __name__ == "__main__":
    main()
