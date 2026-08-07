"""Voltage-sweep scan using PicoScope rapid-block mode.

Sweeps HV voltage across ``voltages`` at a fixed focus point.
Each iteration updates the HV rail (and waits for settle) then fires
one trigger. Because HV settling adds noticeable per-point latency,
individual triggers still work but the rapid-block infrastructure
keeps scope arm/xfer overhead amortized across the sweep.
"""
import argparse
import logging
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from openlifu_verification import VerificationTank

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v-start", type=float, default=5.0)
    parser.add_argument("--v-stop", type=float, default=60.0,
                        help="Last voltage (inclusive).")
    parser.add_argument("--v-step", type=float, default=5.0)
    parser.add_argument("--chunk-size", type=int, default=0,
                        help="Rapid-block segments per chunk (0 = one chunk for the whole sweep).")
    parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True,
                        help="Show the resulting Vpp-vs-voltage plot interactively (default: on).")
    parser.add_argument("--save-data", action=argparse.BooleanOptionalAction, default=True,
                        help="Save NPZ + TXT data to scripts/data/ (default: on).")
    parser.add_argument("--save-plot", type=str, default="",
                        help="If set, save the plot image to this path (e.g. scan_voltage.png).")
    args = parser.parse_args()

    if not args.plot and not args.save_plot and not args.save_data:
        resp = input("Nothing will be plotted or saved (--no-plot, no --save-plot, --no-save-data). "
                     "Continue anyway? [y/N] ")
        if resp.strip().lower() not in ("y", "yes"):
            logger.info("Aborted.")
            return

    xInput, yInput, zInput = 0, 0, 50
    frequency_kHz = 400
    initial_voltage = args.v_start
    duration_msec = 20 / 400
    interval_msec = 20
    num_modules = 1
    voltages = np.arange(args.v_start, args.v_stop + 0.5 * args.v_step, args.v_step)

    # Note: channel A is set to 5V range here (rather than the default
    # 100mV) because the hydrophone signal at high drive voltages will
    # exceed the low-range clip point.
    hydro_range_mv = 5000
    sampling_interval_ns = 100
    time_start_s = 100e-6
    time_stop_s = 200e-6

    chunk_size = args.chunk_size if args.chunk_size > 0 else len(voltages)

    logger.info("Starting Voltage Scan (%d points, chunk_size=%d)",
                len(voltages), chunk_size)

    outputs = []
    timings = []
    hydro = "A"
    t_wall_start = time.perf_counter()

    try:
        with VerificationTank(frequency=frequency_kHz,
                              num_modules=num_modules,
                              ext_power_supply=False,
                              hydrophone_range_mv=hydro_range_mv) as ver:
            ver.configure_lifu(
                frequency_kHz=frequency_kHz,
                voltage=initial_voltage,
                duration_msec=duration_msec,
                interval_msec=interval_msec,
                pulse_count=1,
                trigger_mode="single",
            )
            ver.set_focus(xInput, yInput, zInput)
            ver.enable_hv_output(wait=True)
            hydro = ver.hydrophone_channel

            input("Press Enter to start")

            def apply_point(voltage):
                ver.set_voltage(float(voltage), wait=True)

            def on_progress(voltage, pt):
                logger.info(
                    "V=%5.1f V  apply=%.4fs  trigger=%.4fs  total=%.4fs",
                    voltage, pt["apply_s"], pt["trigger_s"], pt["iter_total_s"],
                )

            outputs, timings = ver.run_rapid_sweep(
                points=list(voltages),
                apply_point=apply_point,
                time_start_s=time_start_s,
                time_stop_s=time_stop_s,
                sampling_interval_ns=sampling_interval_ns,
                chunk_size=chunk_size,
                progress=on_progress,
            )

    except (ConnectionError, ValueError, Exception) as e:
        logger.error(f"An error occurred: {e}")
        return

    t_wall_end = time.perf_counter()
    logger.info("Finished Voltage Scan (%.3f s wall).", t_wall_end - t_wall_start)

    good = [(t, o, v) for t, o, v in zip(timings, outputs, voltages) if o is not None]
    if not good:
        logger.warning("No data was collected.")
        return
    good_timings, good_outputs, good_voltages = zip(*good)
    good_voltages = np.array(good_voltages)
    t_axis = good_outputs[0]["time"]
    a_channel_outputs = np.array([o[hydro] for o in good_outputs])
    hydro_vpp = np.ptp(a_channel_outputs, axis=1)

    out_path = Path(__file__).parent.resolve() / 'data'
    savedata = {
        "t": t_axis,
        "outputs": a_channel_outputs,
        "voltages": good_voltages,
        "chunk_size": chunk_size,
        "apply_s": np.array([t["apply_s"] for t in good_timings]),
        "trigger_s": np.array([t["trigger_s"] for t in good_timings]),
        "arm_s": np.array([t["arm_s"] for t in good_timings]),
        "xfer_s": np.array([t["xfer_s"] for t in good_timings]),
        "iter_total_s": np.array([t["iter_total_s"] for t in good_timings]),
    }
    if args.save_data:
        out_path.mkdir(exist_ok=True)
        np.savez(out_path / "scan_voltage_data.npz", **savedata)
        logger.info("Data saved to scan_voltage_data.npz")
        txt_path = out_path / "voltage_vs_hydrophone.txt"
        np.savetxt(txt_path, np.column_stack((good_voltages, hydro_vpp)),
                   header="Input_Voltage(V)\tHydrophone_Vpp(mV)", fmt="%.6e")
        logger.info(f"Saved {txt_path}")

    if args.plot or args.save_plot:
        fig, ax = plt.subplots()
        ax.plot(good_voltages, hydro_vpp, ".-")
        ax.set_xlabel("Input voltage (V)")
        ax.set_ylabel("Hydrophone Vpp (mV)")
        ax.set_title(f"Voltage sweep @ {frequency_kHz} kHz")
        ax.grid(True)
        fig.tight_layout()
        if args.save_plot:
            fig.savefig(args.save_plot)
            logger.info("Figure saved to %s", args.save_plot)
        if args.plot:
            plt.show()


if __name__ == "__main__":
    main()
