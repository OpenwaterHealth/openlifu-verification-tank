"""Frequency-sweep scan using PicoScope rapid-block mode.

Sweeps the TX pulse frequency across ``frequencies`` at a fixed focus
point. Each iteration re-programs the pulse profile via
``ver.set_pulse`` then fires one trigger.
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
    parser.add_argument("--voltage", type=float, default=20.0)
    parser.add_argument("--f-start", type=float, default=370.0,
                        help="First frequency in kHz.")
    parser.add_argument("--f-stop", type=float, default=430.0,
                        help="Last frequency in kHz (inclusive).")
    parser.add_argument("--f-step", type=float, default=5.0,
                        help="Frequency step in kHz.")
    parser.add_argument("--chunk-size", type=int, default=0,
                        help="Rapid-block segments per chunk (0 = one chunk for the whole sweep).")
    parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True,
                        help="Show the resulting Vpp-vs-frequency plot interactively (default: on).")
    parser.add_argument("--save-data", action=argparse.BooleanOptionalAction, default=True,
                        help="Save NPZ + TXT data to scripts/data/ (default: on).")
    parser.add_argument("--save-plot", type=str, default="",
                        help="If set, save the plot image to this path (e.g. scan_freq.png).")
    args = parser.parse_args()

    if not args.plot and not args.save_plot and not args.save_data:
        resp = input("Nothing will be plotted or saved (--no-plot, no --save-plot, --no-save-data). "
                     "Continue anyway? [y/N] ")
        if resp.strip().lower() not in ("y", "yes"):
            logger.info("Aborted.")
            return

    xInput, yInput, zInput = 0, 0, 50
    voltage = args.voltage
    center_frequency_kHz = 400
    duration_msec = 20 / 400
    interval_msec = 20
    num_modules = 1
    frequencies = np.arange(args.f_start, args.f_stop + 0.5 * args.f_step, args.f_step)

    sampling_interval_ns = 100
    time_start_s = 100e-6
    time_stop_s = 200e-6

    chunk_size = args.chunk_size if args.chunk_size > 0 else len(frequencies)

    logger.info("Starting Frequency Scan (%d points, chunk_size=%d)",
                len(frequencies), chunk_size)

    outputs = []
    timings = []
    hydro = "A"
    t_wall_start = time.perf_counter()

    try:
        with VerificationTank(frequency=center_frequency_kHz,
                              num_modules=num_modules,
                              ext_power_supply=False) as ver:
            ver.configure_lifu(
                frequency_kHz=center_frequency_kHz,
                voltage=voltage,
                duration_msec=duration_msec,
                interval_msec=interval_msec,
                pulse_count=1,
                trigger_mode="single",
            )
            ver.set_focus(xInput, yInput, zInput)
            ver.enable_hv_output(wait=True)
            hydro = ver.hydrophone_channel

            input("Press Enter to start")

            def apply_point(freq_kHz):
                ver.set_pulse(frequency_kHz=freq_kHz, duration_msec=duration_msec)

            def on_progress(freq_kHz, pt):
                logger.info(
                    "f=%6.1f kHz  apply=%.4fs  trigger=%.4fs  total=%.4fs",
                    freq_kHz, pt["apply_s"], pt["trigger_s"], pt["iter_total_s"],
                )

            outputs, timings = ver.run_rapid_sweep(
                points=list(frequencies),
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
    logger.info("Finished Frequency Scan (%.3f s wall).", t_wall_end - t_wall_start)

    good = [(t, o, f) for t, o, f in zip(timings, outputs, frequencies) if o is not None]
    if not good:
        logger.warning("No data was collected.")
        return
    good_timings, good_outputs, good_freqs = zip(*good)
    good_freqs = np.array(good_freqs)
    t_axis = good_outputs[0]["time"]
    a_channel_outputs = np.array([o[hydro] for o in good_outputs])
    voltages_vpp = np.ptp(a_channel_outputs, axis=1)

    out_path = Path(__file__).parent.resolve() / 'data'
    savedata = {
        "t": t_axis,
        "outputs": a_channel_outputs,
        "freq": good_freqs,
        "chunk_size": chunk_size,
        "apply_s": np.array([t["apply_s"] for t in good_timings]),
        "trigger_s": np.array([t["trigger_s"] for t in good_timings]),
        "arm_s": np.array([t["arm_s"] for t in good_timings]),
        "xfer_s": np.array([t["xfer_s"] for t in good_timings]),
        "iter_total_s": np.array([t["iter_total_s"] for t in good_timings]),
    }
    if args.save_data:
        out_path.mkdir(exist_ok=True)
        np.savez(out_path / "scan_freq_data.npz", **savedata)
        logger.info("Data saved to scan_freq_data.npz")
        txt_path = out_path / "Scan_freq_voltage.txt"
        np.savetxt(txt_path, np.column_stack((good_freqs, voltages_vpp)),
                   header="Frequency_kHz\tVpp_mV", fmt="%.3f")
        logger.info(f"Raw frequency-Vpp data saved to {txt_path}")

    if args.plot or args.save_plot:
        fig, ax = plt.subplots()
        ax.plot(good_freqs, voltages_vpp, ".-")
        ax.set_xlabel("Frequency (kHz)")
        ax.set_ylabel("Vpp (mV)")
        ax.set_title(f"Frequency sweep @ V={args.voltage} V")
        ax.grid(True)
        fig.tight_layout()
        if args.save_plot:
            fig.savefig(args.save_plot)
            logger.info("Figure saved to %s", args.save_plot)
        if args.plot:
            plt.show()


if __name__ == "__main__":
    main()
