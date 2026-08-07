"""Lateral (x) scan using PicoScope rapid-block mode.

Sweeps focus x across ``xfoci`` at each y in ``yfoci`` (default 1D
line at y=0). Uses ``VerificationTank.run_rapid_sweep`` so all
captures in each chunk go over USB in a single bulk transfer.
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
    parser.add_argument("--num-x", type=int, default=41,
                        help="Number of x points across the -10..10 mm sweep.")
    parser.add_argument("--chunk-size", type=int, default=0,
                        help="Rapid-block segments per chunk (0 = whole sweep in one chunk).")
    parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True,
                        help="Show the resulting Vpp-vs-x plot interactively (default: on).")
    parser.add_argument("--save-data", action=argparse.BooleanOptionalAction, default=True,
                        help="Save NPZ + TXT data to scripts/data/ (default: on).")
    parser.add_argument("--save-plot", type=str, default="",
                        help="If set, save the plot image to this path (e.g. scan_lat.png).")
    args = parser.parse_args()

    if not args.plot and not args.save_plot and not args.save_data:
        resp = input("Nothing will be plotted or saved (--no-plot, no --save-plot, --no-save-data). "
                     "Continue anyway? [y/N] ")
        if resp.strip().lower() not in ("y", "yes"):
            logger.info("Aborted.")
            return

    zInput = 50
    xfoci = np.linspace(-10, 10, args.num_x)
    yfoci = [0]

    frequency_kHz = 400
    voltage = args.voltage
    duration_msec = 20 / 400
    interval_msec = 20
    num_modules = 1

    sampling_interval_ns = 100
    time_start_s = 100e-6
    time_stop_s = 200e-6
    timeout_s = None  # rapid-block: wait indefinitely

    focus_points = [(float(x), float(y), zInput) for y in yfoci for x in xfoci]
    chunk_size = args.chunk_size if args.chunk_size > 0 else len(focus_points)

    logger.info("Starting Lateral Scan (%d points, chunk_size=%d)",
                len(focus_points), chunk_size)

    outputs = []
    timings = []
    hydro = "A"
    t_wall_start = time.perf_counter()
    t_setup_end = None
    t_teardown_start = None
    t_teardown_end = None

    try:
        t_setup_start = time.perf_counter()
        with VerificationTank(frequency=frequency_kHz,
                              num_modules=num_modules,
                              ext_power_supply=False) as ver:
            ver.configure_lifu(
                frequency_kHz=frequency_kHz,
                voltage=voltage,
                duration_msec=duration_msec,
                interval_msec=interval_msec,
                pulse_count=1,
                trigger_mode="single",
            )
            ver.enable_hv_output(wait=True)
            hydro = ver.hydrophone_channel
            t_setup_end = time.perf_counter()

            input("Press Enter to start")

            def apply_point(point):
                x, y, z = point
                ver.set_focus(x, y, z)

            def on_progress(point, pt):
                x, y, _ = point
                logger.info(
                    "x=%+6.2f y=%+6.2f  apply=%.4fs  trigger=%.4fs  total=%.4fs",
                    x, y, pt["apply_s"], pt["trigger_s"], pt["iter_total_s"],
                )

            outputs, timings = ver.run_rapid_sweep(
                points=focus_points,
                apply_point=apply_point,
                time_start_s=time_start_s,
                time_stop_s=time_stop_s,
                sampling_interval_ns=sampling_interval_ns,
                chunk_size=chunk_size,
                timeout_s=timeout_s,
                progress=on_progress,
            )

            t_teardown_start = time.perf_counter()
        t_teardown_end = time.perf_counter()

    except (ConnectionError, ValueError, Exception) as e:
        logger.error(f"An error occurred: {e}")
        return

    t_wall_end = time.perf_counter()

    if timings:
        apply_arr = np.array([t["apply_s"] for t in timings])
        trigger_arr = np.array([t["trigger_s"] for t in timings])
        arm_arr = np.array([t["arm_s"] for t in timings])
        xfer_arr = np.array([t["xfer_s"] for t in timings])
        iter_arr = np.array([t["iter_total_s"] for t in timings])
        n_captured = sum(1 for t in timings if t["captured"])
        n_chunks = 1 + max((t["chunk_index"] for t in timings), default=0)

        logger.info("--- Timing summary ---")
        logger.info("Wall time:                %.3f s", t_wall_end - t_wall_start)
        logger.info("Setup (open + configure): %.3f s", t_setup_end - t_setup_start)
        if t_teardown_end is not None:
            logger.info("Teardown (__exit__):      %.3f s",
                        t_teardown_end - t_teardown_start)
        logger.info("Iterations:               %d (captured %d)  chunks=%d",
                    len(timings), n_captured, n_chunks)
        logger.info("apply:       mean=%.4fs  total=%.3fs",
                    apply_arr.mean(), apply_arr.sum())
        logger.info("trigger:     mean=%.4fs  total=%.3fs",
                    trigger_arr.mean(), trigger_arr.sum())
        logger.info("arm/point:   mean=%.4fs  total=%.3fs",
                    arm_arr.mean(), arm_arr.sum())
        logger.info("xfer/point:  mean=%.4fs  total=%.3fs",
                    xfer_arr.mean(), xfer_arr.sum())
        logger.info("iter total:  mean=%.4fs  total=%.3fs",
                    iter_arr.mean(), iter_arr.sum())

    logger.info("Finished Lateral Scan.")

    good = [(t, o, p) for t, o, p in zip(timings, outputs, focus_points) if o is not None]
    if not good:
        logger.warning("No data was collected.")
        return
    good_timings, good_outputs, good_points = zip(*good)
    t_axis = good_outputs[0]["time"]
    a_channel_outputs = np.array(
        [o[hydro] for o in good_outputs]
    ).reshape([len(yfoci), len(xfoci), -1])
    positions = np.array([(p[0], p[1]) for p in good_points], dtype=float)
    voltages_vpp = np.array([np.ptp(o[hydro]) for o in good_outputs])
    savedata = {
        "t": t_axis,
        "outputs": a_channel_outputs,
        "xfoci": xfoci,
        "yfoci": yfoci,
        "chunk_size": chunk_size,
        "apply_s": np.array([t["apply_s"] for t in good_timings]),
        "trigger_s": np.array([t["trigger_s"] for t in good_timings]),
        "arm_s": np.array([t["arm_s"] for t in good_timings]),
        "xfer_s": np.array([t["xfer_s"] for t in good_timings]),
        "iter_total_s": np.array([t["iter_total_s"] for t in good_timings]),
    }

    if args.save_data:
        out_path = Path(__file__).parent.resolve() / 'data'
        out_path.mkdir(exist_ok=True)
        np.savez(out_path / "scan_lat_data.npz", **savedata)
        logger.info("Data saved to scan_lat_data.npz")
        txt_path = out_path / "Scan_lat_voltage.txt"
        data_txt = np.column_stack((positions, voltages_vpp))
        np.savetxt(txt_path, data_txt,
                   header="x_focus(mm)\tyfocus(mm)\tVpp(mV)",
                   fmt="%.6e", delimiter="\t")
        logger.info(f"Peak-to-peak voltage data saved to {txt_path}")

    if args.plot or args.save_plot:
        vpp_grid = np.ptp(a_channel_outputs, axis=-1)  # (Y, X)
        fig, ax = plt.subplots()
        if vpp_grid.shape[0] == 1:
            ax.plot(xfoci, vpp_grid[0], ".-")
            ax.set_xlabel("x focus (mm)")
            ax.set_ylabel("Vpp (mV)")
            ax.set_title(f"Lateral scan @ y=0, z={zInput} mm, V={voltage} V")
            ax.grid(True)
        else:
            im = ax.imshow(vpp_grid, aspect="auto", origin="lower",
                           extent=[xfoci[0], xfoci[-1], yfoci[0], yfoci[-1]])
            ax.set_xlabel("x focus (mm)")
            ax.set_ylabel("y focus (mm)")
            ax.set_title(f"Lateral scan Vpp (mV) @ z={zInput} mm, V={voltage} V")
            fig.colorbar(im, ax=ax, label="Vpp (mV)")
        fig.tight_layout()
        if args.save_plot:
            fig.savefig(args.save_plot)
            logger.info("Figure saved to %s", args.save_plot)
        if args.plot:
            plt.show()


if __name__ == "__main__":
    main()
