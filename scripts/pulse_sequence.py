"""Fire a LIFU-generated pulse train and inspect per-pulse alignment.

Like :mod:`single_pulse` but exercises the sequence's ``pulse_count``:
one trigger fires ``N`` consecutive pulses spaced by ``interval_msec``.
The Picoscope is armed in rapid-block mode for ``N`` segments so
every internal pulse gets its own trace. Useful for evaluating
firmware inter-pulse jitter, amplitude drift, and steering-delay
consistency.

Example::

    python scripts/pulse_sequence.py --pulse-count 8 --interval-msec 20
"""
import argparse
import logging

import numpy as np

from openlifu_verification import ScanResult, VerificationTank, paths, set_log_level

logger = logging.getLogger(__name__)


def _configure_root_logger():
    root = logging.getLogger()
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        root.addHandler(h)
    root.setLevel(logging.INFO)


def _find_arrival_us(t_ns, trace, *, skip_us, threshold_frac=0.25):
    """Return the first time (µs) after ``skip_us`` where |trace| crosses
    ``threshold_frac * max(|trace|)``. ``None`` if the trace is flat."""
    t_us = np.asarray(t_ns) * 1e-3
    mask = t_us >= skip_us
    if not mask.any():
        return None
    seg = np.abs(np.asarray(trace)[mask])
    peak = float(seg.max())
    if peak <= 0:
        return None
    thresh = threshold_frac * peak
    idx = np.argmax(seg >= thresh)
    if seg[idx] < thresh:
        return None
    return float(t_us[mask][idx])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    # --- Pulse / drive (None => VerificationTank class defaults) ---
    parser.add_argument("--frequency-khz", type=float, default=None)
    parser.add_argument("--voltage", type=float, default=12.0,
                        help="HV rail (V). Defaults to 12 V for safety; "
                             "override to hit the full drive rail.")
    parser.add_argument("--duration-msec", type=float, default=None,
                        help="Per-pulse duration in ms (default: 20 cycles).")
    parser.add_argument("--interval-msec", type=float, default=None,
                        help="Interval between pulses in the train (ms). "
                             "Default: VerificationTank.DEFAULT_INTERVAL_MSEC.")
    parser.add_argument("--pulse-count", "-n", type=int, default=8,
                        help="Number of pulses to fire in one trigger.")
    # --- Focus ---
    parser.add_argument("--x", type=float, default=0.0)
    parser.add_argument("--y", type=float, default=0.0)
    parser.add_argument("--z", type=float, default=50.0)
    # --- Capture ---
    parser.add_argument("--time-start-us", type=float, default=-14.0,
                        help="Capture window start (µs), relative to the "
                             "start of ultrasound emission (see "
                             "VerificationTank.run_capture).")
    parser.add_argument("--time-stop-us", type=float, default=106.0,
                        help="Capture window stop (µs).")
    parser.add_argument("--sampling-interval-ns", type=float, default=100.0)
    parser.add_argument("--hydro-range-mv", type=int, default=100)
    parser.add_argument("--timeout-s", type=float, default=None,
                        help="Rapid-block wait timeout. Default: "
                             "pulse_count * interval + 2 s.")
    # --- Hydrophone ---
    parser.add_argument("--hydrophone", type=str, default="",
                        help="Hydrophone calibration file or bare ID.")
    parser.add_argument("--calibration-path", type=str,
                        default=str(paths.HYDROPHONE_STATE_PATH))
    parser.add_argument("--num-modules", type=int, default=1)
    # --- Plot / output ---
    parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-plot", type=str, default="",
                        help="Optional PNG output for the overlaid-traces figure.")
    # --- Aggregation ---
    parser.add_argument("--aggregate", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Average across pulses after alignment (default on).")
    parser.add_argument("--align", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Cross-correlate against pulse 0 before averaging "
                             "to remove sub-sample timing jitter (default on).")
    parser.add_argument("--align-max-shift-ns", type=float, default=500.0,
                        help="Bound the cross-correlation lag search (ns). "
                             "Default 500 ns covers typical firmware jitter.")
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument("--verbose", "-v", action="store_true")
    verbosity.add_argument("--quiet", "-q", action="store_true")
    args = parser.parse_args()

    _configure_root_logger()
    if args.verbose:
        set_log_level("DEBUG", sdk_level="DEBUG")
    elif args.quiet:
        set_log_level("WARNING")

    if args.pulse_count < 1:
        parser.error("--pulse-count must be >= 1")

    tank_frequency = int(args.frequency_khz
                         if args.frequency_khz is not None
                         else VerificationTank.DEFAULT_FREQUENCY_KHZ)

    logger.info("Starting Pulse Sequence Script (%d pulses)...", args.pulse_count)
    scan_result = None
    resolved = None
    bulk = None
    try:
        with VerificationTank(frequency=tank_frequency,
                              num_modules=args.num_modules,
                              hydrophone=args.hydrophone or None,
                              calibration_path=args.calibration_path or None,
                              hydrophone_range_mv=args.hydro_range_mv,
                              ext_power_supply=False) as ver:
            resolved = ver.apply_pulse(
                frequency_kHz=args.frequency_khz,
                voltage=args.voltage,
                duration_msec=args.duration_msec,
                interval_msec=args.interval_msec,
                pulse_count=args.pulse_count,
                trigger_mode="single",
            )
            ver.set_focus(args.x, args.y, args.z)
            ver.enable_hv_output(wait=True)

            # A conservative default: enough time for all pulses to fire
            # plus a couple of seconds of USB/scope margin.
            timeout_s = args.timeout_s
            if timeout_s is None:
                train_s = args.pulse_count * resolved["interval_msec"] * 1e-3
                timeout_s = train_s + 2.0

            input("Press Enter to start")

            bulk = ver.capture_pulse_train(
                n_pulses=args.pulse_count,
                time_start_s=args.time_start_us * 1e-6,
                time_stop_s=args.time_stop_us * 1e-6,
                sampling_interval_ns=args.sampling_interval_ns,
                timeout_s=timeout_s,
            )
            if bulk is None:
                logger.warning(
                    "Rapid-block capture timed out waiting for all %d pulses.",
                    args.pulse_count,
                )
            else:
                traces_mv = np.asarray(bulk[ver.hydrophone_channel])
                if ver.hydrophone is not None:
                    traces = ver.hydrophone.mv_to_pa(traces_mv, ver.frequency * 1e3)
                    units = "Pa"
                else:
                    traces = traces_mv
                    units = "mV"
                scan_result = ScanResult(
                    scan_type="pulse_sequence",
                    t=bulk["time"],
                    traces=traces,
                    coords={"pulse_index": np.arange(args.pulse_count)},
                    hydrophone_channel=ver.hydrophone_channel,
                    units=units,
                    metadata={
                        "voltage_V": float(resolved["voltage"]),
                        "frequency_kHz": float(resolved["frequency_kHz"]),
                        "interval_msec": float(resolved["interval_msec"]),
                        "duration_msec": float(resolved["duration_msec"]),
                        "focus_mm": np.array([args.x, args.y, args.z], dtype=float),
                    },
                )
    except (ConnectionError, ValueError, Exception) as e:
        logger.error("An error occurred: %s", e)
        return

    logger.info("Finished Pulse Sequence.")

    if scan_result is None:
        return

    # --- Per-pulse timing / amplitude analysis ---
    t_ns = np.asarray(scan_result.t)
    traces = np.asarray(scan_result.traces)
    n_pulses = traces.shape[0]
    vpp = np.ptp(traces, axis=-1)
    rms = np.sqrt(np.mean(traces.astype(float)**2, axis=-1))
    arrivals_us = np.array([
        _find_arrival_us(t_ns, traces[i], skip_us=0.0) or float("nan")
        for i in range(n_pulses)
    ])

    logger.info("Per-pulse summary (%s):", scan_result.units)
    logger.info("  pulse  arrival_us     Vpp        RMS")
    for i in range(n_pulses):
        arr = arrivals_us[i]
        arr_str = f"{arr:8.3f}" if np.isfinite(arr) else "     nan"
        logger.info("  %5d   %s   %.4g   %.4g",
                    i, arr_str, vpp[i], rms[i])
    if np.isfinite(arrivals_us).sum() >= 2:
        finite = arrivals_us[np.isfinite(arrivals_us)]
        logger.info(
            "  arrival spread: mean=%.3f us, std=%.4f us, peak-to-peak=%.4f us",
            finite.mean(), finite.std(ddof=0), float(np.ptp(finite)),
        )
        # Reference each arrival against the first so drift is obvious.
        drift = arrivals_us - arrivals_us[0]
        logger.info("  arrival drift vs pulse 0 (us): %s",
                    ", ".join(f"{d:+.4f}" if np.isfinite(d) else "nan"
                              for d in drift))
    logger.info("  Vpp:   mean=%.4g, std=%.4g, min=%.4g, max=%.4g",
                vpp.mean(), vpp.std(ddof=0), vpp.min(), vpp.max())

    # --- Aggregation: optional cross-correlation alignment + average ---
    aligned = None
    lags_ns = None
    mean_trace = None
    std_trace = None
    if args.aggregate and n_pulses >= 2:
        from openlifu_verification import align_pulse_traces

        dt_s = float(t_ns[1] - t_ns[0]) * 1e-9
        if dt_s <= 0:
            logger.warning("Non-positive sample interval; skipping aggregation.")
        else:
            if args.align:
                max_shift_samples = max(
                    1, int(np.ceil(args.align_max_shift_ns * 1e-9 / dt_s))
                )
                aligned, lags_s = align_pulse_traces(
                    traces.astype(float),
                    dt_s=dt_s,
                    max_shift_samples=max_shift_samples,
                    reference="first",
                )
                lags_ns = lags_s * 1e9
                logger.info(
                    "  cross-corr lag vs pulse 0 (ns): mean=%+.2f, "
                    "std=%.2f, peak-to-peak=%.2f (search bound: +/-%.0f ns)",
                    lags_ns.mean(), lags_ns.std(ddof=0),
                    float(np.ptp(lags_ns)), args.align_max_shift_ns,
                )
            else:
                aligned = traces.astype(float)
                lags_ns = np.zeros(n_pulses)
            mean_trace = aligned.mean(axis=0)
            std_trace = aligned.std(axis=0, ddof=0)
            agg_vpp = float(np.ptp(mean_trace))
            agg_rms = float(np.sqrt(np.mean(mean_trace**2)))
            # Noise proxy: RMS of the per-sample std across the trace.
            noise_rms = float(np.sqrt(np.mean(std_trace**2)))
            snr = agg_rms / noise_rms if noise_rms > 0 else float("inf")
            logger.info(
                "  aggregated (%s alignment): Vpp=%.4g, RMS=%.4g, "
                "sample-std RMS=%.4g, coherent SNR~=%.2f",
                "with" if args.align else "no",
                agg_vpp, agg_rms, noise_rms, snr,
            )

    if not (args.plot or args.save_plot):
        return

    # --- Plot: overlaid raw traces + (optional) aligned mean + summary ---
    import matplotlib.pyplot as plt

    show_mean = mean_trace is not None
    if show_mean:
        fig, (ax_traces, ax_mean, ax_summary) = plt.subplots(
            3, 1, figsize=(10, 10),
            gridspec_kw={"height_ratios": [3, 3, 1]},
        )
    else:
        fig, (ax_traces, ax_summary) = plt.subplots(
            2, 1, figsize=(10, 7),
            gridspec_kw={"height_ratios": [3, 1]},
        )
        ax_mean = None

    t_us = t_ns * 1e-3
    cmap = plt.get_cmap("viridis")
    for i in range(n_pulses):
        ax_traces.plot(
            t_us, traces[i], lw=0.9,
            color=cmap(i / max(n_pulses - 1, 1)),
            label=f"pulse {i}" if n_pulses <= 12 else None,
        )
    ax_traces.set_xlabel("time (µs, relative to each trigger)")
    ax_traces.set_ylabel(f"raw traces ({scan_result.units})")
    ax_traces.grid(True, alpha=0.3)
    ax_traces.set_title(
        f"{n_pulses} pulses  @ {resolved['frequency_kHz']:.1f} kHz  "
        f"{resolved['voltage']:.1f} V  interval={resolved['interval_msec']:.2f} ms"
    )
    if n_pulses <= 12:
        ax_traces.legend(loc="upper right", fontsize=8, ncol=2)

    if ax_mean is not None:
        for i in range(n_pulses):
            ax_mean.plot(
                t_us, aligned[i], lw=0.6, alpha=0.35,
                color=cmap(i / max(n_pulses - 1, 1)),
            )
        ax_mean.fill_between(
            t_us, mean_trace - std_trace, mean_trace + std_trace,
            color="k", alpha=0.15, label="±1σ across pulses",
        )
        ax_mean.plot(t_us, mean_trace, lw=1.8, color="k", label="mean (aligned)")
        ax_mean.set_xlabel("time (µs)")
        ax_mean.set_ylabel(f"aligned ({scan_result.units})")
        ax_mean.grid(True, alpha=0.3)
        align_note = "cross-corr aligned" if args.align else "no alignment"
        ax_mean.set_title(f"aggregated across {n_pulses} pulses ({align_note})")
        ax_mean.legend(loc="upper right", fontsize=8)

    idx = np.arange(n_pulses)
    ax_summary.plot(idx, vpp, "o-", label=f"Vpp ({scan_result.units})")
    ax_summary.set_xlabel("pulse index")
    ax_summary.set_ylabel(f"Vpp ({scan_result.units})")
    ax_summary.grid(True, alpha=0.3)
    ax_summary.set_xticks(idx)
    if lags_ns is not None:
        ax_lag = ax_summary.twinx()
        ax_lag.plot(idx, lags_ns, "s--", color="tab:red", label="lag (ns)")
        ax_lag.set_ylabel("lag vs pulse 0 (ns)", color="tab:red")
        ax_lag.tick_params(axis="y", labelcolor="tab:red")
    elif np.isfinite(arrivals_us).any():
        ax_arr = ax_summary.twinx()
        ax_arr.plot(idx, arrivals_us, "s--", color="tab:red", label="arrival (µs)")
        ax_arr.set_ylabel("arrival (µs)", color="tab:red")
        ax_arr.tick_params(axis="y", labelcolor="tab:red")

    fig.tight_layout()
    if args.save_plot:
        fig.savefig(args.save_plot, dpi=120)
        logger.info("Saved plot to %s", args.save_plot)
    if args.plot:
        plt.show()


if __name__ == "__main__":
    main()
