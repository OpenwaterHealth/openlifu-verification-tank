"""Benchmark rapid-block chunk sizes on a moderate lateral grid.

Sweeps a focus grid (default 21x21 = 441 points) with several
different rapid-block chunk sizes. For each chunk size, records:

  - total wall time
  - mean per-point apply / trigger / arm / xfer time
  - end-to-end points-per-second

Per-point ``apply + trigger`` cost (~0.33 s with the current SDK/firmware)
is independent of chunk size, so total time is dominated by the point
count, not by the chunk sweep. 441 points x ~0.33 s ~= 145 s per chunk
size; the default 6-value sweep takes ~15 min end to end.

Use this to pick a chunk size for very large scans. Results are
saved to ``scripts/data/benchmark_rapid_data.npz`` and printed to
the log.

Note: chunk_size is capped internally at ``scope.get_max_segments()``.
On the PS5000A that's typically ~125k with the default 12-bit
configuration, so chunk size is effectively unbounded here.
"""
import argparse
import logging
import time
from pathlib import Path
import numpy as np
from openlifu_verification import VerificationTank

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False


def _run_one(ver, focus_points, chunk_size, time_start_s, time_stop_s,
             sampling_interval_ns):
    def apply_point(point):
        x, y, z = point
        ver.set_focus(x, y, z)

    t0 = time.perf_counter()
    outputs, timings = ver.run_rapid_sweep(
        points=focus_points,
        apply_point=apply_point,
        time_start_s=time_start_s,
        time_stop_s=time_stop_s,
        sampling_interval_ns=sampling_interval_ns,
        chunk_size=chunk_size,
    )
    wall_s = time.perf_counter() - t0

    if not timings:
        return None
    apply_arr = np.array([t["apply_s"] for t in timings])
    trigger_arr = np.array([t["trigger_s"] for t in timings])
    arm_arr = np.array([t["arm_s"] for t in timings])
    xfer_arr = np.array([t["xfer_s"] for t in timings])
    iter_arr = np.array([t["iter_total_s"] for t in timings])
    n_captured = sum(1 for t in timings if t["captured"])
    n_chunks = 1 + max((t["chunk_index"] for t in timings), default=0)

    return {
        "wall_s": wall_s,
        "n_points": len(timings),
        "n_captured": n_captured,
        "n_chunks": n_chunks,
        "apply_mean_s": float(apply_arr.mean()),
        "trigger_mean_s": float(trigger_arr.mean()),
        "arm_mean_s": float(arm_arr.mean()),
        "xfer_mean_s": float(xfer_arr.mean()),
        "iter_mean_s": float(iter_arr.mean()),
        "arm_total_s": float(arm_arr.sum()),
        "xfer_total_s": float(xfer_arr.sum()),
        "points_per_s": len(timings) / wall_s if wall_s > 0 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voltage", type=float, default=10.0)
    parser.add_argument("--num-x", type=int, default=21)
    parser.add_argument("--num-y", type=int, default=21)
    parser.add_argument("--chunk-sizes", type=int, nargs="+",
                        default=[1, 4, 16, 64, 256, 441],
                        help="Chunk sizes to benchmark. Per-point apply+trigger "
                             "cost (~0.33s) is fixed; only arm+xfer amortization "
                             "changes with chunk size, so a modest grid is enough "
                             "to find the plateau. Max segments on the scope is "
                             "large (~125k), so chunk size is never capped.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip the Enter prompt (assumes tank is already primed).")
    args = parser.parse_args()

    zInput = 50
    xfoci = np.linspace(-10, 10, args.num_x)
    yfoci = np.linspace(-10, 10, args.num_y)
    focus_points = [(float(x), float(y), zInput) for y in yfoci for x in xfoci]

    frequency_kHz = 400
    duration_msec = 20 / 400
    interval_msec = 20
    num_modules = 1

    sampling_interval_ns = 100
    time_start_s = 100e-6
    time_stop_s = 200e-6

    logger.info(
        "Benchmark: %d x %d = %d points, chunk_sizes=%s",
        args.num_x, args.num_y, len(focus_points), args.chunk_sizes,
    )

    results = []
    try:
        with VerificationTank(frequency=frequency_kHz,
                              num_modules=num_modules,
                              ext_power_supply=False) as ver:
            ver.configure_lifu(
                frequency_kHz=frequency_kHz,
                voltage=args.voltage,
                duration_msec=duration_msec,
                interval_msec=interval_msec,
                pulse_count=1,
                trigger_mode="single",
            )
            ver.enable_hv_output(wait=True)
            max_segments = ver.scope.get_max_segments()
            logger.info("Scope max_segments = %d at current resolution.", max_segments)

            if not args.dry_run:
                input("Press Enter to start benchmark")

            for chunk_size in args.chunk_sizes:
                cs_eff = min(chunk_size, max_segments)
                if cs_eff != chunk_size:
                    logger.warning(
                        "Requested chunk_size=%d exceeds max_segments=%d; capped.",
                        chunk_size, max_segments,
                    )
                logger.info("--- chunk_size=%d (eff=%d) ---", chunk_size, cs_eff)
                r = _run_one(ver, focus_points, cs_eff,
                             time_start_s, time_stop_s, sampling_interval_ns)
                if r is None:
                    logger.warning("chunk_size=%d produced no data.", chunk_size)
                    continue
                r["chunk_size_req"] = chunk_size
                r["chunk_size_eff"] = cs_eff
                results.append(r)
                logger.info(
                    "chunk=%d  wall=%.2fs  pts/s=%.2f  chunks=%d  "
                    "apply=%.4f  trigger=%.4f  arm=%.4f  xfer=%.4f",
                    cs_eff, r["wall_s"], r["points_per_s"], r["n_chunks"],
                    r["apply_mean_s"], r["trigger_mean_s"],
                    r["arm_mean_s"], r["xfer_mean_s"],
                )

    except (ConnectionError, ValueError, Exception) as e:
        logger.error(f"An error occurred: {e}")

    if not results:
        logger.warning("No benchmark results collected.")
        return

    logger.info("=== Benchmark summary ===")
    logger.info(
        "%-6s %-6s %-8s %-8s %-7s %-8s %-8s %-8s %-8s",
        "req", "eff", "wall_s", "pts/s", "chunks",
        "apply", "trigger", "arm", "xfer",
    )
    for r in results:
        logger.info(
            "%-6d %-6d %-8.2f %-8.2f %-7d %-8.4f %-8.4f %-8.4f %-8.4f",
            r["chunk_size_req"], r["chunk_size_eff"],
            r["wall_s"], r["points_per_s"], r["n_chunks"],
            r["apply_mean_s"], r["trigger_mean_s"],
            r["arm_mean_s"], r["xfer_mean_s"],
        )

    out_path = Path(__file__).parent.resolve() / 'data'
    out_path.mkdir(exist_ok=True)
    savedata = {
        "chunk_sizes_req": np.array([r["chunk_size_req"] for r in results]),
        "chunk_sizes_eff": np.array([r["chunk_size_eff"] for r in results]),
        "wall_s": np.array([r["wall_s"] for r in results]),
        "points_per_s": np.array([r["points_per_s"] for r in results]),
        "n_chunks": np.array([r["n_chunks"] for r in results]),
        "apply_mean_s": np.array([r["apply_mean_s"] for r in results]),
        "trigger_mean_s": np.array([r["trigger_mean_s"] for r in results]),
        "arm_mean_s": np.array([r["arm_mean_s"] for r in results]),
        "xfer_mean_s": np.array([r["xfer_mean_s"] for r in results]),
        "arm_total_s": np.array([r["arm_total_s"] for r in results]),
        "xfer_total_s": np.array([r["xfer_total_s"] for r in results]),
        "iter_mean_s": np.array([r["iter_mean_s"] for r in results]),
        "n_points": np.array([r["n_points"] for r in results]),
        "n_captured": np.array([r["n_captured"] for r in results]),
    }
    np.savez(out_path / "benchmark_rapid_data.npz", **savedata)
    logger.info("Saved benchmark_rapid_data.npz")


if __name__ == "__main__":
    main()
