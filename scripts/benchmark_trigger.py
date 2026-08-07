"""Benchmark the per-trigger USB round-trip cost so we can attribute
 the ~440 ms spent inside VerificationTank.trigger_once() during a
 rapid-block sweep.

Runs N cycles of the same TX / HV calls that ``LIFUInterface.start_sonication``
and ``stop_sonication`` issue -- but times each call individually. Also
compares the SDK's ``start_sonication`` / ``stop_sonication`` wrappers to
a bare-metal ``start_trigger`` / ``stop_trigger`` path.

Run from the repo root::

    python scripts/benchmark_trigger.py --iters 10
"""
from __future__ import annotations

import argparse
import logging
import statistics
import time

from openlifu_verification import VerificationTank

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def _stats(label: str, samples: list[float]) -> None:
    if not samples:
        logger.info("%-32s (no samples)", label)
        return
    mean = statistics.mean(samples) * 1000.0
    median = statistics.median(samples) * 1000.0
    lo = min(samples) * 1000.0
    hi = max(samples) * 1000.0
    total = sum(samples) * 1000.0
    logger.info(
        "%-32s mean=%7.2f ms  median=%7.2f ms  min=%7.2f ms  max=%7.2f ms  total=%8.2f ms  n=%d",
        label, mean, median, lo, hi, total, len(samples),
    )


def _time_call(fn, *args, **kwargs) -> float:
    t0 = time.perf_counter()
    fn(*args, **kwargs)
    return time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iters", type=int, default=10,
                    help="Number of trigger cycles per benchmark path.")
    ap.add_argument("--warmup", type=int, default=2,
                    help="Warmup iterations dropped from the summary.")
    ap.add_argument("--voltage", type=float, default=5.0)
    ap.add_argument("--frequency-khz", type=int, default=400)
    args = ap.parse_args()

    logger.info("Opening VerificationTank (no scope)")
    with VerificationTank(frequency=args.frequency_khz,
                          num_modules=1,
                          use_picoscope=False,
                          ext_power_supply=False) as ver:
        ver.configure_lifu(
            frequency_kHz=args.frequency_khz,
            voltage=args.voltage,
            duration_msec=20 / args.frequency_khz,
            interval_msec=20,
            pulse_count=1,
            trigger_mode="single",
        )
        ver.enable_hv_output(wait=True)

        tx = ver.lifu.txdevice
        hv = ver.lifu.hvcontroller
        lifu = ver.lifu

        # ----- Path 1: SDK start_sonication + stop_sonication -----
        logger.info("--- Path 1: SDK start_sonication / stop_sonication (matches trigger_once) ---")
        sdk_start: list[float] = []
        sdk_stop: list[float] = []
        sdk_pair: list[float] = []
        for i in range(args.iters + args.warmup):
            t0 = time.perf_counter()
            t_s = _time_call(lifu.start_sonication,
                             turn_hv_on=False, wait_for_settle=False,
                             async_mode=False)
            t_e = _time_call(lifu.stop_sonication,
                             turn_hv_off=False, wait_for_settle=False)
            t_pair = time.perf_counter() - t0
            if i >= args.warmup:
                sdk_start.append(t_s)
                sdk_stop.append(t_e)
                sdk_pair.append(t_pair)
        _stats("start_sonication", sdk_start)
        _stats("stop_sonication", sdk_stop)
        _stats("start+stop total", sdk_pair)

        # ----- Path 2: bare start_trigger / stop_trigger -----
        logger.info("--- Path 2: bare tx.start_trigger / tx.stop_trigger ---")
        bare_start: list[float] = []
        bare_stop: list[float] = []
        bare_pair: list[float] = []
        for i in range(args.iters + args.warmup):
            t0 = time.perf_counter()
            t_s = _time_call(tx.start_trigger)
            t_e = _time_call(tx.stop_trigger)
            t_pair = time.perf_counter() - t0
            if i >= args.warmup:
                bare_start.append(t_s)
                bare_stop.append(t_e)
                bare_pair.append(t_pair)
        _stats("tx.start_trigger", bare_start)
        _stats("tx.stop_trigger", bare_stop)
        _stats("start+stop total", bare_pair)

        # ----- Path 3: individual USB round-trips -----
        logger.info("--- Path 3: individual USB round-trips ---")
        t_async_off: list[float] = []
        t_hv_status: list[float] = []
        t_tx_ping: list[float] = []
        for i in range(args.iters + args.warmup):
            a = _time_call(tx.async_mode, False)
            b = _time_call(hv.get_hv_status)
            c = _time_call(tx.ping)
            if i >= args.warmup:
                t_async_off.append(a)
                t_hv_status.append(b)
                t_tx_ping.append(c)
        _stats("tx.async_mode(False)", t_async_off)
        _stats("hv.get_hv_status()", t_hv_status)
        _stats("tx.ping()", t_tx_ping)

        # ----- Path 4: minimal fast path (start_trigger only) -----
        # trigger_mode="single" means the trigger auto-disarms after
        # firing one pulse train, so stop_trigger between shots should
        # be unnecessary. If this succeeds N times in a row, we can
        # collapse trigger_once() into one 107 ms USB round-trip.
        logger.info("--- Path 4: bare tx.start_trigger only (no stop_trigger) ---")
        start_only: list[float] = []
        for i in range(args.iters + args.warmup):
            t_s = _time_call(tx.start_trigger)
            if i >= args.warmup:
                start_only.append(t_s)
        _stats("tx.start_trigger (alone)", start_only)

        logger.info(
            "Interpretation: TX round-trip is ~107 ms regardless of command "
            "(async_mode == ping == start_trigger == stop_trigger). The only "
            "lever is to reduce the number of round-trips per pulse."
        )


if __name__ == "__main__":
    main()
