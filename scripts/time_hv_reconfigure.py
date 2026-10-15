"""Time LIFU reconfiguration with and without the HV re-send.

``LIFUInterface.set_solution`` always calls ``hvcontroller.set_voltage``
(setpoint + HV_ON re-assert), even when the voltage is unchanged. This
script measures how much of a ``configure_lifu`` call that costs, and
whether the rail stays up when the re-send is skipped.

Measured (each ``--repeats`` times, HV on at the rail voltage):

  configure (current)     ``apply_pulse`` as the scans call it today
  configure (no HV)       same, with ``hvcontroller.set_voltage`` stubbed
                          out (equivalent to calling ``txdevice`` directly)
  set_voltage (same V)    the HV re-send alone
  settle after re-send    ``wait_for_settle`` right after it
  step to V / settle      ``set_voltage`` to a new target, then settle,
                          in both directions
  get_voltage             one HV readback (poll cost)

"V after" is read back immediately after the call returns: if it is
already at target after a voltage step, the firmware blocks while the
rail ramps.

The HV rail is on for the duration but no pulses are fired. Requires
the console's internal HV supply; no PicoScope needed.

    python scripts/time_hv_reconfigure.py --frequency-khz 155
"""
import argparse
import logging
import statistics
import time

from openlifu_verification import VerificationTank
from openlifu_verification.acceptance import SUPPORTED_FREQUENCIES_KHZ


def _timed(fn):
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frequency-khz", type=float, default=155.0,
                        choices=SUPPORTED_FREQUENCIES_KHZ)
    parser.add_argument("--voltage", type=float, default=None,
                        help="HV rail (V). Default: the tank default for "
                             "--frequency-khz (the report's scan voltage).")
    parser.add_argument("--step-voltage", type=float, default=10.0,
                        help="Second voltage for the step measurements.")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--num-modules", type=int, default=1)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    freq = args.frequency_khz
    volt = (args.voltage if args.voltage is not None
            else VerificationTank.default_voltage_for(freq))
    rows = []  # (label, seconds, V after)

    with VerificationTank(frequency=int(freq), num_modules=args.num_modules,
                          use_picoscope=False, calibration_path=None) as ver:
        hv = ver.lifu.hvcontroller
        ver.apply_pulse(frequency_kHz=freq, voltage=volt)
        ver.enable_hv_output(wait=True)
        print(f"HV on at {volt:g} V ({hv.get_voltage():.2f} V measured), "
              f"{freq:g} kHz; {args.repeats} repeats")

        def record(label, fn):
            rows.append((label, _timed(fn), hv.get_voltage()))

        def configure():
            ver.apply_pulse(frequency_kHz=freq, voltage=volt)

        def configure_no_hv():
            hv.set_voltage = lambda voltage: True
            try:
                configure()
            finally:
                del hv.set_voltage

        for i in range(args.repeats):
            print(f"  repeat {i + 1}/{args.repeats}")
            record("configure (current)", configure)
            record("configure (no HV)", configure_no_hv)
            record("set_voltage (same V)", lambda: hv.set_voltage(volt))
            record("settle after re-send", hv.wait_for_settle)
            for target in (args.step_voltage, volt):
                record(f"step to {target:g} V", lambda t=target: hv.set_voltage(t))
                record(f"settle at {target:g} V", hv.wait_for_settle)
            record("get_voltage", hv.get_voltage)

    print()
    print(f"{'measurement':<24}{'median s':>10}{'min s':>8}{'max s':>8}"
          f"{'V after (min..max)':>22}")
    for label in dict.fromkeys(r[0] for r in rows):
        secs = [s for lbl, s, _ in rows if lbl == label]
        v_after = [v for lbl, _, v in rows if lbl == label]
        print(f"{label:<24}{statistics.median(secs):>10.2f}{min(secs):>8.2f}"
              f"{max(secs):>8.2f}{min(v_after):>14.2f}..{max(v_after):.2f}")


if __name__ == "__main__":
    main()
