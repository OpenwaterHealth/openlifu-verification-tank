import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

from .picoscope import Picoscope
from .qpx600dp import QPX600DP
from .scan_results import ScanResult
from .hydrophone import Hydrophone
from .paths import HYDROPHONE_STATE_PATH
from .pulse_align import align_pulse_traces

from openlifu_sdk.io import LIFUInterface
from openlifu_sdk.io.LIFUTXDevice import Tx7332DelayProfile, Tx7332PulseProfile

logger = logging.getLogger(__name__)
_PACKAGE_LOGGER = logging.getLogger("openlifu_verification")
PICOSCOPE_RESOLUTION = "15BIT"
SPEED_OF_SOUND = 1490  # m/s in water
# Fixed electrical delay between the scope trigger's rising edge and
# the actual start of ultrasound emission (visible in traces as a
# small burst of EM pickup at t=0). Used to convert measured
# time-of-arrival into an axial depth. Hard-coded from calibration on
# current TX7332 firmware; override via ``VerificationTank(
# system_transmit_delay_us=...)`` if a future firmware changes it.
SYSTEM_TRANSMIT_DELAY_US = 115.0
HYDROPHONE_CHANNEL = 'A'
TRIGGER_CHANNEL = 'B'


class _TransducerData:
    """Lightweight transducer container built from the pinmap JSON.

    Holds element positions (in mm), pin numbers, and the raw JSON payload.
    Element order matches the JSON file; call ``sort_by_pin`` to reorder in
    place so element ``i`` corresponds to TX channel ``i``.
    """

    def __init__(self, data: dict):
        self._data = data
        elements = data.get("elements", [])
        self._positions = np.array(
            [el.get("position", [0.0, 0.0, 0.0]) for el in elements],
            dtype=float,
        )
        self._pins = np.array([el.get("pin", i + 1) for i, el in enumerate(elements)], dtype=int)
        self._module_invert = data.get("module_invert", False)

    @classmethod
    def from_file(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            return cls(json.load(f))

    def numelements(self) -> int:
        return int(self._positions.shape[0])

    def get_positions(self, units: str = "mm") -> np.ndarray:
        if units != "mm":
            raise ValueError(f"Only 'mm' units supported, got {units!r}")
        return self._positions

    @property
    def pins(self) -> np.ndarray:
        return self._pins

    @property
    def module_invert(self):
        return self._module_invert

    def sort_by_pin(self) -> None:
        order = np.argsort(self._pins)
        self._positions = self._positions[order]
        self._pins = self._pins[order]
        elements = self._data.get("elements", [])
        self._data["elements"] = [elements[i] for i in order]

    def to_solution_transducer(self) -> dict:
        """Return a solution-compatible transducer dict."""
        return {
            "id": self._data.get("id", ""),
            "name": self._data.get("name", ""),
            "elements": self._data.get("elements", []),
            "module_invert": self._module_invert,
        }


class VerificationTank:
    """
    A context manager to simplify OpenLIFU verification tasks.
    """

    # -- Pulse / drive defaults --------------------------------------
    # Class-level defaults consumed by :meth:`apply_pulse` when the
    # caller passes ``None``. Scripts should keep argparse defaults as
    # ``None`` and pass them through so operator can rely on these
    # values without hunting for hard-coded numbers in every script.
    DEFAULT_FREQUENCY_KHZ: float = 400.0
    DEFAULT_VOLTAGE_V: float = 20.0
    DEFAULT_CYCLES_PER_BURST: float = 20.0
    DEFAULT_INTERVAL_MSEC: float = 20.0
    DEFAULT_PULSE_COUNT: int = 1
    DEFAULT_TRIGGER_MODE: str = "single"

    def __init__(self,
                 frequency=400,
                 use_picoscope=True,
                 num_modules=1,
                 resolution=PICOSCOPE_RESOLUTION,
                 ext_power_supply=False,
                 voltage_table_selection="dvt",
                 hydrophone_channel="A",
                 trigger_channel="B",
                 hydrophone_range_mv=100,
                 trigger_range_mv=5000,
                 trigger_threshold_mv=1000,
                 trigger_direction="rising",
                 hydrophone=None,
                 hydrophone_position=(0.0, 0.0, 50.0),
                 calibration_path=HYDROPHONE_STATE_PATH,
                 use_calibration=True,
                 system_transmit_delay_us=SYSTEM_TRANSMIT_DELAY_US):
        self.use_picoscope = use_picoscope
        self.resolution = resolution
        self.num_modules = num_modules
        self.frequency = frequency
        self.ext_power_supply = ext_power_supply
        self.voltage_table_selection = voltage_table_selection
        self.hydrophone_channel = hydrophone_channel
        self.trigger_channel = trigger_channel
        self.hydrophone_range_mv = hydrophone_range_mv
        self.trigger_range_mv = trigger_range_mv
        self.trigger_threshold_mv = trigger_threshold_mv
        self.trigger_direction = trigger_direction
        self.lifu = None
        self.scope = None
        self.hv = None
        self.hv_enabled = False
        self.hv_voltage = None
        self.arr = None
        # Hydrophone calibration + geometry. ``hydrophone_position`` is
        # the *actual* (calibrated) x,y,z of the hydrophone tip in the
        # transducer coord frame; scans use it as their origin so a
        # sweep of e.g. -1..+1 mm around 0 lands on the true peak.
        if isinstance(hydrophone, (str, Path)):
            hydrophone = Hydrophone(hydrophone)
        self.hydrophone = hydrophone
        self.hydrophone_position = np.array(hydrophone_position, dtype=float).reshape(3)
        self.calibration_path = Path(calibration_path) if calibration_path else None
        # When ``False``, keep the hydrophone attached (so position,
        # id, and any calibration-file lookups still work) but skip
        # the mV \u2192 Pa conversion so all reported traces stay in raw
        # scope volts. Useful when the operator wants to see the raw
        # signal even though a calibration is available.
        self.use_calibration = bool(use_calibration)
        # Fixed electrical delay between trigger and actual ultrasound
        # emission (µs). Used to convert arrival time ↔ axial depth
        # via ``expected_arrival_us = system_transmit_delay_us + z_mm / SOS``.
        self.system_transmit_delay_us = float(system_transmit_delay_us)

    def __enter__(self):
        """
        Initializes and connects to all the required instruments.
        """
        try:
            self.lifu = LIFUInterface(
                ext_power_supply=self.ext_power_supply,
                voltage_table_selection=self.voltage_table_selection,
            )
            if self.ext_power_supply:
                self.hv = QPX600DP()
            if self.use_picoscope:
                self.scope = Picoscope(resolution=self.resolution)
                self.scope.__enter__()
                # Sensible default channel + trigger config so callers
                # don't have to repeat the same boilerplate every script.
                # Override afterwards with set_hydrophone_range /
                # set_trigger_range or by calling scope.set_channel /
                # scope.set_trigger directly.
                self.scope.set_channel(
                    self.hydrophone_channel,
                    range_mv=self.hydrophone_range_mv,
                    coupling='DC',
                )
                self.scope.set_channel(
                    self.trigger_channel,
                    range_mv=self.trigger_range_mv,
                    coupling='DC',
                )
                # auto_trigger_ms=0 disables the scope's auto-trigger
                # fallback so wait_ready only returns on a real edge.
                self.scope.set_trigger(
                    channel=self.trigger_channel,
                    threshold_mv=self.trigger_threshold_mv,
                    direction=self.trigger_direction,
                    auto_trigger_ms=0,
                )
            self.lifu.__enter__()
            if self.hv is not None:
                self.hv.__enter__()

            tx_connected, hv_connected = self.lifu.is_device_connected()
            self.disable_hv_output()

            if tx_connected:
                logger.info(f"  TX Connected: {tx_connected}")
                logger.info("✅ LIFU Device fully connected.")
            else:
                raise ConnectionError("❌ TX NOT fully connected.")

            if not self.lifu.txdevice.ping():
                raise ConnectionError("Failed to ping the transmitter device.")

            tx_firmware_version = self.lifu.txdevice.get_version()
            logger.info(f"TX Firmware Version: {tx_firmware_version}")

            num_tx_devices = self.lifu.txdevice.enum_tx7332_devices()
            if num_tx_devices == 0:
                raise ValueError("No TX7332 devices found.")
            elif num_tx_devices != self.num_modules * 2:
                 raise Exception(f"Number of TX7332 devices found: {num_tx_devices} != 2x{self.num_modules}")
            logger.info(f"Number of TX7332 devices found: {num_tx_devices}")

            transducers_path = Path(__file__).parent.parent.resolve()
            self.arr = _TransducerData.from_file(
                f"{transducers_path}/transducers/openlifu_{self.num_modules}x{self.frequency}_evt1.json"
            )
            self.arr.sort_by_pin()

            # Auto-load a previously-calibrated hydrophone position if
            # one has been saved to ``calibration_path``. On first-time
            # run the file is seeded with the current (default) position
            # so the operator has an editable copy to tweak.
            cal_source = None
            if self.calibration_path is not None:
                if self.calibration_path.is_file():
                    try:
                        self.load_calibration()
                        cal_source = f"loaded from {self.calibration_path}"
                    except Exception as e:
                        logger.warning(
                            "Failed to load hydrophone calibration from %s: %s",
                            self.calibration_path, e,
                        )
                else:
                    try:
                        self.save_calibration()
                        cal_source = f"seeded default at {self.calibration_path}"
                    except Exception as e:
                        logger.warning(
                            "Could not seed default hydrophone calibration at %s: %s",
                            self.calibration_path, e,
                        )

            # Consolidated summary of the actual final hydrophone state:
            # what device is attached (if any) + the position that
            # subsequent relative scans will use as their origin.
            hydro_id = self._current_hydrophone_id()
            if self.hydrophone is not None:
                if self.use_calibration:
                    hydro_desc = (f"attached (id={hydro_id!r})"
                                  if hydro_id else "attached")
                else:
                    hydro_desc = (
                        f"attached (id={hydro_id!r}) but calibration "
                        f"disabled -- traces reported in mV"
                        if hydro_id
                        else "attached but calibration disabled -- "
                             "traces reported in mV"
                    )
            else:
                hydro_desc = "not attached (traces will be reported in mV)"
            src_desc = f" [{cal_source}]" if cal_source else ""
            logger.info(
                "Hydrophone: %s; position=%s mm%s",
                hydro_desc, self.hydrophone_position.tolist(), src_desc,
            )

        except Exception as e:
            logger.error(f"Error during initialization: {e}")
            self.__exit__(None, None, None)
            raise

        return self

    def configure_lifu(self, 
                       frequency_kHz, 
                       voltage, 
                       duration_usec, 
                       interval_msec, 
                       pulse_count=1,
                       pulse_train_interval_msec = 0,
                       pulse_train_count = 1, 
                       trigger_mode="single"):

        pulse_interval_s = interval_msec * 1e-3
        # The SDK/firmware misbehaves when pulse_train_interval is 0. The
        # test app applies the same fallback (see LIFUController
        # ``directSetSequence``): if the caller passes 0, use
        # ``pulse_count * pulse_interval`` so the train interval covers the
        # whole pulse train.
        pulse_train_interval_s = pulse_train_interval_msec * 1e-3

        pulse = {
            "frequency": frequency_kHz * 1e3,
            "duration": duration_usec * 1e-6,
            "amplitude": 1.0,
        }

        sequence = {
            "pulse_interval": pulse_interval_s,
            "pulse_count": pulse_count,
            "pulse_train_interval": pulse_train_interval_s,
            "pulse_train_count": pulse_train_count,
        }

        # Dummy values for delays and apodizations
        delays = np.zeros((1, self.arr.numelements()))
        apodizations = np.ones((1, self.arr.numelements()))

        pin_order = np.argsort(self.arr.pins)
        solution = {
            "id": "verification_solution",
            "name": "VerificationTank Solution",
            "delays": delays[:, pin_order],
            "apodizations": apodizations[:, pin_order],
            "pulse": pulse,
            "sequence": sequence,
            "voltage": voltage,
            "transducer": self.arr.to_solution_transducer(),
        }
        profile_index = 1
        profile_increment = True

        # Log the pieces being sent so we can diff against a known-good
        # solution.json from the test app.
        logger.debug("configure_lifu pulse=%s", pulse)
        logger.debug("configure_lifu sequence=%s", sequence)
        logger.debug(
            "configure_lifu voltage=%s trigger_mode=%s profile_index=%s profile_increment=%s",
            voltage, trigger_mode, profile_index, profile_increment,
        )
        logger.debug(
            "configure_lifu transducer id=%s num_elements=%s module_invert=%s",
            solution["transducer"].get("id"),
            len(solution["transducer"].get("elements", [])),
            solution["transducer"].get("module_invert"),
        )
        logger.debug(
            "configure_lifu delays.shape=%s apodizations.shape=%s",
            solution["delays"].shape, solution["apodizations"].shape,
        )

        self._set_hv_voltage(voltage)

        # Async STATUS frames share the TX device's CDC IN endpoint with
        # command responses; large set_solution writes (many write_block
        # chunks) routinely race with STATUS emissions when async is on.
        # Match the test app: force async OFF around set_solution.
        try:
            self.lifu.txdevice.async_mode(False)
        except Exception as e:
            logger.warning("txdevice.async_mode(False) before set_solution raised: %s", e)

        self.lifu.set_solution(
            solution=solution,
            profile_index=profile_index,
            profile_increment=profile_increment,
            trigger_mode=trigger_mode)

    def apply_pulse(self, *,
                    frequency_kHz: float | None = None,
                    voltage: float | None = None,
                    cycles_per_burst: float | None = None,
                    duration_usec: float | None = None,
                    interval_msec: float | None = None,
                    pulse_count: int | None = None,
                    pulse_train_interval_msec: float | None = None,
                    pulse_train_count: int | None = None,
                    trigger_mode: str | None = None) -> dict:
        """Configure the LIFU with sensible defaults for any ``None`` kwarg.

        Thin wrapper around :meth:`configure_lifu` that lets scripts
        pass their argparse args through unchanged (``default=None``)
        and rely on class-level defaults
        (``DEFAULT_FREQUENCY_KHZ``, ``DEFAULT_VOLTAGE_V``, ...) for
        anything the operator didn't explicitly override.

        ``duration_usec`` is derived from ``cycles_per_burst /
        frequency_kHz * 1000`` when not passed explicitly, so callers
        can just specify "20 cycles" instead of computing the µs.

        Returns the fully-resolved keyword dict actually sent to
        :meth:`configure_lifu` (useful for logging / metadata).
        """
        freq = float(frequency_kHz if frequency_kHz is not None
                     else self.DEFAULT_FREQUENCY_KHZ)
        volt = float(voltage if voltage is not None else self.DEFAULT_VOLTAGE_V)
        if duration_usec is None:
            cyc = float(cycles_per_burst if cycles_per_burst is not None
                        else self.DEFAULT_CYCLES_PER_BURST)
            # cycles / freq_kHz => milliseconds; ×1000 => microseconds.
            duration = cyc / freq * 1000.0
        else:
            duration = float(duration_usec)
        interval = float(interval_msec if interval_msec is not None
                         else self.DEFAULT_INTERVAL_MSEC)
        pc = int(pulse_count if pulse_count is not None else self.DEFAULT_PULSE_COUNT)
        pti = float(pulse_train_interval_msec if pulse_train_interval_msec is not None
                    else 0.0)
        ptc = int(pulse_train_count if pulse_train_count is not None else 1)
        tm = str(trigger_mode if trigger_mode is not None else self.DEFAULT_TRIGGER_MODE)
        resolved = dict(
            frequency_kHz=freq, voltage=volt, duration_usec=duration,
            interval_msec=interval, pulse_count=pc,
            pulse_train_interval_msec=pti, pulse_train_count=ptc,
            trigger_mode=tm,
        )
        logger.info(
            "apply_pulse: frequency=%g kHz, voltage=%g V, duration=%g \u00b5s "
            "(%d cycles), interval=%g ms, pulse_count=%d, trigger_mode=%s",
            freq, volt, duration, int(round(duration * freq / 1000.0)),
            interval, pc, tm,
        )
        self.configure_lifu(**resolved)
        return resolved

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Disconnects from all instruments and cleans up resources.
        """
        try:
            self.disable_hv_output()
        except Exception as e:
            logger.error(f"Error turning off HV outputs: {e}")
        if self.hv:
            self.hv.__exit__(exc_type, exc_val, exc_tb)
        if self.scope:
            self.scope.__exit__(exc_type, exc_val, exc_tb)
        if self.lifu:
            self.lifu.__exit__(exc_type, exc_val, exc_tb)

        logger.info("All instruments disconnected.")

    def set_focus(self, x, y, z, apodizations=None):
        if self.arr is None:
            raise Exception("Transducer array not loaded. Please provide db_path during initialization.")

        focus = np.array([x, y, z])
        logger.debug(f"calculating delays for {focus=}")
        distances = np.sqrt(np.sum((focus - self.arr.get_positions(units="mm"))**2, 1)).reshape(-1)
        tof = distances*1e-3 / SPEED_OF_SOUND
        delays = tof.max() - tof

        if apodizations is None:
            apodizations = np.ones_like(delays)

        delay_profile = Tx7332DelayProfile(
                    profile=1,
                    delays=delays,
                    apodizations=apodizations
                )
        self.lifu.txdevice.tx_registers.add_delay_profile(delay_profile)
        logger.debug("writing delay registers...")
        control_registers = self.lifu.txdevice.tx_registers.get_delay_control_registers()
        data_registers = self.lifu.txdevice.tx_registers.get_delay_data_registers(pack=True, pack_single=True)

        for txi, (ctrl_regs, data_regs) in enumerate(zip(control_registers, data_registers)):
            #if not uniform_apodization: #TODO add this as a parameter
            #    for addr, reg_values in ctrl_regs.items():
            #        if not self.lifu.txdevice.write_register(identifier=txi, address=addr, value=reg_values):
            #            logger.error(f"Error applying TX CHIP ID: {txi} registers")
            for addr, reg_values in data_regs.items():
                if not self.lifu.txdevice.write_block(identifier=txi, start_address=addr, reg_values=reg_values):
                    logger.error(f"Error applying TX CHIP ID: {txi} registers")

    def set_pulse(self, frequency_kHz, duration_usec):
        pulse_profile = Tx7332PulseProfile(
            profile=1,
            frequency=frequency_kHz*1e3,
            cycles=int(duration_usec * frequency_kHz / 1000.0)
        )
        self.lifu.txdevice.tx_registers.add_pulse_profile(pulse_profile)
        logger.debug("writing pulse registers...")
        control_registers = self.lifu.txdevice.tx_registers.get_pulse_control_registers()
        data_registers = self.lifu.txdevice.tx_registers.get_pulse_data_registers(pack=True, pack_single=True)

        for txi, (ctrl_regs, data_regs) in enumerate(zip(control_registers, data_registers)):
            # for addr, reg_value in ctrl_regs.items():
            #     if not self.lifu.txdevice.write_register(identifier=txi, address=addr, value=reg_value):
            #         logger.error(f"Error applying TX CHIP ID: {txi} registers")                    
            #     logger.info(f"{addr:04x}:{reg_value}")
            for addr, reg_values in data_regs.items():
                if not self.lifu.txdevice.write_block(identifier=txi, start_address=addr, reg_values=reg_values):
                    logger.error(f"Error applying TX CHIP ID: {txi} registers")

    def set_scope_trigger(self, channel=TRIGGER_CHANNEL, threshold_mV=100, direction="rising"):
        if not self.scope:
            raise ValueError("No Picoscope Connected")
        self.scope.set_trigger(channel=channel, threshold_mV=threshold_mV, direction=direction)

    def run_trigger(self):
        """Fire a single TX trigger (no scope involvement).

        Useful for running the pulse when an external scope application
        is used for capture. HV must already be enabled (call
        ``enable_hv_output(wait=True)`` first) and
        ``configure_lifu(..., trigger_mode="single")`` must have set the
        firmware to fire exactly one pulse train per ``start_trigger``.

        Equivalent to :meth:`trigger_once`; kept as a separate public
        name for scripts that just want to fire without any scope
        involvement.
        """
        logger.debug("Sending Single Trigger (no scope)...")
        self.lifu.txdevice.start_trigger()

    def set_hydrophone_range(self, range_mv, coupling="DC"):
        """Set the scope's vertical range on the hydrophone channel.

        Args:
            range_mv: Full-scale +/- range in millivolts. Must be one of
                the discrete PicoScope ranges (10, 20, 50, 100, 200, 500,
                1000, 2000, 5000, 10000, 20000, 50000).
            coupling: 'DC' (default) or 'AC'.
        """
        if not self.scope:
            raise ValueError("No Picoscope Connected")
        self.scope.set_channel(
            self.hydrophone_channel, range_mv=range_mv, coupling=coupling
        )
        self.hydrophone_range_mv = range_mv

    def set_trigger_range(self, range_mv, coupling="DC"):
        """Set the scope's vertical range on the trigger channel.

        Note: changing the range invalidates the current trigger config
        (threshold_mV is stored in ADC counts against the range), so this
        also re-applies the trigger using the currently-configured
        threshold / direction.

        Args:
            range_mv: Full-scale +/- range in millivolts.
            coupling: 'DC' (default) or 'AC'.
        """
        if not self.scope:
            raise ValueError("No Picoscope Connected")
        self.scope.set_channel(
            self.trigger_channel, range_mv=range_mv, coupling=coupling
        )
        self.trigger_range_mv = range_mv
        # Re-apply the trigger so its threshold_mV is re-encoded against
        # the new range.
        self.scope.set_trigger(
            channel=self.trigger_channel,
            threshold_mv=self.trigger_threshold_mv,
            direction=self.trigger_direction,
            auto_trigger_ms=0,
        )

    def run_capture(self,
                    time_start_s,
                    time_stop_s,
                    sampling_interval_ns,
                    timeout_s=2.0):
        """Fire a single TX trigger and capture a time-based scope block.

        ``time_start_s`` and ``time_stop_s`` are given relative to the
        **start of ultrasound emission** (``t=0`` == first sample the
        transducer emits), *not* relative to the scope trigger. The
        fixed ``system_transmit_delay_us`` between trigger and emission
        is added internally when programming the scope, and subtracted
        back out of the returned ``time`` axis / ``time_start_s`` /
        ``time_stop_s`` fields. This makes arrivals easy to interpret:
        the time-of-arrival ``t`` at depth ``z`` mm is simply
        ``t = z / SOS`` (µs when SOS is mm/µs).

        - ``time_start_s < 0`` → capture a pre-emission window; the
          scope still sees this as post-trigger up to the transmit
          delay.
        - ``time_start_s >= system_transmit_delay_us`` → delayed
          capture: the scope waits the extra time after the trigger
          before it starts collecting samples.

        The scope only supports discrete sampling intervals and integer
        sample counts, so what actually gets used may differ from what
        was requested. The returned data dict includes
        ``sampling_interval_ns``, ``time_start_s``, and ``time_stop_s``
        (all in the emission frame) so you know what was really
        applied.

        Example::

            # 100 us window straddling the expected 33 us arrival at
            # z=50 mm; sampled every 100 ns.
            data = ver.run_capture(
                time_start_s=-14e-6,
                time_stop_s=86e-6,
                sampling_interval_ns=100,
            )

        Args:
            time_start_s: Start of the capture window relative to
                emission.
            time_stop_s: End of the capture window relative to
                emission.
            sampling_interval_ns: Requested time between samples in ns.
            timeout_s: Max time to wait for the scope trigger to fire.

        Returns:
            The scope data dict with:
              - ``time``: sample times relative to emission (µs).
              - one array per enabled channel (mV).
              - ``sampling_interval_ns``, ``time_start_s``,
                ``time_stop_s``: actual applied values (emission frame).
            Or ``None`` if the scope's trigger timed out.
        """
        if not self.scope:
            raise ValueError("No Picoscope Connected")
        delay_s = self.system_transmit_delay_us * 1e-6
        plan = self.scope.plan_capture(
            sampling_interval_ns=sampling_interval_ns,
            time_start_s=time_start_s + delay_s,
            time_stop_s=time_stop_s + delay_s,
        )
        logger.debug(
            "run_capture: requested %.1f ns / start %.3f us / stop %.3f us "
            "(emission frame); scope start %.3f us / stop %.3f us "
            "(trigger frame); actual %.3f ns / start %.3f us / stop %.3f us "
            "(emission), timebase=%d, pre=%d, post=%d, delay=%d",
            sampling_interval_ns, time_start_s * 1e6, time_stop_s * 1e6,
            plan["time_start_s"] * 1e6, plan["time_stop_s"] * 1e6,
            plan["sampling_interval_ns"],
            (plan["time_start_s"] - delay_s) * 1e6,
            (plan["time_stop_s"] - delay_s) * 1e6,
            plan["timebase"],
            plan["pre_trigger_samples"], plan["post_trigger_samples"],
            plan["delay_samples"],
        )
        # Re-apply the scope trigger with the planned delay. Requires
        # that the user has already called set_trigger to configure
        # channel/threshold/direction/auto_trigger_ms.
        self.scope.set_trigger_delay(plan["delay_samples"])

        result = self._run_capture_block(
            pre_trigger_samples=plan["pre_trigger_samples"],
            post_trigger_samples=plan["post_trigger_samples"],
            timebase=plan["timebase"],
            timeout_s=timeout_s,
        )
        if result is not None:
            # Shift the scope's zero-based time axis into the emission
            # frame: scope-relative start = plan["time_start_s"], then
            # subtract the transmit delay to expose emission-relative
            # times to the caller. Convert ns → µs at the same step so
            # everything downstream (ScanResult.t, waveform plots, ...)
            # is in µs.
            interval_ns = plan["sampling_interval_ns"]
            offset_ns = (plan["time_start_s"] - delay_s) * 1e9
            result["time"] = (result["time"] + offset_ns) * 1e-3
            result["sampling_interval_ns"] = interval_ns
            result["time_start_s"] = plan["time_start_s"] - delay_s
            result["time_stop_s"] = plan["time_stop_s"] - delay_s
        return result

    def capture_pulse_train(self,
                            n_pulses,
                            time_start_s,
                            time_stop_s,
                            sampling_interval_ns,
                            timeout_s=None):
        """Capture a whole LIFU-generated pulse train in a single trigger.

        Requires the LIFU to be configured via ``configure_lifu`` (or
        :meth:`apply_pulse`) with ``pulse_count=n_pulses`` and
        ``trigger_mode="single"``, and HV enabled. The scope is
        armed for ``n_pulses`` rapid-block segments, then one
        ``trigger_once()`` call fires the whole train; each internal
        LIFU pulse re-triggers the scope, filling one segment.

        Unlike :meth:`run_capture` (single pulse) and
        :meth:`run_rapid_sweep` (many triggers, one per point), this
        exercises the LIFU firmware's own inter-pulse timing, so the
        returned segments are useful for inspecting pulse-to-pulse
        jitter and amplitude drift.

        Args:
            n_pulses: Number of internal LIFU pulses to capture. Must
                match the ``pulse_count`` used in the previous
                ``configure_lifu`` call.
            time_start_s, time_stop_s, sampling_interval_ns: Capture
                window applied to every segment (relative to that
                segment's trigger).
            timeout_s: Max time to wait for all ``n_pulses`` triggers.

        Returns:
            Dict from :meth:`finish_rapid_capture` with:

              - ``time``: 1-D sample-time axis (µs), zero at each
                segment's trigger.
              - one entry per enabled channel: ``(n_pulses, samples)``.
              - ``overflow``: 1-D int16, one entry per segment.
              - ``sampling_interval_ns, time_start_s, time_stop_s``.

            Or ``None`` if the scope timed out.
        """
        if not self.scope:
            raise ValueError("No Picoscope Connected")
        plan = self.configure_rapid_capture(
            n_captures=int(n_pulses),
            time_start_s=time_start_s,
            time_stop_s=time_stop_s,
            sampling_interval_ns=sampling_interval_ns,
        )
        self.arm_rapid_capture(plan)
        self.trigger_once()
        return self.finish_rapid_capture(plan, timeout_s=timeout_s)

    # ------------------------------------------------------------------
    # Rapid-block capture (many triggers, one bulk transfer)
    # ------------------------------------------------------------------
    def configure_rapid_capture(self,
                                n_captures,
                                time_start_s,
                                time_stop_s,
                                sampling_interval_ns):
        """Plan a rapid-block capture and configure the scope memory.

        Splits the scope's memory into ``n_captures`` segments, sets
        the scope trigger delay to match ``time_start_s``, and returns
        a plan dict describing the actual applied window / timebase.
        After calling this, use :meth:`arm_rapid_capture`,
        :meth:`trigger_once` (one call per segment), and
        :meth:`finish_rapid_capture` to run the acquisition.

        Args:
            n_captures: Number of triggers/segments to capture in one
                block. Must be <= ``scope.get_max_segments()``.
            time_start_s: Start of the capture window relative to
                emission (see :meth:`run_capture` for the emission-frame
                convention).
            time_stop_s: End of the capture window relative to emission.
            sampling_interval_ns: Requested time between samples in ns.

        Returns:
            The plan dict from ``scope.plan_capture`` (with
            ``time_start_s`` / ``time_stop_s`` still stored in the
            scope's own trigger frame so :meth:`finish_rapid_capture`
            can reconstruct the time axis correctly) plus
            ``n_captures``, ``samples_per_segment``, and
            ``system_transmit_delay_us`` fields.
        """
        if not self.scope:
            raise ValueError("No Picoscope Connected")
        delay_s = self.system_transmit_delay_us * 1e-6
        plan = self.scope.plan_capture(
            sampling_interval_ns=sampling_interval_ns,
            time_start_s=time_start_s + delay_s,
            time_stop_s=time_stop_s + delay_s,
        )
        samples_per_segment = plan["pre_trigger_samples"] + plan["post_trigger_samples"]
        max_per_seg = self.scope.configure_rapid_block(n_captures)
        if samples_per_segment > max_per_seg:
            # Undo segmentation so single-block captures still work.
            self.scope.reset_rapid_block()
            raise ValueError(
                f"Rapid capture needs {samples_per_segment} samples/segment but the "
                f"scope only allows {max_per_seg} samples/segment when split into "
                f"{n_captures} segments. Reduce n_captures, sampling rate, or window."
            )
        plan["n_captures"] = n_captures
        plan["samples_per_segment"] = samples_per_segment
        logger.debug(
            "configure_rapid_capture: n_captures=%d samples/segment=%d "
            "(max %d) window=[%.3f, %.3f] us dt=%.3f ns",
            n_captures, samples_per_segment, max_per_seg,
            plan["time_start_s"] * 1e6, plan["time_stop_s"] * 1e6,
            plan["sampling_interval_ns"],
        )
        return plan

    def arm_rapid_capture(self, plan):
        """Arm the scope for a rapid-block acquisition described by ``plan``.

        Applies the trigger delay from the plan and calls ``run_block``.
        After this returns, the scope is waiting for ``n_captures``
        triggers; call :meth:`trigger_once` that many times.
        """
        if not self.scope:
            raise ValueError("No Picoscope Connected")
        self.scope.set_trigger_delay(plan["delay_samples"])
        self.scope.run_block(
            pre_trigger_samples=plan["pre_trigger_samples"],
            post_trigger_samples=plan["post_trigger_samples"],
            timebase=plan["timebase"],
        )
        # Small settle so the scope is fully armed before the first pulse.
        time.sleep(0.05)

    def trigger_once(self):
        """Fire one TX pulse (one scope trigger).

        Calls ``txdevice.start_trigger()`` directly — one USB round-trip
        (~107 ms on the current firmware) rather than the six that the
        SDK's ``start_sonication`` + ``stop_sonication`` wrappers do
        (~440 ms).

        Assumes:

        - ``configure_lifu(..., trigger_mode="single")`` so the firmware
          fires exactly one pulse train per ``start_trigger`` and
          auto-disarms afterwards.
        - HV is already enabled (``enable_hv_output(wait=True)``).
        - ``async_mode(False)`` is sticky from ``configure_lifu``.
        """
        self.lifu.txdevice.start_trigger()

    def finish_rapid_capture(self, plan, timeout_s=None, reset=True):
        """Wait for the rapid block to complete and bulk-transfer the data.

        Args:
            plan: The dict returned by :meth:`configure_rapid_capture`.
            timeout_s: Max time to wait for all ``n_captures`` triggers.
                ``None`` waits indefinitely.
            reset: If True (default), reset the scope back to
                single-segment mode after retrieval so subsequent
                ``run_capture`` calls work normally.

        Returns:
            Dict with:
              - ``'time'``: 1-D time axis (ns), zero at trigger.
              - one entry per enabled channel: ``(n_captures, samples)`` mV.
              - ``'overflow'``: 1-D int16, one entry per segment.
              - ``sampling_interval_ns``, ``time_start_s``, ``time_stop_s``.
            Or ``None`` if the scope timed out.
        """
        if not self.scope:
            raise ValueError("No Picoscope Connected")
        try:
            if not self.scope.wait_ready(timeout_s=timeout_s):
                logger.warning(
                    "Rapid-block wait_ready timed out after %.3f s", timeout_s or -1.0
                )
                return None
            result = self.scope.get_data_rapid(
                samples_per_segment=plan["samples_per_segment"],
                timebase=plan["timebase"],
            )
        finally:
            if reset:
                try:
                    self.scope.reset_rapid_block()
                except Exception as e:
                    logger.warning("reset_rapid_block raised: %s", e)

        # Shift time axis into the emission frame (see run_capture for
        # details) and convert to µs so downstream ScanResult.t is in
        # µs. Also expose the actual applied window.
        delay_s = self.system_transmit_delay_us * 1e-6
        interval_ns = plan["sampling_interval_ns"]
        offset_ns = (plan["time_start_s"] - delay_s) * 1e9
        result["time"] = (result["time"] + offset_ns) * 1e-3
        result["sampling_interval_ns"] = interval_ns
        result["time_start_s"] = plan["time_start_s"] - delay_s
        result["time_stop_s"] = plan["time_stop_s"] - delay_s
        return result

    def run_averaged_sweep(self,
                           points,
                           apply_point,
                           time_start_s,
                           time_stop_s,
                           sampling_interval_ns,
                           n_averages=1,
                           align=True,
                           align_max_shift_samples=None,
                           chunk_size=None,
                           timeout_s=None,
                           progress="bar",
                           progress_label=None):
        """Repeat-fire each sweep point and (optionally) coherently average.

        Thin wrapper around :meth:`run_rapid_sweep`: expands ``points``
        into ``n_averages`` back-to-back triggers per point (calling
        ``apply_point`` only on the first repeat of each group, so the
        HV rail / delay profile is not re-programmed for the repeats),
        then folds the flat capture list back to per-point groups and
        averages each channel across the repeats.

        When ``align`` is ``True`` the hydrophone channel is aligned by
        cross-correlation (via
        :func:`openlifu_verification.pulse_align.align_pulse_traces`)
        before averaging so sub-sample trigger jitter doesn't smear
        the coherent sum. Non-hydrophone channels are averaged with
        the same per-point lags so they stay coherent with the
        hydrophone.

        Args:
            points, apply_point, time_start_s, time_stop_s,
            sampling_interval_ns, chunk_size, timeout_s, progress,
            progress_label: See :meth:`run_rapid_sweep`.
            n_averages: Repeats per point. ``1`` (default) reduces to
                a plain :meth:`run_rapid_sweep`.
            align: If ``True`` (default), cross-correlate repeats
                against the first repeat before averaging.
            align_max_shift_samples: Optional cap on the lag search
                (samples). ``None`` searches the full range.

        Returns:
            ``(outputs, timings, averaging)`` where ``outputs`` and
            ``timings`` have the same shape as :meth:`run_rapid_sweep`
            (one entry per input point, not per repeat) and
            ``averaging`` is a list of per-point dicts with keys
            ``n_averages``, ``lags_s`` (shape ``(n_averages,)``),
            ``noise_rms`` (per-sample std RMS on the hydrophone
            channel).
        """
        n_averages = int(n_averages)
        if n_averages < 1:
            raise ValueError("n_averages must be >= 1")
        points = list(points)
        n_points = len(points)
        if n_points == 0:
            return [], [], []

        if n_averages == 1:
            outputs, timings = self.run_rapid_sweep(
                points=points,
                apply_point=apply_point,
                time_start_s=time_start_s,
                time_stop_s=time_stop_s,
                sampling_interval_ns=sampling_interval_ns,
                chunk_size=chunk_size,
                timeout_s=timeout_s,
                progress=progress,
                progress_label=progress_label,
            )
            averaging = [{"n_averages": 1,
                          "lags_s": np.zeros(1),
                          "noise_rms": float("nan")} for _ in outputs]
            return outputs, timings, averaging

        # Expand points -> (point, repeat_idx). apply_point is only
        # called on repeat 0 of each group.
        expanded = [(pt, r) for pt in points for r in range(n_averages)]

        def _apply_once(pt_repeat):
            pt, r = pt_repeat
            if r == 0:
                apply_point(pt)

        # Chunk size, if given, applies to physical scope segments.
        # We must ensure whole groups of ``n_averages`` land in the same
        # chunk so a group is never split by an arm/xfer boundary
        # (which would break within-group alignment). Round chunk_size
        # DOWN to a multiple of n_averages.
        if chunk_size is not None and chunk_size > 0:
            eff_chunk = max(n_averages, (chunk_size // n_averages) * n_averages)
        else:
            eff_chunk = n_averages * n_points

        raw_outputs, raw_timings = self.run_rapid_sweep(
            points=expanded,
            apply_point=_apply_once,
            time_start_s=time_start_s,
            time_stop_s=time_stop_s,
            sampling_interval_ns=sampling_interval_ns,
            chunk_size=eff_chunk,
            timeout_s=timeout_s,
            progress=progress,
            progress_label=progress_label,
        )

        outputs = [None] * n_points
        timings = []
        averaging = []
        hydro = self.hydrophone_channel

        for i in range(n_points):
            group = raw_outputs[i * n_averages:(i + 1) * n_averages]
            group_timings = raw_timings[i * n_averages:(i + 1) * n_averages]
            captured = [g for g in group if g is not None]
            # Sum-of-timings so the caller sees the *actual* wall time
            # spent on this point (all repeats combined).
            agg_t = {
                "apply_s": sum(t.get("apply_s", 0.0) for t in group_timings),
                "trigger_s": sum(t.get("trigger_s", 0.0) for t in group_timings),
                "iter_total_s": sum(t.get("iter_total_s", 0.0) for t in group_timings),
                "arm_s": sum(t.get("arm_s", 0.0) for t in group_timings),
                "xfer_s": sum(t.get("xfer_s", 0.0) for t in group_timings),
                "captured": bool(captured),
                "chunk_index": group_timings[0].get("chunk_index")
                if group_timings else None,
                "n_averages": n_averages,
                "n_captured": len(captured),
            }
            timings.append(agg_t)

            if not captured:
                averaging.append({
                    "n_averages": n_averages,
                    "n_captured": 0,
                    "lags_s": np.zeros(n_averages),
                    "noise_rms": float("nan"),
                })
                continue

            t_axis = captured[0]["time"]
            sampling_interval_ns_actual = float(captured[0]["sampling_interval_ns"])
            dt_s = sampling_interval_ns_actual * 1e-9

            hydro_stack = np.stack(
                [np.asarray(g[hydro], dtype=float) for g in captured], axis=0,
            )
            if align and hydro_stack.shape[0] >= 2:
                aligned_hydro, lags_s = align_pulse_traces(
                    hydro_stack,
                    dt_s=dt_s,
                    max_shift_samples=align_max_shift_samples,
                    reference="first",
                )
            else:
                aligned_hydro = hydro_stack
                lags_s = np.zeros(hydro_stack.shape[0])

            # Per-sample std across repeats -> a diagnostic noise floor.
            noise_std = aligned_hydro.std(axis=0, ddof=0)
            noise_rms = float(np.sqrt(np.mean(noise_std ** 2)))

            per_point = {
                "time": t_axis,
                "sampling_interval_ns": sampling_interval_ns_actual,
                "time_start_s": captured[0].get("time_start_s"),
                "time_stop_s": captured[0].get("time_stop_s"),
                "overflow": int(np.max([g.get("overflow", 0) for g in captured])),
                hydro: aligned_hydro.mean(axis=0),
            }
            # Average any other enabled channels with the same lags so
            # they stay coherent with the hydrophone (e.g. sync channel).
            lag_samples = lags_s / dt_s if dt_s > 0 else np.zeros_like(lags_s)
            for ch in self.scope.enabled_channels:
                if ch == hydro:
                    continue
                ch_stack = [np.asarray(g[ch], dtype=float) for g in captured]
                if align and len(ch_stack) >= 2:
                    from .pulse_align import shift_trace
                    aligned_ch = np.stack(
                        [shift_trace(x, l) for x, l in zip(ch_stack, lag_samples)],
                        axis=0,
                    )
                else:
                    aligned_ch = np.stack(ch_stack, axis=0)
                per_point[ch] = aligned_ch.mean(axis=0)
            outputs[i] = per_point
            averaging.append({
                "n_averages": n_averages,
                "n_captured": len(captured),
                "lags_s": lags_s,
                "noise_rms": noise_rms,
            })

        return outputs, timings, averaging

    def run_rapid_sweep(self,
                        points,
                        apply_point,
                        time_start_s,
                        time_stop_s,
                        sampling_interval_ns,
                        chunk_size=None,
                        timeout_s=None,
                        progress="bar",
                        progress_label=None):
        """Run a rapid-block sweep over ``points``.

        Splits ``points`` into chunks of size ``chunk_size``. For each
        chunk:

          1. ``configure_rapid_capture`` + ``arm_rapid_capture`` — arm the
             scope for ``len(chunk)`` segments.
          2. For each point, call ``apply_point(point)`` (a caller
             supplied callback that programs the per-point state, e.g.
             ``set_focus``, ``set_pulse``, ``set_voltage``), then
             ``trigger_once()``.
          3. ``finish_rapid_capture`` — wait for all triggers and pull
             the block back in one bulk transfer.
          4. Split the ``(n_captures, samples)`` bulk buffer into a
             per-point dict (one 1-D array per enabled channel) matching
             the shape of ``run_capture`` results.

        Args:
            points: Iterable of point descriptors. Each is passed
                unchanged to ``apply_point``.
            apply_point: Callable ``apply_point(point) -> None`` that
                programs the device state for one point (delays,
                frequency, HV voltage, etc.).
            time_start_s, time_stop_s, sampling_interval_ns: Capture
                window (same semantics as ``run_capture``).
            chunk_size: Maximum points per rapid block. Defaults to
                ``len(points)`` (one big chunk). Capped internally at
                ``scope.get_max_segments()``.
            timeout_s: Max time to wait for each chunk to finish.
                ``None`` waits indefinitely.
            progress: How to report per-point progress. One of:

                - ``"bar"`` (default): tqdm progress bar with rolling
                  apply/trigger times in the description.
                - ``"log"``: emit one INFO log line per point.
                - ``"both"``: bar + log lines.
                - ``None`` / ``""``: silent.
                - a callable ``fn(point, timing) -> None``: your own
                  callback, invoked after every trigger.

            progress_label: Optional short label shown in the tqdm bar
                (e.g. ``"scan_lat"``). Ignored for other modes.

        Returns:
            ``(outputs, timings)``:

            - ``outputs``: list of per-point data dicts (same shape as
              ``run_capture``) or ``None`` for any point in a chunk
              whose transfer timed out.
            - ``timings``: list of per-point dicts with keys
              ``apply_s``, ``trigger_s``, ``arm_s`` (amortized),
              ``xfer_s`` (amortized), ``iter_total_s``, ``captured``,
              ``chunk_index``.
        """
        if not self.scope:
            raise ValueError("No Picoscope Connected")
        points = list(points)
        n_points = len(points)
        if n_points == 0:
            return [], []
        max_segments = self.scope.get_max_segments()
        if chunk_size is None or chunk_size <= 0:
            chunk_size = n_points
        chunk_size = min(chunk_size, max_segments)

        outputs = [None] * n_points
        timings = []

        progress_cb, bar, close_bar = self._make_progress(
            progress, n_points, progress_label,
        )

        try:
            for chunk_index, chunk_start in enumerate(range(0, n_points, chunk_size)):
                chunk = points[chunk_start:chunk_start + chunk_size]
                n_captures = len(chunk)

                t_arm_start = time.perf_counter()
                plan = self.configure_rapid_capture(
                    n_captures=n_captures,
                    time_start_s=time_start_s,
                    time_stop_s=time_stop_s,
                    sampling_interval_ns=sampling_interval_ns,
                )
                self.arm_rapid_capture(plan)
                t_arm = time.perf_counter() - t_arm_start

                per_point_timings = []
                for point in chunk:
                    t_iter_start = time.perf_counter()

                    t0 = time.perf_counter()
                    apply_point(point)
                    t_apply = time.perf_counter() - t0

                    t0 = time.perf_counter()
                    self.trigger_once()
                    t_trigger = time.perf_counter() - t0

                    t_iter_total = time.perf_counter() - t_iter_start
                    per_point_timings.append({
                        "apply_s": t_apply,
                        "trigger_s": t_trigger,
                        "iter_total_s": t_iter_total,
                    })
                    # Tick the bar as each trigger fires so the user sees
                    # real-time progress. Doing this after finish_rapid_capture
                    # would make the bar sit at 0 for the whole chunk and
                    # then zip to 100 % during the fast output-unpack loop.
                    if bar is not None:
                        bar.set_postfix_str(
                            f"apply={t_apply:.3f}s trigger={t_trigger:.3f}s",
                            refresh=False,
                        )
                        bar.update(1)

                t_xfer_start = time.perf_counter()
                bulk = self.finish_rapid_capture(plan, timeout_s=timeout_s)
                t_xfer = time.perf_counter() - t_xfer_start

                arm_per_pt = t_arm / n_captures
                xfer_per_pt = t_xfer / n_captures
                logger.debug(
                    "chunk %d [%d:%d] arm=%.4fs xfer=%.4fs captured=%s",
                    chunk_index, chunk_start, chunk_start + n_captures,
                    t_arm, t_xfer, bulk is not None,
                )

                for i, (point, pt) in enumerate(zip(chunk, per_point_timings)):
                    pt.update({
                        "chunk_index": chunk_index,
                        "arm_s": arm_per_pt,
                        "xfer_s": xfer_per_pt,
                        "captured": bulk is not None,
                    })
                    if bulk is not None:
                        per_point_data = {
                            "time": bulk["time"],
                            "sampling_interval_ns": bulk["sampling_interval_ns"],
                            "time_start_s": bulk["time_start_s"],
                            "time_stop_s": bulk["time_stop_s"],
                            "overflow": bulk["overflow"][i],
                        }
                        for ch in self.scope.enabled_channels:
                            per_point_data[ch] = bulk[ch][i]
                        outputs[chunk_start + i] = per_point_data
                    timings.append(pt)
                    if progress_cb is not None:
                        progress_cb(point, pt)
        finally:
            if close_bar:
                bar.close()

        return outputs, timings

    def _make_progress(self, progress, n_points, label):
        """Turn the ``progress`` arg into (callback, tqdm_bar, close_bar).

        Returns:
            Tuple of ``(callback, bar, close_bar)``. Either component
            may be ``None`` if it doesn't apply.
        """
        if callable(progress):
            return progress, None, False
        if progress in (None, "", False):
            return None, None, False
        mode = str(progress).lower()
        if mode not in ("bar", "log", "both"):
            raise ValueError(
                f"progress must be None, a callable, or one of "
                f"'bar'/'log'/'both'; got {progress!r}."
            )

        log_cb = None
        if mode in ("log", "both"):
            def _log_cb(point, pt, _label=label):
                logger.info(
                    "%s point=%s  apply=%.4fs trigger=%.4fs iter=%.4fs",
                    _label or "sweep", _fmt_point(point),
                    pt["apply_s"], pt["trigger_s"], pt["iter_total_s"],
                )
            log_cb = _log_cb

        bar = None
        if mode in ("bar", "both"):
            try:
                from tqdm.auto import tqdm  # type: ignore
                bar = tqdm(total=n_points, desc=label or "sweep",
                           unit="pt", leave=True)
            except ImportError:
                logger.warning(
                    "progress=%r requested but tqdm is not installed; "
                    "falling back to log-only.", progress,
                )
                if log_cb is None:
                    def _fallback_cb(point, pt, _label=label):
                        logger.info(
                            "%s point=%s  apply=%.4fs trigger=%.4fs",
                            _label or "sweep", _fmt_point(point),
                            pt["apply_s"], pt["trigger_s"],
                        )
                    log_cb = _fallback_cb

        return log_cb, bar, bar is not None

    def _run_capture_block(self, pre_trigger_samples=2500, post_trigger_samples=10000, timebase=8, timeout_s=2.0):
        """Low-level: fire a TX trigger and capture N samples at a given timebase.

        Prefer :meth:`run_capture`, which accepts time-based inputs
        (``time_start_s``, ``time_stop_s``, ``sampling_interval_ns``)
        and re-applies the scope trigger delay. This method is exposed
        for callers that already know the sample counts and timebase
        they want (e.g. rapid-block mode).

        Args:
            pre_trigger_samples: Samples captured before the scope trigger.
            post_trigger_samples: Samples captured after the scope trigger
                (or after ``trigger + delay`` if a trigger delay is set).
            timebase: Picoscope timebase index.
            timeout_s: Max time to wait for the scope trigger to fire.

        Returns:
            Data dict from the scope, or ``None`` if the scope never
            saw a trigger within *timeout_s*.
        """
        if not self.scope:
            raise ValueError("No Picoscope Connected")
        logger.debug("Sending Single Trigger...")
        self.scope.run_block(pre_trigger_samples=pre_trigger_samples, post_trigger_samples=post_trigger_samples, timebase=timebase)
        # Give the scope a moment to fully arm before firing the TX pulse.
        time.sleep(0.05)
        # Fast path: fire via txdevice.start_trigger() directly (one USB
        # round-trip ~107 ms), skipping the SDK's start_sonication /
        # stop_sonication wrappers which add 5 more round-trips for
        # bookkeeping (HV re-check, async_mode toggles). Assumes
        # configure_lifu was called with trigger_mode="single" so the
        # firmware auto-disarms after one pulse train, enable_hv_output
        # left HV energized, and async_mode(False) is sticky from
        # configure_lifu.
        self.lifu.txdevice.start_trigger()
        if not self.scope.wait_ready(timeout_s=timeout_s):
            logger.warning("Scope trigger timed out after %.3f s; no pulse captured.", timeout_s)
            return None
        return self.scope.get_data(pre_trigger_samples+post_trigger_samples, timebase)

    def set_voltage(self, voltage, wait=False):
        """
        Sets the high-voltage supply voltage.

        When using the external power supply, this sets both channels of the
        QPX600DP. When using the internal supply, this sets the LIFU
        HVController output voltage.
        """
        self._set_hv_voltage(voltage)
        if wait:
            self._wait_hv_ready(voltage)

    def enable_hv_output(self, wait=False):
        """
        Enables the high-voltage output.

        Works with either the external QPX600DP or the internal LIFU
        HVController, depending on how the tank was configured.

        Args:
            wait: If True, block until the HV rail has settled to the
                requested voltage.
        """
        if self.ext_power_supply:
            if self.hv is not None:
                self.hv.set_all_outputs(True)
        else:
            if self.lifu is not None and self.lifu.hvcontroller is not None:
                self.lifu.hvcontroller.turn_hv_on()
        self.hv_enabled = True
        if wait:
            self._wait_hv_ready()

    def disable_hv_output(self):
        """
        Disables the high-voltage output.

        Works with either the external QPX600DP or the internal LIFU
        HVController, depending on how the tank was configured.
        """
        if self.ext_power_supply:
            if self.hv is not None:
                self.hv.set_all_outputs(False)
        else:
            if self.lifu is not None and self.lifu.hvcontroller is not None:
                self.lifu.hvcontroller.turn_hv_off()
        self.hv_enabled = False

    def _set_hv_voltage(self, voltage):
        # set_voltage is a potentially slow USB/serial round-trip; skip it
        # when the requested voltage matches what we last programmed.
        if self.hv_voltage is not None and voltage == self.hv_voltage:
            return
        if self.ext_power_supply:
            self.hv.set_voltage(voltage)
        else:
            self.lifu.hvcontroller.set_voltage(voltage)
        self.hv_voltage = voltage

    def _wait_hv_ready(self, voltage=None):
        if self.ext_power_supply:
            self.hv.wait_ready(target=voltage)
        else:
            self.lifu.hvcontroller.wait_for_settle()

    # ------------------------------------------------------------------
    # Logging utility
    # ------------------------------------------------------------------
    def add_log_file(self, path, level=logging.INFO,
                     fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s"):
        """Stream ``openlifu_verification`` log records to ``path``.

        Adds a :class:`logging.FileHandler` to the top-level
        ``openlifu_verification`` package logger. All submodule loggers
        (``openlifu_verification.verificationtank``,
        ``openlifu_verification.picoscope``, …) inherit the handler.

        This does not touch the root logger, and it does not silence any
        existing handlers — it simply mirrors the same records into the
        file. Call :meth:`remove_log_file` (or close the returned handler
        yourself) when you're done.

        Args:
            path: File path for the log. Parent dirs are created.
            level: Minimum log level captured to the file.
            fmt: ``logging`` format string. The default includes a
                timestamp, level, logger name, and the message.

        Returns:
            The :class:`logging.FileHandler` that was attached.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter(fmt))
        # Make sure the package logger will forward records to the handler.
        if _PACKAGE_LOGGER.level > level or _PACKAGE_LOGGER.level == logging.NOTSET:
            _PACKAGE_LOGGER.setLevel(level)
        _PACKAGE_LOGGER.addHandler(handler)
        logger.info("Logging to %s at level %s.", path, logging.getLevelName(level))
        if not hasattr(self, "_log_handlers"):
            self._log_handlers = []
        self._log_handlers.append(handler)
        return handler

    def remove_log_file(self, handler=None):
        """Detach a file handler installed by :meth:`add_log_file`.

        Args:
            handler: The specific handler to remove; if ``None``, all
                handlers installed via :meth:`add_log_file` are removed.
        """
        handlers = getattr(self, "_log_handlers", [])
        if handler is None:
            for h in list(handlers):
                _PACKAGE_LOGGER.removeHandler(h)
                h.close()
            handlers.clear()
        else:
            _PACKAGE_LOGGER.removeHandler(handler)
            handler.close()
            if handler in handlers:
                handlers.remove(handler)

    # ------------------------------------------------------------------
    # High-level scans
    # ------------------------------------------------------------------
    def scan_1d(self, *,
                dim,
                scan_range=(-10.0, 10.0),
                num=41,
                absolute=False,
                x=0.0,
                y=0.0,
                z=None,
                time_start_s=-14e-6,
                time_stop_s=86e-6,
                sampling_interval_ns=100,
                chunk_size=0,
                timeout_s=None,
                n_averages=1,
                align=True,
                align_max_shift_samples=None,
                progress="bar") -> ScanResult:
        """Sweep the focus along a single spatial axis.

        Generic 1-D scan along ``dim`` (``"x"``, ``"y"``, or ``"z"``).
        The other two coordinates are held fixed at the values given
        by ``x``, ``y``, ``z`` (or, when ``None`` / ``0``, at the
        calibrated hydrophone position).

        Requires the tank to be already configured
        (:meth:`configure_lifu`) and HV enabled
        (:meth:`enable_hv_output` with ``wait=True``).

        Args:
            dim: Which axis to sweep. One of ``"x"``, ``"y"``, ``"z"``.
            scan_range: ``(min, max)`` in mm along ``dim``.
            num: Number of samples across ``scan_range``.
            absolute: If ``False`` (default), all three coordinates are
                interpreted relative to the calibrated
                ``hydrophone_position`` so a sweep of e.g. -1..+1 mm
                lands on the true peak. If ``True``, coordinates are
                absolute in the transducer frame.
            x, y: Fixed offsets on the non-swept lateral axes.
                Ignored for the corresponding swept dimension.
            z: Fixed depth on the non-swept axial axis (mm). ``None``
                means "at the calibrated hydrophone depth"
                (``hydrophone_position[2]``). Ignored when
                ``dim == "z"``. Always interpreted as an *absolute*
                depth when it is a fixed coord; when it is the swept
                coord and ``absolute=False``, ``scan_range`` is added
                to the calibrated ``hydrophone_position[2]``.
            time_start_s, time_stop_s, sampling_interval_ns: Capture
                window for each pulse (see :meth:`run_capture` for the
                emission-relative time-base convention).
            chunk_size: Rapid-block chunk size (0 = whole sweep).
            timeout_s: Passed through to
                :meth:`finish_rapid_capture`.
            n_averages, align, align_max_shift_samples: Passed through
                to :meth:`run_averaged_sweep`.
            progress: See :meth:`run_rapid_sweep`.

        Returns:
            A :class:`ScanResult` with ``scan_type="1d"``, ``metadata["dim"]``
            giving the swept axis, and a single coord axis named
            ``"xfoci"`` / ``"yfoci"`` / ``"zfoci"``. Units are ``"Pa"``
            if a hydrophone calibration is attached, otherwise ``"mV"``.
        """
        dim = str(dim).lower()
        if dim not in ("x", "y", "z"):
            raise ValueError(f"dim must be 'x', 'y', or 'z' (got {dim!r})")

        # Origin for the "relative" mode is the calibrated peak; in
        # absolute mode the origin is (0, 0, 0). The z coord treats the
        # ``z=`` kwarg as absolute either way (matches scan_lateral).
        if absolute:
            x_origin = y_origin = z_origin = 0.0
        else:
            x_origin = float(self.hydrophone_position[0])
            y_origin = float(self.hydrophone_position[1])
            z_origin = float(self.hydrophone_position[2])
        z_fixed = z_origin if z is None else float(z)

        coord_axis = np.linspace(scan_range[0], scan_range[1], num)
        logger.info(
            "scan_1d: dim=%s, range=[%g, %g] mm (%s), num=%d, fixed x=%g mm, "
            "y=%g mm, z=%g mm; capture [%g, %g] µs @ %g ns; n_averages=%d",
            dim, scan_range[0], scan_range[1],
            "absolute" if absolute else "relative to hydrophone",
            num, float(x) + x_origin, float(y) + y_origin, z_fixed,
            time_start_s * 1e6, time_stop_s * 1e6, sampling_interval_ns,
            n_averages,
        )
        if dim == "x":
            focus_points = [
                (float(v) + x_origin, float(y) + y_origin, z_fixed)
                for v in coord_axis
            ]
            coord_key = "xfoci"
            progress_label = "scan_1d_x"
        elif dim == "y":
            focus_points = [
                (float(x) + x_origin, float(v) + y_origin, z_fixed)
                for v in coord_axis
            ]
            coord_key = "yfoci"
            progress_label = "scan_1d_y"
        else:  # dim == "z"
            focus_points = [
                (float(x) + x_origin, float(y) + y_origin, float(v) + z_origin)
                for v in coord_axis
            ]
            coord_key = "zfoci"
            progress_label = "scan_1d_z"

        def apply_point(point):
            xi, yi, zi = point
            self.set_focus(xi, yi, zi)

        outputs, timings, averaging = self.run_averaged_sweep(
            points=focus_points,
            apply_point=apply_point,
            time_start_s=time_start_s,
            time_stop_s=time_stop_s,
            sampling_interval_ns=sampling_interval_ns,
            n_averages=n_averages,
            align=align,
            align_max_shift_samples=align_max_shift_samples,
            chunk_size=chunk_size or len(focus_points),
            timeout_s=timeout_s,
            progress=progress,
            progress_label=progress_label,
        )
        traces, t_axis, ok_mask = self._stack_hydrophone_traces(outputs)
        if traces is None:
            raise RuntimeError(f"No points captured in scan_1d(dim={dim!r}).")

        traces, units = self._convert_to_pressure(traces, self.frequency * 1e3)
        traces = traces.reshape(num, -1)

        return ScanResult(
            scan_type="1d",
            t=t_axis,
            traces=traces,
            coords={coord_key: coord_axis},
            hydrophone_channel=self.hydrophone_channel,
            chunk_size=chunk_size or len(focus_points),
            timings=_collect_timings(timings),
            units=units,
            metadata={
                "dim": dim,
                "z_mm": float(z_fixed),
                "frequency_kHz": float(self.frequency),
                "voltage_V": float(self.hv_voltage) if self.hv_voltage is not None else float("nan"),
                "captured_mask": ok_mask,
                "hydrophone_position_mm": self.hydrophone_position.copy(),
                "absolute": bool(absolute),
                "n_averages": int(n_averages),
                "align": bool(align),
                "averaging": averaging,
            },
        )

    def scan_lateral(self, *,
                     x_range=(-10.0, 10.0),
                     num_x=41,
                     y_range=None,
                     num_y=1,
                     y=0.0,
                     z=None,
                     absolute=False,
                     time_start_s=-14e-6,
                     time_stop_s=86e-6,
                     sampling_interval_ns=100,
                     chunk_size=0,
                     timeout_s=None,
                     n_averages=1,
                     align=True,
                     align_max_shift_samples=None,
                     progress="bar") -> ScanResult:
        """Sweep the focus over an (x, y) grid at a fixed z.

        Requires the tank to be already configured
        (:meth:`configure_lifu`) and HV enabled
        (:meth:`enable_hv_output` with ``wait=True``).

        Args:
            x_range: ``(x_min, x_max)`` in mm.
            num_x: Number of x samples across ``x_range``.
            y_range: Optional ``(y_min, y_max)`` in mm. If ``None`` and
                ``num_y == 1`` the sweep runs a single row at ``y=y``.
            num_y: Number of y samples across ``y_range`` (1 for a line
                scan).
            y: y coordinate used when ``num_y == 1`` and ``y_range`` is
                ``None``.
            z: Fixed z depth (mm). ``None`` uses the calibrated
                ``hydrophone_position[2]``.
            absolute: If ``False`` (default), ``x_range``/``y_range``/``y``
                are relative to the calibrated ``hydrophone_position``
                so a sweep of e.g. -1..+1 mm around 0 lands on the true
                peak. If ``True``, the coordinates are absolute in the
                transducer frame.
            time_start_s, time_stop_s, sampling_interval_ns: Capture
                window for each pulse.
            chunk_size: Rapid-block chunk size (0 = whole sweep).
            timeout_s: Passed through to
                :meth:`finish_rapid_capture`.
            progress: See :meth:`run_rapid_sweep`.

        Returns:
            A :class:`ScanResult` with ``scan_type="lateral"``. Coord
            axes are ``yfoci``, ``xfoci`` (only ``xfoci`` if ``num_y ==
            1``). Units are ``"Pa"`` if a hydrophone calibration is
            attached, otherwise ``"mV"``.
        """
        if z is None:
            z = float(self.hydrophone_position[2])
        x_off, y_off = (0.0, 0.0) if absolute else (
            float(self.hydrophone_position[0]), float(self.hydrophone_position[1])
        )

        xfoci = np.linspace(x_range[0], x_range[1], num_x)
        if num_y > 1:
            if y_range is None:
                raise ValueError("y_range must be given when num_y > 1.")
            yfoci = np.linspace(y_range[0], y_range[1], num_y)
        else:
            yfoci = np.array([float(y)])

        if num_y > 1:
            logger.info(
                "scan_2d: x=[%g, %g] mm (%d), y=[%g, %g] mm (%d), z=%g mm "
                "(%s); capture [%g, %g] µs @ %g ns; n_averages=%d",
                x_range[0], x_range[1], num_x,
                y_range[0], y_range[1], num_y, float(z),
                "absolute" if absolute else "relative to hydrophone",
                time_start_s * 1e6, time_stop_s * 1e6, sampling_interval_ns,
                n_averages,
            )
        else:
            logger.info(
                "scan_lateral: x=[%g, %g] mm (%d) at y=%g mm, z=%g mm (%s); "
                "capture [%g, %g] µs @ %g ns; n_averages=%d",
                x_range[0], x_range[1], num_x, float(y), float(z),
                "absolute" if absolute else "relative to hydrophone",
                time_start_s * 1e6, time_stop_s * 1e6, sampling_interval_ns,
                n_averages,
            )

        focus_points = [(float(xi) + x_off, float(yi) + y_off, float(z))
                        for yi in yfoci for xi in xfoci]

        def apply_point(point):
            xi, yi, zi = point
            self.set_focus(xi, yi, zi)

        outputs, timings, averaging = self.run_averaged_sweep(
            points=focus_points,
            apply_point=apply_point,
            time_start_s=time_start_s,
            time_stop_s=time_stop_s,
            sampling_interval_ns=sampling_interval_ns,
            n_averages=n_averages,
            align=align,
            align_max_shift_samples=align_max_shift_samples,
            chunk_size=chunk_size or len(focus_points),
            timeout_s=timeout_s,
            progress=progress,
            progress_label="scan_lat" if num_y == 1 else "scan_2d",
        )
        traces, t_axis, ok_mask = self._stack_hydrophone_traces(outputs)
        if traces is None:
            raise RuntimeError("No points captured in scan_lateral.")

        traces, units = self._convert_to_pressure(traces, self.frequency * 1e3)

        if num_y > 1:
            traces = traces.reshape(num_y, num_x, -1)
            coords = {"yfoci": yfoci, "xfoci": xfoci}
        else:
            traces = traces.reshape(num_x, -1)
            coords = {"xfoci": xfoci}

        return ScanResult(
            scan_type="lateral" if num_y == 1 else "2d",
            t=t_axis,
            traces=traces,
            coords=coords,
            hydrophone_channel=self.hydrophone_channel,
            chunk_size=chunk_size or len(focus_points),
            timings=_collect_timings(timings),
            units=units,
            metadata={
                "z_mm": float(z),
                "frequency_kHz": float(self.frequency),
                "voltage_V": float(self.hv_voltage) if self.hv_voltage is not None else float("nan"),
                "captured_mask": ok_mask,
                "hydrophone_position_mm": self.hydrophone_position.copy(),
                "absolute": bool(absolute),
                "n_averages": int(n_averages),
                "align": bool(align),
                "averaging": averaging,
            },
        )

    def scan_2d(self, *,
                x_range=(-4.0, 4.0),
                num_x=9,
                y_range=(-4.0, 4.0),
                num_y=9,
                z=None,
                absolute=False,
                time_start_s=-14e-6,
                time_stop_s=86e-6,
                sampling_interval_ns=100,
                chunk_size=0,
                timeout_s=None,
                n_averages=1,
                align=True,
                align_max_shift_samples=None,
                progress="bar") -> ScanResult:
        """Convenience wrapper: 2-D focus grid.

        Same as :meth:`scan_lateral` with ``num_y > 1``. Returns a
        :class:`ScanResult` with ``scan_type="2d"``.
        """
        return self.scan_lateral(
            x_range=x_range, num_x=num_x,
            y_range=y_range, num_y=num_y,
            z=z,
            absolute=absolute,
            time_start_s=time_start_s,
            time_stop_s=time_stop_s,
            sampling_interval_ns=sampling_interval_ns,
            chunk_size=chunk_size,
            timeout_s=timeout_s,
            n_averages=n_averages,
            align=align,
            align_max_shift_samples=align_max_shift_samples,
            progress=progress,
        )

    def scan_frequency(self, *,
                       frequencies_kHz,
                       duration_usec,
                       time_start_s=-14e-6,
                       time_stop_s=86e-6,
                       sampling_interval_ns=100,
                       chunk_size=0,
                       timeout_s=None,
                       n_averages=1,
                       align=True,
                       align_max_shift_samples=None,
                       progress="bar") -> ScanResult:
        """Sweep the TX pulse frequency at the current focus.

        Requires that :meth:`set_focus` and :meth:`enable_hv_output`
        have already been called.

        Args:
            frequencies_kHz: 1-D iterable of frequencies to sweep.
            duration_usec: Pulse duration (\u00b5s) reused at every point;
                the number of cycles per pulse is
                ``int(duration_usec * frequency_kHz / 1000)``.
            time_start_s, time_stop_s, sampling_interval_ns: Capture
                window.
            chunk_size: Rapid-block chunk size (0 = whole sweep).
            timeout_s: Passed through to
                :meth:`finish_rapid_capture`.
            progress: See :meth:`run_rapid_sweep`.

        Returns:
            A :class:`ScanResult` with ``scan_type="frequency"`` and
            coord ``freq_kHz``.
        """
        freqs = np.asarray(list(frequencies_kHz), dtype=float)
        if freqs.size:
            logger.info(
                "scan_frequency: %d freqs from %g to %g kHz, duration=%g µs; "
                "capture [%g, %g] µs @ %g ns; n_averages=%d",
                freqs.size, float(freqs.min()), float(freqs.max()),
                float(duration_usec),
                time_start_s * 1e6, time_stop_s * 1e6, sampling_interval_ns,
                n_averages,
            )

        def apply_point(freq_kHz):
            self.set_pulse(frequency_kHz=freq_kHz, duration_usec=duration_usec)

        outputs, timings, averaging = self.run_averaged_sweep(
            points=freqs.tolist(),
            apply_point=apply_point,
            time_start_s=time_start_s,
            time_stop_s=time_stop_s,
            sampling_interval_ns=sampling_interval_ns,
            n_averages=n_averages,
            align=align,
            align_max_shift_samples=align_max_shift_samples,
            chunk_size=chunk_size or len(freqs),
            timeout_s=timeout_s,
            progress=progress,
            progress_label="scan_freq",
        )
        traces, t_axis, ok_mask = self._stack_hydrophone_traces(outputs)
        if traces is None:
            raise RuntimeError("No points captured in scan_frequency.")
        # Per-point frequency lookup for pressure conversion.
        good_freqs_kHz = freqs[ok_mask] if not ok_mask.all() else freqs
        traces, units = self._convert_to_pressure(traces, good_freqs_kHz * 1e3)
        return ScanResult(
            scan_type="frequency",
            t=t_axis,
            traces=traces,
            coords={"freq_kHz": good_freqs_kHz},
            hydrophone_channel=self.hydrophone_channel,
            chunk_size=chunk_size or len(freqs),
            timings=_collect_timings(timings),
            units=units,
            metadata={
                "voltage_V": float(self.hv_voltage) if self.hv_voltage is not None else float("nan"),
                "duration_usec": float(duration_usec),
                "captured_mask": ok_mask,
                "hydrophone_position_mm": self.hydrophone_position.copy(),
                "n_averages": int(n_averages),
                "align": bool(align),
                "averaging": averaging,
            },
        )

    def scan_voltage(self, *,
                     voltages_V,
                     time_start_s=-14e-6,
                     time_stop_s=86e-6,
                     sampling_interval_ns=100,
                     chunk_size=0,
                     timeout_s=None,
                     n_averages=1,
                     align=True,
                     align_max_shift_samples=None,
                     progress="bar") -> ScanResult:
        """Sweep the HV rail voltage at the current focus/pulse profile.

        Each point calls :meth:`set_voltage` with ``wait=True`` before
        firing the trigger, which adds ~50-200 ms per point for HV
        settling. Requires that :meth:`enable_hv_output` was called
        already.

        Args:
            voltages_V: 1-D iterable of voltages to sweep.
            time_start_s, time_stop_s, sampling_interval_ns: Capture
                window.
            chunk_size: Rapid-block chunk size (0 = whole sweep).
            timeout_s: Passed through to
                :meth:`finish_rapid_capture`.
            progress: See :meth:`run_rapid_sweep`.

        Returns:
            A :class:`ScanResult` with ``scan_type="voltage"`` and
            coord ``voltage_V``.
        """
        voltages = np.asarray(list(voltages_V), dtype=float)
        if voltages.size:
            logger.info(
                "scan_voltage: %d voltages from %g to %g V; capture [%g, %g] "
                "µs @ %g ns; n_averages=%d",
                voltages.size, float(voltages.min()), float(voltages.max()),
                time_start_s * 1e6, time_stop_s * 1e6, sampling_interval_ns,
                n_averages,
            )

        def apply_point(voltage):
            self.set_voltage(float(voltage), wait=True)

        outputs, timings, averaging = self.run_averaged_sweep(
            points=voltages.tolist(),
            apply_point=apply_point,
            time_start_s=time_start_s,
            time_stop_s=time_stop_s,
            sampling_interval_ns=sampling_interval_ns,
            n_averages=n_averages,
            align=align,
            align_max_shift_samples=align_max_shift_samples,
            chunk_size=chunk_size or len(voltages),
            timeout_s=timeout_s,
            progress=progress,
            progress_label="scan_voltage",
        )
        traces, t_axis, ok_mask = self._stack_hydrophone_traces(outputs)
        if traces is None:
            raise RuntimeError("No points captured in scan_voltage.")
        traces, units = self._convert_to_pressure(traces, self.frequency * 1e3)
        return ScanResult(
            scan_type="voltage",
            t=t_axis,
            traces=traces,
            coords={"voltage_V": voltages[ok_mask] if not ok_mask.all() else voltages},
            hydrophone_channel=self.hydrophone_channel,
            chunk_size=chunk_size or len(voltages),
            timings=_collect_timings(timings),
            units=units,
            metadata={
                "frequency_kHz": float(self.frequency),
                "captured_mask": ok_mask,
                "hydrophone_position_mm": self.hydrophone_position.copy(),
                "n_averages": int(n_averages),
                "align": bool(align),
                "averaging": averaging,
            },
        )

    def _stack_hydrophone_traces(self, outputs):
        """Pack the hydrophone channel from ``run_rapid_sweep`` outputs.

        Returns:
            ``(traces, t_axis, mask)``. ``traces`` has shape ``(n_ok,
            samples)``, ``t_axis`` is the common time axis in µs, and
            ``mask`` is a bool array over the input ``outputs``. All
            three are ``None`` if nothing was captured.
        """
        hydro = self.hydrophone_channel
        mask = np.array([o is not None for o in outputs], dtype=bool)
        if not mask.any():
            return None, None, None
        good = [o for o in outputs if o is not None]
        t_axis = good[0]["time"]
        traces = np.stack([o[hydro] for o in good], axis=0)
        return traces, t_axis, mask

    def get_peak_voltage(self, x, y, z,
                         time_start_s=-30e-6,
                         time_stop_s=90e-6,
                         sampling_interval_ns=100):
        """
        Sets the focus to the given coordinates and returns the peak-to-peak
        voltage from the hydrophone channel.
        """
        self.set_focus(x, y, z)
        data = self.run_capture(
            time_start_s=time_start_s,
            time_stop_s=time_stop_s,
            sampling_interval_ns=sampling_interval_ns,
        )
        signal = data[self.hydrophone_channel]
        peak_to_peak = np.max(signal) - np.min(signal)
        return peak_to_peak

    def measure_pressure(self, x, y, z, *,
                         time_start_s=-14e-6,
                         time_stop_s=86e-6,
                         sampling_interval_ns=100,
                         timeout_s=2.0,
                         n_averages=1,
                         align=True,
                         align_max_shift_samples=None):
        """Fire one (or several) pulses at ``(x, y, z)`` and return the trace + RMS.

        Steers to the focus, captures the hydrophone trace, and
        (if a hydrophone calibration is attached) converts it from mV
        to Pa via the single-frequency lookup at ``self.frequency``.

        When ``n_averages > 1`` the hydrophone trace is captured that
        many times back-to-back, optionally aligned by
        cross-correlation (:func:`~openlifu_verification.pulse_align.align_pulse_traces`)
        and coherently averaged before RMS/vpp are computed \u2014 giving
        a much lower-noise measurement at the cost of ``n_averages``\u00d7
        the wall-time.

        Args:
            x, y, z: Focus in mm.
            time_start_s, time_stop_s, sampling_interval_ns: Capture
                window.
            timeout_s: Max wait for the scope trigger.
            n_averages: Number of pulses to fire and coherently
                average at this point. Defaults to ``1``.
            align: If ``True`` (default), cross-correlate repeats
                against the first repeat before averaging.
            align_max_shift_samples: Optional cap on the alignment
                lag search (samples).

        Returns:
            Dict with:

            - ``t``: 1-D time axis (\u00b5s), zero at emission.
            - ``trace``: 1-D signal, in ``units``. When
              ``n_averages > 1``, the coherently averaged trace.
            - ``rms``: scalar RMS over the whole window.
            - ``vpp``: scalar peak-to-peak amplitude.
            - ``units``: ``"Pa"`` if a hydrophone is attached,
              otherwise ``"mV"``.
            - ``n_averages``: number of repeats actually captured
              (may be less than requested if the scope timed out on
              some repeats).

            Or ``None`` if every scope capture timed out.
        """
        n_averages = int(n_averages)
        if n_averages < 1:
            raise ValueError("n_averages must be >= 1")

        self.set_focus(x, y, z)
        traces_mv: list[np.ndarray] = []
        t_axis = None
        for _ in range(n_averages):
            data = self.run_capture(
                time_start_s=time_start_s,
                time_stop_s=time_stop_s,
                sampling_interval_ns=sampling_interval_ns,
                timeout_s=timeout_s,
            )
            if data is None:
                continue
            traces_mv.append(np.asarray(data[self.hydrophone_channel], dtype=float))
            if t_axis is None:
                t_axis = data["time"]

        if not traces_mv:
            return None

        stack = np.stack(traces_mv, axis=0)
        if stack.shape[0] > 1 and align:
            aligned, _lags = align_pulse_traces(
                stack,
                dt_s=float(sampling_interval_ns) * 1e-9,
                max_shift_samples=align_max_shift_samples,
            )
            trace_mv = aligned.mean(axis=0)
        else:
            trace_mv = stack.mean(axis=0)

        if self.hydrophone is not None and self.use_calibration:
            trace = np.asarray(
                self.hydrophone.mv_to_pa(trace_mv, self.frequency * 1e3),
                dtype=float,
            )
            units = "Pa"
        else:
            trace = trace_mv
            units = "mV"
        return {
            "t": t_axis,
            "trace": trace,
            "rms": float(np.sqrt(np.mean(trace ** 2))),
            "vpp": float(np.max(trace) - np.min(trace)),
            "units": units,
            "n_averages": stack.shape[0],
        }

    # ------------------------------------------------------------------
    # Hydrophone calibration + position
    # ------------------------------------------------------------------
    def attach_hydrophone(self, hydrophone):
        """Attach (or replace) the hydrophone calibration.

        Once attached, subsequent scans convert traces from voltage
        (mV) to pressure (Pa) using the calibrated V/Pa sensitivity at
        the pulse frequency. Pass ``None`` to detach and go back to
        raw mV.

        Args:
            hydrophone: A :class:`Hydrophone` instance, a path to a
                ``.txt`` calibration file, or ``None``.
        """
        if hydrophone is None:
            self.hydrophone = None
            return None
        if isinstance(hydrophone, (str, Path)):
            hydrophone = Hydrophone(hydrophone)
        self.hydrophone = hydrophone
        return hydrophone

    def save_calibration(self, path=None):
        """Persist hydrophone state (position + last-used ID) to JSON.

        Writes both ``hydrophone_position_mm`` and, when a
        :class:`Hydrophone` is attached, ``hydrophone_id`` so the next
        run can auto-instantiate the same device.

        Args:
            path: Destination path; defaults to
                ``self.calibration_path``.

        Returns:
            The path written.
        """
        if path is None:
            path = self.calibration_path
        if path is None:
            raise ValueError("No calibration_path configured.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "hydrophone_position_mm": self.hydrophone_position.tolist(),
        }
        hydro_id = self._current_hydrophone_id()
        if hydro_id:
            payload["hydrophone_id"] = hydro_id
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info(
            "Saved hydrophone state to %s (position=%s, id=%r)",
            path, self.hydrophone_position.tolist(), hydro_id or "",
        )
        return path

    def load_calibration(self, path=None):
        """Load hydrophone state (position + optional ID) from JSON.

        If the file carries a ``hydrophone_id`` and no hydrophone is
        currently attached, this method attempts to auto-instantiate
        a :class:`Hydrophone` from that ID and assigns it to
        ``self.hydrophone`` so subsequent scans convert traces from mV
        to Pa without the caller having to pass ``--hydrophone`` again.

        Args:
            path: Source path; defaults to ``self.calibration_path``.

        Returns:
            The loaded 3-vector position in mm.
        """
        if path is None:
            path = self.calibration_path
        if path is None:
            raise ValueError("No calibration_path configured.")
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        pos = np.array(data["hydrophone_position_mm"], dtype=float).reshape(3)
        self.hydrophone_position = pos
        hydro_id = data.get("hydrophone_id")
        if hydro_id and self.hydrophone is None:
            try:
                self.hydrophone = Hydrophone(str(hydro_id))
            except Exception as e:
                logger.warning(
                    "Could not auto-attach hydrophone %r from %s: %s",
                    hydro_id, path, e,
                )
        return pos

    def _current_hydrophone_id(self) -> str:
        """Best-effort hydrophone ID for the currently-attached device."""
        if self.hydrophone is None:
            return ""
        try:
            return str(self.hydrophone.metadata.get("HYD_SN", ""))
        except Exception:
            return ""

    def find_peak(self, *,
                  x0=None, y0=None, z=None,
                  method="grid",
                  # --- grid_walk_search parameters ---
                  grid_step=0.2,
                  max_evaluations=50,
                  fit_window=1,
                  # --- gradient_search parameters ---
                  initial_step=0.25,
                  tol=0.02,
                  max_iter=40,
                  hysteresis=0.005,
                  probe_scale=0.5,
                  min_line_step_scale=0.05,
                  min_step=0.2,
                  max_polish_iter=6,
                  rotate_basis=True,
                  # --- shared ---
                  time_start_s=-14e-6,
                  time_stop_s=106e-6,
                  sampling_interval_ns=100,
                  n_averages=1,
                  align=True,
                  plot=False,
                  store=True,
                  save=False,
                  keep_plot_open=True,
                  pause=False):
        """Locate the (x, y) hydrophone peak.

        Two algorithms are available via the ``method`` argument:

        * ``method="grid"`` (default) uses
          :func:`~openlifu_verification.search.grid_walk_search`.
          Fixed axis-aligned grid at spacing ``grid_step``, cached
          samples (never re-measure a node), cardinal-then-diagonal
          walk until the current-best node is bracketed on all 8
          sides, then a least-squares 2-D paraboloid fit over the
          ``fit_window`` neighborhood returns the vertex as the
          centered peak. Best for typical use: minimal wasted
          probes, robust to per-shot noise, no basis rotation.

        * ``method="gradient"`` uses the older
          :func:`~openlifu_verification.search.gradient_search`
          (direct search + 3-point subsample refinement, with
          basis rotation and step shrinking down to ``min_step``).
          Kept for backward compatibility.

        Starts at the current ``hydrophone_position`` unless
        ``x0``/``y0``/``z`` are passed. When ``store=True`` (default),
        the found ``(x, y, z)`` is written to
        ``self.hydrophone_position`` in place, so any subsequent
        relative scan will be centered on the empirical peak.

        Args:
            x0, y0, z: Starting focus (mm). Default to the current
                ``hydrophone_position``.
            method: ``"grid"`` (default) or ``"gradient"``. Selects
                the underlying search algorithm.

            grid_step: [``grid``] Grid spacing (mm). This is the
                spatial scale over which we require the pressure
                field to roll off measurably \u2014 pick it a couple
                times bigger than the per-shot RMS noise "wobble"
                divided by the local slope. Default 0.2 mm
                (200 \u00b5m).
            max_evaluations: [``grid``] Cap on total new
                measurements. Default 50.
            fit_window: [``grid``] Radius (grid nodes) around the
                converged best used for the paraboloid fit.
                ``1`` \u2192 3\u00d73, ``2`` \u2192 5\u00d75. Default 1.

            initial_step: [``gradient``] Initial trial step (mm).
            tol: [``gradient``] Convergence tolerance (mm).
            max_iter: Maximum iterations (both methods).
            hysteresis: [``gradient``] Retained for signature
                compat; currently unused.
            probe_scale: [``gradient``] Probe distance as a
                fraction of current step.
            min_line_step_scale: [``gradient``] Retained; unused.
            min_step: [``gradient``] Minimum probe spacing
                (mm) below which the grid stops shrinking.
                Default 0.2 mm.
            max_polish_iter: [``gradient``] Cap on
                symmetry-polish iterations at ``min_step``.
                Default 6.
            rotate_basis: [``gradient``] Rotate probe basis
                along accepted shifts. Default ``True``.

            time_start_s, time_stop_s, sampling_interval_ns:
                Capture window used at every point.
            n_averages: Number of pulses to fire and coherently
                average per probe. Default 1.
            align: Cross-correlate repeats before averaging.
                Default ``True``.
            plot: Live matplotlib figure. Default ``False``.
            store: Update ``self.hydrophone_position`` with the
                located ``(x, y, z)``. Default ``True``.
            save: Also call :meth:`save_calibration`. Default
                ``False``.
            keep_plot_open: Leave the figure open after
                convergence. Default ``True``.
            pause: Block on ``input()`` at every iteration
                boundary. Default ``False``.

        Returns:
            ``(x, y)`` \u2014 the located (centered) peak in mm.
        """
        from . import search

        if method not in ("grid", "gradient"):
            raise ValueError(
                f"find_peak method must be 'grid' or 'gradient', "
                f"got {method!r}"
            )

        if z is None:
            z = float(self.hydrophone_position[2])
        if x0 is None:
            x0 = float(self.hydrophone_position[0])
        if y0 is None:
            y0 = float(self.hydrophone_position[1])

        handles = None
        on_progress = None
        if plot:
            handles = search.make_live_figure()
            handles["ax_scatter"].set_title(
                f"find_peak @ z={z:.2f} mm  (color = RMS)"
            )

            def on_progress(**kw):
                search.update_live_figure(handles, **kw)
                if pause and kw.get("iter_end") and not kw.get("done"):
                    try:
                        input(
                            f"[iter {kw.get('iteration')} done] "
                            "press Enter for next iteration "
                            "(Ctrl-C to abort)... "
                        )
                    except EOFError:
                        pass
        elif pause:
            def on_progress(**kw):
                if kw.get("iter_end") and not kw.get("done"):
                    try:
                        input(
                            f"[iter {kw.get('iteration')} done] "
                            "press Enter for next iteration "
                            "(Ctrl-C to abort)... "
                        )
                    except EOFError:
                        pass

        def measure_fn(x, y):
            return self.measure_pressure(
                x, y, z,
                time_start_s=time_start_s,
                time_stop_s=time_stop_s,
                sampling_interval_ns=sampling_interval_ns,
                n_averages=n_averages,
                align=align,
            )

        if method == "grid":
            logger.info(
                "find_peak (grid): starting at (%.3f, %.3f, %.3f) mm  "
                "grid_step=%.3f mm  max_evaluations=%d  "
                "fit_window=%d  n_averages=%d",
                x0, y0, z, grid_step, max_evaluations,
                fit_window, n_averages,
            )
            result = search.grid_walk_search(
                measure_fn,
                x0=x0, y0=y0,
                step=grid_step,
                max_evaluations=max_evaluations,
                max_iter=max_iter,
                fit_window=fit_window,
                on_progress=on_progress,
            )
        else:  # method == "gradient"
            logger.info(
                "find_peak (gradient): starting at (%.3f, %.3f, %.3f) mm  "
                "initial_step=%.3f mm  tol=%.3f mm  n_averages=%d",
                x0, y0, z, initial_step, tol, n_averages,
            )
            result = search.gradient_search(
                measure_fn,
                x0=x0, y0=y0,
                initial_step=initial_step,
                tol=tol,
                max_iter=max_iter,
                hysteresis=hysteresis,
                probe_scale=probe_scale,
                min_line_step_scale=min_line_step_scale,
                min_step=min_step,
                max_polish_iter=max_polish_iter,
                rotate_basis=rotate_basis,
                on_progress=on_progress,
            )
        # Use the *symmetry-center* the search settled on, not the
        # highest single RMS sample. On a noisy top the max sample
        # is a lucky noise spike; the center is what we actually
        # asked the algorithm to find.
        x, y = result["center_x"], result["center_y"]
        logger.info(
            "find_peak: %s at center (%.4f, %.4f, %.4f) mm  "
            "RMS=%.4g %s  (global max sample=%.4g %s at "
            "(%.4f, %.4f))  (%d iterations, %d evaluations)",
            "converged" if result["converged"] else "hit max_iter",
            x, y, z, result["center_rms"], result["units"],
            result["best_rms"], result["units"],
            result["best_x"], result["best_y"],
            result["iterations"], result["evaluations"],
        )

        if store:
            self.hydrophone_position = np.array([x, y, z], dtype=float)
        if save:
            if result["converged"]:
                self.save_calibration()
            else:
                logger.warning(
                    "find_peak did not converge (step > tol); "
                    "skipping save_calibration.",
                )

        if plot and keep_plot_open:
            import matplotlib.pyplot as plt
            plt.ioff()
            plt.show()

        return x, y

    def find_xy_peak(self, *,
                     z=None,
                     x0=None,
                     y0=None,
                     step_size=0.5,
                     iterations=10,
                     learning_rate=0.1,
                     time_start_s=-30e-6,
                     time_stop_s=90e-6,
                     sampling_interval_ns=100,
                     store=True,
                     save=False):
        """Locate the true (x, y) peak of the hydrophone via gradient ascent.

        Uses finite-difference gradients of the peak-to-peak
        hydrophone voltage at each iteration. Starts from
        ``(x0, y0, z)`` — defaulting to the current
        ``self.hydrophone_position`` — so calling this after a rough
        first-time setup refines the stored position.

        Args:
            z: Depth (mm) to search at. ``None`` uses
                ``hydrophone_position[2]``.
            x0, y0: Starting (x, y) in mm. ``None`` uses the current
                ``hydrophone_position``.
            step_size: Finite-difference step (mm) used to estimate
                the gradient.
            iterations: Number of ascent steps.
            learning_rate: Ascent step size (mm per unit gradient).
            time_start_s, time_stop_s, sampling_interval_ns: Capture
                window used for each measurement.
            store: If ``True`` (default), update
                ``self.hydrophone_position`` with the found (x, y, z).
            save: If ``True``, also write the updated position to
                ``self.calibration_path`` (call
                :meth:`save_calibration`).

        Returns:
            ``(x, y, z, vpp)`` — the located peak and its Vpp (mV).
        """
        if z is None:
            z = float(self.hydrophone_position[2])
        if x0 is None:
            x0 = float(self.hydrophone_position[0])
        if y0 is None:
            y0 = float(self.hydrophone_position[1])
        x, y = float(x0), float(y0)

        v_current = self.get_peak_voltage(
            x, y, z,
            time_start_s=time_start_s, time_stop_s=time_stop_s,
            sampling_interval_ns=sampling_interval_ns,
        )
        logger.info(
            "find_xy_peak start: x=%.3f y=%.3f z=%.3f Vpp=%.3f mV",
            x, y, z, v_current,
        )
        for i in range(iterations):
            v_x = self.get_peak_voltage(
                x + step_size, y, z,
                time_start_s=time_start_s, time_stop_s=time_stop_s,
                sampling_interval_ns=sampling_interval_ns,
            )
            v_y = self.get_peak_voltage(
                x, y + step_size, z,
                time_start_s=time_start_s, time_stop_s=time_stop_s,
                sampling_interval_ns=sampling_interval_ns,
            )
            grad_x = (v_x - v_current) / step_size
            grad_y = (v_y - v_current) / step_size
            x += learning_rate * grad_x
            y += learning_rate * grad_y
            v_current = self.get_peak_voltage(
                x, y, z,
                time_start_s=time_start_s, time_stop_s=time_stop_s,
                sampling_interval_ns=sampling_interval_ns,
            )
            logger.info(
                "find_xy_peak iter %d/%d: x=%.3f y=%.3f Vpp=%.3f mV "
                "(grad=(%.3f, %.3f))",
                i + 1, iterations, x, y, v_current, grad_x, grad_y,
            )

        if store:
            self.hydrophone_position = np.array([x, y, z], dtype=float)
        if save:
            self.save_calibration()
        return x, y, z, v_current

    def find_peak_by_gradient_ascent(self, x_start, y_start, z,
                                     step_size=0.5, iterations=10,
                                     learning_rate=0.1):
        """Legacy wrapper for :meth:`find_xy_peak`.

        Kept for backward compatibility with older scripts. Does not
        update ``self.hydrophone_position`` (pass ``store=True`` to
        :meth:`find_xy_peak` for the new behavior).
        """
        x, y, _z, _v = self.find_xy_peak(
            z=z, x0=x_start, y0=y_start,
            step_size=step_size, iterations=iterations,
            learning_rate=learning_rate,
            store=False, save=False,
        )
        return x, y

    # ------------------------------------------------------------------
    # Pressure conversion helper (used by scans)
    # ------------------------------------------------------------------
    def _convert_to_pressure(self, traces, frequency_hz):
        """Convert an (N, samples) trace array from mV to Pa.

        Args:
            traces: 2-D array of hydrophone voltage in mV. First axis
                is the sweep index, second is time.
            frequency_hz: Frequency (Hz) at which to look up the V/Pa
                sensitivity. Scalar (single-frequency scan) or 1-D
                array of length ``traces.shape[0]`` (per-point
                frequency, e.g. scan_frequency).

        Returns:
            ``(traces_out, units)`` where ``units`` is ``"Pa"`` when
            the hydrophone is attached and ``self.use_calibration``
            is ``True``. Otherwise ``units`` is ``"mV"`` and
            ``traces`` is returned unchanged.
        """
        if self.hydrophone is None or not self.use_calibration:
            return traces, "mV"
        freq_hz_arr = np.asarray(frequency_hz, dtype=float)
        pa_per_v = np.asarray(
            self.hydrophone.get_frequency_response(freq_hz_arr), dtype=float,
        )
        voltage_v = np.asarray(traces, dtype=float) * 1e-3
        if pa_per_v.ndim == 0:
            return voltage_v * float(pa_per_v), "Pa"
        # Broadcast per-row sensitivities against (N, samples) traces.
        pa_per_v = pa_per_v.reshape((-1,) + (1,) * (voltage_v.ndim - 1))
        return voltage_v * pa_per_v, "Pa"


def _fmt_point(point) -> str:
    """Short string representation of a sweep point for log messages."""
    if isinstance(point, (list, tuple)):
        parts = []
        for v in point:
            try:
                parts.append(f"{float(v):+.2f}")
            except (TypeError, ValueError):
                parts.append(repr(v))
        return "(" + ", ".join(parts) + ")"
    try:
        return f"{float(point):.4g}"
    except (TypeError, ValueError):
        return repr(point)


def _collect_timings(timings):
    """Convert the list-of-dicts from run_rapid_sweep to a dict-of-arrays."""
    if not timings:
        return {}
    keys = ("apply_s", "trigger_s", "arm_s", "xfer_s", "iter_total_s")
    return {k: np.array([t.get(k, np.nan) for t in timings], dtype=float)
            for k in keys}
