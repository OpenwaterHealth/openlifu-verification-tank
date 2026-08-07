import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

from .picoscope import Picoscope
from .qpx600dp import QPX600DP

from openlifu_sdk.io import LIFUInterface
from openlifu_sdk.io.LIFUTXDevice import Tx7332DelayProfile, Tx7332PulseProfile

logger = logging.getLogger(__name__)
PICOSCOPE_RESOLUTION = "15BIT"
SPEED_OF_SOUND = 1500  # m/s in water
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

    def __init__(self,
                 frequency=400,
                 use_picoscope=True,
                 num_modules=1,
                 resolution=PICOSCOPE_RESOLUTION,
                 ext_power_supply=True,
                 voltage_table_selection="dvt",
                 hydrophone_channel="A",
                 trigger_channel="B",
                 hydrophone_range_mv=100,
                 trigger_range_mv=5000,
                 trigger_threshold_mv=1000,
                 trigger_direction="rising"):
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


        except Exception as e:
            logger.error(f"Error during initialization: {e}")
            self.__exit__(None, None, None)
            raise

        return self

    def configure_lifu(self, 
                       frequency_kHz, 
                       voltage, 
                       duration_msec, 
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
            "duration": duration_msec * 1e-3,
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
        logger.info("configure_lifu pulse=%s", pulse)
        logger.info("configure_lifu sequence=%s", sequence)
        logger.info(
            "configure_lifu voltage=%s trigger_mode=%s profile_index=%s profile_increment=%s",
            voltage, trigger_mode, profile_index, profile_increment,
        )
        logger.info(
            "configure_lifu transducer id=%s num_elements=%s module_invert=%s",
            solution["transducer"].get("id"),
            len(solution["transducer"].get("elements", [])),
            solution["transducer"].get("module_invert"),
        )
        logger.info(
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
        logger.info(f"calculating delays for {focus=}")
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
        logger.info("writing registers...")
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

    def set_pulse(self, frequency_kHz, duration_msec):
        pulse_profile = Tx7332PulseProfile(
            profile=1,
            frequency=frequency_kHz*1e3,
            cycles=int(duration_msec * frequency_kHz)
        )
        self.lifu.txdevice.tx_registers.add_pulse_profile(pulse_profile)
        logger.info("writing registers...")
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

    def run_trigger(self, hold_s: float = 1.0):
        """Fire a single TX trigger (no scope involvement) and stop.

        Useful for running the pulse when an external scope application is
        used for capture. HV must already be enabled (call
        ``enable_hv_output(wait=True)`` first).

        Args:
            hold_s: How long to leave sonication running before calling
                ``stop_sonication``. For ``trigger_mode="single"`` the
                pulse train completes on its own, so this just needs to
                cover the pulse-train duration.
        """
        logger.info("Sending Single Trigger (no scope)...")
        # Keep async STATUS frames off during the trigger so they don't race
        # command responses on the shared TX CDC endpoint.
        self.lifu.start_sonication(turn_hv_on=False, wait_for_settle=False, async_mode=False)
        try:
            time.sleep(hold_s)
        finally:
            try:
                self.lifu.stop_sonication(turn_hv_off=False, wait_for_settle=False)
            except Exception as e:
                logger.warning("stop_sonication after trigger raised: %s", e)

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

        The window is specified relative to the scope trigger event:

        - ``time_start_s < 0`` → pre-trigger capture; the scope buffers
          data from ``time_start_s`` up through ``time_stop_s``.
        - ``time_start_s >= 0`` → delayed (advanced-trigger) capture;
          the scope waits ``time_start_s`` after the trigger before it
          starts collecting samples. This lets you skim the front of
          long captures without wasting samples/RAM.

        The scope only supports discrete sampling intervals and integer
        sample counts, so what actually gets used may differ from what
        was requested. The returned data dict includes
        ``sampling_interval_ns``, ``time_start_s``, and ``time_stop_s``
        so you know what was really applied.

        This method assumes the trigger has already been configured via
        ``self.scope.set_trigger(channel=self.trigger_channel, ...)``.
        The ``delay_samples`` field of that trigger is re-applied here
        based on ``time_start_s``.

        Example::

            # samples every 100 ns for 100 us, starting 10 us before trigger
            data = ver.run_capture(
                time_start_s=-10e-6,
                time_stop_s=100e-6,
                sampling_interval_ns=100,
            )

        Args:
            time_start_s: Start of the capture window relative to trigger.
            time_stop_s: End of the capture window relative to trigger.
            sampling_interval_ns: Requested time between samples in ns.
            timeout_s: Max time to wait for the scope trigger to fire.

        Returns:
            The scope data dict with:
              - ``time``: sample times relative to the trigger (ns).
              - one array per enabled channel (mV).
              - ``sampling_interval_ns``, ``time_start_s``,
                ``time_stop_s``: actual applied values.
            Or ``None`` if the scope's trigger timed out.
        """
        if not self.scope:
            raise ValueError("No Picoscope Connected")
        plan = self.scope.plan_capture(
            sampling_interval_ns=sampling_interval_ns,
            time_start_s=time_start_s,
            time_stop_s=time_stop_s,
        )
        logger.info(
            "run_capture: requested %.1f ns / start %.3f us / stop %.3f us; "
            "actual %.3f ns / start %.3f us / stop %.3f us "
            "(timebase=%d, pre=%d, post=%d, delay=%d)",
            sampling_interval_ns, time_start_s * 1e6, time_stop_s * 1e6,
            plan["sampling_interval_ns"],
            plan["time_start_s"] * 1e6, plan["time_stop_s"] * 1e6,
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
            # Shift the scope's zero-based time axis so t=0 is the trigger.
            interval_ns = plan["sampling_interval_ns"]
            offset_ns = plan["time_start_s"] * 1e9
            result["time"] = result["time"] + offset_ns
            result["sampling_interval_ns"] = interval_ns
            result["time_start_s"] = plan["time_start_s"]
            result["time_stop_s"] = plan["time_stop_s"]
        return result

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
        logger.info("Sending Single Trigger...")
        self.scope.run_block(pre_trigger_samples=pre_trigger_samples, post_trigger_samples=post_trigger_samples, timebase=timebase)
        # Give the scope a moment to fully arm before firing the TX pulse.
        time.sleep(0.05)
        # Mirror the test app: fire via LIFUInterface.start_sonication rather
        # than calling txdevice.start_trigger directly. HV is already on and
        # settled (enable_hv_output(wait=True) is called earlier), so
        # skip the SDK's HV re-check/settle steps. Keep async STATUS off so
        # STATUS frames don't race the start_trigger command response.
        self.lifu.start_sonication(turn_hv_on=False, wait_for_settle=False, async_mode=False)
        try:
            if not self.scope.wait_ready(timeout_s=timeout_s):
                logger.warning("Scope trigger timed out after %.3f s; no pulse captured.", timeout_s)
                return None
            return self.scope.get_data(pre_trigger_samples+post_trigger_samples, timebase)
        finally:
            # Match the test app's Start/Stop pairing: always stop sonication
            # after the capture completes (or times out). Leave HV energized
            # so subsequent captures don't need to re-settle.
            try:
                self.lifu.stop_sonication(turn_hv_off=False, wait_for_settle=False)
            except Exception as e:
                logger.warning("stop_sonication after capture raised: %s", e)

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

    def get_peak_voltage(self, x, y, z,
                         time_start_s=-10e-6,
                         time_stop_s=200e-6,
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

    def find_peak_by_gradient_ascent(self, x_start, y_start, z, step_size=0.5, iterations=10, learning_rate=0.1):
        """
        Finds the x-y coordinates that produce the maximum peak voltage using gradient ascent.
        """
        x = x_start
        y = y_start

        for i in range(iterations):
            # Calculate the gradient
            v_current = self.get_peak_voltage(x, y, z)
            v_x = self.get_peak_voltage(x + step_size, y, z)
            v_y = self.get_peak_voltage(x, y + step_size, z)

            grad_x = (v_x - v_current) / step_size
            grad_y = (v_y - v_current) / step_size

            # Update the coordinates
            x += learning_rate * grad_x
            y += learning_rate * grad_y

            logger.info(f"Iteration {i+1}/{iterations}: x={x:.2f}, y={y:.2f}, Vp-p={v_current:.2f}")

        return x, y
