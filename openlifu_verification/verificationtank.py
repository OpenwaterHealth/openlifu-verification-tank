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
        logger.info("Sending Single Trigger (no scope)...")
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
            time_start_s: Start of the capture window relative to each trigger.
            time_stop_s: End of the capture window relative to each trigger.
            sampling_interval_ns: Requested time between samples in ns.

        Returns:
            The plan dict from ``scope.plan_capture`` plus
            ``n_captures`` and ``samples_per_segment`` fields.
        """
        if not self.scope:
            raise ValueError("No Picoscope Connected")
        plan = self.scope.plan_capture(
            sampling_interval_ns=sampling_interval_ns,
            time_start_s=time_start_s,
            time_stop_s=time_stop_s,
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
        logger.info(
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

        # Shift time axis so t=0 is the trigger, and expose the actual plan.
        interval_ns = plan["sampling_interval_ns"]
        offset_ns = plan["time_start_s"] * 1e9
        result["time"] = result["time"] + offset_ns
        result["sampling_interval_ns"] = interval_ns
        result["time_start_s"] = plan["time_start_s"]
        result["time_stop_s"] = plan["time_stop_s"]
        return result

    def run_rapid_sweep(self,
                        points,
                        apply_point,
                        time_start_s,
                        time_stop_s,
                        sampling_interval_ns,
                        chunk_size=None,
                        timeout_s=None,
                        progress=None):
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
            progress: Optional callable ``progress(point, timing) ->
                None`` invoked after every trigger. ``timing`` is the
                per-point dict described below.

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

            t_xfer_start = time.perf_counter()
            bulk = self.finish_rapid_capture(plan, timeout_s=timeout_s)
            t_xfer = time.perf_counter() - t_xfer_start

            arm_per_pt = t_arm / n_captures
            xfer_per_pt = t_xfer / n_captures
            logger.info(
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
                if progress is not None:
                    progress(point, pt)

        return outputs, timings

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
