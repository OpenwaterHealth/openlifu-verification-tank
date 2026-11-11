"""Simulated instruments for ``--dry-run``.

A dry run must not touch real hardware, so everything the launcher and
the run would read from the instruments comes from here instead:

* :class:`SimulatedHardware` holds which simulated instruments are
  plugged in. Its :meth:`~SimulatedHardware.probe` stands in for
  :func:`hardware_status.probe_hardware` (same keys, no PicoSDK or
  serial-port enumeration), and the launcher toggles presence to test
  Start blocking and console re-detection.
* :class:`SimulatedLifu` is the SDK's
  :class:`~openlifu_sdk.ui.simulated_interface.SimulatedLIFUInterface`
  with a console that has an S/N in its user config, usable as a
  context manager like ``LIFUInterface``. It is passed to
  :func:`device_info.read_console_identity` and is the dry-run tank's
  ``lifu``, so the console S/N and hardware ID the launcher detects are
  the ones :class:`DeviceInfo` puts in the report.

The SDK's simulation builds Qt objects on whichever Qt binding the
application imported first, so import this module only after PySide6
(the package's ``__init__`` doesn't import it).
"""
from __future__ import annotations

import threading
from typing import Optional

from openlifu_sdk.io.LIFUUserConfig import LifuUserConfig
from openlifu_sdk.ui.simulated_interface import (
    SimulatedLIFUHVController,
    SimulatedLIFUInterface,
    SimulatedLIFUTxDevice,
)

#: S/N in the simulated console's user config. Its hardware ID is the
#: SDK simulation's (see :func:`simulated_console_identity`).
SIMULATED_CONSOLE_SN = "WL134Q00000"

#: Names :meth:`SimulatedHardware.probe` reports, as
#: :func:`hardware_status.probe_hardware` does.
DEVICE_NAMES = ("Scope", "Transducer", "Console", "HV supply")


class _SimulatedConsole(SimulatedLIFUHVController):
    """The SDK's simulated console plus a user config holding its S/N
    (the SDK's simulated console has no user config)."""

    def read_config(self) -> LifuUserConfig:
        return LifuUserConfig(json_data={"sn": SIMULATED_CONSOLE_SN})


class SimulatedLifu(SimulatedLIFUInterface):
    """:class:`SimulatedLIFUInterface` with an S/N in the console's user
    config, usable as ``with SimulatedLifu() as lifu:`` like
    ``LIFUInterface``.

    ``tx_connected`` / ``console_connected`` set what
    :meth:`is_device_connected` reports. ``fw_version``, if given,
    replaces the SDK's placeholder version on both devices (the
    firmware acceptance gate needs a real version number)."""

    def __init__(self, *, tx_connected: bool = True,
                 console_connected: bool = True,
                 fw_version: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        if tx_connected:
            self.txdevice.emit_connected()
        if console_connected:
            self.hvcontroller.emit_connected()
        if fw_version is not None:
            self.txdevice.set_version(fw_version)
            self.hvcontroller.set_version(fw_version)

    def _create_devices(self, num_modules, preset_flash):
        return (SimulatedLIFUTxDevice(num_modules=num_modules,
                                      preset_flash=preset_flash),
                _SimulatedConsole())

    def __enter__(self) -> "SimulatedLifu":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def simulated_console_identity() -> tuple[str, str]:
    """``(sn, base58 hwid)`` of the simulated console."""
    with SimulatedLifu() as lifu:
        return SIMULATED_CONSOLE_SN, lifu.hvcontroller.get_hardware_id(
            base58=True)


class SimulatedHardware:
    """Which simulated instruments are plugged in (all, initially).

    Thread-safe: the launcher's poller and console reader run on
    background threads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._present = dict.fromkeys(DEVICE_NAMES, True)

    def is_present(self, name: str) -> bool:
        with self._lock:
            return self._present[name]

    def set_present(self, name: str, present: bool) -> None:
        with self._lock:
            if name not in self._present:
                raise KeyError(name)
            self._present[name] = bool(present)

    def toggle(self, name: str) -> bool:
        """Plug ``name`` in or out; returns whether it is now present."""
        with self._lock:
            self._present[name] = not self._present[name]
            return self._present[name]

    def probe(self, *, ext_power_supply: bool = False) -> dict[str, bool]:
        """Stand-in for :func:`hardware_status.probe_hardware`."""
        names = ("Scope", "Transducer",
                 "HV supply" if ext_power_supply else "Console")
        with self._lock:
            return {name: self._present[name] for name in names}

    def open_lifu(self) -> SimulatedLifu:
        """A :class:`SimulatedLifu` whose devices are connected as
        plugged in here; pass it as ``open_interface`` to
        :func:`device_info.read_console_identity`."""
        return SimulatedLifu(tx_connected=self.is_present("Transducer"),
                             console_connected=self.is_present("Console"))
