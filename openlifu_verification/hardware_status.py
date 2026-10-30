"""Cheap hardware presence checks for the launcher's status LEDs.

Nothing is opened: the transducer, console and QPX600DP are looked up
by USB VID/PID in the COM port list (the same match their drivers use
to connect), and the PicoScope by driver enumeration. A full probe
takes ~0.2 s, dominated by the PicoScope enumeration.

PicoScope enumeration is not allowed while this process has a scope
open, so stop polling before a run opens the hardware.
"""
from __future__ import annotations

import ctypes

import serial.tools.list_ports
from openlifu_sdk.io.LIFUConfig import OW_CONSOLE_PID, OW_TRANSMITTER_PID, OW_VID
from picosdk.ps5000a import ps5000a as ps

from .qpx600dp import QPX600DP


def picoscope_present() -> bool:
    """Whether a PicoScope 5000A unit is plugged in."""
    count = ctypes.c_int16()
    length = ctypes.c_int16(256)
    serials = ctypes.create_string_buffer(256)
    status = ps.ps5000aEnumerateUnits(ctypes.byref(count), serials,
                                      ctypes.byref(length))
    return status == 0 and count.value > 0


def probe_hardware(*, ext_power_supply: bool = False) -> dict[str, bool]:
    """Presence of each instrument a run needs, keyed by display name.

    The console supplies HV unless ``ext_power_supply``, in which case
    the QPX600DP is checked instead.
    """
    ports = {(p.vid, p.pid) for p in serial.tools.list_ports.comports()}
    status = {
        "Scope": picoscope_present(),
        "Transducer": (OW_VID, OW_TRANSMITTER_PID) in ports,
    }
    if ext_power_supply:
        status["HV supply"] = (QPX600DP.VID, QPX600DP.PID) in ports
    else:
        status["Console"] = (OW_VID, OW_CONSOLE_PID) in ports
    return status
