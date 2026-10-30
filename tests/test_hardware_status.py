"""probe_hardware: presence from COM port VID/PIDs and PicoScope enumeration."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from openlifu_sdk.io.LIFUConfig import OW_CONSOLE_PID, OW_TRANSMITTER_PID, OW_VID

from openlifu_verification import hardware_status
from openlifu_verification.qpx600dp import QPX600DP


def _ports(*ids):
    return [SimpleNamespace(vid=vid, pid=pid) for vid, pid in ids]


@pytest.mark.parametrize("ext_power, plugged, expected", [
    (False, [(OW_VID, OW_TRANSMITTER_PID), (OW_VID, OW_CONSOLE_PID)],
     {"Scope": True, "Transducer": True, "Console": True}),
    (False, [(OW_VID, OW_CONSOLE_PID), (None, None)],
     {"Scope": True, "Transducer": False, "Console": True}),
    (True, [(OW_VID, OW_TRANSMITTER_PID), (OW_VID, OW_CONSOLE_PID)],
     {"Scope": True, "Transducer": True, "HV supply": False}),
    (True, [(QPX600DP.VID, QPX600DP.PID)],
     {"Scope": True, "Transducer": False, "HV supply": True}),
])
def test_probe_matches_usb_ids(monkeypatch, ext_power, plugged, expected):
    monkeypatch.setattr(hardware_status.serial.tools.list_ports, "comports",
                        lambda: _ports(*plugged))
    monkeypatch.setattr(hardware_status, "picoscope_present", lambda: True)
    assert hardware_status.probe_hardware(
        ext_power_supply=ext_power) == expected


@pytest.mark.parametrize("status, count, present", [
    (0, 1, True), (0, 0, False), (3, 1, False)])
def test_picoscope_present_reads_enumeration(monkeypatch, status, count,
                                             present):
    def _enumerate(count_ref, serials, length_ref):
        count_ref._obj.value = count
        return status

    monkeypatch.setattr(hardware_status.ps, "ps5000aEnumerateUnits",
                        _enumerate)
    assert hardware_status.picoscope_present() is present
