"""``_read_hwid`` returns ``(raw hex, base58 of the full 12-byte UID)``
through the SDK's ``get_hardware_id``; only the transport is mocked."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from openlifu_sdk.io.LIFUHVController import HVController
from openlifu_sdk.io.LIFUTXDevice import TxDevice
from openlifu_sdk.util.hwid import encode_hwid

from openlifu_verification.device_info import DeviceInfo, _read_hwid
from openlifu_verification.dry_run import DryRunTank

UID = bytes(range(12))
UID_HEX = "000102030405060708090a0b"
UID_B58 = "1FVk6iLh9oT6QH8"        # base58 of all 12 bytes
UID_B58_LEGACY = "17bWpTW"         # base58 of the first 6 bytes (old form)


def _with_uid(dev):
    """Answer the SDK's HWID command with ``UID`` instead of a UART."""
    def send_checked(command, addr=0, **_kw):
        return SimpleNamespace(data=bytearray(UID), data_len=len(UID))
    dev.send_checked = send_checked
    return dev


@pytest.mark.parametrize("cls", [TxDevice, HVController])
def test_read_hwid_returns_hex_and_full_base58(cls):
    dev = _with_uid(cls())
    assert _read_hwid(dev, 0, "dev") == (UID_HEX, UID_B58)
    # The SDK still offers the legacy form; the report must not carry it.
    assert dev.get_hardware_id(0, base58=True, legacy_6byte=True) \
        == UID_B58_LEGACY


class _NoModuleDevice:
    """A device whose ``get_hardware_id`` takes no ``module``, like the
    SDK's simulated console."""

    def get_hardware_id(self, raw_hex=False, base58=False,
                        legacy_6byte=False):
        return encode_hwid(UID, raw_hex, base58, legacy_6byte)


def test_read_hwid_falls_back_when_module_not_accepted():
    assert _read_hwid(_NoModuleDevice(), 0, "console") == (UID_HEX, UID_B58)


def test_read_hwid_unreadable_device_gives_empty_strings():
    class _Broken:
        def get_hardware_id(self, *_a, **_k):
            raise RuntimeError("not connected")
    assert _read_hwid(_Broken(), 0, "TXM") == ("", "")


def test_dry_run_device_info_carries_full_form():
    info = DeviceInfo.collect(DryRunTank(rng_seed=0))
    assert (info.txm_hwid_hex, info.txm_hwid) == (
        "d00000000000000000000001", "4vfAkNk3axRSP1x4Q")
    assert (info.console_hwid_hex, info.console_hwid) == (
        "d00000000000000000000002", "4vfAkNk3axRSP1x4R")
