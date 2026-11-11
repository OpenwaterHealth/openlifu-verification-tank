"""``_read_hwid`` returns ``(raw hex, base58 of the full 12-byte UID)``
through the SDK's ``get_hardware_id``; only the transport is mocked."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from openlifu_sdk.io.LIFUHVController import HVController
from openlifu_sdk.io.LIFUTXDevice import TxDevice
from openlifu_sdk.util.hwid import encode_hwid

from openlifu_verification.device_info import (
    DeviceInfo,
    _read_config_sn,
    _read_hwid,
    read_console_identity,
)
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


def test_dry_run_device_info_comes_from_the_sdk_simulation():
    info = DeviceInfo.collect(DryRunTank(rng_seed=0))
    # The SDK simulation's MCU UIDs; FW versions the acceptance gate takes.
    assert (info.txm_hwid_hex, info.txm_hwid) == (
        "a0a1a2a3a4a5a6a7b0b1b2b3", "42p8mJivqN6tpZqe2")
    assert (info.console_hwid_hex, info.console_hwid) == (
        "c0c1c2c3c4c5c6c7d0d1d2d3", "4dyWcpSyLJHurUomU")
    assert (info.txm_fw_version, info.console_fw_version) == ("2.0.8", "1.2.7")


def test_simulated_lifu_firmware_versions():
    from openlifu_verification.simulated_hardware import SimulatedLifu
    with SimulatedLifu(txm_fw_version="2.0.8",
                       console_fw_version="1.2.7") as lifu:
        assert lifu.txdevice.get_version() == "2.0.8"
        assert lifu.hvcontroller.get_version() == "1.2.7"
    with SimulatedLifu(fw_version="3.0.0", console_fw_version="") as lifu:
        assert lifu.txdevice.get_version() == "3.0.0"
        assert lifu.hvcontroller.get_version() == ""


def test_read_console_identity_from_simulated_console():
    from openlifu_verification.simulated_hardware import (
        SIMULATED_CONSOLE_SN,
        SimulatedLifu,
    )
    assert read_console_identity(SimulatedLifu) == (
        SIMULATED_CONSOLE_SN, "4dyWcpSyLJHurUomU")
    with pytest.raises(RuntimeError, match="not connected"):
        read_console_identity(
            lambda: SimulatedLifu(console_connected=False))


class _ConfigDevice:
    def __init__(self, config):
        self._config = config

    def read_config(self):
        if isinstance(self._config, Exception):
            raise self._config
        from openlifu_sdk.io.LIFUUserConfig import LifuUserConfig
        return LifuUserConfig(json_data=self._config)


@pytest.mark.parametrize("config, sn", [
    ({"sn": " WL134Q00007 "}, "WL134Q00007"),
    ({"hwid": "abc"}, ""),
    ({"sn": 7}, ""),
    (RuntimeError("timeout"), ""),
])
def test_read_config_sn(config, sn):
    assert _read_config_sn(_ConfigDevice(config), "console") == sn
