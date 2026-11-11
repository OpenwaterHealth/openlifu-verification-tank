"""Auto-extracted device metadata for the characterization report.

Reads what the SDK can tell us about the connected LIFU interface
(SDK version, TXM/console HW ID + FW version) and packs it into
:class:`DeviceInfo`. HW IDs are exposed in both hex and base58 forms
(the base58 form is the full 12-byte UID, as calibration records carry).
"""
from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class DeviceInfo:
    """Auto-extractable device metadata.

    All fields default to empty strings so a partial read (e.g. HV
    controller disconnected) still produces a serializable object.
    """
    test_date: str = ""
    sdk_version: str = ""
    txm_hwid_hex: str = ""
    txm_hwid: str = ""              # base58-encoded, matches app schema
    txm_fw_version: str = ""
    console_hwid_hex: str = ""
    console_hwid: str = ""
    console_fw_version: str = ""
    hydrophone_model: str = ""      # e.g. "Onda HNR0500"
    hydrophone_cal_date: str = ""   # verbatim from cal file, e.g. "19-Dec-2022"
    hydrophone_cal_file: str = ""   # basename only, no path
    picoscope_variant: str = ""     # e.g. "5244D"
    picoscope_sn: str = ""          # "batch/serial" string from the driver
    picoscope_cal_date: str = ""    # factory cal date, driver-formatted

    @classmethod
    def collect(cls, ver, *, module: int = 0) -> "DeviceInfo":
        """Pull every field we can from ``ver`` (a :class:`VerificationTank`).

        Failures on individual reads are logged and left empty rather
        than raising -- the report can still be filled with the
        remaining values.
        """
        info = cls(test_date=datetime.date.today().isoformat())

        # SDK version.
        try:
            from openlifu_sdk.io import LIFUInterface
            info.sdk_version = LIFUInterface.get_sdk_version()
        except Exception as e:
            logger.warning("Could not read SDK version: %s", e)

        # TXM.
        tx = getattr(getattr(ver, "lifu", None), "txdevice", None)
        if tx is not None:
            info.txm_hwid_hex, info.txm_hwid = _read_hwid(tx, module, "TXM")
            info.txm_fw_version = _read_version(tx, module, "TXM")

        # Console (HV controller).
        hv = getattr(getattr(ver, "lifu", None), "hvcontroller", None)
        if hv is not None:
            info.console_hwid_hex, info.console_hwid = _read_hwid(hv, module, "console")
            info.console_fw_version = _read_version(hv, module, "console")

        # Hydrophone calibration metadata (inferred from the loaded
        # calibration file). Silent no-op for stub hydrophones (dry
        # run) or if the file couldn't be parsed.
        hyd = getattr(ver, "hydrophone", None)
        if hyd is not None:
            try:
                meta = getattr(hyd, "metadata", None) or {}
                mfg = str(meta.get("HYD_MFG", "")).strip()
                model = str(meta.get("HYD_MODEL", "")).strip()
                aperture = str(meta.get("HYD_APERTURE_NOM_UM", "")).strip()
                # Build "Onda HNR0500" if we have the pieces, else fall
                # back to whatever ``hyd.model`` reports.
                if model and aperture:
                    model_full = f"{model}{aperture}"
                else:
                    model_full = str(getattr(hyd, "model", "") or "")
                info.hydrophone_model = (
                    f"{mfg} {model_full}".strip() if mfg else model_full
                )
                info.hydrophone_cal_date = str(meta.get("Calibration_DATE", "")).strip()
                cal_path = getattr(hyd, "calibration_file_path", None)
                if cal_path is not None:
                    from pathlib import Path
                    info.hydrophone_cal_file = Path(str(cal_path)).name
            except Exception as e:
                logger.warning("Could not read hydrophone cal metadata: %s", e)

        # PicoScope identification + factory cal metadata.
        scope = getattr(ver, "scope", None)
        get_unit_info = getattr(scope, "get_unit_info", None)
        if callable(get_unit_info):
            try:
                psi = get_unit_info()
                info.picoscope_variant = psi.get("variant", "")
                info.picoscope_sn = psi.get("batch_and_serial", "")
                info.picoscope_cal_date = psi.get("cal_date", "")
            except Exception as e:
                logger.warning("Could not read PicoScope unit info: %s", e)

        return info


def _read_hwid(dev, module: int, label: str) -> tuple[str, str]:
    """Return ``(hex_hwid, base58_hwid)`` for a device, or empty strings.

    ``base58_hwid`` is the SDK's base58 form of the full 12-byte MCU
    UID -- the form calibration records and the report carry. Records
    written before this form hold base58 of only the first 6 bytes
    (``base58=True, legacy_6byte=True`` in the SDK).
    """
    try:
        try:
            hex_hwid = dev.get_hardware_id(module, raw_hex=True)
            args: tuple = (module,)
        except TypeError:
            # Some devices (e.g. HV controller) don't take ``module``.
            hex_hwid = dev.get_hardware_id(raw_hex=True)
            args = ()
    except Exception as e:
        logger.warning("Could not read %s HW ID: %s", label, e)
        return "", ""

    try:
        b58 = dev.get_hardware_id(*args, base58=True)
    except Exception as e:
        logger.warning("Could not read %s base58 HW ID: %s", label, e)
        b58 = ""
    return hex_hwid, b58


def read_console_identity(open_interface: Optional[Callable] = None
                          ) -> tuple[str, str]:
    """Briefly connect to the console and return ``(sn, hwid)``.

    ``sn`` is the serial number stored in the console's user config
    (``"sn"``; ``""`` if the config has none or can't be read) and
    ``hwid`` its base58 hardware ID (the form shown in the report).
    Raises ``RuntimeError`` if the console isn't connected or the
    hardware ID can't be read.

    ``open_interface`` returns the interface to read, as a context
    manager; default: a new ``LIFUInterface`` (the dry run passes
    :meth:`SimulatedHardware.open_lifu
    <openlifu_verification.simulated_hardware.SimulatedHardware.open_lifu>`)."""
    if open_interface is None:
        from openlifu_sdk.io import LIFUInterface as open_interface

    with open_interface() as lifu:
        if not lifu.is_device_connected()[1]:
            raise RuntimeError("Console is not connected.")
        hv = lifu.hvcontroller
        _hex, b58 = _read_hwid(hv, 0, "console")
        sn = _read_config_sn(hv, "console")
    if not b58:
        raise RuntimeError("Could not read the console hardware ID.")
    return sn, b58


def read_console_hwid(open_interface: Optional[Callable] = None) -> str:
    """Briefly connect to the console and return its base58 hardware ID
    (the form shown in the report). Raises ``RuntimeError`` if the
    console isn't connected or the ID can't be read. ``open_interface``
    as in :func:`read_console_identity`."""
    return read_console_identity(open_interface)[1]


def _read_config_sn(dev, label: str) -> str:
    """Serial number (``"sn"``) from ``dev``'s user config, or ``""``."""
    try:
        sn = dev.read_config().get("sn")
    except Exception as e:
        logger.warning("Could not read %s user config: %s", label, e)
        return ""
    return sn.strip() if isinstance(sn, str) else ""


def _read_version(dev, module: int, label: str) -> str:
    try:
        return dev.get_version(module)
    except TypeError:
        try:
            return dev.get_version()
        except Exception as e:
            logger.warning("Could not read %s firmware version: %s", label, e)
            return ""
    except Exception as e:
        logger.warning("Could not read %s firmware version: %s", label, e)
        return ""
