"""Auto-extracted device metadata for the characterization report.

Reads what the SDK can tell us about the connected LIFU interface
(SDK version, TXM/console HW ID + FW version) and packs it into
:class:`DeviceInfo`. HW IDs are exposed in both hex and base58 forms
so the report matches whatever the app expects.
"""
from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass, field
from typing import Optional

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

    @classmethod
    def collect(cls, ver, *, module: int = 0) -> "DeviceInfo":
        """Pull every field we can from ``ver`` (a :class:`VerificationTank`).

        Failures on individual reads are logged and left empty rather
        than raising \u2014 the report can still be filled with the
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

        return info


def _read_hwid(dev, module: int, label: str) -> tuple[str, str]:
    """Return ``(hex_hwid, base58_hwid)`` for a device, or empty strings."""
    try:
        hex_hwid = dev.get_hardware_id(module, raw_hex=True)
    except TypeError:
        # Some devices (e.g. HV controller) don't take ``module``.
        try:
            hex_hwid = dev.get_hardware_id(raw_hex=True)
        except Exception as e:
            logger.warning("Could not read %s HW ID: %s", label, e)
            return "", ""
    except Exception as e:
        logger.warning("Could not read %s HW ID: %s", label, e)
        return "", ""

    try:
        import base58
        from openlifu_sdk.io.LIFUConfig import HW_ID_DATA_LENGTH
        b58 = base58.b58encode(bytes.fromhex(hex_hwid[:HW_ID_DATA_LENGTH * 2])).decode("utf-8")
    except Exception as e:
        logger.warning("Could not base58-encode %s HW ID: %s", label, e)
        b58 = ""
    return hex_hwid, b58


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
