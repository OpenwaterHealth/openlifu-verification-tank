import logging as _logging

from .picoscope import Picoscope
from .qpx600dp import QPX600DP
from .scan_results import ScanResult
from .verificationtank import VerificationTank
from .hydrophone import Hydrophone
from .operator_prefs import OperatorPrefs
from .acceptance import AcceptanceCriteria
from .device_info import DeviceInfo
from .scan_config import ScanConfig
from .characterization import Characterization, TestReport, ReportRow
from .dry_run import DryRunTank
from . import characterization, paths, report_io

# The upstream openlifu_sdk logs a lot at INFO/DEBUG during device
# setup and per-command traffic. Quiet it to WARNING by default so
# scans don't drown the console; users can re-enable via
# ``logging.getLogger("openlifu_sdk").setLevel(logging.DEBUG)`` or the
# ``set_log_level`` helper below.
_logging.getLogger("openlifu_sdk").setLevel(_logging.WARNING)


def set_log_level(level=_logging.INFO, *, sdk_level=None):
    """Set console verbosity for openlifu_verification (and optionally SDK).

    Args:
        level: Level for the ``openlifu_verification`` package logger.
            Accepts ``logging.DEBUG`` / ``INFO`` / ``WARNING`` etc. or
            the string equivalents (``"DEBUG"``, …).
        sdk_level: Optional level for the ``openlifu_sdk`` logger. If
            ``None`` (default) the SDK level is left at whatever it
            currently is (WARNING at import time).
    """
    if isinstance(level, str):
        level = _logging.getLevelName(level.upper())
    _logging.getLogger("openlifu_verification").setLevel(level)
    if sdk_level is not None:
        if isinstance(sdk_level, str):
            sdk_level = _logging.getLevelName(sdk_level.upper())
        _logging.getLogger("openlifu_sdk").setLevel(sdk_level)


__all__ = [
    "Picoscope",
    "QPX600DP",
    "ScanResult",
    "VerificationTank",
    "Hydrophone",
    "OperatorPrefs",
    "AcceptanceCriteria",
    "DeviceInfo",
    "ScanConfig",
    "Characterization",
    "TestReport",
    "ReportRow",
    "DryRunTank",
    "characterization",
    "paths",
    "report_io",
    "set_log_level",
]
