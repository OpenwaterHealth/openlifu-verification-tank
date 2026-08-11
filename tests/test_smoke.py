"""Import-level smoke tests for the public entry points.

Cheap safety net that catches "someone renamed a public symbol"
without needing hardware.
"""
from __future__ import annotations


def test_public_top_level_names_import():
    import openlifu_verification as ov  # noqa: F401
    for name in (
        "Picoscope", "QPX600DP", "ScanResult", "VerificationTank",
        "Hydrophone", "OperatorPrefs", "AcceptanceCriteria", "DeviceInfo",
        "ScanConfig", "Characterization", "TestReport", "ReportRow",
        "DryRunTank", "paths", "characterization", "report_io",
        "set_log_level",
    ):
        assert hasattr(ov, name), f"openlifu_verification missing {name}"


def test_verificationtank_class_defaults_exist():
    """The scripts read these class constants directly at argparse
    build time — renaming any of them silently breaks every CLI."""
    from openlifu_verification import VerificationTank as VT
    for name in (
        "DEFAULT_FREQUENCY_KHZ", "DEFAULT_VOLTAGE_V",
        "DEFAULT_CYCLES_PER_BURST", "DEFAULT_INTERVAL_MSEC",
        "DEFAULT_PULSE_COUNT", "DEFAULT_TRIGGER_MODE",
    ):
        assert hasattr(VT, name), f"VerificationTank missing {name}"


def test_verificationtank_has_capture_pulse_train():
    """New method for the pulse_sequence script."""
    from openlifu_verification import VerificationTank as VT
    assert callable(getattr(VT, "capture_pulse_train", None)), (
        "VerificationTank.capture_pulse_train is missing"
    )


def test_script_modules_are_parseable():
    """Each script should at least be import-parseable via ``py_compile``
    so a syntax error can't sneak past a rewrite."""
    import py_compile
    from pathlib import Path
    scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
    for py in scripts_dir.glob("*.py"):
        # Notebooks and data files live alongside; only compile .py.
        py_compile.compile(str(py), doraise=True)
