"""Tests for :meth:`VerificationTank.apply_pulse` argument resolution.

The full :class:`VerificationTank` requires real hardware, but
``apply_pulse`` just resolves keyword arguments against class-level
``DEFAULT_*`` constants and forwards to ``configure_lifu``. We bind
it as an unbound method against a stub self so we can exercise the
defaults logic without touching any USB device.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from openlifu_verification import VerificationTank
from openlifu_verification.acceptance import SUPPORTED_FREQUENCIES_KHZ


class _StubTank:
    """Just the bits of VerificationTank that ``apply_pulse`` reads."""

    # Mirror the class-level constants exactly.
    DEFAULT_FREQUENCY_KHZ = VerificationTank.DEFAULT_FREQUENCY_KHZ
    DEFAULT_VOLTAGE_BY_FREQ_KHZ = VerificationTank.DEFAULT_VOLTAGE_BY_FREQ_KHZ
    default_voltage_for = VerificationTank.default_voltage_for
    DEFAULT_DURATION_US_BY_FREQ_KHZ = VerificationTank.DEFAULT_DURATION_US_BY_FREQ_KHZ
    default_duration_us_for = VerificationTank.default_duration_us_for
    frequency = VerificationTank.DEFAULT_FREQUENCY_KHZ
    DEFAULT_INTERVAL_MSEC = VerificationTank.DEFAULT_INTERVAL_MSEC
    DEFAULT_PULSE_COUNT = VerificationTank.DEFAULT_PULSE_COUNT
    DEFAULT_TRIGGER_MODE = VerificationTank.DEFAULT_TRIGGER_MODE

    def __init__(self):
        self.configure_lifu = MagicMock()


def _call_apply_pulse(**kwargs) -> dict:
    """Invoke ``apply_pulse`` bound to a stub self and return the
    dict it hands to ``configure_lifu``."""
    stub = _StubTank()
    resolved = VerificationTank.apply_pulse(stub, **kwargs)
    stub.configure_lifu.assert_called_once()
    passed = stub.configure_lifu.call_args.kwargs
    assert passed == resolved, "apply_pulse must forward the resolved dict"
    return resolved


def test_defaults_when_nothing_passed():
    """No kwargs => every DEFAULT_* comes through, with voltage and
    duration from the per-frequency tables."""
    r = _call_apply_pulse()
    f = VerificationTank.DEFAULT_FREQUENCY_KHZ
    assert r["frequency_kHz"] == f
    assert r["voltage"] == VerificationTank.DEFAULT_VOLTAGE_BY_FREQ_KHZ[f]
    assert r["duration_usec"] == VerificationTank.DEFAULT_DURATION_US_BY_FREQ_KHZ[f]
    assert r["interval_msec"] == VerificationTank.DEFAULT_INTERVAL_MSEC
    assert r["pulse_count"] == VerificationTank.DEFAULT_PULSE_COUNT
    assert r["trigger_mode"] == VerificationTank.DEFAULT_TRIGGER_MODE


def test_voltage_default_is_frequency_dependent():
    assert _call_apply_pulse(frequency_kHz=155.0)["voltage"] == 40.0
    assert _call_apply_pulse(frequency_kHz=400.0)["voltage"] == 20.0
    # Unlisted frequency has no default; caller must pass voltage.
    with pytest.raises(ValueError):
        _call_apply_pulse(frequency_kHz=250.0)
    assert _call_apply_pulse(frequency_kHz=250.0, voltage=10.0,
                             cycles_per_burst=20.0)["voltage"] == 10.0


def test_drive_tables_cover_all_supported_frequencies():
    supported = set(SUPPORTED_FREQUENCIES_KHZ)
    assert set(VerificationTank.DEFAULT_VOLTAGE_BY_FREQ_KHZ) == supported
    assert set(VerificationTank.DEFAULT_DURATION_US_BY_FREQ_KHZ) == supported


def test_duration_default_is_frequency_dependent():
    assert _call_apply_pulse(frequency_kHz=155.0)["duration_usec"] == 70.0
    assert _call_apply_pulse(frequency_kHz=400.0)["duration_usec"] == 50.0
    # Explicit cycles still win over the per-frequency table.
    r = _call_apply_pulse(frequency_kHz=155.0, cycles_per_burst=20.0)
    assert r["duration_usec"] == 20.0 / 155.0 * 1000.0


def test_explicit_kwargs_override_defaults():
    r = _call_apply_pulse(
        frequency_kHz=155.0, voltage=42.5,
        duration_usec=500.0, interval_msec=100.0,
        pulse_count=8, trigger_mode="single",
    )
    assert r["frequency_kHz"] == 155.0
    assert r["voltage"] == 42.5
    assert r["duration_usec"] == 500.0
    assert r["interval_msec"] == 100.0
    assert r["pulse_count"] == 8
    assert r["trigger_mode"] == "single"


def test_duration_derived_from_cycles_when_omitted():
    """When ``duration_usec`` is not passed but ``cycles_per_burst`` is,
    duration should track cycles / freq * 1000."""
    r = _call_apply_pulse(frequency_kHz=200.0, voltage=10.0,
                          cycles_per_burst=40.0)
    assert r["duration_usec"] == (40.0 / 200.0) * 1000.0


def test_explicit_duration_wins_over_cycles():
    """Explicit ``duration_usec`` must not be overwritten by the
    cycles/frequency derivation."""
    r = _call_apply_pulse(
        frequency_kHz=200.0, voltage=10.0, cycles_per_burst=40.0,
        duration_usec=123.0,
    )
    assert r["duration_usec"] == 123.0


def test_pulse_count_and_trigger_mode_forwarded():
    """The new pulse_sequence script relies on ``pulse_count`` +
    ``trigger_mode="single"`` making it to ``configure_lifu``."""
    r = _call_apply_pulse(pulse_count=16, trigger_mode="single")
    assert r["pulse_count"] == 16
    assert r["trigger_mode"] == "single"


def test_pulse_count_defaults_to_class_constant_when_omitted():
    r = _call_apply_pulse(voltage=5.0)
    assert r["pulse_count"] == VerificationTank.DEFAULT_PULSE_COUNT


def test_apodization_default_is_none():
    """When the caller doesn't pass ``apodization``, the resolved
    dict carries ``None`` so ``configure_lifu`` falls through to
    its all-elements-on branch."""
    r = _call_apply_pulse()
    assert r["apodization"] is None


def test_apodization_normalised_to_canonical_hex():
    """Any accepted representation should end up as a 16-digit
    upper-case ``0x...`` string in the resolved dict."""
    r = _call_apply_pulse(apodization="0x1")
    assert r["apodization"] == "0x0000000000000001"

    r = _call_apply_pulse(apodization=0xDEADBEEF)
    assert r["apodization"] == "0x00000000DEADBEEF"


def test_apodization_bad_hex_raises():
    """Malformed hex strings should surface immediately at the
    apply_pulse boundary, not later inside configure_lifu."""
    import pytest
    with pytest.raises(ValueError):
        _call_apply_pulse(apodization="0xNOTHEX")
