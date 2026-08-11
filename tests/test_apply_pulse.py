"""Tests for :meth:`VerificationTank.apply_pulse` argument resolution.

The full :class:`VerificationTank` requires real hardware, but
``apply_pulse`` just resolves keyword arguments against class-level
``DEFAULT_*`` constants and forwards to ``configure_lifu``. We bind
it as an unbound method against a stub self so we can exercise the
defaults logic without touching any USB device.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from openlifu_verification import VerificationTank


class _StubTank:
    """Just the bits of VerificationTank that ``apply_pulse`` reads."""

    # Mirror the class-level constants exactly.
    DEFAULT_FREQUENCY_KHZ = VerificationTank.DEFAULT_FREQUENCY_KHZ
    DEFAULT_VOLTAGE_V = VerificationTank.DEFAULT_VOLTAGE_V
    DEFAULT_CYCLES_PER_BURST = VerificationTank.DEFAULT_CYCLES_PER_BURST
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
    """No kwargs => every DEFAULT_* comes through, and duration
    is derived from cycles / frequency."""
    r = _call_apply_pulse()
    assert r["frequency_kHz"] == VerificationTank.DEFAULT_FREQUENCY_KHZ
    assert r["voltage"] == VerificationTank.DEFAULT_VOLTAGE_V
    expected_duration = (VerificationTank.DEFAULT_CYCLES_PER_BURST
                         / VerificationTank.DEFAULT_FREQUENCY_KHZ)
    assert r["duration_msec"] == expected_duration
    assert r["interval_msec"] == VerificationTank.DEFAULT_INTERVAL_MSEC
    assert r["pulse_count"] == VerificationTank.DEFAULT_PULSE_COUNT
    assert r["trigger_mode"] == VerificationTank.DEFAULT_TRIGGER_MODE


def test_explicit_kwargs_override_defaults():
    r = _call_apply_pulse(
        frequency_kHz=155.0, voltage=42.5,
        duration_msec=0.5, interval_msec=100.0,
        pulse_count=8, trigger_mode="single",
    )
    assert r["frequency_kHz"] == 155.0
    assert r["voltage"] == 42.5
    assert r["duration_msec"] == 0.5
    assert r["interval_msec"] == 100.0
    assert r["pulse_count"] == 8
    assert r["trigger_mode"] == "single"


def test_duration_derived_from_cycles_when_omitted():
    """When ``duration_msec`` is not passed but ``cycles_per_burst`` is,
    duration should track cycles / freq."""
    r = _call_apply_pulse(frequency_kHz=200.0, cycles_per_burst=40.0)
    assert r["duration_msec"] == 40.0 / 200.0


def test_explicit_duration_wins_over_cycles():
    """Explicit ``duration_msec`` must not be overwritten by the
    cycles/frequency derivation."""
    r = _call_apply_pulse(
        frequency_kHz=200.0, cycles_per_burst=40.0,
        duration_msec=0.123,
    )
    assert r["duration_msec"] == 0.123


def test_pulse_count_and_trigger_mode_forwarded():
    """The new pulse_sequence script relies on ``pulse_count`` +
    ``trigger_mode="single"`` making it to ``configure_lifu``."""
    r = _call_apply_pulse(pulse_count=16, trigger_mode="single")
    assert r["pulse_count"] == 16
    assert r["trigger_mode"] == "single"


def test_pulse_count_defaults_to_class_constant_when_omitted():
    r = _call_apply_pulse(voltage=5.0)
    assert r["pulse_count"] == VerificationTank.DEFAULT_PULSE_COUNT
