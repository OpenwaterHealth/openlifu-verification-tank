"""Pre-emission hydrophone baseline removal.

The hydrophone channel is DC-coupled, so a DC level on the hydrophone
adds straight onto every sample and biases the windowed PNP. Each
capture's mean before emission is subtracted in ``run_capture`` and
``finish_rapid_capture``, and a large baseline is logged.
"""
from __future__ import annotations

import logging
import types

import numpy as np
import pytest

from openlifu_verification.verificationtank import (
    HYDROPHONE_BASELINE_WARN_MV,
    VerificationTank,
    remove_pre_emission_baseline,
)

T_US = np.arange(-10.0, 120.0, 0.1)


def _burst(t_us, amp=50.0, f_MHz=0.155, start_us=35.0, dur_us=60.0):
    on = (t_us >= start_us) & (t_us < start_us + dur_us)
    return np.where(on, amp * np.sin(2 * np.pi * f_MHz * (t_us - start_us)), 0.0)


def test_single_trace_offset_is_removed():
    sig = _burst(T_US)
    result = {"time": T_US, "A": sig - 14.0}
    baseline = remove_pre_emission_baseline(result, "A")
    assert baseline == pytest.approx(-14.0)
    np.testing.assert_allclose(result["A"], sig, atol=1e-9)


def test_each_capture_gets_its_own_baseline():
    sig = _burst(T_US)
    offsets = np.array([-14.0, -29.0, 0.5])
    result = {"time": T_US, "A": sig[None, :] + offsets[:, None]}
    baseline = remove_pre_emission_baseline(result, "A")
    np.testing.assert_allclose(baseline, offsets)
    np.testing.assert_allclose(result["A"], np.tile(sig, (3, 1)), atol=1e-9)


def test_pnp_no_longer_follows_the_offset():
    # The R&R spike: same acoustic burst, offset doubled at one point.
    sig = _burst(T_US)
    a = {"time": T_US, "A": sig - 14.0}
    b = {"time": T_US, "A": sig - 29.0}
    remove_pre_emission_baseline(a, "A")
    remove_pre_emission_baseline(b, "A")
    assert -a["A"].min() == pytest.approx(-b["A"].min())
    assert -a["A"].min() == pytest.approx(50.0, rel=1e-3)


@pytest.mark.parametrize("t_start_us", [0.0, -0.5, -1.0])
def test_too_few_pre_emission_samples_leaves_trace_alone(t_start_us):
    t = np.arange(t_start_us, 120.0, 0.1)
    trace = _burst(t) - 14.0
    result = {"time": t, "A": trace.copy()}
    assert remove_pre_emission_baseline(result, "A") is None
    np.testing.assert_array_equal(result["A"], trace)


def test_missing_channel_is_ignored():
    result = {"time": T_US, "B": np.ones_like(T_US)}
    assert remove_pre_emission_baseline(result, "A") is None
    assert remove_pre_emission_baseline(None, "A") is None


def _tank(delay_us=113.5):
    tank = types.SimpleNamespace(system_transmit_delay_us=delay_us,
                                 hydrophone_channel="A")
    tank.scope = types.SimpleNamespace(
        plan_capture=lambda *, sampling_interval_ns, time_start_s, time_stop_s: {
            "sampling_interval_ns": float(sampling_interval_ns),
            "time_start_s": float(time_start_s),
            "time_stop_s": float(time_stop_s),
            "timebase": 3, "pre_trigger_samples": 0,
            "post_trigger_samples": int(round((time_stop_s - time_start_s)
                                              / (sampling_interval_ns * 1e-9))),
            "delay_samples": 0,
        },
        set_trigger_delay=lambda _n: None,
        wait_ready=lambda timeout_s=None: True,
        reset_rapid_block=lambda: None,
    )
    return tank


def _capture(tank, offset_mV):
    n = len(T_US)
    tank._run_capture_block = lambda **_: {
        "time": np.arange(n, dtype=float) * 100.0,
        "A": _burst(T_US) + offset_mV,
        "B": np.full(n, 7.0),
    }
    return VerificationTank.run_capture(
        tank, time_start_s=-10e-6, time_stop_s=120e-6, sampling_interval_ns=100)


def test_run_capture_removes_baseline_on_hydrophone_only():
    tank = _tank()
    result = _capture(tank, -14.0)
    assert result["time"][0] == pytest.approx(-10.0, abs=1e-6)
    np.testing.assert_allclose(result["A"], _burst(T_US), atol=1e-9)
    np.testing.assert_array_equal(result["B"], 7.0)  # trigger channel untouched


def test_large_baseline_warns_once_then_only_when_worse(caplog):
    tank = _tank()
    caplog.set_level(logging.WARNING, logger="openlifu_verification.verificationtank")
    _capture(tank, -0.5 * HYDROPHONE_BASELINE_WARN_MV)
    assert not caplog.records
    _capture(tank, -14.0)
    _capture(tank, -14.3)
    assert len(caplog.records) == 1
    assert "14.0 mV" in caplog.records[0].getMessage()
    _capture(tank, -29.0)
    assert len(caplog.records) == 2


def test_finish_rapid_capture_removes_per_segment_baseline():
    tank = _tank()
    offsets = np.array([-14.0, -29.0, -14.5])
    n = len(T_US)
    tank.scope.get_data_rapid = lambda **_: {
        "time": np.arange(n, dtype=float) * 100.0,
        "A": _burst(T_US)[None, :] + offsets[:, None],
        "overflow": np.zeros(3, dtype=np.int16),
    }
    plan = {"n_captures": 3, "samples_per_segment": n, "timebase": 3,
            "sampling_interval_ns": 100.0,
            "time_start_s": -10e-6 + 113.5e-6, "time_stop_s": 120e-6 + 113.5e-6}
    result = VerificationTank.finish_rapid_capture(tank, plan)
    assert result["time"][0] == pytest.approx(-10.0, abs=1e-6)
    np.testing.assert_allclose(result["A"], np.tile(_burst(T_US), (3, 1)), atol=1e-9)
