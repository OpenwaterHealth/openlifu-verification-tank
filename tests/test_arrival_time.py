"""Tests for the first-arrival RF-peak picker."""
import numpy as np
import pytest

from openlifu_verification.arrival_time import find_first_arrival_us


def _boxcar_burst(*, t_us, onset_us, freq_kHz, n_cycles=8, amp=1.0):
    """Hard-onset sine burst: zero before ``onset_us``, then ``amp *
    sin(2\u03c0 f (t - onset_us))`` for ``n_cycles`` periods. The
    first positive RF peak sits at ``onset_us + T/4``, so the picker
    should report ``arrival_us == onset_us``."""
    T_us = 1000.0 / freq_kHz
    stop_us = onset_us + n_cycles * T_us
    trace = np.zeros_like(t_us)
    mask = (t_us >= onset_us) & (t_us <= stop_us)
    trace[mask] = amp * np.sin(2 * np.pi * freq_kHz * 1e3 *
                                (t_us[mask] - onset_us) * 1e-6)
    return trace


def _time_axis(dt_us=0.02, t_stop_us=100.0):
    return np.arange(0.0, t_stop_us, dt_us)


@pytest.mark.parametrize("onset_us", [20.0, 33.3, 50.0, 66.7])
def test_arrival_recovers_boxcar_onset(onset_us):
    """A hard-onset tone-burst starts with a rising sinusoid; the first
    positive crest is at onset + T/4, so the quarter-cycle-corrected
    arrival must equal the onset."""
    freq_kHz = 400.0
    t_us = _time_axis()
    trace = _boxcar_burst(t_us=t_us, onset_us=onset_us, freq_kHz=freq_kHz)
    res = find_first_arrival_us(t_us, trace, frequency_kHz=freq_kHz,
                                skip_us=12.0)
    assert res is not None
    # Sub-sample refinement should land well inside 5% of a period.
    tol_us = 0.05 * 1000.0 / freq_kHz
    assert abs(res.arrival_us - onset_us) < tol_us


def test_arrival_returns_none_when_empty_window():
    freq_kHz = 400.0
    t_us = _time_axis(t_stop_us=10.0)
    trace = _boxcar_burst(t_us=t_us, onset_us=5.0, freq_kHz=freq_kHz)
    # skip_us beyond the record.
    assert find_first_arrival_us(t_us, trace, frequency_kHz=freq_kHz,
                                 skip_us=20.0) is None


def test_arrival_skips_early_transient():
    """A big transient before skip_us must not fool the picker."""
    freq_kHz = 400.0
    t_us = _time_axis()
    # Real burst at 50 us + big trigger flash centered at 3 us.
    trace = _boxcar_burst(t_us=t_us, onset_us=50.0, freq_kHz=freq_kHz)
    trace = trace + _boxcar_burst(t_us=t_us, onset_us=3.0,
                                   freq_kHz=freq_kHz, n_cycles=4, amp=5.0)
    res = find_first_arrival_us(t_us, trace, frequency_kHz=freq_kHz,
                                skip_us=12.0)
    assert res is not None
    assert abs(res.arrival_us - 50.0) < 0.05


def test_returns_none_when_no_positive_signal():
    """When the trace has no positive-going signal above zero, the
    picker's threshold is undefined and it must return None."""
    freq_kHz = 400.0
    t_us = _time_axis()
    # All-negative trace: global_peak (max) <= 0 -> None.
    trace = -np.ones_like(t_us)
    assert find_first_arrival_us(t_us, trace, frequency_kHz=freq_kHz,
                                 skip_us=12.0) is None


def test_quarter_cycle_correction_value():
    freq_kHz = 400.0
    t_us = _time_axis()
    trace = _boxcar_burst(t_us=t_us, onset_us=40.0, freq_kHz=freq_kHz)
    res = find_first_arrival_us(t_us, trace, frequency_kHz=freq_kHz,
                                skip_us=12.0)
    assert res is not None
    # 250 / 400 = 0.625 us at 400 kHz.
    assert res.quarter_cycle_us == pytest.approx(0.625, rel=1e-9)
    assert res.first_peak_us - res.arrival_us == pytest.approx(
        0.625, rel=1e-9,
    )


def test_threshold_gates_early_low_amplitude_wobble():
    """A tiny low-amplitude sinusoid before the real burst should be
    rejected by the prominence threshold and not fool the picker."""
    freq_kHz = 400.0
    t_us = _time_axis()
    # 5% amplitude early wobble at 20us + 100% real burst at 60us.
    tiny = _boxcar_burst(t_us=t_us, onset_us=20.0, freq_kHz=freq_kHz,
                         amp=0.05, n_cycles=6)
    real = _boxcar_burst(t_us=t_us, onset_us=60.0, freq_kHz=freq_kHz,
                         amp=1.0)
    trace = tiny + real
    # Threshold at 0.25 * 1.0 = 0.25 rejects the 0.05 wobble.
    res = find_first_arrival_us(t_us, trace, frequency_kHz=freq_kHz,
                                skip_us=12.0, min_prominence_frac=0.25)
    assert res is not None
    # Real onset at 60 us; picker should land within a small fraction
    # of a period.
    assert abs(res.arrival_us - 60.0) < 0.05
