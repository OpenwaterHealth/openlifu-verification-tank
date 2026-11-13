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
    positive crest is at onset + T/4, and the walk-back must land on
    the onset."""
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


def test_onset_to_peak_is_quarter_cycle_for_clean_burst():
    freq_kHz = 400.0
    t_us = _time_axis()
    trace = _boxcar_burst(t_us=t_us, onset_us=40.0, freq_kHz=freq_kHz)
    res = find_first_arrival_us(t_us, trace, frequency_kHz=freq_kHz,
                                skip_us=12.0)
    assert res is not None
    # 250 / 400 = 0.625 us at 400 kHz.
    assert res.onset_to_peak_us == pytest.approx(0.625, abs=0.03)


def _ramped_burst(*, t_us, onset_us, freq_kHz, tau_cycles, amp=1.0,
                  n_cycles=10):
    """Resonant ring-up: ``amp * (1 - exp(-t/tau)) * sin(2π f t)``."""
    T_us = 1000.0 / freq_kHz
    dt = t_us - onset_us
    on = (dt >= 0) & (dt <= n_cycles * T_us)
    env = 1.0 - np.exp(-np.clip(dt, 0, None) / (tau_cycles * T_us))
    return np.where(on, amp * env * np.sin(2 * np.pi * dt / T_us), 0.0)


@pytest.mark.parametrize("freq_kHz", [155.0, 400.0])
def test_sub_threshold_first_cycle_is_recovered(freq_kHz):
    """Slow ring-up whose first crest is below the prominence threshold:
    the first qualifying crest is a period late, and peak - T/4 would be
    off by a whole period. The walk-back must recover the onset."""
    t_us = _time_axis()
    rng = np.random.default_rng(1)
    trace = _ramped_burst(t_us=t_us, onset_us=33.6, freq_kHz=freq_kHz,
                          tau_cycles=4.0)
    trace = trace + rng.normal(0.0, 0.002, t_us.size)
    res = find_first_arrival_us(t_us, trace, frequency_kHz=freq_kHz,
                                skip_us=14.0, min_prominence_frac=0.2)
    assert res is not None
    T_us = 1000.0 / freq_kHz
    assert res.first_peak_us - 33.6 > T_us  # crest picked is not the 1st
    assert abs(res.arrival_us - 33.6) < 0.1 * T_us


def test_negative_first_lobe_is_included():
    """Inverted burst: the first positive crest is 3T/4 after the onset;
    the walk-back must step over the leading negative lobe."""
    freq_kHz = 155.0
    t_us = _time_axis()
    trace = -_boxcar_burst(t_us=t_us, onset_us=33.6, freq_kHz=freq_kHz,
                           n_cycles=2)
    res = find_first_arrival_us(t_us, trace, frequency_kHz=freq_kHz,
                                skip_us=14.0)
    assert res is not None
    assert abs(res.arrival_us - 33.6) < 0.05


def test_slow_baseline_drift_is_not_walked_into():
    """A residual baseline slope wider than a carrier period before the
    burst must not be mistaken for an earlier signal lobe."""
    freq_kHz = 400.0
    t_us = _time_axis()
    rng = np.random.default_rng(2)
    drift = np.where(t_us < 40.0, -0.05 * (40.0 - t_us) / 20.0, 0.0)
    trace = (_boxcar_burst(t_us=t_us, onset_us=40.0, freq_kHz=freq_kHz)
             + drift + rng.normal(0.0, 0.002, t_us.size))
    res = find_first_arrival_us(t_us, trace, frequency_kHz=freq_kHz,
                                skip_us=14.0)
    assert res is not None
    assert abs(res.arrival_us - 40.0) < 0.1


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


@pytest.mark.parametrize("freq_kHz", [155.0, 400.0])
@pytest.mark.parametrize("dc", [0.003, 0.01])
def test_flat_positive_foot_is_not_walked_into(freq_kHz, dc):
    """A small positive residual baseline under a 32-pulse median keeps
    the sign from flipping before the burst, so the first crest's
    same-sign run reaches back across the flat foot (seen on 155 kHz
    focused depth calibrations as a ~2 us early arrival). The onset must
    stay at the real start of the leading edge."""
    t_us = np.arange(0.0, 100.0, 0.1)
    rng = np.random.default_rng(3)
    trace = (_boxcar_burst(t_us=t_us, onset_us=37.3, freq_kHz=freq_kHz)
             + dc * (t_us > 22.0) + rng.normal(0.0, 0.001, t_us.size))
    res = find_first_arrival_us(t_us, trace, frequency_kHz=freq_kHz,
                                skip_us=25.0)
    assert res is not None
    assert abs(res.arrival_us - 37.3) < 0.05


@pytest.mark.parametrize("freq_kHz", [155.0, 400.0])
def test_flat_foot_check_keeps_ring_up_onset(freq_kHz):
    """A slow ring-up's first lobe starts as ~t^2, not a flat foot; its
    zero-crossing onset must still be used (no late extrapolation)."""
    t_us = np.arange(0.0, 100.0, 0.02)
    trace = _ramped_burst(t_us=t_us, onset_us=33.6, freq_kHz=freq_kHz,
                          tau_cycles=4.0)
    res = find_first_arrival_us(t_us, trace, frequency_kHz=freq_kHz,
                                skip_us=14.0, min_prominence_frac=0.01)
    assert res is not None
    assert abs(res.arrival_us - 33.6) < 0.05


def _plane_wave_tank(amp_mv, freq_kHz=155.0):
    """Stand-in ``self`` for :meth:`VerificationTank.calibrate_hydrophone_depth`
    whose 32-pulse capture is a plane-wave burst of ``amp_mv`` at 34 us."""
    from types import SimpleNamespace

    from openlifu_verification.verificationtank import VerificationTank

    t_us = np.arange(0.0, 100.0, 0.1)
    burst = _boxcar_burst(t_us=t_us, onset_us=34.0, freq_kHz=freq_kHz,
                          amp=amp_mv, n_cycles=3)
    return SimpleNamespace(
        arr=object(), frequency=freq_kHz, hydrophone_channel="B",
        hydrophone_position=[0.0, 0.0, 0.0], system_transmit_delay_us=0.0,
        MIN_PLANE_WAVE_PEAK_MV_PER_V_BY_FREQ_KHZ=(
            VerificationTank.MIN_PLANE_WAVE_PEAK_MV_PER_V_BY_FREQ_KHZ),
        apply_pulse=lambda **kw: {"interval_msec": 20.0},
        set_focus=lambda *a: None,
        enable_hv_output=lambda **kw: None,
        capture_pulse_train=lambda **kw: {"time": t_us,
                                          "B": np.tile(burst, (32, 1))},
    )


@pytest.mark.parametrize("amp_mv, ok", [(8.0, True), (2.0, False)])
def test_plane_wave_depth_rejects_low_signal(amp_mv, ok):
    """At 155 kHz / 40 V the passing reports peak at ~8-9 mV; a bubble
    that drops the signal below 4 mV must stop the calibration with an
    operator-actionable message instead of picking noise."""
    from openlifu_verification.verificationtank import VerificationTank

    tank = _plane_wave_tank(amp_mv)
    if ok:
        res = VerificationTank.calibrate_hydrophone_depth(tank, voltage_V=40.0)
        assert res["peak_mv"] == pytest.approx(amp_mv, rel=0.05)
        assert abs(res["arrival_us"] - 34.0) < 0.1
    else:
        with pytest.raises(RuntimeError, match="plane wave signal is too low"
                           ".*Check for bubbles"):
            VerificationTank.calibrate_hydrophone_depth(tank, voltage_V=40.0)


# ----------------------------------------------------------------------
# Depth calibration detection: expected-depth window, SNR, edge picks
# ----------------------------------------------------------------------
def _noisy_tank(*, bursts=(), noise_mv=0.0, freq_kHz=155.0, seed=0):
    """Like ``_plane_wave_tank`` but with Gaussian noise (independent per
    pulse) and any number of ``(onset_us, amp_mv, n_cycles)`` bursts."""
    tank = _plane_wave_tank(0.0, freq_kHz)
    t_us = np.arange(0.0, 100.0, 0.1)
    clean = np.zeros_like(t_us)
    for onset, amp, cycles in bursts:
        clean += _boxcar_burst(t_us=t_us, onset_us=onset, freq_kHz=freq_kHz,
                               amp=amp, n_cycles=cycles)
    rng = np.random.default_rng(seed)
    traces = clean + rng.normal(0.0, noise_mv, size=(32, t_us.size))
    tank.capture_pulse_train = lambda **kw: {"time": t_us, "B": traces}
    return tank


def _cal(tank, **kw):
    from openlifu_verification.verificationtank import VerificationTank
    return VerificationTank.calibrate_hydrophone_depth(
        tank, voltage_V=40.0, align=False, **kw)


DETECT = dict(expected_mm=50.0, min_snr=6.0)


def test_noise_only_used_to_give_a_bogus_early_depth():
    """The bench failure: noise above the amplitude gate and no real
    arrival. Without the new checks the picker returns an "arrival"
    near the start of its search window (a ~27 mm depth)."""
    tank = _noisy_tank(noise_mv=6.0)
    res = _cal(tank)                       # old behaviour (no DETECT)
    assert res["distance_mm"] < 40.0


def test_noise_only_is_not_found_with_detection():
    from openlifu_verification.verificationtank import DepthNotFound
    tank = _noisy_tank(noise_mv=6.0)
    with pytest.raises(DepthNotFound) as info:
        _cal(tank, **DETECT)
    assert info.value.details["snr"] < 6.0


def test_weak_but_clean_arrival_is_found():
    """A real arrival at 34 us (50.7 mm) well above the noise is found,
    and its signal-to-noise ratio is reported."""
    tank = _noisy_tank(bursts=[(34.0, 8.0, 3)], noise_mv=0.5)
    res = _cal(tank, **DETECT)
    assert abs(res["arrival_us"] - 34.0) < 0.2
    assert res["snr"] > 6.0
    lo, hi = res["search_window_us"]
    assert lo < 34.0 < hi


def test_search_window_ignores_an_early_spike():
    """A strong early transient at 18 us (~27 mm) is outside the
    50 +/- 10 mm window, so the real arrival at 34 us is picked."""
    tank = _noisy_tank(bursts=[(18.0, 20.0, 1), (34.0, 8.0, 3)], noise_mv=0.3)
    assert _cal(tank)["arrival_us"] < 25.0          # old behaviour
    assert abs(_cal(tank, **DETECT)["arrival_us"] - 34.0) < 0.2


def test_arrival_at_the_window_edge_is_not_found():
    """Signal already ringing when the search window opens (a burst from
    26.5 us; the 50 +/- 10 mm window opens at 26.8 us) has no onset
    inside the window: not found."""
    from openlifu_verification.verificationtank import DepthNotFound
    tank = _noisy_tank(bursts=[(26.5, 8.0, 6)], noise_mv=0.3)
    with pytest.raises(DepthNotFound, match="start of the search window"):
        _cal(tank, **DETECT)


def test_low_amplitude_gate_raises_depth_not_found():
    from openlifu_verification.verificationtank import DepthNotFound
    tank = _plane_wave_tank(2.0)
    with pytest.raises(DepthNotFound, match="plane wave signal is too low"):
        _cal(tank)


def test_detection_never_reports_noise_as_a_depth():
    """Seeded sweep: a real 155 kHz arrival at signal-to-noise ~7 is
    found within 1 us (1.5 mm) every time, and pure noise is never
    reported as a depth - it is "not found" (which triggers the focused
    fallback) instead."""
    from openlifu_verification.verificationtank import DepthNotFound
    for seed in range(40):
        res = _cal(_noisy_tank(bursts=[(34.0, 8.0, 3)], noise_mv=5.0, seed=seed),
                   **DETECT)
        assert abs(res["arrival_us"] - 34.0) < 1.0, seed
        with pytest.raises(DepthNotFound):
            _cal(_noisy_tank(noise_mv=6.0, seed=seed), **DETECT)


@pytest.mark.parametrize("freq_kHz", [155.0, 400.0])
@pytest.mark.parametrize("depth_mm", [41.0, 44.0, 47.0, 48.5, 49.5, 50.0, 52.0, 55.0, 58.0, 59.0])
def test_depths_across_the_search_window_are_found(freq_kHz, depth_mm):
    """Regression (v1.0.2-rc): the edge-pick margin is one carrier period,
    ~9.6 mm at 155 kHz, and it used to sit inside the 50 +/- 10 mm window,
    so every 155 kHz depth below ~49.6 mm (typical focused D.16 values)
    was rejected as "at the start of the search window"."""
    tank = _noisy_tank(bursts=[(depth_mm / 1.49, 8.0, 3)], noise_mv=0.5,
                       freq_kHz=freq_kHz)
    res = _cal(tank, **DETECT)
    assert abs(res["distance_mm"] - depth_mm) < 1.0
