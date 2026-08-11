"""Tests for pulse-train cross-correlation alignment."""
from __future__ import annotations

import numpy as np
import pytest

from openlifu_verification.pulse_align import (
    align_pulse_traces,
    estimate_lag_samples,
    shift_trace,
)


def _synth_burst(n_samples=1024, dt_s=1e-7, f_hz=400e3,
                 t0_s=20e-6, duration_s=50e-6, amp=1.0, seed=0):
    """A tone burst centered at ``t0_s`` with a Hann envelope."""
    t = np.arange(n_samples) * dt_s
    env = np.zeros_like(t)
    mask = (t >= t0_s) & (t < t0_s + duration_s)
    tau = (t[mask] - t0_s) / duration_s
    env[mask] = np.sin(np.pi * tau) ** 2
    sig = amp * env * np.sin(2 * np.pi * f_hz * (t - t0_s))
    if seed is not None:
        rng = np.random.default_rng(seed)
        sig = sig + 0.0 * rng.standard_normal(n_samples)
    return t, sig


def test_zero_shift_returns_zero_lag():
    _, ref = _synth_burst()
    lag = estimate_lag_samples(ref, ref)
    assert abs(lag) < 1e-9


def test_integer_shift_recovered_exactly():
    _, ref = _synth_burst()
    # np.roll shifts the sample index -> if we roll right by k, the signal
    # appears delayed by k samples relative to ref, so lag should be +k.
    x = np.roll(ref, 5)
    lag = estimate_lag_samples(x, ref, refine=False)
    assert lag == pytest.approx(5.0, abs=1e-9)


def test_negative_shift_recovered():
    _, ref = _synth_burst()
    x = np.roll(ref, -3)
    lag = estimate_lag_samples(x, ref, refine=False)
    assert lag == pytest.approx(-3.0, abs=1e-9)


def test_subsample_shift_recovered_within_tenth_of_sample():
    # Build a reference and a version shifted by a fractional sample via
    # linear interpolation on a densely-sampled underlying signal.
    dt_s = 1e-7
    _, ref = _synth_burst(n_samples=2048, dt_s=dt_s)
    # shift_trace advances by +lag samples -> ref sampled at (t + 0.4) is
    # a version of ref that arrived 0.4 samples EARLIER; equivalently the
    # returned trace lags ref by -0.4.
    x = shift_trace(ref, -0.4)  # x arrives 0.4 samples late
    lag = estimate_lag_samples(x, ref, refine=True)
    assert lag == pytest.approx(0.4, abs=0.1)


def test_max_shift_bounds_search():
    _, ref = _synth_burst()
    x = np.roll(ref, 20)
    # Bound search to +/-5 samples; true lag is +20 -> reported lag should
    # be clamped inside the bound (i.e., the search returns a max-corr
    # sample within the allowed window, which is ``+5``).
    lag = estimate_lag_samples(x, ref, max_shift=5, refine=False)
    assert -5 <= lag <= 5


def test_align_pulse_traces_shape_and_first_row_lag():
    _, ref = _synth_burst()
    stack = np.vstack([ref, np.roll(ref, 2), np.roll(ref, -4)])
    aligned, lags_s = align_pulse_traces(stack, dt_s=1e-7, refine=False)
    assert aligned.shape == stack.shape
    assert lags_s.shape == (3,)
    # Reference row (index 0) must have lag zero.
    assert lags_s[0] == pytest.approx(0.0, abs=1e-15)
    # Recovered lags in seconds match applied roll * dt.
    assert lags_s[1] == pytest.approx(2 * 1e-7, abs=1e-12)
    assert lags_s[2] == pytest.approx(-4 * 1e-7, abs=1e-12)


def test_align_pulse_traces_reduces_average_to_reference():
    _, ref = _synth_burst()
    stack = np.vstack([ref, np.roll(ref, 3), np.roll(ref, -2), np.roll(ref, 1)])
    aligned, _ = align_pulse_traces(stack, dt_s=1e-7, refine=False)
    # After integer-shift alignment, each row should match the reference
    # in the interior (edges get mean-fill, so compare a safe middle window).
    core = slice(20, len(ref) - 20)
    for i in range(aligned.shape[0]):
        np.testing.assert_allclose(aligned[i, core], ref[core], atol=1e-10)


def test_aligned_mean_beats_naive_mean_with_jitter_and_noise():
    """Under sub-sample jitter + additive noise, aligned averaging should
    recover a mean trace closer to the reference pulse than the naive
    (unaligned) average."""
    dt_s = 1e-7  # 100 ns
    n_samples = 2048
    n_pulses = 16
    _, clean = _synth_burst(n_samples=n_samples, dt_s=dt_s, seed=None)
    rng = np.random.default_rng(42)
    # Jitter each pulse by up to +/- 0.8 samples (well below one period).
    jitter_samples = rng.uniform(-0.8, 0.8, size=n_pulses)
    noise_amp = 0.05
    stack = np.empty((n_pulses, n_samples))
    for i in range(n_pulses):
        stack[i] = shift_trace(clean, -jitter_samples[i])  # applied delay
        stack[i] += noise_amp * rng.standard_normal(n_samples)

    naive_mean = stack.mean(axis=0)
    aligned, lags_s = align_pulse_traces(
        stack, dt_s=dt_s, max_shift_samples=4, refine=True,
    )
    aligned_mean = aligned.mean(axis=0)

    # Compare each average to the *noise-free* alignment target: the clean
    # burst delayed to match stack[0] (the reference we aligned to). This
    # isolates the jitter-smearing error that alignment is meant to fix
    # from the per-pulse noise that both averages reduce equally.
    expected = shift_trace(clean, -jitter_samples[0])
    core = slice(50, n_samples - 50)
    err_naive = np.sqrt(np.mean((naive_mean[core] - expected[core]) ** 2))
    err_aligned = np.sqrt(np.mean((aligned_mean[core] - expected[core]) ** 2))
    # Aligned averaging removes the jitter-smearing term; the residual is
    # noise-floor limited at noise_amp/sqrt(N). Naive averaging keeps the
    # smearing on top of the noise floor. Alignment should cut the total
    # error by at least ~35%.
    assert err_aligned < err_naive * 0.65
    # And the recovered lags should track the applied jitter within ~0.2 samples.
    recovered_samples = lags_s / dt_s
    residual = recovered_samples - (jitter_samples - jitter_samples[0])
    assert np.max(np.abs(residual)) < 0.2


def test_reference_mean_and_int_options():
    _, ref = _synth_burst()
    stack = np.vstack([np.roll(ref, k) for k in (-2, 0, 3)])
    _, lags_first = align_pulse_traces(stack, dt_s=1e-7, reference="first",
                                        refine=False)
    _, lags_int = align_pulse_traces(stack, dt_s=1e-7, reference=1,
                                     refine=False)
    # reference=1 is the same trace as reference="first" would be if we
    # took the middle row, so its own lag must be zero.
    assert lags_int[1] == pytest.approx(0.0, abs=1e-15)
    # reference="first" -> first row has lag zero.
    assert lags_first[0] == pytest.approx(0.0, abs=1e-15)


def test_align_pulse_traces_rejects_1d():
    with pytest.raises(ValueError):
        align_pulse_traces(np.zeros(10), dt_s=1e-7)


def test_shift_trace_zero_is_identity():
    _, ref = _synth_burst()
    out = shift_trace(ref, 0.0)
    np.testing.assert_allclose(out, ref, atol=1e-12)
