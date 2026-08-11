"""Tests for :meth:`VerificationTank.run_averaged_sweep`.

We can't instantiate a real :class:`VerificationTank` without hardware,
so we test the averaging helper by binding it as an unbound method to a
lightweight stub that owns the minimum surface it touches: a
``hydrophone_channel`` attribute, a ``scope.enabled_channels`` list, and
a fake ``run_rapid_sweep`` that returns pre-cooked chunks.
"""
from __future__ import annotations

import types

import numpy as np
import pytest

from openlifu_verification.pulse_align import shift_trace
from openlifu_verification.verificationtank import VerificationTank


def _burst(n_samples=512, dt_s=1e-7, f_hz=400e3, amp=1.0,
           t0_s=10e-6, duration_s=30e-6):
    t = np.arange(n_samples) * dt_s
    env = np.zeros_like(t)
    mask = (t >= t0_s) & (t < t0_s + duration_s)
    tau = (t[mask] - t0_s) / duration_s
    env[mask] = np.sin(np.pi * tau) ** 2
    return t, amp * env * np.sin(2 * np.pi * f_hz * (t - t0_s))


class _Scope:
    def __init__(self, channels):
        self.enabled_channels = list(channels)


class _StubTank:
    """Minimum stub for :meth:`VerificationTank.run_averaged_sweep`.

    ``per_repeat_generator(point, repeat_idx) -> dict`` supplies the
    per-repeat capture (same layout as ``run_rapid_sweep`` output).
    ``fail_idx`` (set of ``(point_idx, repeat_idx)``) forces those
    repeats to appear as scope-timeout misses (``None``).
    """

    def __init__(self, per_repeat_generator, channels=("A", "B"),
                 fail_idx=frozenset()):
        self.hydrophone_channel = "A"
        self.scope = _Scope(channels)
        self._gen = per_repeat_generator
        self._fail_idx = fail_idx
        self.apply_calls = []

    # Bound copies of the real methods we want under test.
    run_averaged_sweep = VerificationTank.run_averaged_sweep

    def run_rapid_sweep(self, *, points, apply_point, time_start_s,
                        time_stop_s, sampling_interval_ns, chunk_size=None,
                        timeout_s=None, progress=None, progress_label=None):
        # Mirror the real method: call apply_point on every entry and
        # emit one output + timings dict per entry.
        outputs = []
        timings = []
        for pt in points:
            apply_point(pt)
            # In n_averages>1 the outer wrapper expands to (real_pt, r)
            # tuples; in n_averages==1 it passes points through as-is.
            if isinstance(pt, tuple) and len(pt) == 2 and isinstance(pt[1], int):
                real_pt, repeat_idx = pt
            else:
                real_pt, repeat_idx = pt, 0
            outputs.append(None if (real_pt, repeat_idx) in self._fail_idx
                           else self._gen(real_pt, repeat_idx))
            timings.append({
                "apply_s": 0.0,
                "trigger_s": 0.0,
                "iter_total_s": 0.0,
                "arm_s": 0.0,
                "xfer_s": 0.0,
                "captured": outputs[-1] is not None,
                "chunk_index": 0,
            })
        return outputs, timings


def _output(t_axis, dt_s, hydro, aux=None):
    d = {
        "time": t_axis,
        "sampling_interval_ns": dt_s * 1e9,
        "time_start_s": 0.0,
        "time_stop_s": t_axis[-1] * 1e-9,
        "overflow": 0,
        "A": hydro,
    }
    if aux is not None:
        d["B"] = aux
    return d


def test_n_averages_1_matches_run_rapid_sweep_output_shape():
    dt_s = 1e-7
    t_axis, ref = _burst(dt_s=dt_s)
    t_ns = t_axis * 1e9

    def gen(point, r):
        return _output(t_ns, dt_s, ref.copy(), np.zeros_like(ref))

    tank = _StubTank(gen)
    outputs, timings, averaging = tank.run_averaged_sweep(
        points=[(0.0,), (1.0,), (2.0,)],
        apply_point=lambda p: tank.apply_calls.append(p),
        time_start_s=0.0, time_stop_s=1e-4,
        sampling_interval_ns=100,
        n_averages=1,
    )
    assert len(outputs) == 3
    assert len(timings) == 3
    assert len(averaging) == 3
    for entry in averaging:
        assert entry["n_averages"] == 1
    for out in outputs:
        assert out["A"].shape == ref.shape


def test_apply_point_called_once_per_group():
    dt_s = 1e-7
    t_axis, ref = _burst(dt_s=dt_s)
    t_ns = t_axis * 1e9

    def gen(point, r):
        return _output(t_ns, dt_s, ref.copy())

    calls = []

    def apply_pt(pt):
        calls.append(pt)

    tank = _StubTank(gen, channels=("A",))
    tank.run_averaged_sweep(
        points=[10.0, 20.0, 30.0],
        apply_point=apply_pt,
        time_start_s=0.0, time_stop_s=1e-4,
        sampling_interval_ns=100,
        n_averages=5,
    )
    # apply_point must be called exactly once per real point regardless
    # of how many repeats are captured.
    assert calls == [10.0, 20.0, 30.0]


def test_averaging_produces_folded_shape_and_metadata():
    dt_s = 1e-7
    t_axis, ref = _burst(dt_s=dt_s)
    t_ns = t_axis * 1e9

    def gen(point, r):
        return _output(t_ns, dt_s, ref.copy())

    tank = _StubTank(gen, channels=("A",))
    outputs, timings, averaging = tank.run_averaged_sweep(
        points=[0.0, 1.0],
        apply_point=lambda p: None,
        time_start_s=0.0, time_stop_s=1e-4,
        sampling_interval_ns=100,
        n_averages=4,
    )
    # One entry per input point (not per repeat).
    assert len(outputs) == 2
    assert len(averaging) == 2
    for entry in averaging:
        assert entry["n_averages"] == 4
        assert entry["n_captured"] == 4
        assert entry["lags_s"].shape == (4,)
    # Traces should still be 1-D per channel (averaged).
    assert outputs[0]["A"].shape == ref.shape


def test_alignment_reduces_smearing_from_subsample_jitter():
    dt_s = 1e-7
    n = 1024
    t_axis = np.arange(n) * dt_s
    t_ns = t_axis * 1e9
    _, clean = _burst(n_samples=n, dt_s=dt_s)
    rng = np.random.default_rng(2026)
    jitter = rng.uniform(-0.7, 0.7, size=6)
    noise_amp = 0.02

    def gen(point, r):
        # Distinct sub-sample delay per repeat, same at every point.
        trace = shift_trace(clean, -jitter[r])
        trace = trace + noise_amp * rng.standard_normal(n)
        return _output(t_ns, dt_s, trace)

    tank_aligned = _StubTank(gen, channels=("A",))
    out_aligned, _, avg_meta = tank_aligned.run_averaged_sweep(
        points=[0.0],
        apply_point=lambda p: None,
        time_start_s=0.0, time_stop_s=1e-4,
        sampling_interval_ns=100,
        n_averages=6, align=True, align_max_shift_samples=4,
    )

    # Reset RNG so the "no align" path sees the same noise realization.
    rng2 = np.random.default_rng(2026)
    _ = rng2.uniform(-0.7, 0.7, size=6)  # skip past jitter draws

    def gen_noalign(point, r):
        trace = shift_trace(clean, -jitter[r])
        trace = trace + noise_amp * rng2.standard_normal(n)
        return _output(t_ns, dt_s, trace)

    tank_naive = _StubTank(gen_noalign, channels=("A",))
    out_naive, _, _ = tank_naive.run_averaged_sweep(
        points=[0.0],
        apply_point=lambda p: None,
        time_start_s=0.0, time_stop_s=1e-4,
        sampling_interval_ns=100,
        n_averages=6, align=False,
    )

    expected = shift_trace(clean, -jitter[0])  # aligned reference is repeat 0
    core = slice(30, n - 30)
    err_aligned = np.sqrt(np.mean((out_aligned[0]["A"][core] - expected[core]) ** 2))
    err_naive = np.sqrt(np.mean((out_naive[0]["A"][core] - expected[core]) ** 2))
    assert err_aligned < err_naive * 0.75
    # Recovered lags should track applied jitter (relative to repeat 0).
    recovered = avg_meta[0]["lags_s"] / dt_s
    assert np.max(np.abs(recovered - (jitter - jitter[0]))) < 0.25


def test_second_channel_uses_same_lags_as_hydrophone():
    """Non-hydrophone channels must be shifted by the same per-repeat lag
    so they stay coherent with the hydrophone."""
    dt_s = 1e-7
    n = 512
    t_axis = np.arange(n) * dt_s
    t_ns = t_axis * 1e9
    _, hyd_clean = _burst(n_samples=n, dt_s=dt_s)
    # A sharp sync channel (different signal shape, same jitter).
    aux_clean = np.zeros(n)
    aux_clean[120:125] = 1.0

    jitter = np.array([0.0, 0.6, -0.4])

    def gen(point, r):
        return _output(
            t_ns, dt_s,
            shift_trace(hyd_clean, -jitter[r]),
            aux=shift_trace(aux_clean, -jitter[r]),
        )

    tank = _StubTank(gen, channels=("A", "B"))
    outputs, _, _ = tank.run_averaged_sweep(
        points=[0.0],
        apply_point=lambda p: None,
        time_start_s=0.0, time_stop_s=1e-4,
        sampling_interval_ns=100,
        n_averages=3, align=True, align_max_shift_samples=3,
    )
    # If the aux channel had NOT been coherently aligned, three copies
    # of a narrow spike offset by ~0.6 and -0.4 samples would smear
    # and reduce peak amplitude; with coherent alignment the averaged
    # aux peak should stay close to 1.0.
    peak = float(outputs[0]["B"].max())
    assert peak > 0.9


def test_partial_capture_failure_is_tolerated():
    dt_s = 1e-7
    t_axis, ref = _burst(dt_s=dt_s)
    t_ns = t_axis * 1e9

    def gen(point, r):
        return _output(t_ns, dt_s, ref.copy())

    # Fail one repeat of point 1 entirely -- the remaining ones should
    # still get averaged and the output should be populated.
    tank = _StubTank(gen, channels=("A",),
                     fail_idx=frozenset({(1.0, 1)}))
    outputs, timings, averaging = tank.run_averaged_sweep(
        points=[0.0, 1.0],
        apply_point=lambda p: None,
        time_start_s=0.0, time_stop_s=1e-4,
        sampling_interval_ns=100,
        n_averages=3,
    )
    assert outputs[0] is not None
    assert outputs[1] is not None
    assert averaging[1]["n_captured"] == 2
    assert timings[1]["captured"] is True


def test_total_failure_leaves_output_none():
    dt_s = 1e-7
    t_axis, ref = _burst(dt_s=dt_s)
    t_ns = t_axis * 1e9

    def gen(point, r):
        return _output(t_ns, dt_s, ref.copy())

    tank = _StubTank(
        gen, channels=("A",),
        fail_idx=frozenset({(5.0, 0), (5.0, 1)}),
    )
    outputs, timings, averaging = tank.run_averaged_sweep(
        points=[5.0],
        apply_point=lambda p: None,
        time_start_s=0.0, time_stop_s=1e-4,
        sampling_interval_ns=100,
        n_averages=2,
    )
    assert outputs[0] is None
    assert averaging[0]["n_captured"] == 0
    assert timings[0]["captured"] is False


def test_invalid_n_averages_rejected():
    tank = _StubTank(lambda p, r: None)
    with pytest.raises(ValueError):
        tank.run_averaged_sweep(
            points=[0.0], apply_point=lambda p: None,
            time_start_s=0.0, time_stop_s=1e-4,
            sampling_interval_ns=100, n_averages=0,
        )


def test_averaging_reduces_random_noise_rms():
    """Repeat-averaging with align=False (no jitter, just white noise)
    should reduce the per-sample noise floor by ~sqrt(N)."""
    dt_s = 1e-7
    n = 256
    t_axis = np.arange(n) * dt_s
    t_ns = t_axis * 1e9
    rng = np.random.default_rng(7)
    N = 16

    def gen(point, r):
        # Pure noise, no signal, so any RMS after averaging is measurement noise.
        return _output(t_ns, dt_s, rng.standard_normal(n))

    tank = _StubTank(gen, channels=("A",))
    outputs, _, _ = tank.run_averaged_sweep(
        points=[0.0],
        apply_point=lambda p: None,
        time_start_s=0.0, time_stop_s=1e-4,
        sampling_interval_ns=100,
        n_averages=N, align=False,
    )
    avg_noise_rms = float(np.sqrt(np.mean(outputs[0]["A"] ** 2)))
    # Expected 1/sqrt(N); allow generous slack (finite N + edge effects).
    assert avg_noise_rms < 0.5  # << 1.0 (single-shot RMS)
