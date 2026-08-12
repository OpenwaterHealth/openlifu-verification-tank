"""Tests for the 2-D gradient-ascent peak search.

Runs against a synthetic gaussian measurement so the tests are fast
and don't need scope hardware.
"""
from __future__ import annotations

import numpy as np
import pytest

from openlifu_verification.search import gradient_search, _quadratic_subsample_frac


def _gaussian_source(*, peak_xy, sigma=1.5, noise=0.0, seed=0):
    """Build a synthetic ``measure_fn`` around a known peak."""
    rng = np.random.default_rng(seed)

    def measure(x, y):
        r2 = (x - peak_xy[0]) ** 2 + (y - peak_xy[1]) ** 2
        rms = 100.0 * np.exp(-0.5 * r2 / sigma**2)
        if noise:
            rms += rng.normal(0, noise)
        return {
            "t": np.arange(4),
            "trace": np.zeros(4),
            "rms": float(rms),
            "vpp": float(rms * 2),
            "units": "Pa",
        }

    return measure


def test_returns_expected_keys():
    """Regression guard against silently renaming the return schema."""
    res = gradient_search(
        _gaussian_source(peak_xy=(0.5, -0.5)),
        x0=0.0, y0=0.0, initial_step=0.5, tol=0.05, max_iter=20,
    )
    for k in ("best_x", "best_y", "best_rms",
              "center_x", "center_y", "center_rms",
              "units", "converged", "iterations", "evaluations",
              "xs", "ys", "rms_values"):
        assert k in res, f"missing key {k!r} in gradient_search result"


def test_reported_best_equals_max_measured():
    """Regression test for the bug where ``best_rms`` reported the
    search-center RMS instead of the highest measurement ever seen.
    The colorbar max on the live plot must match the reported best."""
    res = gradient_search(
        _gaussian_source(peak_xy=(1.2, -0.8), noise=0.15, seed=42),
        x0=0.0, y0=0.0, initial_step=0.5, tol=0.02,
        hysteresis=0.01, max_iter=40,
    )
    assert res["best_rms"] == max(res["rms_values"])
    # And ``best_x, best_y`` must be the exact point that produced it.
    peak_idx = int(np.argmax(res["rms_values"]))
    assert res["best_x"] == res["xs"][peak_idx]
    assert res["best_y"] == res["ys"][peak_idx]


def test_finds_gaussian_peak_within_tolerance():
    res = gradient_search(
        _gaussian_source(peak_xy=(1.2, -0.8)),
        x0=0.0, y0=0.0, initial_step=0.5, tol=0.02, max_iter=40,
    )
    assert res["converged"], f"expected convergence, got {res}"
    dist = np.hypot(res["best_x"] - 1.2, res["best_y"] + 0.8)
    assert dist < 0.1, f"peak {dist:.3f} mm from truth"


def test_converges_from_off_axis_start():
    """Search should still find the peak when the start is >1 mm off."""
    res = gradient_search(
        _gaussian_source(peak_xy=(2.5, -1.5), sigma=2.0),
        x0=0.0, y0=0.0, initial_step=0.5, tol=0.02, max_iter=60,
    )
    dist = np.hypot(res["best_x"] - 2.5, res["best_y"] + 1.5)
    assert dist < 0.2


def test_flat_field_stays_put_and_converges():
    """On a perfectly flat field, all probes tie the center; the
    algorithm should just halve step until it converges without
    wandering."""
    def flat(x, y):
        return {"t": np.arange(4), "trace": np.zeros(4),
                "rms": 42.0, "vpp": 84.0, "units": "Pa"}

    res = gradient_search(
        flat, x0=0.5, y0=-0.25, initial_step=0.5, tol=0.02, max_iter=20,
    )
    assert res["converged"]
    assert res["best_rms"] == 42.0
    # No wandering off origin.
    assert abs(res["center_x"] - 0.5) < 1e-9
    assert abs(res["center_y"] + 0.25) < 1e-9


def test_scope_timeout_at_start_raises():
    """``measure_fn`` returning ``None`` on the very first call is
    fatal because we can't seed the ``units`` field."""
    with pytest.raises(RuntimeError):
        gradient_search(lambda x, y: None,
                        x0=0.0, y0=0.0, initial_step=0.5, tol=0.05)


def test_progress_callback_receives_global_peak():
    """The ``best_*`` kwargs handed to the ``on_progress`` callback
    must always describe the running global peak, so live figures
    can highlight it correctly."""
    seen_bests: list[tuple[float, float, float]] = []

    def measure(x, y):
        # Synthetic: value increases the further we walk in +x.
        rms = 10.0 + x
        return {"t": np.arange(2), "trace": np.zeros(2),
                "rms": float(rms), "vpp": float(rms * 2), "units": "Pa"}

    def on_progress(**kw):
        seen_bests.append((kw["best_x"], kw["best_y"], kw["best_rms"]))

    gradient_search(measure, x0=0.0, y0=0.0, initial_step=0.5,
                    tol=0.05, max_iter=10, on_progress=on_progress)

    # ``best_rms`` seen by the callback must be non-decreasing.
    rmses = [b[2] for b in seen_bests]
    assert all(b >= a for a, b in zip(rmses, rmses[1:])), (
        f"best_rms went backwards in on_progress: {rmses}"
    )


def test_returned_measurement_arrays_are_consistent():
    """``xs``, ``ys``, ``rms_values`` should all be the same length
    and equal to ``evaluations``."""
    res = gradient_search(
        _gaussian_source(peak_xy=(1.0, 1.0)),
        x0=0.0, y0=0.0, initial_step=0.5, tol=0.05, max_iter=20,
    )
    n = res["evaluations"]
    assert len(res["xs"]) == n
    assert len(res["ys"]) == n
    assert len(res["rms_values"]) == n


def test_max_iter_stops_the_search():
    """Passing ``max_iter=1`` must return quickly without hanging."""
    res = gradient_search(
        _gaussian_source(peak_xy=(0.0, 0.0), noise=0.0),
        x0=10.0, y0=10.0, initial_step=0.5, tol=0.001, max_iter=1,
    )
    assert res["iterations"] <= 1


def test_quadratic_subsample_frac_concave_down_interior():
    """Symmetric samples -> peak sits on the center (frac ~= 0)."""
    assert _quadratic_subsample_frac(0.5, 1.0, 0.5) == pytest.approx(0.0, abs=1e-12)


def test_quadratic_subsample_frac_biased_toward_higher_side():
    """When the +side sample is higher than the -side, the fit vertex
    shifts toward +1 (still in (-1, +1))."""
    frac = _quadratic_subsample_frac(0.5, 1.0, 0.8)
    assert frac is not None
    assert 0.0 < frac < 1.0


def test_quadratic_subsample_frac_rejects_non_concave():
    # Concave-up parabola: no local max.
    assert _quadratic_subsample_frac(1.0, 0.5, 1.0) is None
    # Flat: denom == 0.
    assert _quadratic_subsample_frac(1.0, 1.0, 1.0) is None


def test_quadratic_subsample_frac_recovers_gaussian_center(tmp_path=None):
    """A Gaussian sampled at ``[-h, 0, +h]`` off-center should give a
    quadratic vertex close to the true offset (for small ``h/sigma``)."""
    sigma = 1.5
    true_offset = 0.3        # mm
    h = 0.4                  # probe half-spacing (mm)
    center = 0.0             # sampled at 0

    def g(x):
        return np.exp(-0.5 * (x - true_offset) ** 2 / sigma ** 2)

    frac = _quadratic_subsample_frac(g(center - h), g(center), g(center + h))
    assert frac is not None
    # Fractional offset in units of h -> multiply by h to compare in mm.
    assert frac * h == pytest.approx(true_offset, abs=0.05)


def test_gaussian_search_uses_quadratic_refinement(monkeypatch):
    """The final refinement step near the peak should come from the
    quadratic fit on both axes, not just probe-halving. We assert
    via the emitted info_extra string on the ``on_progress`` callback."""
    seen_infos: list[str] = []

    def on_progress(**kw):
        seen_infos.append(kw.get("info_extra", ""))

    res = gradient_search(
        _gaussian_source(peak_xy=(0.15, -0.10), sigma=1.5),
        x0=0.0, y0=0.0, initial_step=0.5, tol=0.02, max_iter=40,
        on_progress=on_progress,
    )
    assert res["converged"]
    assert any("u=quadratic v=quadratic" in s for s in seen_infos), (
        "expected at least one iteration with quadratic refinement "
        f"on both axes; saw: {seen_infos}"
    )
    dist = np.hypot(res["best_x"] - 0.15, res["best_y"] + 0.10)
    assert dist < 0.05


def test_shift_shift_diagonal_is_weighted_by_relative_gain():
    """When both axes want a shift step but the peak is much
    stronger in one direction, the combined step should skew toward
    the stronger axis rather than moving 45\u00b0 diagonally."""

    # Peak far in +x, only slightly in +y.
    def measure(x, y):
        rms = 100.0 * np.exp(-0.5 * ((x - 2.0) ** 2 + (y - 0.1) ** 2) / 1.5**2)
        return {"t": np.arange(2), "trace": np.zeros(2),
                "rms": float(rms), "vpp": float(rms * 2), "units": "Pa"}

    positions: list[tuple[float, float]] = []

    def on_progress(**kw):
        positions.append((kw["x"], kw["y"]))

    gradient_search(
        measure, x0=0.0, y0=0.0, initial_step=0.5, tol=0.02,
        max_iter=1, on_progress=on_progress, rotate_basis=False,
    )
    # positions[0] = start (0, 0)
    # positions[1..4] = ±u, ±v probes at h=0.5
    # positions[5] = the "probes complete" summary event (x, y = center)
    # positions[6] = the refined move that closes iter 1
    assert len(positions) >= 7
    refined_x, refined_y = positions[6]
    # The refined move must skew far more toward +x than +y since
    # +x is the strong-gain direction. The unweighted algorithm
    # would have gone to (0.5, 0.5); the weighted one should keep
    # dx close to h and dy noticeably smaller.
    assert refined_x > 0.35
    assert refined_y < 0.25
    assert refined_x > 2 * refined_y
