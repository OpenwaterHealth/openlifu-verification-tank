"""2-D gradient-ascent peak search for RMS pressure.

Implements a rotating-basis, central-difference gradient ascent with
backtracking. Kept independent of ``VerificationTank`` so it can be
tested against mocks; ``VerificationTank.find_peak`` is the public
entry point that wires this up to the scope and hydrophone.

Also exposes optional live matplotlib helpers that are only imported
when the caller asks for them (``plot=True``).
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _quadratic_subsample_frac(y_minus: float, y_center: float,
                              y_plus: float) -> Optional[float]:
    """Fractional offset (in units of the sample spacing) of the peak of
    a quadratic fit through three equally-spaced samples ``[y_minus,
    y_center, y_plus]`` located at ``[-1, 0, +1]``.

    Returns the offset in ``(-1, +1)`` when the quadratic is
    concave-down with an interior maximum; ``None`` otherwise (flat
    axis, concave-up, or the estimated peak lies outside the sampled
    interval so the fit shouldn't be trusted).
    """
    denom = y_minus - 2.0 * y_center + y_plus
    if denom >= 0:
        # Not concave-down (flat or upward-facing) -> no interior peak.
        return None
    frac = (y_minus - y_plus) / (2.0 * denom)
    if not np.isfinite(frac) or abs(frac) >= 1.0:
        return None
    return float(frac)


def gradient_search(
    measure_fn: Callable[[float, float], Optional[dict]],
    *,
    x0: float,
    y0: float,
    initial_step: float = 0.5,
    tol: float = 0.02,
    max_iter: int = 40,
    hysteresis: float = 0.01,
    probe_scale: float = 1.0,
    min_line_step_scale: float = 0.05,
    min_step: float = 0.2,
    max_polish_iter: int = 6,
    rotate_basis: bool = True,
    on_progress: Optional[Callable[..., None]] = None,
):
    """Walk toward the RMS peak from ``(x0, y0)``.

    Direct-search algorithm (no gradient/line-search anywhere): on
    each iteration, probe ``\u00b1u`` and ``\u00b1v`` at distance
    ``h = probe_scale * step`` around the current center. For each
    axis independently:

    * If one of the neighbors beats the center, shift by ``\u00b1h``
      toward that neighbor.
    * Else (center dominates both neighbors on that axis), fit a
      quadratic through the three samples and shift by the
      fractional vertex offset (subsample peak estimate).
    * Else (all three equal), don't shift on that axis.

    The per-axis shifts are combined in the local basis and the
    center moves to the refined position. If the shift was nonzero
    and ``rotate_basis`` is set, the basis is rotated so ``u``
    aligns with the shift direction (a diagonal shift of one probe
    step on each axis rotates the basis by 45\u00b0; a pure quadratic
    refinement rotates by an arbitrary angle). Once both axes are
    in the subsample regime (center dominates on both axes) the
    probe grid shrinks by half \u2014 but only down to ``min_step``.
    Below ``min_step`` the algorithm switches to a **symmetry
    polish** phase: the probe spacing stays fixed at ``min_step``
    and each iteration moves the center to the estimated
    symmetry point of the roll-off. This trades noisy micro-peak
    hunting for a stable "center the mountain" objective.

    Convergence: both axes used the quadratic branch **and** the
    combined shift magnitude is smaller than ``tol``. During the
    polish phase we additionally cap iteration count at
    ``max_polish_iter`` to prevent noise-driven oscillation.

    Every measurement is compared against a *global peak* tracker
    so the returned ``best_x, best_y, best_rms`` are the
    highest-RMS point ever measured (not the point the walk happens
    to settle on).

    Args:
        measure_fn: ``measure_fn(x, y)`` fires one pulse at ``(x, y)``
            and returns a dict with keys ``{"t", "trace", "rms",
            "vpp", "units"}`` (matching
            :meth:`VerificationTank.measure_pressure`) or ``None`` on
            scope timeout.
        x0, y0: Starting position (mm).
        initial_step: Initial probe spacing control (mm).
        tol: Convergence tolerance (mm).
        max_iter: Maximum iterations.
        hysteresis: Currently unused (kept for signature stability).
        probe_scale: Probe distance as a fraction of ``step``. Must
            be > 0.
        min_line_step_scale: Currently unused (kept for signature
            stability).
        min_step: Minimum probe spacing (mm) below which we stop
            halving. This is the spatial scale over which we
            require the mountain to roll off symmetrically. On a
            noisy top, setting this too small makes the parabola
            fit dominated by RMS noise; the default of 0.2 mm
            (200 \u00b5m) is roughly the transducer's spot
            radius \u2014 large enough that the side probes see a
            meaningful pressure drop. Set to 0 or a very small
            value to recover the old "always shrink" behavior.
        max_polish_iter: Cap on how many symmetry-polish iterations
            (once ``step == min_step``) are allowed before we
            declare convergence. Prevents endless jitter around a
            noisy top.
        rotate_basis: If ``True``, rotate the probe basis so ``u``
            aligns with each accepted shift direction.
        on_progress: Optional callback invoked after every
            measurement, with kwargs ``(meas, x, y, r, xs, ys,
            rms_values, best_x, best_y, best_rms, units, iteration,
            step, evaluations, converged, done, info_extra)``. The
            ``best_*`` fields carry the *global peak* seen so far,
            so the live-plot "best" marker sits on the highest-RMS
            measurement.

    Returns:
        Dict with:
          - ``best_x, best_y, best_rms``: global peak seen (the
            highest RMS across every measurement made).
          - ``center_x, center_y, center_rms``: final search-center
            location and its RMS (where the walk settled).
          - ``units, converged, iterations, evaluations``
          - ``xs, ys, rms_values``: every measurement in order.
    """
    if probe_scale <= 0:
        raise ValueError("probe_scale must be > 0")
    del hysteresis, min_line_step_scale  # accepted but no longer used

    xs: list[float] = []
    ys: list[float] = []
    rms_values: list[float] = []
    # Trajectory of best-guess (search-center) positions across
    # iterations. Starts with the initial guess and grows every time
    # the center moves. Exposed to the ``on_progress`` callback so
    # the live figure can draw the convergence path.
    center_history: list[tuple[float, float]] = []

    # Global peak seen so far. Independent of the *search center*
    # (``center_x, center_y, center_rms``) so a probe that measures
    # higher than the center still gets remembered and reported.
    # confusion: the returned peak = highest RMS ever measured.
    peak_x: float
    peak_y: float
    peak_rms: float = float("-inf")
    peak_meas = None

    def _measure(x, y):
        nonlocal peak_x, peak_y, peak_rms, peak_meas
        meas = measure_fn(x, y)
        if meas is None:
            return None, float("-inf")
        r = meas["rms"]
        if r > peak_rms:
            peak_x, peak_y, peak_rms, peak_meas = x, y, r, meas
        return meas, r

    def _emit(meas, x, y, r, *,
              iteration, step, converged=False, done=False,
              iter_end=False, info_extra=""):
        xs.append(x)
        ys.append(y)
        rms_values.append(r)
        if on_progress is not None:
            # Report the global peak (not the search center) as
            # ``best_*`` so the live figure's "best" marker sits at
            # the highest-RMS measurement instead of wherever the
            # walk happens to be centered.
            on_progress(
                meas=meas, x=x, y=y, r=r,
                xs=xs, ys=ys, rms_values=rms_values,
                best_x=peak_x, best_y=peak_y, best_rms=peak_rms,
                units=units,
                iteration=iteration, step=step,
                evaluations=len(rms_values),
                converged=converged, done=done,
                iter_end=iter_end,
                centers=list(center_history),
                info_extra=info_extra,
            )

    # Seed the peak with the starting point so ``_measure`` can safely
    # compare against it on subsequent calls.
    peak_x, peak_y = x0, y0
    center_meas, center_rms = _measure(x0, y0)
    if center_meas is None:
        raise RuntimeError("Scope timed out on initial measurement.")
    units = center_meas["units"]
    center_x, center_y = x0, y0
    center_history.append((center_x, center_y))
    _emit(center_meas, x0, y0, center_rms,
          iteration=0, step=initial_step,
          info_extra=f"start (RMS={center_rms:.4g} {units})")

    # Local probe basis. Starts axis-aligned; rotates to follow the
    # last accepted shift direction (so a diagonal move rotates the
    # probes 45\u00b0 for the next iteration, and a quadratic-only
    # refinement rotates by an arbitrary sub-degree angle).
    u = np.array([1.0, 0.0])
    v = np.array([0.0, 1.0])

    step = float(initial_step)
    # Clamp min_step to be at most initial_step (otherwise we'd
    # start at the floor, which is fine \u2014 but never let it exceed
    # the starting spacing).
    min_step = max(0.0, min(float(min_step), step))
    iteration = 0
    # Count of iterations spent at ``step == min_step`` (the
    # symmetry-polish phase). Capped at ``max_polish_iter``.
    polish_count = 0
    converged = False

    while iteration < max_iter:
        iteration += 1
        h = step * probe_scale
        probes = [
            ("+u", center_x + h * u[0], center_y + h * u[1]),
            ("-u", center_x - h * u[0], center_y - h * u[1]),
            ("+v", center_x + h * v[0], center_y + h * v[1]),
            ("-v", center_x - h * v[0], center_y - h * v[1]),
        ]
        rvals: dict[str, float] = {}
        # Track the best probe. If it beats the accepted line-search
        # trial (or the line search doesn't accept anything), we jump
        # to the probe instead of leaving free RMS on the table.
        best_probe = None  # (label, x, y, r, meas)
        for label, px, py in probes:
            meas, r = _measure(px, py)
            if meas is None:
                continue
            _emit(meas, px, py, r,
                  iteration=iteration, step=step,
                  info_extra=f"probe {label}  h={h:.4f} mm")
            rvals[label] = r
            if best_probe is None or r > best_probe[3]:
                best_probe = (label, px, py, r, meas)
        if len(rvals) < 4:
            step /= 2.0
            logger.info(
                "iter %d: incomplete probe set (scope timeouts) \u2192 halve step to %.4f",
                iteration, step,
            )
            if step < tol:
                converged = True
                break
            continue

        # Publish the probe triplets so the live figure can draw
        # ``P vs u'`` and ``P vs v'`` for this iteration. Quadratic
        # fractions are precomputed here purely for visualization;
        # the actual refinement logic recomputes them below.
        probe_u_rms = (rvals["-u"], center_rms, rvals["+u"])
        probe_v_rms = (rvals["-v"], center_rms, rvals["+v"])
        vis_frac_u = (
            _quadratic_subsample_frac(*probe_u_rms)
            if rvals["+u"] < center_rms and rvals["-u"] < center_rms
            else None
        )
        vis_frac_v = (
            _quadratic_subsample_frac(*probe_v_rms)
            if rvals["+v"] < center_rms and rvals["-v"] < center_rms
            else None
        )
        if on_progress is not None:
            on_progress(
                meas=None, x=center_x, y=center_y, r=center_rms,
                xs=xs, ys=ys, rms_values=rms_values,
                best_x=peak_x, best_y=peak_y, best_rms=peak_rms,
                units=units,
                iteration=iteration, step=step,
                evaluations=len(rms_values),
                converged=False, done=False,
                info_extra=f"probes complete  h={h:.4f} mm",
                probe_h=h,
                probe_u_rms=probe_u_rms,
                probe_v_rms=probe_v_rms,
                quad_frac_u=vis_frac_u,
                quad_frac_v=vis_frac_v,
            )

        # --- Per-axis refinement ---
        # For each axis independently:
        #   * If a neighbor beats the center, we're in "shift" mode
        #     on that axis \u2014 the peak lies beyond the sampled
        #     interval; the raw target is \u00b1h toward the higher
        #     neighbor.
        #   * Elif the center dominates both neighbors, fit a
        #     quadratic through (-h, center, +h) and the raw target
        #     is the fractional vertex offset in (-h, +h).
        #   * Else (flat), no shift.
        #
        # When either axis is in shift mode we don't blindly move
        # by (\u00b1h, \u00b1h). Instead we weight the two axes by
        # their relative "improvement" (how much higher the
        # winning neighbor is above the center) so that when one
        # direction is a much stronger climb we skew toward it
        # rather than moving 45\u00b0 diagonally. We also cap the
        # combined step magnitude to ``h`` \u2014 the user's "step
        # size" \u2014 so a single iteration never jumps more than
        # one probe spacing when we haven't yet bracketed the peak
        # on every axis.
        #
        # Rules:
        #   * Combined shift is applied in the local (u, v) basis.
        #   * Basis rotates to align u with the shift direction so
        #     next iteration's probes point toward the peak.
        #   * The probe grid ``step`` shrinks only when neither
        #     axis is in shift mode (both axes bracket the peak).
        #   * Convergence when both axes are quadratic AND the
        #     combined offset is < ``tol``.
        u_dominates = rvals["+u"] < center_rms and rvals["-u"] < center_rms
        v_dominates = rvals["+v"] < center_rms and rvals["-v"] < center_rms
        frac_u = (
            _quadratic_subsample_frac(rvals["-u"], center_rms, rvals["+u"])
            if u_dominates else None
        )
        frac_v = (
            _quadratic_subsample_frac(rvals["-v"], center_rms, rvals["+v"])
            if v_dominates else None
        )
        # Raw per-axis shift in mm and a positive "gain" magnitude
        # (0 for flat/quadratic-at-peak axes) that we'll use to
        # weight diagonal shift-mode moves.
        if rvals["+u"] > center_rms or rvals["-u"] > center_rms:
            du = h if rvals["+u"] >= rvals["-u"] else -h
            gain_u = max(rvals["+u"], rvals["-u"]) - center_rms
            u_kind = "shift"
        elif frac_u is not None:
            du = frac_u * h
            gain_u = 0.0
            u_kind = "quadratic"
        else:
            du = 0.0
            gain_u = 0.0
            u_kind = "flat"
        if rvals["+v"] > center_rms or rvals["-v"] > center_rms:
            dv = h if rvals["+v"] >= rvals["-v"] else -h
            gain_v = max(rvals["+v"], rvals["-v"]) - center_rms
            v_kind = "shift"
        elif frac_v is not None:
            dv = frac_v * h
            gain_v = 0.0
            v_kind = "quadratic"
        else:
            dv = 0.0
            gain_v = 0.0
            v_kind = "flat"

        # Weighted-diagonal correction. Only kicks in when either
        # axis is still in shift mode; pure-quadratic moves are
        # already the actual estimated peak offsets and should not
        # be scaled.
        weighted = False
        if u_kind == "shift" and v_kind == "shift":
            # Both axes: weight each by its relative improvement
            # so the axis with the steeper climb dominates.
            gain_max = max(gain_u, gain_v)
            if gain_max > 0:
                du *= gain_u / gain_max
                dv *= gain_v / gain_max
                weighted = True
        # If at least one axis is in shift mode, cap the combined
        # step magnitude to h so a single move never jumps more
        # than one probe spacing when we haven't yet bracketed the
        # peak on every axis.
        if u_kind == "shift" or v_kind == "shift":
            mag_raw = float(np.hypot(du, dv))
            if mag_raw > h:
                scale = h / mag_raw
                du *= scale
                dv *= scale
                weighted = True

        offset_mag = float(np.hypot(du, dv))
        moved = False
        if du != 0.0 or dv != 0.0:
            refined_x = center_x + du * u[0] + dv * v[0]
            refined_y = center_y + du * u[1] + dv * v[1]
            refined_meas, refined_rms = _measure(refined_x, refined_y)
            if refined_meas is not None:
                center_x, center_y, center_rms = refined_x, refined_y, refined_rms
                center_history.append((center_x, center_y))
                moved = True
                _emit(refined_meas, refined_x, refined_y, refined_rms,
                      iteration=iteration, step=step,
                      info_extra=(
                          f"refine u={u_kind} v={v_kind}"
                          f"{' (weighted)' if weighted else ''}  "
                          f"du={du:+.4f} dv={dv:+.4f} mm  "
                          f"|d|={offset_mag:.4f}  "
                          f"RMS={refined_rms:.4g} {units}"
                      ))

            # Rotate basis so u aligns with the shift direction.
            # A pure diagonal (h, h) rotates by 45\u00b0; a weighted
            # shift-mode move rotates by whatever the gain ratio
            # produced; a pure (quadratic, quadratic) move rotates
            # by whatever the fractional offsets dictate.
            if rotate_basis:
                shift_world = np.array([du * u[0] + dv * v[0],
                                        du * u[1] + dv * v[1]])
                norm = float(np.linalg.norm(shift_world))
                if norm > 1e-12:
                    u = shift_world / norm
                    v = np.array([-u[1], u[0]])

        logger.info(
            "iter %d: refine u=%s v=%s \u2192 (%.4f, %.4f) mm  "
            "offset=%.4f mm  RMS=%.4g %s  peak=%.4g %s",
            iteration, u_kind, v_kind, center_x, center_y,
            offset_mag, center_rms, units, peak_rms, units,
        )

        # Convergence: both axes are quadratic-refined AND the
        # combined offset is below ``tol``.
        both_quadratic = (u_kind == "quadratic" and v_kind == "quadratic")
        converged_now = both_quadratic and offset_mag < tol
        # Fully-flat corner case: neither axis moved and neither
        # bracketed a peak. Halving step won't help, so just
        # declare success (we have nothing better to do).
        if not converged_now and du == 0.0 and dv == 0.0:
            converged_now = True

        # ---- Shrink / polish decision -------------------------------
        # If we're already at (or below) ``min_step`` we're in the
        # symmetry-polish phase: probe spacing stays fixed at the
        # roll-off distance, and each iteration recenters the
        # mountain rather than hunting for a micro-peak. Cap total
        # polish iterations so noise-driven jitter can't run
        # forever.
        at_floor = step <= min_step + 1e-12
        if at_floor and not converged_now:
            polish_count += 1
            if polish_count >= max_polish_iter:
                converged_now = True
        # Only shrink when we've bracketed a sample peak on both
        # axes AND the subsample-peak estimate lies inside what
        # the halved probe band would still cover. This avoids
        # collapsing ``step`` before we've actually landed on the
        # mountain. Never shrink below ``min_step``.
        shrink_now = (not converged_now
                      and not at_floor
                      and both_quadratic
                      and offset_mag < step / 2.0)
        if shrink_now:
            new_step = step / 2.0
            if new_step < min_step:
                new_step = min_step
            step = new_step
            # Reaching min_step is itself entering polish phase;
            # count this shrink as the first polish iteration so
            # the cap still applies.
            if step <= min_step + 1e-12:
                polish_count = max(polish_count, 1)
            if step < tol:
                converged_now = True

        # End-of-iteration progress emit. Always fires: gives
        # ``find_peak`` a chance to pause between iterations, keeps
        # the trajectory line current, and marks convergence when
        # relevant. If the iteration made no measurement of its own
        # (all axes flat, no probe timeouts) we still push a marker
        # here so the live figure gets refreshed exactly once per
        # iteration.
        if on_progress is not None:
            on_progress(
                meas=None, x=center_x, y=center_y, r=center_rms,
                xs=xs, ys=ys, rms_values=rms_values,
                best_x=peak_x, best_y=peak_y, best_rms=peak_rms,
                units=units,
                iteration=iteration, step=step,
                evaluations=len(rms_values),
                converged=converged_now,
                done=converged_now,
                iter_end=True,
                centers=list(center_history),
                info_extra=("converged" if converged_now
                            else f"iter {iteration} done"),
            )

        if converged_now:
            converged = True
            break

    return {
        # ``best_*`` == global peak seen (not the search center).
        # The search center coordinates are also returned as
        # ``center_*`` for callers that want to inspect where the
        # walk settled.
        "best_x": peak_x,
        "best_y": peak_y,
        "best_rms": peak_rms,
        "center_x": center_x,
        "center_y": center_y,
        "center_rms": center_rms,
        "units": units,
        "converged": converged,
        "iterations": iteration,
        "evaluations": len(rms_values),
        "xs": xs,
        "ys": ys,
        "rms_values": rms_values,
    }


# ----------------------------------------------------------------------
# grid_walk_search: fixed-grid walk + LSQ paraboloid centering
# ----------------------------------------------------------------------
def _fit_paraboloid_vertex(points, *, guard_radius=None):
    """Least-squares fit ``z = A x\u00b2 + B y\u00b2 + C xy + D x + E y + F``
    to ``points`` (iterable of ``(x, y, z)``) and return the vertex
    ``(xv, yv, zv)`` where the gradient vanishes.

    Returns ``None`` if:
      * fewer than 6 points (system is underdetermined),
      * the Hessian is not concave-down (``4AB - C\u00b2 <= 0`` or
        ``A >= 0``),
      * or the vertex lies farther than ``guard_radius`` from the
        centroid of the input points (fit is extrapolating).
    """
    if len(points) < 6:
        return None
    px = np.array([p[0] for p in points], dtype=float)
    py = np.array([p[1] for p in points], dtype=float)
    pz = np.array([p[2] for p in points], dtype=float)
    M = np.column_stack([px * px, py * py, px * py,
                         px, py, np.ones_like(px)])
    try:
        coeffs, *_ = np.linalg.lstsq(M, pz, rcond=None)
    except np.linalg.LinAlgError:
        return None
    a, b, c, d, e, f = (float(v) for v in coeffs)
    det = 4.0 * a * b - c * c
    if det <= 1e-18 or a >= 0.0:
        # Not a concave-down elliptic paraboloid (saddle, plane, or
        # concave-up bowl) \u2014 no interior maximum.
        return None
    # Solve [[2A, C], [C, 2B]] @ [x; y] = -[D; E]
    xv = (c * e - 2.0 * b * d) / det
    yv = (c * d - 2.0 * a * e) / det
    if guard_radius is not None:
        cx = float(px.mean())
        cy = float(py.mean())
        if np.hypot(xv - cx, yv - cy) > guard_radius:
            return None
    zv = (a * xv * xv + b * yv * yv + c * xv * yv
          + d * xv + e * yv + f)
    return xv, yv, zv


def grid_walk_search(
    measure_fn: Callable[[float, float], Optional[dict]],
    *,
    x0: float,
    y0: float,
    step: float = 0.2,
    max_evaluations: int = 50,
    max_iter: int = 40,
    fit_window: int = 1,
    on_progress: Optional[Callable[..., None]] = None,
):
    """Fixed-grid peak search with 2-D paraboloid centering.

    All samples live on an axis-aligned grid with spacing ``step``,
    keyed by integer indices ``(i, j)`` with coordinates
    ``(x0 + i*step, y0 + j*step)``. Once measured, a node is
    cached and never re-visited (so a walk that doubles back is
    free of wall-time cost).

    Algorithm
    ---------
    1. **Seed**: measure origin ``(0, 0)`` plus the 4 cardinal
       neighbors ``(\u00b11, 0)``, ``(0, \u00b11)``. (5 measurements.)
    2. **Cardinal walk**: identify the current best grid node.
       Sample any of its 4 cardinal neighbors we don't have yet.
       If the max moves to one of those neighbors, iterate.
    3. **Diagonal bracket**: once the cardinals are all sampled and
       none exceed the center, sample the 4 diagonals of the best
       node. If any diagonal exceeds the center, resume step 2
       from there. Otherwise the max is a local grid maximum with
       all 8 neighbors lower \u2014 converged.
    4. **Paraboloid fit**: fit ``z = A x\u00b2 + B y\u00b2 + C xy + D x
       + E y + F`` by least squares to all cached samples within
       ``fit_window`` grid nodes of the converged best. Return the
       vertex as ``(center_x, center_y)``. If the fit isn't
       concave-down or the vertex falls outside the window, fall
       back to the grid-max coordinates.

    Compared to :func:`gradient_search`, this trades subsample
    refinement of individual triples for a many-sample LSQ fit,
    which averages down single-sample noise. It also never rotates
    the coordinate system and never re-measures a node, giving a
    predictable sample budget of roughly ``5 + 3*(walk_steps) +
    4`` measurements for a peak ``walk_steps`` grid-steps away.

    Args:
        measure_fn: ``measure_fn(x, y) -> {"rms", "vpp", "trace",
            "t", "units"} | None``, matching
            :meth:`VerificationTank.measure_pressure`.
        x0, y0: Origin of the grid (mm).
        step: Grid spacing (mm). Also the side length of the fit
            window's central cell. Choose so a probe at ``\u00b1step``
            from a peak sees a clearly detectable roll-off vs the
            per-shot noise level. Default 0.2 mm (200 \u00b5m).
        max_evaluations: Cap on total new measurements before
            giving up. Cached re-visits don't count. Default 50.
        max_iter: Cap on walk iterations (independent of
            ``max_evaluations``). Default 40.
        fit_window: Radius (in grid nodes) of the neighborhood
            around the converged best used for the paraboloid fit.
            ``fit_window=1`` \u2192 3\u00d73 (up to 9 points),
            ``fit_window=2`` \u2192 5\u00d75 (up to 25 points).
            Default 1.
        on_progress: Callback fired after every new measurement
            and once at every iteration boundary. See
            :func:`gradient_search` for the kwarg contract.

    Returns:
        Dict with the same keys as :func:`gradient_search`:
        ``best_x, best_y, best_rms`` (global-max sample),
        ``center_x, center_y, center_rms`` (paraboloid vertex),
        ``units, converged, iterations, evaluations, xs, ys,
        rms_values``.
    """
    if step <= 0:
        raise ValueError("step must be > 0")
    if fit_window < 1:
        raise ValueError("fit_window must be >= 1")

    grid_rms: dict[tuple[int, int], float] = {}
    grid_meas: dict[tuple[int, int], dict] = {}
    xs: list[float] = []
    ys: list[float] = []
    rms_values: list[float] = []
    center_history: list[tuple[float, float]] = []

    peak_x: float = x0
    peak_y: float = y0
    peak_rms: float = float("-inf")
    units: str = "?"

    def gcoord(i, j):
        return (x0 + i * step, y0 + j * step)

    def _measure(i, j):
        """Measure node (i, j) unless cached. Return rms or ``None``
        on scope timeout. Updates the global peak."""
        nonlocal peak_x, peak_y, peak_rms, units
        if (i, j) in grid_rms:
            return grid_rms[(i, j)]
        x, y = gcoord(i, j)
        meas = measure_fn(x, y)
        if meas is None:
            return None
        r = float(meas["rms"])
        grid_rms[(i, j)] = r
        grid_meas[(i, j)] = meas
        xs.append(x)
        ys.append(y)
        rms_values.append(r)
        if units == "?":
            units = str(meas.get("units", "?"))
        if r > peak_rms:
            peak_x, peak_y, peak_rms = x, y, r
        _emit(meas, x, y, r, info_extra=f"grid ({i:+d},{j:+d}) "
                                        f"RMS={r:.4g} {units}")
        return r

    def _emit(meas, x, y, r, *, iter_end=False, converged=False,
              done=False, info_extra=""):
        if on_progress is None:
            return
        on_progress(
            meas=meas, x=x, y=y, r=r,
            xs=xs, ys=ys, rms_values=rms_values,
            best_x=peak_x, best_y=peak_y, best_rms=peak_rms,
            units=units,
            iteration=iteration, step=step,
            evaluations=len(rms_values),
            converged=converged, done=done,
            iter_end=iter_end,
            centers=list(center_history),
            info_extra=info_extra,
        )

    def _emit_iter_end(*, converged=False, done=False, info_extra=""):
        if on_progress is None:
            return
        on_progress(
            meas=None,
            x=(center_history[-1][0] if center_history else x0),
            y=(center_history[-1][1] if center_history else y0),
            r=(grid_rms[best_ij] if best_ij in grid_rms else 0.0),
            xs=xs, ys=ys, rms_values=rms_values,
            best_x=peak_x, best_y=peak_y, best_rms=peak_rms,
            units=units,
            iteration=iteration, step=step,
            evaluations=len(rms_values),
            converged=converged, done=done,
            iter_end=True,
            centers=list(center_history),
            info_extra=info_extra,
        )

    iteration = 0
    best_ij: tuple[int, int] = (0, 0)

    # ---- Seed: origin + 4 cardinals -------------------------------
    r0 = _measure(0, 0)
    if r0 is None:
        raise RuntimeError("Scope timed out on initial measurement.")
    for (di, dj) in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        _measure(di, dj)

    best_ij = max(grid_rms, key=grid_rms.__getitem__)
    center_history.append(gcoord(*best_ij))
    _emit_iter_end(info_extra=f"seed done  best=({best_ij[0]:+d},"
                              f"{best_ij[1]:+d})")

    # ---- Cardinal walk + diagonal bracket -------------------------
    converged = False
    while iteration < max_iter and len(grid_rms) < max_evaluations:
        iteration += 1
        # Phase A: fill cardinals of current best.
        cardinals = [(best_ij[0] + di, best_ij[1] + dj)
                     for (di, dj) in [(1, 0), (-1, 0), (0, 1), (0, -1)]]
        for n in cardinals:
            if n not in grid_rms and len(grid_rms) < max_evaluations:
                _measure(*n)
        new_best = max(grid_rms, key=grid_rms.__getitem__)
        if new_best != best_ij:
            best_ij = new_best
            center_history.append(gcoord(*best_ij))
            _emit_iter_end(info_extra=f"walk \u2192 ({best_ij[0]:+d},"
                                      f"{best_ij[1]:+d})")
            continue

        # Phase B: fill diagonals of current best.
        diagonals = [(best_ij[0] + di, best_ij[1] + dj)
                     for (di, dj) in [(1, 1), (1, -1), (-1, 1), (-1, -1)]]
        for n in diagonals:
            if n not in grid_rms and len(grid_rms) < max_evaluations:
                _measure(*n)
        new_best = max(grid_rms, key=grid_rms.__getitem__)
        if new_best != best_ij:
            best_ij = new_best
            center_history.append(gcoord(*best_ij))
            _emit_iter_end(info_extra=f"walk (diag) \u2192 "
                                      f"({best_ij[0]:+d},{best_ij[1]:+d})")
            continue

        # All 8 neighbors sampled and lower \u2014 bracketed.
        converged = True
        _emit_iter_end(converged=True, done=False,
                       info_extra=f"bracketed at ({best_ij[0]:+d},"
                                  f"{best_ij[1]:+d})")
        break

    # ---- Paraboloid fit over fit_window window around best_ij -----
    fit_pts = []
    for di in range(-fit_window, fit_window + 1):
        for dj in range(-fit_window, fit_window + 1):
            n = (best_ij[0] + di, best_ij[1] + dj)
            if n in grid_rms:
                fx, fy = gcoord(*n)
                fit_pts.append((fx, fy, grid_rms[n]))

    grid_max_x, grid_max_y = gcoord(*best_ij)
    grid_max_r = grid_rms[best_ij]
    # Vertex must lie inside the fit window (radius = fit_window * step).
    fit_result = _fit_paraboloid_vertex(
        fit_pts, guard_radius=fit_window * step,
    )
    if fit_result is not None:
        center_x, center_y, center_rms = fit_result
        info_fit = (f"paraboloid vertex from {len(fit_pts)} pts \u2192 "
                    f"({center_x:.4f}, {center_y:.4f}) mm  "
                    f"RMS={center_rms:.4g} {units}")
    else:
        center_x, center_y = grid_max_x, grid_max_y
        center_rms = grid_max_r
        info_fit = (f"paraboloid fit rejected \u2014 fell back to "
                    f"grid-max ({grid_max_x:.4f}, {grid_max_y:.4f}) mm")
    center_history.append((center_x, center_y))
    _emit_iter_end(converged=converged, done=True, info_extra=info_fit)

    return {
        "best_x": peak_x,
        "best_y": peak_y,
        "best_rms": peak_rms,
        "center_x": center_x,
        "center_y": center_y,
        "center_rms": center_rms,
        "units": units,
        "converged": converged,
        "iterations": iteration,
        "evaluations": len(rms_values),
        "xs": xs,
        "ys": ys,
        "rms_values": rms_values,
    }


# ----------------------------------------------------------------------
# Optional live-plot helpers (matplotlib imported lazily).
# ----------------------------------------------------------------------
def _fit_parabola_coeffs(y_minus, y_center, y_plus, h):
    """Coefficients ``(a, b, c)`` of ``y = a*u^2 + b*u + c`` through
    the three points ``(-h, y_minus), (0, y_center), (+h, y_plus)``."""
    a = (y_minus + y_plus - 2.0 * y_center) / (2.0 * h * h)
    b = (y_plus - y_minus) / (2.0 * h)
    c = y_center
    return a, b, c


def make_live_figure():
    """Set up an interactive multi-panel figure. Returns a handles dict.

    Layout (left column top->bottom): latest hydrophone trace, P vs
    u' (along-gradient probe axis), P vs v' (perpendicular probe
    axis). Right column: scatter of visited (x, y) colored by RMS.
    """
    import matplotlib.pyplot as plt

    plt.ion()
    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.2, 1.0],
                          height_ratios=[1.2, 1.0, 1.0])
    ax_trace = fig.add_subplot(gs[0, 0])
    ax_pu = fig.add_subplot(gs[1, 0])
    ax_pv = fig.add_subplot(gs[2, 0])
    ax_scatter = fig.add_subplot(gs[:, 1])

    trace_line, = ax_trace.plot([], [], lw=1)
    ax_trace.set_xlabel("time (\u00b5s)")
    ax_trace.set_ylabel("hydrophone")
    ax_trace.grid(True)
    ax_trace.set_title("latest trace")

    # P vs u' and P vs v' -- three probe samples + fitted parabola.
    pu_samples, = ax_pu.plot([], [], "o", color="tab:blue", label="probes")
    pu_fit, = ax_pu.plot([], [], "-", color="tab:orange", lw=1,
                        label="quadratic fit")
    pu_peak, = ax_pu.plot([], [], "rx", markersize=10, markeredgewidth=2,
                          label="subsample peak")
    ax_pu.set_xlabel("u' offset (mm)")
    ax_pu.set_ylabel("RMS")
    ax_pu.set_title("P vs u' (along gradient)")
    ax_pu.grid(True)
    ax_pu.legend(loc="lower center", fontsize=7, ncol=3)

    pv_samples, = ax_pv.plot([], [], "o", color="tab:blue", label="probes")
    pv_fit, = ax_pv.plot([], [], "-", color="tab:orange", lw=1,
                        label="quadratic fit")
    pv_peak, = ax_pv.plot([], [], "rx", markersize=10, markeredgewidth=2,
                          label="subsample peak")
    ax_pv.set_xlabel("v' offset (mm)")
    ax_pv.set_ylabel("RMS")
    ax_pv.set_title("P vs v' (perp. to gradient)")
    ax_pv.grid(True)

    scatter = ax_scatter.scatter(
        [np.nan], [np.nan], c=[np.nan], cmap="viridis",
        s=40, edgecolors="k", linewidths=0.3,
    )
    ax_scatter.set_xlabel("x (mm)")
    ax_scatter.set_ylabel("y (mm)")
    ax_scatter.set_aspect("equal", "box")
    ax_scatter.grid(True)
    ax_scatter.set_title("visited points (color = RMS)")
    # Trajectory of best-guess (search-center) positions across
    # iterations, connected with a red line and dotted markers so
    # the convergence path is easy to follow.
    trajectory, = ax_scatter.plot(
        [], [], "-o", color="red", lw=1.2, markersize=5,
        markerfacecolor="none", markeredgecolor="red",
        label="best-guess path",
    )
    best_marker, = ax_scatter.plot(
        [], [], "rx", markersize=12, markeredgewidth=2, label="best",
    )
    ax_scatter.legend(loc="upper right", fontsize=8)
    cbar = fig.colorbar(scatter, ax=ax_scatter, shrink=0.8)

    info_text = fig.text(
        0.01, 0.005, "", va="bottom", ha="left",
        family="monospace", fontsize=8,
    )

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.show()
    # Force an initial draw so the window appears immediately instead of
    # waiting for the first measurement to complete.
    try:
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
    except Exception:
        pass

    return {
        "fig": fig,
        "ax_trace": ax_trace,
        "trace_line": trace_line,
        "ax_pu": ax_pu,
        "pu_samples": pu_samples,
        "pu_fit": pu_fit,
        "pu_peak": pu_peak,
        "ax_pv": ax_pv,
        "pv_samples": pv_samples,
        "pv_fit": pv_fit,
        "pv_peak": pv_peak,
        "ax_scatter": ax_scatter,
        "scatter": scatter,
        "cbar": cbar,
        "best_marker": best_marker,
        "trajectory": trajectory,
        "info_text": info_text,
    }


def _update_probe_axis(ax, samples_line, fit_line, peak_marker,
                       *, h, rms_triplet, frac):
    """Update one of the P vs u' / P vs v' axes with 3 probe samples
    and (if available) the fitted quadratic + subsample peak marker."""
    y_minus, y_center, y_plus = rms_triplet
    us_samples = np.array([-h, 0.0, h])
    ys_samples = np.array([y_minus, y_center, y_plus])
    samples_line.set_data(us_samples, ys_samples)

    a, b, c = _fit_parabola_coeffs(y_minus, y_center, y_plus, h)
    us_fit = np.linspace(-1.2 * h, 1.2 * h, 61)
    ys_fit = a * us_fit ** 2 + b * us_fit + c
    fit_line.set_data(us_fit, ys_fit)

    if frac is not None and a < 0:
        u_peak = frac * h
        y_peak = a * u_peak ** 2 + b * u_peak + c
        peak_marker.set_data([u_peak], [y_peak])
    else:
        peak_marker.set_data([], [])

    ax.relim()
    ax.autoscale_view()


def update_live_figure(handles, *, meas, xs, ys, rms_values, units,
                       best_x, best_y, best_rms,
                       iteration, step, evaluations,
                       converged, done, info_extra="",
                       probe_h=None, probe_u_rms=None, probe_v_rms=None,
                       quad_frac_u=None, quad_frac_v=None,
                       centers=None, iter_end=False,
                       **_ignored):
    """Refresh the live figure with the latest measurement and state.

    ``probe_h``, ``probe_u_rms``, ``probe_v_rms``: when the caller
    just finished the 4-point ``\u00b1u/\u00b1v`` probe pass they may
    pass ``h`` (probe spacing) and the ``(-h, 0, +h)`` RMS triplets
    for each axis so the corresponding subplot can show the samples
    and the fitted parabola.

    ``quad_frac_u``, ``quad_frac_v``: fractional (in units of ``h``)
    location of the quadratic-subsample peak on each axis. When
    provided (and the quadratic is concave-down) a red x marks the
    fitted peak on the subplot.

    ``centers``: optional list of ``(x, y)`` search-center positions
    across iterations. Drawn as a connected red-dotted line.

    ``iter_end`` is accepted (for symmetry with the ``on_progress``
    signature) but not used here; ``find_peak`` uses it to know when
    to pause.
    """
    import matplotlib.pyplot as plt
    del iter_end  # consumed by ``find_peak`` for pausing, not us.

    if meas is not None:
        handles["trace_line"].set_data(meas["t"], meas["trace"])
        handles["ax_trace"].relim()
        handles["ax_trace"].autoscale_view()
        handles["ax_trace"].set_ylabel(f"hydrophone ({units})")

    if xs and ys and rms_values:
        pts = np.column_stack((xs, ys))
        handles["scatter"].set_offsets(pts)
        handles["scatter"].set_array(np.asarray(rms_values, dtype=float))
        handles["scatter"].set_clim(np.min(rms_values), np.max(rms_values))
        pad = max(step * 2.0, 0.1)
        handles["ax_scatter"].set_xlim(min(xs) - pad, max(xs) + pad)
        handles["ax_scatter"].set_ylim(min(ys) - pad, max(ys) + pad)
        handles["best_marker"].set_data([best_x], [best_y])
        handles["cbar"].update_normal(handles["scatter"])

    if centers is not None and len(centers) > 0:
        cx = [c[0] for c in centers]
        cy = [c[1] for c in centers]
        handles["trajectory"].set_data(cx, cy)

    if probe_h is not None and probe_u_rms is not None:
        _update_probe_axis(
            handles["ax_pu"], handles["pu_samples"],
            handles["pu_fit"], handles["pu_peak"],
            h=probe_h, rms_triplet=probe_u_rms, frac=quad_frac_u,
        )
        handles["ax_pu"].set_ylabel(f"RMS ({units})")
    if probe_h is not None and probe_v_rms is not None:
        _update_probe_axis(
            handles["ax_pv"], handles["pv_samples"],
            handles["pv_fit"], handles["pv_peak"],
            h=probe_h, rms_triplet=probe_v_rms, frac=quad_frac_v,
        )
        handles["ax_pv"].set_ylabel(f"RMS ({units})")

    status = "CONVERGED" if converged else ("DONE" if done else "searching")
    latest_rms = rms_values[-1] if rms_values else float("nan")
    info = (
        f"iter        : {iteration}\n"
        f"evaluations : {evaluations}\n"
        f"step (mm)   : {step:.4f}\n"
        f"best        : x={best_x:.4f}  y={best_y:.4f}\n"
        f"best RMS    : {best_rms:.4g} {units}\n"
        f"latest RMS  : {latest_rms:.4g} {units}\n"
        f"status      : {status}"
    )
    if info_extra:
        info += "   |   " + info_extra
    handles["info_text"].set_text(info)

    # Belt-and-braces redraw. ``draw_idle`` schedules a repaint,
    # ``flush_events`` runs the GUI event loop so the repaint happens
    # now instead of being deferred until the next ``plt.pause``,
    # and ``plt.pause`` gives the backend a slice of time to actually
    # show the update. Without ``flush_events`` some backends
    # (notably Qt on Windows) will let the figure appear to freeze
    # while measurements are in progress.
    fig = handles["fig"]
    try:
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
    except Exception:
        pass
    plt.pause(0.001)
