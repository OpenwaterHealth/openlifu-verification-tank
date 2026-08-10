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


def gradient_search(
    measure_fn: Callable[[float, float], Optional[dict]],
    *,
    x0: float,
    y0: float,
    initial_step: float = 0.5,
    tol: float = 0.02,
    max_iter: int = 40,
    hysteresis: float = 0.01,
    probe_scale: float = 0.5,
    min_line_step_scale: float = 0.05,
    rotate_basis: bool = True,
    on_progress: Optional[Callable[..., None]] = None,
):
    """Walk toward the RMS peak from ``(x0, y0)``.

    On each iteration:

    1. Probe the RMS at ``\u00b1u`` and ``\u00b1v`` at distance
       ``h = probe_scale * step`` in the local basis.
    2. Estimate the gradient by central differences and pick the
       search direction.
    3. Do a backtracking line search along the gradient direction:
       try ``step, step/2, step/4, ...`` down to
       ``min_line_step_scale * step``, accepting the first trial that
       beats the current best by ``hysteresis``. If one succeeds,
       ``step`` is grown (up to a cap) for the next iteration.
    4. If the line search fails, fall back to the best probe if it
       beat the center by ``hysteresis``.
    5. Only if both the line search and probe fallback fail, halve
       ``step`` and iterate again.

    Args:
        measure_fn: ``measure_fn(x, y)`` fires one pulse at ``(x, y)``
            and returns a dict with keys ``{"t", "trace", "rms",
            "vpp", "units"}`` (matching
            :meth:`VerificationTank.measure_pressure`) or ``None`` on
            scope timeout.
        x0, y0: Starting position (mm).
        initial_step: Initial trial step length (mm).
        tol: Convergence tolerance (mm). Search stops when ``step``
            falls below this.
        max_iter: Maximum iterations.
        hysteresis: Fractional RMS improvement required to accept a
            move (noise guard).
        probe_scale: Probe distance as a fraction of ``step``. Must
            be > 0. Values around 0.5 give a genuinely local gradient
            estimate.
        min_line_step_scale: Smallest backtracking line-search step,
            as a fraction of the current ``step``. Must be > 0.
        rotate_basis: If ``True``, rotate the probe basis so ``u``
            aligns with each accepted gradient direction. If
            ``False``, probes stay axis-aligned.
        on_progress: Optional callback invoked after every
            measurement, with kwargs ``(meas, x, y, r, xs, ys,
            rms_values, best_x, best_y, best_rms, units, iteration,
            step, evaluations, converged, done, info_extra)``. Used by
            the live-plot layer.

    Returns:
        Dict with keys ``best_x, best_y, best_rms, units, converged,
        iterations, evaluations, xs, ys, rms_values``.
    """
    if probe_scale <= 0:
        raise ValueError("probe_scale must be > 0")
    if min_line_step_scale <= 0 or min_line_step_scale > 1:
        raise ValueError("min_line_step_scale must be in (0, 1]")

    xs: list[float] = []
    ys: list[float] = []
    rms_values: list[float] = []

    def _measure(x, y):
        meas = measure_fn(x, y)
        if meas is None:
            return None, float("-inf")
        return meas, meas["rms"]

    def _emit(meas, x, y, r, *, best_x, best_y, best_rms,
              iteration, step, converged=False, done=False, info_extra=""):
        xs.append(x)
        ys.append(y)
        rms_values.append(r)
        if on_progress is not None:
            on_progress(
                meas=meas, x=x, y=y, r=r,
                xs=xs, ys=ys, rms_values=rms_values,
                best_x=best_x, best_y=best_y, best_rms=best_rms,
                units=units,
                iteration=iteration, step=step,
                evaluations=len(rms_values),
                converged=converged, done=done,
                info_extra=info_extra,
            )

    center_meas, center_rms = _measure(x0, y0)
    if center_meas is None:
        raise RuntimeError("Scope timed out on initial measurement.")
    units = center_meas["units"]
    best_x, best_y, best_rms = x0, y0, center_rms
    best_meas = center_meas
    _emit(best_meas, x0, y0, center_rms,
          best_x=best_x, best_y=best_y, best_rms=best_rms,
          iteration=0, step=initial_step,
          info_extra=f"start (RMS={center_rms:.4g} {units})")

    # Local probe basis. u is the "along-gradient" direction, v its
    # perpendicular. Starts axis-aligned; rotates to follow the
    # accepted gradient direction after each successful gradient step.
    u = np.array([1.0, 0.0])
    v = np.array([0.0, 1.0])

    step = float(initial_step)
    iteration = 0
    converged = False
    grow_factor = 1.4
    max_step = float(initial_step) * 4.0

    while iteration < max_iter:
        iteration += 1
        h = step * probe_scale
        probes = [
            ("+u", best_x + h * u[0], best_y + h * u[1]),
            ("-u", best_x - h * u[0], best_y - h * u[1]),
            ("+v", best_x + h * v[0], best_y + h * v[1]),
            ("-v", best_x - h * v[0], best_y - h * v[1]),
        ]
        rvals: dict[str, float] = {}
        # Track the best probe (in case both the gradient step and
        # its line search fail; we'd rather jump to a probe that
        # actually improved than stay put and halve).
        best_probe = None  # (label, x, y, r, meas)
        for label, px, py in probes:
            meas, r = _measure(px, py)
            if meas is None:
                continue
            _emit(meas, px, py, r,
                  best_x=best_x, best_y=best_y, best_rms=best_rms,
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

        # Central-difference gradient in the local (u, v) basis.
        g_u = (rvals["+u"] - rvals["-u"]) / (2.0 * h)
        g_v = (rvals["+v"] - rvals["-v"]) / (2.0 * h)
        grad_xy = g_u * u + g_v * v
        gnorm = float(np.linalg.norm(grad_xy))
        threshold = best_rms * (1.0 + hysteresis)

        moved = False
        if gnorm >= 1e-12:
            direction = grad_xy / gnorm
            # Backtracking line search along the gradient direction:
            # start at step, then step/2, step/4, ..., down to
            # step * min_line_step_scale.
            trial_step = step
            min_line_step = step * min_line_step_scale
            while trial_step >= min_line_step:
                trial_x = best_x + trial_step * direction[0]
                trial_y = best_y + trial_step * direction[1]
                trial_meas, trial_rms = _measure(trial_x, trial_y)
                if trial_meas is not None:
                    _emit(trial_meas, trial_x, trial_y, trial_rms,
                          best_x=best_x, best_y=best_y, best_rms=best_rms,
                          iteration=iteration, step=step,
                          info_extra=(
                              f"trial step={trial_step:.4f} "
                              f"dir=({direction[0]:+.3f}, {direction[1]:+.3f}) "
                              f"|g|={gnorm:.3g}"
                          ))
                if trial_meas is not None and trial_rms > threshold:
                    prev_x, prev_y = best_x, best_y
                    best_x, best_y, best_rms, best_meas = (
                        trial_x, trial_y, trial_rms, trial_meas,
                    )
                    logger.info(
                        "iter %d: gradient step %s(%+.4f, %+.4f) \u2192 "
                        "(%.4f, %.4f)  RMS=%.4g %s  "
                        "trial_step=%.4f  dir=(%+.3f, %+.3f)",
                        iteration,
                        "from (%.4f, %.4f) " % (prev_x, prev_y),
                        trial_step * direction[0], trial_step * direction[1],
                        best_x, best_y, best_rms, units,
                        trial_step, direction[0], direction[1],
                    )
                    if rotate_basis:
                        u = direction.copy()
                        v = np.array([-u[1], u[0]])
                    # Grow the step budget for next iteration, but
                    # relative to the accepted trial size so we don't
                    # keep overshooting.
                    step = min(max(trial_step, step) * grow_factor, max_step)
                    moved = True
                    break
                trial_step /= 2.0

        if moved:
            continue

        # --- Fallback 1: best probe beats center by hysteresis ---
        if best_probe is not None and best_probe[3] > threshold:
            label, px, py, r, meas = best_probe
            logger.info(
                "iter %d: line search failed, moving to best probe %s "
                "(%.4f, %.4f)  RMS=%.4g %s  step held at %.4f",
                iteration, label, px, py, r, units, step,
            )
            best_x, best_y, best_rms, best_meas = px, py, r, meas
            # No basis rotation on a probe fallback (we don't have a
            # trustworthy gradient direction here).
            continue

        # --- Fallback 2: refine by halving step (probes get finer too) ---
        step /= 2.0
        reason = "flat gradient" if gnorm < 1e-12 else "no trial or probe improved"
        logger.info(
            "iter %d: %s \u2192 halve step to %.4f (best RMS=%.4g %s)",
            iteration, reason, step, best_rms, units,
        )
        if step < tol:
            converged = True
            if on_progress is not None:
                on_progress(
                    meas=best_meas, x=best_x, y=best_y, r=best_rms,
                    xs=xs, ys=ys, rms_values=rms_values,
                    best_x=best_x, best_y=best_y, best_rms=best_rms,
                    units=units,
                    iteration=iteration, step=step,
                    evaluations=len(rms_values),
                    converged=True, done=True, info_extra="",
                )
            break

    return {
        "best_x": best_x,
        "best_y": best_y,
        "best_rms": best_rms,
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
def make_live_figure():
    """Set up an interactive 3-panel figure. Returns a handles dict."""
    import matplotlib.pyplot as plt

    plt.ion()
    fig = plt.figure(figsize=(11, 5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.4, 1.0], height_ratios=[1, 1])
    ax_trace = fig.add_subplot(gs[0, 0])
    ax_scatter = fig.add_subplot(gs[:, 1])
    ax_info = fig.add_subplot(gs[1, 0])
    ax_info.axis("off")

    trace_line, = ax_trace.plot([], [], lw=1)
    ax_trace.set_xlabel("time (\u00b5s)")
    ax_trace.set_ylabel("hydrophone")
    ax_trace.grid(True)
    ax_trace.set_title("latest trace")

    scatter = ax_scatter.scatter(
        [np.nan], [np.nan], c=[np.nan], cmap="viridis",
        s=40, edgecolors="k", linewidths=0.3,
    )
    ax_scatter.set_xlabel("x (mm)")
    ax_scatter.set_ylabel("y (mm)")
    ax_scatter.set_aspect("equal", "box")
    ax_scatter.grid(True)
    ax_scatter.set_title("visited points (color = RMS)")
    best_marker, = ax_scatter.plot(
        [], [], "rx", markersize=12, markeredgewidth=2, label="best",
    )
    ax_scatter.legend(loc="upper right", fontsize=8)
    cbar = fig.colorbar(scatter, ax=ax_scatter, shrink=0.8)

    info_text = ax_info.text(
        0.02, 0.98, "", va="top", ha="left", family="monospace",
        transform=ax_info.transAxes,
    )

    fig.tight_layout()
    fig.show()

    return {
        "fig": fig,
        "ax_trace": ax_trace,
        "trace_line": trace_line,
        "ax_scatter": ax_scatter,
        "scatter": scatter,
        "cbar": cbar,
        "best_marker": best_marker,
        "info_text": info_text,
    }


def update_live_figure(handles, *, meas, xs, ys, rms_values, units,
                       best_x, best_y, best_rms,
                       iteration, step, evaluations,
                       converged, done, info_extra="", **_ignored):
    """Refresh the live figure with the latest measurement and state."""
    import matplotlib.pyplot as plt

    if meas is not None:
        handles["trace_line"].set_data(meas["t"] * 1e-3, meas["trace"])
        handles["ax_trace"].relim()
        handles["ax_trace"].autoscale_view()
        handles["ax_trace"].set_ylabel(f"hydrophone ({units})")

    pts = np.column_stack((xs, ys))
    handles["scatter"].set_offsets(pts)
    handles["scatter"].set_array(np.asarray(rms_values, dtype=float))
    handles["scatter"].set_clim(np.min(rms_values), np.max(rms_values))
    pad = max(step * 2.0, 0.1)
    handles["ax_scatter"].set_xlim(min(xs) - pad, max(xs) + pad)
    handles["ax_scatter"].set_ylim(min(ys) - pad, max(ys) + pad)
    handles["best_marker"].set_data([best_x], [best_y])
    handles["cbar"].update_normal(handles["scatter"])

    status = "CONVERGED" if converged else ("DONE" if done else "searching")
    latest_rms = rms_values[-1] if rms_values else float("nan")
    info = (
        f"iter        : {iteration}\n"
        f"evaluations : {evaluations}\n"
        f"step (mm)   : {step:.4f}\n"
        f"best        : x={best_x:.4f}  y={best_y:.4f}\n"
        f"best RMS    : {best_rms:.4g} {units}\n"
        f"latest RMS  : {latest_rms:.4g} {units}\n"
        f"status      : {status}\n"
    )
    if info_extra:
        info += "\n" + info_extra
    handles["info_text"].set_text(info)

    handles["fig"].canvas.draw_idle()
    plt.pause(0.01)
