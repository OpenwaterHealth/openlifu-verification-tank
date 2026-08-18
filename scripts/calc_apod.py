"""Compute an apodization hex mask for the openlifu transducer array.

Prints a 16-digit hex code suitable for passing to
``single_pulse.py`` / ``pulse_sequence.py`` via
``--apodization``. Also draws a small ASCII map of the selected
elements so you can eyeball the pattern before firing.

Examples::

    # Single element nearest (0, 0):
    python scripts/calc_apod.py --type single --x 0 --y 0

    # Disk of radius 4 mm centered on (3, 4.5):
    python scripts/calc_apod.py --type disk --x0 3 --y0 4.5 --r 4

    # Every element:
    python scripts/calc_apod.py --type all

    # Explicit list of 1-based element indices:
    python scripts/calc_apod.py --type indices --indices 1 5 12 33

    # Complement of a disk (annulus outside r=6):
    python scripts/calc_apod.py --type disk --x0 0 --y0 0 --r 6 --invert

The ``--transducer`` flag selects which JSON pinmap to consult;
by default the tool auto-detects ``transducers/openlifu_1x400_evt1.json``
matching the ``--num-modules`` / ``--frequency-khz`` flags used
elsewhere.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from openlifu_verification import apodization as apo
from openlifu_verification.verificationtank import _TransducerData


REPO_ROOT = Path(__file__).parent.parent.resolve()


def _default_transducer_path(num_modules: int, frequency_khz: float) -> Path:
    """Mirror ``VerificationTank`` transducer path resolution."""
    return (REPO_ROOT / "transducers" /
            f"openlifu_{num_modules}x{int(frequency_khz)}_evt1.json")


def _load_transducer(path: Path) -> _TransducerData:
    if not path.is_file():
        raise FileNotFoundError(f"No transducer JSON at {path}")
    arr = _TransducerData.from_file(str(path))
    arr.sort_by_pin()
    return arr


def _estimate_pitch(positions_xy: np.ndarray) -> float:
    """Estimate the element pitch (min positive spacing along each
    axis). Used to auto-size the ASCII map cells so we don't leave
    blank rows between element rows."""
    xs = np.unique(np.round(positions_xy[:, 0], 3))
    ys = np.unique(np.round(positions_xy[:, 1], 3))
    dx = np.diff(np.sort(xs))
    dy = np.diff(np.sort(ys))
    candidates = [d for d in [*dx, *dy] if d > 0.01]
    return float(min(candidates)) if candidates else 5.0


def _draw_map(positions_xy: np.ndarray, active_idx: Iterable[int],
              *, cell_mm: Optional[float] = None) -> str:
    """Return a compact ASCII map of the array with active elements
    marked ``X`` and inactive ``.``.

    ``positions_xy`` is a ``(N, 2)`` array in mm; ``active_idx`` is a
    set of 0-based array indices. ``cell_mm`` defaults to the array's
    element pitch so each row of elements gets exactly one row of
    map cells.
    """
    if positions_xy.size == 0:
        return "(empty)"
    if cell_mm is None:
        cell_mm = _estimate_pitch(positions_xy)
    active_set = set(int(i) for i in active_idx)
    xs = positions_xy[:, 0]
    ys = positions_xy[:, 1]
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    # Pad the bounding box slightly so edge elements land inside a
    # cell instead of at the very edge of the grid.
    pad = cell_mm * 0.5
    x_min -= pad; x_max += pad
    y_min -= pad; y_max += pad
    n_cols = max(1, int(np.ceil((x_max - x_min) / cell_mm)))
    n_rows = max(1, int(np.ceil((y_max - y_min) / cell_mm)))

    grid: list[list[str]] = [[" " for _ in range(n_cols)]
                             for _ in range(n_rows)]
    for i, (x, y) in enumerate(positions_xy):
        col = int((x - x_min) / cell_mm)
        # Flip Y so +y prints upward.
        row = n_rows - 1 - int((y - y_min) / cell_mm)
        col = max(0, min(n_cols - 1, col))
        row = max(0, min(n_rows - 1, row))
        cell = grid[row][col]
        marker = "X" if i in active_set else "."
        # Prefer active over inactive when two elements fall in the
        # same cell (rare with cell_mm >= element pitch).
        if cell == "X":
            continue
        grid[row][col] = marker

    lines = ["".join(row) for row in grid]
    header = f"+{'-' * n_cols}+"
    return header + "\n" + "\n".join(f"|{line}|" for line in lines) + "\n" + header


def _resolve_type(args: argparse.Namespace,
                  arr: _TransducerData) -> tuple[str, list[int]]:
    """Dispatch on ``--type`` and return ``(hex_mask, active_0based)``.

    Returns the raw list of 0-based active indices too so the ASCII
    map matches whatever indices actually ended up selected."""
    t = args.type
    if t == "single":
        if args.x is None or args.y is None:
            raise SystemExit(
                "--type single requires --x and --y"
            )
        max_d = args.max_distance_mm
        hex_code = apo.element_at(arr, args.x, args.y,
                                  max_distance_mm=max_d)
    elif t == "disk":
        if args.x0 is None or args.y0 is None or args.r is None:
            raise SystemExit(
                "--type disk requires --x0, --y0, and --r"
            )
        hex_code = apo.element_disk(arr, args.x0, args.y0, args.r,
                                    inclusive=not args.exclusive)
    elif t == "rect":
        for name in ("x_min", "x_max", "y_min", "y_max"):
            if getattr(args, name) is None:
                raise SystemExit(
                    "--type rect requires --x-min --x-max --y-min --y-max"
                )
        hex_code = apo.element_rect(arr, args.x_min, args.x_max,
                                    args.y_min, args.y_max)
    elif t == "indices":
        if not args.indices:
            raise SystemExit(
                "--type indices requires one or more --indices values"
            )
        hex_code = apo.elements(args.indices)
    elif t == "all":
        hex_code = apo.all_elements(arr.numelements())
    elif t == "none":
        hex_code = apo.HEX_ALL_OFF
    else:  # pragma: no cover - argparse choices already restricts it
        raise SystemExit(f"unknown --type {t!r}")

    if args.invert:
        hex_code = apo.invert(hex_code, n=arr.numelements())

    # Expand the mask back to 0-based indices for the ASCII map + count.
    mask_int = apo.parse_hex(hex_code)
    active = [i for i in range(arr.numelements())
              if (mask_int >> i) & 1]
    return hex_code, active


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--type", choices=(
        "single", "disk", "rect", "indices", "all", "none",
    ), required=True, help="Shape of the region to select.")

    # single
    parser.add_argument("--x", type=float,
                        help="[single] X (mm) of the target point.")
    parser.add_argument("--y", type=float,
                        help="[single] Y (mm) of the target point.")
    parser.add_argument("--max-distance-mm", type=float, default=None,
                        help="[single] Reject if nearest element is "
                             "farther than this (mm). Default: no limit.")

    # disk
    parser.add_argument("--x0", type=float,
                        help="[disk] X (mm) of disk center.")
    parser.add_argument("--y0", type=float,
                        help="[disk] Y (mm) of disk center.")
    parser.add_argument("--r", type=float,
                        help="[disk] Radius (mm).")
    parser.add_argument("--exclusive", action="store_true",
                        help="[disk] Use strict inequality (< r) instead "
                             "of <=, so elements exactly on the boundary "
                             "are excluded.")

    # rect
    parser.add_argument("--x-min", type=float, help="[rect] X min (mm).")
    parser.add_argument("--x-max", type=float, help="[rect] X max (mm).")
    parser.add_argument("--y-min", type=float, help="[rect] Y min (mm).")
    parser.add_argument("--y-max", type=float, help="[rect] Y max (mm).")

    # indices
    parser.add_argument("--indices", type=int, nargs="+", default=None,
                        help="[indices] 1-based element indices.")

    # Universal modifiers.
    parser.add_argument("--invert", action="store_true",
                        help="Complement the resulting mask (only bits "
                             "for real elements are toggled).")

    # Transducer selection.
    parser.add_argument("--transducer", type=Path, default=None,
                        help="Path to the transducer pinmap JSON. "
                             "Default: auto-detect from "
                             "--num-modules / --frequency-khz.")
    parser.add_argument("--num-modules", type=int, default=1,
                        help="Auto-detect: modules count. Default 1.")
    parser.add_argument("--frequency-khz", type=float, default=400.0,
                        help="Auto-detect: nominal frequency. Default 400.")

    # Output controls.
    parser.add_argument("--no-map", action="store_true",
                        help="Skip the ASCII map (print only the hex).")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Print ONLY the hex mask (no header, no map). "
                             "Useful for shell substitution: "
                             "`--apodization $(python calc_apod.py ...)`.")

    args = parser.parse_args(argv)

    # Resolve transducer path.
    path = args.transducer or _default_transducer_path(
        args.num_modules, args.frequency_khz
    )
    try:
        arr = _load_transducer(path)
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 2

    try:
        hex_code, active = _resolve_type(args, arr)
    except SystemExit:
        raise
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if args.quiet:
        print(hex_code)
        return 0

    n_active = len(active)
    n_total = arr.numelements()
    print(f"Transducer: {path.name} ({n_total} elements)")
    print(f"Active:     {n_active} / {n_total}")
    print(f"Mask:       {hex_code}")
    if not args.no_map:
        pos_xy = arr.get_positions(units="mm")[:, :2]
        print()
        print(_draw_map(pos_xy, active))
    return 0


if __name__ == "__main__":
    sys.exit(main())
