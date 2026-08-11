"""Result containers for scans run through :class:`VerificationTank`.

Each scan (:meth:`~openlifu_verification.VerificationTank.scan_lateral`,
:meth:`~openlifu_verification.VerificationTank.scan_2d`,
:meth:`~openlifu_verification.VerificationTank.scan_frequency`,
:meth:`~openlifu_verification.VerificationTank.scan_voltage`) returns a
:class:`ScanResult` bundling the captured traces, the coordinate arrays
that describe the sweep, per-point timings, and the metadata needed to
reproduce the scan.

The class provides two conveniences:

- :meth:`ScanResult.plot` — quick-look plots (line / heatmap / single
  trace / trace-image) with sensible defaults per scan type.
- :meth:`ScanResult.save` — write a self-describing NPZ (and, where
  relevant, a companion TSV) that matches the layout the old
  ``plot_*`` scripts expected.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ScanResult:
    """Bundle of arrays + metadata produced by a single sweep.

    Attributes:
        scan_type: One of ``"lateral"``, ``"2d"``, ``"frequency"``,
            ``"voltage"`` (or ``"generic"`` for anything else).
        t: 1-D time axis in µs, shared by every trace.
        traces: N-D array of hydrophone traces. The last axis is the
            sample axis; the leading axes correspond to the coord axes
            in ``coords`` in the same order.
        coords: Ordered mapping of coordinate axis name to 1-D array.
            The keys give the axis names for plots; the shape of
            ``traces[..., 0]`` matches ``tuple(len(v) for v in
            coords.values())``.
        hydrophone_channel: Which scope channel the traces came from
            (e.g. ``"A"``).
        chunk_size: Rapid-block chunk size actually used.
        timings: Per-point timing arrays (``apply_s``, ``trigger_s``,
            ``arm_s``, ``xfer_s``, ``iter_total_s``), flat in sweep
            order.
        metadata: Free-form dict of extra fields (voltage, focus,
            frequency, etc.) written into the NPZ under ``meta_<key>``.
        units: Physical units of ``traces``. ``"mV"`` (raw
            hydrophone voltage) or ``"Pa"`` (converted via a
            :class:`~openlifu_verification.Hydrophone` calibration).
    """

    scan_type: str
    t: np.ndarray
    traces: np.ndarray
    coords: dict[str, np.ndarray]
    hydrophone_channel: str = "A"
    chunk_size: int = 0
    timings: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    units: str = "mV"

    # ---- reductions ---------------------------------------------------
    @property
    def vpp(self) -> np.ndarray:
        """Peak-to-peak amplitude per point (same shape as coord grid)."""
        return np.ptp(self.traces, axis=-1)

    @property
    def vmin(self) -> np.ndarray:
        return self.traces.min(axis=-1)

    @property
    def vmax(self) -> np.ndarray:
        return self.traces.max(axis=-1)

    def reduce(self, kind: str) -> np.ndarray:
        """Return a scalar-per-point summary of the traces.

        Args:
            kind: One of ``"vpp"``, ``"vmin"``, ``"vmax"``.
        """
        kind = kind.lower()
        if kind == "vpp":
            return self.vpp
        if kind == "vmin":
            return self.vmin
        if kind == "vmax":
            return self.vmax
        raise ValueError(f"Unknown reduction {kind!r}; use vpp/vmin/vmax.")

    # ---- plotting -----------------------------------------------------
    def plot(self,
             kind: str = "auto",
             reduce: str = "vpp",
             index: Any = None,
             ax=None,
             show: bool = False,
             save_as: str | Path = "",
             title: str | None = None,
             **kwargs):
        """Render a quick-look plot of the scan.

        Args:
            kind: Plot style.

                - ``"auto"`` (default): pick based on coord layout —
                  ``"line"`` for 1-D coords, ``"heatmap"`` for 2-D.
                - ``"line"``: line plot of the reduction vs the single
                  coord axis. Requires 1-D coords.
                - ``"heatmap"``: 2-D imshow of the reduction over both
                  coord axes. Requires 2-D coords.
                - ``"trace"``: plot a single time-trace. ``index`` picks
                  which one (int, tuple of ints, or dict of coord ->
                  value; nearest match).
                - ``"trace_image"``: imshow of all traces along the
                  coord axis (only meaningful for 1-D sweeps).

            reduce: Reduction used by ``"line"``/``"heatmap"`` —
                ``"vpp"`` (default), ``"vmin"``, or ``"vmax"``.
            index: Trace picker for ``"trace"``.
            ax: Optional matplotlib Axes to draw into.
            show: Call ``plt.show()`` at the end.
            save_as: If truthy, save the figure to this path.
            title: Override the auto-generated title.
            **kwargs: Passed through to the underlying matplotlib call
                (``plot``/``imshow``).

        Returns:
            The matplotlib ``Figure``.
        """
        import matplotlib.pyplot as plt  # local import: keeps top-level cheap

        coord_names = list(self.coords.keys())
        n_coord_axes = len(coord_names)

        if kind == "auto":
            kind = "heatmap" if n_coord_axes == 2 else "line"

        fig = ax.figure if ax is not None else None
        if ax is None:
            fig, ax = plt.subplots()

        # Display convention: calibrated (Pa) scans are shown in kPa
        # (closer to the working scale). Raw mV traces are left alone.
        y_scale, y_unit = self._display_scale_unit()

        if kind == "line":
            if n_coord_axes != 1:
                raise ValueError(
                    f"kind='line' needs 1 coord axis, got {n_coord_axes} ({coord_names})."
                )
            values = self.reduce(reduce) * y_scale
            xname = coord_names[0]
            x = self.coords[xname]
            ax.plot(x, values, ".-", **kwargs)
            ax.set_xlabel(self._coord_axis_label(xname))
            ax.set_ylabel(self._reduction_label(reduce, y_unit))
            ax.grid(True)

        elif kind == "heatmap":
            if n_coord_axes != 2:
                raise ValueError(
                    f"kind='heatmap' needs 2 coord axes, got {n_coord_axes} ({coord_names})."
                )
            values = self.reduce(reduce) * y_scale  # shape (rows, cols)
            row_name, col_name = coord_names  # dict is insertion-ordered
            rows = self.coords[row_name]
            cols = self.coords[col_name]
            im = ax.imshow(
                values, aspect="auto", origin="lower",
                extent=[float(cols[0]), float(cols[-1]),
                        float(rows[0]), float(rows[-1])],
                **kwargs,
            )
            ax.set_xlabel(self._coord_axis_label(col_name))
            ax.set_ylabel(self._coord_axis_label(row_name))
            fig.colorbar(im, ax=ax, label=self._reduction_label(reduce, y_unit))

        elif kind == "trace":
            idx = self._resolve_trace_index(index)
            trace = self.traces[idx] * y_scale
            ax.plot(self.t, trace, **kwargs)
            ax.set_xlabel("time (\u00b5s)")
            ax.set_ylabel(self._trace_label(y_unit))
            ax.grid(True)

        elif kind == "trace_image":
            if n_coord_axes != 1:
                raise ValueError(
                    f"kind='trace_image' needs 1 coord axis, got {n_coord_axes}."
                )
            traces = self.traces * y_scale  # (N, samples)
            xname = coord_names[0]
            x = self.coords[xname]
            im = ax.imshow(
                traces, aspect="auto", origin="lower",
                extent=[float(self.t[0]), float(self.t[-1]),
                        float(x[0]), float(x[-1])],
                **kwargs,
            )
            ax.set_xlabel("time (\u00b5s)")
            ax.set_ylabel(self._coord_axis_label(xname))
            fig.colorbar(im, ax=ax, label=self._trace_label(y_unit))

        else:
            raise ValueError(
                f"Unknown kind={kind!r}. Use auto/line/heatmap/trace/trace_image."
            )

        ax.set_title(title if title is not None else self._auto_title(kind, reduce))
        fig.tight_layout()
        if save_as:
            fig.savefig(save_as)
            logger.info("Figure saved to %s", save_as)
        if show:
            plt.show()
        return fig

    # ---- I/O ----------------------------------------------------------
    def save(self, path: str | Path,
             save_txt: bool = True) -> Path:
        """Save the scan to ``path`` as NPZ (and TSV where sensible).

        The NPZ layout is:

        - ``t``: time axis (ns).
        - ``outputs``: the full ``traces`` array (kept as ``outputs``
          for backwards compatibility with existing plot notebooks).
        - one entry per coord axis, keyed by its axis name (e.g.
          ``xfoci``, ``freq``, ``voltages``).
        - one entry per timing key (``apply_s``, …).
        - ``chunk_size`` and ``hydrophone_channel`` scalars.
        - one ``meta_<key>`` entry per metadata field.

        If ``save_txt`` is True, a companion ``.txt`` file with the
        coord columns and the per-point Vpp is written next to the NPZ
        (except for 2-D scans, where a flat TSV is less useful).

        Args:
            path: Destination NPZ path. Parent directories are created.
            save_txt: Also write the per-point Vpp TSV next to the NPZ.

        Returns:
            The resolved ``Path`` written for the NPZ.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "t": self.t,
            "outputs": self.traces,
            "chunk_size": np.int64(self.chunk_size),
            "hydrophone_channel": np.array(self.hydrophone_channel),
            "scan_type": np.array(self.scan_type),
            "units": np.array(self.units),
        }
        for name, arr in self.coords.items():
            payload[name] = np.asarray(arr)
        for name, arr in self.timings.items():
            payload[name] = np.asarray(arr)
        for name, value in self.metadata.items():
            payload[f"meta_{name}"] = np.asarray(value)
        np.savez(path, **payload)
        logger.info("Scan data saved to %s", path)

        if save_txt and len(self.coords) == 1:
            xname, x = next(iter(self.coords.items()))
            vpp = self.vpp
            txt_path = path.with_suffix(".txt")
            np.savetxt(
                txt_path,
                np.column_stack((x, vpp)),
                header=f"{xname}\tVpp_{self.units}",
                fmt="%.6e",
                delimiter="\t",
            )
            logger.info("Per-point Vpp table saved to %s", txt_path)

        return path

    # ---- internals ----------------------------------------------------
    def _resolve_trace_index(self, index) -> tuple:
        """Turn an int/tuple/dict into a numpy index into ``traces``."""
        coord_names = list(self.coords.keys())
        n = len(coord_names)
        if index is None:
            if n == 1 and self.traces.shape[0] == 1:
                return (0,)
            raise ValueError(
                "kind='trace' requires an index; pass an int, tuple, or dict."
            )
        if isinstance(index, dict):
            axes = []
            for name in coord_names:
                if name not in index:
                    raise ValueError(f"trace index dict missing coord {name!r}.")
                axes.append(int(np.argmin(np.abs(self.coords[name] - index[name]))))
            return tuple(axes)
        if isinstance(index, tuple):
            if len(index) != n:
                raise ValueError(
                    f"trace index tuple must have {n} entries, got {len(index)}."
                )
            return tuple(int(i) for i in index)
        if n == 1:
            return (int(index),)
        raise ValueError(
            f"scalar trace index only allowed for 1-coord scans; this scan has "
            f"{n} coord axes ({coord_names})."
        )

    def _auto_title(self, kind: str, reduce: str) -> str:
        base = f"{self.scan_type} scan"
        if kind in ("line", "heatmap"):
            base += f" — {self._reduction_label(reduce)}"
        elif kind == "trace":
            base += " — single trace"
        elif kind == "trace_image":
            base += " — all traces"
        meta_bits = []
        if "voltage_V" in self.metadata:
            meta_bits.append(f"V={self.metadata['voltage_V']} V")
        if "frequency_kHz" in self.metadata:
            meta_bits.append(f"f={self.metadata['frequency_kHz']} kHz")
        if "z_mm" in self.metadata:
            meta_bits.append(f"z={self.metadata['z_mm']} mm")
        if meta_bits:
            base += " (" + ", ".join(meta_bits) + ")"
        return base

    def _reduction_label(self, reduction: str, unit: str | None = None) -> str:
        """Human-readable label for a reduction, respecting ``self.units``."""
        base = _REDUCTION_BASES.get(reduction.lower(), reduction)
        if unit is None:
            unit = self._display_scale_unit()[1]
        return f"{base} ({unit})"

    def _trace_label(self, unit: str | None = None) -> str:
        """Y-axis label for a raw trace plot."""
        if unit is None:
            unit = self._display_scale_unit()[1]
        return f"hydrophone ({unit})"

    def _display_scale_unit(self) -> tuple[float, str]:
        """Return ``(scale, display_unit)`` for plot y-values.

        Pa scans are rendered in kPa (closer to the working range for
        LIFU output); mV and any other units pass through unchanged.
        """
        if self.units == "Pa":
            return 1e-3, "kPa"
        return 1.0, self.units

    def _coord_axis_label(self, name: str) -> str:
        """X/Y axis label for a coord axis.

        For ``scan_type == "1d"`` sweeps we render the axis as
        ``Δ<dim> (<dim>_0 = <val> mm)`` where ``<val>`` is the
        calibrated origin (hydrophone position) in relative mode or
        0 in absolute mode.
        """
        if self.scan_type == "1d" and name in _1D_COORD_TO_DIM:
            dim = _1D_COORD_TO_DIM[name]
            axis_index = _DIM_TO_INDEX[dim]
            meta = self.metadata or {}
            if meta.get("absolute", False):
                origin = 0.0
            else:
                pos = meta.get("hydrophone_position_mm")
                origin = float(np.asarray(pos)[axis_index]) if pos is not None else 0.0
            subscript = "\u2080"  # unicode subscript zero
            return f"\u0394{dim} ({dim}{subscript} = {origin:.2f} mm)"
        return _axis_label(name)


# ----------------------------------------------------------------------
# Axis labels
# ----------------------------------------------------------------------
_AXIS_LABELS = {
    "xfoci": "x focus (mm)",
    "yfoci": "y focus (mm)",
    "zfoci": "z focus (mm)",
    "freq": "frequency (kHz)",
    "freq_kHz": "frequency (kHz)",
    "voltages": "input voltage (V)",
    "voltage_V": "input voltage (V)",
}
# Coord-name → axis letter used by scan_1d and DryRunTank.scan_1d. Used
# by ``ScanResult._coord_axis_label`` to render the "Δ<dim>" label.
_1D_COORD_TO_DIM = {"xfoci": "x", "yfoci": "y", "zfoci": "z"}
_DIM_TO_INDEX = {"x": 0, "y": 1, "z": 2}
_REDUCTION_LABELS = {
    "vpp": "Vpp (mV)",
    "vmin": "V min (mV)",
    "vmax": "V max (mV)",
}
# Base names used by ``ScanResult._reduction_label`` (units are added
# from ``ScanResult.units`` so mV/Pa scans get the right axis label).
_REDUCTION_BASES = {
    "vpp": "Vpp",
    "vmin": "V min",
    "vmax": "V max",
}


def _axis_label(name: str) -> str:
    return _AXIS_LABELS.get(name, name)


def _reduction_label(reduction: str) -> str:
    return _REDUCTION_LABELS.get(reduction.lower(), reduction)
