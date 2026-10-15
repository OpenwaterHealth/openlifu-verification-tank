"""Configuration for the characterization scans.

Holds everything about *how* the sweep runs -- spatial extents /
point counts, scope capture window + sampling, frequency & voltage
sweep grids, and the PicoScope range plan for the voltage sweep.

The built-in defaults (``ScanConfig()``) ARE the standard production
configuration; nothing is read from disk unless an override file is
passed explicitly (:meth:`ScanConfig.with_overrides`, used by
``run_report.py --scan-config``). Any run that uses overrides is
marked non-standard in the report and cannot write to the device.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Optional

from .acceptance import SUPPORTED_FREQUENCIES_KHZ, AcceptanceCriteria, _freq_key

logger = logging.getLogger(__name__)


# PicoScope 5000a valid full-scale ranges (mV, +/-).
PICOSCOPE_RANGES_MV = (10, 20, 50, 100, 200, 500,
                       1000, 2000, 5000, 10000, 20000, 50000)


def choose_range_mv(expected_pk_mV: float, *,
                    headroom_pct: float = 30.0) -> int:
    """Smallest PicoScope range that fits ``expected_pk_mV`` with headroom.

    ``expected_pk_mV`` is the *single-sided* peak amplitude (not Vpp).
    """
    target = float(expected_pk_mV) * (1.0 + headroom_pct / 100.0)
    for r in PICOSCOPE_RANGES_MV:
        if r >= target:
            return r
    return PICOSCOPE_RANGES_MV[-1]


@dataclass
class ScopeCapture:
    """Scope capture window applied to every phase.

    ``time_start_us`` / ``time_stop_us`` are given relative to the
    start of ultrasound emission (t=0 = emission), so a pulse arriving
    at depth ``z`` mm shows up at ``t = z / SOS`` (us in water).
    """
    time_start_us: float = -10.0
    time_stop_us: float = 120.0
    sampling_interval_ns: float = 100.0
    #: Default vertical range on the hydrophone channel (mV, +/-). Used
    #: for the peak search + 1D/2D scans + freq sweep.
    hydrophone_range_mv: int = 200
    #: Headroom above the predicted peak amplitude when auto-picking a
    #: scope range for the voltage sweep. 30 % is a reasonable default.
    voltage_scan_headroom_pct: float = 30.0
    #: Pulses coherently averaged per point in the beam scans and the
    #: voltage sweep.
    n_averages: int = 3
    #: Pulses coherently averaged for the graded PNP measurements: the
    #: waveform-at-peak capture and the frequency sweep.
    pnp_n_averages: int = 5


@dataclass
class Scan1D:
    """1-D lateral / elevation / axial scan geometry (relative to peak)."""
    extent_mm: float
    points: int


@dataclass
class Scan2D:
    """2-D XY scan geometry (relative to peak)."""
    extent_mm: float
    points: int


@dataclass
class FrequencySweep:
    """Frequency-sweep offsets around the nominal center (kHz)."""
    offsets_kHz: list[float]


@dataclass
class VoltageSweep:
    """Voltage-sweep points (V rail)."""
    voltages_V: list[float]
    #: Floor for the PicoScope vertical range (mV, +/-) used during the
    #: voltage sweep. The actual range picked by
    #: :meth:`Characterization.sweep_voltage` is
    #: ``max(range_mv, auto_predicted_range)``, quantized up to the
    #: next valid PicoScope range. Prevents an underestimated
    #: prediction (e.g. from a clipped baseline capture) from sizing
    #: the sweep too small. 200 mV gives ~720 kPa headroom at 400 kHz
    #: for an HNR-0500.
    range_mv: int


@dataclass
class ScanPlan:
    """Scan geometry and sweep grids for one module frequency."""
    lateral_1d: Scan1D
    elevation_1d: Scan1D
    axial_1d: Scan1D
    scan_2d: Scan2D
    frequency_sweep: FrequencySweep
    voltage_sweep: VoltageSweep


def _standard_plans() -> dict[str, ScanPlan]:
    """Standard per-frequency scan plans (one explicit entry each)."""
    return {
        "155": ScanPlan(
            lateral_1d=Scan1D(extent_mm=9.0, points=37),
            elevation_1d=Scan1D(extent_mm=9.0, points=37),
            axial_1d=Scan1D(extent_mm=10.0, points=21),
            scan_2d=Scan2D(extent_mm=2.0, points=9),
            frequency_sweep=FrequencySweep(offsets_kHz=[
                -25.0, -20.0, -15.0, -10.0, -5.0, 0.0,
                5.0, 10.0, 15.0, 20.0, 25.0]),
            voltage_sweep=VoltageSweep(
                voltages_V=[10.0, 15.0, 20.0, 25.0, 30.0], range_mv=200),
        ),
        "400": ScanPlan(
            lateral_1d=Scan1D(extent_mm=9.0, points=37),
            elevation_1d=Scan1D(extent_mm=9.0, points=37),
            axial_1d=Scan1D(extent_mm=10.0, points=21),
            scan_2d=Scan2D(extent_mm=2.0, points=9),
            frequency_sweep=FrequencySweep(offsets_kHz=[
                -25.0, -20.0, -15.0, -10.0, -5.0, 0.0,
                5.0, 10.0, 15.0, 20.0, 25.0]),
            voltage_sweep=VoltageSweep(
                voltages_V=[10.0, 15.0, 20.0, 25.0, 30.0], range_mv=200),
        ),
    }


@dataclass
class ScanConfig:
    """Configuration for :class:`Characterization`.

    ``ScanConfig()`` is the standard production configuration. Use
    :meth:`with_overrides` for diagnostic runs; :meth:`differences_from_standard`
    reports what an override changed.
    """
    scope: ScopeCapture = field(default_factory=ScopeCapture)
    #: Per-frequency scan plans keyed by ``"155"`` / ``"400"``.
    by_freq_kHz: dict[str, ScanPlan] = field(default_factory=_standard_plans)
    acceptance: AcceptanceCriteria = field(default_factory=AcceptanceCriteria)
    #: Speed of sound in water (m/s), used to convert pulse arrival
    #: time to axial depth. 1500 m/s is standard for degassed water at
    #: ~22 \u00b0C; tweak here if the tank is at a different temperature
    #: or the medium changes.
    sos_water_m_per_s: float = 1500.0

    # ------------------------------------------------------------------
    # Validation / overrides
    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        self.by_freq_kHz = {_freq_key(k): v for k, v in self.by_freq_kHz.items()}
        missing = [_freq_key(f) for f in SUPPORTED_FREQUENCIES_KHZ
                   if _freq_key(f) not in self.by_freq_kHz]
        if missing:
            raise ValueError(
                f"scan config by_freq_kHz has no entry for "
                f"{', '.join(missing)} kHz; every supported frequency "
                f"needs an explicit scan plan")

    @classmethod
    def with_overrides(cls, path: Path) -> "ScanConfig":
        """Standard config with the values in ``path`` applied on top.

        ``path`` is a partial JSON document with the same shape as
        ``asdict(ScanConfig())``; only the keys it contains change and
        lists are replaced wholesale. Unknown keys, wrongly typed
        values and an ``acceptance`` section raise ``ValueError``
        (acceptance criteria are fixed in code).
        """
        path = Path(path)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}: invalid JSON ({e})") from e
        if not isinstance(data, dict):
            raise ValueError(f"{path}: top level must be a JSON object")
        if "acceptance" in data:
            raise ValueError(
                f"{path}: acceptance criteria cannot be overridden")
        cfg = cls()
        try:
            _apply_overrides(cfg, data, "")
        except ValueError as e:
            raise ValueError(f"{path}: {e}") from e
        return cfg

    def differences_from_standard(self) -> list[str]:
        """One ``"<key> = <value> (standard <value>)"`` line per
        setting that differs from ``ScanConfig()``."""
        mine = _flatten(asdict(self))
        std = _flatten(asdict(ScanConfig()))
        return [f"{k} = {mine.get(k)} (standard {std.get(k)})"
                for k in sorted(set(mine) | set(std))
                if mine.get(k) != std.get(k)]

    def save(self, path: Path) -> Path:
        """Serialize to ``path`` as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return path

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------
    def plan_for(self, freq_kHz: float) -> ScanPlan:
        """Scan plan for the module's nominal frequency (strict)."""
        key = _freq_key(freq_kHz)
        try:
            return self.by_freq_kHz[key]
        except KeyError:
            raise ValueError(
                f"scan config has no scan plan for {key} kHz") from None

    def scope_kwargs(self, *, min_time_stop_us: Optional[float] = None) -> dict:
        """Kwargs common to every scope-driven phase method.

        ``min_time_stop_us`` extends ``time_stop_us`` when a phase
        needs a longer window (e.g. long bursts at 155 kHz); the
        configured value is used when it is already long enough.
        """
        stop_us = float(self.scope.time_stop_us)
        if min_time_stop_us is not None:
            stop_us = max(stop_us, float(min_time_stop_us))
        return {
            "time_start_s": self.scope.time_start_us * 1e-6,
            "time_stop_s":  stop_us * 1e-6,
            "sampling_interval_ns": self.scope.sampling_interval_ns,
        }


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _apply_overrides(obj: Any, data: Any, where: str) -> None:
    """Recursively apply the partial dict ``data`` onto ``obj`` in place.

    ``obj`` is a dataclass or the ``by_freq_kHz`` dict. Leaves must be
    numbers (or lists of numbers) matching the type of the default.
    """
    if not isinstance(data, dict):
        raise ValueError(f"'{where}' must be a JSON object")
    if isinstance(obj, dict):
        for k, v in data.items():
            key = str(k)
            if key not in obj:
                raise ValueError(
                    f"unknown frequency '{where}.{key}' "
                    f"(expected one of {sorted(obj)})")
            _apply_overrides(obj[key], v, f"{where}.{key}")
        return
    known = {f.name for f in fields(obj)}
    for k, v in data.items():
        path = f"{where}.{k}" if where else k
        if k not in known:
            raise ValueError(f"unknown key '{path}'")
        cur = getattr(obj, k)
        if is_dataclass(cur) or isinstance(cur, dict):
            _apply_overrides(cur, v, path)
        elif isinstance(cur, list):
            if not isinstance(v, list) or not v or not all(map(_is_number, v)):
                raise ValueError(f"'{path}' must be a non-empty list of numbers")
            setattr(obj, k, [float(x) for x in v])
        elif isinstance(cur, int):
            if not _is_number(v) or v != int(v):
                raise ValueError(f"'{path}' must be an integer")
            setattr(obj, k, int(v))
        else:
            if not _is_number(v):
                raise ValueError(f"'{path}' must be a number")
            setattr(obj, k, float(v))


def _flatten(d: dict, prefix: str = "") -> dict[str, Any]:
    """``{"a": {"b": 1}}`` -> ``{"a.b": 1}`` (lists are leaves)."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out
