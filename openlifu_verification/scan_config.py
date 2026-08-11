"""Editable configuration for the characterization scans.

Holds everything about *how* the sweep runs \u2014 spatial extents /
point counts, scope capture window + sampling, frequency & voltage
sweep grids, and the PicoScope range plan for the voltage sweep.

Loaded from an editable JSON file (default:
``config/scan_config.json`` in the CWD, see
:mod:`openlifu_verification.paths`). Missing fields fall back to the
built-in defaults; a missing file is seeded on first-time run so the
operator has a copy to tweak.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from .acceptance import AcceptanceCriteria
from .paths import SCAN_CONFIG_PATH  # re-exported convenience

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
    at depth ``z`` mm shows up at ``t = z / SOS`` (\u00b5s in water).
    """
    time_start_us: float = -14.0
    time_stop_us: float = 86.0
    sampling_interval_ns: float = 100.0
    #: Default vertical range on the hydrophone channel (mV, +/-). Used
    #: for the arrival check + peak search + 1D/2D scans + freq sweep.
    hydrophone_range_mv: int = 100
    #: Headroom above the predicted peak amplitude when auto-picking a
    #: scope range for the voltage sweep. 30 % is a reasonable default.
    voltage_scan_headroom_pct: float = 30.0


@dataclass
class Scan1D:
    """1-D lateral / elevation scan geometry (relative to peak)."""
    extent_mm: float = 5.0
    points: int = 21


@dataclass
class Scan2D:
    """2-D XY scan geometry (relative to peak)."""
    extent_mm: float = 3.0
    points: int = 13


@dataclass
class FrequencySweep:
    """Frequency-sweep offsets around the nominal center (kHz)."""
    offsets_kHz: list[float] = field(default_factory=lambda: [
        -25.0, -20.0, -15.0, -10.0, -5.0, 0.0, 5.0, 10.0,
    ])
    #: Cycles per burst; the pulse duration is ``cycles / frequency``.
    cycles_per_burst: float = 20.0


@dataclass
class VoltageSweep:
    """Voltage-sweep points (V rail)."""
    voltages_V: list[float] = field(default_factory=lambda: [
        5.0, 10.0, 15.0, 20.0, 25.0, 30.0,
    ])


@dataclass
class ScanConfig:
    """Editable configuration for :class:`Characterization`.

    Load with :meth:`load_or_create` (mirrors
    :class:`AcceptanceCriteria`). Individual phase methods on
    :class:`Characterization` pick up their settings from here.
    """
    scope: ScopeCapture = field(default_factory=ScopeCapture)
    lateral_1d: Scan1D = field(default_factory=Scan1D)
    elevation_1d: Scan1D = field(default_factory=Scan1D)
    scan_2d: Scan2D = field(default_factory=Scan2D)
    frequency_sweep: FrequencySweep = field(default_factory=FrequencySweep)
    voltage_sweep: VoltageSweep = field(default_factory=VoltageSweep)
    acceptance: AcceptanceCriteria = field(default_factory=AcceptanceCriteria)
    #: Speed of sound in water (m/s), used to convert pulse arrival
    #: time to axial depth. 1500 m/s is standard for degassed water at
    #: ~22 \u00b0C; tweak here if the tank is at a different temperature
    #: or the medium changes.
    sos_water_m_per_s: float = 1500.0

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------
    @classmethod
    def from_file(cls, path: Optional[Path] = None) -> "ScanConfig":
        """Load config from ``path``; unknown keys are ignored."""
        if path is None:
            return cls()
        path = Path(path)
        if not path.is_file():
            logger.warning("scan config %s does not exist; using defaults", path)
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            scope=ScopeCapture(**data.get("scope", {})),
            lateral_1d=Scan1D(**data.get("lateral_1d", {})),
            elevation_1d=Scan1D(**data.get("elevation_1d", {})),
            scan_2d=Scan2D(**data.get("scan_2d", {})),
            frequency_sweep=FrequencySweep(**data.get("frequency_sweep", {})),
            voltage_sweep=VoltageSweep(**data.get("voltage_sweep", {})),
            acceptance=AcceptanceCriteria.from_dict(data.get("acceptance", {})),
            sos_water_m_per_s=float(
                data.get("sos_water_m_per_s", cls.sos_water_m_per_s)
            ),
        )

    @classmethod
    def load_or_create(cls, path: Path) -> "ScanConfig":
        """Load config, seeding ``path`` with defaults if missing."""
        path = Path(path)
        if path.is_file():
            return cls.from_file(path)
        cfg = cls()
        cfg.save(path)
        logger.info("Seeded default scan config at %s", path)
        return cfg

    def save(self, path: Path) -> Path:
        """Serialize to ``path`` as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return path

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------
    def scope_kwargs(self) -> dict:
        """Kwargs common to every scope-driven phase method."""
        return {
            "time_start_s": self.scope.time_start_us * 1e-6,
            "time_stop_s":  self.scope.time_stop_us * 1e-6,
            "sampling_interval_ns": self.scope.sampling_interval_ns,
        }
