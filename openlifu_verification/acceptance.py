"""Acceptance criteria for the TXM characterization report.

Loaded from an editable JSON file (default: ``acceptance.json`` in
the repo root). Each threshold is applied by
:class:`Characterization` after the corresponding phase; the results
are collected as PASS / FAIL / NA per report row.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ArrivalTime:
    """Round-trip arrival time (\u00b5s) for the setup pulse."""
    nominal_us: float = 33.3          # ~50 mm / 1500 m/s
    tol_pct: float = 10.0


@dataclass
class PeakOffset:
    """Max allowed distance from the nominal focal center (mm)."""
    max_mm: float = 3.0


@dataclass
class PnpAtPeak:
    """Minimum PNP at the empirical peak, keyed by nominal freq (kHz)."""
    min_by_freq_kHz: dict[str, float] = field(default_factory=lambda: {
        "155": 0.5,
        "400": 1.0,
    })


@dataclass
class FreqResponse:
    """Ripple of the PNP-vs-frequency curve, in dB (peak-to-peak)."""
    max_ripple_dB: float = 3.0


@dataclass
class VoltageLinearity:
    """Linear-fit quality across the voltage sweep."""
    r2_min: float = 0.99


@dataclass
class Temperature:
    """TX board temperature during scans."""
    max_C: float = 35.0


@dataclass
class AcceptanceCriteria:
    """Complete threshold set.

    Load with :meth:`from_file`; missing / partial JSON entries fall
    back to the defaults defined on each subsection.
    """
    arrival_time: ArrivalTime = field(default_factory=ArrivalTime)
    peak_offset: PeakOffset = field(default_factory=PeakOffset)
    pnp_at_peak: PnpAtPeak = field(default_factory=PnpAtPeak)
    freq_response: FreqResponse = field(default_factory=FreqResponse)
    voltage_linearity: VoltageLinearity = field(default_factory=VoltageLinearity)
    temperature: Temperature = field(default_factory=Temperature)

    @classmethod
    def from_file(cls, path: Optional[Path] = None) -> "AcceptanceCriteria":
        """Load criteria from ``path``; unknown keys are ignored."""
        if path is None:
            return cls()
        path = Path(path)
        if not path.is_file():
            logger.warning(
                "acceptance file %s does not exist; using defaults", path,
            )
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            arrival_time=ArrivalTime(**data.get("arrival_time", {})),
            peak_offset=PeakOffset(**data.get("peak_offset", {})),
            pnp_at_peak=PnpAtPeak(**data.get("pnp_at_peak", {})),
            freq_response=FreqResponse(**data.get("freq_response", {})),
            voltage_linearity=VoltageLinearity(**data.get("voltage_linearity", {})),
            temperature=Temperature(**data.get("temperature", {})),
        )

    @classmethod
    def load_or_create(cls, path: Path) -> "AcceptanceCriteria":
        """Load criteria, seeding ``path`` with defaults if missing.

        On first-time run the file is written from the built-in
        defaults so the operator has an editable copy to tweak.
        """
        path = Path(path)
        if path.is_file():
            return cls.from_file(path)
        crit = cls()
        crit.save(path)
        logger.info("Seeded default acceptance criteria at %s", path)
        return crit

    def save(self, path: Path) -> Path:
        """Serialize criteria to ``path`` as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "arrival_time":     {"nominal_us": self.arrival_time.nominal_us,
                                 "tol_pct":     self.arrival_time.tol_pct},
            "peak_offset":      {"max_mm":      self.peak_offset.max_mm},
            "pnp_at_peak":      {"min_by_freq_kHz": dict(self.pnp_at_peak.min_by_freq_kHz)},
            "freq_response":    {"max_ripple_dB": self.freq_response.max_ripple_dB},
            "voltage_linearity":{"r2_min":      self.voltage_linearity.r2_min},
            "temperature":      {"max_C":       self.temperature.max_C},
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def pnp_min_for(self, freq_kHz: float) -> Optional[float]:
        """Look up the minimum PNP for a nominal frequency.

        Returns ``None`` if the frequency is not listed \u2014 the grader
        will then mark it ``NA`` rather than fail.
        """
        key = str(int(round(float(freq_kHz))))
        val = self.pnp_at_peak.min_by_freq_kHz.get(key)
        return float(val) if val is not None else None
