"""Acceptance criteria for the TXM characterization report.

Loaded from an editable JSON file (default: ``acceptance.json`` in
the repo root). Each threshold is applied by
:class:`Characterization` after the corresponding phase; the results
are collected as PASS / FAIL / NA per report row.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _filtered(cls, data: dict) -> Any:
    """Instantiate ``cls`` from ``data``, silently dropping unknown
    keys so a JSON file written by a previous version of the schema
    still loads.

    Warns per dropped key so operators notice when their edits
    aren't being applied because they targeted a renamed field."""
    if not data:
        return cls()
    known = {f.name for f in fields(cls)}
    kwargs = {}
    for k, v in data.items():
        if k in known:
            kwargs[k] = v
        else:
            logger.warning(
                "Ignoring unknown %s field %r "
                "(likely a renamed key from an older acceptance.json)",
                cls.__name__, k,
            )
    return cls(**kwargs)


def _peak_depth_from_dict(data: dict) -> "PeakDepth":
    """Load :class:`PeakDepth` with backward-compat for the pre-LUT
    schema where the sole nominal depth was ``nominal_mm`` and the
    tolerance was ``tol_pct``.

    Old shape (single-freq):
        ``{"nominal_mm": 50.0, "tol_pct": 5.0}``

    New shape (per-freq LUT):
        ``{"nominal_by_freq_kHz": {"155": 50.0, "400": 50.0},
           "default_nominal_mm": 50.0,
           "hydrophone_tol_mm": 2.0,
           "focused_tol_pct": 5.0,
           "peak_z_focus_tol_pct": 5.0}``
    """
    data = dict(data or {})
    # Migrate legacy keys in place so the filter step below picks them
    # up as their new names.
    if "nominal_mm" in data and "default_nominal_mm" not in data:
        data["default_nominal_mm"] = data.pop("nominal_mm")
    else:
        data.pop("nominal_mm", None)
    if "tol_pct" in data and "focused_tol_pct" not in data:
        data["focused_tol_pct"] = data.pop("tol_pct")
    else:
        data.pop("tol_pct", None)
    return _filtered(PeakDepth, data)


@dataclass
class ArrivalTime:
    """Round-trip arrival time (\u00b5s) for the setup pulse."""
    nominal_us: float = 33.3          # ~50 mm / 1500 m/s
    tol_pct: float = 10.0


@dataclass
class PeakOffset:
    """Max allowed distance from the nominal focal center (mm).

    ``max_mm`` bounds the Euclidean distance ``|(x, y)|`` and is
    reported log-only. ``max_axis_mm`` bounds each axis
    independently (``|x|`` and ``|y|``) and is what grades the
    D.5 / D.6 Hydrophone X/Y Position report rows.

    ``max_1d_axis_mm`` is the max separation between the peak of a
    1-D lateral (x) or elevation (y) scan and the peak located by
    the 2-D scan. Grades the D-section rows inserted right after
    the 1-D lateral / elevation figures.
    """
    max_mm: float = 3.0
    max_axis_mm: float = 1.0
    max_1d_axis_mm: float = 0.2


@dataclass
class PeakDepth:
    """Focused-arrival axial depth at the located peak, per nominal frequency.

    Both the plane-wave-derived hydrophone depth (D.3) and the
    focused-arrival hydrophone depth (D.14) are graded against
    ``nominal_by_freq_kHz`` for the current drive frequency. D.3
    uses the absolute-tolerance form (``hydrophone_tol_mm``) so
    small nominal shifts don't require re-tuning a percentage; D.14
    uses the percentage form (``focused_tol_pct``) so tighter
    depths automatically get tighter windows.

    ``peak_z_focus_tol_pct`` grades the commanded focus depth
    (D.10, ``peak_z_focus_mm``) against the measured plane-wave
    hydrophone depth (D.3) - i.e., the two should agree because
    the pipeline uses the plane-wave value as the seed for the
    focused-pulse measurement.
    """
    # LUT: frequency (kHz) -> nominal hydrophone depth (mm). Populate
    # more entries as new devices are calibrated; missing frequencies
    # fall back to :attr:`default_nominal_mm`.
    nominal_by_freq_kHz: dict[str, float] = field(default_factory=lambda: {
        "155": 50.0,
        "400": 50.0,
    })
    default_nominal_mm: float = 50.0
    hydrophone_tol_mm: float = 2.0
    focused_tol_pct: float = 5.0
    peak_z_focus_tol_pct: float = 5.0

    def nominal_for(self, freq_kHz: float) -> float:
        """Look up the nominal depth (mm) for ``freq_kHz`` with fallback."""
        key = str(int(round(float(freq_kHz))))
        val = self.nominal_by_freq_kHz.get(key)
        return float(val) if val is not None else float(self.default_nominal_mm)


@dataclass
class PnpAtPeak:
    """Minimum PNP at the empirical peak, keyed by nominal freq (kHz)."""
    min_by_freq_kHz: dict[str, float] = field(default_factory=lambda: {
        "155": 0.5,
        "400": 1.0,
    })


@dataclass
class FreqResponse:
    """Deviation of the nominal-frequency PNP from the peak PNP in the
    frequency sweep, expressed as a percentage of the peak."""
    max_deviation_pct: float = 5.0


@dataclass
class VoltageLinearity:
    """Linear-fit quality across the voltage sweep."""
    r2_min: float = 0.99


@dataclass
class Temperature:
    """TX board temperature during scans."""
    max_C: float = 35.0


@dataclass
class Firmware:
    """Minimum firmware versions (dotted semver strings, e.g. ``"1.2.4"``).

    Empty strings disable the check for that device \u2014 useful during
    bring-up when a device may report ``""`` for its firmware
    version. Version comparison uses tuple-of-ints ordering after
    stripping a leading ``"v"`` and ignoring build/pre-release
    suffixes.
    """
    min_txm_version: str = "1.2.4"
    min_console_version: str = "1.2.4"


@dataclass
class Calibration:
    """Age limit for factory calibration certificates.

    ``max_age_years`` is applied identically to the hydrophone and
    picoscope calibration dates parsed out of the device metadata.
    """
    max_age_years: float = 5.0


@dataclass
class AcceptanceCriteria:
    """Complete threshold set.

    Load with :meth:`from_file`; missing / partial JSON entries fall
    back to the defaults defined on each subsection.
    """
    arrival_time: ArrivalTime = field(default_factory=ArrivalTime)
    peak_offset: PeakOffset = field(default_factory=PeakOffset)
    peak_depth: PeakDepth = field(default_factory=PeakDepth)
    pnp_at_peak: PnpAtPeak = field(default_factory=PnpAtPeak)
    freq_response: FreqResponse = field(default_factory=FreqResponse)
    voltage_linearity: VoltageLinearity = field(default_factory=VoltageLinearity)
    temperature: Temperature = field(default_factory=Temperature)
    firmware: Firmware = field(default_factory=Firmware)
    calibration: Calibration = field(default_factory=Calibration)

    @classmethod
    def from_dict(cls, data: dict) -> "AcceptanceCriteria":
        """Construct criteria from a plain dict; unknown keys ignored.

        Used both by :meth:`from_file` and by :class:`ScanConfig` when
        acceptance is embedded inside ``scan_config.json``.
        """
        data = data or {}
        return cls(
            arrival_time=_filtered(ArrivalTime, data.get("arrival_time", {})),
            peak_offset=_filtered(PeakOffset, data.get("peak_offset", {})),
            peak_depth=_peak_depth_from_dict(data.get("peak_depth", {})),
            pnp_at_peak=_filtered(PnpAtPeak, data.get("pnp_at_peak", {})),
            freq_response=_filtered(FreqResponse, data.get("freq_response", {})),
            voltage_linearity=_filtered(VoltageLinearity,
                                        data.get("voltage_linearity", {})),
            temperature=_filtered(Temperature, data.get("temperature", {})),
            firmware=_filtered(Firmware, data.get("firmware", {})),
            calibration=_filtered(Calibration, data.get("calibration", {})),
        )

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
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

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
            "peak_offset":      {"max_mm":        self.peak_offset.max_mm,
                                 "max_axis_mm":   self.peak_offset.max_axis_mm,
                                 "max_1d_axis_mm": self.peak_offset.max_1d_axis_mm},
            "peak_depth":       {"nominal_by_freq_kHz": dict(self.peak_depth.nominal_by_freq_kHz),
                                 "default_nominal_mm":  self.peak_depth.default_nominal_mm,
                                 "hydrophone_tol_mm":   self.peak_depth.hydrophone_tol_mm,
                                 "focused_tol_pct":     self.peak_depth.focused_tol_pct,
                                 "peak_z_focus_tol_pct": self.peak_depth.peak_z_focus_tol_pct},
            "pnp_at_peak":      {"min_by_freq_kHz": dict(self.pnp_at_peak.min_by_freq_kHz)},
            "freq_response":    {"max_deviation_pct": self.freq_response.max_deviation_pct},
            "voltage_linearity":{"r2_min":      self.voltage_linearity.r2_min},
            "temperature":      {"max_C":       self.temperature.max_C},
            "firmware":         {"min_txm_version":     self.firmware.min_txm_version,
                                 "min_console_version": self.firmware.min_console_version},
            "calibration":      {"max_age_years": self.calibration.max_age_years},
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
