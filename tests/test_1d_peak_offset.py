"""1-D lateral / elevation peak-offset grading.

The grader and the report figure must agree on which point is the
peak, so both reduce the scan with PNP (not Vpp).
"""
from __future__ import annotations

import numpy as np

from openlifu_verification import AcceptanceCriteria, ScanConfig
from openlifu_verification.characterization import (
    ROW,
    Characterization,
    _1d_scan_pnp_peak,
)
from openlifu_verification.dry_run import DryRunTank
from openlifu_verification.operator_prefs import OperatorPrefs
from openlifu_verification.scan_results import ScanResult


def _scan_pnp_vs_vpp_disagree() -> ScanResult:
    """Coord 0 has the largest PNP; coord +0.5 has the largest Vpp."""
    ys = np.array([-0.5, 0.0, 0.5])
    traces = np.array([
        [0.0, -0.8e6, 0.8e6],   # PNP 0.8, Vpp 1.6
        [0.0, -1.0e6, 0.9e6],   # PNP 1.0, Vpp 1.9
        [0.0, -0.9e6, 1.2e6],   # PNP 0.9, Vpp 2.1
    ])
    return ScanResult(scan_type="1d", t=np.arange(3.0), traces=traces,
                      coords={"yfoci": ys}, units="Pa")


def _chz(freq_kHz: float) -> Characterization:
    prefs = OperatorPrefs(tester_name="t", test_app_version="t",
                          hydrophone_sn="t", txm_sn="t", console_sn="t")
    return Characterization(DryRunTank(rng_seed=0), prefs=prefs,
                            criteria=AcceptanceCriteria(),
                            scan_config=ScanConfig(),
                            frequency_kHz=freq_kHz)


def test_peak_helper_uses_pnp_not_vpp():
    coord, pnp, idx = _1d_scan_pnp_peak(_scan_pnp_vs_vpp_disagree())
    assert coord[idx] == 0.0
    assert pnp[idx] == 1.0


def test_elevation_grade_matches_plotted_peak():
    chz = _chz(400.0)
    chz.report.scans["elevation_1d"] = _scan_pnp_vs_vpp_disagree()
    chz.report.set_row(ROW["elevation_peak_offset_mm"], "x", 0.0, unit="mm")
    assert chz._grade_elevation_peak_offset() is True


def test_1d_offset_limit_is_frequency_dependent():
    crit = AcceptanceCriteria()
    assert crit.peak_offset.max_1d_axis_mm_for(155.0) == 1.0
    assert crit.peak_offset.max_1d_axis_mm_for(400.0) == 0.5

    scan = _scan_pnp_vs_vpp_disagree()
    # Shift so the PNP peak lands at +0.75 mm.
    scan.coords["yfoci"] = scan.coords["yfoci"] + 0.75
    for freq, expected in ((155.0, True), (400.0, False)):
        chz = _chz(freq)
        chz.report.scans["elevation_1d"] = scan
        chz.report.set_row(ROW["elevation_peak_offset_mm"], "x", 0.0, unit="mm")
        assert chz._grade_elevation_peak_offset() is expected
