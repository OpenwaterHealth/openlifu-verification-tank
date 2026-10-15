"""Every frequency-dependent acceptance criterion must have an explicit
entry for each supported frequency (no scalar fallback)."""
from __future__ import annotations

from pathlib import Path

import pytest

from openlifu_verification import AcceptanceCriteria, ScanConfig
from openlifu_verification.acceptance import SUPPORTED_FREQUENCIES_KHZ

SCAN_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "scan_config.json"


@pytest.mark.parametrize("freq", SUPPORTED_FREQUENCIES_KHZ)
def test_default_tables_cover_supported_frequencies(freq):
    crit = AcceptanceCriteria()
    crit.peak_offset.max_1d_axis_mm_for(freq)
    crit.peak_depth.nominal_for(freq)
    crit.pnp_min_for(freq)


@pytest.mark.parametrize("freq", SUPPORTED_FREQUENCIES_KHZ)
def test_live_config_covers_supported_frequencies(freq):
    crit = ScanConfig.from_file(SCAN_CONFIG_PATH).acceptance
    crit.peak_offset.max_1d_axis_mm_for(freq)
    crit.peak_depth.nominal_for(freq)
    crit.pnp_min_for(freq)


@pytest.mark.parametrize("section, data", [
    ("peak_offset", {"max_1d_axis_mm_by_freq_kHz": {"155": 1.0}}),
    ("peak_depth", {"nominal_by_freq_kHz": {"400": 50.0}}),
    ("pnp_at_peak", {"min_by_freq_kHz": {"155": 0.5}}),
])
def test_missing_frequency_entry_is_rejected(section, data):
    with pytest.raises(ValueError, match="no entry for"):
        AcceptanceCriteria.from_dict({section: data})


def test_legacy_single_nominal_depth_applies_to_all_frequencies():
    crit = AcceptanceCriteria.from_dict(
        {"peak_depth": {"nominal_mm": 42.0, "tol_pct": 3.0}})
    for f in SUPPORTED_FREQUENCIES_KHZ:
        assert crit.peak_depth.nominal_for(f) == 42.0
    assert crit.peak_depth.focused_tol_pct == 3.0
