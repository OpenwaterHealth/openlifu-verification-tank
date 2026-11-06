"""Every frequency-dependent acceptance criterion must have an explicit
entry for each supported frequency (no scalar fallback)."""
from __future__ import annotations

import pytest

from openlifu_verification import AcceptanceCriteria, ScanConfig
from openlifu_verification.acceptance import SUPPORTED_FREQUENCIES_KHZ, PnpAtPeak
from openlifu_verification.characterization import ROW, Characterization
from openlifu_verification.dry_run import DryRunTank
from openlifu_verification.operator_prefs import OperatorPrefs


@pytest.mark.parametrize("freq", SUPPORTED_FREQUENCIES_KHZ)
def test_default_tables_cover_supported_frequencies(freq):
    crit = AcceptanceCriteria()
    crit.peak_offset.max_1d_axis_mm_for(freq)
    crit.peak_depth.nominal_for(freq)
    crit.pnp_min_MPa_for(freq)
    crit.freq_response.max_deviation_pct_for(freq)


@pytest.mark.parametrize("section, data", [
    ("peak_offset", {"max_1d_axis_mm_by_freq_kHz": {"155": 1.0}}),
    ("peak_depth", {"nominal_by_freq_kHz": {"400": 50.0}}),
    ("pnp_at_peak", {"min_MPa_by_freq_kHz": {"155": 0.225}}),
    ("freq_response", {"max_deviation_pct_by_freq_kHz": {"400": 15.0}}),
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


def _chz(crit: AcceptanceCriteria, voltage_V: float = 20.0,
         frequency_kHz: float = 400.0) -> Characterization:
    prefs = OperatorPrefs(tester_name="t", test_app_version="t",
                          hydrophone_sn="t", txm_sn="t", console_sn="t")
    return Characterization(DryRunTank(rng_seed=0), prefs=prefs,
                            criteria=crit, scan_config=ScanConfig(),
                            frequency_kHz=frequency_kHz, voltage_V=voltage_V)


@pytest.mark.parametrize("freq, expected", [(155.0, True), (400.0, False)])
def test_freq_response_tolerance_is_per_frequency(freq, expected):
    # Nominal sits 25% below the sweep peak: within the 30% allowed at
    # 155 kHz, outside the 15% allowed at 400 kHz.
    chz = _chz(AcceptanceCriteria(), frequency_kHz=freq)
    chz.report.freq_response = {
        "frequencies_kHz": [freq - 20.0, freq, freq + 20.0],
        "pnp_MPa": [1.0, 0.75, 0.9],
    }
    assert chz._grade_freq_response() is expected
    row = chz.report.rows[ROW["freq_deviation_pct"]]
    assert row.value == pytest.approx(25.0)
    assert row.threshold == {155.0: 30.0, 400.0: 15.0}[freq]


def test_waveform_at_peak_is_captured_at_reference_voltage():
    chz = _chz(AcceptanceCriteria(), voltage_V=20.0)
    chz.ver.set_voltage(20.0)
    wf = chz.measure_waveform_at_peak()

    assert wf["pnp_voltage_V"] == 50.0
    x, y, _ = chz.ver.hydrophone_position
    expected_MPa = chz.ver._amplitude_Pa(x, y, voltage_V=50.0) / 1e6
    assert wf["pnp_MPa"] == pytest.approx(expected_MPa, rel=0.05)
    # The scan voltage is restored for the phases that follow.
    assert chz.ver.hv_voltage == 20.0
    assert "50V/100Vpp" in chz.report.rows[ROW["pnp_at_peak_MPa"]].label


@pytest.mark.parametrize("pnp_voltage_V, expected", [(50.0, False), (25.0, True)])
def test_pnp_at_peak_threshold_rescales_with_capture_voltage(pnp_voltage_V, expected):
    # Limit 0.6 MPa at 50 V. 0.4 MPa fails at 50 V, but passes when
    # captured at 25 V, where the limit scales to 0.3 MPa.
    crit = AcceptanceCriteria(pnp_at_peak=PnpAtPeak(
        reference_voltage_V=50.0, min_MPa_by_freq_kHz={"155": 0.2, "400": 0.6}))
    chz = _chz(crit)
    chz.report.waveform_at_peak = {"pnp_MPa": 0.4, "pnp_voltage_V": pnp_voltage_V}
    chz.report.set_row(ROW["pnp_at_peak_MPa"], "x", 0.4, unit="MPa")

    assert chz._grade_pnp_at_peak() is expected
    row = chz.report.rows[ROW["pnp_at_peak_MPa"]]
    assert row.threshold == pytest.approx(0.6 * pnp_voltage_V / 50.0)
