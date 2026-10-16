"""Report figures show the aligned pulses, the acceptance limits and
the row verdicts, so a failure (or a near miss) is readable from the
plot alone."""
from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from openlifu_verification import AcceptanceCriteria, ScanConfig  # noqa: E402
from openlifu_verification.characterization import ROW, Characterization  # noqa: E402
from openlifu_verification.dry_run import DryRunTank  # noqa: E402
from openlifu_verification.operator_prefs import OperatorPrefs  # noqa: E402
from openlifu_verification.report_io import build_figures  # noqa: E402

FIGURE_ROWS = {
    "scan_2d": ("peak_x_mm", "peak_y_mm"),
    "lateral_1d": ("lateral_peak_offset_mm",),
    "elevation_1d": ("elevation_peak_offset_mm",),
    "axial_1d": ("peak_z_focus_mm",),
    "waveform_at_peak": ("pnp_at_peak_MPa",),
    "freq_response": ("freq_deviation_pct",),
    "voltage_linearity": ("voltage_r2",),
    "depth_cal_plane_wave": ("plane_wave_depth_mm",),
    "depth_cal_focused": ("focused_depth_mm",),
}


@pytest.fixture(scope="module")
def report():
    prefs = OperatorPrefs(tester_name="t", test_app_version="t",
                          hydrophone_sn="t", txm_sn="t", console_sn="t")
    chz = Characterization(DryRunTank(rng_seed=0), prefs=prefs,
                           criteria=AcceptanceCriteria(),
                           scan_config=ScanConfig(),
                           frequency_kHz=400.0)
    chz.run()
    return chz.report


@pytest.fixture(scope="module")
def figs(report):
    import matplotlib.pyplot as plt
    out = build_figures(report)
    yield out
    for fig in out.values():
        plt.close(fig)


def test_figure_order(figs):
    assert list(figs) == list(FIGURE_ROWS)


@pytest.mark.parametrize("name", list(FIGURE_ROWS))
def test_title_carries_row_verdict_and_margin(report, figs, name):
    statuses = {report.rows[ROW[k]].status for k in FIGURE_ROWS[name]}
    expected = "FAIL" if "FAIL" in statuses else statuses.pop()
    assert expected in ("PASS", "FAIL")
    title = figs[name].axes[0].get_title()
    verdict_line = title.splitlines()[1]
    assert verdict_line.startswith(expected)
    assert "margin" in verdict_line


def test_waveform_shows_pulses_and_pnp_threshold(report, figs):
    ax = figs["waveform_at_peak"].axes[0]
    n = int(report.scan_config.scope.pnp_n_averages)
    labels = [ln.get_label() for ln in ax.get_lines()]
    assert f"aligned pulses (n={n})" in labels
    assert "median" in labels
    crit = report.criteria
    wf = report.waveform_at_peak
    thr_MPa = (crit.pnp_min_MPa_for(400.0) * wf["pnp_voltage_V"]
               / crit.pnp_at_peak.reference_voltage_V)
    hlines = [ln for ln in ax.get_lines()
              if ln.get_label().startswith("pass: PNP")]
    assert len(hlines) == 1
    assert np.allclose(hlines[0].get_ydata(), -thr_MPa * 1e3)


def test_depth_cal_pass_window_brackets_nominal(report, figs):
    ax = figs["depth_cal_plane_wave"].axes[0]
    dc = report.depth_cal_plane_wave
    pd = report.criteria.peak_depth
    nominal, tol = pd.nominal_for(400.0), pd.hydrophone_tol_mm
    (span,) = [p for p in ax.patches if p.get_label().startswith("pass")]
    x = span.get_x()
    w = span.get_width()
    sos = dc["sos_m_per_s"]
    assert x == pytest.approx((nominal - tol) * 1e3 / sos)
    assert x + w == pytest.approx((nominal + tol) * 1e3 / sos)
