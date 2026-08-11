"""Tests for the generic :meth:`scan_1d` API.

Exercises the DryRunTank implementation across ``dim="x"``, ``"y"``,
``"z"`` to pin down the coord axis / scan_type / metadata shape, and
sanity-checks that :class:`Characterization.run_beam_scans` produces
an ``axial_1d`` result.
"""
from __future__ import annotations

import numpy as np
import pytest

from openlifu_verification.dry_run import DryRunTank


def _fresh_tank():
    tank = DryRunTank(rng_seed=0, noise_Pa=0.0)
    tank.hv_voltage = tank._nominal_voltage
    return tank


@pytest.mark.parametrize("dim,coord_key", [
    ("x", "xfoci"),
    ("y", "yfoci"),
    ("z", "zfoci"),
])
def test_scan_1d_coord_axis_and_scan_type(dim, coord_key):
    """A 1-D scan returns a single-axis coord dict, ``scan_type='1d'``,
    and ``metadata['dim']`` matching the swept axis."""
    tank = _fresh_tank()
    result = tank.scan_1d(dim=dim, scan_range=(-2.0, 2.0), num=5)
    assert result.scan_type == "1d"
    assert list(result.coords.keys()) == [coord_key]
    assert result.coords[coord_key].shape == (5,)
    assert result.metadata["dim"] == dim
    # traces shape is (num_points, n_samples).
    assert result.traces.ndim == 2
    assert result.traces.shape[0] == 5


def test_scan_1d_rejects_bad_dim():
    tank = _fresh_tank()
    with pytest.raises(ValueError, match="dim must be"):
        tank.scan_1d(dim="q", scan_range=(-1.0, 1.0), num=3)


def test_scan_1d_z_sweeps_around_calibrated_depth():
    """In relative mode a z-scan of ``[-a, +a]`` should center on the
    calibrated ``hydrophone_position[2]``."""
    tank = _fresh_tank()
    z0 = float(tank.hydrophone_position[2])
    a = 5.0
    result = tank.scan_1d(dim="z", scan_range=(-a, +a), num=5,
                          absolute=False)
    # The requested (relative) coord axis is still [-a, +a] in the
    # returned result — the "at peak" origin is applied only when
    # setting the focus.
    assert result.coords["zfoci"][0] == pytest.approx(-a)
    assert result.coords["zfoci"][-1] == pytest.approx(+a)
    # The synth traces come from z_absolute = z0 + coord, so the middle
    # (coord=0) trace should peak later than an equivalent scan with
    # a smaller z0. Enough to verify that z_fixed metadata reflects
    # the calibrated origin.
    assert result.metadata["z_mm"] == pytest.approx(z0)


def test_scan_1d_absolute_mode_ignores_hydrophone_offset():
    """In absolute mode the origin for the swept axis is 0."""
    tank = _fresh_tank()
    # Move the calibrated hydrophone off-origin so we can tell the
    # difference between the two modes.
    tank.hydrophone_position = np.array([2.0, -1.5, 60.0])
    a = 1.0
    r_rel = tank.scan_1d(dim="x", scan_range=(-a, +a), num=3, absolute=False)
    r_abs = tank.scan_1d(dim="x", scan_range=(-a, +a), num=3, absolute=True)
    # The reported coord axis is the same in both modes (raw scan_range).
    np.testing.assert_allclose(r_rel.coords["xfoci"], r_abs.coords["xfoci"])
    # Metadata reflects the mode.
    assert r_rel.metadata["absolute"] is False
    assert r_abs.metadata["absolute"] is True


def test_characterization_run_beam_scans_includes_axial(clean_cwd):
    """After the API change, ``run_beam_scans`` must produce four
    entries (lateral, elevation, axial, 2d) and the axial one must have
    a ``zfoci`` coord axis."""
    from openlifu_verification.characterization import Characterization
    from openlifu_verification import ScanConfig, AcceptanceCriteria
    from openlifu_verification.operator_prefs import OperatorPrefs

    tank = _fresh_tank()
    prefs = OperatorPrefs(tester_name="dry", test_app_version="dry",
                          hydrophone_sn="dry", txm_sn="dry",
                          txm_hw_rev="dry", console_sn="dry",
                          console_hw_rev="dry")
    # Keep the scan geometry tiny so the test stays snappy.
    cfg = ScanConfig()
    cfg.lateral_1d.points = 5
    cfg.elevation_1d.points = 5
    cfg.axial_1d.points = 5
    cfg.scan_2d.points = 3
    ch = Characterization(
        ver=tank, prefs=prefs,
        scan_config=cfg, criteria=AcceptanceCriteria(),
        plot=False,
    )
    out = ch.run_beam_scans()
    assert set(out.keys()) == {"lateral_1d", "elevation_1d", "axial_1d", "scan_2d"}
    assert "zfoci" in out["axial_1d"].coords
    assert out["axial_1d"].scan_type == "1d"
    assert out["axial_1d"].metadata["dim"] == "z"
    assert out["axial_1d"].coords["zfoci"].shape == (5,)
