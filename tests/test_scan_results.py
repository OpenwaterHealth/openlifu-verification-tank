"""Tests for :class:`openlifu_verification.ScanResult`."""
from __future__ import annotations

import numpy as np

from openlifu_verification import ScanResult


def _fake_lateral(n=5) -> ScanResult:
    t = np.linspace(0, 100, 51)
    # Traces: (n, samples). Each pulse a sine with growing amplitude.
    traces = np.array([
        (0.5 + i * 0.5) * np.sin(2 * np.pi * 5e-2 * t)
        for i in range(n)
    ])
    return ScanResult(
        scan_type="lateral",
        t=t,
        traces=traces,
        coords={"xfoci": np.linspace(-1.0, 1.0, n)},
        hydrophone_channel="A",
        units="Pa",
    )


def test_construction_accepts_full_kwarg_set():
    r = _fake_lateral()
    assert r.scan_type == "lateral"
    assert r.units == "Pa"
    assert r.traces.shape[0] == r.coords["xfoci"].size


def test_reductions_shape_matches_coords():
    r = _fake_lateral(n=7)
    assert r.vpp.shape == (7,)
    assert r.vmin.shape == (7,)
    assert r.vmax.shape == (7,)
    assert (r.vpp == np.ptp(r.traces, axis=-1)).all()


def test_pulse_sequence_scan_type_supported():
    """The new pulse_sequence script builds a ScanResult with
    ``scan_type="pulse_sequence"``. Make sure ScanResult accepts it
    (and reductions still work)."""
    n_pulses = 4
    t = np.linspace(0, 200, 101)
    traces = np.random.default_rng(0).normal(size=(n_pulses, 101))
    r = ScanResult(
        scan_type="pulse_sequence",
        t=t,
        traces=traces,
        coords={"pulse_index": np.arange(n_pulses)},
        hydrophone_channel="A",
        units="mV",
    )
    assert r.vpp.shape == (n_pulses,)


def test_save_roundtrip_npz(clean_cwd, tmp_path):
    r = _fake_lateral()
    out = tmp_path / "lat.npz"
    r.save(out, save_txt=False)
    assert out.is_file()
    loaded = np.load(out, allow_pickle=True)
    # ScanResult.save writes traces under the ``outputs`` key.
    assert "outputs" in loaded.files
    assert loaded["outputs"].shape == r.traces.shape
    assert loaded["scan_type"].item() == "lateral"
