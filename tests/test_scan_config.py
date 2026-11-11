"""Tests for :class:`openlifu_verification.ScanConfig`.

The built-in defaults are the standard production configuration, so
they are pinned here; overrides are strict and detectable.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from openlifu_verification import AcceptanceCriteria, ScanConfig
from openlifu_verification.acceptance import SUPPORTED_FREQUENCIES_KHZ

_STANDARD_PLAN = {
    "lateral_1d": {"extent_mm": 9.0, "points": 37},
    "elevation_1d": {"extent_mm": 9.0, "points": 37},
    "axial_1d": {"extent_mm": 20.0, "points": 41},
    "scan_2d": {"extent_mm": 2.0, "points": 9},
    "frequency_sweep": {"offsets_kHz": [-25.0, -20.0, -15.0, -10.0, -5.0, 0.0,
                                        5.0, 10.0, 15.0, 20.0, 25.0]},
    "voltage_sweep": {"voltages_V": [10.0, 15.0, 20.0, 25.0, 30.0],
                      "range_mv": 200},
}


def test_standard_configuration_is_pinned():
    """Changing a standard setting must be a deliberate, reviewed edit
    of this test as well as of ``scan_config.py``."""
    d = asdict(ScanConfig())
    assert d["scope"] == {
        "time_start_us": -10.0,
        "time_stop_us": 120.0,
        "sampling_interval_ns": 100.0,
        "hydrophone_range_mv": 200,
        "voltage_scan_headroom_pct": 30.0,
        "n_averages": 3,
        "pnp_n_averages": 5,
    }
    assert d["by_freq_kHz"] == {"155": _STANDARD_PLAN, "400": _STANDARD_PLAN}
    assert d["sos_water_m_per_s"] == 1500.0
    assert d["acceptance"] == asdict(AcceptanceCriteria())
    assert ScanConfig().differences_from_standard() == []


@pytest.mark.parametrize("freq", SUPPORTED_FREQUENCIES_KHZ)
def test_every_supported_frequency_has_its_own_plan(freq):
    cfg = ScanConfig()
    plan = cfg.plan_for(freq)
    other = [cfg.plan_for(f) for f in SUPPORTED_FREQUENCIES_KHZ if f != freq]
    assert all(plan is not p for p in other)


def test_missing_frequency_plan_is_rejected():
    plans = ScanConfig().by_freq_kHz
    del plans["155"]
    with pytest.raises(ValueError, match="no entry for 155"):
        ScanConfig(by_freq_kHz=plans)


def _write(tmp_path, payload) -> Path:
    path = tmp_path / "override.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_partial_override_changes_only_given_keys(tmp_path):
    path = _write(tmp_path, {
        "scope": {"n_averages": 1},
        "by_freq_kHz": {"400": {"lateral_1d": {"points": 11},
                                "voltage_sweep": {"voltages_V": [10, 20]}}},
    })
    cfg = ScanConfig.with_overrides(path)
    assert cfg.scope.n_averages == 1
    assert cfg.plan_for(400).lateral_1d.points == 11
    assert cfg.plan_for(400).lateral_1d.extent_mm == 9.0
    assert cfg.plan_for(400).voltage_sweep.voltages_V == [10.0, 20.0]
    assert cfg.plan_for(155) == ScanConfig().plan_for(155)
    assert cfg.differences_from_standard() == [
        "by_freq_kHz.400.lateral_1d.points = 11 (standard 37)",
        "by_freq_kHz.400.voltage_sweep.voltages_V = [10.0, 20.0] "
        "(standard [10.0, 15.0, 20.0, 25.0, 30.0])",
        "scope.n_averages = 1 (standard 3)",
    ]


@pytest.mark.parametrize("payload, match", [
    ({"lateral_1d": {"points": 5}}, "unknown key 'lateral_1d'"),
    ({"scope": {"n_avg": 1}}, "unknown key 'scope.n_avg'"),
    ({"by_freq_kHz": {"250": {}}}, "unknown frequency"),
    ({"scope": {"n_averages": 1.5}}, "must be an integer"),
    ({"scope": {"time_stop_us": "120"}}, "must be a number"),
    ({"by_freq_kHz": {"400": {"frequency_sweep": {"offsets_kHz": 5}}}},
     "list of numbers"),
    ({"acceptance": {}}, "acceptance criteria cannot be overridden"),
])
def test_invalid_override_is_rejected(tmp_path, payload, match):
    with pytest.raises(ValueError, match=match):
        ScanConfig.with_overrides(_write(tmp_path, payload))


def test_save_writes_effective_config(tmp_path):
    path = ScanConfig().save(tmp_path / "cfg.json")
    assert json.loads(path.read_text(encoding="utf-8")) == asdict(ScanConfig())


def test_scope_kwargs_shape():
    """``scope_kwargs`` must return only the keys the scope-driven
    methods actually accept."""
    cfg = ScanConfig()
    kw = cfg.scope_kwargs()
    assert set(kw) >= {"time_start_s", "time_stop_s", "sampling_interval_ns"}
    # Nothing weird slipped in that the scan_* methods wouldn't take.
    assert all(isinstance(v, (int, float)) for v in kw.values())


def test_gui_section_is_not_a_measurement_override(tmp_path):
    cfg = ScanConfig.with_overrides(
        _write(tmp_path, {"gui": {"add_items": True}}))
    assert cfg.gui.add_items is True
    assert not cfg.overrides_measurement
    assert cfg.differences_from_standard() == []
    assert ScanConfig().gui.add_items is False


def test_any_measurement_key_is_an_override(tmp_path):
    """Even a measurement setting restated at its standard value counts."""
    cfg = ScanConfig.with_overrides(_write(tmp_path, {
        "gui": {"add_items": True}, "scope": {"n_averages": 3}}))
    assert cfg.overrides_measurement
    assert not ScanConfig().overrides_measurement


@pytest.mark.parametrize("value", [1, "true", None])
def test_gui_add_items_must_be_boolean(tmp_path, value):
    with pytest.raises(ValueError, match="must be true or false"):
        ScanConfig.with_overrides(
            _write(tmp_path, {"gui": {"add_items": value}}))


def test_bundled_gui_config_enables_add_items(repo_root):
    cfg = ScanConfig.with_overrides(repo_root / "config/gui/add_items.json")
    assert cfg.gui.add_items and not cfg.overrides_measurement
