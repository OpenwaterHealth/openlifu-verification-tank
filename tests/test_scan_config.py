"""Tests for :class:`openlifu_verification.ScanConfig`.

Focuses on load / save / seed round-trip and the new
``sos_water_m_per_s`` field.
"""
from __future__ import annotations

import json

from openlifu_verification import ScanConfig
from openlifu_verification import paths


def test_defaults_roundtrip(clean_cwd):
    """A default config should survive a save + reload round-trip."""
    cfg = ScanConfig()
    path = paths.SCAN_CONFIG_PATH  # config/scan_config.json in CWD
    cfg.save(path)
    assert path.is_file()
    reloaded = ScanConfig.from_file(path)
    assert reloaded == cfg


def test_load_or_create_seeds_missing_file(clean_cwd):
    """``load_or_create`` should write a defaults file on first call
    so users always end up with an editable copy."""
    path = paths.SCAN_CONFIG_PATH
    assert not path.exists()
    cfg = ScanConfig.load_or_create(path)
    assert path.is_file()
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert "sos_water_m_per_s" in on_disk
    assert on_disk["sos_water_m_per_s"] == cfg.sos_water_m_per_s


def test_unknown_top_level_keys_are_ignored(clean_cwd):
    """Extra keys in a hand-edited config must not crash the loader
    (forward compatibility for older release binaries)."""
    path = paths.SCAN_CONFIG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"scope": {}, "future_key": {"anything": 1}}
    path.write_text(json.dumps(payload), encoding="utf-8")
    cfg = ScanConfig.from_file(path)
    # Should have fallen through to defaults everywhere.
    assert cfg.sos_water_m_per_s == ScanConfig().sos_water_m_per_s


def test_scope_kwargs_shape():
    """``scope_kwargs`` must return only the keys the scope-driven
    methods actually accept."""
    cfg = ScanConfig()
    kw = cfg.scope_kwargs()
    assert set(kw) >= {"time_start_s", "time_stop_s", "sampling_interval_ns"}
    # Nothing weird slipped in that the scan_* methods wouldn't take.
    assert all(isinstance(v, (int, float)) for v in kw.values())
