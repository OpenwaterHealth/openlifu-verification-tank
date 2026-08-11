"""Tests for :mod:`openlifu_verification.paths`."""
from pathlib import Path

from openlifu_verification import paths


def test_paths_constants_are_paths():
    assert isinstance(paths.REPO_ROOT, Path)
    assert isinstance(paths.CONFIG_DIR, Path)
    assert isinstance(paths.SCAN_CONFIG_PATH, Path)
    assert isinstance(paths.HYDROPHONE_STATE_PATH, Path)
    assert isinstance(paths.CALIBRATIONS_DIR, Path)
    assert isinstance(paths.REPO_CALIBRATIONS_DIR, Path)


def test_cwd_relative_paths_are_relative():
    """Anything a script writes to must be CWD-relative so tests /
    users can point a fresh working directory at it."""
    assert not paths.CONFIG_DIR.is_absolute()
    assert not paths.SCAN_CONFIG_PATH.is_absolute()
    assert not paths.HYDROPHONE_STATE_PATH.is_absolute()
    assert not paths.CALIBRATIONS_DIR.is_absolute()


def test_repo_calibrations_dir_exists_and_ships_a_file():
    """The repo-shipped fallback should have at least one calibration."""
    assert paths.REPO_CALIBRATIONS_DIR.is_dir()
    hits = list(paths.REPO_CALIBRATIONS_DIR.glob("HNR0500-*.txt"))
    assert hits, f"no HNR0500 calibrations found in {paths.REPO_CALIBRATIONS_DIR}"


def test_scan_config_path_lives_under_config_dir():
    """Both editable files must live under ``config/`` so a single
    gitignore rule covers them."""
    assert paths.SCAN_CONFIG_PATH.parent == paths.CONFIG_DIR
    assert paths.HYDROPHONE_STATE_PATH.parent == paths.CONFIG_DIR
    assert paths.CALIBRATIONS_DIR.parent == paths.CONFIG_DIR
