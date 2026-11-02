"""conftest.py: shared fixtures for the openlifu_verification test suite."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from openlifu_verification import paths


@pytest.fixture(autouse=True)
def isolated_user_dir(tmp_path, monkeypatch) -> Path:
    """Point the per-user setup data location at a tmp dir so tests
    never read or change the machine's active setup data folder."""
    user = tmp_path / "user"
    monkeypatch.setattr(paths, "DEFAULT_SETUP_DATA_DIR", user / "setup_data")
    monkeypatch.setattr(paths, "SETUP_DATA_LOCATION_FILE",
                        user / "setup_data_location.txt")
    return user


@pytest.fixture()
def clean_cwd(tmp_path, monkeypatch):
    """Change into an empty tmp directory for the duration of the test.

    Useful for anything that reads or writes ``config/*.json`` at
    module scope — keeps the repo tree clean.
    """
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture()
def repo_root() -> Path:
    """Absolute path to the repository root."""
    return Path(__file__).resolve().parent.parent
