"""conftest.py: shared fixtures for the openlifu_verification test suite.

Runs from a clean, empty CWD in a tmp dir so tests can exercise the
CWD-relative ``config/`` layout without polluting the repo.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest


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
