"""Centralized default paths for configuration and reference data.

All machine-local state lives under ``config/`` in the CWD
so scripts, notebooks, and the report generator agree on one spot::

    <cwd>/
      config/
        hydrophone.json           # last-known hydrophone position + ID
        hydrophone_calibrations/  # HNR0500-<sn>_*.txt files

Scan settings are NOT read from here: the standard configuration is
built into :class:`~openlifu_verification.ScanConfig`.

The repo ships a ``setup_data/`` template (hydrophone calibrations and
empty tester / console registries, see :mod:`.setup_data`). The GUI
copies it into its setup data folder on first use; bare hydrophone IDs
in scripts still resolve against the bundled calibrations.
"""
from __future__ import annotations

from pathlib import Path

#: Repo root (parent of ``openlifu_verification/``).
REPO_ROOT = Path(__file__).resolve().parent.parent

# ----------------------------------------------------------------------
# CWD-relative defaults (what user commands write to / read from).
# ----------------------------------------------------------------------
#: User-editable configuration folder (relative to CWD).
CONFIG_DIR = Path("config")

#: Persisted hydrophone state (position + last-used ID).
HYDROPHONE_STATE_PATH = CONFIG_DIR / "hydrophone.json"

#: Local calibration-file cache (CWD-relative).
CALIBRATIONS_DIR = CONFIG_DIR / "hydrophone_calibrations"

# ----------------------------------------------------------------------
# Repo-shipped fallbacks (absolute).
# ----------------------------------------------------------------------
#: Repo-shipped setup data template (see :mod:`.setup_data`).
BUNDLED_SETUP_DATA_DIR = REPO_ROOT / "setup_data"

#: Repo-bundled calibration files. Used as the last-resort search
#: directory so the shipped hydrophones "just work" from any CWD.
REPO_CALIBRATIONS_DIR = BUNDLED_SETUP_DATA_DIR / "hydrophone_calibrations"

#: Default setup data folder for the GUI (changeable in the launcher).
DEFAULT_SETUP_DATA_DIR = Path.home() / ".openlifu_verification" / "setup_data"
