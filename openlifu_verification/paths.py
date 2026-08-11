"""Centralized default paths for configuration and reference data.

All user-editable configuration lives under ``config/`` in the CWD
so scripts, notebooks, and the report generator agree on one spot::

    <cwd>/
      config/
        scan_config.json          # scan geometry + acceptance criteria
        hydrophone.json           # last-known hydrophone position + ID
        hydrophone_calibrations/  # HNR0500-<sn>_*.txt files

The repo ships a fallback ``config/hydrophone_calibrations/`` folder
so hydrophones ship with the source tree; the CWD copy always wins
if present.
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

#: Editable scan-configuration JSON path.
SCAN_CONFIG_PATH = CONFIG_DIR / "scan_config.json"

#: Persisted hydrophone state (position + last-used ID).
HYDROPHONE_STATE_PATH = CONFIG_DIR / "hydrophone.json"

#: Local calibration-file cache (CWD-relative).
CALIBRATIONS_DIR = CONFIG_DIR / "hydrophone_calibrations"

# ----------------------------------------------------------------------
# Repo-shipped fallbacks (absolute).
# ----------------------------------------------------------------------
#: Repo-bundled calibration files. Used as the last-resort search
#: directory so the shipped hydrophones "just work" from any CWD.
REPO_CALIBRATIONS_DIR = REPO_ROOT / "config" / "hydrophone_calibrations"
