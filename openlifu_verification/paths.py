"""Centralized default paths.

Machine-local state (registries, hydrophone calibrations, transducer
pinmaps, last hydrophone position) lives in one setup data folder
shared by the GUI and every script (see :mod:`.setup_data`). Its
location is stored in :data:`SETUP_DATA_LOCATION_FILE` and defaults
to :data:`DEFAULT_SETUP_DATA_DIR`. New folders are seeded from the
template shipped inside the package.

Scan settings are NOT read from here: the standard configuration is
built into :class:`~openlifu_verification.ScanConfig`.
"""
from __future__ import annotations

from pathlib import Path

#: Setup data template shipped as package data (see :mod:`.setup_data`).
BUNDLED_SETUP_DATA_DIR = Path(__file__).resolve().parent / "setup_data_template"

#: Shipped calibration files. Last-resort search directory for bare
#: hydrophone IDs.
BUNDLED_CALIBRATIONS_DIR = BUNDLED_SETUP_DATA_DIR / "hydrophone_calibrations"

#: Per-user folder for app state.
USER_DIR = Path.home() / ".openlifu_verification"

#: Default setup data folder.
DEFAULT_SETUP_DATA_DIR = USER_DIR / "setup_data"

#: Text file holding the active setup data folder (written by the GUI
#: launcher and ``openlifu-setup-data``).
SETUP_DATA_LOCATION_FILE = USER_DIR / "setup_data_location.txt"
