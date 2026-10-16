"""Setup data: registries of the testers, consoles and hydrophones the
GUI accepts, plus the hydrophone calibration files and per-tank state.

Layout::

    <root>/
      testers/testers.json                      {id: {"Name": name}}
      consoles/consoles.json                    {sn: {"HWID": hwid}}
      hydrophone_calibrations/hydrophones.json  {id: {"Calibration": filename}}
      hydrophone_calibrations/<calibration files>
      hydrophone.json                           last hydrophone position
      last_setup.json                           last tester / console / hydrophone

A new folder is seeded from the repo's ``setup_data/`` template.
"""
from __future__ import annotations

import json
import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from . import paths

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IdFormat:
    """Valid format of a scanned ID (matched against the whole value)."""
    pattern: re.Pattern
    hint: str

    def matches(self, value: str) -> bool:
        return self.pattern.fullmatch(value) is not None


#: TXM serial numbers are format-checked only (no registry).
TXM_SN = IdFormat(re.compile(r"WLM(24|44)Q\d{5}"), "WLM24Q/WLM44Q#####")

#: Model code in a TXM S/N ("WLM<code>Q...") -> nominal frequency.
TXM_FREQUENCY_KHZ_BY_MODEL = {"24": 155.0, "44": 400.0}


def txm_frequency_kHz(txm_sn: str) -> Optional[float]:
    """Nominal drive frequency encoded in ``txm_sn`` (``None`` if the
    S/N isn't a valid TXM S/N)."""
    if not TXM_SN.matches(txm_sn):
        return None
    return TXM_FREQUENCY_KHZ_BY_MODEL[txm_sn[3:5]]


@dataclass(frozen=True)
class Registry:
    """One registry file: ``{id: {field: value}}``."""
    kind: str          # "tester" / "console" / "hydrophone"
    folder: str
    filename: str
    field: str         # metadata key stored for each ID
    id_format: IdFormat


TESTERS = Registry("tester", "testers", "testers.json", "Name",
                   IdFormat(re.compile(r".+"), "any text"))
CONSOLES = Registry("console", "consoles", "consoles.json", "HWID",
                    IdFormat(re.compile(r"WL134Q\d{5}"), "WL134Q#####"))
HYDROPHONES = Registry("hydrophone", "hydrophone_calibrations",
                       "hydrophones.json", "Calibration",
                       IdFormat(re.compile(r"HYDRO\d{4}"), "HYDRO####"))
REGISTRIES = (TESTERS, CONSOLES, HYDROPHONES)


class SetupData:
    """Read / update the setup data folder at ``root``."""

    def __init__(self, root: Path):
        self.root = Path(root)

    @property
    def calibrations_dir(self) -> Path:
        return self.root / HYDROPHONES.folder

    @property
    def hydrophone_state_path(self) -> Path:
        """Hydrophone position file passed to ``VerificationTank``."""
        return self.root / "hydrophone.json"

    @property
    def last_setup_path(self) -> Path:
        return self.root / "last_setup.json"

    def registry_path(self, registry: Registry) -> Path:
        return self.root / registry.folder / registry.filename

    # ------------------------------------------------------------------
    # Registries
    # ------------------------------------------------------------------
    def load(self, registry: Registry) -> dict[str, dict]:
        """Return the registry contents (``{}`` if the file is missing).

        Raises ``ValueError`` if the file is not a JSON object of
        objects."""
        path = self.registry_path(registry)
        if not path.is_file():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not all(
                isinstance(v, dict) for v in data.values()):
            raise ValueError(f"{path} must map each ID to an object")
        return data

    def lookup(self, registry: Registry,
               key: str) -> tuple[bool, Optional[dict]]:
        """Return ``(found, metadata)`` for ``key`` (metadata ``None``
        when not found)."""
        meta = self.load(registry).get(key)
        return meta is not None, meta

    def add(self, registry: Registry, key: str, value: str) -> None:
        """Register ``key`` with ``{registry.field: value}``."""
        if not registry.id_format.matches(key):
            raise ValueError(f"{key!r} is not a valid {registry.kind} ID "
                             f"({registry.id_format.hint})")
        data = self.load(registry)
        data[key] = {registry.field: value}
        path = self.registry_path(registry)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
        logger.info("Added %s %s to %s", registry.kind, key, path)

    def add_hydrophone(self, key: str, calibration_file: Path) -> None:
        """Copy ``calibration_file`` into the calibrations folder and
        register it for hydrophone ``key``.

        The file must parse as a hydrophone calibration. An existing
        file of the same name is only accepted if it is identical."""
        from .hydrophone import Hydrophone

        src = Path(calibration_file)
        if Hydrophone(src).sensitivity_interp is None:
            raise ValueError(f"{src.name} has no FREQ_MHz / SENS_VPERPA "
                             "calibration data")
        dest = self.calibrations_dir / src.name
        if dest.exists():
            if dest.read_bytes() != src.read_bytes():
                raise FileExistsError(
                    f"A different {dest.name} already exists in "
                    f"{self.calibrations_dir}")
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
        self.add(HYDROPHONES, key, src.name)

    def calibration_path(self, hydrophone_id: str) -> Optional[Path]:
        """Path of the calibration file registered for
        ``hydrophone_id`` (``None`` if not registered)."""
        found, meta = self.lookup(HYDROPHONES, hydrophone_id)
        if not found or not meta.get(HYDROPHONES.field):
            return None
        return self.calibrations_dir / meta[HYDROPHONES.field]

    # ------------------------------------------------------------------
    # Last setup (only applied with --use-last-setup)
    # ------------------------------------------------------------------
    def load_last_setup(self) -> dict[str, str]:
        """Return ``{registry.kind: id}`` from the last Start (``{}`` if
        missing or unreadable)."""
        try:
            data = json.loads(self.last_setup_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return {k: str(v) for k, v in data.items()} if isinstance(data, dict) else {}

    def save_last_setup(self, ids: dict[str, str]) -> None:
        self.last_setup_path.write_text(json.dumps(ids, indent=2) + "\n",
                                        encoding="utf-8")

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def is_initialized(self) -> bool:
        return all(self.registry_path(r).is_file() for r in REGISTRIES)

    def initialize(self, template: Path = paths.BUNDLED_SETUP_DATA_DIR) -> None:
        """Copy every file from ``template`` that doesn't already
        exist under :attr:`root` (never overwrites)."""
        for src in Path(template).rglob("*"):
            dest = self.root / src.relative_to(template)
            if src.is_file() and not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)
        logger.info("Initialized setup data at %s from %s", self.root, template)
