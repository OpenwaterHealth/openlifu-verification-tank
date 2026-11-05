"""Setup data: registries of the testers, consoles and hydrophones the
GUI accepts, plus the hydrophone calibration files, transducer pinmaps
and per-tank state. One folder is shared by the GUI and the scripts.

Layout::

    <root>/
      testers/testers.json                      {id: {"Name": name}}
      consoles/consoles.json                    {sn: {"HWID": hwid}}
      hydrophone_calibrations/hydrophones.json  {id: {"Calibration": filename}}
      hydrophone_calibrations/<calibration files>
      transducers/openlifu_<n>x.json            transducer pinmaps
      hydrophone.json                           last hydrophone position + ID
      last_setup.json                           last tester / console / hydrophone

Missing files are copied from the template shipped in the package
(:data:`paths.BUNDLED_SETUP_DATA_DIR`); existing files are never
overwritten. Set up or switch the active folder with::

    openlifu-setup-data [FOLDER]
"""
from __future__ import annotations

import argparse
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
                    IdFormat(re.compile(r"WL1(1|3)4Q\d{5}"),
                             "WL114Q/WL134Q#####"))
HYDROPHONES = Registry("hydrophone", "hydrophone_calibrations",
                       "hydrophones.json", "Calibration",
                       IdFormat(re.compile(r"HYDRO\d{4}"), "HYDRO####"))
REGISTRIES = (TESTERS, CONSOLES, HYDROPHONES)


class SetupData:
    """Read / update the setup data folder at ``root``."""

    def __init__(self, root: Path):
        self.root = Path(root)

    @classmethod
    def active(cls) -> "SetupData":
        """The folder saved by :meth:`make_active`, else
        :data:`paths.DEFAULT_SETUP_DATA_DIR`."""
        try:
            saved = paths.SETUP_DATA_LOCATION_FILE.read_text(
                encoding="utf-8").strip()
        except OSError:
            saved = ""
        return cls(Path(saved) if saved else paths.DEFAULT_SETUP_DATA_DIR)

    def make_active(self) -> None:
        """Save :attr:`root` as the folder the GUI and scripts use."""
        location = paths.SETUP_DATA_LOCATION_FILE
        location.parent.mkdir(parents=True, exist_ok=True)
        location.write_text(str(self.root.resolve()) + "\n", encoding="utf-8")

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

    def transducer_path(self, num_modules: int) -> Path:
        return self.root / "transducers" / f"openlifu_{num_modules}x.json"

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
    def _missing_files(self, template: Optional[Path]) -> list[Path]:
        """Template files (relative paths) not present under root."""
        template = Path(template or paths.BUNDLED_SETUP_DATA_DIR)
        return [src.relative_to(template) for src in template.rglob("*")
                if src.is_file()
                and not (self.root / src.relative_to(template)).exists()]

    def is_initialized(self, template: Optional[Path] = None) -> bool:
        return not self._missing_files(template)

    def initialize(self, template: Optional[Path] = None) -> list[Path]:
        """Copy every file from ``template`` (default: the bundled
        template) that doesn't already exist under :attr:`root` (never
        overwrites). Returns the copied relative paths."""
        template = Path(template or paths.BUNDLED_SETUP_DATA_DIR)
        missing = self._missing_files(template)
        for rel in missing:
            dest = self.root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(template / rel, dest)
        if missing:
            logger.info("Copied %d template file(s) into setup data at %s",
                        len(missing), self.root)
        return missing


def main(argv: Optional[list[str]] = None) -> int:
    """``openlifu-setup-data``: create / update the setup data folder
    and make it the one the GUI and scripts use."""
    parser = argparse.ArgumentParser(
        prog="openlifu-setup-data",
        description=("Copy any missing template files (registries, "
                     "hydrophone calibrations, transducer pinmaps) into "
                     "the setup data folder and make it the active one. "
                     "Existing files are never overwritten."),
    )
    parser.add_argument(
        "folder", nargs="?", type=Path, default=None,
        help=("Setup data folder. Default: the active folder "
              f"({SetupData.active().root})."),
    )
    args = parser.parse_args(argv)

    setup = SetupData(args.folder) if args.folder else SetupData.active()
    copied = setup.initialize()
    setup.make_active()
    for rel in copied:
        print(f"  added {rel.as_posix()}")
    print(f"Setup data folder: {setup.root.resolve()}")
    for registry in REGISTRIES:
        print(f"  {registry.kind}s: {len(setup.load(registry))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
