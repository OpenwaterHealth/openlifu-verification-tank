"""Operator-preferences cache.

Persists the last-entered values for interactive test-report prompts
(tester name, TXM serial, console serial, hydrophone S/N, etc.) so a
subsequent run can pre-fill them.

Stored as JSON at ``~/.openlifu_verification/operator_prefs.json``.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_PREFS_PATH = Path.home() / ".openlifu_verification" / "operator_prefs.json"


@dataclass
class OperatorPrefs:
    """Cacheable interactive-prompt values.

    All fields default to empty strings so a freshly created cache is
    valid. Only string-typed fields are supported so the cache round-
    trips cleanly through JSON.

    ``test_app_version`` is *not* prompted — it is populated by the
    CLI from ``openlifu_verification``'s installed version.
    ``hydrophone_sn`` is pre-filled from the loaded
    :class:`Hydrophone` metadata (when a calibration is available) and
    the operator is asked to confirm or override it.
    """
    tester_name: str = ""
    test_app_version: str = ""
    hydrophone_sn: str = ""
    txm_sn: str = ""
    console_sn: str = ""

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "OperatorPrefs":
        """Load prefs from ``path`` (default: ``DEFAULT_PREFS_PATH``).

        Missing file returns a fresh, empty instance. Malformed JSON
        logs a warning and returns an empty instance.
        """
        if path is None:
            path = DEFAULT_PREFS_PATH
        path = Path(path)
        if not path.is_file():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Could not read operator prefs at %s: %s", path, e)
            return cls()
        # Only pull known fields; ignore anything unexpected.
        known = {f.name for f in fields(cls)}
        filtered = {k: str(v) for k, v in data.items() if k in known}
        return cls(**filtered)

    def save(self, path: Optional[Path] = None) -> Path:
        """Persist prefs to ``path`` (default: ``DEFAULT_PREFS_PATH``)."""
        if path is None:
            path = DEFAULT_PREFS_PATH
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        logger.info("Saved operator prefs to %s", path)
        return path

    def prompt_interactively(self, *, freq_kHz: Optional[float] = None) -> None:
        """Update fields in place by prompting the user.

        Each prompt shows the cached value in ``[brackets]``; pressing
        Enter accepts it. Empty response *and* empty cache re-prompts
        for required fields.

        ``freq_kHz`` is displayed but not stored (it comes from the CLI
        flag) \u2014 shown so the operator can double-check they set the
        right label.
        """
        if freq_kHz is not None:
            print(f"\n--- Test-report metadata (nominal freq = {freq_kHz:g} kHz) ---")
        else:
            print("\n--- Test-report metadata ---")
        self.tester_name    = _ask("Tester name",     self.tester_name,     required=True)
        # test_app_version is set by the CLI from the installed package
        # version — don't prompt for it. hydrophone_sn is pre-filled by
        # the CLI from the loaded Hydrophone calibration; we still
        # prompt so the operator can confirm or correct it.
        self.hydrophone_sn  = _ask("Hydrophone S/N",  self.hydrophone_sn,   required=True)
        self.txm_sn         = _ask("TXM S/N",         self.txm_sn,          required=True)
        self.console_sn     = _ask("Console S/N",     self.console_sn,      required=True)


def _ask(label: str, default: str, *, required: bool) -> str:
    """Prompt with a bracketed default. Blank input keeps the default."""
    while True:
        shown = f" [{default}]" if default else ""
        raw = input(f"  {label}{shown}: ").strip()
        value = raw if raw else default
        if value or not required:
            return value
        print("    (required)")
