"""Interactive CLI: run the full TXM characterization report.

Prompts (with cached defaults) for tester name + serial numbers,
connects to the hardware, runs every characterization phase, grades
the results against the criteria in ``scan_config.json``, and writes
the report bundle to
``test_reports/<TXM-SN>/<YYYYMMDD>_<HHMMSS>/``.

Examples::

    # Full run at the defaults (400 kHz, 20 V), interactive prompts
    python scripts/run_report.py --hydrophone 2246

    # 155 kHz, skip the voltage sweep, no HW config-write
    python scripts/run_report.py --frequency-khz 155 --skip-voltage

    # Dry-run (no hardware) to sanity-check the pipeline
    python scripts/run_report.py --dry-run --hydrophone 2246

    # Also load the freshly-generated device_config.json onto the TXM
    python scripts/run_report.py --write-config
"""
from __future__ import annotations

import argparse
import json
import logging
import logging.handlers
import sys
import threading
import time
from contextlib import ExitStack
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from pathlib import Path
from typing import Callable, Optional

from openlifu_verification import (
    Characterization,
    DryRunTank,
    Hydrophone,
    OperatorPrefs,
    RunAborted,
    RunHalted,
    ScanConfig,
    VerificationTank,
    characterization,
    paths,
    report_io,
    set_log_level,
)
from openlifu_verification.acceptance import SUPPORTED_FREQUENCIES_KHZ

logger = logging.getLogger(__name__)


DEFAULT_SCAN_CONFIG_PATH = paths.SCAN_CONFIG_PATH
DEFAULT_CALIBRATION_PATH = paths.HYDROPHONE_STATE_PATH
DEFAULT_OUTPUT_DIR = Path("test_reports")


def _configure_root_logger(log_file: Path | None = None) -> None:
    root = logging.getLogger()
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        root.addHandler(h)
    if log_file is not None:
        target = str(Path(log_file).resolve())
        already = any(isinstance(h, logging.FileHandler)
                      and h.baseFilename == target for h in root.handlers)
        if not already:
            fh = logging.FileHandler(log_file)
            fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
            root.addHandler(fh)
    root.setLevel(logging.INFO)


def _write_config_to_device(ver, config_path: Path, *, module: int = 0) -> bool:
    """Push a device_config.json onto the TXM via the SDK.

    The SDK exposes two entry points on ``TxDevice``:

    - ``write_config(cfg: LifuUserConfig, module: int) -> LifuUserConfig``
    - ``write_config_json(json_str: str, module: int) -> LifuUserConfig``

    We prefer ``write_config_json`` because it takes the raw JSON
    string and avoids having to hand-construct a ``LifuUserConfig``
    header (magic / version / seq / crc) — the SDK does that
    internally. If it's not available on this SDK version we fall
    back to constructing a ``LifuUserConfig`` and calling
    ``write_config``.

    Returns ``True`` on success. Any failure is logged and returns
    ``False`` so the CLI can continue.
    """
    # Read the file first — cheap failure mode that shouldn't need
    # the SDK.
    try:
        raw_json = Path(config_path).read_text(encoding="utf-8")
    except Exception as e:
        logger.error("Cannot read %s: %s", config_path, e)
        return False

    tx = ver.lifu.txdevice

    # Preferred path: JSON-string API.
    if hasattr(tx, "write_config_json"):
        try:
            tx.write_config_json(raw_json, module)
            logger.info("Wrote device_config.json to TXM module %d "
                        "via write_config_json.", module)
            return True
        except Exception as e:
            logger.error("write_config_json failed: %s", e)
            return False

    # Fallback: LifuUserConfig object API.
    if hasattr(tx, "write_config"):
        try:
            from openlifu_sdk.io.LIFUUserConfig import LifuUserConfig  # type: ignore
        except Exception as e:
            logger.error("Cannot import LifuUserConfig: %s", e)
            return False
        try:
            cfg_dict = json.loads(raw_json)
            cfg = LifuUserConfig(json_data=cfg_dict)
            tx.write_config(cfg, module)
            logger.info("Wrote device_config.json to TXM module %d "
                        "via write_config.", module)
            return True
        except Exception as e:
            logger.error("write_config failed: %s", e)
            return False

    logger.error("TX device has no write_config/write_config_json method.")
    return False


def _prompt_yes_no(prompt: str, *, default: bool = False) -> bool:
    yn = "Y/n" if default else "y/N"
    while True:
        raw = input(f"{prompt} [{yn}]: ").strip().lower()
        if not raw:
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run the full TXM characterization report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # --- Device / measurement ---
    p.add_argument("--frequency-khz", type=float,
                   default=VerificationTank.DEFAULT_FREQUENCY_KHZ,
                   choices=list(SUPPORTED_FREQUENCIES_KHZ),
                   help="Nominal center frequency of the TXM.")
    p.add_argument("--voltage", type=float, default=None,
                   help="HV rail (V) for peak scans + freq sweep. Default: "
                        "per-frequency (VerificationTank.DEFAULT_VOLTAGE_BY_FREQ_KHZ).")
    p.add_argument("--num-modules", type=int, default=1)
    p.add_argument("--ext-power", action="store_true",
                   help="Use the QPX600DP external HV supply (default: internal HV).")
    # --- Hydrophone ---
    p.add_argument("--hydrophone", type=str, default="",
                   help="Hydrophone calibration file or ID (e.g. '2246').")
    p.add_argument("--calibration-path", type=Path,
                   default=DEFAULT_CALIBRATION_PATH,
                   help="Auto-load / save destination for hydrophone_position.")
    # --- Report inputs / outputs ---
    p.add_argument("--scan-config", type=Path, default=DEFAULT_SCAN_CONFIG_PATH,
                   help="Path to scan_config.json (seeded from defaults if missing; "
                        "contains acceptance criteria + scan geometry + scope settings).")
    p.add_argument("--prefs", type=Path, default=None,
                   help="Path to operator prefs JSON (defaults to ~/.openlifu_verification/operator_prefs.json).")
    p.add_argument("--reports-root", type=Path, default=None,
                   help="Root reports folder; each run is written to "
                        "<root>/<TXM-SN>/<timestamp>/. Default: ./test_reports.")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Parent output directory (overrides --reports-root). "
                        "Default: <reports-root>/<TXM-SN>/.")
    p.add_argument("--no-prompt", action="store_true",
                   help="Skip the interactive prefs prompt; use cached values as-is.")
    p.add_argument("--no-start-prompt", action="store_true",
                   help="Skip the 'Press Enter to start' confirmation prompt "
                        "before the scan campaign kicks off. Implied by "
                        "--no-prompt; expose separately so callers (e.g. the "
                        "GUI launcher) can suppress just the start prompt "
                        "while still driving prefs collection themselves.")
    # --- Phase skips ---
    p.add_argument("--skip-2d", action="store_true",
                   help="Skip the 1-D + 2-D beam scans.")
    p.add_argument("--skip-frequency", action="store_true",
                   help="Skip the frequency sweep (implies --no-write-config).")
    p.add_argument("--skip-voltage", action="store_true",
                   help="Skip the voltage sweep.")
    # --- Device write-back ---
    p.add_argument("--write-config", action="store_true",
                   help="Push the freshly-generated device_config.json onto the TXM.")
    p.add_argument("--confirm-write-config", action="store_true",
                   help="Prompt before writing config to hardware (default: prompt).")
    # --- Misc ---
    p.add_argument("--dry-run", action="store_true",
                   help="Use DryRunTank instead of real hardware.")
    p.add_argument("--plot-peak", action="store_true",
                   help="Open a live matplotlib figure during the find_peak "
                        "stage so the operator can visually verify the "
                        "gradient-ascent search behavior.")
    p.add_argument("--log-file", type=Path, default=None,
                   help="Optional file to tee logs into.")
    verbosity = p.add_mutually_exclusive_group()
    verbosity.add_argument("--verbose", "-v", action="store_true")
    verbosity.add_argument("--quiet", "-q", action="store_true")
    return p


@dataclass
class RunResult:
    """Return value from :func:`run` capturing everything the GUI /
    caller needs to know after the pipeline finishes.

    ``passed`` reflects the overall PASS/FAIL verdict. ``run_dir``
    points at the timestamped output folder (``None`` if the run
    aborted before ``write_report`` ran). ``files`` maps short names
    (``"xlsx"``, ``"pdf"``, ``"device_config"``) to the corresponding
    Paths. ``exit_code`` is what ``main`` returns to the OS: 0 on
    pass, 1 on fail, 130 on user abort, 2 on unexpected error.
    ``aborted`` is set when the run was stopped via ``abort_event``.
    ``halted_reason`` is set (exit code 1, no report) when the depth
    calibration or peak search failed and the run was stopped.
    """
    passed: bool = False
    run_dir: Optional[Path] = None
    files: dict[str, Path] = field(default_factory=dict)
    exit_code: int = 1
    device_write_ok: Optional[bool] = None
    aborted: bool = False
    halted_reason: Optional[str] = None


#: Signature of the ``on_complete`` hook accepted by :func:`run`. The
#: second argument writes the run's device config to the TXM and
#: returns success; it is ``None`` when there is nothing writable
#: (report FAILED, or the frequency sweep was skipped).
OnComplete = Callable[["RunResult", Optional[Callable[[], bool]]], None]


def run(args: argparse.Namespace, *,
        abort_event: Optional[threading.Event] = None,
        on_complete: Optional[OnComplete] = None) -> RunResult:
    """Execute the report pipeline with parsed ``args``.

    Split out of :func:`main` so a GUI launcher can call the same
    code path in-process, then read ``RunResult.run_dir`` /
    ``RunResult.passed`` to build a completion dialog.

    Args:
        abort_event: When set, the characterization stops after the
            phase in progress and the result comes back with
            ``aborted=True`` (no report is written).
        on_complete: Called after the report is written, while the
            hardware is still connected (HV is switched off first).
            Receives the result and a device-write callable (see
            :data:`OnComplete`). The connection is closed when the
            hook returns, so a GUI can block here until the operator
            has decided whether to write.

    Every log record emitted during the run is also saved to
    ``<run_dir>/<stem>_console.log`` (buffered in memory until the
    report folder exists, then written live). Runs that abort or
    halt before the report is written leave no log file.
    """
    console_log = logging.handlers.MemoryHandler(
        capacity=sys.maxsize, flushLevel=logging.CRITICAL + 1
    )
    root = logging.getLogger()
    root.addHandler(console_log)
    try:
        return _run(args, abort_event, on_complete, console_log)
    finally:
        root.removeHandler(console_log)
        target = console_log.target
        console_log.close()  # flushes any tail into the log file
        if target is not None:
            target.close()


def _run(args: argparse.Namespace,
         abort_event: Optional[threading.Event],
         on_complete: Optional[OnComplete],
         console_log: logging.handlers.MemoryHandler) -> RunResult:
    _configure_root_logger(args.log_file)
    if getattr(args, "verbose", False):
        set_log_level("DEBUG", sdk_level="DEBUG")
    elif getattr(args, "quiet", False):
        set_log_level("WARNING")

    result = RunResult()

    # --- Prefs ---
    prefs = OperatorPrefs.load(args.prefs)
    # Auto-populate test_app_version (never prompted) and pre-fill
    # hydrophone_sn from the loaded calibration so the operator can
    # confirm or override it at the prompt.
    try:
        prefs.test_app_version = _pkg_version("openlifu-verification")
    except PackageNotFoundError:
        prefs.test_app_version = ""
    hydro_sn_from_file = ""
    if args.hydrophone:
        try:
            hydro_sn_from_file = str(
                Hydrophone(args.hydrophone).metadata.get("HYD_SN", "")
            )
        except Exception as e:
            logger.warning("Could not read hydrophone S/N from %r: %s",
                           args.hydrophone, e)
    if hydro_sn_from_file:
        prefs.hydrophone_sn = hydro_sn_from_file

    if not args.no_prompt:
        try:
            prefs.prompt_interactively(freq_kHz=args.frequency_khz)
        except (KeyboardInterrupt, EOFError):
            print("\nAborted at prompt.", file=sys.stderr)
            result.exit_code = 130
            return result
        prefs.save(args.prefs)
    logger.info("Operator prefs: tester=%r txm_sn=%r hydrophone_sn=%r app_ver=%r",
                prefs.tester_name, prefs.txm_sn, prefs.hydrophone_sn,
                prefs.test_app_version)

    # --- Scan configuration (seed if missing) ---
    # Bundles acceptance criteria + scan geometry + scope capture.
    scan_config = ScanConfig.load_or_create(args.scan_config)
    criteria = scan_config.acceptance

    voltage_V = (args.voltage if args.voltage is not None
                 else VerificationTank.default_voltage_for(args.frequency_khz))
    logger.info("HV rail: %g V at %g kHz", voltage_V, args.frequency_khz)

    # --- Tank ---
    if args.dry_run:
        logger.info("=== DRY RUN (no hardware) ===")
        tank_cm = DryRunTank(frequency=args.frequency_khz)
    else:
        tank_cm = VerificationTank(
            frequency=int(args.frequency_khz),
            num_modules=args.num_modules,
            ext_power_supply=args.ext_power,
            hydrophone_range_mv=scan_config.scope.hydrophone_range_mv,
            hydrophone=args.hydrophone or None,
            calibration_path=args.calibration_path or None,
        )

    with ExitStack() as stack:
        ver = stack.enter_context(tank_cm)
        if args.log_file and hasattr(ver, "add_log_file"):
            try:
                ver.add_log_file(str(args.log_file))
            except Exception:
                pass

        # Configure the LIFU pulse + HV rail so scans start off correct.
        if not args.dry_run:
            ver.apply_pulse(
                frequency_kHz=args.frequency_khz,
                voltage=voltage_V,
            )
            ver.enable_hv_output(wait=True)
        else:
            ver.enable_hv_output()

        # Give the operator a chance to verify HV is up + the tank is
        # ready before the (potentially long) scan campaign kicks off.
        # Skip when either --no-prompt or --no-start-prompt is set, or
        # in --dry-run (nothing to physically verify).
        suppress_start = (args.no_prompt or args.no_start_prompt
                          or args.dry_run)
        if not suppress_start:
            try:
                input("\nReady to begin characterization. "
                      "Press Enter to start (Ctrl+C to abort)... ")
            except (KeyboardInterrupt, EOFError):
                print("\nAborted before start.", file=sys.stderr)
                result.exit_code = 130
                return result

        # --- Run the characterization ---
        chz = Characterization(
            ver,
            prefs=prefs,
            criteria=criteria,
            scan_config=scan_config,
            frequency_kHz=args.frequency_khz,
            voltage_V=voltage_V,
            plot=args.plot_peak,
        )
        t_start = time.perf_counter()
        try:
            report = chz.run(
                skip_2d=args.skip_2d,
                skip_frequency=args.skip_frequency,
                skip_voltage=args.skip_voltage,
                abort_event=abort_event,
            )
        except RunAborted:
            logger.warning("Run aborted by operator; no report written.")
            result.exit_code = 130
            result.aborted = True
            return result
        except RunHalted as e:
            logger.error("Run halted: %s No report written.", e)
            result.exit_code = 1
            result.halted_reason = str(e)
            return result
        except KeyboardInterrupt:
            print("\nInterrupted during characterization.", file=sys.stderr)
            result.exit_code = 130
            return result
        elapsed = time.perf_counter() - t_start
        logger.info("Characterization finished in %.1f s. Overall: %s",
                    elapsed, "PASS" if report.overall_pass else "FAIL")

        # --- Persist artifacts ---
        # write_report defaults to <reports_root>/<TXM-SN>/<ts>/ when
        # output_dir is None.
        run_dir = report_io.write_report(
            report,
            args.output_dir,
            reports_root=args.reports_root,
            write_device_config_json=not args.skip_frequency,
        )
        # Match the on-disk stem (SN + test start timestamp) so the
        # printed paths point at real files.
        file_stem = report_io.report_file_stem(report)
        log_path = run_dir / f"{file_stem}_console.log"
        log_fh = logging.FileHandler(log_path, encoding="utf-8")
        log_fh.setFormatter(logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        ))
        console_log.setTarget(log_fh)
        console_log.flush()
        # Write each later record straight through (device write etc.).
        console_log.flushLevel = logging.NOTSET
        xlsx_path = run_dir / f"{file_stem}_Report.xlsx"
        pdf_path = run_dir / f"{file_stem}_Report.pdf"
        config_path = run_dir / f"{file_stem}_device_config.json"
        print(f"\nReport directory: {run_dir}")
        print(f"  XLSX : {xlsx_path}")
        print(f"  PDF  : {pdf_path}")
        if not args.skip_frequency:
            print(f"  JSON : {config_path}")
        print(f"  LOG  : {log_path}")
        print(f"  Verdict: {'PASS' if report.overall_pass else 'FAIL'}")

        result.passed = bool(report.overall_pass)
        result.run_dir = run_dir
        result.files["xlsx"] = xlsx_path
        result.files["pdf"] = pdf_path
        result.files["console_log"] = log_path
        if not args.skip_frequency:
            result.files["device_config"] = config_path
        result.exit_code = 0 if result.passed else 1

        # --- Optionally push config back onto the device ---
        # Only fire when the run PASSED so we never overwrite a
        # good on-device config with the calibration numbers from a
        # failing run. This is what backs the GUI's "save calibration
        # data to device" checkbox: caller sets --write-config, and
        # the pass gate here decides whether the write actually
        # happens.
        logger.info(
            "Device-write decision: write_config=%s dry_run=%s "
            "skip_frequency=%s passed=%s",
            bool(getattr(args, "write_config", False)),
            bool(getattr(args, "dry_run", False)),
            bool(getattr(args, "skip_frequency", False)),
            bool(result.passed),
        )
        if args.write_config and not args.dry_run:
            if args.skip_frequency:
                logger.warning("--skip-frequency set; no device_config.json to write.")
            elif not result.passed:
                logger.warning(
                    "Report FAILED; skipping --write-config (device "
                    "calibration is only pushed on PASS)."
                )
                print("  Write to device: SKIPPED (report FAILED)")
            else:
                do_write = True
                if args.confirm_write_config:
                    do_write = _prompt_yes_no(
                        f"Write {config_path.name} to hardware?",
                        default=False,
                    )
                if do_write:
                    ok = _write_config_to_device(ver, config_path)
                    result.device_write_ok = bool(ok)
                    print(f"  Write to device: {'OK' if ok else 'FAILED'}")

        if on_complete is not None:
            write_fn: Optional[Callable[[], bool]] = None
            if result.passed and not args.skip_frequency:
                if args.dry_run:
                    def write_fn() -> bool:
                        logger.info("DRY RUN: would write %s to device.",
                                    config_path.name)
                        return True
                else:
                    def write_fn() -> bool:
                        return _write_config_to_device(ver, config_path)
            # Don't leave HV up while the operator reviews the result.
            try:
                ver.disable_hv_output()
            except Exception as e:  # noqa: BLE001
                logger.warning("Could not disable HV before review: %s", e)
            on_complete(result, write_fn)

    return result


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    result = run(args)
    return result.exit_code


if __name__ == "__main__":
    sys.exit(main())
