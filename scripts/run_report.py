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
import sys
import time
from contextlib import ExitStack
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from pathlib import Path

from openlifu_verification import (
    Characterization,
    DryRunTank,
    Hydrophone,
    OperatorPrefs,
    ScanConfig,
    VerificationTank,
    characterization,
    paths,
    report_io,
    set_log_level,
)

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
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
        root.addHandler(fh)
    root.setLevel(logging.INFO)


def _write_config_to_device(ver, config_path: Path, *, module: int = 0) -> bool:
    """Push a device_config.json onto the TXM via the SDK.

    Returns ``True`` on success. Any failure is logged and returns
    ``False`` so the CLI can continue.
    """
    try:
        from openlifu_sdk.io.LIFUConfig import LIFUConfig  # type: ignore
    except Exception as e:
        logger.error("Cannot import openlifu_sdk.io.LIFUConfig: %s", e)
        return False

    try:
        cfg_dict = json.loads(Path(config_path).read_text(encoding="utf-8"))
    except Exception as e:
        logger.error("Cannot read %s: %s", config_path, e)
        return False

    try:
        cfg = LIFUConfig.from_dict(cfg_dict) if hasattr(LIFUConfig, "from_dict") \
            else LIFUConfig(**cfg_dict)
    except Exception as e:
        logger.error("Cannot instantiate LIFUConfig from %s: %s", config_path, e)
        return False

    try:
        tx = ver.lifu.txdevice
        # The SDK's method name has changed a few times; try both.
        if hasattr(tx, "write_config"):
            tx.write_config(cfg, module)
        elif hasattr(tx, "write_config_json"):
            tx.write_config_json(cfg_dict, module)
        else:
            logger.error("TX device has no write_config/write_config_json method.")
            return False
        logger.info("Wrote device_config.json to TXM module %d.", module)
        return True
    except Exception as e:
        logger.error("write_config failed: %s", e)
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
                   choices=[155.0, 400.0],
                   help="Nominal center frequency of the TXM.")
    p.add_argument("--voltage", type=float,
                   default=VerificationTank.DEFAULT_VOLTAGE_V,
                   help="HV rail (V) for peak scans + freq sweep.")
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
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Parent output directory. Default: ./test_reports/<TXM-SN>/.")
    p.add_argument("--no-prompt", action="store_true",
                   help="Skip the interactive prefs prompt; use cached values as-is.")
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
    p.add_argument("--log-file", type=Path, default=None,
                   help="Optional file to tee logs into.")
    verbosity = p.add_mutually_exclusive_group()
    verbosity.add_argument("--verbose", "-v", action="store_true")
    verbosity.add_argument("--quiet", "-q", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    _configure_root_logger(args.log_file)
    if args.verbose:
        set_log_level("DEBUG", sdk_level="DEBUG")
    elif args.quiet:
        set_log_level("WARNING")

    # --- Prefs ---
    prefs = OperatorPrefs.load(args.prefs)
    # Auto-populate the two fields we don't prompt for:
    # (1) test_app_version = version of this library, and
    # (2) hydrophone_sn    = read from the loaded Hydrophone calibration.
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
            return 130
        prefs.save(args.prefs)
    logger.info("Operator prefs: tester=%r txm_sn=%r hydrophone_sn=%r app_ver=%r",
                prefs.tester_name, prefs.txm_sn, prefs.hydrophone_sn,
                prefs.test_app_version)

    # --- Scan configuration (seed if missing) ---
    # Bundles acceptance criteria + scan geometry + scope capture.
    scan_config = ScanConfig.load_or_create(args.scan_config)
    criteria = scan_config.acceptance

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
                voltage=args.voltage,
            )
            ver.enable_hv_output(wait=True)
        else:
            ver.enable_hv_output()

        # --- Run the characterization ---
        chz = Characterization(
            ver,
            prefs=prefs,
            criteria=criteria,
            scan_config=scan_config,
            frequency_kHz=args.frequency_khz,
            voltage_V=args.voltage,
            plot=False,
        )
        t_start = time.perf_counter()
        try:
            report = chz.run(
                skip_2d=args.skip_2d,
                skip_frequency=args.skip_frequency,
                skip_voltage=args.skip_voltage,
            )
        except KeyboardInterrupt:
            print("\nInterrupted during characterization.", file=sys.stderr)
            return 130
        elapsed = time.perf_counter() - t_start
        logger.info("Characterization finished in %.1f s. Overall: %s",
                    elapsed, "PASS" if report.overall_pass else "FAIL")

        # --- Persist artifacts ---
        # write_report defaults to test_reports/<TXM-SN>/<ts>/ when
        # output_dir is None.
        run_dir = report_io.write_report(
            report,
            args.output_dir,
            write_device_config_json=not args.skip_frequency,
        )
        sn = report.rows[characterization.ROW["txm_sn"]].value or "unknown"
        sn_stem = report_io._sanitize_stem(sn)
        print(f"\nReport directory: {run_dir}")
        print(f"  XLSX : {run_dir / f'{sn_stem}_Report.xlsx'}")
        print(f"  PDF  : {run_dir / f'{sn_stem}_Report.pdf'}")
        if not args.skip_frequency:
            print(f"  JSON : {run_dir / f'{sn_stem}_device_config.json'}")
        print(f"  Verdict: {'PASS' if report.overall_pass else 'FAIL'}")

        # --- Optionally push config back onto the device ---
        if args.write_config and not args.dry_run:
            if args.skip_frequency:
                logger.warning("--skip-frequency set; no device_config.json to write.")
            else:
                config_path = run_dir / f"{sn_stem}_device_config.json"
                do_write = True
                if args.confirm_write_config:
                    do_write = _prompt_yes_no(
                        f"Write {config_path.name} to hardware?",
                        default=False,
                    )
                if do_write:
                    ok = _write_config_to_device(ver, config_path)
                    print(f"  Write to device: {'OK' if ok else 'FAILED'}")

    return 0 if report.overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
