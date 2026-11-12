"""Runs with any override are marked NON-STANDARD and cannot write to
the device; standard runs are unmarked."""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_report  # noqa: E402

from openlifu_verification import ScanConfig  # noqa: E402
from openlifu_verification.report_io import NONSTANDARD_BANNER  # noqa: E402


def _args(tmp_path, *extra):
    return run_report.build_parser().parse_args([
        "--dry-run", "--no-prompt",
        "--prefs", str(tmp_path / "prefs.json"),
        "--reports-root", str(tmp_path / "reports"),
        "--calibration-path", str(tmp_path / "hydrophone.json"),
        *extra,
    ])


def test_standard_args_have_no_reasons(tmp_path):
    assert run_report.nonstandard_reasons(_args(tmp_path), ScanConfig()) == []


@pytest.mark.parametrize("extra, reason", [
    (["--voltage", "25"], "--voltage 25"),
    (["--skip-2d"], "--skip-2d"),
    (["--skip-frequency"], "--skip-frequency"),
    (["--skip-voltage"], "--skip-voltage"),
])
def test_override_options_are_reasons(tmp_path, extra, reason):
    assert run_report.nonstandard_reasons(
        _args(tmp_path, *extra), ScanConfig()) == [reason]


def _run(args):
    seen = {}

    def on_complete(result, write_fn):
        seen["write_fn"] = write_fn

    result = run_report.run(args, on_complete=on_complete)
    rows = []
    if result.files.get("xlsx"):
        from openpyxl import load_workbook
        rows = list(load_workbook(result.files["xlsx"])["Summary"]
                    .iter_rows(values_only=True))
    return result, seen["write_fn"], rows


def test_standard_run_is_unmarked_and_can_write(tmp_path):
    result, write_fn, rows = _run(_args(tmp_path))
    assert result.passed
    assert result.nonstandard_reasons == []
    assert write_fn is not None
    assert ("Configuration", "Standard") in rows
    snapshot = next(result.run_dir.glob("*_scan_config.json"))
    assert json.loads(snapshot.read_text(encoding="utf-8")) == json.loads(
        json.dumps(asdict(ScanConfig())))


def test_nonstandard_run_is_marked_and_cannot_write(tmp_path):
    override = tmp_path / "override.json"
    override.write_text(json.dumps({"scope": {"n_averages": 1}}),
                        encoding="utf-8")

    result, write_fn, rows = _run(
        _args(tmp_path, "--scan-config", str(override)))

    # A passing run, so the only thing blocking the write is the override.
    assert result.passed
    assert write_fn is None
    assert result.nonstandard_reasons == [
        f"--scan-config {override}",
        "scope.n_averages = 1 (standard 3)",
    ]
    snapshot = next(result.run_dir.glob("*_scan_config.json"))
    assert json.loads(snapshot.read_text(encoding="utf-8"))["scope"]["n_averages"] == 1
    assert ("Configuration", NONSTANDARD_BANNER) in rows
    assert ("Non-standard setting", "scope.n_averages = 1 (standard 3)") in rows


def _failing_depth_tank(monkeypatch):
    real = run_report.DryRunTank

    def tank(**kw):
        t = real(**kw)

        def boom(**_k):
            raise RuntimeError("no arrival found")
        t.calibrate_hydrophone_depth = boom
        return t
    monkeypatch.setattr(run_report, "DryRunTank", tank)


def test_depth_failure_continue_writes_failing_report(tmp_path, monkeypatch):
    _failing_depth_tank(monkeypatch)
    asked = []
    result = run_report.run(
        _args(tmp_path),
        on_setup_failure=lambda failure: asked.append(failure.step) or True)
    assert asked == ["depth"]
    assert result.run_dir is not None and not result.passed
    assert result.files.get("xlsx") and result.files.get("pdf")
    depth = next(r for r in result.graded_rows
                 if r.label == "Plane-Wave Hydrophone Depth")
    assert depth.status == "FAIL"


def test_depth_failure_stop_is_acknowledged_abort(tmp_path, monkeypatch):
    _failing_depth_tank(monkeypatch)
    result = run_report.run(_args(tmp_path),
                            on_setup_failure=lambda _failure: False)
    assert result.aborted and result.abort_acknowledged
    assert result.exit_code == 130 and result.run_dir is None


def test_depth_failure_without_prompt_still_halts(tmp_path, monkeypatch):
    _failing_depth_tank(monkeypatch)
    result = run_report.run(_args(tmp_path))  # --no-prompt: no handler
    assert result.halted_reason.startswith(
        "Plane-wave depth calibration failed")


def test_depth_failure_cli_prompt(tmp_path, monkeypatch):
    _failing_depth_tank(monkeypatch)
    args = _args(tmp_path)
    args.no_prompt = False
    args.no_start_prompt = False  # --dry-run skips the start prompt anyway
    answers = iter(["n"])
    monkeypatch.setattr("builtins.input", lambda *_a: next(answers))
    result = run_report.run(args, prefs=run_report.OperatorPrefs(
        tester_name="t", test_app_version="t", hydrophone_sn="t",
        txm_sn="WLM44Q00001", console_sn="t"))
    assert result.aborted and result.abort_acknowledged


def test_peak_failure_continue_writes_failing_report(tmp_path, monkeypatch):
    real = run_report.DryRunTank

    def tank(**kw):
        t = real(**kw)

        def boom(**_k):
            raise RuntimeError("no signal above noise")
        t.find_peak = boom
        return t
    monkeypatch.setattr(run_report, "DryRunTank", tank)
    result = run_report.run(_args(tmp_path),
                            on_setup_failure=lambda failure: True)
    assert result.run_dir is not None and not result.passed
    assert result.files.get("xlsx") and result.files.get("pdf")
    peak = [r for r in result.graded_rows
            if r.label in ("Hydrophone X Position", "Hydrophone Y Position")]
    assert [r.status for r in peak] == ["FAIL", "FAIL"]


def test_outdated_firmware_result_is_marked(tmp_path, monkeypatch):
    real = run_report.DryRunTank

    def old_firmware(**kw):
        return real(txm_fw_version="2.0.0", **kw)
    monkeypatch.setattr(run_report, "DryRunTank", old_firmware)
    result = run_report.run(_args(tmp_path))
    assert result.halted_kind == "firmware"
    assert result.exit_code == 1 and result.run_dir is None
    assert "TXM firmware: version 2.0.0" in result.halted_reason


def test_focused_fallback_report_marks_d3_warn(tmp_path, monkeypatch):
    from openlifu_verification import DepthNotFound
    from openpyxl import load_workbook
    real = run_report.DryRunTank

    def tank(**kw):
        t = real(**kw)
        calibrate = t.calibrate_hydrophone_depth

        def weak_plane_wave(**k):
            if k.get("z_focus_mm") is None:
                raise DepthNotFound("No hydrophone signal above the noise")
            return calibrate(**k)
        t.calibrate_hydrophone_depth = weak_plane_wave
        return t
    monkeypatch.setattr(run_report, "DryRunTank", tank)
    result = run_report.run(_args(tmp_path))
    assert result.passed and result.run_dir is not None
    d3 = next(r for r in result.graded_rows if r.id == "D.3")
    assert d3.status == "WARN"
    ws = load_workbook(result.files["xlsx"])["Grading"]
    cell = next(row[4] for row in ws.iter_rows(min_row=2) if row[0].value == "D.3")
    assert cell.value == "WARN"
    assert cell.fill.start_color.rgb.endswith("FFEB9C")
