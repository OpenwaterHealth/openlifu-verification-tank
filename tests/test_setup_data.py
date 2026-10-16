"""Setup data registries: formats, lookup, add, initialization."""
from __future__ import annotations

import json

import pytest

from openlifu_verification import paths
from openlifu_verification.acceptance import SUPPORTED_FREQUENCIES_KHZ
from openlifu_verification.setup_data import (
    CONSOLES,
    HYDROPHONES,
    TESTERS,
    TXM_FREQUENCY_KHZ_BY_MODEL,
    TXM_SN,
    SetupData,
    txm_frequency_kHz,
)


@pytest.fixture
def setup(tmp_path) -> SetupData:
    s = SetupData(tmp_path / "setup")
    s.initialize()
    return s


@pytest.mark.parametrize("fmt, good, bad", [
    (TXM_SN, "WLM44Q00012", ["WLM44Q0001", "WLM44Q000123", "wlm44q00012",
                              "WLM34Q00012"]),
    (TXM_SN, "WLM24Q00012", ["WLM24W00012"]),
    (CONSOLES.id_format, "WL134Q00007", ["WL134Q0007", "XWL134Q00007",
                                        "WL134W00007"]),
    (HYDROPHONES.id_format, "HYDRO2246", ["HYDRO224", "HYDRO22460", "2246"]),
    (TESTERS.id_format, "Jane Doe 7", [""]),
])
def test_id_formats(fmt, good, bad):
    assert fmt.matches(good)
    assert not any(fmt.matches(b) for b in bad)


def test_txm_frequency_from_model_code():
    assert txm_frequency_kHz("WLM24Q00012") == 155.0
    assert txm_frequency_kHz("WLM44Q00012") == 400.0
    assert txm_frequency_kHz("WLM44Q0001") is None
    assert sorted(TXM_FREQUENCY_KHZ_BY_MODEL.values()) \
        == sorted(SUPPORTED_FREQUENCIES_KHZ)


def test_initialize_copies_template_without_overwriting(tmp_path):
    s = SetupData(tmp_path / "setup")
    assert not s.is_initialized()
    s.initialize()
    assert s.is_initialized()
    assert s.lookup(TESTERS, "anyone") == (False, None)
    for key in ("HYDRO2246", "HYDRO2581", "HYDRO2585"):
        assert s.calibration_path(key).is_file()

    s.add(TESTERS, "T1", "Jane Doe")
    s.initialize()
    assert s.lookup(TESTERS, "T1") == (True, {"Name": "Jane Doe"})


def test_bundled_template_matches_calibrations():
    s = SetupData(paths.BUNDLED_SETUP_DATA_DIR)
    assert s.is_initialized()
    for key in s.load(HYDROPHONES):
        assert s.calibration_path(key).is_file(), key


def test_add_rejects_bad_format(setup):
    with pytest.raises(ValueError, match="WL134Q#####"):
        setup.add(CONSOLES, "WL134Q1", "abc")
    assert setup.load(CONSOLES) == {}


def test_add_console_and_lookup(setup):
    setup.add(CONSOLES, "WL134Q00007", "4ZGd1kX")
    assert setup.lookup(CONSOLES, "WL134Q00007") == (True, {"HWID": "4ZGd1kX"})
    on_disk = json.loads(setup.registry_path(CONSOLES).read_text())
    assert on_disk == {"WL134Q00007": {"HWID": "4ZGd1kX"}}


def test_malformed_registry_raises(setup):
    setup.registry_path(TESTERS).write_text('{"T1": "Jane"}')
    with pytest.raises(ValueError):
        setup.lookup(TESTERS, "T1")


def test_add_hydrophone_copies_calibration(setup, tmp_path):
    src = next(paths.REPO_CALIBRATIONS_DIR.glob("HNR0500-2246*.txt"))
    copy = tmp_path / src.name
    copy.write_bytes(src.read_bytes())
    setup.add_hydrophone("HYDRO9999", copy)
    assert setup.calibration_path("HYDRO9999") == setup.calibrations_dir / src.name

    other = tmp_path / "other" / src.name
    other.parent.mkdir()
    other.write_bytes(src.read_bytes() + b"\n")
    with pytest.raises(FileExistsError):
        setup.add_hydrophone("HYDRO9998", other)
    assert setup.lookup(HYDROPHONES, "HYDRO9998") == (False, None)


def test_add_hydrophone_rejects_non_calibration(setup, tmp_path):
    bogus = tmp_path / "notes.txt"
    bogus.write_text("not a calibration")
    with pytest.raises(ValueError, match="no FREQ_MHz"):
        setup.add_hydrophone("HYDRO9999", bogus)
    assert not (setup.calibrations_dir / "notes.txt").exists()
    assert setup.lookup(HYDROPHONES, "HYDRO9999") == (False, None)


def test_last_setup_round_trip(setup):
    assert setup.load_last_setup() == {}
    ids = {"tester": "T1", "console": "WL134Q00007", "hydrophone": "HYDRO2246"}
    setup.save_last_setup(ids)
    assert setup.load_last_setup() == ids
