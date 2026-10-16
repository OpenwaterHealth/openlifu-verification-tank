"""Calibration date strings from device metadata parse to dates."""
from __future__ import annotations

import datetime

import pytest

from openlifu_verification.characterization import _parse_cal_date


@pytest.mark.parametrize("raw", [
    "2024-02-28",   # ISO
    "28-Feb-2024",  # Onda hydrophone file
    "20240228",     # PicoScope driver
    "28Feb24",      # PicoScope driver
])
def test_parse_cal_date(raw):
    assert _parse_cal_date(raw) == datetime.date(2024, 2, 28)


@pytest.mark.parametrize("raw", ["", "   ", "not a date"])
def test_unparseable_cal_date_is_none(raw):
    assert _parse_cal_date(raw) is None
