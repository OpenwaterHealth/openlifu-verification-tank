"""Realized TX drive frequency and the hydrophone-cal frequency convention.

The TX pattern generator quantizes the drive period to whole clock
samples, so the emitted tone differs from the commanded frequency.
``ScanConfig.hydrophone_cal_at_realized_frequency`` (experimental, off
by default) makes the tank read the hydrophone calibration at the
emitted tone instead of the commanded one.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from openlifu_verification import Hydrophone, ScanConfig, paths
from openlifu_verification.drive_frequency import (
    estimate_tone_frequency, realized_frequency_hz)
from openlifu_verification.verificationtank import VerificationTank

# Commanded kHz -> emitted kHz with the default 10 MHz beamformer clock
# for the standard +/-25 kHz sweeps (period = int(10 MHz / f) samples).
REALIZED_KHZ = {
    130: 131.579, 135: 135.135, 140: 140.845, 145: 147.059, 150: 151.515,
    155: 156.250, 160: 161.290, 165: 166.667, 170: 172.414, 175: 175.439,
    180: 181.818,
    375: 384.615, 380: 384.615, 385: 400.0, 390: 400.0, 395: 400.0,
    400: 400.0, 405: 416.667, 410: 416.667, 415: 416.667, 420: 434.783,
    425: 434.783,
}


@pytest.mark.parametrize("cmd_kHz, real_kHz", sorted(REALIZED_KHZ.items()))
def test_realized_frequency_matches_sdk_pattern(cmd_kHz, real_kHz):
    assert realized_frequency_hz(cmd_kHz * 1e3) / 1e3 == pytest.approx(real_kHz, abs=1e-3)


def test_realized_frequency_is_vectorized_and_passes_through_nan():
    out = realized_frequency_hz(np.array([[390e3, np.nan]]))
    assert out.shape == (1, 2)
    assert out[0, 0] == pytest.approx(400e3)
    assert np.isnan(out[0, 1])


def test_realized_frequency_follows_bf_clk():
    # 20 MHz clock: 390 kHz -> int(51.28) = 51 samples -> 392.16 kHz.
    assert realized_frequency_hz(390e3, bf_clk_hz=20e6) == pytest.approx(20e6 / 51)


def _burst(f_hz, *, dt=96e-9, onset=40e-6, dur=60e-6, n=1354, noise=0.0, seed=0):
    t = -10e-6 + np.arange(n) * dt
    on = (t >= onset) & (t < onset + dur)
    y = np.where(on, np.sin(2 * np.pi * f_hz * (t - onset)), 0.0)
    y += 0.2 * np.where(on, np.sin(3 * 2 * np.pi * f_hz * (t - onset)), 0.0)
    y += 0.3  # scope DC offset
    if noise:
        y += np.random.default_rng(seed).normal(0, noise, n)
    return t, y


@pytest.mark.parametrize("f_hz", [156.25e3, 384.615e3, 400e3, 434.783e3])
def test_estimate_tone_frequency_recovers_synthetic_burst(f_hz):
    t, y = _burst(f_hz, noise=0.02)
    # Guess off by a few kHz (e.g. the commanded frequency).
    est = estimate_tone_frequency(t, y, f_hz * 0.98, burst_s=60e-6)
    assert est == pytest.approx(f_hz, rel=1e-3)


def test_estimate_tone_frequency_short_window_is_nan():
    t, y = _burst(400e3, n=200)
    assert np.isnan(estimate_tone_frequency(t, y, 400e3))


# ---------------------------------------------------------------------
# Config flag
# ---------------------------------------------------------------------
def test_flag_is_off_in_the_standard_config():
    assert ScanConfig().hydrophone_cal_at_realized_frequency is False


def test_flag_override_marks_run_non_standard(tmp_path):
    path = tmp_path / "o.json"
    path.write_text(json.dumps({"hydrophone_cal_at_realized_frequency": True}))
    cfg = ScanConfig.with_overrides(path)
    assert cfg.hydrophone_cal_at_realized_frequency is True
    assert cfg.differences_from_standard() == [
        "hydrophone_cal_at_realized_frequency = True (standard False)"]


def test_flag_override_must_be_boolean(tmp_path):
    path = tmp_path / "o.json"
    path.write_text(json.dumps({"hydrophone_cal_at_realized_frequency": 1}))
    with pytest.raises(ValueError, match="true or false"):
        ScanConfig.with_overrides(path)


# ---------------------------------------------------------------------
# Tank conversion
# ---------------------------------------------------------------------
@pytest.fixture()
def hydrophone():
    hits = sorted(paths.BUNDLED_CALIBRATIONS_DIR.glob("HNR0500-2580*.txt"))
    assert hits
    return Hydrophone(hits[0])


def _tank(hydrophone, realized):
    return VerificationTank(use_picoscope=False, hydrophone=hydrophone,
                            hydrophone_cal_at_realized_frequency=realized)


def test_default_tank_reads_cal_at_commanded_frequency(hydrophone):
    tank = _tank(hydrophone, False)
    assert tank.hydrophone_cal_frequency_hz(155.0) == pytest.approx(155e3)
    np.testing.assert_allclose(
        tank.hydrophone_cal_frequency_hz(np.array([390.0, 420.0])), [390e3, 420e3])


def test_realized_tank_reads_cal_at_emitted_tone(hydrophone):
    tank = _tank(hydrophone, True)
    assert tank.hydrophone_cal_frequency_hz(155.0) == pytest.approx(156.25e3)
    np.testing.assert_allclose(
        tank.hydrophone_cal_frequency_hz(np.array([390.0, 420.0])),
        [400e3, 434.783e3], rtol=1e-5)


class _FakeTX:
    def __init__(self):
        self.commanded = None


def _fake_sweep_tank(hydrophone, realized, monkeypatch):
    """Tank whose capture returns a clean burst at the *realized* tone,
    1 mV amplitude, so the Pa conversion can be checked directly."""
    tank = _tank(hydrophone, realized)
    tank.lifu_config = {"interval_msec": 20.0}
    tx = _FakeTX()
    monkeypatch.setattr(tank, "configure_lifu", lambda **kw: None)
    monkeypatch.setattr(
        tank, "set_pulse",
        lambda frequency_kHz, duration_usec: setattr(tx, "commanded", frequency_kHz))

    def capture(n_pulses, time_start_s, time_stop_s, sampling_interval_ns, timeout_s):
        f = realized_frequency_hz(tx.commanded * 1e3)
        t, y = _burst(f)
        return {"time": t * 1e6, "sampling_interval_ns": 96.0,
                "A": np.tile(y - 0.3, (n_pulses, 1))}

    monkeypatch.setattr(tank, "capture_pulse_train", capture)
    return tank


@pytest.mark.parametrize("realized", [False, True])
def test_scan_frequency_conversion_and_metadata(hydrophone, realized, monkeypatch):
    tank = _fake_sweep_tank(hydrophone, realized, monkeypatch)
    freqs = np.array([385.0, 400.0, 420.0])
    res = tank.scan_frequency(frequencies_kHz=freqs, duration_usec=60.0,
                              n_averages=2, progress=None)
    meta = res.metadata
    np.testing.assert_allclose(meta["realized_freq_kHz"], [400.0, 400.0, 434.783], rtol=1e-5)
    np.testing.assert_allclose(meta["measured_freq_kHz"], meta["realized_freq_kHz"], rtol=2e-3)
    cal_kHz = meta["realized_freq_kHz"] if realized else freqs
    np.testing.assert_allclose(meta["hydrophone_cal_freq_kHz"], cal_kHz, rtol=1e-6)
    assert meta["hydrophone_cal_at_realized_frequency"] is realized
    # 1 mV peak -> Pa = 1e-3 V * (Pa/V at the cal frequency).
    expected_pa = 1e-3 * hydrophone.get_frequency_response(np.asarray(cal_kHz) * 1e3)
    peak = np.max(np.asarray(res.traces), axis=1)
    np.testing.assert_allclose(peak, expected_pa, rtol=0.25)  # 3rd harmonic adds ~20 %
    ratio = peak / expected_pa
    np.testing.assert_allclose(ratio, ratio[0], rtol=1e-6)
    if realized:
        # 385 and 400 kHz commanded emit the same tone -> same pressure.
        assert peak[0] == pytest.approx(peak[1], rel=1e-9)


def test_characterization_pushes_flag_to_tank(hydrophone, tmp_path):
    from openlifu_verification import Characterization, OperatorPrefs
    path = tmp_path / "o.json"
    path.write_text(json.dumps({"hydrophone_cal_at_realized_frequency": True}))
    tank = _tank(hydrophone, False)
    Characterization(tank, prefs=OperatorPrefs(), frequency_kHz=155.0,
                     scan_config=ScanConfig.with_overrides(path))
    assert tank.hydrophone_cal_at_realized_frequency is True
    Characterization(tank, prefs=OperatorPrefs(), frequency_kHz=155.0)
    assert tank.hydrophone_cal_at_realized_frequency is False
