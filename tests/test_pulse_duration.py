"""Per-frequency acceptance pulse duration."""
import pytest

from openlifu_verification.characterization import Characterization


@pytest.mark.parametrize("freq_kHz, expected_us", [(155.0, 80.0), (400.0, 60.0)])
def test_pulse_duration_by_frequency(freq_kHz, expected_us):
    assert Characterization.pulse_duration_us_for(freq_kHz) == expected_us


def test_pulse_duration_falls_back_to_default():
    assert (Characterization.pulse_duration_us_for(250.0)
            == Characterization.PULSE_DURATION_US)


def test_155khz_pulse_gives_pnp_window_more_ring_up_time():
    """The clean window opens when transmission stops, so the extra
    burst length at 155 kHz moves the window later by the same amount."""
    from openlifu_verification.pnp import clean_pnp_window_us, emitted_duration_us

    def window(duration_us):
        return clean_pnp_window_us(
            duration_us=emitted_duration_us(duration_us, 155.0),
            arrival_us=37.0, max_delay_us=4.0)

    lo60, hi60 = window(60.0)
    lo, hi = window(Characterization.pulse_duration_us_for(155.0))
    assert lo - lo60 >= 15.0
    assert hi - lo == pytest.approx(hi60 - lo60)
