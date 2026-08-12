"""Synthetic stand-in for :class:`VerificationTank`.

Provides just enough of the interface (``measure_pressure``,
``scan_lateral``, ``scan_2d``, ``scan_frequency``, ``scan_voltage``,
``find_peak``, ``hydrophone_position``, ``frequency``, ``hv_voltage``,
``system_transmit_delay_us``, ``configure_lifu``, ``enable_hv_output``,
``set_voltage``, plus context-manager semantics) so
:class:`Characterization` can be exercised end-to-end without any
hardware.

Every measurement is a rotated-anisotropic 2-D Gaussian in ``(x, y)``,
scaled by voltage and modulated by a low-Q frequency-response curve
centered on ``nominal_frequency_kHz``. Traces are single-cycle
raised-cosine bursts arriving at ``z / 1.5`` µs in the
emission-relative time frame that :class:`VerificationTank` also
exposes.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from .scan_results import ScanResult

logger = logging.getLogger(__name__)


class DryRunTank:
    """Zero-hardware stand-in for :class:`VerificationTank`."""

    def __init__(
        self,
        *,
        frequency: float = 400.0,                        # kHz
        hydrophone_position=(0.6, -0.4, 50.0),
        peak_xy=(0.7, -0.3),                             # ground-truth peak
        peak_sigma=(0.9, 0.6),                           # mm
        peak_theta=np.pi / 6,                             # rad, rotation of Gaussian
        peak_amp_Pa=1.3e6,                                # peak PNP in Pa @ nominal voltage
        nominal_voltage_V=20.0,
        freq_bandwidth_kHz=40.0,                          # -3dB half-width of response
        noise_Pa=8e3,
        system_transmit_delay_us: float = 114.0,
        rng_seed: Optional[int] = 0,
        use_calibration: bool = True,
    ):
        self.frequency = float(frequency)
        self.hydrophone_position = np.array(hydrophone_position, dtype=float).reshape(3)
        self.system_transmit_delay_us = float(system_transmit_delay_us)
        self.hv_voltage: Optional[float] = None
        self.hydrophone_channel = "A"
        # Placeholder for the SDK-shaped bits that DeviceInfo pokes at.
        self.lifu = None
        # Ground truth used for signal synthesis.
        self._peak_xy = tuple(peak_xy)
        self._peak_sigma = tuple(peak_sigma)
        self._peak_theta = float(peak_theta)
        self._peak_amp_Pa = float(peak_amp_Pa)
        self._nominal_voltage = float(nominal_voltage_V)
        self._freq_bw_kHz = float(freq_bandwidth_kHz)
        self._noise_Pa = float(noise_Pa)
        self._rng = np.random.default_rng(rng_seed)
        # No-op hydrophone attribute so downstream code sees "attached".
        self.hydrophone = _DryHydrophone()
        self.calibration_path = None
        # Kept for API parity with :class:`VerificationTank`; the
        # dry-run synth always emits Pa, so toggling this flag has
        # no effect on the returned traces. Consumers that inspect
        # ``ver.use_calibration`` still see the intended value.
        self.use_calibration = bool(use_calibration)

    # --- context manager ------------------------------------------------
    def __enter__(self):
        logger.info("DryRunTank: context entered (no hardware)")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.info("DryRunTank: context exited")
        return False

    # --- SDK / configure API stubs -------------------------------------
    def configure_lifu(self, **_):  # noqa: D401 - stub
        return None

    def enable_hv_output(self, *_, **kwargs):
        self.hv_voltage = self._nominal_voltage
        return None

    def disable_hv_output(self, *_, **__):
        return None

    def set_voltage(self, voltage_V, *, wait=True):
        self.hv_voltage = float(voltage_V)
        return None

    def set_focus(self, x, y, z):  # noqa: D401 - stub
        return None

    def set_pulse(self, *, frequency_kHz=None, **_):
        if frequency_kHz is not None:
            self.frequency = float(frequency_kHz)
        return None

    def add_log_file(self, *_):
        return None

    # --- signal synthesis helpers -------------------------------------
    def _amplitude_Pa(self, x, y, freq_kHz=None, voltage_V=None) -> float:
        """Return the peak amplitude of a burst at (x, y)."""
        if voltage_V is None:
            voltage_V = self.hv_voltage or self._nominal_voltage
        if freq_kHz is None:
            freq_kHz = self.frequency
        dx = x - self._peak_xy[0]
        dy = y - self._peak_xy[1]
        ct, st = np.cos(self._peak_theta), np.sin(self._peak_theta)
        u = dx * ct + dy * st
        v = -dx * st + dy * ct
        sx, sy = self._peak_sigma
        gauss = np.exp(-(u ** 2) / (2 * sx ** 2) - (v ** 2) / (2 * sy ** 2))
        # Frequency response: Gaussian around nominal.
        df = freq_kHz - self.frequency
        # Actually we want the nominal to be the *device* nominal not the
        # currently-set frequency. Approximate: pull from ``self._peak_amp_Pa``
        # curve peaking at 400 or 155 depending on init frequency.
        # Since we don't retain nominal separately, just use current
        # nominal as a moving target; the freq sweep changes it.
        # To make the freq sweep interesting, use init-time nominal:
        freq_gain = 1.0 / (1.0 + (df / self._freq_bw_kHz) ** 2)
        # Voltage: linear
        v_scale = voltage_V / self._nominal_voltage
        return self._peak_amp_Pa * gauss * freq_gain * v_scale

    def _synth_trace(self, amp_Pa, z_mm, t_us):
        """Return a synthetic trace: raised-cosine burst at expected arrival.

        ``t_us`` is the emission-referenced time axis (\u00b5s), so the
        arrival time is simply ``z_mm / SOS`` (SOS = 1.5 mm/\u00b5s in
        water) with no transmit-delay offset applied.
        """
        arrival_us = float(z_mm) / 1.5
        # 20-cycle burst @ current frequency.
        cycles = 20.0
        f_Hz = self.frequency * 1e3
        burst_dur_us = 1e6 * cycles / f_Hz
        env = np.zeros_like(t_us)
        mask = (t_us >= arrival_us) & (t_us <= arrival_us + burst_dur_us)
        # raised-cosine envelope
        tau = (t_us[mask] - arrival_us) / burst_dur_us
        env[mask] = 0.5 * (1 - np.cos(2 * np.pi * tau))
        carrier = np.sin(2 * np.pi * f_Hz * (t_us - arrival_us) * 1e-6)
        trace = amp_Pa * env * carrier
        trace += self._rng.normal(0.0, self._noise_Pa, size=trace.shape)
        return trace

    def _make_time_axis(self, time_start_s, time_stop_s, sampling_interval_ns):
        """Emission-referenced time axis in \u00b5s (matches VerificationTank)."""
        n = int(round((time_stop_s - time_start_s) / (sampling_interval_ns * 1e-9))) + 1
        return np.linspace(time_start_s * 1e6, time_stop_s * 1e6, n)

    # --- measurement APIs ---------------------------------------------
    def measure_pressure(self, x, y, z, *,
                         time_start_s=-14e-6, time_stop_s=86e-6,
                         sampling_interval_ns=100, timeout_s=2.0):
        t_us = self._make_time_axis(time_start_s, time_stop_s, sampling_interval_ns)
        amp = self._amplitude_Pa(x, y)
        trace = self._synth_trace(amp, z, t_us)
        return {
            "t": t_us,
            "trace": trace,
            "rms": float(np.sqrt(np.mean(trace ** 2))),
            "vpp": float(np.max(trace) - np.min(trace)),
            "units": "Pa",
        }

    def find_peak(self, *, x0=None, y0=None, z=None, plot=False, store=True,
                  save=False, keep_plot_open=False, **_):
        if z is None:
            z = float(self.hydrophone_position[2])
        # Cheat: return the ground truth + small noise.
        x = self._peak_xy[0] + self._rng.normal(0, 0.02)
        y = self._peak_xy[1] + self._rng.normal(0, 0.02)
        if store:
            self.hydrophone_position = np.array([x, y, z], dtype=float)
        return float(x), float(y)

    # --- scan APIs -----------------------------------------------------
    def _scan_grid(self, xs, ys, z, time_start_s, time_stop_s, sampling_interval_ns):
        t_us = self._make_time_axis(time_start_s, time_stop_s, sampling_interval_ns)
        traces = np.empty((len(ys), len(xs), t_us.size), dtype=float)
        for iy, y in enumerate(ys):
            for ix, x in enumerate(xs):
                amp = self._amplitude_Pa(x, y)
                traces[iy, ix] = self._synth_trace(amp, z, t_us)
        return t_us, traces

    def scan_1d(self, *, dim, scan_range=(-10.0, 10.0), num=41,
                absolute=False, x=0.0, y=0.0, z=None,
                time_start_s=-14e-6, time_stop_s=86e-6,
                sampling_interval_ns=100, chunk_size=0, timeout_s=None,
                n_averages=1, align=True, align_max_shift_samples=None,
                progress="bar") -> ScanResult:
        """1-D synthetic scan along ``dim`` (``"x"``, ``"y"``, or ``"z"``).

        Mirrors :meth:`VerificationTank.scan_1d`. The other two
        coordinates are held fixed at the calibrated hydrophone
        position (or at ``(x, y, z)`` when ``absolute=True``).
        """
        dim = str(dim).lower()
        if dim not in ("x", "y", "z"):
            raise ValueError(f"dim must be 'x', 'y', or 'z' (got {dim!r})")
        if absolute:
            x_origin = y_origin = z_origin = 0.0
        else:
            x_origin = float(self.hydrophone_position[0])
            y_origin = float(self.hydrophone_position[1])
            z_origin = float(self.hydrophone_position[2])
        z_fixed = z_origin if z is None else float(z)

        coord_axis = np.linspace(scan_range[0], scan_range[1], num)
        t_us = self._make_time_axis(time_start_s, time_stop_s, sampling_interval_ns)
        traces = np.empty((num, t_us.size), dtype=float)
        for i, v in enumerate(coord_axis):
            if dim == "x":
                xi, yi, zi = float(v) + x_origin, float(y) + y_origin, z_fixed
            elif dim == "y":
                xi, yi, zi = float(x) + x_origin, float(v) + y_origin, z_fixed
            else:  # "z"
                xi, yi, zi = float(x) + x_origin, float(y) + y_origin, float(v) + z_origin
            amp = self._amplitude_Pa(xi, yi)
            traces[i] = self._synth_trace(amp, zi, t_us)

        coord_key = {"x": "xfoci", "y": "yfoci", "z": "zfoci"}[dim]
        return ScanResult(
            scan_type="1d",
            t=t_us,
            traces=traces,
            coords={coord_key: coord_axis},
            hydrophone_channel=self.hydrophone_channel,
            chunk_size=chunk_size or num,
            units="Pa",
            metadata={
                "dim": dim,
                "z_mm": float(z_fixed),
                "frequency_kHz": float(self.frequency),
                "voltage_V": float(self.hv_voltage) if self.hv_voltage else float("nan"),
                "hydrophone_position_mm": self.hydrophone_position.copy(),
                "absolute": bool(absolute),
            },
        )

    def scan_lateral(self, *, x_range=(-10.0, 10.0), num_x=41,
                     y_range=None, num_y=1, y=0.0, z=None, absolute=False,
                     time_start_s=-14e-6, time_stop_s=86e-6,
                     sampling_interval_ns=100, chunk_size=0, timeout_s=None,
                     progress="bar") -> ScanResult:
        if z is None:
            z = float(self.hydrophone_position[2])
        x_off, y_off = (0.0, 0.0) if absolute else (
            float(self.hydrophone_position[0]), float(self.hydrophone_position[1])
        )
        xs = np.linspace(x_range[0], x_range[1], num_x)
        if num_y > 1:
            if y_range is None:
                raise ValueError("y_range required when num_y > 1")
            ys = np.linspace(y_range[0], y_range[1], num_y)
        else:
            ys = np.array([float(y)])
        xs_abs = xs + x_off
        ys_abs = ys + y_off
        t_us, grid = self._scan_grid(xs_abs, ys_abs, z, time_start_s,
                                     time_stop_s, sampling_interval_ns)
        if num_y > 1:
            coords = {"yfoci": ys, "xfoci": xs}
            traces = grid
            scan_type = "2d"
        else:
            coords = {"xfoci": xs}
            traces = grid[0]
            scan_type = "lateral"
        return ScanResult(
            scan_type=scan_type,
            t=t_us,
            traces=traces,
            coords=coords,
            hydrophone_channel=self.hydrophone_channel,
            chunk_size=chunk_size or (num_x * num_y),
            units="Pa",
            metadata={
                "z_mm": float(z),
                "frequency_kHz": float(self.frequency),
                "voltage_V": float(self.hv_voltage) if self.hv_voltage else float("nan"),
                "hydrophone_position_mm": self.hydrophone_position.copy(),
                "absolute": bool(absolute),
            },
        )

    def scan_2d(self, *, x_range=(-4.0, 4.0), num_x=9,
                y_range=(-4.0, 4.0), num_y=9, z=None, absolute=False,
                time_start_s=-14e-6, time_stop_s=86e-6,
                sampling_interval_ns=100, chunk_size=0, timeout_s=None,
                progress="bar") -> ScanResult:
        return self.scan_lateral(
            x_range=x_range, num_x=num_x,
            y_range=y_range, num_y=num_y,
            z=z, absolute=absolute,
            time_start_s=time_start_s, time_stop_s=time_stop_s,
            sampling_interval_ns=sampling_interval_ns,
            chunk_size=chunk_size, timeout_s=timeout_s, progress=progress,
        )

    def scan_frequency(self, *, frequencies_kHz, duration_usec,
                       time_start_s=-14e-6, time_stop_s=86e-6,
                       sampling_interval_ns=100, chunk_size=0,
                       timeout_s=None, progress="bar") -> ScanResult:
        freqs = np.asarray(list(frequencies_kHz), dtype=float)
        t_us = self._make_time_axis(time_start_s, time_stop_s, sampling_interval_ns)
        traces = np.empty((freqs.size, t_us.size), dtype=float)
        x, y, z = self.hydrophone_position
        original_freq = self.frequency
        try:
            for i, f in enumerate(freqs):
                self.frequency = float(f)
                amp = self._amplitude_Pa(x, y, freq_kHz=f)
                traces[i] = self._synth_trace(amp, z, t_us)
        finally:
            self.frequency = original_freq
        return ScanResult(
            scan_type="frequency",
            t=t_us,
            traces=traces,
            coords={"freq_kHz": freqs},
            hydrophone_channel=self.hydrophone_channel,
            chunk_size=chunk_size or freqs.size,
            units="Pa",
            metadata={
                "voltage_V": float(self.hv_voltage) if self.hv_voltage else float("nan"),
                "duration_usec": float(duration_usec),
                "hydrophone_position_mm": self.hydrophone_position.copy(),
            },
        )

    def scan_voltage(self, *, voltages_V,
                     time_start_s=-14e-6, time_stop_s=86e-6,
                     sampling_interval_ns=100, chunk_size=0,
                     timeout_s=None, progress="bar") -> ScanResult:
        voltages = np.asarray(list(voltages_V), dtype=float)
        t_us = self._make_time_axis(time_start_s, time_stop_s, sampling_interval_ns)
        traces = np.empty((voltages.size, t_us.size), dtype=float)
        x, y, z = self.hydrophone_position
        original_v = self.hv_voltage
        try:
            for i, v in enumerate(voltages):
                self.hv_voltage = float(v)
                amp = self._amplitude_Pa(x, y)
                traces[i] = self._synth_trace(amp, z, t_us)
        finally:
            self.hv_voltage = original_v
        return ScanResult(
            scan_type="voltage",
            t=t_us,
            traces=traces,
            coords={"voltage_V": voltages},
            hydrophone_channel=self.hydrophone_channel,
            chunk_size=chunk_size or voltages.size,
            units="Pa",
            metadata={
                "frequency_kHz": float(self.frequency),
                "hydrophone_position_mm": self.hydrophone_position.copy(),
            },
        )


class _DryHydrophone:
    """Minimal stub so downstream code sees a hydrophone as 'attached'."""
    model = "DRY"
    serial_number = "DRY-0000"

    def mv_to_pa(self, trace_mv, frequency_hz):
        return trace_mv  # not used in dry-run

    def get_frequency_response(self, frequencies_hz):
        arr = np.asarray(frequencies_hz, dtype=float)
        return np.full_like(arr, 1.0)
