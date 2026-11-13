"""Re-derive the verification-tank focal-gain LUT with k-Wave via OpenLIFU-Python.

The device_config sensitivity is ``PNP / (focal_gain * V)``, where
``focal_gain`` is the steady-state focal pressure amplitude produced by
an 8x8 TXM whose every element has a surface-pressure sensitivity of
1 Pa/V. The gain depends on the measurement geometry, so it must match
how the mini tank measures the frequency sweep: hydrophone fixed at the
plane-wave depth (~50 mm), focus commanded *at* the hydrophone, burst
long enough to reach steady state, PNP taken as the median negative
peak inside the steady-state window.

For each frequency the script runs two linear k-Wave simulations on the
same grid, recording pressure on the beam axis:

* ``direct``   - every element driven with its focusing delay.
* ``neighbor`` - every element driven with the *sum* of its crosstalk
  neighbours' drive signals (elements within ``--crosstalk-dist``), the
  same nearest-neighbour model as
  ``openlifu.sim.kwave_if.run_simulation``.

k-Wave is linear in the source, so ``p(t; xt) = p_direct + xt * p_neighbor``
exactly and one pair of runs gives the gain at every crosstalk fraction.

Requirements (NOT dependencies of this package): ``openlifu``
(OpenLIFU-Python) and ``k-wave-python`` with its CUDA/OMP binaries.

Example::

    python scripts/derive_focal_gain_lut.py --depth-mm 50 --out focal_gain_lut_50mm.json
"""
from __future__ import annotations

import argparse
import json
import time
from copy import deepcopy

import numpy as np

PITCH_MM = 5.0
KERF_MM = 0.3
C0 = 1500.0  # medium and focusing speed of sound (m/s) == ScanConfig.sos_water_m_per_s
RHO0 = 1000.0

DEFAULT_FREQS_KHZ = ([130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180]
                     + [375, 380, 385, 390, 395, 400, 405, 410, 415, 420, 425])
DEFAULT_CROSSTALK = [0.0, 0.05, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]


def make_array():
    from openlifu.xdc import Transducer
    return Transducer.gen_matrix_array(nx=8, ny=8, pitch=PITCH_MM, kerf=KERF_MM, units="mm",
                                       id="txm", name="TXM")


def neighbor_matrix(arr, crosstalk_dist_m: float) -> np.ndarray:
    pos = np.asarray(arr.get_positions(units="m"))
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    a = (d <= crosstalk_dist_m).astype(float)
    np.fill_diagonal(a, 0.0)
    return a  # a[dst, src]


def focus_delays(arr, focus_mm) -> np.ndarray:
    pos = np.asarray(arr.get_positions(units="mm"))
    tof = np.linalg.norm(pos - np.asarray(focus_mm)[None, :], axis=1) * 1e-3 / C0
    return tof.max() - tof


def make_params(dx_mm: float, xy_half_mm: float, z_range_mm):
    import xarray as xa
    nxy = int(round(2 * xy_half_mm / dx_mm)) + 1
    x = np.linspace(-xy_half_mm, xy_half_mm, nxy)
    nz = int(round((z_range_mm[1] - z_range_mm[0]) / dx_mm)) + 1
    z = z_range_mm[0] + np.arange(nz) * dx_mm
    coords = {"x": xa.DataArray(x, dims="x", attrs={"units": "mm"}),
              "y": xa.DataArray(x.copy(), dims="y", attrs={"units": "mm"}),
              "z": xa.DataArray(z, dims="z", attrs={"units": "mm"})}
    shape = (len(x), len(x), len(z))
    return xa.Dataset(
        {"sound_speed": (("x", "y", "z"), np.full(shape, C0), {"units": "m/s", "ref_value": C0}),
         "density": (("x", "y", "z"), np.full(shape, RHO0), {"units": "kg/m^3", "ref_value": RHO0}),
         "attenuation": (("x", "y", "z"), np.zeros(shape),
                         {"units": "dB/cm/MHz", "ref_value": 0.0})},
        coords=coords)


def simulate_axis(freq_hz: float, depth_mm: float, *, crosstalk_dist_m: float, dx_mm: float = 0.5,
                  burst_s: float = 60e-6, gpu: bool = True, cfl: float = 0.3) -> dict:
    """Two linear k-Wave runs; returns axial time traces for direct + neighbour sources."""
    from kwave.ksensor import kSensor
    from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D
    from kwave.options.simulation_execution_options import SimulationExecutionOptions
    from kwave.options.simulation_options import SimulationOptions
    from openlifu.sim.kwave_if import get_karray, get_kgrid, get_medium, get_source

    arr = make_array()
    z_range = (-3.0, depth_mm + 12.0)
    params = make_params(dx_mm, 24.0, z_range)
    cycles = int(np.ceil(burst_s * freq_hz))
    dt = cfl * dx_mm * 1e-3 / C0
    t_end = cycles / freq_hz + (z_range[1] + 10) * 1e-3 / C0 + 10e-6
    kgrid = get_kgrid(params.coords, dt=dt, t_end=t_end)
    delays = focus_delays(arr, (0.0, 0.0, depth_mm))
    src_direct = arr.calc_output(amplitude=1.0, cycles=cycles, frequency=freq_hz, dt=kgrid.dt,
                                 delays=delays)[:, :kgrid.Nt]
    src_neigh = neighbor_matrix(arr, crosstalk_dist_m) @ src_direct

    array_offset = [-float(c.mean()) * 1e-3 for c in params.coords.values()]
    medium = get_medium(params, ref_values_only=True)
    karray = get_karray(arr, translation=array_offset)
    nx, ny, nz = params.sizes["x"], params.sizes["y"], params.sizes["z"]
    mask = np.zeros((nx, ny, nz))
    mask[nx // 2, ny // 2, :] = 1
    sensor = kSensor(mask, record=["p"])

    traces = {}
    for name, sig in (("direct", src_direct), ("neighbor", src_neigh)):
        source = get_source(kgrid, karray, sig)
        out = kspaceFirstOrder3D(
            kgrid=kgrid, source=source, sensor=deepcopy(sensor), medium=medium,
            simulation_options=SimulationOptions(pml_auto=True, pml_inside=False,
                                                 save_to_disk=True, data_cast="single"),
            execution_options=SimulationExecutionOptions(is_gpu_simulation=gpu,
                                                         show_sim_log=False))
        p = np.asarray(out["p"])
        traces[name] = p.T if p.shape[0] == kgrid.Nt else p  # -> [z, t]
    return {"t": np.arange(kgrid.Nt) * kgrid.dt, "z_mm": params.z.values.copy(),
            "freq_hz": freq_hz, "cycles": cycles, "delays": delays, "arr": arr, **traces}


def steady_state_window(sim: dict, point_mm) -> tuple[float, float]:
    """After the last direct/crosstalk arrival, before the first burst end."""
    pos = np.asarray(sim["arr"].get_positions(units="mm"))
    tof = np.linalg.norm(pos - np.asarray(point_mm)[None, :], axis=1) * 1e-3 / C0
    arrive = sim["delays"] + tof
    period = 1.0 / sim["freq_hz"]
    burst = sim["cycles"] * period
    return arrive.max() + 3 * period + 4e-6, arrive.min() + burst - 2 * period - 4e-6


def median_pnp(t, p, freq_hz, t_start, t_stop) -> float:
    """Median |negative peak| per cycle in the window (the tank's PNP estimator)."""
    sel = (t >= t_start) & (t <= t_stop)
    tt, pp = t[sel], p[sel]
    period = 1.0 / freq_hz
    n = int(np.floor((tt[-1] - tt[0]) / period))
    mins = [-pp[(tt >= tt[0] + k * period) & (tt < tt[0] + (k + 1) * period)].min()
            for k in range(n)]
    return float(np.median(mins))


def gains_from_sim(sim: dict, depth_mm: float, crosstalk) -> list[float]:
    iz = int(np.argmin(np.abs(sim["z_mm"] - depth_mm)))
    win = steady_state_window(sim, (0.0, 0.0, sim["z_mm"][iz]))
    return [median_pnp(sim["t"], sim["direct"][iz] + xt * sim["neighbor"][iz],
                       sim["freq_hz"], *win) for xt in crosstalk]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--depth-mm", type=float, default=50.0,
                    help="Hydrophone depth = commanded focus (mm). Default 50.")
    ap.add_argument("--freqs-khz", type=float, nargs="+", default=DEFAULT_FREQS_KHZ)
    ap.add_argument("--crosstalk", type=float, nargs="+", default=DEFAULT_CROSSTALK)
    ap.add_argument("--crosstalk-dist-mm", type=float, default=5.05)
    ap.add_argument("--dx-mm", type=float, default=0.5)
    ap.add_argument("--cpu", action="store_true", help="Use the OMP binary instead of CUDA.")
    ap.add_argument("--out", default="focal_gain_lut.json")
    args = ap.parse_args(argv)

    rows = []
    for f_khz in args.freqs_khz:
        t0 = time.time()
        sim = simulate_axis(f_khz * 1e3, args.depth_mm, crosstalk_dist_m=args.crosstalk_dist_mm * 1e-3,
                            dx_mm=args.dx_mm, gpu=not args.cpu)
        rows.append(gains_from_sim(sim, args.depth_mm, args.crosstalk))
        print(f"{f_khz:g} kHz: " + " ".join(f"{g:.4f}" for g in rows[-1])
              + f"  ({time.time() - t0:.0f} s)", flush=True)

    lut = {"depth_mm": args.depth_mm, "crosstalk_dist_mm": args.crosstalk_dist_mm,
           "dx_mm": args.dx_mm, "c_m_per_s": C0,
           "f0_hz": [f * 1e3 for f in args.freqs_khz], "crosstalk": list(args.crosstalk),
           "focal_gain": rows}
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(lut, fh, indent=1)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
