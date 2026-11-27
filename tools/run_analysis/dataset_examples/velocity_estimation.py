#!/usr/bin/env python3
"""Velocity estimation: wheel speed and the IMU against the speed the map pose shows.

    python3 tools/run_analysis/dataset_examples/velocity_estimation.py data/rb1/racerbot_runs/20261026-194514

Model.
  * Wheel speed: /odom's v = motor ERPM / speed_to_erpm_gain (the VESC, through
    vesc_to_odom). If the gain is right and the tyres do not slip, v_wheel = v_true.
  * Map speed (the reference): the forward component of the map pose's velocity,
        v_map = (dx cos(yaw) + dy sin(yaw)) / dt      over a centred 0.4 s span.
  * IMU: on level ground the forward accelerometer reads the change of speed,
        a_x = dv/dt                                    (car axes, /imu/data)
    so a_x is compared with the derivative of the wheel speed.
  * /odom/fused is imu_bridge's Kalman blend of both (clamped to +-0.3 m/s of
    the wheel speed, src/imu_bridge/README.md).

Only samples with LB held, the map speed at least 0.5 m/s, and no map-pose jump
within 0.5 s are compared.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from run_analysis.dataset_examples import common as c  # noqa: E402

TOPICS = ["/odom", "/odom/fused", *c.MAP_POSES, "/imu/data", "/joy"]
SMOOTH_S = 0.2        # s, accelerometer and wheel-speed smoothing before comparing


def speed_errors(v_est, v_ref, ok) -> dict:
    """Error stats of v_est - v_ref on the ok samples, and the distance ratio sum(v_est) / sum(v_ref)."""
    v_est, v_ref = np.asarray(v_est, dtype=float), np.asarray(v_ref, dtype=float)
    ok = np.asarray(ok, dtype=bool) & np.isfinite(v_est) & np.isfinite(v_ref)
    out = c.stats(v_est[ok] - v_ref[ok])
    total = float(np.sum(v_ref[ok]))
    out["distance_ratio"] = float(np.sum(v_est[ok]) / total) if total > 0 else float("nan")
    return out


def analyse(run: c.Run) -> dict:
    ref = run[next(t for t in c.MAP_POSES if run.has(t))]
    odom = run["/odom"]
    grid = c.time_grid(max(ref.t[0], odom.t[0]), min(ref.t[-1], odom.t[-1]))
    ref_topic, v_map, jumps = c.map_speed(run, grid)
    v_wheel = c.on_grid(grid, odom.t, odom.v)
    gyro = run["/imu/data"] if run.has("/imu/data") else None
    ok = c.inside(grid, c.lb_spans(run)) & (v_map >= c.MIN_MAP_SPEED)
    result = dict(reference=ref_topic, compared_s=float(ok.sum() / c.FS), map_pose_jumps=int(len(jumps)),
                  wheel=speed_errors(v_wheel, v_map, ok))
    curves = dict(t=grid - grid[0], v_map=v_map, v_wheel=v_wheel, ok=ok)
    if run.has("/odom/fused"):
        v_fused = c.on_grid(grid, run["/odom/fused"].t, run["/odom/fused"].v)
        result["fused"] = speed_errors(v_fused, v_map, ok)
        curves["v_fused"] = v_fused
    if gyro is not None:
        n = max(1, int(round(SMOOTH_S * c.FS)))
        a_x = c.moving_average(c.on_grid(grid, gyro.t, gyro.ax), n)
        dv_wheel = c.derivative(c.moving_average(v_wheel, n), c.FS)
        moving = c.inside(grid, c.lb_spans(run)) & (np.abs(v_wheel) >= c.MOVING_SPEED)
        a_y = c.moving_average(c.on_grid(grid, gyro.t, gyro.ay), n)
        lateral = v_wheel * c.on_grid(grid, gyro.t, gyro.wz)
        result["accelerometer_axes"] = c.accel_axes(a_x, a_y, dv_wheel, lateral, moving)
        result["accel_vs_wheel"] = c.line_fit(dv_wheel[moving], a_x[moving])
        result["accel_vs_wheel"]["accel_bias_mps2"] = float(np.nanmedian(a_x[moving] - dv_wheel[moving])) \
            if moving.any() else float("nan")
        curves.update(a_x=a_x, dv_wheel=dv_wheel, moving=moving)
    result["_curves"] = curves
    return result


def plots(run: c.Run, result: dict, folder: Path) -> list:
    plt = c.pyplot()
    if plt is None:
        return []
    k = result["_curves"]
    paths = []
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(k["t"], k["v_map"], "k-", lw=1.2, label=f"map pose ({result['reference']})")
    ax.plot(k["t"], k["v_wheel"], "-", color="tab:red", lw=0.9, label="wheels (/odom)")
    if "v_fused" in k:
        ax.plot(k["t"], k["v_fused"], "-", color="tab:blue", lw=0.9, label="Kalman (/odom/fused)")
    ax.fill_between(k["t"], 0, 1, where=k["ok"], transform=ax.get_xaxis_transform(), color="0.9", label="compared")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("forward speed (m/s)")
    ax.legend(fontsize=7)
    ax.set_title(f"{run.car} {run.run_id}: speed, three ways")
    paths.append(c.save(fig, folder / "velocity_speed.png"))
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ok = k["ok"]
    axes[0].plot(k["v_map"][ok], (k["v_wheel"] - k["v_map"])[ok], ".", ms=2, color="tab:red", label="wheels")
    if "v_fused" in k:
        axes[0].plot(k["v_map"][ok], (k["v_fused"] - k["v_map"])[ok], ".", ms=2, color="tab:blue", label="Kalman")
    axes[0].axhline(0, color="k", lw=0.6)
    axes[0].set_xlabel("map speed (m/s)")
    axes[0].set_ylabel("estimate - map speed (m/s)")
    axes[0].legend(fontsize=7)
    if "a_x" in k:
        m = k["moving"]
        axes[1].plot(k["dv_wheel"][m], k["a_x"][m], ".", ms=2, color="tab:purple")
        fit = result["accel_vs_wheel"]
        if np.isfinite(fit["slope"]):
            xs = np.array([np.nanmin(k["dv_wheel"][m]), np.nanmax(k["dv_wheel"][m])])
            axes[1].plot(xs, fit["slope"] * xs + fit["intercept"], "k-", lw=1,
                         label=f"fit: slope {fit['slope']:.2f}, r {fit['r']:.2f}")
            axes[1].plot(xs, xs, "k:", lw=0.8, label="a_x = dv/dt")
        axes[1].set_xlabel("d(wheel speed)/dt (m/s^2)")
        axes[1].set_ylabel("forward accelerometer a_x (m/s^2)")
        axes[1].legend(fontsize=7)
    paths.append(c.save(fig, folder / "velocity_errors.png"))
    plt.close(fig)
    return paths


def report(run: c.Run, result: dict) -> list:
    lines = [f"velocity_estimation  {run.car}/{run.run_id}  reference {result['reference']}, "
             f"{result['compared_s']:.0f} s compared, {result['map_pose_jumps']} map-pose jumps left out"]
    for name in ("wheel", "fused"):
        if name in result:
            e = result[name]
            lines.append(f"  {name:6} - map: bias {c.fmt(e['mean'])} m/s, RMS {c.fmt(e['rms'])} m/s, "
                         f"p95 |e| {c.fmt(e['p95_abs'])} m/s; distance ratio {c.fmt(e['distance_ratio'], 4)}")
    if "accel_vs_wheel" in result:
        f, axes = result["accel_vs_wheel"], result["accelerometer_axes"]
        if axes["mapping_ok"]:
            lines.append(f"  a_x vs d(v_wheel)/dt: slope {c.fmt(f['slope'])}, r {c.fmt(f['r'])}, "
                         f"accelerometer bias {c.fmt(f['accel_bias_mps2'])} m/s^2")
        else:
            lines.append(f"  WARNING accelerometer axes do not match the car: {axes['problem']}; its a_x is not "
                         f"forward acceleration (fit slope {c.fmt(f['slope'])}, r {c.fmt(f['r'])}). "
                         f"Check imu/car.yaml rpy_deg.")
    return lines


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="run directory, data/<car>/racerbot_runs/<time>")
    ap.add_argument("--out", default=str(c.DEFAULT_OUT), help="results folder (default %(default)s)")
    args = ap.parse_args(argv)
    run = c.load(args.run, TOPICS)
    if not any(run.has(t) for t in c.MAP_POSES) or not run.has("/odom"):
        print(f"{args.run}: needs /odom and a map pose ({', '.join(c.MAP_POSES)})", file=sys.stderr)
        return 1
    folder = c.out_dir(run, args.out)
    result = analyse(run)
    paths = plots(run, result, folder)
    print("\n".join(report(run, result)))
    c.write_summary(folder, "velocity_estimation", {k: v for k, v in result.items() if not k.startswith("_")})
    print(f"  results: {folder}" + (f" ({len(paths)} plots)" if paths else " (no matplotlib: numbers only)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
