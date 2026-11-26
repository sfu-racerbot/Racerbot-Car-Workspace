#!/usr/bin/env python3
"""Lateral acceleration: how hard the car really corners, measured two ways.

    python3 tools/run_analysis/dataset_examples/lateral_acceleration.py data/rb1/racerbot_runs/20261026-194514

Model. A car going round a corner accelerates sideways, toward the centre:

    from the gyro          a_y = v r                     (v forward speed, r yaw rate)
    from the steering      a_y = v^2 tan(delta) / L      (what the command asks for)
    from the accelerometer a_y = /imu/data's y axis       (also reads g sin(roll))

v r is the reference: whatever the tyres do, speed times yaw rate IS the
centripetal acceleration. The accelerometer is compared with it (slope 1 if it
is calibrated and the body does not roll), after removing its straight-line
median as grip_truth.py does. Before that, the accelerometer's axes are checked
(common.accel_axes): a mount rotated in car.yaml makes ay read something else.

Grip curve: delivered (v r) against asked (v^2 tan(delta)/L), binned with
grip_truth.curve. The command is first delayed by the steering delay measured
the same way as steering_response.py, so a corner is compared with the command
that made it. grip_truth.py takes only steady corners; on these short courses
there are seldom enough of them (their count is reported), so every driving
sample is used and the bins need 1 s of data each (grip_truth.sustained_limit).

v is the map pose's speed when the run has one (the wheel speed can read
several per cent high, see velocity_estimation.py), else the wheel speed.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from run_analysis import grip_truth  # noqa: E402
from run_analysis.dataset_examples import common as c  # noqa: E402
from run_analysis.dataset_examples.steering_response import MAX_LAG_S, predicted_yaw_rate, shift  # noqa: E402
from run_analysis.speed_tracking import hold  # noqa: E402
from run_analysis.yaw_tracking import xcorr_lag  # noqa: E402

TOPICS = ["/odom", "/imu/data", "/ackermann_cmd", "/joy", *c.MAP_POSES]
CORNERING = 1.0                          # m/s^2, |v r| above this counts as cornering
MIN_BIN_S = 1.0                          # s of data a grip-curve bin needs


def kinematic_lateral(v, yaw_rate) -> np.ndarray:
    """Centripetal acceleration (m/s^2) of a car at speed v turning at yaw_rate: v r."""
    return np.asarray(v, dtype=float) * np.asarray(yaw_rate, dtype=float)


def analyse(run: c.Run) -> dict:
    odom, imu = run["/odom"], run["/imu/data"]
    grid = c.time_grid(max(odom.t[0], imu.t[0]), min(odom.t[-1], imu.t[-1]))
    v_wheel = c.on_grid(grid, odom.t, odom.v)
    topic, v_map, _ = c.map_speed(run, grid)
    use_map = topic is not None and np.isfinite(v_map).any()
    v = v_map if use_map else v_wheel
    r = c.on_grid(grid, imu.t, imu.wz)
    n = max(1, int(round(grip_truth.AVERAGE_SEC * c.FS)))
    ay = c.moving_average(c.on_grid(grid, imu.t, imu.ay), n)
    ax = c.moving_average(c.on_grid(grid, imu.t, imu.ax), n)
    driving = c.inside(grid, c.lb_spans(run)) & (np.abs(v) >= c.MOVING_SPEED)
    a_kin = kinematic_lateral(v, r)
    axes = c.accel_axes(ax, ay, c.derivative(c.moving_average(v_wheel, n), c.FS), a_kin, driving)
    ay_corrected, bias, straight = grip_truth.remove_straight_bias(np.where(driving, v, np.nan), r, ay)
    result = dict(speed_from=topic if use_map else "/odom (wheels)", driving_s=float(driving.sum() / c.FS),
                  accelerometer_axes=axes,
                  accelerometer_straight_bias_mps2=bias, straight_samples=straight,
                  accel_vs_kinematic=c.line_fit(a_kin[driving], ay_corrected[driving]),
                  kinematic=dict(p95_abs_mps2=float(np.nanpercentile(np.abs(a_kin[driving]), 95))
                                 if driving.any() else math.nan,
                                 cornering_s=float(np.sum(np.abs(a_kin[driving]) >= CORNERING) / c.FS)))
    if run.has("/ackermann_cmd"):
        delta = hold(run["/ackermann_cmd"].t, run["/ackermann_cmd"].steer, grid)
        lag_s, _ = xcorr_lag(np.nan_to_num(predicted_yaw_rate(v_wheel, delta)), np.nan_to_num(r), c.FS, MAX_LAG_S)
        delta_d = shift(delta, int(round(lag_s * c.FS)) if math.isfinite(lag_s) else 0)
        steady = grip_truth.steady_corners(grid, np.where(driving, v, np.nan), r, ay_corrected, delta)
        use = driving & np.isfinite(delta_d)
        asked = grip_truth.asked_lateral_accel(v[use], delta_d[use])
        gyro_rows = grip_truth.curve(asked, np.abs(a_kin[use]))
        acc_rows = grip_truth.curve(asked, np.abs(ay_corrected[use]))
        limit, at = grip_truth.sustained_limit(gyro_rows, int(MIN_BIN_S * c.FS))
        result["grip"] = dict(steering_delay_s=lag_s, steady_corner_samples=int(steady.sum()),
                              samples=int(use.sum()), sustained_limit_mps2=limit, at_asked_mps2=at,
                              curve_gyro=gyro_rows, curve_accelerometer=acc_rows)
    result["_curves"] = dict(a_kin=a_kin, ay=ay_corrected, ax=ax, driving=driving)
    return result


def plots(run: c.Run, result: dict, folder: Path) -> list:
    plt = c.pyplot()
    if plt is None:
        return []
    k = result["_curves"]
    m = k["driving"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    axes[0].plot(k["a_kin"][m], k["ay"][m], ".", ms=2, color="tab:purple")
    lim = float(np.nanmax(np.abs(k["a_kin"][m]))) if m.any() else 3.0
    axes[0].plot([-lim, lim], [-lim, lim], "k:", lw=0.8, label="equal")
    fit = result["accel_vs_kinematic"]
    if math.isfinite(fit["slope"]):
        axes[0].plot([-lim, lim], [-fit["slope"] * lim + fit["intercept"], fit["slope"] * lim + fit["intercept"]],
                     "k-", lw=1, label=f"fit: slope {fit['slope']:.2f}, r {fit['r']:.2f}")
    axes[0].set_xlabel(f"v r (m/s^2), v from {result['speed_from']}")
    axes[0].set_ylabel("accelerometer a_y, straight bias removed (m/s^2)")
    if not result["accelerometer_axes"]["mapping_ok"]:
        axes[0].set_title("accelerometer axes wrong: " + result["accelerometer_axes"]["problem"], fontsize=7,
                          color="tab:red")
    axes[0].legend(fontsize=7)
    if "grip" in result and result["grip"]["curve_gyro"]:
        g = result["grip"]
        curves = [(g["curve_gyro"], "o-", "v r (median)")]
        if result["accelerometer_axes"]["mapping_ok"]:      # a mis-mounted accelerometer would only mislead here
            curves.append((g["curve_accelerometer"], "s--", "accelerometer"))
        for rows, style, label in curves:
            axes[1].plot([r["asked"] for r in rows], [r["median"] for r in rows], style, label=f"delivered, {label}")
        top = max([r["asked"] for r in g["curve_gyro"]] + [1.0])
        axes[1].plot([0, top], [0, top], "k:", lw=0.8, label="delivered = asked")
        axes[1].set_xlabel(f"asked, v^2 tan(delta)/L, command delayed {g['steering_delay_s'] * 1000:.0f} ms (m/s^2)")
        axes[1].set_ylabel("delivered (m/s^2)")
        axes[1].set_title("grip curve, all driving samples", fontsize=9)
        axes[1].legend(fontsize=7)
    axes[2].plot(k["ay"][m], k["ax"][m], ".", ms=2, color="0.3")
    axes[2].set_xlabel("lateral a_y (m/s^2)")
    axes[2].set_ylabel("forward a_x (m/s^2)")
    axes[2].set_aspect("equal")
    axes[2].set_title("accelerometer g-g diagram" + ("" if result["accelerometer_axes"]["mapping_ok"]
                                                       else " (axes as recorded: wrong)"), fontsize=9)
    fig.suptitle(f"{run.car} {run.run_id}: lateral acceleration")
    path = c.save(fig, folder / "lateral_acceleration.png")
    plt.close(fig)
    return [path]


def report(run: c.Run, result: dict) -> list:
    f, k, ax = result["accel_vs_kinematic"], result["kinematic"], result["accelerometer_axes"]
    lines = [f"lateral_acceleration  {run.car}/{run.run_id}  v from {result['speed_from']}, {result['driving_s']:.0f} s driving",
             f"  v r: p95 |a_y| {c.fmt(k['p95_abs_mps2'])} m/s^2, {c.fmt(k['cornering_s'])} s above {CORNERING} m/s^2"]
    if ax["mapping_ok"]:
        lines.append(f"  accelerometer vs v r: slope {c.fmt(f['slope'])}, r {c.fmt(f['r'])} "
                     f"(straight-line bias {c.fmt(result['accelerometer_straight_bias_mps2'])} m/s^2 removed)")
    else:
        lines.append(f"  WARNING accelerometer axes do not match the car: {ax['problem']}; its a_y is not lateral "
                     f"acceleration (fit slope {c.fmt(f['slope'])}, r {c.fmt(f['r'])}). Check imu/car.yaml rpy_deg.")
    if "grip" in result:
        g = result["grip"]
        lines.append(f"  grip: sustained {c.fmt(g['sustained_limit_mps2'])} m/s^2 delivered when asked about "
                     f"{c.fmt(g['at_asked_mps2'])} (command delayed {c.fmt(g['steering_delay_s'] * 1000)} ms; "
                     f"{g['steady_corner_samples']} steady-corner samples)")
    return lines


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="run directory, data/<car>/racerbot_runs/<time>")
    ap.add_argument("--out", default=str(c.DEFAULT_OUT), help="results folder (default %(default)s)")
    args = ap.parse_args(argv)
    run = c.load(args.run, TOPICS)
    missing = [t for t in ("/odom", "/imu/data") if not run.has(t)]
    if missing:
        print(f"{args.run}: needs {', '.join(missing)}", file=sys.stderr)
        return 1
    folder = c.out_dir(run, args.out)
    result = analyse(run)
    paths = plots(run, result, folder)
    print("\n".join(report(run, result)))
    c.write_summary(folder, "lateral_acceleration", {k: v for k, v in result.items() if not k.startswith("_")})
    print(f"  results: {folder}" + (f" ({len(paths)} plots)" if paths else " (no matplotlib: numbers only)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
