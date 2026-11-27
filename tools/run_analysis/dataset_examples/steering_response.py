#!/usr/bin/env python3
"""Steering response: how the car really turns for a steering command.

    python3 tools/run_analysis/dataset_examples/steering_response.py data/rb1/racerbot_runs/20261026-194514

Model (kinematic bicycle, rear-axle reference, wheelbase L = 0.36 m):

    predicted yaw rate      r_hat = v tan(delta_cmd) / L
    effective steering      delta_eff = atan(L r / v)       from the measured gyro r

delta_cmd is /ackermann_cmd's steering, what reached the VESC after
ackermann_mux. r is the calibrated gyro (/imu/data, car z axis). The fitted
numbers are:

  * delay: the shift of r_hat that best lines it up with r (cross-correlation,
    yaw_tracking.xcorr_lag), covering servo travel, tyre build-up and transport;
  * yaw gain: r ~ k r_hat once shifted (k = 1 if the model is right);
  * steering map: delta_eff = a delta_cmd + b (a = 1, b = 0 if the servo
    calibration is right; b is the straight-ahead offset);
  * the residual of r after the fit.

Only samples with LB held and the car at least 0.5 m/s are fitted: below that
delta_eff is dominated by gyro noise (vehicle_state_node uses 0.3 m/s).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from run_analysis.dataset_examples import common as c  # noqa: E402
from run_analysis.speed_tracking import hold  # noqa: E402
from run_analysis.yaw_tracking import ls_gain, xcorr_lag  # noqa: E402

TOPICS = ["/ackermann_cmd", "/odom", "/imu/data", "/steering/measured", "/joy", *c.MAP_POSES]
MIN_SPEED = 0.5            # m/s, the fit's floor (yaw_tracking.MOVING_SPEED)
MAX_LAG_S = 0.5            # s, the delay search range
STEER_BINS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.45)   # rad, |delta_cmd|
SPEED_BINS = (0.5, 1.5, 2.5, 3.5, 6.0)                          # m/s
MIN_STEER = 0.05           # rad, |delta_cmd| below which turned / commanded is mostly noise


def predicted_yaw_rate(v, delta, wheelbase: float = c.WHEELBASE):
    """Kinematic bicycle yaw rate (rad/s): v tan(delta) / L."""
    return np.asarray(v, dtype=float) * np.tan(np.asarray(delta, dtype=float)) / wheelbase


def effective_steering(v, yaw_rate, wheelbase: float = c.WHEELBASE, min_speed: float = MIN_SPEED):
    """The steering angle (rad) that would give the measured yaw rate: atan(L r / v); NaN below min_speed."""
    v, r = np.asarray(v, dtype=float), np.asarray(yaw_rate, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.arctan(wheelbase * r / v)
    return np.where(np.abs(v) >= min_speed, out, np.nan)


def shift(x, samples: int) -> np.ndarray:
    """x delayed by ``samples`` (x[n - samples]); NaN where nothing came earlier."""
    x = np.asarray(x, dtype=float)
    out = np.full(len(x), np.nan)
    if samples >= 0:
        out[samples:] = x[:len(x) - samples]
    else:
        out[:samples] = x[-samples:]
    return out


def by_steering(delta_cmd, delta_eff, bins=STEER_BINS) -> list:
    """Median delta_eff / delta_cmd per |delta_cmd| bin: 1 if the car turns as much as asked."""
    delta_cmd, delta_eff = np.asarray(delta_cmd, dtype=float), np.asarray(delta_eff, dtype=float)
    ok = np.isfinite(delta_cmd) & np.isfinite(delta_eff)
    rows = []
    for low, high in zip(bins[:-1], bins[1:]):
        m = ok & (np.abs(delta_cmd) >= max(low, 1e-3)) & (np.abs(delta_cmd) < high)
        if m.sum() >= 25:
            rows.append(dict(low=low, high=high, n=int(m.sum()),
                             ratio_median=float(np.median(delta_eff[m] / delta_cmd[m]))))
    return rows


def by_speed(v, delta_cmd, delta_eff, bins=SPEED_BINS) -> list:
    """Median delta_eff / delta_cmd per speed bin, for |delta_cmd| >= MIN_STEER. A servo
    calibration error is the same at every speed; tyre slip (understeer) grows with speed."""
    v, delta_cmd, delta_eff = (np.asarray(a, dtype=float) for a in (v, delta_cmd, delta_eff))
    ok = np.isfinite(v) & np.isfinite(delta_cmd) & np.isfinite(delta_eff) & (np.abs(delta_cmd) >= MIN_STEER)
    rows = []
    for low, high in zip(bins[:-1], bins[1:]):
        m = ok & (np.abs(v) >= low) & (np.abs(v) < high)
        if m.sum() >= 25:
            rows.append(dict(low=low, high=high, n=int(m.sum()),
                             ratio_median=float(np.median(delta_eff[m] / delta_cmd[m]))))
    return rows


def analyse(run: c.Run) -> dict:
    cmd, odom, gyro = run["/ackermann_cmd"], run["/odom"], run["/imu/data"]
    spans = c.lb_spans(run)
    t0 = max(cmd.t[0], odom.t[0], gyro.t[0])
    t1 = min(cmd.t[-1], odom.t[-1], gyro.t[-1])
    grid = c.time_grid(t0, t1)
    delta = hold(cmd.t, cmd.steer, grid)
    v = c.on_grid(grid, odom.t, odom.v)
    r = c.on_grid(grid, gyro.t, gyro.wz)
    r_hat = predicted_yaw_rate(v, delta)
    # Stopped stretches add zeros to both series, so the whole run can be correlated at once.
    lag_s, lag_r = xcorr_lag(np.nan_to_num(r_hat), np.nan_to_num(r), c.FS, MAX_LAG_S)
    k = int(round(lag_s * c.FS)) if math.isfinite(lag_s) else 0
    delta_d, r_hat_d = shift(delta, k), shift(r_hat, k)
    ok = c.inside(grid, spans) & (np.abs(v) >= MIN_SPEED) & np.isfinite(r) & np.isfinite(r_hat_d)
    gain = ls_gain(r[ok], r_hat_d[ok])
    delta_eff = effective_steering(v, r)
    steering_map = c.line_fit(delta_d[ok], delta_eff[ok])
    result = dict(fitted_s=float(ok.sum() / c.FS), delay_s=lag_s, delay_correlation=lag_r,
                  yaw_gain=gain, steering_map=steering_map,
                  straight_ahead_offset_deg=math.degrees(steering_map["intercept"])
                  if math.isfinite(steering_map["intercept"]) else math.nan,
                  residual_yaw_rate=c.stats((r - gain * r_hat_d)[ok]) if math.isfinite(gain) else c.stats([]),
                  by_steering=by_steering(delta_d[ok], delta_eff[ok]))
    # The model's v is the wheel speed. If that reads high (velocity_estimation), the steering looks
    # weak by the same factor; the map pose's speed takes the wheels out of it.
    topic, v_map, _ = c.map_speed(run, grid)
    if topic is not None:
        ok_map = ok & np.isfinite(v_map) & (v_map >= c.MIN_MAP_SPEED)
        eff_map = effective_steering(v_map, r)
        result["with_map_speed"] = dict(
            speed_from=topic, fitted_s=float(ok_map.sum() / c.FS),
            yaw_gain=ls_gain(r[ok_map], predicted_yaw_rate(v_map, delta_d)[ok_map]),
            steering_map=c.line_fit(delta_d[ok_map], eff_map[ok_map]),
            by_speed=by_speed(v_map[ok_map], delta_d[ok_map], eff_map[ok_map]))
    if run.has("/steering/measured"):
        measured = c.on_grid(grid, run["/steering/measured"].t, run["/steering/measured"].position)
        both = ok & np.isfinite(measured) & np.isfinite(delta_eff)
        result["steering_measured_vs_gyro_deg"] = c.stats(np.degrees(measured[both] - delta_eff[both]))
    result["_curves"] = dict(t=grid - grid[0], r=r, r_hat=r_hat, r_hat_d=r_hat_d, delta_d=delta_d,
                             delta_eff=delta_eff, ok=ok)
    result["_xcorr"] = correlation_curve(np.nan_to_num(r_hat), np.nan_to_num(r), int(MAX_LAG_S * c.FS))
    return result


def correlation_curve(a, b, max_samples: int):
    """(lags in s, correlation of b with a delayed by each lag): the curve xcorr_lag picks its peak from."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    lags, corr = [], []
    for k in range(-max_samples, max_samples + 1):
        x, y = (a[:len(a) - k], b[k:]) if k >= 0 else (a[-k:], b[:len(b) + k])
        lags.append(k / c.FS)
        corr.append(float(np.corrcoef(x, y)[0, 1]) if np.ptp(x) > 0 and np.ptp(y) > 0 else float("nan"))
    return lags, corr


def plots(run: c.Run, result: dict, folder: Path) -> list:
    plt = c.pyplot()
    if plt is None:
        return []
    k = result["_curves"]
    paths = []
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(k["t"], k["r"], "k-", lw=0.8, label="gyro yaw rate r (/imu/data)")
    ax.plot(k["t"], k["r_hat_d"], "-", color="tab:orange", lw=0.9,
            label=f"v tan(delta)/L, delayed {result['delay_s'] * 1000:.0f} ms")
    ax.fill_between(k["t"], 0, 1, where=k["ok"], transform=ax.get_xaxis_transform(), color="0.92", label="fitted")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("yaw rate (rad/s)")
    ax.legend(fontsize=7)
    ax.set_title(f"{run.car} {run.run_id}: measured vs predicted yaw rate")
    paths.append(c.save(fig, folder / "steering_yaw_rate.png"))
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ok = k["ok"]
    axes[0].plot(np.degrees(k["delta_d"][ok]), np.degrees(k["delta_eff"][ok]), ".", ms=2, color="tab:green")
    lim = np.degrees(np.nanmax(np.abs(k["delta_d"][ok]))) if ok.any() else 20
    axes[0].plot([-lim, lim], [-lim, lim], "k:", lw=0.8, label="turns as commanded")
    fit = result["steering_map"]
    if math.isfinite(fit["slope"]):
        xs = np.radians([-lim, lim])
        axes[0].plot(np.degrees(xs), np.degrees(fit["slope"] * xs + fit["intercept"]), "k-", lw=1,
                     label=f"fit: {fit['slope']:.2f} x + {math.degrees(fit['intercept']):.2f} deg")
    axes[0].set_xlabel("commanded steering (deg, delayed)")
    axes[0].set_ylabel("effective steering atan(L r / v) (deg)")
    axes[0].legend(fontsize=7)
    lags, corr = result["_xcorr"]
    axes[1].plot(np.array(lags) * 1000, corr, "k-")
    axes[1].axvline(result["delay_s"] * 1000, color="tab:orange", lw=1,
                    label=f"delay {result['delay_s'] * 1000:.0f} ms")
    axes[1].set_xlabel("shift of the prediction (ms)")
    axes[1].set_ylabel("correlation with the gyro")
    axes[1].legend(fontsize=7)
    paths.append(c.save(fig, folder / "steering_map.png"))
    plt.close(fig)
    return paths


def report(run: c.Run, result: dict) -> list:
    m = result["steering_map"]
    lines = [f"steering_response  {run.car}/{run.run_id}  {result['fitted_s']:.0f} s fitted "
             f"(LB held, >= {MIN_SPEED} m/s)",
             f"  delay {c.fmt(result['delay_s'] * 1000)} ms (correlation {c.fmt(result['delay_correlation'])}), "
             f"yaw gain {c.fmt(result['yaw_gain'])}",
             f"  steering map: delta_eff = {c.fmt(m['slope'])} delta_cmd + "
             f"{c.fmt(result['straight_ahead_offset_deg'])} deg"
             f"  (r {c.fmt(m['r'])})",
             f"  residual yaw rate RMS {c.fmt(result['residual_yaw_rate']['rms'])} rad/s",
             "  turned / commanded by |steering|: " + ", ".join(
                 f"{b['low']:.2f}-{b['high']:.2f} rad {b['ratio_median']:.2f}" for b in result["by_steering"])]
    if "with_map_speed" in result:
        w = result["with_map_speed"]
        lines.append(f"  with {w['speed_from']}'s speed instead of the wheels': yaw gain {c.fmt(w['yaw_gain'])}, "
                     f"steering map slope {c.fmt(w['steering_map']['slope'])} ({w['fitted_s']:.0f} s)")
        lines.append("    turned / commanded by speed: " + ", ".join(
            f"{b['low']:g}-{b['high']:g} m/s {b['ratio_median']:.2f}" for b in w["by_speed"]))
    if "steering_measured_vs_gyro_deg" in result:
        lines.append(f"  /steering/measured vs atan(L r/v): median offset "
                     f"{c.fmt(result['steering_measured_vs_gyro_deg']['mean'])} deg")
    return lines


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="run directory, data/<car>/racerbot_runs/<time>")
    ap.add_argument("--out", default=str(c.DEFAULT_OUT), help="results folder (default %(default)s)")
    args = ap.parse_args(argv)
    run = c.load(args.run, TOPICS)
    missing = [t for t in ("/ackermann_cmd", "/odom", "/imu/data") if not run.has(t)]
    if missing:
        print(f"{args.run}: needs {', '.join(missing)}", file=sys.stderr)
        return 1
    folder = c.out_dir(run, args.out)
    result = analyse(run)
    paths = plots(run, result, folder)
    print("\n".join(report(run, result)))
    c.write_summary(folder, "steering_response", {k: v for k, v in result.items() if not k.startswith("_")})
    print(f"  results: {folder}" + (f" ({len(paths)} plots)" if paths else " (no matplotlib: numbers only)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
