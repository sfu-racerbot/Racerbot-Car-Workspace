#!/usr/bin/env python3
"""Replay a recorded event in the simulator: start the rb1 plant at the recorded pose and speed, run the mirror for a few seconds.

    python3 tools/f1tenth_sim/event_replay.py --seeds 6 --out /tmp/event_replay.json

Starts the plant on car2_course_2026_11_11 at the pose and wheel speed that fixture steer_into_wall_rb1_20261111-155542.npz (rb1 run
20261111-155542, clock 85.60 s on intent_report's clock = 1.10 s into the fixture) has, steering 0, then runs
gap_mirror.GapController with the fixture's recorded parameters for 3.0 s, awareness as recorded and awareness off. Per run it reports the
speed at +0.7 s and +1.3 s, max(v - cmd) over the window, the minimum chassis-to-map clearance (the plant's own distance transform,
never gym's collision flag) with its time and position, and whether the chassis reached occupied space. For seed 12345 it also prints the
real trace against the simulated one every 0.25 s. The controller is cold-started (no history before the first tick).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.append(str(HERE.parent))
from sim_fidelity.bootstrap import bootstrap  # noqa: E402

bootstrap()
import eval_gap  # noqa: E402
from eval_gap import CONTROL_DT  # noqa: E402
from gap_mirror import GapController  # noqa: E402
from sim_fidelity import calibration_for_car  # noqa: E402

FIXTURE = HERE.parents[1] / "src" / "gap_follow" / "test" / "data" / "steer_into_wall_rb1_20261111-155542.npz"
TRACK = "car2_course_2026_11_11"
CLOCK_OFFSET = 84.5            # fixture time + 84.5 = intent_report's clock
START_CLOCK = 85.60
WINDOW = 3.0


def real_reference(data=None):
    """The recorded trace from the start index: (start pose, start speed, arrays relative to the start), cmd/steering held forward."""
    z = np.load(FIXTURE) if data is None else data
    t = z["times"]
    i0 = int(np.argmin(np.abs(t - (START_CLOCK - CLOCK_OFFSET))))
    finite = np.where(np.isfinite(z["poses"][:, 0]))[0]
    ip = int(finite[np.argmin(np.abs(finite - i0))])          # the nearest finite pose
    sel = (t >= t[i0]) & (t <= t[i0] + WINDOW)
    cmd = np.array(z["drive_speed"], dtype=float)
    st = np.array(z["drive_steering"], dtype=float)
    for arr in (cmd, st):                                       # zero-order hold over ticks without a /drive
        last = math.nan
        for k in range(len(arr)):
            last = arr[k] if np.isfinite(arr[k]) else last
            arr[k] = last
    return dict(pose=z["poses"][ip], pose_index=ip, start_index=i0, speed=float(z["odom_speeds"][i0]),
                t=t[sel] - t[i0], v=z["odom_speeds"][sel], cmd=cmd[sel], steering=st[sel], poses=z["poses"][sel])


def run_once(seed, awareness, ref, params, car_id="rb1"):
    calibration = calibration_for_car(car_id)
    plant = eval_gap.make_plant(TRACK, seed, calibration=calibration, car_id=car_id)
    pose = np.array([ref["pose"]], dtype=float)
    plant.reset(pose, speeds=[ref["speed"]])
    plant.step([(0.0, 0.0)])                                     # builds the map's distance transform
    start_clearance = plant.chassis_map_clearance(0)
    obs, _ = plant.reset(pose, speeds=[ref["speed"]])
    lid = type("Lidar", (), {})()
    lid.angle_min = float(-eval_gap.components_half_fov(car_id))
    lid.angle_increment = float(eval_gap.components_increment(car_id))
    lid.range_min = float(eval_gap.components_range_min(car_id))
    p = dict(params)
    p["enable_opponent_awareness"] = bool(awareness)
    ctrl = GapController(p, lid)
    rows = []
    for step in range(int(round(WINDOW / CONTROL_DT))):
        ego = obs["agent_0"]
        state = np.asarray(ego["std_state"], dtype=float)
        speed_meas = float(state[3])
        steering, cmd = ctrl.command(ego["scan"], speed_meas, step, horizontal_accel=0.0, lb_pressed=True,
                                     fused_speed=float(plant.truth(0)[3]), yaw_rate=float(state[5]),
                                     measured_steering=eval_gap.steering_from_yaw_rate(float(state[5]), float(state[3]), p["wheelbase"], 0.3))
        rows.append((step * CONTROL_DT, float(plant.truth(0)[3]), cmd, steering, float(plant.truth(0)[0]), float(plant.truth(0)[1]),
                     plant.chassis_map_clearance(0)))
        obs, _r, done, _t, _info = plant.step([(steering, cmd)])
        if done:
            break
    contact_steps = plant.report().get("contact_steps", 0)
    plant.close()
    a = np.array(rows)
    t, v, cmd, st, x, y, clr = a.T
    k = int(np.nanargmin(clr))
    at = lambda s: float(np.interp(s, t, v))
    return dict(seed=seed, awareness=bool(awareness), start_clearance=start_clearance, v_07=at(0.7), v_13=at(1.3),
                max_v_minus_cmd=float(np.max(v - cmd)), min_clearance=float(clr[k]), t_min=float(t[k]), x_min=float(x[k]), y_min=float(y[k]),
                contact=bool(contact_steps > 0 or clr.min() <= 0.0), contact_steps=int(contact_steps), steps=len(rows),
                trace=dict(t=t.tolist(), v=v.tolist(), cmd=cmd.tolist(), steering=st.tolist(), x=x.tolist(), y=y.tolist()))


def overlay(ref, run):
    print(f"-- overlay, seed {run['seed']}, awareness {'on' if run['awareness'] else 'off'} (t from the start; real = recorded rb1 run)")
    print("   t     real v  sim v | real cmd sim cmd | real steer sim steer")
    t = np.array(run["trace"]["t"])
    for s in np.arange(0.0, WINDOW + 1e-9, 0.25):
        j = min(int(np.searchsorted(t, s)), len(t) - 1)
        rv, rc, rs = (float(np.interp(s, ref["t"], arr)) for arr in (ref["v"], ref["cmd"], ref["steering"]))
        print(f"  {s:4.2f}   {rv:5.2f} {run['trace']['v'][j]:6.2f} | {rc:7.2f} {run['trace']['cmd'][j]:7.2f} | {rs:+9.3f} {run['trace']['steering'][j]:+9.3f}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--first-seed", type=int, default=12345)
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    data = np.load(FIXTURE)
    params = json.loads(str(data["params_json"]))
    ref = real_reference(data)
    t = ref["t"]
    real_over = float(np.max((ref["v"] - ref["cmd"])[t <= 1.74]))
    print(f"start: fixture index {ref['start_index']} (pose index {ref['pose_index']}), pose x {ref['pose'][0]:.3f} y {ref['pose'][1]:.3f} "
          f"yaw {ref['pose'][2]:.3f}, wheel speed {ref['speed']:.3f} m/s")
    print(f"real: v at +0.7 s {np.interp(0.7, t, ref['v']):.2f}, +1.3 s {np.interp(1.3, t, ref['v']):.2f}, max(v - cmd) up to the contact {real_over:.2f} m/s; "
          "contact at +1.74 s near x -9.6, y 11.5")
    runs = []
    for awareness in (True, False):
        for k in range(args.seeds):
            r = run_once(args.first_seed + k, awareness, ref, params)
            runs.append(r)
            print(f"awareness {'on ' if awareness else 'off'} seed {r['seed']}: start clearance {r['start_clearance']:.3f} m; v +0.7 {r['v_07']:.2f} "
                  f"+1.3 {r['v_13']:.2f}; max(v-cmd) {r['max_v_minus_cmd']:.2f}; min clearance {r['min_clearance']:.3f} m at +{r['t_min']:.2f} s "
                  f"(x {r['x_min']:.2f}, y {r['y_min']:.2f}); contact {'YES' if r['contact'] else 'no'}", flush=True)
    for r in runs:
        if r["seed"] == args.first_seed:
            overlay(ref, r)
    if args.out:
        Path(args.out).write_text(json.dumps(dict(reference={k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in ref.items()}, runs=runs),
                                             indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
