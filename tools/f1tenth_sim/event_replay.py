#!/usr/bin/env python3
"""Replay a recorded event in the simulator: start the rb1 plant at the recorded pose and speed, run the mirror for a few seconds.

    python3 tools/f1tenth_sim/event_replay.py --seeds 6 --out /tmp/event_replay.json

Starts the plant on car2_course_2026_11_11 at the pose and wheel speed that fixture steer_into_wall_rb1_20261111-155542.npz (rb1 run
20261111-155542, clock 85.60 s on intent_report's clock = 1.10 s into the fixture) has, steering 0, then runs
gap_mirror.GapController with the fixture's recorded parameters for 3.0 s, awareness as recorded and awareness off. Per run it reports the
speed at +0.7 s and +1.3 s, max(v - cmd) over the window, the minimum chassis-to-map clearance (the plant's own distance transform,
never gym's collision flag) with its time and position, and whether the chassis reached occupied space. For seed 12345 it also prints the
real trace against the simulated one every 0.25 s. The controller is cold-started (no history before the first tick) unless
``--warm RUN_DIR``: then it first replays the run's recorded scans from the LB press up to the start, as the real node saw them
(steer_stages.replay_window's method, which reproduces /drive steering within 0.0164 rad over this window).
The plant uses rb1's own calibration (its fitted speed loop, cars/rb1/sim/speed_loop.yaml); ``--shared-speed-loop`` swaps in the shared one.
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


def shared_speed_loop(calibration):
    """``calibration`` with the shared speed loop (calibration.SHARED_SPEED_LOOP) instead of the car's own fitted one."""
    from dataclasses import replace
    from sim_fidelity.calibration import SHARED_SPEED_LOOP
    return replace(calibration, **SHARED_SPEED_LOOP)


def warm_up(ctrl, scans, scan_t, speed_at, gyro_t, gyro_wz, press, imu_timeout=0.1):
    """Feed recorded scans through ``ctrl`` open-loop, exactly as steer_stages.replay_window does (step index (t - press) / DT,
    the wheel speed the node had, the /imu/data gyro if fresh within ``imu_timeout``, LB held). ``scans``/``scan_t`` are the
    scans BEFORE the closed-loop start, from the LB press. Returns the last (steering, speed) and the step index of the last scan."""
    out, step = (math.nan, math.nan), -1.0
    if len(scan_t):
        ctrl.last_command_time = (scan_t[0] - press) / CONTROL_DT - 1.0
    for scan, t in zip(scans, scan_t):
        k = int(np.searchsorted(gyro_t, t, side="right")) - 1
        yaw = float(gyro_wz[k]) if k >= 0 and t - gyro_t[k] <= imu_timeout else None
        step = (t - press) / CONTROL_DT
        out = ctrl.command(np.asarray(scan, dtype=np.float64), float(speed_at(t)), step, yaw_rate=yaw)
    return out, step


def warm_inputs(run_dir, data=None):
    """From the recorded run: the scans from the LB press up to (not including) the one that is the fixture's start scan, their
    times, the wheel speed, the gyro and the press time; plus the recorded /drive just before and at the start scan."""
    sys.path.insert(0, str(HERE.parent))
    from run_analysis import runlog
    from run_analysis.steer_stages import gyro_series
    z = np.load(FIXTURE) if data is None else data
    ref = real_reference(z)
    run = runlog.load_run(run_dir)
    target = np.asarray(z["ranges"][ref["start_index"]], dtype=float)
    hits = [i for i in range(len(run.scan)) if np.allclose(np.asarray(run.scan[i], dtype=float), target, equal_nan=True)]
    if len(hits) != 1:
        raise ValueError(f"the fixture's start scan matches {len(hits)} scans of {run_dir}")
    i0 = hits[0]
    t0 = float(run.scan_t[i0])
    press = max((a for a, b in run.lb_windows(min_duration=0.0) if a <= t0 < b), default=None)
    if press is None:
        raise ValueError("the start scan is not inside an LB hold")
    idx = np.where((run.scan_t > press) & (run.scan_t < t0))[0]
    gyro_t, gyro_wz = gyro_series(run_dir)
    return dict(scans=[run.scan[i] for i in idx], scan_t=run.scan_t[idx], speed_at=run.odom_speed_at, gyro_t=gyro_t,
                gyro_wz=gyro_wz, press=press, start_t=t0)


def run_once(seed, awareness, ref, params, car_id="rb1", warm=None, calibration=None, data=None):
    """One closed-loop run from the recorded state. ``warm``: warm_inputs(...) of the recorded run; the controller first replays
    the recorded scans from the LB press up to the start (warm_up), so it starts with the real node's history instead of none.
    ``calibration``: default the car's own."""
    calibration = calibration_for_car(car_id) if calibration is None else calibration
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
    first_step = 0
    if warm is not None:
        _last, last_step = warm_up(ctrl, warm["scans"], warm["scan_t"], warm["speed_at"], warm["gyro_t"], warm["gyro_wz"],
                                   warm["press"])
        first_step = (warm["start_t"] - warm["press"]) / CONTROL_DT if last_step >= 0 else 0
    rows = []
    open_loop_ticks, first_closed = 0, None
    for k in range(int(round(WINDOW / CONTROL_DT))):
        step = first_step + k            # the node's clock runs on from the warm-up
        ego = obs["agent_0"]
        state = np.asarray(ego["std_state"], dtype=float)
        speed_meas = float(state[3])
        scan_now = np.asarray(ego["scan"], dtype=float)
        if warm is not None and np.count_nonzero(np.isfinite(scan_now) & (scan_now > lid.range_min)) < 0.1 * scan_now.size:
            # The plant's first scans after a reset are zeros (its LiDAR delay line is still filling; one beam inf). The real node had
            # real scans here; a warm controller fed zeros would stop and lose the history it was warmed with. Play the
            # recorded /drive for those ticks instead and do not step the controller (cold runs keep the old behaviour).
            steering, cmd = float(np.interp(k * CONTROL_DT, ref["t"], ref["steering"])), float(np.interp(k * CONTROL_DT, ref["t"], ref["cmd"]))
            open_loop_ticks += 1
        else:
            steering, cmd = ctrl.command(ego["scan"], speed_meas, step, horizontal_accel=0.0, lb_pressed=True,
                                         fused_speed=float(plant.truth(0)[3]), yaw_rate=float(state[5]),
                                         measured_steering=eval_gap.steering_from_yaw_rate(float(state[5]), float(state[3]), p["wheelbase"], 0.3))
            if first_closed is None:
                first_closed = (steering, cmd)
        truth = plant.truth(0)
        rows.append((k * CONTROL_DT, float(truth[3]), cmd, steering, float(truth[0]), float(truth[1]),
                     plant.chassis_map_clearance(0), float(truth[5]), float(truth[6])))
        obs, _r, done, _t, _info = plant.step([(steering, cmd)])
        if done:
            break
    contact_steps = plant.report().get("contact_steps", 0)
    plant.close()
    a = np.array(rows)
    t, v, cmd, st, x, y, clr, yaw_rate, slip = a.T
    k = int(np.nanargmin(clr))
    at = lambda s: float(np.interp(s, t, v))
    return dict(seed=seed, awareness=bool(awareness), warm=bool(warm), first_command=dict(steering=(first_closed or (math.nan,))[0], speed=(first_closed or (math.nan, math.nan))[1]),
                open_loop_ticks=open_loop_ticks,
                start_clearance=start_clearance, v_07=at(0.7), v_13=at(1.3),
                max_v_minus_cmd=float(np.max(v - cmd)), max_yaw_rate=float(np.max(np.abs(yaw_rate))),
                max_slip=float(np.max(np.abs(slip))), min_clearance=float(clr[k]), t_min=float(t[k]), x_min=float(x[k]), y_min=float(y[k]),
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
    ap.add_argument("--warm", metavar="RUN_DIR",
                    help="the recorded run (data/rb1/racerbot_runs/20261111-155542): replay its scans from the LB press up to the start "
                         "through the controller first, as the real node saw them")
    ap.add_argument("--shared-speed-loop", action="store_true",
                    help="use the shared speed loop (calibration.SHARED_SPEED_LOOP) instead of the car's own fitted one")
    ap.add_argument("--tyres", metavar="FRONT,REAR,SHAPE[,COUPLING]",
                    help="saturating tyres (sim_fidelity.tyres.TyreModel): peak friction front and rear, magic-formula shape, "
                         "longitudinal coupling; default gym's linear tyres under the steering clamp, which cannot spin")
    args = ap.parse_args(argv)
    calibration = calibration_for_car("rb1")
    if args.shared_speed_loop:
        calibration = shared_speed_loop(calibration)
    if args.tyres:
        from dataclasses import replace
        from sim_fidelity.tyres import TyreModel
        calibration = replace(calibration, tyre_model=TyreModel(*(float(v) for v in args.tyres.split(","))))
    data = np.load(FIXTURE)
    params = json.loads(str(data["params_json"]))
    ref = real_reference(data)
    t = ref["t"]
    real_over = float(np.max((ref["v"] - ref["cmd"])[t <= 1.74]))
    if args.warm:
        print(f"warm start: recorded /drive at the start: speed {ref['cmd'][0]:.2f} m/s, steering {ref['steering'][0]:+.3f} rad")
    print(f"start: fixture index {ref['start_index']} (pose index {ref['pose_index']}), pose x {ref['pose'][0]:.3f} y {ref['pose'][1]:.3f} "
          f"yaw {ref['pose'][2]:.3f}, wheel speed {ref['speed']:.3f} m/s")
    print(f"real: v at +0.7 s {np.interp(0.7, t, ref['v']):.2f}, +1.3 s {np.interp(1.3, t, ref['v']):.2f}, max(v - cmd) up to the contact {real_over:.2f} m/s; "
          "contact at +1.74 s near x -9.6, y 11.5")
    warm = warm_inputs(args.warm, data) if args.warm else None
    if warm is not None:
        print(f"warm-up: {len(warm['scan_t'])} recorded scans from the LB press ({warm['start_t'] - warm['press']:.2f} s before the start)")
    runs = []
    for awareness in (True, False):
        for k in range(args.seeds):
            r = run_once(args.first_seed + k, awareness, ref, params, warm=warm, calibration=calibration, data=data)
            runs.append(r)
            print(f"awareness {'on ' if awareness else 'off'} seed {r['seed']}: start clearance {r['start_clearance']:.3f} m; v +0.7 {r['v_07']:.2f} "
                  f"+1.3 {r['v_13']:.2f}; max(v-cmd) {r['max_v_minus_cmd']:.2f}; max yaw rate {r['max_yaw_rate']:.2f} rad/s, slip {r['max_slip']:.2f} rad; min clearance {r['min_clearance']:.3f} m at +{r['t_min']:.2f} s "
                  f"(x {r['x_min']:.2f}, y {r['y_min']:.2f}); contact {'YES' if r['contact'] else 'no'}; "
                  f"first command {r['first_command']['speed']:.2f} m/s {r['first_command']['steering']:+.3f} rad", flush=True)
    for r in runs:
        if r["seed"] == args.first_seed:
            overlay(ref, r)
    if args.out:
        Path(args.out).write_text(json.dumps(dict(reference={k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in ref.items()}, runs=runs),
                                             indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
