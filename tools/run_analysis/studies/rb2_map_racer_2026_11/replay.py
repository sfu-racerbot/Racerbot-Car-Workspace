"""Read-only reproduction of the November 2026 Car 2 map-racer evidence."""
import argparse
from collections import defaultdict
from itertools import groupby
import json
import math
from pathlib import Path
import re
import sys

import numpy as np
import yaml
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
from PIL import Image
from scipy.integrate import cumulative_trapezoid

ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / "data/rb2"
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src/pure_pursuit"))
from run_analysis.loc import Grid, global_search, match, search_points
from pure_pursuit.racing_math import vehicle_boundary_distances

PRIMARY = "20261109-190202"
FLANK = "20261111-152132"
RUNS = (PRIMARY, "20261109-190356", "20261109-190617", "20261109-190837", FLANK)
TOPICS = ("/joy", "/odom", "/scan", "/drive_intent", "/drive", "/map", "/rosout",
          "/pf/viz/inferred_pose", "/sensors/imu/raw")


def yaw(q):
    return math.atan2(2 * (q.w*q.z + q.x*q.y), 1 - 2 * (q.y*q.y + q.z*q.z))


def check_grid(msg, map_dir):
    cfg = yaml.safe_load((map_dir / "map.yaml").read_text())
    pixels = np.asarray(Image.open(map_dir / cfg["image"]))[::-1] / 255
    occ = pixels if cfg["negate"] else 1 - pixels
    saved = np.where(occ > cfg["occupied_thresh"], 100, np.where(occ < cfg["free_thresh"], 0, -1))
    info = msg.info
    origin = [info.origin.position.x, info.origin.position.y, yaw(info.origin.orientation)]
    if not (np.allclose(cfg["origin"], origin, atol=1e-7, rtol=0)
            and abs(cfg["resolution"] - info.resolution) < 1e-9
            and np.array_equal(saved, np.asarray(msg.data).reshape(info.height, info.width))):
        raise ValueError(f"Recorded occupancy grid does not match {map_dir.name}")


def read_run(run_id):
    run_dir = DATA / "racerbot_runs" / run_id
    map_dir = DATA / "racerbot_auto" / (
        "20261111-151026-rebuilt" if run_id == FLANK else "20261109-185146-rebuilt")
    line = map_dir / "raceline_profiled.csv"
    topic_data = defaultdict(list)
    line_confirmed = False
    scan_geom = None
    for bag in sorted((run_dir / "bag").glob("*.mcap")):
        with bag.open("rb") as stream:
            reader = make_reader(stream, decoder_factories=[DecoderFactory()])
            for _, ch, msg, m in reader.iter_decoded_messages(topics=TOPICS):
                t = msg.log_time * 1e-9
                if ch.topic == "/rosout":
                    if m.name == "pure_pursuit_node" and "waypoints from" in m.msg:
                        if f"/{line.relative_to(ROOT)}'" not in m.msg:
                            raise ValueError(f"Unexpected line: {m.msg}")
                        line_confirmed = True
                    continue
                if ch.topic == "/map":
                    check_grid(m, map_dir)
                    row = (t, m.info.width, m.info.height)
                elif ch.topic == "/drive_intent":
                    intent = json.loads(m.data)
                    row = (t, intent["state"], intent["commanded_speed"], intent.get("reason", ""))
                else:
                    h = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
                    if ch.topic == "/scan":
                        geom = (m.angle_min, m.angle_increment, m.range_min, m.time_increment)
                        if scan_geom is not None and geom != scan_geom:
                            raise ValueError("Scan geometry changed within the run")
                        scan_geom = geom
                        row = (t, h, np.asarray(m.ranges, dtype=np.float32))
                    elif ch.topic == "/pf/viz/inferred_pose":
                        row = (t, h, m.pose.position.x, m.pose.position.y, yaw(m.pose.orientation))
                    elif ch.topic == "/sensors/imu/raw":
                        row = (t, h, m.angular_velocity.z)
                    elif ch.topic == "/odom":
                        row = (t, h, m.twist.twist.linear.x, yaw(m.pose.pose.orientation))
                    elif ch.topic == "/drive":
                        row = (t, h, m.drive.speed)
                    elif ch.topic == "/joy":
                        row = (t, h, m.buttons[4])
                topic_data[ch.topic].append(row)
    if not (topic_data["/map"] and line_confirmed):
        raise ValueError(f"{run_id}: missing recorded map or controller line selection")
    for topic, rows in topic_data.items():
        rows.sort(key=lambda r: r[0])
        if topic not in ("/scan", "/drive_intent"):
            topic_data[topic] = np.asarray(rows)
    params_file = run_dir / "metadata/params/pure_pursuit_node.yaml"
    params = (yaml.safe_load(params_file.read_text())["/pure_pursuit_node"]["ros__parameters"]
              if params_file.exists() else None)
    if params and not params["waypoints_file"].endswith("/" + str(line.relative_to(ROOT))):
        raise ValueError("Recorded waypoints parameter disagrees with startup log")
    return topic_data, map_dir, params, scan_geom


def heading_study(data, map_dir, laser_x, geom, hold):
    """Independent scan alignment; filter-seeded fits are diagnostics, not the reference."""
    pf = data["/pf/viz/inferred_pose"]
    imu = data["/sensors/imu/raw"]
    od = data["/odom"]
    start = hold[0]
    grid = Grid.from_map(map_dir)
    line = np.loadtxt(map_dir / "raceline_profiled.csv", delimiter=",", skiprows=1)[:, :2]
    origin, step, _, time_increment = geom
    scans = [s for s in data["/scan"] if start-.6 <= s[0] <= hold[-1]+.3]
    center = (len(scans[0][2])-1) / 2
    t = np.array([h + center*time_increment for _, h, _ in scans])
    pf_yaw = np.interp(t, pf[:, 1], np.unwrap(pf[:, 4]))
    pf_xy = np.array([np.interp(t, pf[:, 1], pf[:, dim]) for dim in (2, 3)]).T
    pf_xy -= laser_x * np.c_[np.cos(pf_yaw), np.sin(pf_yaw)]

    points, poses, diagnostics = [], [], []
    for i, (_, _, ranges) in enumerate(scans):
        ids = np.flatnonzero(np.isfinite(ranges) & (ranges > .35) & (ranges < 10))[::3]
        angles = origin + ids*step
        bx, by = laser_x + ranges[ids]*np.cos(angles), ranges[ids]*np.sin(angles)
        points.append((bx, by, (ids-center)*time_increment))
        alt = match(grid, bx, by, (*pf_xy[i], pf_yaw[i]), iters=20, trim=.45)
        if not poses:
            fit = global_search(grid, bx, by, search_points(grid, line))
        else:
            prior = np.array(poses[-1])
            if i >= 2:
                prior += (prior - poses[-2]) * min(1.5, (t[i]-t[i-1])/(t[i-1]-t[i-2]))
            fit = match(grid, bx, by, prior, iters=20, trim=.45)
        poses.append(fit[0])
        diagnostics.append(np.degrees(alt[0][2]-fit[0][2]))
    poses = np.array(poses)
    poses[:, 2] = np.unwrap(poses[:, 2])
    # Deskew each beam using motion from the preliminary scan fits, not the gyro.
    velocities = np.gradient(poses, t, axis=0)
    ref, h10 = [], []
    for (bx, by, dt), (vx, vy, w), pose in zip(points, velocities, poses):
        th = pose[2]
        vx, vy = vx*np.cos(th)+vy*np.sin(th), -vx*np.sin(th)+vy*np.cos(th)
        c, s = np.cos(w*dt), np.sin(w*dt)
        fit, score, _ = match(grid, c*bx-s*by+dt*vx, s*bx+c*by+dt*vy, pose, iters=20)
        ref.append(fit[2])
        h10.append(score)
    bias = np.median(imu[(imu[:, 0] < start) & (imu[:, 0] >= start-2), 2])
    gyro = cumulative_trapezoid(imu[:, 2]-bias, imu[:, 1], initial=0)  # VESC: deg/s
    pf_last = np.searchsorted(pf[:, 0], [s[0] for s in scans], side="right")-1
    if np.any(pf_last < 0):
        raise ValueError("No filter estimate before first scan")
    ref = np.degrees(np.unwrap(ref))
    wrap = lambda deg: (deg+180) % 360-180
    fields = {
        "odom": np.interp(t, od[:, 1], np.degrees(np.unwrap(od[:, 3]))) - ref,
        "gyro": np.interp(t, imu[:, 1], gyro) - ref,
        "PF by header": wrap(np.degrees(pf_yaw)-ref),
        "PF latest by receipt": wrap(np.degrees(pf[pf_last, 4])-ref),
        "fraction of endpoints within 10 cm": np.array(h10),
        "seed heading difference": wrap(np.array(diagnostics)),
    }
    brake = od[(od[:, 1] > start+2) & (od[:, 2] <= 4.91)][0]
    end = od[(od[:, 1] > brake[1]) & (od[:, 2] <= 3.07)][0]
    lo, hi = brake[1]-start, end[1]-start
    print(f"  wheel speed: {brake[2]:.3f} to {end[2]:.3f} m/s in {hi-lo:.3f} s")
    for lo, hi in [(hi-2, hi), (lo, hi), (3.05, 3.5)]:
        elapsed = t-start
        ts = np.r_[lo, elapsed[(elapsed > lo) & (elapsed < hi)], hi]
        print(f"  {lo:.3f}–{hi:.3f} s, {len(ts)-2} scans")
        for name, values in fields.items():
            v = np.interp(ts, elapsed, values)
            if name in ("odom", "gyro"):
                print(f"    {name}: max heading-change error {abs(v-v[0]).max():.2f}°")
            elif name.startswith("PF"):
                print(f"    {name}: median/max abs error {np.median(abs(v)):.2f}/{abs(v).max():.2f}°")
            elif name.startswith("fraction"):
                print(f"    {name}: min {v.min():.4f}")
            else:
                print(f"    {name}: max {abs(v).max():.2f}°")
    for topic in ("/pf/viz/inferred_pose", "/odom", "/sensors/imu/raw", "/drive", "/drive_intent", "/scan"):
        rx = np.array([r[0]-start for r in data[topic]])
        rx = rx[(rx > 1.5) & (rx < 3.5)]
        i = np.argmax(np.diff(rx))
        print(f"  {topic} longest receipt gap: {rx[i]:.3f}–{rx[i+1]:.3f} s")


def flank_study(data, geom, params, lo, hi):
    ori, inc, range_min, _ = geom
    n = len(data["/scan"][0][2])
    angles = ori + np.arange(n)*inc
    envelope = vehicle_boundary_distances(angles, params["car_width"], params["car_length"],
                                         params["wheelbase"], params["laser_offset_x"],
                                         params["laser_offset_y"])
    half = np.radians(params["body_clearance_fov_deg"])/2
    # Inclusive beam bounds, rounded as in the controller recorded on race day.
    cone = (np.arange(n) >= int((-half-ori)/inc)) & (np.arange(n) <= int((half-ori)/inc))
    close = []
    for rx, _, rs in data["/scan"]:
        if lo <= rx <= hi:
            usable = cone & np.isfinite(rs) & (rs >= range_min)
            cl = np.where(usable, rs-envelope, np.inf)
            idx = cl.argmin()
            close.append((cl[idx], np.degrees(angles[idx]), cl[abs(angles) < np.radians(15)].min()))
    print(f"  {hi-lo:.3f} s; {len(close)} scans")
    for col, name in enumerate(("body clearance (m)", "bearing (°)", "front ±15° clearance (m)")):
        print(f"    {name} min/median/max:", np.round(np.quantile(np.asarray(close)[:, col], [0, .5, 1]), 4))
    for topic in ("/drive", "/odom"):
        a = data[topic]
        speeds = a[(a[:, 0] >= lo) & (a[:, 0] <= hi), 2]
        print(f"    {topic}: {len(speeds)} speeds; min/max {speeds.min():.3f}/{speeds.max():.3f} m/s")


def report(run):
    data, map_dir, params, geom = read_run(run)
    joy = data["/joy"]
    hold = joy[joy[:, 2] == 1, 0]
    start = hold[0]
    print(f"{run}: map and line {map_dir.name}")
    intents = data["/drive_intent"]
    stops = [list(g) for key, g in groupby(intents, lambda row: row[1]) if key == "body_contact"]
    for g in stops:
        t, _, _, reason = g[0]
        print(f"  body_contact: {t-start:.3f} s; {reason.split(';')[0]}; "
              f"{re.search(r'nearest waypoint=[0-9]+', reason).group()}; "
              f"{sum(r[2] == 0 for r in g)}/{len(g)} zero-speed intents")
    if not stops:
        print("  No body_contact; recorded states:", sorted({row[1] for row in intents}))
    if run == PRIMARY:
        first_avoid = next(r for r in intents if r[1] == "lidar_avoidance")
        print(f"  avoidance {first_avoid[0]-start:.3f} s: {first_avoid[3].split('; base plan')[0]}")
        drive = data["/drive"]
        print(f"  first zero drive: {drive[(drive[:, 0] > start) & (drive[:, 2] == 0)][0, 0]-start:.3f} s")
        heading_study(data, map_dir, params["laser_offset_x"], geom, hold)
    elif run == FLANK:
        longest = max(stops, key=lambda g: g[-1][0]-g[0][0])
        flank_study(data, geom, params, longest[0][0], longest[-1][0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", choices=RUNS, action="append", help="repeatable; default: all five bags")
    for run_id in parser.parse_args().run or RUNS:
        report(run_id)
