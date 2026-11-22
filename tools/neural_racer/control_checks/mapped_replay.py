"""Recorded bag -> map-conditioned policy -> bounded Ackermann values, in memory only.

This diagnostic has no ROS publisher, motor interface, health-status generator,
model approval or training code. Nothing it returns is an authorized command.
"""

from collections import Counter
from dataclasses import fields
import json, math, os, sys
from pathlib import Path
import numpy as np
import yaml
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

ROOT = Path(os.environ["DATA_WORKSPACE"]).resolve()
RUN = ROOT / "data/rb2/racerbot_runs/20261111-152132"
ASSET = ROOT / "data/rb2/racerbot_auto/20261111-151026-rebuilt"
for pkg in ("gap_follow", "neural_racer", "racerbot_car", "pure_pursuit", "drive_intent"):
    sys.path.insert(0, str(ROOT / "src" / pkg))
from neural_racer import observation as obs, preview, model_store
from neural_racer.policy_numpy import PolicyNumpy
from neural_racer.safety_wrapper import SafetyWrapper, WrapperConfig, scan_angles
from pure_pursuit.occupancy_map import OccupancyMap
from racerbot_car.layers import effective_config


def params(key):
    p = RUN / "metadata/params" / f"{key}.yaml"
    return yaml.safe_load(p.read_text())["/" + key]["ros__parameters"]


def load_assets():
    def asset_suffix(s):
        return Path(s).parts[-2:]

    assert asset_suffix(params("pure_pursuit_node")["waypoints_file"]) == (
        ASSET.name,
        "raceline_profiled.csv",
    )
    assert asset_suffix(params("map_server")["yaml_filename"]) == (ASSET.name, "map.yaml")
    grid = OccupancyMap.from_yaml(str(ASSET / "map.yaml"), inflate_cells=0)
    cells = np.where(grid.grid > 0, 100, grid.grid).astype(np.int8)
    profile = np.genfromtxt(ASSET / "raceline_profiled.csv", names=True, delimiter=",")
    xy = np.c_[profile["x"], profile["y"]]
    grid2 = preview.GridMap.from_occupancy(
        cells, cells.shape[1], cells.shape[0], grid.resolution, grid.origin_x, grid.origin_y
    )
    return cells, grid2, xy, preview.ReferencePath.from_grid(xy, grid2)


def ns(h):
    return h.stamp.sec * 10**9 + h.stamp.nanosec


def yaw(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def maps_match(m, cells, grid):
    info = m.info
    origin = info.origin
    return (
        m.header.frame_id == "map"
        and cells.shape == (info.height, info.width)
        and np.allclose(
            [info.resolution, origin.position.x, origin.position.y, yaw(origin.orientation)],
            [grid.res, grid.ox, grid.oy, 0],
            rtol=0,
            atol=1e-6,
        )
        and np.array_equal(cells.ravel(), np.asarray(m.data))
    )


def events():
    p = RUN / "bag"
    info = yaml.safe_load((p / "metadata.yaml").read_text())["rosbag2_bagfile_information"]
    paths = info["relative_file_paths"]
    assert set(paths) == {x.name for x in p.glob("*.mcap")} and len(set(paths)) == len(paths)
    prev = -1
    for name in paths:
        with (p / name).open("rb") as f:
            for _, ch, r, msg in make_reader(
                f, decoder_factories=[DecoderFactory()]
            ).iter_decoded_messages(
                topics=["/scan", "/joy", "/odom", "/pf/viz/inferred_pose", "/map", "/imu/status"]
            ):
                assert r.log_time >= prev
                prev = r.log_time
                yield ch.topic, r.log_time, msg


def evaluate():
    cells, grid, xy, line = load_assets()
    model = Path(os.environ["MODEL"])
    h = model_store.weights_hash(model / "weights.npz")
    assert h == "a0eef2af0a3757b6"
    w, card = model_store.load_dir(model)
    policy = PolicyNumpy(w, card)
    assert "preview" in card["modalities"]
    cfg = effective_config("neural_racer.yaml", "neural_racer_node", "rb2")
    assert (
        cfg["enable_deadman"]
        and cfg["deadman_button"] == 4
        and cfg["require_imu"]
        and cfg["shadow"]
    )
    wrapper = SafetyWrapper(WrapperConfig(**{f.name: cfg[f.name] for f in fields(WrapperConfig)}))
    hist = obs.History()
    ready = False
    joy = odom = pose = None
    hint = None
    previous = 0.0
    prior_t = None
    counts = Counter()
    dispositions = Counter()
    binding = Counter()
    net = []
    bounded = []
    nearest = []
    map_hits = []
    gaps = 0
    health = 0
    stop_reasons = Counter()
    clearance = []
    invalid = []
    examples = []
    hit_angles = []
    hit_ranges = []

    def reset():
        nonlocal hint, previous, prior_t
        hist.reset()
        policy.reset()
        hint = None
        previous = 0.0
        prior_t = None

    def fresh(t, m, seconds):
        return m is not None and -5_000_000 <= t - m[0] < seconds * 1e9

    for topic, t, m in events():
        counts[topic] += 1
        if topic == "/imu/status":
            health += 1
            continue
        if topic == "/map":
            if not maps_match(m, cells, grid):
                raise ValueError("published map does not match the recorded reference")
            ready = True
            continue
        hdr = ns(m.header)
        stamp = min(t, hdr) if hdr - t <= 5_000_000 else None
        if topic == "/joy":
            joy = (
                (stamp, None)
                if stamp is not None and len(m.buttons) > 4 and m.buttons[4] == 1
                else None
            )
            if not joy:
                reset()
            continue
        if topic == "/odom":
            v = float(m.twist.twist.linear.x)
            odom = (stamp, v) if stamp is not None and np.isfinite(v) else None
            continue
        if topic == "/pf/viz/inferred_pose":
            p = m.pose
            val = (p.position.x, p.position.y, yaw(p.orientation))
            pose = (
                (stamp, val)
                if stamp is not None
                and m.header.frame_id in ("map", "/map")
                and np.isfinite(val).all()
                else None
            )
            continue
        reason = None
        if not fresh(t, joy, cfg["joy_timeout_sec"]):
            reason = "deadman_or_joy_timing"
        elif stamp is None or t - stamp > cfg["scan_timeout_sec"] * 1e9:
            reason = "scan_time"
        elif not fresh(t, odom, cfg["odom_timeout_sec"]):
            reason = "odom_time"
        elif not ready or not fresh(t, pose, cfg["pose_timeout_sec"]):
            reason = "map_or_pose_not_ready"
        elif (
            len(m.ranges) != 1081
            or abs(m.angle_min + 3 * math.pi / 4) > 0.0022
            or abs(m.angle_min + 1080 * m.angle_increment - 3 * math.pi / 4) > 0.0022
        ):
            reason = "scan_geometry"
        if reason:
            dispositions[reason] += 1
            reset()
            continue
        if prior_t is not None and t - prior_t > cfg["scan_timeout_sec"] * 1e9:
            reset()
            gaps += 1
        ranges = np.asarray(m.ranges, float)
        encoded, valid = obs.pool_scan(ranges, m.range_min, m.range_max)
        x, y, heading = pose[1]
        block, hint, s0 = line.preview_block(x, y, heading, hint)
        hist.push_tick(encoded, valid, odom[1])
        observation = obs.build_observation(
            hist,
            imu_present=False,
            mode=obs.MODE_MAP,
            v_cap=wrapper.cfg.max_speed,
            preview=block,
            race=obs.race_context(s0 / line.length, 1, None, None),
        )
        out = policy.step(observation)
        world = preview.car_to_world(np.vstack([[0.0, 0.0], out.ego_path]), x, y, heading)
        hit = grid.first_blocked_along(world) if np.isfinite(world).all() else None
        dt = 0.025 if prior_t is None else (t - prior_t) * 1e-9
        r = wrapper.apply(
            out.steering,
            out.speed,
            out.ego_path,
            ranges,
            scan_angles(m.angle_min, m.angle_increment, ranges.size),
            m.range_min,
            m.range_max,
            previous,
            dt,
            hit,
        )
        assert (
            np.isfinite([r.steering, r.speed]).all()
            and abs(r.steering) <= wrapper.cfg.max_steering_angle
            and 0 <= r.speed <= wrapper.cfg.max_speed
        )
        invalid.append(1 - obs.beam_validity(ranges, m.range_min, m.range_max).mean())
        if math.isfinite(r.clearance):
            angles = scan_angles(m.angle_min, m.angle_increment, len(ranges))
            distance = np.where(
                obs.beam_validity(ranges, m.range_min, m.range_max),
                ranges - wrapper.boundary(angles),
                np.inf,
            )
            i = int(np.argmin(distance))
            assert abs(distance[i] - r.clearance) < 1e-10
            hit_angles.append(math.degrees(angles[i]))
            hit_ranges.append(ranges[i])
            clearance.append(r.clearance)
        if r.stopped:
            stop_reasons[r.reason] += 1
            reset()
        else:
            assert (
                abs(r.steering - previous)
                <= wrapper.cfg.max_steering_rate * min(dt, wrapper.cfg.command_slew_max_dt) + 1e-7
            )
            hist.push_command(r.steering, r.speed)
            previous = r.steering
            prior_t = t
        dispositions["map_inference"] += 1
        binding[r.binding] += 1
        net.append((out.steering, out.speed))
        bounded.append((r.steering, r.speed))
        nearest.append(float(np.min(np.linalg.norm(xy - [x, y], axis=1))))
        if hit is not None:
            map_hits.append(hit)
        if len(examples) < 8:
            examples.append(
                dict(
                    bag_time_ns=t,
                    raw=[out.steering, out.speed],
                    wrapped=[r.steering, r.speed],
                    reason=r.reason or r.binding,
                )
            )
    assert counts["/scan"] == sum(dispositions.values()) and counts["/map"] == 1 and health == 0
    a = np.asarray(bounded)
    assert len(a) > 0
    return dict(
        run=RUN.name,
        map_package=ASSET.name,
        model_hash=h,
        mode="map diagnostic only",
        cap_mps=wrapper.cfg.max_speed,
        topics=dict(counts),
        map_cells=int(cells.size),
        reference_points=len(xy),
        reference_length_m=line.length,
        map_grid_equal=True,
        imu_health_messages=health,
        ros_publishers_created=0,
        scan_outcomes=dict(dispositions),
        stop_reasons=dict(stop_reasons),
        median_invalid_beam_fraction=float(np.median(invalid)),
        clearance_min_median_max=(
            np.percentile(clearance, [0, 50, 100]).tolist() if clearance else None
        ),
        limiting_bearing_deg_min_median_max=(
            np.percentile(hit_angles, [0, 50, 100]).tolist() if hit_angles else None
        ),
        limiting_return_m_min_median_max=(
            np.percentile(hit_ranges, [0, 50, 100]).tolist() if hit_ranges else None
        ),
        raw_speed_min_median_max=np.percentile(np.asarray(net)[:, 1], [0, 50, 100]).tolist(),
        examples=examples,
        scan_gap_resets=gaps,
        projected_path_hits=len(map_hits),
        bindings=dict(binding),
        nonzero_bounded_speed=int(np.count_nonzero(a[:, 1] > 0)),
        max_abs_steer_rad=float(abs(a[:, 0]).max()),
        max_bounded_speed_mps=float(a[:, 1].max()),
        median_pose_to_nearest_waypoint_m=float(np.median(nearest)),
        preview_enabled=True,
        scope="Recorded inputs; no IMU health inferred, driver authorization impossible. Diagnostic proposal is not a published command.",
    )


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2, allow_nan=False))
