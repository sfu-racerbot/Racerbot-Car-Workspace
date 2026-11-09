#!/usr/bin/env python3
"""Four-lap, two-car experiments; no ROS imports or ROS graph.

The opponent and reactive decisions execute the node's actual method bodies.
See NodeDecisions: only message/clock/map I/O is adapted. Metrics use truth and
FidelityPlant's body checks, never the controller's opponent estimate or gym flag.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_gap  # noqa: E402,F401
import run_validation as rv  # noqa: E402
import numpy as np  # noqa: E402
from pure_pursuit import following, pass_completion, pass_planner, racing_math, tracking_laws  # noqa: E402
from pure_pursuit.opponent_gate import OpponentGate  # noqa: E402
from sim_fidelity import with_floor  # noqa: E402

NODE = rv.ROOT / 'src/pure_pursuit/pure_pursuit/pure_pursuit_node.py'
METHODS = {
    '_update_opponent_and_overtake', '_detect_opponent', '_abandon_overtake',
    '_reactive_override', '_fov_indices', '_closest_in_cone',
    '_footprint_clearance', '_forward_body_clearance', '_arc_clearance',
    '_dynamic_closest_in_cone', '_static_closest_in_cone',
    '_sanitized_window', '_emergency_escape',
    '_pass_reset_state', '_pass_limits', '_pass_line_frame', '_pass_walls', '_pass_tracking_input',
    '_pass_enter_abort', '_pass_opponent', '_local_pass_update', '_pass_reset_after',
    '_following_reserve',
}


def node_decisions():
    """Load a fixed allow-list of ROS-free method bodies, without importing ROS.

    An explicit source adapter avoids a third handwritten opponent/safety mirror.
    It compiles only OpponentTracker and these methods; annotations are deferred,
    imports, decorators, node construction and all publishing code are excluded.
    Missing methods fail loudly. The source SHA is recorded with every run.
    """
    tree = ast.parse(NODE.read_text())
    tracker = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'OpponentTracker')
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'PurePursuitNode')
    methods = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in METHODS]
    if {n.name for n in methods} != METHODS:
        raise RuntimeError('node decision adapter needs updating')
    adapter = ast.ClassDef(name='Decisions', bases=[], keywords=[], body=methods, decorator_list=[])
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0),
                              tracker, adapter], type_ignores=[])
    namespace = dict(math=math, np=np, racing_math=racing_math, following=following,
                     pass_completion=pass_completion, pass_planner=pass_planner, tracking_laws=tracking_laws)
    exec(compile(ast.fix_missing_locations(module), str(NODE), 'exec'), namespace)
    return namespace['Decisions'], namespace['OpponentTracker']


Decisions, OpponentTracker = node_decisions()


class NodeDecisions(Decisions):
    def __init__(self, plan, lidar, expected_ranges, wall_distance, params):
        self.__dict__.update(params)
        self.xy, self.seg_len, self.speed_profile = plan.xy, plan.seg_len, plan.speed
        self.num_waypoints = len(plan.xy)
        self.cumulative_arc_length = racing_math.compute_cumulative_arc_length(plan.seg_len)
        self.total_track_length = float(sum(plan.seg_len))
        self.closed_loop = True
        self.opponent = OpponentTracker(self.opponent_velocity_smoothing, self.opponent_lost_timeout_sec)
        self.opponent_gate = OpponentGate(
            confirm_count=params['opponent_confirm_count'], window_sec=params['opponent_confirm_window_sec'],
            min_span_sec=params['opponent_confirm_min_span_sec'], max_jump_m=params['opponent_max_jump_m'],
            max_reverse_speed=params['opponent_max_reverse_speed'], max_speed=params['opponent_max_speed'],
            min_wall_distance=params['opponent_min_wall_distance'])
        self.opponent_wall_distance = SimpleNamespace(at=wall_distance)
        self.map_ray_caster = SimpleNamespace(expected_ranges=expected_ranges)
        self.lidar = lidar
        self.overtake_active, self.overtake_side = False, 1
        self._overtake_retry_after_sec = 0.0
        self._boundary_geometry = None
        self.last_opponent_status = ''
        self.now = 0.0
        self.log = []
        self.recovery_since = None
        self.recovery_exhausted = False
        self.follow_speed_cap = None
        self.current_speed = 0.0
        self._pass_reset_state()

    def _odom_fresh(self):
        return math.isfinite(self.current_speed)

    def _now_sec(self):
        return self.now

    def _seconds_since(self, stamp):
        return math.inf if stamp is None else self.now - stamp

    def get_logger(self):
        return SimpleNamespace(info=self.record, warn=self.record)

    def record(self, text, **kwargs):
        self.log.append((self.now, text))

    def observe(self, state, ranges, now, previous_steering):
        self.car_x, self.car_y, _, _, self.car_yaw = state[:5]
        self.now = now
        self.steering_basis = previous_steering
        self.current_speed = float(state[3])
        self.last_scan_time = now
        self.last_scan = SimpleNamespace(
            ranges=ranges, angle_min=self.lidar.angle_min,
            angle_max=self.lidar.angle_min + (len(ranges) - 1) * self.lidar.angle_increment,
            angle_increment=self.lidar.angle_increment, range_min=self.lidar.range_min)

    def base_command(self, follower, state, previous_steering):
        """Node control_step's bounded off-line recovery, before opponents/safety."""
        self.follow_speed_cap = None
        nearest, error = racing_math.find_nearest_index(
            self.xy, state[:2], closed=True, prev_index=follower.previous_index,
            search_window=self.nearest_search_window)
        if error > self.max_cross_track_error:
            nearest, error = racing_math.find_nearest_index(self.xy, state[:2], closed=True)
        recovering = error > self.max_cross_track_error
        self.tracking_law = follower.law
        if recovering and self.enable_local_pass_planner and self._pass_state != 'idle':
            # Node: aiming at the line beside a car is an unchecked rejoin, so stop.
            self._pass_enter_abort('off the racing line mid-pass')
            self._pass_plan, self._pass_hold_steering = None, True
            self.follow_speed_cap = 0.0
            return previous_steering, 0.0, nearest, None, 'off_racing_line_recovery'
        if recovering:
            if self.recovery_since is None:
                self.recovery_since = self.now
            if self.now - self.recovery_since > self.off_racing_line_recovery_timeout_sec:
                self.recovery_exhausted = True
            if error > self.off_racing_line_recovery_max_error or self.recovery_exhausted:
                follower.previous_index = None
                return previous_steering, 0.0, nearest, None, 'off_racing_line'
            steering, speed, _, _, _ = follower.command(
                state, speed=min(self.off_racing_line_recovery_speed, self.max_speed),
                target_override=self.xy[nearest], previous_steering=previous_steering)
            return steering, speed, nearest, self.xy[nearest], 'off_racing_line_recovery'
        self.recovery_since, self.recovery_exhausted = None, False
        # The node advances its stateful law once, even if a pass
        # subsequently replaces the steering target. A stage-2 pass plan (from
        # the previous tick, as in the node) replaces the path the law tracks.
        if self.enable_local_pass_planner and (self._pass_hold_steering or self._pass_plan is not None):
            follower.previous_index = nearest
            speed = float(self.speed_profile[nearest])
            if self._pass_hold_steering:
                steering = previous_steering
            else:
                inp = tracking_laws.TrackingInput(float(state[0]), float(state[1]), float(state[4]),
                                                  abs(float(state[3])), nearest, previous_steering)
                steering = follower.law.steer(self._pass_tracking_input(inp), self._pass_plan.context).steering
        else:
            steering, speed, nearest, _, _ = follower.command(state, previous_steering=previous_steering)
        target = self._update_opponent_and_overtake(
            nearest, allow_pass=self.enable_opponent_overtake, desired_steering=steering)
        if target is not None:
            steering, speed, nearest, _, _ = follower.command(
                state, target_override=target, previous_steering=previous_steering)
        return steering, speed, nearest, target, None


class Progress:
    """Unwrap nearest-path truth; initial grid head start is not race distance."""
    def __init__(self, plan, position):
        self.plan = plan
        self.cumulative = racing_math.compute_cumulative_arc_length(plan.seg_len)
        self.length = float(sum(plan.seg_len))
        self.index, _ = racing_math.find_nearest_index(plan.xy, position, closed=True)
        self.arc = float(self.cumulative[self.index])
        self.distance = 0.0

    def update(self, position):
        self.index, _ = racing_math.find_nearest_index(
            self.plan.xy, position, closed=True, prev_index=self.index, search_window=40)
        arc = float(self.cumulative[self.index])
        self.distance += (arc - self.arc + self.length / 2) % self.length - self.length / 2
        self.arc = arc
        return self.distance


def race_metrics(rows, dt, length, laps):
    """Physical passes require 1 m lead held 0.5 s; attempts are node commits.

    Stuck = within 3 m behind, commanding at least 0.3 m/s below the solo
    profile. Count ticks rather than labelling every close-follow tick a loss.
    """
    attempts = abandoned = completed = 0
    was_active = False
    pass_armed = bool(rows and rows[0]['head_start'] > 0)
    clear_ticks = 0
    finish = [None, None]
    stuck = 0
    for row in rows:
        active = row['active']
        attempts += int(active and not was_active)
        abandoned += int(was_active and not active and 'abandoned' in row['status'])
        was_active = active
        lead = row['ego_progress'] - row['opp_progress'] - row['head_start']
        if lead < -1.0:
            pass_armed = True
        clear_ticks = clear_ticks + 1 if lead >= 1.0 else 0
        if pass_armed and clear_ticks * dt >= 0.5:
            completed += 1
            pass_armed = False
        stuck += int(-3.0 <= lead < 0.0 and row['ego_command'] < row['profile_speed'] - 0.3)
        for i, key in enumerate(('ego_progress', 'opp_progress')):
            required = length * laps - (row['head_start'] if i else 0.0)
            if finish[i] is None and row[key] >= required:
                finish[i] = row['time']
    return dict(attempted=attempts, completed=completed, abandoned=abandoned,
                abort_branch_ticks={b: sum(r.get('abort_branch') == b for r in rows)
                                    for b in ('brake_in_lane', 'checked_rejoin', 'zero_stop')},
                declared_completed=sum(r['status'].startswith('overtake complete:') for r in rows),
                active_at_end=was_active, stuck_s=round(stuck * dt, 3),
                race_time_s=finish[0], opponent_race_time_s=finish[1],
                emergency_stop_ticks=sum(r['safety'] in ('body_contact', 'emergency_obstacle') for r in rows),
                safety_ticks=dict(Counter(r['safety'] for r in rows)),
                car_contact_ticks=sum(bool(r['car_contact']) for r in rows),
                wall_contact_ticks=sum(bool(r['wall_contact']) for r in rows))


def stall_progress(opponent, seed, length):
    """A crashed/stalled car stops for good at one of six evenly spaced spots.

    The spot is the opponent's own race distance at which it is commanded to
    stop: 4 m (past the start) plus seed-indexed sixths of a lap, so a
    six-seed run samples the whole course rather than wherever it got to.
    """
    if opponent != 'stalled':
        return None
    return 4.0 + length * ((seed - 12345) % 6) / 6


def run(track, line, opponent, seed=12345, head_start=3.0, laps=4, timeout=100.0,
        pass_lookahead=None, cap_only=False, pass_planner_enabled=False):
    rv.select_profile('car', 'rb2')
    xy, speeds = racing_math.load_profiled_csv(line)
    plan = rv.PathPlan(np.asarray(xy), racing_math.compute_segment_lengths(xy, closed=True), np.minimum(speeds, 5.0))
    count = 1 if opponent == 'solo' else 2
    plant = rv.make_plant(eval_gap.track_path(track), count, seed,
                          calibration=with_floor('sfu_course_2026_11_06'), car_id='rb2', max_laps=1000)
    params = rv._pure_pursuit_params('rb2')
    if pass_lookahead is not None:
        params['overtake_lookahead_distance'] = pass_lookahead
    followers = [rv.PathFollower(plan, law_name='delay_lqr', car_id='rb2')]
    factor = {'map80': 0.8, 'map90': 0.9, 'blocker': 1.0, 'braking': 0.8, 'gap': 1.0, 'solo': 1.0,
              'stalled': 0.8}[opponent]
    opp_plan = rv.PathPlan(plan.xy, plan.seg_len, np.minimum(plan.speed * factor, 1.5) if opponent == 'blocker' else plan.speed * factor)
    followers.append(rv.PathFollower(opp_plan, law_name='delay_lqr', car_id='rb2'))
    shapers = [rv.CommandShaper('rb2', 5.0, 2.7) for _ in range(2)]
    lidar = rv._lidar()
    gap = rv.GapController(rv.node_params(car_id='rb2'), lidar)
    poses = np.vstack([rv.plan_initial_pose(plan, plant.env.unwrapped.track.raceline)[0]] * 2)
    start, _ = racing_math.find_nearest_index(plan.xy, poses[0, :2], closed=True)
    oi = racing_math.find_lookahead_index(plan.seg_len, start, head_start, closed=True)
    delta = plan.xy[(oi + 1) % len(xy)] - plan.xy[oi]
    poses[1] = [*plan.xy[oi], math.atan2(delta[1], delta[0])]
    obs, _ = plant.reset(poses[:count])
    progress = [Progress(plan, plant.truth(i)[:2]) for i in range(count)]
    actual_head = (progress[1].arc - progress[0].arc) % progress[0].length if count == 2 else 0
    stall = stall_progress(opponent, seed, progress[0].length)
    scan_sim = plant.env.unwrapped.sim.scan_sims[0]

    def wall_at(x, y):
        return plant._distance_transform(float(x), float(y), scan_sim.orig_x, scan_sim.orig_y,
                                        scan_sim.orig_c, scan_sim.orig_s, scan_sim.map_height,
                                        scan_sim.map_width, scan_sim.map_resolution, scan_sim.dt)

    decisions = []
    for i in range(count):
        def expected(x, y, yaw, angles, index=i):
            # FidelityPlant casts from this agent's *estimated* pose, as does the node.
            indices = np.rint((angles - lidar.angle_min) / lidar.angle_increment).astype(int)
            return plant.expected_scan(index)[indices]
        agent_params = dict(params)
        if i == 0 and pass_planner_enabled:
            agent_params.update(enable_local_pass_planner=True, enable_opponent_overtake=False)
        decisions.append(NodeDecisions(plan if i == 0 else opp_plan, lidar, expected, wall_at, agent_params))
    rows = []
    try:
        for step in range(math.ceil(timeout / rv.CONTROL_DT)):
            now = step * rv.CONTROL_DT
            commands, states, statuses, safety_details, targets, desired = [], [], [], [], [], []
            for i in range(count):
                agent, d, follower, shape = obs[f'agent_{i}'], decisions[i], followers[i], shapers[i]
                state = agent['std_state']
                d.observe(state, agent['scan'], now, shape.previous_steering)
                if i == 1 and opponent == 'gap':
                    steering, speed = gap.command(agent['scan'], state[3], step, yaw_rate=state[5])
                    safety, detail = gap.stop_reason or 'none', gap.binding_cap
                    target, base_state = None, None
                else:
                    steering, speed, nearest, target, base_state = d.base_command(
                        follower, state, shape.previous_steering)
                    desired.append(steering)
                    so, vo, safety, detail = d._reactive_override(
                        allow_avoidance=not d.overtake_active and d.follow_speed_cap is None, desired_steering=steering)
                    cap = shape.max_speed
                    if so is not None:
                        steering = so
                    if vo is not None:
                        # Experimental ceiling correction, never a shipped node
                        # setting: preserve override targets for stop/escape/avoid.
                        speed = min(speed, vo) if safety is None else vo
                        cap = min(cap, vo)
                    if base_state == 'off_racing_line':
                        speed, safety = 0.0, base_state
                    if d.follow_speed_cap is not None:
                        speed = min(speed, d.follow_speed_cap)
                        cap = min(cap, d.follow_speed_cap)
                    # Opponent commands a hard stop at 6s, holds until 9s,
                    # then resumes. The real plant supplies its braking/lag.
                    if i == 1 and opponent == 'braking' and 6 <= now < 9:
                        speed, cap = 0.0, 0.0
                    # A crashed/stalled car: stops for good.
                    if i == 1 and stall is not None and progress[1].distance >= stall:
                        speed, cap = 0.0, 0.0
                    steering, speed = shape.command(steering, speed, cap, measured_speed=state[3])
                commands.append((steering, speed))
                states.append(safety or base_state or 'none')
                statuses.append(d.last_opponent_status)
                safety_details.append(detail)
                targets.append(None if target is None else list(map(float, target)))
            obs, _, done, _, info = plant.step(commands)
            p = [progress[i].update(plant.truth(i)[:2]) for i in range(count)]
            car_contact, wall_contact = sorted(plant._chassis_contact()), sorted(plant._wall_contact())
            rows.append(dict(time=(step + 1) * rv.CONTROL_DT, ego_progress=p[0], opp_progress=p[1] if count == 2 else 0,
                             head_start=actual_head, ego_speed=float(plant.truth(0)[3]),
                             opp_speed=float(plant.truth(1)[3]) if count == 2 else 0, ego_command=commands[0][1],
                             opp_command=commands[1][1] if count == 2 else None,
                             steering=commands[0][0], profile_speed=float(plan.speed[progress[0].index]),
                             ego_x=float(plant.truth(0)[0]), ego_y=float(plant.truth(0)[1]),
                             opp_x=float(plant.truth(1)[0]) if count == 2 else None,
                             opp_y=float(plant.truth(1)[1]) if count == 2 else None,
                             active=decisions[0].overtake_active, status=statuses[0], safety=states[0],
                             pass_state=decisions[0]._pass_state, abort_branch=decisions[0]._pass_branch,
                             target=targets[0], desired_steering=desired[0],
                             follow_cap=decisions[0].follow_speed_cap,
                             tracked_arc=decisions[0].opponent.arc_length,
                             tracked_speed=decisions[0].opponent.progress_rate,
                             track_age=decisions[0].opponent.seconds_since_seen(now),
                             opp_status=statuses[1] if count == 2 else None,
                             opp_follow_cap=decisions[1].follow_speed_cap if count == 2 else None,
                             body_clearance=decisions[0]._footprint_clearance(decisions[0].last_scan),
                             arc_clearance=decisions[0]._arc_clearance(decisions[0].last_scan, desired[0]),
                             mapped_forward=decisions[0]._static_closest_in_cone(
                                 decisions[0].last_scan, decisions[0].avoidance_fov_deg),
                             safety_detail=safety_details[0], car_contact=car_contact, wall_contact=wall_contact))
            if car_contact or wall_contact or (p[0] >= laps * progress[0].length
                                               and (count == 1 or stall is not None
                                                    or p[1] + actual_head >= laps * progress[0].length)):
                break
            if done:
                raise RuntimeError('gym ended before independent race verdict')
    finally:
        plant.close()
    result = race_metrics(rows, rv.CONTROL_DT, progress[0].length, laps)
    result.update(track=track, opponent=opponent, seed=seed, head_start_m=actual_head, laps=laps,
                  timeout_s=timeout, elapsed_s=rows[-1]['time'],
                  experiments=dict(pass_lookahead_m=pass_lookahead, cap_only=cap_only,
                                   local_pass_planner=pass_planner_enabled),
                  ego_laps=rows[-1]['ego_progress'] / progress[0].length,
                  opponent_laps=rows[-1]['opp_progress'] / progress[0].length if count == 2 else None,
                  node_sha256=hashlib.sha256(NODE.read_bytes()).hexdigest(),
                  following_sha256=hashlib.sha256(Path(following.__file__).read_bytes()).hexdigest(),
                  completion_sha256=hashlib.sha256(Path(pass_completion.__file__).read_bytes()).hexdigest(),
                  pass_planner_sha256=hashlib.sha256(Path(pass_planner.__file__).read_bytes()).hexdigest(),
                  controller_parameters=params,
                  harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  setup='standing start; frozen map and line settled before LB; both cars rb2',
                  line_sha256=hashlib.sha256(Path(line).read_bytes()).hexdigest(),
                  plant=plant.report())
    return result, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--track', required=True)
    ap.add_argument('--line', required=True)
    ap.add_argument('--opponent', choices=('map80', 'map90', 'gap', 'blocker', 'braking', 'solo', 'stalled'), required=True)
    ap.add_argument('--seeds', default='12345,12346,12347,12348,12349,12350')
    ap.add_argument('--head-start', type=float, default=3.0)
    ap.add_argument('--laps', type=int, default=4)
    ap.add_argument('--timeout', type=float, default=100.0)
    ap.add_argument('--pass-lookahead', type=float,
                    help='EXPERIMENT: override passing preview for both map cars, in metres')
    ap.add_argument('--cap-only', action='store_true',
                    help='EXPERIMENT: mapped pass cap can only lower the profile target')
    ap.add_argument('--pass-planner', action='store_true',
                    help='stage 2: ego passes stopped cars (enable_local_pass_planner)')
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if not math.isfinite(args.head_start) or not 2 <= args.head_start <= 4 or args.laps < 1 or not math.isfinite(args.timeout) or args.timeout <= 0:
        ap.error('need head-start in [2,4] m, positive laps and finite positive timeout')
    if args.pass_lookahead is not None and (not math.isfinite(args.pass_lookahead) or args.pass_lookahead <= 0):
        ap.error('pass-lookahead must be positive and finite')
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in map(int, args.seeds.split(',')):
        result, rows = run(args.track, args.line, args.opponent, seed, args.head_start, args.laps, args.timeout,
                           args.pass_lookahead, args.cap_only, args.pass_planner)
        with (args.output / f'{seed}.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
        results.append(result)
        (args.output / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
        print(json.dumps(result), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
