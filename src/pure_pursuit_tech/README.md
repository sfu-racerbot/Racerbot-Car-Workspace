# Pure Pursuit (tech)

A C++ pure pursuit waypoint follower, laid out like racerbot_a's
[`pure_pursuit_a`](https://github.com/sfu-racerbot/racerbot_a/tree/main/src/pure_pursuit_a)
and using the same control law, except for the goal point: the point
where the line crosses the lookahead circle, interpolated between waypoints,
instead of the first waypoint at least the lookahead distance away. The goal
then slides along the line instead of jumping from waypoint to waypoint
(`k = 2y/d²`, `steering = atan(wheelbase · k)`). It slows down linearly from `max_speed`
on straights to `min_speed` at full steering lock, and publishes
`AckermannDriveStamped` on `/drive`.

It only drives while **both** of these are true; otherwise it publishes speed 0:

- **LB is held** on the F710 (`/joy` button 4, heard within the last
  `joy_timeout_sec`). The node checks this itself, as every driving node in
  this workspace must (`docs/architecture.md`). It cannot be turned off.
- **racerbot_a's [safety node](../racerbot_a/src/safety_node/README.md)** is
  not braking. `/safety/brake` must be `false` and heard within the last
  `safety_timeout_sec`. `enable_safety:=false` drops only this check.

## Building
```bash
cd ~/racerbot-ws
colcon build --packages-select pure_pursuit_tech safety_node
```

## Launching in real life

```bash
ros2 launch f1tenth_stack bringup_launch.py         # terminal 1: drivers + mux
ros2 launch particle_filter localize_launch.py      # terminal 2: /pf/pose/odom on the map
ros2 launch pure_pursuit_tech pure_pursuit_tech_launch.py track:=mapname
```

`track:=mapname` reads `waypoints/mapname.csv`. For a track mapped on this
car, point at the run's line instead:
`waypoints_file:=~/racerbot-ws/data/<car>/racerbot_auto/<time>/raceline_profiled.csv`.
`track:=latest` takes the newest saved line instead: a line drawn in the
simulator (`data/<car>/racerbot_sim/drawn_lines/`) or a mapping run
(`data/<car>/racerbot_auto/`). The launch prints which one it chose. Add
`map_yaml:=<the map localization runs on>` and it refuses a line saved on a
different map. Without it, nothing checks that the two maps match.

Don't run `teleop_launch.py` at the same time: its `/teleop` always outranks
`/drive` in `ackermann_mux`.

The launch file also starts racerbot_a's safety node. It remaps the node's
odometry input (the simulator's `/ego_racecar/odom`) to `/odom`, because on
the car its time-to-collision brake would otherwise read speed 0 and never
fire. Use `launch_safety:=false` if you run it yourself.

The wheelbase and the steering limit come from this car's calibration
(`src/racerbot_car/cars/<car>/calibration/pure_pursuit.yaml`): 0.33 rad on
rb1 and 0.349 rad on rb2. A higher `max_steering:=` is refused.

## Launching in sim

In the RViz simulator (`docs/ros-simulator.md`), pick **"Map, then race
(gap_follow, then pure_pursuit_tech C++)"** in the Racerbot panel. gap_follow
maps the track and saves a racing line. The car is then put back on the start
line, and this node races that line on the particle filter, with the safety
node, as on the car. Its driven line is red.

To race your own line, use the panel's **Draw your own racing line** box. Press
**Draw**, click dots around the track with RViz's **Publish Point** tool, then
press **Race with pure_pursuit_tech**. The line is saved under
`data/<car>/racerbot_sim/drawn_lines/<time>/`, with a `commands.txt` that races
it again. There is nothing to copy into `waypoints/`.

## Safety Node
The safety node needs to be running, or this node will not drive. The launch
file starts it by default. See the
[Safety Node Readme](../racerbot_a/src/safety_node/README.md) for details.

Known issue: the safety node gives up on `/joy` after 50 ms
(`safety_message_timeout_ms`), but the car's `joy_node` repeats every 50 ms
(20 Hz). Its brake may flicker while LB is held, so check this with the wheels
off the ground first.

## Parameters

Run `ros2 launch pure_pursuit_tech pure_pursuit_tech_launch.py --show-args`
for all options.

- `track` or `waypoints_file`: which line to follow.
- `use_pf`: `true` takes the pose from `/pf/pose/odom`; `false` takes it from `odom_topic`.
- `min_speed`, `max_speed` (m/s): default 0.5 and 1.0, for first floor runs.
- `medium_angle_speed` (m/s): the steering rule's speed at half of
  `max_steering`. The rule runs in straight lines from `max_speed` (wheels
  straight) through this to `min_speed` (full lock), so you can bend it: hold
  speed through gentle steering and drop late, or the reverse. Default -1 =
  halfway between the two, the old straight line. Must lie between
  `min_speed` and `max_speed`.
- `lookahead_base`, `lookahead_min`, `lookahead_max` (m) and
  `lookahead_speed_gain` (extra metres of lookahead per m/s).
- `max_steering` (rad).
- `speed_mode`: `steering` (default) uses `pure_pursuit_a`'s rule, `max_speed`
  when straight, `medium_angle_speed` at half lock and `min_speed` at full
  steering lock. `profile` plans the
  speed along the whole line instead:
  - corners: `speed ≤ √(max_lateral_accel ÷ curvature)`. This wins over
    `min_speed`: a corner that needs 0.4 m/s gets 0.4 m/s even with
    `min_speed` 0.5 (`min_speed` only sets the steering rule's low end);
  - braking before each corner at `max_brake`, and speeding up after it at
    `max_accel`. Cornering shares the same grip, so while the car is also
    turning it gets only what is left of each (friction ellipse:
    `available = max × √(1 − (lateral ÷ max_lateral_accel)²)`, all of it on
    a straight, none at a corner's speed limit). The same model as
    `pure_pursuit`'s planner (`racing_math.compute_velocity_profile`);
  - the command rises no faster than `max_accel`. Slowing down is never delayed.

  The steering rule stays on as a cap, for when the car is off the line.
  The default limits are lateral 2.5, accel 3.0 and brake 3.0 m/s², the same
  as this workspace's planner (`src/pure_pursuit/config/auto_map_race.yaml`).
  A higher `max_lateral_accel` takes corners faster.
- `laps` (loops only): stop at the start/finish line after this many laps.
  The default, 0, never stops.
- `goal_tolerance` (m, default 0.3): stop this close to the goal, which is the
  end of an open path or the end of the last lap. The car slows to it
  (`v ≤ √(2 × max_brake × distance left)`). It then stays stopped until you
  release LB and press it again.
- `max_path_error` (m, default 1.5): stop if the car is further than this from
  the line. Usually localization is lost, or the line was saved on another map.
- `loop_closure_distance` (m, default 0.5, node parameter): a path whose last
  point is this close to its first is a loop; otherwise it is open. An open
  path never wraps back to its start. Near its end, the car aims along the
  last segment.

The particle filter reports the LiDAR's pose, which is 0.26 m ahead of the
rear axle. With `use_pf:=true`, the launch file passes the car's
`laser_offset_x/y` (from `pure_pursuit.yaml`), and the node shifts the pose
back to the rear axle, where pure pursuit's geometry is measured from.

## Code layout

| File | What it does |
|---|---|
| `src/pure_pursuit_tech_node.cpp` | ROS glue: settings, the LB and `/safety/brake` gates, topics |
| `src/path.cpp` (`include/path.hpp`) | the line: CSV parsing, loop/open, lengths, curvature, speed profile, nearest waypoint |
| `src/tracker.cpp` (`include/tracker.hpp`) | where the car is on the line: progress, laps, distance to the goal |
| `src/control.cpp` (`include/control.hpp`) | the pure pursuit law: rear axle, lookahead, goal point, steering, speed rules |

Only the node uses ROS, so the other three are tested on their own.

## Tests

```bash
source /opt/ros/jazzy/setup.bash && source install/setup.bash
export ROS_DOMAIN_ID=79 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
colcon test --packages-select pure_pursuit_tech && colcon test-result --verbose
```

- `test/test_core.cpp` (gtest, 31 tests): the ROS-free core.
- `test/test_pure_pursuit_tech_node.py` (pytest): the built node on
  `/test_only/*` topics. It covers the four LB deny paths, the safety brake,
  the control law, the laser offset, profile mode's ramp, the goal and lap
  stops, and the off-path stop.
- `test/test_pure_pursuit_tech_launch.py` (pytest): the launch file's steering
  cap, its `safety_node` remap and the settings it passes to the node.
