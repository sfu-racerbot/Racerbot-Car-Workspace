# NLOS opponent detection with a first-return 2D LiDAR: implementation plan

> **Who this is for:** coding agents and team members who will build or run the non-line-of-sight (NLOS) experiment. It is a work plan, not a tutorial.
> **Read first:** [CLAUDE.md](../../../CLAUDE.md), [TEST_QUALITY_STANDARDS.md](../../../TEST_QUALITY_STANDARDS.md) and the [glossary](../../glossary.md).
> **You'll be able to:** pick up one work package or one experiment and carry it out without further design decisions.
> **Time:** about 40 minutes for sections 0 to 5; each work package is then self-contained.

**Date:** 2026-11-20. **Repo:** `sfu-racerbot/Racerbot-Car-Workspace` at `7b2003f`. **Status:** proposed, not yet approved by the team lead.

**Goal.** A race-legal Hokuyo 2D LiDAR detects and tracks a car hidden around a corner, in real time on the Jetson, by reading the mirror image ("ghost") that a specular panel on the track wall puts into the scan. The paper reports how much earlier than line-of-sight the hidden car is seen, and for which materials and geometries that works.

**Architecture in one paragraph.** A new, non-driving package `src/nlos_perception/` reads `/scan` and the localization pose. For each registered panel it (1) predicts where the panel is in the scan, (2) re-fits the panel line from the scan itself, (3) flags beams that came back longer than the panel, (4) reflects those beams about the fitted line to get real positions, (5) removes the static hidden scene with a per-panel background grid, (6) clusters and tracks what is left in the panel's own frame, and (7) publishes JSON on `/perception/nlos_opponents`. Everything except the ROS adapter is NumPy and runs without ROS.

**Who executes what.** Work packages (WP) are for coding agents, one WP per session and per pull request. Experiments (E) are for humans; agents only build the tools and analyse the recordings.

---

## 0. Rules for whoever executes a work package

Read these before any WP. They are not optional.

1. **Read first:** `CLAUDE.md`, `AGENTS.md`, `TEST_QUALITY_STANDARDS.md`, and sections 2, 4 and 5 of this plan. For docs, follow `.claude/skills/novice-docs/SKILL.md`.
2. **The repo wins.** If a path, name or behaviour in this plan does not match the checkout, stop and report the mismatch. Do not guess and do not "fix" the repo to match the plan.
3. **Never publish a command.** Nothing in `nlos_perception` may create a publisher on `/drive`, `/teleop`, `/ackermann_cmd` or any `AckermannDriveStamped` topic. WP8 has a test for this.
4. **Do not edit** vendored or submodule trees: `src/f1tenth_system`, `src/particle_filter`, `src/range_libc`, `src/transport_drivers`, `src/realsense-ros`, `src/web_dashboards`, `src/racerbot_a`, `src/racerbot_b`.
5. **Do not edit** `gap_follow` or `pure_pursuit` except in WP14, and there only behind a parameter that defaults to off with an "off is identical" test.
6. **Values tagged `[MEASURE]` are unknown.** Keep them as parameters with the stated default. Never write them into code, docs or the paper as facts until the named experiment has measured them.
7. **Tests.** pytest, files named `test_*.py` in `src/nlos_perception/test/`. No `pytest.ini`, no `[tool:pytest]` in `setup.cfg` (the repo forbids it; register markers in `test/conftest.py` as `src/opponent_perception/test/conftest.py` does). ROS tests carry the `ros` marker and run on `ROS_DOMAIN_ID=79 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST`.
8. **Every test must be able to fail.** For each function, do the stub swap from `TEST_QUALITY_STANDARDS.md` section 0. Add one entry per function to `tools/mutations/nlos_perception.json` and run `python3 tools/check_mutations.py tools/mutations/nlos_perception.json`. A WP is not done until every mutation is caught.
9. **No new third-party dependency** beyond NumPy, SciPy, PyYAML (already used in this workspace). Bag reading uses `mcap` and `mcap-ros2-support`, which exist only in `$HOME/.local/racerbot-analysis-venv`; code that imports them lives in `tools/nlos/`, never in `src/nlos_perception/`.
10. **Determinism.** Every random choice takes a `numpy.random.Generator` argument. Tests pass a fixed seed.
11. **Stop conditions.** Stop and report, without workarounds, if: a test in another package starts failing; a guard test (`src/racerbot_launch/test/test_car_profiles.py`, `src/racerbot_car/test/test_no_package_pytest_config.py`) fails; or an acceptance number in this plan cannot be met.
12. **One WP, one PR.** PR description lists the packages touched, the safety impact ("none: no command topic"), and the exact test commands and results.

Test commands, from the workspace root:

```bash
# ROS-free tests (must pass with ROS not sourced)
python3 -m pytest src/nlos_perception/test -m 'not ros' -q
# ROS tests
source /opt/ros/jazzy/setup.bash && source install/setup.bash
export ROS_DOMAIN_ID=79 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
python3 -m pytest src/nlos_perception/test -v
# offline tools (bag reading)
"$HOME/.local/racerbot-analysis-venv/bin/python" -m pytest tools/nlos/tests -q
```

---

## 1. Scope

**In scope.** Flat specular panels that the team mounts on its own track; one or more panels per track; one hidden car per panel at a time; the stopped-car-beyond-a-blind-corner scenario and the car-ahead-disappears scenario; simulation, bench, static and on-car experiments.

**Out of scope.** Diffuse-wall NLOS (the sensor cannot measure it). Curved relays. Any change to localization or SLAM code. Emergency braking from NLOS output.

**The honest limit for race day.** Race barriers are the organisers' 33 cm air ducts. Experiment E0 measures whether a duct gives any ghost. If it does not, the paper states that the method needs a flat specular surface and reports the duct as a negative result.

---

## 2. Facts, with where each came from

Do not re-derive these. Do not contradict them without new evidence.

### 2.1 Sensor

| Fact | Value | Source |
|---|---|---|
| Car 2 LiDAR | Hokuyo UST-10LX, Ethernet, `192.168.0.10:10940` | `CLAUDE.md`, `src/racerbot_car/cars/rb2/calibration/sensors.yaml` |
| Car 1 LiDAR | Repo says "UST-30LX over USB" with a TODO to verify. USB id `15d1:0000` and a USB link point to a UTM-30LX, since the UST-30LX is Ethernet. `[MEASURE U8]` read the label | `src/racerbot_car/cars/rb1/car.yaml`, Hokuyo spec sheets |
| Wavelength, class | 905 nm, class 1 | UST-10LX specification |
| Beams | 1081 beams, -135° to +135°, 0.25° (0.004363 rad) | bag `data/rb2/racerbot_runs/20261030-184010` |
| Timing | `scan_time` 0.025 s, `time_increment` 17.36 µs (25 ms / 1440), sweep of 1081 beams takes 18.75 ms | same bag |
| Measured rate | 38.3 Hz median, 5 ms arrival jitter | same bag |
| `range_min`, `range_max` in the message | 0.02 m, 30.0 m | same bag |
| No-return code | The driver reports **65.533 m**, not `inf`. Spec: 65533 = "object does not exist or low reflectivity". Treat `r >= range_max`, `r < range_min` and non-finite as no-return | same bag, UST-10LX specification |
| Guaranteed range | 10 m on white paper, 4 m at 10 % diffuse reflectance | UST-10LX specification |
| Accuracy (spec) | ±40 mm, repeatability σ < 30 mm | UST-10LX specification |
| Range noise (measured, car 2, stationary, 849 scans) | median per-beam σ: 3 mm at 0-2 m, 7 mm at 2-4 m, about 10 mm at 4-7 m with outliers | same bag, one scene only |
| Beam spread | about 250 × 40 mm at 10 m for the standard UST-10LX. Which axis lies in the scan plane, and whether it rotates with scan angle, is `[MEASURE U2, U3]` | Hokuyo UST-10LXB-H02 product page |
| Multi-echo | UST-10LX: not supported (SCIP spec). UTM-30LX: not supported (only the -EW model). `[MEASURE U7]` on car 1 | Hokuyo protocol spec, distributor tables |
| Intensity | Available on the UST-10LX; currently off (`publish_intensity: false`), so `intensities` is empty in every existing bag | `src/f1tenth_system/f1tenth_stack/config/sensors.yaml`, bag |
| Mount | 0.26 m ahead of `base_link` (rear axle), 0.11 m above it | `docs/hardware-reference.md` |

### 2.2 Repo

| Fact | Where |
|---|---|
| New packages go under `src/`, one per feature, `ament_python`, layout as in `CLAUDE.md` "Package anatomy" | `CLAUDE.md` |
| Closest template: `src/opponent_perception/` (non-driving, JSON on `std_msgs/String`, `pipeline.py` shared by offline replay and ROS adapter, `ros` test marker) | that package |
| Localization pose: `/pf/viz/inferred_pose`, `geometry_msgs/PoseStamped`, frame `map`, **pose of the laser**, not `base_link`. `/slam_pose` is a `base_link` pose | `src/pure_pursuit/config/pure_pursuit.yaml` lines 22-30 |
| Localization can be wrong by 0.8-2.2 m along a straight (car 2 run `20261106-175300`), which produced a phantom opponent in map subtraction | `src/pure_pursuit/pure_pursuit/opponent_gate.py` docstring |
| Map-predicted ranges already exist: `pure_pursuit.map_subtraction.MapRayCaster` (needs `range_libc`). This plan does **not** use it, so the package stays NumPy-only | that file |
| The simulator's scan comes from `SimBridge.scan(agent)`; occupancy lookup is `SimBridge.occupied(xs, ys)`; scan dropout is reported as `inf`; sim noise σ is 0.01 m | `src/racerbot_sim/racerbot_sim/sim_bridge.py` |
| Simulator ground truth: `/sim/ground_truth_pose`, `/sim/car2/ground_truth_pose`, `/sim/opponent_poses` | `docs/ros-simulator.md`, `gym_bridge_node.py` |
| `bringup_launch.py` takes `sensors_config:=<file>`; that file replaces the shared `sensors.yaml` and the per-car calibration overlay is applied on top | `src/f1tenth_system/f1tenth_stack/launch/bringup_launch.py`, `src/racerbot_car/racerbot_car/layers.py` |
| Runs are bagged as MCAP under `data/<car>/racerbot_runs/<timestamp>/bag/`. Default topics include `/scan`, `/odom`, `/tf`, `/pf/viz/inferred_pose`; they do not include `/perception/*` | `src/race_diagnostics/race_diagnostics/recording.py` |
| A file over 95 MiB cannot be committed | `data/README.md` |
| Every launch file is listed in `tools/dev/launch_matrix.yaml` | that file |
| Car body model used everywhere: 0.58 m long, 0.31 m wide (padded) | `sim_bridge.py`, `gap_follow` |

### 2.3 Rules of the competition

| Fact | Source |
|---|---|
| Allowed LiDAR: Hokuyo UST-30LX, equivalent, or lower specification | RoboRacer rules, 2025-02-03 |
| Track borders: air ducts of 33 cm diameter, with gaps the beams can pass through | same |
| Track at least 1.2 m wide (4 car widths) | same |
| A car must fill at least 12 × 12 cm at every height between 10 and 30 cm | same |

### 2.4 Physics numbers (computed, checked numerically on 2026-11-20)

- Specular reflectance of a rough surface: $\rho_s(\alpha)=F(\alpha)\exp\!\left[-\left(4\pi\sigma_h\cos\alpha/\lambda\right)^2\right]$, with $\alpha$ the incidence angle from the normal, $\sigma_h$ the RMS roughness, $F$ the Fresnel reflectance.
- At 905 nm the exponent is 1 when $\sigma_h\cos\alpha = 72$ nm. Roughness factor: 100 nm gives 0.145 at 0°, 0.38 at 45°, 0.88 at 75°; 200 nm gives 0.0004, 0.02, 0.60.
- Fresnel reflectance of clear acrylic (n = 1.49, unpolarised): 0.039 at 0°, 0.049 at 45°, 0.088 at 60°, 0.25 at 75°, 0.39 at 80°. A metal mirror is about 0.9 at all angles. The ghost power goes with $\rho_s^2$ because the light meets the panel twice.
- Beam spacing is 4.36 mm per metre of path. At 4 m path a 0.28 m wide target spans 16 beams; at 6 m, 11 beams.
- Line-fit angle error for $N$ points with perpendicular noise $\sigma_d$ spread evenly over span $S$: $\sigma_\phi\approx\sigma_d\sqrt{12}/(S\sqrt N)$. With 5 mm, 0.8 m, 9 points: 0.0072 rad. Position error across the reflected ray is $2\ell\,\sigma_\phi$: 3 cm at 2 m, 9 cm at 6 m.
- A duct of radius 0.165 m turns its normal 0.061 rad per cm of hit error, which is 0.24 m of error 2 m past the bounce.
- Within a panel sector of 80 beams the sweep takes 1.4 ms; at 5 m/s the car moves 7 mm. Motion inside one sector is ignored.

### 2.5 Prototype results (simulation only)

Scene: L-corner, both corridors 1.5 m wide, 0.80 m chamfer panel (pattern P1) across the outer corner, perfect mirror, range noise 5 mm, pose prior error 0.15 m and 3°, ego on the centre line at 4 m/s, hidden car 0.50 × 0.28 m. Reference code and raw outputs: `tools/nlos/reference/` (committed in WP0). "Footprint" is the beam width in the scan plane per metre of path; a beam whose footprint touches matte is assumed to return the matte.

| Quantity | Thin beam, no path limit | Footprint 25 mm/m, no path limit | Footprint 12 mm/m, ghost path ≤ 6 m (the defaults) |
|---|---|---|---|
| Hidden positions detected, ego 2-5 m from the panel | 56-68 % | 26-54 % at 2-4 m, 0 % beyond 5 m | not run |
| Same positions visible by line of sight | 3-12 % | 3-12 % | not run |
| Surface error of unfolded points (RMS) | 2.3-2.9 cm | 1.3-2.2 cm | not run |
| Distance from cluster centroid to car centre | 0.21-0.26 m | 0.23-0.26 m | not run |
| Stopped car 1-2 m past the corner: ego travel gained over line of sight | 4.9-5.9 m | 1.5-2.6 m | 1.2-1.9 m (0.30-0.47 s at 4 m/s) |
| Stopped car 0.5 m past the corner | 3.4-4.0 m | -0.5 to 1.1 m | -0.1 to 0.9 m |
| Stopped car 2-3 m past the corner, middle or inner lane | seen 5.5-6.5 m before the corner | seen 1.6-2.7 m before | seen 0.4-1.4 m before; never seen by line of sight before the turn |

Panel line fit (footprint 12 mm/m, 150 random poses per distance bin): angle error median 0.003 rad, 95th percentile below 0.010 rad; error of the `p0` end along the panel median 0.5-1.4 cm, 95th percentile below 3.2 cm; fit failures 0-3 % up to 5.5 m, 7 % at 5.5-6.5 m.

False alarms on four empty approaches (316 scans, defaults): with a background learned only up to 0.2 m before the corner and on 3 lateral lines, 35 scans had a foreground cluster and 28 were in the 4-of-6 confirmed state, almost all from ego positions the background pass never visited. With the background learned over the whole approach on 7 lateral lines: 0 scans with a cluster. This is why 4.4 has a coverage rule.

Read this as: geometry and tracking are not the limit. The return strength (how long a ghost path the sensor still reports) and the beam footprint are. E0 measures both. The thin-beam column includes detections at 8-10 m total path, which a real sensor is unlikely to return from a dark car.

---

## 3. Unknowns and the experiment that settles each

| Id | Unknown | Settled by | Default until then |
|---|---|---|---|
| U1 | Probability that a mirror beam returns the ghost, as a function of material, incidence angle, path length, target reflectance | E0-A | parameter table, all 1.0 inside `max_ghost_path_m = 6.0` |
| U2 | Beam footprint in the scan plane, per metre of path | E0-B | `footprint_k = 0.012` |
| U3 | Whether the footprint changes with scan angle | E0-B | no |
| U4 | Which echo is reported when a beam covers both matte and mirror | E0-B | the matte (nearer) one |
| U5 | Range bias of ghost returns | E0-A | 0 |
| U6 | Whether `header.stamp` is the time of the first beam | WP8 step 1 (read the driver source) | first beam |
| U7 | Multi-echo on car 1 | E0-D | unavailable |
| U8 | Car 1's exact LiDAR model | read the label | UTM-30LX |
| U9 | Whether a race duct gives any ghost | E0-C | no |
| U10 | Whether gap-follow steers at a mirror window | WP9 (simulation), then E1 | unsafe until WP9 passes |
| U11 | Cost of uncovered panels to particle-filter localization | E2 | unknown |

---

## 4. Design

### 4.1 The panel (physical)

One standard panel, so the software has one thing to recognise:

- Board 0.80 m long, 0.25 m tall, rigid and flat (flatness matters more than anything else: a 1 mm bow over 0.8 m is 0.005 rad).
- Mounted vertical, its centre line at the LiDAR's height above the floor `[MEASURE on each car with a tape: floor to the middle of the Hokuyo window]`.
- Surface, along its length: **5 cm matte, 20 cm mirror, 5 cm matte, 20 cm mirror, 5 cm matte, 20 cm mirror, 5 cm matte**. Call this pattern P1.
- Mirror: the best material from E0-A (start with 3 mm acrylic mirror sheet). Matte: white paper or matte white tape.
- A matte cover (cardboard) that hides the whole panel. It is on during mapping laps and off otherwise.

Why stripes: (a) the matte returns give the panel line in the scan itself, so unfolding does not depend on localization; (b) a 20 cm window is narrower than a car (0.30 m), so a gap-seeking controller should not read it as an opening `[U10]`; (c) the particle filter still sees a wall there.

Placement: across the **outside** of a corner at 45° to both legs (a chamfer), drivable side facing the inside of the corner. A flat panel parallel to a wall is nearly useless: its reflection only reaches into the hidden leg when the ego is already at the corner. WP12 quantifies alternatives.

### 4.2 Frames

All 2D, metres, radians, right-handed, angles counter-clockwise.

- **map**: the SLAM map frame.
- **laser**: origin at the LiDAR, x forward. Beam $i$ has direction $\mathbf u_i=(\cos\theta_i,\sin\theta_i)$ with $\theta_i=\texttt{angle\_min}+i\cdot\texttt{angle\_increment}$.
- **panel**: origin at panel endpoint `p0`, x axis $\mathbf t=(\mathbf p_1-\mathbf p_0)/\lVert\mathbf p_1-\mathbf p_0\rVert$, y axis $\mathbf n=(-t_y,\ t_x)$. **Convention: the drivable side is on the left when walking from `p0` to `p1`**, so real objects have panel-frame $y>0$ and raw ghost points have $y<0$. The panel frame is fixed to the world, so tracking in it is immune to localization error.

### 4.3 Per-scan algorithm

Input: `ranges[N]`, `angle_min`, `angle_increment`, `range_min`, `range_max`, `time_increment`, scan stamp, latest laser pose in map (may be stale or wrong), panel registry.

For each panel:

1. **Predict.** Transform `p0`, `p1` into the laser frame with the pose. Skip the panel if its centre is farther than `max_panel_range_m`, if the laser is on the wrong side ($\mathbf n\cdot(\mathbf s-\mathbf p_0)\le 0$), or if $|\cos\alpha|$ at the centre is below `min_incidence_cos`.
2. **Select beams** whose angle lies within the panel's angular extent widened by `angle_margin_rad` on each side.
3. **Fit the panel line in the laser frame.** Among selected valid points within `gate_perp_m` of the predicted line and within `gate_perp_m` of its ends, run RANSAC for a line whose direction is within `ransac_max_angle_rad` of the predicted direction; inlier tolerance `ransac_tol_m`. Refit by total least squares on the inliers. Reject the fit if inliers < `min_inliers` or their span < `min_span_fraction` × panel length. On rejection publish the panel as `visible: false` and stop (do **not** fall back to the map line unless `allow_map_line` is true).
4. **Classify beams** that intersect the fitted segment. With $\hat r_i$ the intersection range and $e_i=r_i-\hat r_i$: matte if $|e_i|\le$ `matte_tol_m`; **ghost** if $e_i>$ `ghost_min_excess_m` and the beam is valid; front object if $e_i<-$`matte_tol_m`; no-return otherwise.
5. **Unfold** every ghost beam: $\mathbf p_i=\mathbf w_i+e_i\,\mathbf u_i'$ with $\mathbf w_i=\hat r_i\mathbf u_i$ and $\mathbf u_i'=\mathbf u_i-2(\mathbf u_i\cdot\mathbf n)\mathbf n$. Convert to the panel frame.
6. **Reject** points within `self_ghost_radius_m` of the laser origin (the ego's own image) and points with panel-frame $y\le 0$.
7. **Background.** Drop points that fall in a static cell of the panel's background grid (4.4).
8. **Cluster** the rest (single linkage, `cluster_eps_m`); keep clusters with at least `cluster_min_beams` points and extent at most `cluster_max_extent_m`.
9. **Track** cluster centroids in the panel frame with a constant-velocity Kalman filter; confirm on `confirm_m` of the last `confirm_n` scans.
10. **Publish** (4.6). Map-frame positions use the pose and are marked stale if the pose is older than `pose_max_age_sec`.

Timestamp of a panel observation: scan stamp + (middle selected beam index) × `time_increment`.

### 4.4 Background grid (per panel)

A 2D grid in the panel frame, cell `bg_cell_m`, covering $x\in[-4,4)$ m and $y\in[-0.5,7.5)$ m. During solo laps with the panel uncovered and no other car on track, every unfolded ghost point adds one hit to its cell. Afterwards a cell is static if hits ≥ `bg_min_hits`, then the static mask is dilated by `bg_dilate_cells`. Saved as `.npz` next to the panel registry. A point outside the grid counts as static.

**Coverage rule.** Static ghosts that the background pass never saw look like a hidden car. So the background also stores where the laser was while learning: a second grid over the same area with 0.25 m cells, marking every cell the laser occupied (panel frame), dilated by one cell when finalised. At run time, if the laser's panel-frame position is not in a marked cell, the panel produces no tracks and reports `"background": "uncovered"`. Learning laps must therefore cover every line the car may drive, including lines 0.3 m either side of the racing line, all the way through the corner.

### 4.5 Files

```
src/nlos_perception/
  package.xml  setup.py  setup.cfg  resource/nlos_perception
  nlos_perception/
    __init__.py
    geometry.py        WP1   pure math
    panels.py          WP2   Panel, PanelRegistry, YAML
    forward_model.py   WP3   2D specular scan simulator
    locator.py         WP4   panel line from the scan
    ghosts.py          WP5   classify + unfold
    background.py      WP5   per-panel static grid
    clustering.py      WP6
    tracking.py        WP6
    pipeline.py        WP7   one class used offline and by the node
    offline.py         WP7   synthetic demo CLI (no bag reading here)
    nlos_node.py       WP8   ROS adapter, the only file that imports rclpy
  config/nlos_perception.yaml  config/panels.example.yaml  config/sensors_intensity.yaml
  launch/nlos_perception_launch.py
  test/conftest.py  test/test_*.py
tools/nlos/
  reference/nlos_proto.py      WP0   verified prototype, read-only reference
  bag_io.py                    WP10  MCAP to arrays
  bench_analyze.py             WP10  E0
  register_panels.py           WP11
  learn_background.py          WP11
  eval_static.py eval_dynamic.py make_figures.py   WP11
  sweep_sim.py slurm_sweep.sh  WP12
  tests/
tools/mutations/nlos_perception.json
docs/nlos-perception.md  docs/nlos-experiments.md   WP15
```

### 4.6 Interfaces

**Subscriptions** (node): `scan_topic` (`sensor_msgs/LaserScan`, default `/scan`), `pose_topic` (`geometry_msgs/PoseStamped`, default `/pf/viz/inferred_pose`).

**Publications**: `/perception/nlos_opponents`, `/perception/nlos_status` (both `std_msgs/String`, JSON, `allow_nan=False`), `/perception/nlos_markers` (`visualization_msgs/MarkerArray`).

`/perception/nlos_opponents`, one message per scan:

```json
{
  "schema_version": 1,
  "stamp_sec": 1763692810.125,
  "pose_age_sec": 0.031,
  "pose_stale": false,
  "panels": [
    {"id": "p1", "visible": true, "background": "ok", "fit_source": "scan", "n_inliers": 14,
     "sigma_phi_rad": 0.006, "p0_laser_m": [3.91, -0.62], "p1_laser_m": [4.47, -0.05],
     "n_ghost_beams": 21, "n_foreground_beams": 9}
  ],
  "tracks": [
    {"track_id": 3, "panel_id": "p1", "confirmed": true, "hits": 7, "age_sec": 0.0,
     "position_panel_m": [0.31, 2.44], "velocity_panel_mps": [0.0, -1.2],
     "position_laser_m": [4.9, 1.8], "position_map_m": [12.3, 4.1],
     "position_sigma_m": [0.03, 0.06], "n_beams": 9, "path_length_m": 6.7}
  ]
}
```

`position_*` is the centroid of the **visible surface**, not the car's centre. `position_map_m` is `null` when `pose_stale` is true. `position_sigma_m` is (along the reflected ray, across it) = (`range_sigma_m`, $2\ell\,\sigma_\phi$).

**Panel registry** (`panels.yaml`):

```yaml
schema_version: 1
frame: map
panels:
  - id: p1
    p0: [12.40, 3.10]        # metres, map frame; drivable side on the LEFT walking p0 -> p1
    p1: [12.97, 3.67]
    pattern: [[0.05, matte], [0.20, mirror], [0.05, matte], [0.20, mirror], [0.05, matte], [0.20, mirror], [0.05, matte]]
    material: acrylic_mirror_3mm
    background: p1_background.npz   # optional, relative to this file
```

Validation: unique ids; finite coordinates; length within 0.01 m of the pattern sum; pattern kinds only `matte` or `mirror`; first and last element `matte`.

### 4.7 Parameters (defaults; all in `config/nlos_perception.yaml`)

| Name | Default | Meaning |
|---|---|---|
| `scan_topic` | `/scan` | |
| `pose_topic` | `/pf/viz/inferred_pose` | |
| `pose_frame` | `laser` | `laser` or `base_link`; for `base_link` add `laser_offset_x` along the heading |
| `laser_offset_x` | 0.26 | m |
| `panels_file` | `''` | empty: node starts, publishes `"panels": []`, logs once |
| `pose_max_age_sec` | 0.2 | |
| `max_panel_range_m` | 8.0 | |
| `min_incidence_cos` | 0.17 | about 80° from the normal |
| `angle_margin_rad` | 0.15 | covers pose error |
| `gate_perp_m` | 0.5 | |
| `ransac_iters` | 200 | fixed seed per node, reseeded never |
| `ransac_tol_m` | 0.02 | |
| `ransac_max_angle_rad` | 0.35 | |
| `min_inliers` | 6 | |
| `min_span_fraction` | 0.5 | |
| `allow_map_line` | false | |
| `matte_tol_m` | 0.05 | |
| `ghost_min_excess_m` | 0.10 | |
| `self_ghost_radius_m` | 0.5 | |
| `bg_cell_m` | 0.05 | |
| `bg_min_hits` | 3 | |
| `bg_dilate_cells` | 2 | |
| `cluster_eps_m` | 0.25 | |
| `cluster_min_beams` | 3 | |
| `cluster_max_extent_m` | 0.8 | |
| `track_gate_m` | 0.6 | association radius |
| `track_accel_sigma` | 4.0 | m/s², process noise |
| `confirm_m`, `confirm_n` | 4, 6 | |
| `track_max_misses` | 8 | scans |
| `range_sigma_m` | 0.01 | measurement noise used by the tracker |

---

## 5. Math reference and test vectors

All vectors below were computed numerically; use them as exact expected values (tolerance 1e-6 unless stated).

**Reflect a direction.** $\mathbf u'=\mathbf u-2(\mathbf u\cdot\mathbf n)\mathbf n$, $\lVert\mathbf n\rVert=1$. The sign of $\mathbf n$ does not matter.

**Beam and line.** Laser at $\mathbf s$, line through $\mathbf c$ with unit normal $\mathbf n$: $\hat r=\dfrac{(\mathbf c-\mathbf s)\cdot\mathbf n}{\mathbf u\cdot\mathbf n}$, valid when $\mathbf u\cdot\mathbf n\neq0$ and $\hat r>0$. Hit point $\mathbf w=\mathbf s+\hat r\,\mathbf u$.

**Unfold.** Measured range $r$: $\mathbf p=\mathbf w+(r-\hat r)\,\mathbf u'$.

| Laser $\mathbf s$ | Beam angle | $r$ | Line point $\mathbf c$ | Line direction | $\hat r$ | $\mathbf w$ | $\mathbf u'$ | $\mathbf p$ |
|---|---|---|---|---|---|---|---|---|
| (0, 0) | 0° | 5.0 | (2, 0) | (1, 1) | 2.000000 | (2.000000, 0.000000) | (0.000000, 1.000000) | (2.000000, 3.000000) |
| (0, 0) | 10° | 6.0 | (2, 0) | (1, 1) | 2.465606 | (2.428148, 0.428148) | (0.173648, 0.984808) | (3.041889, 3.908847) |
| (0, 0) | -5° | 4.5 | (3, 0.5) | (0, 1) | 3.011460 | (3.000000, -0.262466) | (-0.996195, -0.087156) | (1.517124, -0.392201) |
| (1, 0.5) | 20° | 7.25 | (4, 1) | (1, 2) | 3.577550 | (4.361797, 1.723594) | (-0.290199, 0.956966) | (3.296054, 5.238005) |

**Map to laser.** Laser pose $(x,y,\psi)$: $\mathbf q_{laser}=R(-\psi)\,(\mathbf q_{map}-(x,y))$. Example: pose (1, 2, 90°), map point (1, 5) gives laser (3, 0).

**Self image.** A beam at normal incidence on a mirror at range $d$ with nothing else in the way returns $r=2d$ and unfolds to the laser origin.

**Fit uncertainty.** $\sigma_\phi=\sigma_d\sqrt{12}/(S\sqrt N)$ with $\sigma_d$ the RMS perpendicular residual of the inliers, $S$ their span, $N$ their count. Examples: (0.005, 0.8, 9) gives 0.00722; (0.005, 0.8, 20) gives 0.00484; (0.003, 1.0, 25) gives 0.00208.

**Fresnel, unpolarised, n = 1.49** (for `forward_model` material tables): 0° 0.039, 30° 0.040, 45° 0.049, 60° 0.088, 70° 0.169, 75° 0.251, 80° 0.386, 85° 0.612 (tolerance 0.001).

**Roughness factor** $\exp[-(4\pi\sigma_h\cos\alpha/\lambda)^2]$, λ = 905 nm: (100 nm, 0°) 0.145; (100 nm, 75°) 0.879; (50 nm, 45°) 0.786; (200 nm, 60°) 0.145 (tolerance 0.001).

**Kalman filter** (per axis independent, state $[p, v]$, step $\Delta t$, acceleration noise $\sigma_a$): $F=\begin{bmatrix}1&\Delta t\\0&1\end{bmatrix}$, $Q=\sigma_a^2\begin{bmatrix}\Delta t^4/4&\Delta t^3/2\\ \Delta t^3/2&\Delta t^2\end{bmatrix}$, $H=[1\ 0]$, $R=\sigma_m^2$. New track: $p$ = measurement, $v=0$, $P=\mathrm{diag}(\sigma_m^2,\ 3^2)$.

---

## 6. Work packages

Order and dependencies:

```
WP0 ─┬─ WP1 ─ WP2 ─ WP3 ─ WP4 ─ WP5 ─ WP6 ─ WP7 ─┬─ WP8 ─ WP9 ─ WP14 (stretch)
     │                                            ├─ WP11
     │                                            ├─ WP12
     └─ WP10 (needs only WP0; E0 needs it in week 1)  └─ WP13 (stretch, needs WP12)
WP15 (docs) after WP8 and again after WP11
```

Each WP lists: files, interface, behaviour, tests with expected values, acceptance. "Acceptance" is the definition of done together with rule 8 (mutations).

### WP0. Scaffold

**Create**
- `src/nlos_perception/` with `package.xml`, `setup.py`, `setup.cfg`, `resource/nlos_perception`, empty `nlos_perception/__init__.py`, `test/conftest.py`. Copy the shape of `src/opponent_perception/` (same dependency tag style, same `conftest.py` marker registration, no pytest section in `setup.cfg`). Dependencies: `rclpy`, `sensor_msgs`, `geometry_msgs`, `std_msgs`, `visualization_msgs`, NumPy, SciPy, PyYAML. No console scripts yet.
- `src/nlos_perception/config/sensors_intensity.yaml`: a byte-for-byte copy of `src/f1tenth_system/f1tenth_stack/config/sensors.yaml` with only `publish_intensity: true` changed.
- `tools/nlos/reference/`: the reference files already committed with this plan (`nlos_proto.py`, `run1.py`, `run2.py`, `fitstats.py`, `fa_kf.py`, `fa2.py`, `results/*.txt`) and a `README.md` saying: reference only, slow on purpose, do not import from `src/`, tests may import it.
- `tools/nlos/tests/__init__.py` (empty) and `tools/mutations/nlos_perception.json` containing `{"copy": ["src/nlos_perception"], "mutations": []}`.

**Tests**
- `test_sensors_intensity.py::test_only_publish_intensity_differs_from_shared`: load both YAML files; the dictionaries are equal after setting `publish_intensity` to the same value in both; and the NLOS file has it `true`. Fails if the shared file gains a key the copy lacks.

**Acceptance**
- `colcon build --symlink-install --packages-select nlos_perception` succeeds.
- `python3 -m pytest src/racerbot_car/test/test_no_package_pytest_config.py src/racerbot_launch/test/test_car_profiles.py -q` passes (ROS sourced, domain 79).

### WP1. `geometry.py`

Pure functions, NumPy only, vectorised, no state. Arrays of points are `(N, 2)` float64.

```python
def rotation(angle: float) -> np.ndarray                       # (2, 2), counter-clockwise
def map_to_laser(points_map, pose) -> np.ndarray               # pose = (x, y, yaw) of the LASER in map
def laser_to_map(points_laser, pose) -> np.ndarray
def beam_dirs(angle_min: float, angle_increment: float, n: int) -> np.ndarray   # (n, 2) unit vectors
def left_normal(t) -> np.ndarray                               # (-t_y, t_x)
def reflect_dirs(u, n) -> np.ndarray                           # u (N, 2), n (2,) unit
def line_ranges(u, c, n) -> np.ndarray                         # laser at origin; NaN where |u.n| < 1e-9 or range <= 0
def unfold(u, rhat, r, n) -> np.ndarray                        # w + (r - rhat) u'
def fit_line_tls(points) -> tuple[np.ndarray, np.ndarray, float]   # centroid c, unit direction t, RMS perpendicular residual
def ransac_line(points, tol, iters, rng, t_prior=None, max_angle=None) -> np.ndarray | None   # inlier mask
def line_angle_sigma(sigma_d: float, span: float, n: int) -> float
def to_frame(points, origin, t) -> np.ndarray                  # coordinates along t and left_normal(t)
def from_frame(points_frame, origin, t) -> np.ndarray
```

`ransac_line`: draw `iters` point pairs with `rng`; skip pairs closer than 0.15 m; skip hypotheses whose direction differs from `t_prior` by more than `max_angle` (compare with `abs(t . t_prior) < cos(max_angle)`); keep the hypothesis with the most points within `tol`; then refit three times by total least squares, with tolerances `tol`, `tol/2`, `tol/2`, each time keeping points within the tolerance whose position along the line is inside the current inliers' extent plus 0.02 m; stop refitting if fewer than 4 points remain. Return `None` if fewer than 2 points are given or no hypothesis is accepted. Reference: `fit_line_ransac` in `tools/nlos/reference/nlos_proto.py`.

**Tests** (`test_geometry.py`)
- The four unfold rows of section 5, each checking `rhat`, `w`, `u'` and `p`.
- `map_to_laser` with pose (1, 2, 90°) maps (1, 5) to (3, 0); `laser_to_map` inverts it for 50 seeded random poses and points (1e-9).
- `reflect_dirs` gives the same result for `n` and `-n`; preserves length; a direction along the line is unchanged.
- `line_ranges` returns NaN for a beam parallel to the line and for a line behind the laser.
- Self image: beam at normal incidence, `r = 2 * rhat`, unfolds to the origin.
- `fit_line_tls` on 200 points spread evenly over 2 m of a 30° line with 5 mm noise (fixed seed): angle within 0.003 rad, residual between 4 and 6 mm.
- `ransac_line`: 12 collinear points plus 40 points of a second line 60° away, `t_prior` along the first: the mask selects exactly the 12. Without `t_prior` it selects the 40. One point: `None`.
- `line_angle_sigma`: the three examples of section 5.
- Empty `(0, 2)` input returns empty output for every array function. NaN in gives NaN out, no exception.

### WP2. `panels.py`

```python
@dataclass(frozen=True)
class Panel:
    id: str
    p0: tuple[float, float]
    p1: tuple[float, float]
    pattern: tuple[tuple[float, str], ...]     # (length_m, 'matte' | 'mirror')
    material: str = ''
    background: str | None = None
    # properties: length, t (unit p0->p1), n (left normal), edges (cumulative lengths, starts at 0.0)
    def kind_at(self, s_m: float) -> str | None   # element intervals are half-open [start, end); None outside [0, length)

P1_PATTERN = ((0.05, 'matte'), (0.20, 'mirror'), (0.05, 'matte'), (0.20, 'mirror'), (0.05, 'matte'), (0.20, 'mirror'), (0.05, 'matte'))

class PanelRegistry:
    panels: tuple[Panel, ...]
    @classmethod
    def from_mapping(cls, data: dict, base_dir: Path | None = None) -> 'PanelRegistry'
    @classmethod
    def from_yaml(cls, path) -> 'PanelRegistry'
    def by_id(self, panel_id: str) -> Panel

@dataclass(frozen=True)
class PanelPrior:
    p0_laser: np.ndarray; p1_laser: np.ndarray
    center_range: float; cos_incidence: float; facing: bool

def predict_in_laser(panel: Panel, pose) -> PanelPrior
```

Validation raises one `ValueError` listing every problem found: wrong `schema_version`; `frame` not `map`; unknown top-level or panel key; duplicate id; non-finite coordinate; zero-length panel; pattern kind not `matte`/`mirror`; non-positive element length; first or last element not `matte`; `|length - sum(pattern)| > 0.01`. An empty `panels` list is valid.

`facing` is `n . (laser - p0) > 0`. `cos_incidence` is `|d . n|` with `d` the unit vector from the laser to the panel centre.

**Tests** (`test_panels.py`)
- `config/panels.example.yaml` loads; one test per validation rule, each asserting its own message fragment.
- Pattern sum off by 0.009 loads; off by 0.011 raises.
- `kind_at` for P1: 0.0 matte; 0.0499 matte; 0.05 mirror; 0.2499 mirror; 0.25 matte; 0.7999 matte; 0.80 None; -0.001 None.
- `predict_in_laser`, pose (1, 2, 90°), panel `p0 = (2, 5)`, `p1 = (1, 5)`: `p0_laser` (3, -1), `p1_laser` (3, 0), `center_range` 3.0414, `cos_incidence` 0.9864, `facing` true. Swapping `p0` and `p1`: `facing` false.

### WP3. `forward_model.py`

A 2D scan simulator with one specular bounce. It is the test fixture for WP4-WP7, the engine of the simulator overlay (WP9) and of the sweeps (WP12).

```python
LABEL_WALL, LABEL_MATTE, LABEL_GHOST_STATIC, LABEL_GHOST_CAR, LABEL_DIRECT_CAR, LABEL_NO_RETURN = 0, 1, 2, 3, 4, 5

@dataclass(frozen=True)
class SensorModel:
    range_sigma_m: float = 0.005
    range_min_m: float = 0.02
    range_max_m: float = 30.0
    no_return_m: float = 65.533
    footprint_k: float = 0.012        # [MEASURE U2] beam width in the scan plane per metre of path
    max_ghost_path_m: float = 6.0     # [MEASURE U1] longest ghost path that still returns
    p_ghost: float = 1.0              # [MEASURE U1] probability a mirror beam inside the path limit returns

@dataclass(frozen=True)
class Scene2D:
    walls: np.ndarray                 # (K, 2, 2) diffuse segments, scene frame
    panels: tuple[Panel, ...]         # endpoints in the scene frame
    cars: tuple[tuple[float, float, float, float, float], ...] = ()   # (cx, cy, yaw, length, width)

def simulate_scan(scene, laser_pose, beam_angles, sensor, rng) -> tuple[np.ndarray, np.ndarray]   # ranges, labels
def l_corner_scene(width=1.5, panel_len=0.80, far=10.0, pattern=P1_PATTERN, cars=()) -> Scene2D
def rect_segments(car) -> np.ndarray                                   # (4, 2, 2)
def raymarch(occupied_fn, starts, dirs, max_range, step) -> np.ndarray # used by WP9
```

Behaviour of `simulate_scan`, per beam:
1. Find the nearest hit among walls, car edges and panels.
2. Wall: range = distance, `LABEL_WALL`. Car edge: `LABEL_DIRECT_CAR`.
3. Panel hit from behind (`u . n >= 0`): range = distance, `LABEL_WALL`.
4. Panel hit from the front at arclength `s`, distance `t1`, with `cos_a = -(u . n)`: footprint half-width along the panel `h = 0.5 * footprint_k * t1 / max(cos_a, 1e-3)`. If `[s - h, s + h]` lies inside a single mirror element, the beam is a mirror beam; otherwise range = `t1`, `LABEL_MATTE`.
5. Mirror beam: trace the reflected ray against walls and car edges only. If it hits at `t2`, and `t1 + t2 <= min(max_ghost_path_m, range_max_m)`, and `rng.random() < p_ghost`: range = `t1 + t2`, label `LABEL_GHOST_CAR` or `LABEL_GHOST_STATIC`. Otherwise no return.
6. No hit, or hit beyond `range_max_m`: range = `no_return_m`, `LABEL_NO_RETURN`.
7. Add `N(0, range_sigma_m)` to every returned range. Draw random numbers in beam order so results are reproducible.

`l_corner_scene(width=W)`: corridor A is `y in [0, W]`, `x in [-far, W]`; corridor B is `x in [0, W]`, `y in [0, far]`; inner corner at `(0, W)`. With `c = panel_len / sqrt(2)`: walls `(-far,0)-(W-c,0)`, `(-far,W)-(0,W)`, `(W,c)-(W,far)`, `(0,W)-(0,far)`; one panel `p0 = (W-c, 0)`, `p1 = (W, c)`. The far ends are open.

`raymarch`: step all rays together by `step` until `occupied_fn(xs, ys)` is true or `max_range` is passed; return the distance of the first occupied sample, `inf` otherwise.

**Tests** (`test_forward_model.py`)
- Analytic scene: laser pose (0, 0, 0); panel `p0 = (1.6, -0.4)`, `p1 = (2.4, 0.4)`, pattern `((0.05, matte), (1.03137085, mirror), (0.05, matte))`; one wall `(-1, 3)-(4, 3)`; noise 0, footprint 0, path limit 30. Beam 0°: range 5.0, `LABEL_GHOST_STATIC`. Beam 90°: range 3.0, `LABEL_WALL`. Beam 180°: 65.533, `LABEL_NO_RETURN`. Add a car `(2, 2, 0, 0.5, 0.28)`: beam 0° gives 3.86, `LABEL_GHOST_CAR`.
- Same scene, `max_ghost_path_m = 4.9`: beam 0° is no-return. `p_ghost = 0`: no-return.
- Footprint: same scene, beam 0° hits the panel 0.5657 m from `p0`. `footprint_k = 0.5` gives `h = 0.707` m, which exceeds the mirror element: `LABEL_MATTE`, range 2.0.
- Equivalence with the reference: for 20 seeded poses in `l_corner_scene()` (x uniform in [-6, -0.5], y in [0.3, 1.2], yaw N(0, 0.1)), with and without a car at (1.0, 3.0, 90°, 0.5, 0.28), noise 0: labels identical to `nlos_proto.trace_scan` and ranges equal within 1e-9, for `footprint_k` 0 and 0.012.
- Noise: 2000 draws of one wall beam at 3 m with σ 0.01 have sample σ between 0.009 and 0.011.
- `raymarch` against a half-plane `x > 2`: a ray from the origin along +x returns 2.0 within one `step`; along -x returns `inf`.
- Speed: one 1081-beam scan of `l_corner_scene()` with one car takes under 5 ms (mean of 50, after one warm-up call). Mark this test `slow` and report the number in the PR.

### WP4. `locator.py`

```python
@dataclass(frozen=True)
class PanelFit:
    origin: np.ndarray        # the p0 end, laser frame
    t: np.ndarray             # unit, p0 -> p1, laser frame
    n: np.ndarray             # left_normal(t); points toward the laser
    n_inliers: int
    sigma_d: float            # RMS perpendicular residual of inliers
    sigma_phi: float          # line_angle_sigma(sigma_d, span, n_inliers)
    span: float
    origin_source: str        # 'scan' or 'prior'
    beam_lo: int; beam_hi: int

def locate_panel(ranges, valid, dirs, prior: PanelPrior, panel_length: float, params, rng) -> PanelFit | None
```

Steps: 4.3 steps 2 and 3. Then orient `t` so that `t . (prior.p1_laser - prior.p0_laser) > 0`. Project inliers on `t`; let `s_lo`, `s_hi` be their extent. If `|(s_hi - s_lo) - panel_length| <= 0.06`: `origin` = the line point at `s_lo`, `origin_source = 'scan'`. Otherwise: `origin` = projection of the prior panel centre on the fitted line, minus `t * panel_length / 2`, `origin_source = 'prior'`. Return `None` if the fitted normal does not face the laser (`left_normal(t) . (-origin) <= 0`).

`valid` is `isfinite(r) & (r >= range_min) & (r < range_max)`.

**Tests** (`test_locator.py`), all on `simulate_scan` output from `l_corner_scene()`:
- Noise-free, exact prior, laser at (-3, 0.75, 0): angle error below 1e-3 rad, origin within 0.02 m of the true `p0` (in the laser frame), `origin_source == 'scan'`.
- Statistics, `footprint_k = 0.012`, σ 5 mm, prior error N(0.15 m, 0.15 m, 3°), 150 seeded poses with panel distance uniform in 2-5 m: failures at most 5 %; angle error 95th percentile at most 0.012 rad; origin error along `t` 95th percentile at most 0.04 m. (Reference: 0.010 rad and 0.032 m.)
- A distractor wall: the reference scene already has the outer wall of corridor B inside the gate; with `ransac_max_angle_rad` set to 3.14 the noise-free case above returns `None` or a line more than 0.3 rad from the truth (this proves the direction prior is doing work).
- Panel covered (pattern all matte via a scene whose panel is replaced by a wall): fit succeeds and the origin is correct.
- Panel absent (no panel, no wall there): returns `None`.
- Fewer than `min_inliers` valid points, all-no-return scan, all-NaN scan: `None`, no exception.
- Half the panel occluded by a car parked in front of the `p0` end: either `None` or `origin_source == 'prior'`, never a scan-anchored origin more than 0.06 m wrong.

### WP5. `ghosts.py` and `background.py`

```python
@dataclass(frozen=True)
class GhostResult:
    beam_idx: np.ndarray          # (K,)
    points_laser: np.ndarray      # (K, 2)
    points_panel: np.ndarray      # (K, 2)
    excess: np.ndarray            # (K,) r - rhat
    path_length: np.ndarray       # (K,) r
    n_matte: int; n_front: int; n_no_return: int; n_on_panel: int

def classify_and_unfold(ranges, valid, dirs, fit: PanelFit, panel_length: float, params) -> GhostResult

class PanelBackground:
    def __init__(self, cell_m=0.05, x_range=(-4.0, 4.0), y_range=(-0.5, 7.5))
    def add(self, points_panel, laser_panel_xy) -> None   # also marks the laser's coverage cell; raises RuntimeError after finalize()
    def finalize(self, min_hits: int, dilate_cells: int) -> None
    def is_static(self, points_panel) -> np.ndarray    # True outside the grid; raises RuntimeError before finalize()
    def covers(self, laser_panel_xy) -> bool           # coverage rule of 4.4; False outside the grid
    def save(self, path) -> None                   # .npz: counts, static mask, coverage mask, cells, ranges, schema_version
    @classmethod
    def load(cls, path) -> 'PanelBackground'
```

A beam is "on the panel" when `u . n < -0.05`, `rhat > 0` and its hit lies between `origin` and `origin + t * panel_length`. Classification and rejection as in 4.3 steps 4-6. Dilation uses `scipy.ndimage.binary_dilation` with the default cross structuring element, `dilate_cells` iterations.

**Tests** (`test_ghosts.py`, `test_background.py`)
- Section 5 rows 1 and 2 through `classify_and_unfold` with a hand-built `PanelFit`.
- Reference scene with a car at (1.0, 3.0, 90°): every point labelled `LABEL_GHOST_CAR` by the simulator unfolds to within 0.03 m of the car outline (noise 5 mm, true fit); every `LABEL_GHOST_STATIC` point unfolds to within 0.03 m of a wall segment.
- Counts: `n_matte + n_front + n_no_return + K` equals `n_on_panel`.
- Excess exactly `ghost_min_excess_m` is not a ghost; 1 mm more is.
- Self image is removed; a point with panel `y = 0` is removed.
- Background: a cell with 3 hits is static with `min_hits = 3` and not with 4; a point two cells (0.10 m) along x from a static cell's centre is static with `dilate_cells = 2`; a point four cells (0.20 m) away is not; a point at `x = 4.5` is static; save and load give identical `is_static` on 1000 seeded points; `add` after `finalize` raises; `is_static` before `finalize` raises.
- Coverage: after learning with the laser at panel position (-2.0, 1.0), `covers((-2.0, 1.0))` and `covers((-1.8, 1.0))` are true, `covers((-1.0, 1.0))` and `covers((9.0, 9.0))` are false; coverage survives save and load.

### WP6. `clustering.py` and `tracking.py`

```python
def cluster_points(points, eps: float, min_pts: int, max_extent: float) -> list[np.ndarray]
# single linkage: two points join if closer than eps. Clusters ordered by smallest member index.
# extent = larger side of the axis-aligned bounding box. Clusters outside [min_pts, max_extent] are dropped.

@dataclass
class Track:
    track_id: int
    state: np.ndarray      # [px, py, vx, vy], panel frame
    cov: np.ndarray        # (4, 4)
    hits: int; misses: int; n_beams: int
    last_stamp: float
    confirmed: bool

class PanelTracker:
    def __init__(self, params)
    def update(self, stamp: float, centroids, n_beams, sigma_m: float) -> list[Track]
    def reset(self) -> None
```

`update`: `dt = stamp - last update stamp`. If `dt <= 0` or `dt > 0.5`: `reset()` and treat this as a first update. Predict every track with the model of section 5. Associate greedily: repeatedly take the globally closest (track, measurement) pair within `track_gate_m` of the predicted position. Matched: Kalman update, `hits += 1`, `misses = 0`, history gets True. Unmatched track: `misses += 1`, history gets False; delete when `misses > track_max_misses`. Unmatched measurement: new track. `confirmed` becomes true when the last `confirm_n` history entries hold at least `confirm_m` True, and stays true until deletion. Track ids increase from 1 and are never reused within a tracker's life. Non-finite centroids are ignored.

**Tests** (`test_clustering.py`, `test_tracking.py`)
- Two blobs of 6 points 1 m apart: two clusters; with `eps = 1.5`: one cluster, dropped if its extent exceeds `max_extent`.
- A chain of points 0.2 m apart over 1.0 m with `eps = 0.25`: one cluster, dropped by `max_extent = 0.8`.
- 2 points with `min_pts = 3`: dropped. Empty input: empty list.
- Target moving at (0, -1.0) m/s, noise 2 cm, `sigma_m` 0.02, 40 Hz, 2 s: confirmed on exactly the 4th update; over 50 seeds, after 1 s the median absolute velocity error is below 0.10 m/s and no position error exceeds 0.06 m. (Reference over 300 seeds: velocity median 0.07, 95th percentile 0.20 m/s; position maximum 0.046 m.)
- Hits on updates 1, 3, 5 only: not confirmed after 6 updates. Hits on 1, 2, 4, 5: confirmed at update 5.
- Deleted on the 9th consecutive miss, not the 8th.
- Two targets 1.0 m apart moving in parallel keep their ids for 2 s.
- A measurement 0.61 m from the prediction starts a new track; 0.59 m updates the old one.
- `stamp` going backwards resets (ids continue, tracks gone). A gap of 0.6 s resets.

### WP7. `pipeline.py` and `offline.py`

```python
class NlosPipeline:
    def __init__(self, registry: PanelRegistry, params, backgrounds: dict[str, PanelBackground | None], seed: int = 0)
    def process(self, ranges, angle_min, angle_increment, range_min, range_max, time_increment,
                stamp_sec, pose, pose_stamp_sec, learn_background: bool = False) -> dict
```

`process` returns the dictionary of 4.6 (plain Python types, finite numbers only, JSON-serialisable with `allow_nan=False`). `pose` may be `None`: then every panel is `visible: false` and there are no tracks. With `learn_background=True` unfolded points are added to the panel's background and no tracks are produced. A panel whose background is `None` produces no tracks and reports `"background": "missing"` in its panel entry; a panel whose background does not cover the laser's position reports `"background": "uncovered"` and produces no tracks; otherwise `"background": "ok"`. One `PanelTracker` per panel.

`offline.py`: `python3 -m nlos_perception.offline demo --output <dir> [--footprint-k K] [--max-ghost-path M] [--speed V]`. The directory must not exist. Scenario: `l_corner_scene()`; background learned from solo passes (x from -7 to 0.9 in 0.05 m steps, on the seven lines y = 0.40, 0.52, 0.64, 0.75, 0.86, 0.98, 1.10, each pose with y noise N(0, 0.03) and yaw N(0, 0.05), and the same pose-prior error as below); then the ego drives the centre line from x = -7 at `V` (default 4 m/s), one scan per 1/40 s, with a stopped car at (1.25, 2.5, 90°, 0.5, 0.28); the pose given to the pipeline carries N(0.15 m, 0.15 m, 3°) error. Writes `tracks.jsonl` (one snapshot per line) and `summary.json` with `first_confirm_ego_x`, `first_los_ego_x` (first x where at least 3 `LABEL_DIRECT_CAR` beams occur in 4 of 6 consecutive scans), `gain_m`, `surface_rmse_m`, `n_scans`, and every parameter used.

**Tests** (`test_pipeline.py`, `test_offline.py`)
- Snapshot schema: exact key set of 4.6 for an empty registry, a visible panel with no track, and a confirmed track; `json.dumps(snapshot, allow_nan=False)` succeeds in all three.
- `pose=None`, stale pose, missing background, wrong beam count between calls (1081 then 541), all-no-return scan: no exception, documented output.
- Demo, thin beam (`--footprint-k 0 --max-ghost-path 30`): `gain_m >= 3.0`, `surface_rmse_m <= 0.05`. (Reference: 5.5 m, 0.03 m.)
- Demo, defaults: `0.8 <= gain_m <= 2.5`. (Reference: 1.5 m.)
- False alarms: four empty approaches (y uniform in [0.4, 1.1], x from -7 to 0.9 in 0.1 m steps, yaw N(0, 0.05), fixed seed): no confirmed track in any of the 316 scans and at most 2 scans with any cluster. (Reference: 0 and 0.)
- Coverage: the same four approaches with a background learned only for x < -0.2 report `"background": "uncovered"` for every scan with x > 0.1 and produce no track there.
- Demo with `learn_background` skipped: the summary reports zero confirmed tracks and the reason.

### WP8. `nlos_node.py`, launch, config

1. **Settle U6 first.** Find how the installed `urg_node` sets `header.stamp` (upstream `ros-drivers/urg_node`, the branch matching the installed apt version; look at `urg_c_wrapper.cpp`). Write the answer and the source lines into `docs/nlos-perception.md`. If the stamp is not the first beam's time, add the correction to the observation time and say so in the doc.
2. Node `nlos_perception_node`: declares every parameter of 4.7; loads the registry and backgrounds at start (a bad file is a fatal error with the validation message); keeps only the newest pose; processes each scan in the scan callback; publishes the snapshot, markers (one sphere per confirmed track and one line per visible panel, in the laser frame, lifetime 0.2 s) and, once per second, `/perception/nlos_status` with `schema_version`, scans processed, processing time p50 and p99 over the last second in ms, and the last error string.
3. `launch/nlos_perception_launch.py` with arguments `config_file`, `panels_file`, `pose_topic`, `scan_topic`. Add it to `tools/dev/launch_matrix.yaml` in the style of the `opponent-perception` entry.
4. Add `/perception/nlos_opponents` and `/perception/nlos_status` to `DEFAULT_TOPICS` in `src/race_diagnostics/race_diagnostics/recording.py`. If a test pins that list, update the pin in the same PR; if the test's intent is unclear, stop and report.
5. `setup.py` console scripts: `nlos_perception_node`, `nlos_perception_offline`.

**Tests** (`test_ros_adapter.py`, marker `ros`)
- The node's publishers are exactly the three topics above plus ROS built-ins; none has type `ackermann_msgs/msg/AckermannDriveStamped`.
- One synthetic `LaserScan` plus pose in gives one snapshot out with matching `stamp_sec`.
- Pose older than `pose_max_age_sec`: `pose_stale` true and `position_map_m` null.
- `pose_frame: base_link` shifts the laser 0.26 m along the heading (check `position_map_m` against a hand-computed value).
- Empty `panels_file`: snapshots with `"panels": []`.
- Scan with NaN, `inf`, 65.533 everywhere, or a changed beam count: the node keeps publishing.
- Clock going backwards (bag loop): tracker resets, node keeps publishing.

**Acceptance**
- On the dev machine the demo scenario replayed through the node gives p99 processing time under 5 ms for one panel. On the Jetson the target is under 10 ms `[MEASURE]`; a human reports it in E2.

### WP9. Simulator overlay and controller safety check

**Modify** `src/racerbot_sim/racerbot_sim/sim_bridge.py` and `gym_bridge_node.py`; **create** `tools/nlos/place_sim_panels.py`.

- `SimConfig` gains `nlos_panels_file: str = ''`. Empty means `SimBridge.scan()` returns exactly what it returns today.
- When set, `scan(agent)` post-processes the gym scan: for each beam compute its intersection `t1` with each panel from the laser pose (base_link pose plus `LIDAR_OFFSET_X` along the heading). A beam hits a panel if `t1 <= r_gym + 0.10`. Matte (same footprint rule as WP3): range `t1`. Mirror: reflect, find `t2` as the smaller of `raymarch(self.occupied, ...)` with `step = resolution / 2` and the exact intersection with every other car's rectangle (`_body_corners`); range `t1 + t2` if within `max_ghost_path_m`, else `inf` (the simulator's no-return value). Add the gym noise σ to ghost ranges.
- `nlos_perception.forward_model` is imported lazily, only when the file is set.
- `gym_bridge_node` gets a parameter `nlos_panels_file`; `sim_bringup_launch.py` and `sim_race_launch.py` get a launch argument `nlos_panels` passed through to it.
- `place_sim_panels.py --centerline <csv> --half-width <m> --out panels.yaml`: for each run of centre-line points whose curvature magnitude exceeds 0.3 1/m, take the middle point `q` with unit tangent `g` and outward unit normal `o` (pointing away from the centre of curvature); panel centre `q + o * (half_width - 0.10)`; endpoints `centre -/+ g * 0.40`; order them so the centre line is on the left of `p0 -> p1`.

**Tests**
- `test_sim_bridge.py`: with `nlos_panels_file=''`, `scan()` is bit-identical to the current implementation for 100 steps (compare against a second bridge built without the field set).
- With a panel file on a generated track: beams on matte stripes report the panel distance within 0.02 m; at least one mirror beam reports a range longer than the panel distance; with a second car parked in the mirror's view, unfolding those beams with the true panel line lands within 0.10 m of that car's rectangle (0.05 m map cells limit this).
- `place_sim_panels.py` on a rounded-rectangle centre line gives 4 panels, each 0.80 m long, each with the centre line on its left.

**Controller safety check (U10).** Run the existing solo validations with and without panels and compare. Start from `tools/racerbot_sim/run_gap_follow_validation.py` and `tools/f1tenth_sim/eval_gap.py`; read them first and use their own contact verdicts (never gym's flag).

| Check | Pass |
|---|---|
| Body contacts over 5 laps × 5 seeds, gap_follow, panels on | 0 |
| Mean lap time, panels on vs off | within 3 % |
| Number of scans in which gap_follow's opponent layer reports a moving car on a solo run, panels on | same as panels off (expected 0) |
| Same three checks for pure_pursuit on a mapped track | same |

If any check fails, **stop**. Do not tune gap_follow or the panel to make it pass. Report the numbers; the team decides between a narrower mirror window, a filtered scan for the controller, or restricting panel tests to pure_pursuit.

### WP10. Bench tools for E0

**Create** `tools/nlos/bag_io.py`, `tools/nlos/bench_analyze.py`, `tools/nlos/tests/test_bench_analyze.py`. Run with the analysis venv.

```python
def read_scans(bag_path, topic='/scan') -> dict
# keys: stamps (M,), ranges (M, N) float32, intensities (M, N) float32 or None,
#       angle_min, angle_increment, range_min, range_max, time_increment, scan_time
def read_poses(bag_path, topic) -> dict     # stamps (M,), x, y, yaw
```

Use `mcap_ros2.reader.read_ros2_messages` (see how `tools/run_analysis/` reads bags and follow that style).

Manifest (`e0_manifest.yaml`), written by the humans running E0:

```yaml
schema_version: 1
sensor: {car: rb2, model: UST-10LX, intensity: true}
recordings:
  - bag: E0A_acrylic-mirror_d2.0_a45_l1.0_white/      # folder containing the .mcap
    experiment: A                # A, B, C or D
    material: acrylic_mirror_3mm
    sample_width_m: 0.30
    d1_m: 2.00                   # laser to sample centre, sample centre on the laser's x axis (beam angle 0)
    incidence_deg: 45            # angle between the beam and the sample normal
    target: white_board          # white_board | black_board | car | none
    l_m: 1.00                    # sample centre to target face along the mirror direction; null if target is none
    slot_width_m: null           # experiment B only
    scan_angle_deg: 0            # experiment B only: where in the scan the sample sits
    notes: ''
```

`bench_analyze.py --manifest <yaml> --out <csv>`: for each recording, use the beams whose angle is inside the central 80 % of the sample's angular width, `atan(0.5 * sample_width * cos(incidence) / d1)` either side of `scan_angle_deg`. With `rhat` the beam's intersection with the sample plane from the measured geometry, classify every (scan, beam) sample: `wall` if `|r - rhat| <= 0.05`; `ghost_target` if `target != none` and `|r - (rhat + l)| <= 0.15`; `no_return` if invalid; `short` if `r < rhat - 0.05`; `ghost_other` otherwise. One CSV row per recording: manifest fields, number of scans and beams, fraction of each class, mean and standard deviation of `r - (rhat + l)` over `ghost_target` samples, mean intensity per class (empty if no intensity), scan rate.

**Tests**: build three synthetic recordings with `forward_model` (perfect mirror with target; all-matte sample; mirror with no target), write them with a small in-memory fake of `read_scans`, and check the fractions are 1.0 `ghost_target`, 1.0 `wall`, and 1.0 `ghost_other`-or-`no_return` respectively; check the bias column equals an injected 0.03 m offset within 2 mm.

### WP11. Registration, background and evaluation tools

**Create** in `tools/nlos/`: `register_panels.py`, `learn_background.py`, `eval_static.py`, `eval_dynamic.py`, `make_figures.py`, tests for each.

- `register_panels.py --map <map.yaml> --panel <id> <x0> <y0> <x1> <y1> [--panel ...] --out panels.yaml`: checks orientation against the map (the point 0.3 m to the left of the panel centre must be free, the point 0.3 m to the right must be occupied or unknown; if it is the other way round, swap the endpoints and print a warning; if neither, error). `--refine <bag>`: run the pipeline's locator over the bag; for scans with `origin_source == 'scan'`, map the fitted endpoints to the map with the bag's pose; replace each endpoint by the median; print the standard deviation and warn above 0.15 m.
- `learn_background.py --bag <solo bag> --panels <yaml>`: replay with `learn_background=True`, finalise with the configured parameters, save one `.npz` per panel, write the `background` keys into the YAML, print per panel: scans used, static cells.
- `eval_static.py --manifest <e1.yaml> --panels <yaml> --out <csv>` (E1). Per recording the manifest gives the hidden car's rear-axle position and yaw **in the panel frame** (tape-measured), or `none`. Metrics per recording: fraction of scans with a confirmed track within 0.5 m of the true outline (recall); number of confirmed-track scans farther than 0.5 m or with no car (false alarms); RMS distance from foreground unfolded points to the true outline (surface error); centroid-to-centre distance; number of foreground beams; ghost path length.
- `eval_dynamic.py --ego-bag <dir> --other-bag <dir> --panels <yaml> --out <csv>` (E2, E3). Ground truth is the other car's localization pose, converted to the panel frame with the registry. Time offset between the two bags: over the scans in which the ego sees the other car directly, find the offset in [-0.5, 0.5] s (1 ms steps) that minimises the RMS distance between the ego's direct points and the other car's outline; report it and the residual; refuse to continue if the residual exceeds 0.15 m. **Oracle line of sight**: a scan counts as a line-of-sight detection when at least 3 direct beams end within 0.10 m of the true outline; apply the same 4-of-6 rule. This is the earliest any line-of-sight detector could report the car. Metrics: everything in `eval_static.py`, plus first-confirm and first-oracle-line-of-sight times and ego positions along its path, the gain in metres and seconds, and track velocity error.
- `make_figures.py --results <dir> --out <dir>`: one function per figure of section 9, reading CSV only. Matplotlib, no seaborn.

**Tests**: each tool on recordings synthesised with `forward_model` and written through the fake reader of WP10: a registration with swapped endpoints is corrected; a learned background removes all static ghosts of the reference scene (zero foreground points on a solo pass with a different seed); `eval_static` reports recall 1.0 and surface error below 0.05 m on a clean synthetic recording and recall 0.0 with the car outside the mirror's view; `eval_dynamic` recovers an injected 0.120 s offset within 5 ms.

### WP12. Parameter sweep on the cluster

**Create** `tools/nlos/sweep_sim.py`, `tools/nlos/slurm_sweep.sh`.

`sweep_sim.py --grid <yaml> --shard I --num-shards N --out <dir>`: expands the grid, takes every N-th configuration starting at I, runs 200 Monte Carlo approaches per configuration with the WP7 demo logic, and writes one CSV row per configuration: the parameters, detection rate, median and 10th/90th percentile of `gain_m`, surface RMS, false alarms per 1000 empty scans. Seeds are derived from the configuration index so shards are reproducible.

Grid for the paper:

| Parameter | Values |
|---|---|
| Corridor width | 1.2, 1.5, 2.0 m |
| Panel placement | chamfer 45°; chamfer 35°; chamfer 55°; flat on the end wall |
| Panel length | 0.6, 0.8, 1.2 m |
| Pattern | P1; two 5 cm matte ends with one open mirror; 10 cm matte / 15 cm mirror |
| `footprint_k` | 0.004, 0.012, 0.025 |
| `max_ghost_path_m` | 4, 6, 8, 10 |
| Range noise | 3, 10, 30 mm |
| Pose prior error | (0.05 m, 1°), (0.15 m, 3°), (0.5 m, 8°) |
| Ego speed | 2, 4, 6 m/s |
| Hidden car | stopped at depth 0.5, 1, 2, 3 m in outer, middle, inner lane; moving away at 0.5× and 1× ego speed |

Run the full factorial only over the first six rows with the others at their middle value, then one-at-a-time for the rest. `l_corner_scene` needs two new arguments for this (`panel_angle_deg`, `panel_on_end_wall`); add them with tests that the default scene is unchanged. `slurm_sweep.sh` is a job-array template; if the cluster does not use SLURM, keep the Python interface and replace only this file.

After E0, rerun with the measured `footprint_k`, `max_ghost_path_m` and `p_ghost`.

### WP13 (stretch). Learned baseline

A small 1D CNN per panel sector. Input: 128 samples across the panel's angular extent, channels (range, `rhat`, excess, intensity or zero). Output: presence logit and (x, y) in the panel frame. Train on WP12 simulation with randomised parameters; test on E1-E3 recordings without fine-tuning, then with fine-tuning on half of E1. PyTorch on the cluster only; nothing from this WP is installed on the car. Report next to the geometric method in Table 3. Skip this WP if week 12 arrives with E3 incomplete.

### WP14 (stretch). Closed loop

Only after WP9 passes and E3 is analysed, and only with the team lead's written approval in the PR.

- `pure_pursuit_node` gains `enable_nlos_speed_cap` (default false), `nlos_topic`, `nlos_stop_margin_m`, `nlos_max_age_sec`. When enabled and a **confirmed** track newer than `nlos_max_age_sec` lies within 0.5 m of the racing line ahead, cap the speed so the car can stop `nlos_stop_margin_m` short of it with the node's existing braking model. The cap can only lower speed. It never steers, never triggers the emergency stop, and never replaces any existing check.
- Tests: with the parameter false, the node's outputs are identical on the existing test scenarios; with it true and no message, identical; a stale message is ignored; the cap is never above the uncapped speed. The LB deadman tests stay untouched and passing.
- Validation order: simulator (WP9 overlay, stopped car past each corner, 5 seeds), wheels off the ground with LB held, floor at 1 m/s, then E4.

### WP15. Documentation

`docs/nlos-perception.md` (what it is, the panel, how to register panels and learn a background, how to run, the output format, the limits) and `docs/nlos-experiments.md` (section 7 as runbooks with checklists). Follow `.claude/skills/novice-docs/SKILL.md`; add both to the index in `docs/README.md`. Update after WP11 with the evaluation commands.

---

## 7. Experiments (humans)

General rules for every experiment:
- A car that is not meant to move is on a stand with its wheels off the ground.
- Start the hardware with `ros2 launch f1tenth_stack bringup_launch.py sensors_config:=<install path of nlos_perception>/config/sensors_intensity.yaml record:=false`. Never start a second `urg_node`.
- Record bench data with `ros2 bag record -s mcap -o <name> /scan /tf_static`. Record driving runs with the normal recorder.
- Every recording gets a manifest entry before the next one starts. Measure with a tape to the nearest 5 mm; photograph each setup with the tape in view.
- Bags above 95 MiB cannot be committed: keep bench recordings to 5 s and driving runs to 3 minutes.

### E0. Bench characterisation (weeks 1-2, needs WP0 and WP10)

**Buy or borrow.** 3 mm acrylic mirror sheet; aluminised polyester film taped flat on a board; polished aluminium or stainless sheet; mirror-finish adhesive vinyl on a board; glossy white acrylic; glossy black acrylic; clear acrylic; matte white board (control); one real race air duct. Each flat sample at least 0.30 × 0.25 m. Targets: white paper board and matte black board, each 0.30 × 0.30 m, and the other car. A protractor or printed angle template, tape measure, matte white tape.

**Setup.** Sensor on the stand. Sample centre on the laser's forward axis at distance `d1`, at LiDAR height, turned so the beam meets it at the incidence angle. Target centred on the mirror direction (angle of reflection = angle of incidence) at distance `l` from the sample, facing the sample.

**E0-A, ghost return (U1, U5).** Acrylic mirror, full grid: `d1` 1, 2, 3 m × incidence 15, 30, 45, 60, 75° × `l` 0.5, 1, 2, 3 m × target white, black; plus one "none" per (`d1`, incidence). Every other material: `d1` 2 m × incidence 30, 45, 60, 75° × `l` 1, 2 m × white, black. Then the best two materials with the other car as target at `d1` 2 m, 45°, `l` 1, 2, 3, 4 m. 5 s each.

**E0-B, footprint and mixed beams (U2, U3, U4).** Best material, 45°, white target at `l` = 1 m. Mask the sample with matte tape leaving one vertical slot of width 5, 10, 20, 30 cm; `d1` 1, 2, 3, 4 m; repeat the 20 cm slot with the sample placed at scan angles 0°, 45°, 90° (turn the car on the stand, not the sample). The fraction of slot beams that report the ghost, against slot width and distance, gives the footprint.

**E0-C, duct (U9).** Race duct at 2 m, the other car beside the specular direction at 1 m; and the duct's flattest face at 30, 45, 60, 75°. Look for any `ghost_target` fraction above the "none" recording.

**E0-D, sensor options (U7, U8).** Read car 1's LiDAR label. On each car try `publish_multiecho: true` once and record whether the driver accepts it. Record 60 s with intensity on and note the scan rate and bag size per minute.

**Analysis.** `bench_analyze.py`; then fit per material: `p_ghost` by incidence bin, `max_ghost_path_m` as the longest `d1 + l` with `ghost_target` fraction ≥ 0.5 for each target, range bias, `footprint_k`. Put the numbers in `config/nlos_perception.yaml` comments and in the forward-model material table.

**Gate G0 (end of week 2).**
- Pass: at least one flat material gives `ghost_target` fraction ≥ 0.5 at 45°, `d1 + l` ≥ 3 m, on the black board. Continue with that material.
- Partial: it passes only on the white board or only below 3 m. Continue, but E1-E3 use a light-coloured dummy car and the paper says so.
- Fail: no material passes. Stop the detector work after WP7 and write the paper as the measurement study (section 10).
- Record separately whether the duct shows any ghost. That decides the race-day claim, not the gate.

### E1. Static corner (weeks 5-6, needs WP7 and WP11)

Build an L-corner from track barriers, both legs 1.5 m wide and at least 4 m long. Mount one panel as a 45° chamfer across the outside corner. Mark a floor grid with tape in the panel frame.

- Ego positions (static, on the stand, laser at the marked point): on the approach centre line 1.5, 2.5, 3.5, 4.5 m from the inner corner; and at 2.5 m, 0.3 m left and right of the centre line.
- Hidden car positions: depth 0.5, 1, 2, 3 m past the inner corner × outer, middle, inner lane, heading along the leg; plus the middle lane at depth 1 and 2 m turned 90°.
- For each ego position: one 20 s recording with no car and the cover on (wall reference), one 20 s with no car and cover off (background and false alarms), then 5 s per hidden position.
- Learn the background from the cover-off empty recordings. For this static experiment the evaluated ego position must be in the learning set (coverage rule, 4.4), so record the empty scene twice at every ego position: learn from the first recording, count false alarms on the second.

**Gate G1 (end of week 6).** With the car between 1 and 3 m past the corner and the ego 1.5-3.5 m from it: recall ≥ 0.7 over positions the simulation says are visible; surface error ≤ 0.10 m RMS; zero confirmed tracks in the empty recordings. If recall fails but ghost beams are present, the problem is software: fix and re-run offline. If ghost beams are absent, revisit E0 numbers and the panel height and tilt.

### E2. Hidden car moving, ego static (weeks 7-8, needs WP8)

Same corner. Both cars localise in the **same** saved map (copy the map files to both cars). Ego on the stand at 2.5 and 3.5 m. The other car drives through the hidden leg and around the corner under teleop at about 0.5, 1 and 2 m/s, 10 passes each, both directions. Each pass starts and ends with the cars in direct view of each other for at least 2 s (needed for the time offset). Run the node live and record its output. Report the node's p99 processing time from `/perception/nlos_status`. Also record 3 laps of the ego's normal localization with the panel covered and 3 with it uncovered, and compare pose lag with `race_diagnostics` (U11).

### E3. Ego moving (weeks 9-10)

Before any driving near a panel: WP9 must have passed, and the first laps are with the panel covered, then uncovered at 1 m/s with a person ready on LB.

- **Stopped object past a blind corner.** A cardboard dummy of 0.50 × 0.28 × 0.25 m (use the surface the gate G0 result allows) at depth 0.5, 1, 2, 3 m in the outer and middle lane, placed **off** the ego's line or with the ego's speed capped so it can stop in what it sees. Ego approaches at 2, 3, 4 m/s, 10 repeats each. Ground truth: tape, panel frame.
- **Car ahead disappears.** The other car drives ahead at 0.5-1× the ego's speed; 10 laps.
- Background learned on the same day from at least 10 solo laps: 4 on the racing line, 3 held about 0.3 m to its left, 3 about 0.3 m to its right, all the way through the corner (coverage rule, 4.4). Check with `learn_background.py` that no evaluated run reports `uncovered`.

**Gate G2 (end of week 10).** For the stopped object 1-2 m past the corner: median gain over oracle line of sight ≥ 1.0 m of ego travel, and at most 1 false confirmed track per 5 minutes of solo laps. If the gain is positive but below 1.0 m, report it as is and drop E4. If negative or zero, the paper's claim becomes "localises hidden cars that line of sight cannot see before the turn" (the 2-3 m deep, inner-lane case) instead of "earlier warning".

### E4. Closed loop (weeks 11-12, stretch, needs WP14)

Stopped dummy on the racing line 1.5 m past the corner; ego at 2, 3, 4 m/s; cap on and off; 10 runs each with the cap, and without the cap only at speeds where the existing safety net is known to stop in time. Report minimum distance to the dummy, and lap time on 10 clear laps with the cap on and off (the cost of false alarms).

---

## 8. Schedule

Five people, about 10 h a week each. Roles: **A** experiments and hardware; **B** forward model and sweeps; **C** detector and ROS node; **D** recordings, ground truth, evaluation; **E** literature, paper, reviews of agent PRs.

| Week | Agent WPs | Humans | Gate |
|---|---|---|---|
| 1 | WP0, WP10, WP1 | Buy samples; read the starred papers; prior-art search (E) | |
| 2 | WP2, WP3 | E0-A to E0-D | **G0** |
| 3 | WP4, WP5 | Fit E0 numbers into the forward model; build panels | |
| 4 | WP6, WP7, WP12 | First sweeps; choose panel placement | |
| 5 | WP11 | Build the corner; E1 | |
| 6 | WP8 | E1 analysis | **G1** |
| 7 | WP9 | E2; shared-map setup on both cars | |
| 8 | WP15 | E2 analysis; draft sections 3-5 of the paper | |
| 9 | (fixes) | E3 | |
| 10 | WP13 or WP14 if approved | E3 analysis | **G2** |
| 11 | WP14 | Sweeps with measured parameters; ablations; E4 if approved | |
| 12 | WP13 | E4; draft sections 6-7 | |
| 13 | `make_figures.py` final | Repeat any weak recordings | |
| 14 | | Full draft | |
| 15 | | Internal review; fix | |
| 16 | | arXiv; code and data release | |

Cut order if time runs out: WP14 and E4, then WP13, then the "car ahead disappears" half of E3.

---

## 9. Paper outputs and where each comes from

| Item | Content | Produced by |
|---|---|---|
| Fig. 1 | Corner, panel, raw ghost points and unfolded car | one E1 recording, `make_figures.fig_teaser` |
| Fig. 2 | Geometry of the forward model | drawn by hand |
| Fig. 3 | Ghost-return fraction against incidence angle, one curve per material | E0-A, `bench_analyze.py` |
| Fig. 4 | Ghost-return fraction against path length, white and black target | E0-A |
| Fig. 5 | Slot-width result and fitted footprint | E0-B |
| Fig. 6 | Pipeline diagram | drawn by hand |
| Fig. 7 | Simulated against measured detection map over hidden positions | WP12 and E1 |
| Fig. 8 | Surface error against ghost path length | E1, E2 |
| Fig. 9 | Gain over oracle line of sight, by depth and speed | E3 |
| Fig. 10 | Panel-placement comparison | WP12 |
| Table 1 | Sensor facts and what they rule out | section 2 |
| Table 2 | Materials: return fraction, path limit, bias | E0 |
| Table 3 | Methods against metrics: oracle line of sight, prediction through occlusion, geometric NLOS, learned NLOS | `eval_dynamic.py` |
| Table 4 | Ablations | below |
| Table 5 | Runtime on the Jetson; effect on localization | E2 |

**Ablations** (each is one parameter change replayed over the E1-E3 bags, no new recordings unless stated): scan-fitted line against map line (`allow_map_line` with the fit disabled); background grid against map-distance gating; background off; confirmation rule 1-of-1 against 4-of-6; stripe pattern (new recordings: P1 against open mirror); intensity as an extra gate (if intensity separates ghost from wall in E0); injected pose error 0.5 m and 8°; observation-time correction off; ego speed; target reflectance.

---

## 10. If a gate fails

| Event | Decision |
|---|---|
| G0 fails | Paper becomes "what a first-return 2D LiDAR can and cannot see around a corner": E0 in full, the forward model, the simulator, and the simulated detector as the upper bound. WP8, WP9, WP14 dropped |
| G0 partial | Continue with light-coloured targets; state the reflectance limit in the abstract |
| Duct shows no ghost | No race-day claim. The paper says the method needs a flat specular surface |
| WP9 safety check fails | No uncovered panel on a track with gap_follow driving. E3 runs under pure_pursuit only, or the team approves a narrower window and E0-B is repeated for it |
| G1 fails with ghost beams present | Software fault: debug offline on the E1 bags |
| G1 fails with no ghost beams | Check panel height, tilt and flatness; re-measure with the E0 rig in place |
| G2 gain below 1.0 m | Report the measured gain; change the headline claim as in G2; drop E4 |
| Localization degrades with panels uncovered (U11) | Report it as a cost; keep the cover on outside NLOS runs; do not modify the particle filter |
| Prior art found in week 1 | Keep the plan; reposition the contribution as the measurements and the racing evaluation |

---

## 11. Glossary

- **Ghost**: a LiDAR return whose range is the folded path sensor → mirror → object, so the point appears behind the mirror.
- **Unfold**: reflect a ghost point about the mirror line to get the object's real position.
- **Panel frame**: coordinates fixed to a panel (4.2).
- **Excess**: measured range minus the range to the panel along the same beam.
- **Oracle line of sight**: a detector that uses the true position of the hidden car to decide whether direct beams hit it; the best any line-of-sight method could do.
- **Gain**: how much farther from the corner the ego is at the first confirmed NLOS track than at the first oracle line-of-sight detection.
