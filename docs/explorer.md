# The explorer: let the car find its own limits for the simulator

> **Who this is for:** anyone collecting car data for the simulator, or retuning before a race on a new floor.
> **Read first:** [operations.md](operations.md), to bring the car up and run gap follow. [sim-calibration.md](sim-calibration.md) explains what the simulator's numbers are.
> **You'll be able to:** run a session where the car steps its own speed and cornering up until it finds the limit, check it in the simulator first, and fit the simulator to what it recorded.
> **Time:** 10 minutes in the simulator; a 15-minute session on the floor.

The explorer is a small helper that runs beside gap follow. It never drives. Between stretches of clean driving it raises two of gap follow's settings, a notch at a time, until the car shows it has reached its limit. Then it backs off. The runs it produces cover a spread of speeds and corner loads, which is what the simulator fit needs.

## Highlights

- **It cannot drive the car.** It publishes no drive topic. It only changes `max_speed` and `max_lateral_accel` through gap follow's live-tuning service, the same path as the [web dashboard](web-dashboard.md)'s sliders. LB, TTC braking and every other gap follow safety layer stay exactly as they are.
- **No cap of its own.** It climbs to gap follow's own slider maximum, 5.0 m/s and 4.0 m/s² (club decision, 2026-11-28). The limit is what it is measuring.
- **Three ways it knows it has hit the limit.** The car slides: it turns under 70% of what the steering asks, for 0.2 s. Gap follow has to emergency-brake. Or the car gets within its reaction distance plus 3 cm of something on its path.
- **Tested closed loop in the simulator.** On the 2026-11-11 course with a slippery planted floor (0.32 g), it found the slide, held cornering back and kept raising speed to 5 m/s. The simulator's own chassis check saw no wall contact (`tools/f1tenth_sim/tests/test_explore_sim.py`).
- **The fitter finds numbers it was never told.** Given simulated laps with planted values, it found grip 0.40 against 0.42, the steering delay exactly, and braking 2.94 against 3.0 m/s² (`tools/f1tenth_sim/tests/test_explorer_fit.py`).
- **It says what it could not pin down.** Each fitted number comes with how much the score cares about it. In the closed-loop session the braking ceiling scored 0: the data could not tell, and the report says so.
- **Light enough for the Jetson.** The explorer does a few hundred additions per scan and nothing heavy while the car moves.

**Limits, plainly:** simulator-validated only. It has never run on the floor. Braking strength and steering delay are not well measured by gap follow laps alone (see [What the fit can and cannot learn](#what-the-fit-can-and-cannot-learn)).

### Why it exists

Ordinary gap follow laps repeat the same corners at the same speeds. Two earlier simulator fits were rejected because that data rarely comes near the car's limits ([sim-calibration.md](sim-calibration.md#tyre-fit-both-rejected)). The explorer drives the spread on purpose.

## What it does, step by step

1. It sets gap follow to a gentle start: 2.0 m/s and 1.5 m/s².
2. After every 15 seconds of clean driving with LB held, it raises one setting by one notch. It alternates speed (0.5 m/s a notch) and cornering (0.25 m/s² a notch). A raise waits until the car is driving straight.
3. When the car shows a limit, it drops that setting two notches at once, wherever the car is, and never raises it again that session. A slide lowers cornering. A near-miss or an emergency brake lowers whichever setting it raised last.
4. After every third raise it runs a **brake test**: on a straight with room to stop plus 1 m, it drops `max_speed` to 1.5 m/s for 1.5 s, then puts it back.
5. It stops on a low battery (under 10.5 V), after 15 minutes of driving, or when both settings have found their limit.

It writes a report to `~/.ros/car_explorer/<time>.json`: the limits found, every back-off and why, the brake tests, and seconds of driving in each speed and cornering bin. It also publishes the report on `/car_explorer/status`. Every change it makes is also in the run's bag, on `/parameter_events`, which bringup's recorder already saves.

Releasing LB pauses it: nothing counts and nothing rises until LB is held again. When it exits, gap follow keeps the last setting it applied; restart gap follow to return to the shipped settings.

## Safety

- **LB is still the stop.** The explorer does not touch the deadman. Releasing LB stops the car, because gap follow refuses to drive without it.
- **gap follow's own braking is unchanged.** Its time-to-collision brake (0.30 s) and clearance stops still decide when the car brakes. The explorer's 3 cm margin only decides when to lower the settings. Loosening gap follow's braking is not part of this and would mean touching walls.
- **Test order, never skip ahead:** simulator, then wheels off the ground, then the floor at low speed in open space. [writing-your-own-node.md](writing-your-own-node.md#testing-before-its-on-wheels).
- **It looks for limits on purpose.** Expect slides near the end of a session. Run it only on an open course with nobody standing near the walls.

## Try it in the simulator first

**Any terminal, from the workspace root; no ROS or car needed.** Needs the simulator set up once ([simulator.md](simulator.md)).

```bash
export PYTHONPATH=.sim/python:.sim/f1tenth_gym:src/gap_follow:src/pure_pursuit:tools/f1tenth_sim:tools
export NUMBA_CACHE_DIR=.sim/numba-cache
python3 tools/f1tenth_sim/explore_sim.py --car rb2 --floor sfu_course_2026_11_11_rb2 --minutes 6
```

**Working when:** it prints a JSON report ending with `"contact_steps": 0`, a `limits` entry, and a list of `events`. It exits 1 if the simulated car touched a wall.

Add `--friction 0.32` to plant a slippery floor, and `--fit` to fit the recording afterwards (about 7 minutes on 4 cores). The `fit` section then shows each fitted number next to the true one.

## On car 2

**Before you start:**

- [ ] The workspace is built with `car_explorer`: `colcon build --symlink-install --packages-select car_explorer`, then `source install/setup.bash`.
- [ ] The battery is charged (the explorer ends the session below 10.5 V).
- [ ] The course is open and nobody is near the walls.

**Terminal 1:** the hardware. Leave it running.

```bash
source /opt/ros/jazzy/setup.bash && source ~/racerbot-ws/install/setup.bash
ros2 launch f1tenth_stack bringup_launch.py
```

**Working when:** the log settles with no `[ERROR]` lines.

**Terminal 2:** gap follow, without opponent awareness (one car, so every "opponent" would be a wall).

```bash
source /opt/ros/jazzy/setup.bash && source ~/racerbot-ws/install/setup.bash
ros2 launch gap_follow gap_follow_launch.py opponent_awareness:=false
```

**Working when:** gap follow logs that it is waiting for LB.

**Terminal 3:** the explorer.

```bash
source /opt/ros/jazzy/setup.bash && source ~/racerbot-ws/install/setup.bash
ros2 launch car_explorer explore_launch.py
```

**Working when:** it logs `explorer ready`, then `applied {'max_speed': 2.0, 'max_lateral_accel': 1.5}`.

**If it doesn't:** if no `applied` line appears, gap follow is not running or has another node name; check `ros2 node list` shows `/gap_follow_node`. A line saying gap follow `refused` a setting names the reason.

**First time on the floor:** keep the session short and the tops low. You change nothing in code. **Terminal 3, instead of the launch above:**

```bash
ros2 run car_explorer explorer_node --ros-args \
  --params-file $(ros2 pkg prefix --share car_explorer)/config/car_explorer.yaml \
  -p speed_top:=3.0 -p lateral_top:=2.5 -p max_session_sec:=300.0
```

**Working when:** it logs `explorer ready`, then the same first `applied` line. It stops raising at 3.0 m/s and 2.5 m/s², and ends after 5 minutes of driving.

Hold LB and let it drive. Watch the third terminal: each `applied` line is a new setting.

**Afterwards:** save the runs as usual with `tools/save_runs.sh`, and keep the report from `~/.ros/car_explorer/`.

## Fitting the simulator to a session

`tools/f1tenth_sim/sim_fidelity/explorer_fit.py` takes recorded runs and cuts them into short windows. It replays each window's commands through the simulated car, and searches for the numbers that make it move like the real one. Measured numbers (mass, wheelbase, width and the rest in `FIXED`) are never fitted.

**Terminal: development computer, workspace root, analysis Python with `rosbags==0.10.11`** (as in [sim-calibration.md](sim-calibration.md#run-the-three-stages)). Locate the runs first with `loc.py`, then:

```bash
python3 - <<'EOF'
from sim_fidelity import explorer_fit as ef
from sim_fidelity.calibration import calibration_for_car
runs = ['data/rb2/racerbot_runs/<time>']
windows = [w for r in runs for w in ef.windows_from_run(r, '<loc output dir>')]
result = ef.fit(calibration_for_car('rb2'), windows,
                ['surface_friction', 'steering_delay_ticks', 'speed_loop_accel_gain'], workers=4)
print(result)
EOF
```

**Working when:** it prints the fitted values, the score before and after, and a `sensitivity` for each. A sensitivity near 0 means the data did not pin that number down.

## What the fit can and cannot learn

<details>
<summary>Measured in the simulator, 2026-11-28 (safe to skip)</summary>

| Number | Planted-answer test (open-loop explorer-like commands) | Closed-loop session (gap follow + explorer, 8 min, 43 windows) |
|---|---|---|
| Grip, `surface_friction` | 0.40 found, 0.42 planted | 0.302 found, 0.32 planted |
| Steering delay | 5 ticks found, 5 planted | 2 found, 3 planted; sensitivity 0.007 (weak) |
| Braking ceiling | 2.94 found, 3.0 planted | not pinned down: sensitivity 0 |

Why braking is not learned from gap follow laps: gap follow ramps its own speed command down no faster than its `max_braking_decel` (3.0 m/s²), so the car never brakes at its physical ceiling. Measure that with a hard stop and `tools/run_analysis/braking_lag.py` instead.

Steering delay needs quick steering changes, which smooth gap follow laps rarely contain.

The open-loop windows start each replay with the wheels straight and the car not turning, which leaves some error the fit cannot remove (score 0.41 at the planted answer). That costs accuracy, not correctness, in the tests above.

</details>

## Tuning the explorer

Every value is in `src/car_explorer/config/car_explorer.yaml`, with a comment. The ones you might change:

| Setting | Default | What it does |
|---|---|---|
| `wall_margin` | 0.03 m | Room left after the reaction distance before it backs off. Also a launch argument: `wall_margin:=0.05` |
| `brake_delay` | 0.24 s | Car 2's braking delay; the reaction distance is speed × this |
| `slide_ratio` | 0.7 | Turning under this share of what the steering asks counts as a slide |
| `segment_sec` | 15 s | Clean driving per notch |
| `speed_top`, `lateral_top` | 5.0, 4.0 | gap follow's own maxima; lower them for a first floor session |
| `front_reach` | 0.41 m | Rear axle to front bumper. **An estimate, not measured.** Tape-measure it on car 2 |

<details>
<summary>Why the wall check counts only the reaction distance (safe to skip)</summary>

The first version asked: if the car braked now, on its current steering, would it stop within 3 cm of a wall? If so, it backed off.

In the simulator on the 2026-11-11 course it backed off at 2.5 m/s on ordinary corner approaches. Gap follow turns before a wall rather than stopping for it, so a full stopping distance on the current arc is nearly always "too close". The explorer never got past 2.5 m/s.

The check now asks a narrower question: is the car within the distance it covers before any braking bites (speed × 0.24 s), plus 3 cm, of something on its current path? That is a real near-miss whatever gap follow does next. With it the same simulated session climbed to 5 m/s with no contact.

</details>

## Tests

```bash
python3 -m pytest src/car_explorer/test/ -q                                   # 65 tests, no ROS needed
python3 -m pytest tools/f1tenth_sim/tests/test_explorer_fit.py tools/f1tenth_sim/tests/test_explore_sim.py -q   # with the sim PYTHONPATH above
```

The first runs in under a second. The second takes about 2 minutes: the planted-answer fit and three closed-loop simulator sessions.

Every rule in the explorer has a test that fails when the rule is broken (checked by hand, 2026-11-28). The ROS node itself has not been run here: no ROS install in the build environment. Run it on car 2 with its wheels off the ground first.
