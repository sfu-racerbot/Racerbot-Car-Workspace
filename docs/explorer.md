# The explorer: let the car find its own limits for the simulator

> **Who this is for:** anyone collecting car data for the simulator, or retuning before a race on a new floor.
> **Read first:** [operations.md](operations.md), to bring the car up, run gap follow and run SLAM. [sim-calibration.md](sim-calibration.md) explains what the simulator's numbers are.
> **You'll be able to:** run a session where the car maps the course, steps its own speed and cornering up until it finds the limit, runs its own stop tests, and then fit the simulator to what it recorded.
> **Time:** 10 minutes in the simulator; a 15-minute session on the floor.

The explorer is a small helper that runs beside gap follow. It never drives.

It first lets gap follow drive one gentle lap, to learn the course. Then, between stretches of clean driving, it raises two of gap follow's settings, a notch at a time, until the car shows it has reached its limit, and backs off.

Where the course has room, it runs its own stop tests. It keeps going until the recording holds what the simulator fit needs.

## Highlights

- **It cannot drive the car.** It publishes no drive topic. It only changes gap follow's settings through gap follow's live-tuning service, the same path as the [web dashboard](web-dashboard.md)'s sliders: `max_speed` and `max_lateral_accel`, and `min_speed` during a stop test. LB, TTC braking and every other gap follow safety layer stay exactly as they are.
- **No cap of its own.** It climbs to gap follow's own slider maximum, 5.0 m/s and 4.0 m/s² (club decision, 2026-11-28). The limit is what it is measuring.
- **It learns the course first.** With SLAM running, it drives one lap at gentle settings and remembers how much straight room there is at each place and heading. Stop tests then happen only where the map *and* the live scan both show room. Each place is used once, so the tests spread round the course.
- **It collects what the fit needs, then stops.** It counts the seconds of driving that teach each number: cornering near the limit (grip), braking harder than gentle slowing (braking strength), and quick steering (steering delay).
- **It decides its own tests.** A stop test runs when braking is the number furthest behind. The session ends only when both settings have found their limit and every number has its seconds.
- **Three ways it knows it has hit the limit.** The car slides: it turns under 70% of what the steering asks, for 0.2 s. Gap follow has to emergency-brake. Or the car gets within its reaction distance plus 3 cm of something on its path.
- **It blames the right setting.** The first slide holds cornering. After that only speed is still rising, so the next slide lowers speed. A slide below a speed it already found is flagged as a possible floor change. In the simulator on a slippery floor it stopped speed at 3.0 m/s after two slides.
- **One session teaches grip, braking and steering delay.** In the simulator we planted values in the simulated car that the explorer never sees. A 3-minute session plus the fit found grip 0.51 (planted 0.50), steering delay exactly, and braking 3.15 m/s² (planted 3.0). The simulator's own chassis check saw no wall contact.
- **Every fit is checked.** Each fitted number comes with a 90% range. One window in five is held back from the fit and scored afterwards: 0.36 before, 0.017 after in that session. A grip-change check compares the start of the session with the end.
- **Light enough for the Jetson.** The explorer does a few hundred additions per scan and nothing heavy while the car moves. The fit runs afterwards, on a laptop or the Jetson.

**Limits, plainly:** simulator-validated only. The ROS node has never run, on the car or anywhere else.

The 90% ranges count only the scatter between windows, not the simulator's own errors. In the simulator they came out narrower than the real miss (grip 0.504 to 0.505 against a true 0.50), so read them as a floor on the uncertainty.

The distance from the rear axle to the front bumper is a bound from car 2's measurements, not a measurement (see [Tuning the explorer](#tuning-the-explorer)).

### Why it exists

Ordinary gap follow laps repeat the same corners at the same speeds. Two earlier simulator fits were rejected because that data rarely comes near the car's limits ([sim-calibration.md](sim-calibration.md#tyre-fit-both-rejected)). The explorer drives the spread on purpose, and adds the hard stops ordinary laps never contain.

## What it does, step by step

1. It sets gap follow to a gentle start: 2.0 m/s and 1.5 m/s².
2. **It learns the course.** With SLAM running, it drives the first lap at that start and remembers the room ahead at each place. Without SLAM it waits 90 s, then carries on using the live scan alone.
3. After every 15 seconds of clean driving with LB held, it raises one setting by one notch. It alternates speed (0.5 m/s a notch) and cornering (0.25 m/s² a notch). A raise waits until the car is driving straight.
4. When the car shows a limit, it drops that setting two notches at once, wherever the car is, and never raises it again that session.

   The first slide lowers cornering. A later slide lowers speed, since cornering is already held and only speed is still rising. If both are already at their lowest notch and the car still slides, the session ends. A near-miss or an emergency brake lowers whichever setting it raised last.
5. **Stop tests.** When braking is the number most short of data, it waits for all of these:
   - a straight, at 2.5 m/s or more;
   - room to stop plus 1 m, on both the live scan and the course map;
   - a place it has not tested yet, and 20 s since the last stop test.

   Then it sets `max_speed` and `min_speed` to 0.5 m/s. When the car is under 0.8 m/s, or after 3 s, it puts gap follow's own `min_speed` and its ladder speed back.
6. It stops on a low battery (under 10.5 V) or after 15 minutes of driving. It also stops when both settings have found their limit and the recording holds 20 s of near-limit cornering, 4 s of hard braking and 15 s of quick steering.

It writes a report to `~/.ros/car_explorer/<time>.json`. The report holds the limits found, every back-off and why, and the stop tests. It also holds the course map's laps and places, the seconds of each kind of data, and the seconds of driving in each speed and cornering bin.

It also publishes the report on `/car_explorer/status`. Every change it makes is also in the run's bag, on `/parameter_events`, which bringup's recorder already saves.

Releasing LB pauses it: nothing counts and nothing rises until LB is held again. When it exits, gap follow keeps the last setting it applied; restart gap follow to return to the shipped settings.

## Safety

- **LB is still the stop.** The explorer does not touch the [deadman](glossary.md#deadman) (the rule that the car moves only while LB is held). Releasing LB stops the car, because gap follow refuses to drive without it.
- **gap follow's own braking is unchanged.** Its time-to-collision brake (0.30 s) and clearance stops still decide when the car brakes. The explorer's 3 cm margin only decides when to lower the settings. Loosening gap follow's braking is not part of this and would mean touching walls.
- **A stop test only ever slows the car.** It lowers `max_speed` and `min_speed`; gap follow keeps steering and braking as usual. Any limit sign or a low battery during a test puts both back at once.
- **If the explorer crashes during a stop test, gap follow stays at 0.5 m/s.** Release LB and restart gap follow.
- **No steering tests.** The simulator tried raising gap follow's steering gain to make quick steering flicks. At 2.5 the simulated car hit a wall. Ordinary gap follow driving already holds enough quick steering for the fit.
- **Test order, never skip ahead:** simulator, then wheels off the ground, then the floor at low speed in open space. [writing-your-own-node.md](writing-your-own-node.md#testing-before-its-on-wheels).
- **It looks for limits on purpose.** Expect slides and hard stops. Run it only on an open course with nobody standing near the walls.

## Try it in the simulator first

**Any terminal, from the workspace root; no ROS or car needed.** Needs the simulator set up once ([simulator.md](simulator.md)).

```bash
export RACERBOT_CAR=rb2
export PYTHONPATH=.sim/python:.sim/f1tenth_gym:src/gap_follow:src/pure_pursuit:tools/f1tenth_sim:tools
export NUMBA_CACHE_DIR=.sim/numba-cache
python3 tools/f1tenth_sim/explore_sim.py --car rb2 --floor sfu_course_2026_11_11_rb2 --minutes 6
```

**Working when:** it prints a JSON report ending with `"contact_steps": 0`, a `limits` entry, a `course` entry with `"learned": true`, and a list of `events`. It exits 1 if the simulated car touched a wall.

Add `--friction 0.32` to plant a slippery floor, and `--fit` to fit the recording afterwards (about 12 minutes on 4 cores). The `fit` section then shows each fitted number next to the true one, its `range_90`, the held-back scores and a `grip_change` check.

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

**Terminal 3:** SLAM, which gives the explorer the car's place on the course.

```bash
source /opt/ros/jazzy/setup.bash && source ~/racerbot-ws/install/setup.bash
ros2 launch racerbot_launch slam_launch.py
```

**Working when:** it settles without repeating errors, as in [operations.md](operations.md#building-a-map).

**Terminal 4:** the explorer.

```bash
source /opt/ros/jazzy/setup.bash && source ~/racerbot-ws/install/setup.bash
ros2 launch car_explorer explore_launch.py
```

**Working when:** it logs `explorer ready`, then `applied {'max_speed': 2.0, 'max_lateral_accel': 1.5, 'min_speed': 1.429}`.

**If it doesn't:** if no `applied` line appears, gap follow is not running or has another node name; check `ros2 node list` shows `/gap_follow_node`. A warning that it could not read `min_speed` means no stop tests this session. A line saying gap follow `refused` a setting names the reason.

**First time on the floor:** keep the session short and the tops low. You change nothing in code. **Terminal 4, instead of the launch above:**

```bash
ros2 run car_explorer explorer_node --ros-args \
  --params-file $(ros2 pkg prefix --share car_explorer)/config/car_explorer.yaml \
  -p speed_top:=3.0 -p lateral_top:=2.5 -p max_session_sec:=300.0
```

**Working when:** it logs `explorer ready`, then the same first `applied` line. It stops raising at 3.0 m/s and 2.5 m/s², and ends after 5 minutes of driving.

Hold LB and let it drive. The first lap stays at the start settings while it maps the course. Watch the explorer's terminal: each `applied` line is a new setting, and one with `'max_speed': 0.5` is a stop test.

**Afterwards:** save the runs as usual with `tools/save_runs.sh`, and keep the report from `~/.ros/car_explorer/`.

## Fitting the simulator to a session

`tools/f1tenth_sim/sim_fidelity/explorer_fit.py` takes recorded runs and cuts them into short windows. It replays each window's commands through the simulated car, and searches for the numbers that make it move like the real one. Measured numbers (mass, wheelbase, width and the rest in `FIXED`) are never fitted.

Each replay first drives a 0.75 s lead-in that is not scored. That lets the simulated wheels and turn rate catch up with the real car's before the comparison starts.

**Terminal: development computer, workspace root, analysis Python with `rosbags==0.10.11`** (as in [sim-calibration.md](sim-calibration.md#run-the-three-stages)). Locate the runs first with `loc.py`, then:

```bash
python3 - <<'EOF'
from dataclasses import replace
from sim_fidelity import explorer_fit as ef
from sim_fidelity.calibration import calibration_for_car
start = calibration_for_car('rb2')
runs = ['data/rb2/racerbot_runs/<time>']
windows = [w for r in runs for w in ef.windows_from_run(r, '<loc output dir>')]
names = ['surface_friction', 'steering_delay_ticks', 'speed_loop_brake_max']
result = ef.fit(start, windows, names, workers=4, holdout_every=5, with_intervals=True)
print(result)
print(ef.drift(replace(start, **result.values), windows, workers=4))
EOF
```

**Working when:** it prints the fitted values, the score before and after, the held-back score before and after, a 90% range for each number, and a grip-change check with `'changed': False` or `True`.

Read it like this:

- **Held-back score:** it should fall about as much as the fit score. If it barely falls, the fit is learning the session's quirks, not the car.
- **90% range:** a wide range means the session did not pin that number down. Drive another session; the explorer collects more of whatever the ledger says is short.
- **Grip change:** `True` means the floor changed during the session, for example it got dustier. Fit the two halves separately rather than trusting one number for both.

## What the fit can and cannot learn

<details>
<summary>Measured in the simulator, 2026-11-28 (safe to skip)</summary>

| Number | Planted-answer test (open-loop explorer-like commands) | Closed-loop session (gap follow + explorer, 3 min, 56 windows) |
|---|---|---|
| Grip, `surface_friction` | 0.42 planted, see the test | 0.51 found, 0.50 planted |
| Steering delay | found exactly | 5 ticks found, 5 planted |
| Braking ceiling, `speed_loop_brake_max` | 3.0 planted, within 10% | 3.15 found, 3.0 planted |
| Held-back score | falls below 30% of its start | 0.36 before, 0.017 after |

Ordinary gap follow driving already teaches braking and steering delay. Gap follow limits how fast its speed command *rises*, never how fast it falls, so every corner entry is a hard brake. An earlier version of this page said the opposite. That was wrong, and two faults in the fit's own harness had hidden the truth:

- Every replay started with the wheels straight and the car not turning, while the real car was mid-corner. That start-up error was larger than anything a number could fix, and it drowned the steering delay. The lead-in removed it: replaying a planted car with its own numbers scored 0.205 before and 0.001 after.
- The simulator's recording was one tick out of step with what the replay reports. The fit made up for it with one extra tick of steering delay and weaker brakes.

On a slippery floor the braking ceiling cannot be learned. With grip 0.32 the tyres slide at about 3.1 m/s², before the brakes reach their 4.0 ceiling. The grip number then covers braking too. That is the physics, not a fault.

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
| `brake_speed` | 0.5 m/s | What a stop test sets `max_speed` and `min_speed` to |
| `brake_gap_sec` | 20 s | Driving between stop tests |
| `need_grip_sec`, `need_braking_sec`, `need_steering_sec` | 20, 4, 15 s | Data each number needs before the session may end. Raise one if its 90% range came out wide |
| `learn_timeout_sec` | 90 s | How long to wait for the mapping lap to close before carrying on with the scan alone |
| `pose_frame` | `map` | SLAM's frame (the coordinates it reports the car's position in). The course map needs one that does not drift |
| `front_reach` | 0.445 m | Rear axle to front bumper. **A bound, not a measurement:** car 2 is 0.495 m long with a 0.325 m wheelbase, so the nose is 0.375 to 0.445 m ahead of the rear axle. The explorer assumes the long end. One tape measurement, rear axle to front bumper, would give it back up to 7 cm |

<details>
<summary>Why the wall check counts only the reaction distance (safe to skip)</summary>

The first version asked: if the car braked now, on its current steering, would it stop within 3 cm of a wall? If so, it backed off.

In the simulator on the 2026-11-11 course it backed off at 2.5 m/s on ordinary corner approaches. Gap follow turns before a wall rather than stopping for it, so a full stopping distance on the current arc is nearly always "too close". The explorer never got past 2.5 m/s.

The check now asks a narrower question: is the car within the distance it covers before any braking bites (speed × 0.24 s), plus 3 cm, of something on its current path? That is a real near-miss whatever gap follow does next. With it the same simulated session climbed to 5 m/s with no contact.

</details>

## Tests

```bash
python3 -m pytest src/car_explorer/test/ -q                                   # no ROS needed
python3 -m pytest tools/f1tenth_sim/tests/test_explorer_fit.py tools/f1tenth_sim/tests/test_explore_sim.py -q   # with the sim PYTHONPATH above
```

The first runs in under a second. The second takes several minutes: the planted-answer fit, the grip-change check and closed-loop simulator sessions.

Every rule in the explorer has a test that fails when the rule is broken. A sweep of 22 deliberate bugs in the new logic was each caught (2026-11-28). The ROS node itself has not been run here: no ROS install in the build environment. Run it on car 2 with its wheels off the ground first.
