# Changelog

Notable changes to the workspace's own packages, newest first. Format:
one dated section per work session, grouped by package, with behavior
changes and new/removed parameters called out explicitly. Upstream
submodule bumps don't go here (see `docs/git-setup.md`) — this file is
for changes the team made.

## 2026-10-30 (latest) — neural_racer student stages: race rules, rewards and the league's opponents

A CPU audit raced teacher v5 (update 150) in the league's own races, with
the league's rewards, 40 races per setup. Alone in a 1-seat sim, 19
finished and 16 were still racing at the 90 s limit. Alone in a 4-seat
sim, it stalled in 35 and finished none: the empty seats' cars sit at the
origin, `privileged()` still reports them, and the solo-trained teacher's
normaliser clips those inputs. With them masked, it was back to 19
finishes. In traffic with them masked, it finished 10 and stalled behind
opponents in 16. A car that stopped was never ended alone, and in traffic
lost −73 on average to a contact charge paid every tick. The league also
re-drew every opponent's driver every rollout, mid-race.

### neural_racer_train — race rules and rewards (behavior changes)

- **Reward defaults are the teacher's** (`RewardConfig`): wall cost
  25 + 15(v/4)² (was 5 + 15(v/4)²), stall 40 (was 20). `PPOConfig.gamma`
  0.998 (was 0.995). The Runner's episode limit is `learn.MAX_SECONDS`,
  180 s (was 90). `teacher.py` already passed all of these, so teacher runs
  are unchanged; distillation and the league get them.
- **Contact with another car costs once per contact**, on its first tick,
  and only between cars still racing. It was charged on every tick, wrecks
  included. Running into a wreck is still a fault and still ends the race.
- **A car below 0.1 m/s for 3 s is stalled, whatever the cause**
  (`IDLE_TICKS`). With the wrapper on, only a car the wrapper held counted,
  so stopping by choice was never ended or charged.
- **A race ends when the learner's does** (`Runner(end_with_learner=True)`,
  `sim.end_with_seat0`). After seat 0 finished or crashed, the env had raced
  on until every car was done, and a parked opponent never is. `sim.step`
  also returns `crashed`. PLR level scores count only the learner's racing
  ticks (`Runner.add_scores(..., mask)`).
- League: a curvature-cap cut costs `--cap-penalty` 0.5 per v_ref m/s.
  Gate S counts those cuts as interventions, and the student's squash
  doesn't hold the cap.

### neural_racer_train — a solo teacher in seats for more cars

- `model.solo_view`: the privileged state as a teacher trained alone saw it,
  with no other cars and in first of one. It feeds distillation's labels
  with `--max-cars` over 1, and the league's kickstart, unless the teacher's
  checkpoint says `traffic`.
- `model.fresh_opponent_inputs`: the league's `--critic-from` critic gets
  normaliser statistics for the opponent block and the place from the
  league's own state. Their first-layer weights are zeroed, since they never
  trained, and the sample count is capped at 1e7 so the statistics can still
  move (the teacher's count is about 3e9).

### neural_racer_train — league_learner

- `OpponentTable`: every race keeps its drivers to the finish. New races draw
  from one half of the table. Matchmaking refills the other half once no
  car still racing uses it. Archived snapshots are read once and cached.
- `StackedOpponents(recurrent=...)`: an MLP snapshot races with its memory
  input at zero, as it trained. Run as recurrent, its actions differed by up
  to 1.8.
- Win rule (`won`): a car that crashed loses to one that didn't; otherwise
  the place decides.
- An exploiter reset restores log_std −1.5 and a fresh actor optimiser and
  step size. It used to keep the student's −1.0 and the old Adam state.
- The main roles race the held-out tracks every `--eval-every` (200) updates
  (student_eval) and keep `best.pt`. Archive entries record their width.
- `./run.sh league STUDENT MLP_STUDENT TEACHER [RUN] [ROLES] [SEGMENTS]`
  replaces the stale distill-v1 path. `fir_league_submit.sh` adds
  `--init-mlp` (main1_mlp starts from the `--memory none` student),
  `--teacher`, `--critic-from`, `--library` and `--roles`, and uses eight
  3-hour segments by default (was six of 12 h).

### neural_racer_train — distillation

- The episode-start flag carries across rollouts. About 1 in 64 new
  episodes started with the previous episode's GRU memory.
- Distillation and the league skip a held-out eval once the time-limit
  signal has arrived.
- `./run.sh distill-sweep` has no traffic run, because the solo teacher
  stalls behind other cars. `full` and `masked` are submitted first.

### neural_racer_train — referee

- `NeuralDriver` passes the cornering cap that observation spec v2 added.
  Every neural race raised a `TypeError` on its first tick.
- `eval_worker`: a race that fails in a worker comes back as an error. The
  gym's bootstrap called `SystemExit` inside the worker, and `Pool.map`
  waited forever. A matchup where every race failed is left unrecorded, so
  it is retried. Snapshots load at their recorded width. SIGTERM ends the
  worker at once, where before the job script waited up to an hour on it.
- Item 3a, the referee in Fir jobs: the gym's `f1tenth_gym.envs` imports cv2,
  which loaded the host's GLib 2.68. The software stack's libgio needs 2.70's
  `g_source_set_static_name`, so every import failed with `undefined symbol`.
  `fidelity.ensure_paths` now loads the stack's own GLib, GObject and GIO
  first (`prefer_compat_glib`, from `$EBROOTGENTOO`; a no-op elsewhere).
  Preloading those three fixed the import in job 1524538. All 33 tests that
  need the gym now pass on a Fir compute node (job 1530304); none of them
  could run there before.
- `test_trajectory_matches_fidelity_plant`, which was never run where the gym
  was missing, failed by 5.9 m. The race's `a_lat_cap` (default 2.0) has set
  the sim's cornering cap since teacher v4, so its "never caps" wrapper config
  no longer uncapped it. The test now uncaps the race; the sim was not at fault.

### neural_racer_train — benched against the car's current algorithms

- `python -m neural_racer_train.bench CKPT...` races students against what car 1
  runs today, in the referee: gap_follow (its `gap_follow.yaml` race defaults,
  through `gap_mirror`) and pure pursuit on the map's line. Each driver races
  alone (finish, lap time, wall contacts), and each student races each of them
  head to head with the grid order alternating (wins, at-fault contacts). It
  runs on car 1's course, the other obstacle-free real maps and held-out
  procedural tracks, with the student at gap_follow's 5.0 m/s and a chosen
  cornering cap. The league's cars start here: it is their floor.
- `tools/f1tenth_sim`, `src/gap_follow` and `src/pure_pursuit` are taken as
  they are on main (ae7ceeb), so the referee and the bench race the shipped
  algorithms. `test_config.py` and `neural_racer.yaml` take main's change: the
  cornering cap is no longer paired with gap_follow's, which moved to 2.5.
- `referee.NeuralDriver(a_lat=...)`: the wrapper's cornering cap, also given
  to the policy.
- `referee.NeuralDriver` adds the training sim's IMU noise (`imu_vibration`, default
  1.0; per-race seed), so the referee tests a student on an IMU as noisy as car
  1's is while driving. `bench --imu-vibration 0` gives the exact IMU.

### neural_racer_train — sensor noise from car 1's drives (data/racerbot_runs)

Read from the bags of runs 20261030-175701 (gap_follow, race defaults, 4.8 m/s
top), 20261026-194913 and 20261026-180228 (the course). The analysis scripts
live outside the repo.

- **IMU vibration.** The node feeds the policy `/imu/data`'s latest message
  each tick, with no averaging. While driving, that sample's scatter around a
  0.08 s mean was 0.023–0.034 rad/s (gyro z), 0.4–1.0 m/s² (accel x) and
  0.6–1.9 m/s² (accel y), growing with speed. That is 5–13× the sim's 0.005
  and 0.15. At rest it is 0.001–0.008 rad/s and 0.03–0.09 m/s². Now the noise
  is the rest level (`imu_gyro_noise` 0.004, `imu_accel_noise` 0.06) plus,
  once moving, `imu.VIBRATION`'s fitted curves (std = c0 + c1·v), scaled by
  `imu_vibration`, which is drawn from 0.4–1.5 per episode.
- **LiDAR dropout in sectors.** 95% of the car's invalid beams come in runs of
  4 or more (median 9, p90 56). In the 2026-10-30 room, a sector at +16° was
  invalid in half the scans. The sim dropped single beams at 0.2%. Now
  `lidar.sector_dropout` adds transient sectors (`lidar_sectors` per scan on
  average, drawn from 0–1, each 4–60 beams wide) and, in about 57% of
  episodes, one persistent sector (`lidar_persist*`). The range noise
  (6–12 mm) already matched the sim's 10 mm.
- Not changed: car 1 commands speed with a gain of 3650 ERPM per m/s
  (`ackermann_to_vesc`) but reads odometry with 4040 (`vesc_to_odom`). Its
  odometry therefore reads 0.90 of the command, and SLAM agrees with the
  odometry (0.97 over 136 m), so the car drives about 10% slower than
  commanded. That is for the team to fix on the car; the sim follows
  commands exactly.

### neural_racer_train — car 1's measured IMU position

- `CarParams.imu_offset_x` 0.45 → 0.36 m from the rear axle. `/imu/data` is the
  BNO085, rotated into car axes but not moved, and car 1 mounts it at the
  front axle's centre (`tools/imu_calibration/config/car.yaml`). It was the
  team's front-bumper estimate. The referee's `NeuralDriver` takes the same
  point (0.18 m ahead of the CoG, was 0.27). Its gyro noise measured at rest is
  0.0014 rad/s, under the simulated 0.005, which also has to cover driving
  vibration, so the noise model is unchanged.

### neural_racer_train — teacher v6, and a teacher's last races inside its time limit

- Teacher v6 (jobs 1721555/7, from v5 on `tracklib-v2`) trained all 5e8 steps
  (plain) and 3e8 (masked) with a crash hazard of 5–9e-5 per tick on the
  tight-corner tracks. Its held-out finish stayed at ~88%, and car 1's course
  only reached 14–19%. levine_blocked stayed at 0%, where the LiDAR-driven
  student finishes 92%, so the teacher's privileged view of the real maps is
  suspect and its real-map score says little. Both runs were killed at the 3 h
  limit: each wrapper-on eval took 13–29 min, ten of them filled the segment,
  and the reraces ran past the time-limit signal.
- `teacher.py`: the final whole-plan race and the reraces start only before
  the time-limit signal, and the reraces stop between snapshots. Otherwise
  train() saves and returns as checkpointed, and the next segment (past its
  step budget) runs them again.
- `./run.sh teacher6`: `--eval-every 100`, and the signal 25 min before the
  limit (`--signal=B:USR1@1500`).
- `./run.sh distill-sweep TEACHER [MASKED|-] [STEPS] [LIB] [TAG] [VARIANTS]`:
  the track library, the run-name tag and which runs to submit. The
  teacher's reference bar is `--min-teacher-finish 0.7`: it cannot see two
  real maps' boxes, so it tops out near 0.8 on the student's races.

### Fir job scripts

- `run.sh` keeps a broken GPU node out for 7 days, up from 24 h. fc10512 (no
  CUDA device) came back after 24 h, and the scheduler handed it to six jobs
  in a row on 2026-10-31, each failing in 30 s.

### neural_racer_train — tight corners in training (track config v2, teacher v6)

distill-v3-full (teacher v5) finished 81.3% of its 300 held-out races, level
with the teacher's 80.0% on the same races. Both stalled in every race on car 1's
course: the braking cap held them nose-on to the outside wall after the
hairpin, from which no reachable arc was clear. Every real map has corners
tighter than 1.3 m centreline radius (car 1's course 0.71 m, room 0.65 m,
levine_blocked 0.50 m), and the training generator never made one
(`min_radius` 1.3 m). The car's tightest turn is 1.05 m, so those corners
need a wide line.

- `TrackGenConfig.turn_room`: a corner tighter than this must have its radius
  plus its outside width at least this, so the car's turning circle fits.
  0 (the default) is off, and v1 is unchanged.
- `learn.TRAIN_CONFIGS`: `v1` (as before) and `v2` (`min_radius` 0.45 m,
  `turn_room` 1.4 m, which covers the outer front corner's ~1.34 m sweep at
  full lock). In v2, 67% of tracks have a corner under 1.3 m and 30% one under
  0.8 m. The obstacle-free real maps meet v2's corner rules (each has at
  least 1.55 m of radius plus outside width).
- A library records its config, and a run takes it from there: the feeder
  generates with it once the library runs out (`Library(dir).cfg`). A library
  written before `turn_room` existed still loads as v1.
  `tracklib --config v1|v2`, `./run.sh tracklib [v1|v2]`.
- `teacher.py --eval-real`: its held-out races and reraces also run on the
  obstacle-free real maps (`teacher_eval.OBSTACLE_FREE_REAL`), so best.pt
  answers for car 1's course. `Evaluator(real=...)` takes a tuple of names.
- `./run.sh teacher6 [plain|masked]`: fine-tunes v5 on `tracklib-v2` with the
  whole wrapper and `--eval-real`.

### neural_racer_train — car 1's hairpin in training (track config v3, teacher v7)

Teacher v6 (best 88.9% in-loop; masked 86.6%) and distill-v4-full (82.0%
of 300 held-out races) still finished no race on car 1's course. In 3 of 4
races, teacher and student both stopped at the same spot. They came out of
the left hairpin round the divider's tip too wide and stopped at
(−1.0…−1.5, −0.4), heading −46° to −71°, 0.38 m from the bottom wall, at
full left lock. The swept clearance along every arc the wrapper allows was
0.00–0.14 m, under the 0.2 m reserve, so only reversing would get them
out. No generated track, v1 or v2, has a U-turn round a thin divider.
gap_follow, with the same 1.05 m turning radius, gets round it in the
referee.

- `tracks.hairpin_loop`: a long loop round a thin divider (0.25–0.8 m), its
  corridor 1.1–1.8 m wide. Each U-turn has room on the outside mid-turn (up
  to +1.0 m) and a pinch just after its exit, where the outside wall comes
  in to 45–95% of the half-width. The loop is mirrored at random, so the
  U-turns go both ways. 93% pass `check()` with v2's rules. They reproduce
  car 1's failure: on 32 fresh ones (CPU check, wrapper on), teacher v6
  finished 13 of 64 races and distill-v4-full 9, and every stall came
  within 3 m after a U-turn. 31 of the 32 have a minimum-curvature line no
  tighter than car 1's 1.05 m turn at the 0.33 rad cap.
- `TrackGenConfig.hairpin_share` (default 0, so v1 and v2 are unchanged).
  `learn.TRAIN_CONFIGS['v3']` is v2 with `hairpin_share` 0.3.
- `realmaps.variant(r, mirror, reverse, scale)` and `realmaps.variants(name)`
  give a real map mirrored (left and right swapped), driven the other way
  round, and scaled ×0.92, ×1 or ×1.08. That makes 12 per map, each with its
  own digest.
- `tracklib --real MAPS --real-copies N` adds every variant of those maps
  to a library, N copies each (default 40, about 5.5% of an 8192-track
  library). They take seeds from `REAL_SEED_BASE` (999 000 000), below the
  held-out base. `./run.sh tracklib v3` builds the v3 library with car 1's
  course in it.
- **Car 1's course is no longer held out for anything trained on a v3
  library.** For those runs its score in the held-out races and the bench
  measures fit to the car's track, not generalisation. The other four real
  maps and the procedural held-out tracks stay unseen.
- `./run.sh teacher7 [plain|masked]`: fine-tunes teacher v6 (or v6-masked)
  on `tracklib-v3`, with the same settings as teacher6.

### neural_racer_train — car 2's course and car 2's sensors

Car 2's runs reached the repo on `unify/merge-car2` (`data/rb2/`). Its
2026-10-30 drives (gap_follow, and map_course's auto race on a 51.3 m
course) were extracted and compared with car 1's and with the sim.

- **Car 2's course is the sixth real map, as `car2_course_2026_10_30`**
  (`heldout.REAL`, `teacher_eval.REAL_MAPS` and `OBSTACLE_FREE_REAL`,
  `heldout.yaml`, catalog). Upstream has two versions (map files from
  `origin/main`). `course_2026_10_30` has its outside filled in, but its
  walls stand 7.5 cm (median) to 20 cm (p90) into the corridor: narrowest
  0.80 m against 1.00 m. `course_2026_10_30_scans` has its walls corrected to
  where real scans landed, but they are gappy and its outside is free.
  car2_course takes the `_scans` walls and closes them with the filled
  version, its unreachable pockets removed and shrunk 0.25 m. Then every
  free cell a 0.3 m disk can't reach from the centreline becomes wall: the
  insides of the infield islands, and slivers out through gaps in the outer
  wall, which LiDAR beams in the sim had gone through. Widths match the
  `_scans` map (median 1.55 m, min 1.00 m). On the filled map, nothing
  finished in the referee with car 1's geometry (gap_follow, pure pursuit,
  distill-v4-full 0/3 each), though the minimum-curvature line needs no
  tighter than a 1.33 m radius. New `realmaps.reachable_free(occ, centre,
  origin, res, radius=0.15)` and `write_clean` (`python -m
  neural_racer_train.realmaps SRC OUT [--fill MAP]`). Lap 51.3 m. Held out:
  no library trains on it.
- **Real-map centrelines are smoothed on load** (`realmaps.CENTRE_SMOOTH`,
  a 0.25 m Gaussian). The CSV centrelines jog: every real map had 3–12
  points whose curvature was over 1/m above their neighbours'. Car 2's course
  had a 0.32 m radius over 5 cm, at which teacher v6 turned into a wall with
  1.5 m clear straight ahead. Generated tracks never jog, and the teacher
  reads the curvature ahead. Smoothing moves the line by at most 8 cm and
  keeps its clearance from the walls; laps get 0.3–0.4 m shorter. A real
  map's digest now covers its centreline. `tracklib --real-refresh` /
  `add_real(..., refresh=True)` rewrites a library's real-map tracks, and
  `tracklib-v3`'s 480 were rewritten.
- **The gyro has its own vibration scale** (`imu_gyro_vibration`, drawn
  0.4–2.3). The accelerometers keep `imu_vibration`, whose range is now
  0.3–1.5 (was 0.4–1.5). While driving, car 2's gyro reads 0.020–0.048
  rad/s, up to 1.5× the sim's old top. Its accelerometers read 0.27–0.48
  (x) and 0.28–0.39 (y) m/s², quieter than car 1's. One shared scale could
  not cover both cars. `imu.noise_std(..., gyro_vibration=)`.
- **`lidar_sectors` is drawn 0–1.5** (was 0–1). Car 2's UST-10LX loses
  0.9–1.0 runs of beams per scan while driving (median run 4–5 beams, p90
  16–27, max 72). Car 1 loses 0.1–0.6. Neither car had a sector invalid in
  more than 30% of moving scans.
- Measured on car 2 and already inside the sim's randomization: wheelbase
  0.33 m and a 0.349 rad steering limit. That gives a 9% tighter turn than
  car 1 for the same angle, inside the ±10% steering-gain error. Speed
  follows the command 0.2 s late. /imu/data runs at 50 Hz, inside the
  0–2 tick IMU delay. Under gap_follow (`max_lateral_accel` 2.5) the car
  reached lateral accelerations of p50 2.0, p90 2.9 and max 4.1 m/s².

### neural_racer_train — league library, and an escape-creep experiment switch

- `./run.sh league ... [LIB]`: the league's track library, by default
  `tracklib-v3`. Before, it was always `tracklib-v1`, which has none of the
  tight corners, hairpins or car 1's course.
- `SafetyTorch.creep` (sim only, default 0 = off, as on the car): gap_follow's
  escape creep (`escape_creep_speed`). Inside the 0.2 m reserve the braking
  cap allows up to that speed, using the braking formula on half the
  reserve. Experiments set it to ask whether the car's wrapper should have
  the creep: teacher v7 still stops at car 1's hairpin exit, 0.13 m from the
  wall, with every arc it could take under the reserve. The car's wrapper
  (`src/neural_racer`) is unchanged.

### neural_racer_train — top speeds, overtaking, contact cost, release at full speed (2026-11-08)

The team's call: "bump speeds as high as we can; the neural racers will
figure out what works best; crashed cars should stay on the track". A
release crossplay at 5.0 m/s (2.5 m/s²) found both of our best snapshots
beating gap_follow and pure pursuit, yet vetoed for 3–11 at-fault contacts.

- **Training's motor goes to 10 m/s** (`gpu_sim.params.TRAIN_SPEED_MAX`;
  `learn.V_CAP_RANGE` 1.0–10.0). 5.05 m/s is only the VESC's configured
  limit (`vesc.yaml` speed_max 23250 at gain 4600, still
  `CAR1_OVERRIDES['speed_max']`); the car can reach about 27 m/s. 10 m/s is
  the most any course we race can use. With the measured 4.5 m/s²
  acceleration, 4.0 braking and grip, levine_blocked's fastest line peaks at
  9.8 m/s, car 1's course at 7.6, car 2's at 6.5–8.2 and the room at 5.0.
  Raising the cap from 5.05 to 8 m/s takes 1–14% off those laps. Nothing
  above about 5.4 m/s is measured, so the motor's power limit `v_switch` is
  drawn ±30%. The CPU referee, the release crossplay and the car keep the
  VESC's limit until the team raises it. Car-side limits above about 8 m/s,
  for the team: the student's LiDAR encoding stops at 10 m (`R_MAX`), and
  the wrapper's braking look-ahead is 3 m (`swept_clearance_max_length`).
- **Speed caps near the top.** 70% of levels draw their speed cap from
  4.0 m/s to the top of `V_CAP_RANGE` (`learn.V_CAP_HIGH`, `V_CAP_HIGH_SHARE`). Only 10% sit at
  exactly 2.0 m/s² cornering (`A_LAT_CAR_SHARE`, was 40%). Both come from
  a separate random stream, so every other draw of a level is unchanged.
- **Overtaking levels.** In 30% of multi-car levels (`OVERTAKE_SHARE`), the
  learner starts at the back of the grid. The opponents ahead are held to
  70–90% of its speed cap. `Level.v_cap_scale` gives each car its share,
  and `BatchedRaceSim.car_v_cap()` uses it in the observation, the
  privileged state and the wrapper. `tracks.start_grid(..., order=)` places
  the grid.
- **At-fault contact costs 60** (`RewardConfig.fault_contact`, was 20),
  above a wall touched at walking pace (25). Crashed cars stay on the
  track, as before.
- **The floor's measured grip curve** (plan workstream 4). It replaces the
  friction circle in training: `actuation.grip_clamp` with
  `CarParams.grip_model` 1 (the default) turns the car by what `main`'s
  `ProgressiveGrip` delivers. That is the mean of the two measured floors
  (`sfu_course_2026_10_30`, `sfu_course_2026_11_06`), stretched per race by
  `grip_scale`, drawn from 0.85–1.3. Up to 2.0 m/s² the car turns as asked;
  asking 4.5 gives about 3.3. `grip_model` 0 keeps the circle; the parity test
  against the CPU referee's plain calibration uses it. In the replay of the
  cars' `/drive` commands, car 2's turning response came out 0.77–0.81
  (real 0.69–0.81) and lateral p99 3.16 m/s² (real 3.25). Car 1 needs a
  `grip_scale` near 1.25 for its 3.39.
- **Head-to-heads are scored on finishing order** (plan workstream 6,
  `referee.finish_outcome`). A car that crashes or runs out of time never
  beats one that finishes, and two non-finishers draw, however far each got.
  Crashed cars stay on the track (the team's call), and in the Nov 7 round
  robin 87% of head-to-heads had no finisher, so distance had decided them.
  The bench reports wins, losses and draws, the wins from races the student
  started second (the overtaking measure, workstream 2), and at-fault
  contacts per 100 races. The release crossplay (`eval_worker`) counts a
  win the same way. league-v1's crossplay scored on distance moved to
  `eval-max-distance/`.
- **Wrapper reserve stays at 0.20 m.** In the sim what-if, 0.12 m (gap_follow's
  since `main` 2fb626e) changed nothing: the smoke snapshot's results were
  identical race for race on the 6 real maps, and teacher v9's differed by
  one race.
- **Release crossplay at full speed.** `eval_worker` now defaults to
  5.05 m/s and a 5.0 m/s² cornering cap, writing to `<archive>/eval-max/`
  (`RELEASE_V_CAP`, `RELEASE_A_LAT`, `RELEASE_TAG`). The 2.5 m/s results stay
  in `eval/`. `fir_run.sh select` and `export` and the release lane read
  `eval-max/`.

### neural_racer_train — improvement plan, first steps (2026-11-08)

The Nov 7 round robin (10 algorithms on `main` 2fb626e's simulator) put
gap_follow narrowly ahead of our best neural racers. Each of our models
failed at least one of car 2's courses, and nobody overtook. The plan is in
the team's doc, "Neural Racer Improvement Plan". First steps:

- **Car 2's Nov 6 courses are held-out real maps**:
  `car2_course_2026_11_06a_clean` (lap 32.0 m) and
  `car2_course_2026_11_06b_clean` (21.1 m), in `heldout.REAL`,
  `teacher_eval.REAL_MAPS` and `OBSTACLE_FREE_REAL`, `heldout.yaml` and the
  catalog. Each is built like `car2_course_2026_10_30`: the `_scans` walls,
  closed by the filled map shrunk 0.25 m, then `reachable_free`. Upstream's
  `car2_course_2026_11_06b_scans_centerline.csv` is a symlink to the plain
  course's centreline; here it is a copy. Both courses loop round a thin
  divider, like car 1's.
- `tracklib-v4` (built with `python -m neural_racer_train.tracklib --config v3
  --real car1_course_2026_10_26,car2_course_2026_11_06a_clean,car2_course_2026_11_06b_clean
  --real-copies 10`): v3's tracks plus every variant of those three courses.
- `eval_worker --a-lat --tag --only`: a crossplay at other caps writes to
  `<archive>/eval-<tag>/`, beside the standard one (the release crossplay at
  deploy speed); `--only` races just the named entries.

### neural_racer_train — the floor's real grip, the car's real top speed (teacher v9)

The cars' recorded `/drive` commands were replayed open loop through
gpu_sim's actuation and single-track model, at several values of `mu`, and
compared with each car's gyro. Car 1 (runs 20261030-175701 and
20261026-194913) matched at `mu` ≈ 0.5–0.55. Car 2 (20261030-184556 and
-185036) matched at ≈ 0.4: yaw gain 0.69–0.81 by speed band, lateral
acceleration p99 3.25 m/s². Car 2's own analysis
(`data/rb2/.../20261026-174911/analysis/yaw_tracking.txt`) puts its grip limit
near 2.0 m/s² and its steering lag at 145 ms. Training drew `mu` from
0.40–1.10, so both real floors sat at its bottom end.

- **`mu` is drawn from 0.30–0.80** (was 0.40–1.10).
- **The car's top speed is 5.05 m/s** (`speed_max` 23250/4600, was 23250/3650
  = 6.37). Car 1's motor command gain is 4600 since `main` ea5f114 (it
  delivered 0.79 of every command at 3650). `learn.V_CAP_RANGE` is 1.0–5.05.
- `fidelity`: `NR_SIM_ROOT` makes the referee use another workspace's
  `tools/f1tenth_sim` (with its `src/gap_follow` and `src/pure_pursuit`), so
  the bench can race the team's current algorithms from an upstream checkout.
- `./run.sh teacher9 [plain|masked]`: from teacher v8's best.pt on
  `tracklib-v3r10`, which is v3 with 10 copies of each car-1 variant, not 40.
  The longer v8 trained, the more it learnt car 1's course itself: 0.36 →
  0.64 there, while car 2's course fell 0.56 → 0.24 and levine_blocked 0.94
  → 0.64.

### neural_racer_train — teacher v8, from teacher v7's last update

distill-v5 (teacher v7, tracklib-v3) finished in one segment each: 85.6% of
held-out races at its best, level with distill-v4-full's 85.0%. Car 1's
course: 25% (full) and 75% (masked), against 0% for distill-v4 (4 races
each, so read these as a direction, not a number). Car 2's course: 50% and 75%.

- `teacher.py --init-actor best|latest`: with `--init-from`, start from the
  run's best.pt actor (as before) or from its latest.pt. Teacher v7's best.pt
  is its update 100, chosen on the procedural tracks, but car 1's course kept
  improving to its last update (rerace finish 0.22 → 0.30).
- `./run.sh teacher8 [plain|masked]`: teacher v7 (or v7-masked) trained on
  for another 5e8 steps on `tracklib-v3`, from its latest actor.

### neural_racer_train — job logs on scratch

- `run.sh` and `fir_league_submit.sh` send every job's log to
  `$SCRATCH/neural_racer/logs/%x-%j.out`. Before, logs landed in the
  workspace root, beside the runs' code.

### neural_racer_train — car 1's course in the held-out races

- `car1_course_2026_10_26`, the cleaned SLAM map of car 1's run
  20261026-194514 (31.1 m lap, 1.06–1.52 m wide), is the fifth real map
  (`heldout.REAL`, `teacher_eval.REAL_MAPS`, `heldout.yaml`). The student eval
  that picks distillation's and the league's `best.pt` now races the course
  the car drives, and the CPU referee can race it too. The map files come
  from `origin/main`'s `tools/f1tenth_sim/maps`.

## 2026-10-29 — neural_racer safety wrapper brakes on the swept path, as gap_follow does

The first distill-v2 run (Fir job 1084316, 2 h 50 min) trained a student that
finished 0% of its held-out races, and the teacher, raced in the same
student world, also finished 0% (98% in its own races). The wrapper's LiDAR
braking cap took the nearest return in *any* direction, so a corridor's side
walls capped the speed: it bound on 99.6% of ticks at ~0.3 m/s, and the
held cars counted as stalled (crashed). The rest of the sweep was cancelled
before it started.

### neural_racer (car side)

- **Behavior change, `SafetyWrapper.apply` step 5**: the braking cap's
  clearance is `gap_logic.swept_path_clearance` (how far the body, plus a
  margin, can drive along the arc it steers before reaching a return),
  over gap_follow's 180° forward window, for both the arc asked for and the
  slew-limited one. Before, it was `minimum_footprint_clearance` over
  all 270°. Same reserve, deceleration and latency. `WrapperResult.clearance`
  is now that free length.
- **New parameters** (`WrapperConfig`, `neural_racer.yaml`, the node), all
  gap_follow.yaml's: `forward_fov_deg` 180, `swept_clearance_margin` 0.05 m,
  `swept_clearance_max_length` 3.0 m. `test_config` holds them equal to
  gap_follow's.
- Tests: driving straight down a 0.6–1.5 m corridor is not braked. Steering
  into its wall is. A return beside the body stops the car. Returns behind
  the forward window are ignored.

### neural_racer_train

- `SafetyTorch.swept_clearance`: the same clearance, batched;
  `test_swept_clearance_matches_gap_logic` holds it to gap_logic's on
  corridors at every steering. A scripted driver with the wrapper on now
  runs 2.06 m/s against 2.08 without it (before the fix it was 1.09 m/s,
  with the braking cap active on 90% of ticks).
- `distill.py`: a new run stops before training if the teacher, raced in
  the student's world, finishes under `--min-teacher-finish` (0.8), or the
  braking cap binds on more than `--max-teacher-stop` (0.2) of its ticks.
- `student_eval.evaluate` races the three episode types in one batch
  (`Evaluator.races(episode_type=[...])`, `teacher_eval.split`), about 3×
  faster. `races()` now returns its tick counts and lateral-acceleration
  histogram per race.
- `./run.sh distill-sweep` names its runs `distill-v3-*`.

### neural_racer_train — teacher v5 inside the whole wrapper

Even with the swept path, teacher v4 finished 1 of 8 student-world races
(6 of 8 without the wrapper). Six of the seven failures were stalls: it
races 2–12 cm from walls, inside the swept band's 5 cm side margin, where
the braking cap is zero, so the wrapper held it. With no margin, 4 of 8
finished and the rest braked late into a wall from up to 5.6 m/s. v4
trained with the wrapper off and never met either limit.

- `teacher.py --safety on` (already in the trainer, so far unused) is teacher
  v5's regime. Every race (held-out, final and rerace) now has the wrapper on
  too (`wrapper` plan entries, `teacher_eval.evaluate_file`). Because they cast
  the LiDAR, those races run on the GPU between updates instead of in the
  CPU worker. The training log adds `safety_stop` (share of live ticks the
  braking cap bound).
- `./run.sh teacher5 [plain|masked] [seeds]`: fine-tunes 5e8 steps with
  `--safety on --cap-mode squash`. `plain` starts from teacher-v4 seed 1,
  `masked` from teacher-v4-masked.

### neural_racer_train — the Fir job environment installs triton

`requirements-fir.txt` never listed triton, so every training job's LiDAR
cast ran on the PyTorch fallback. Only the GPU check installed triton by
hand. distill-v2 rolled out at 2.7e4 steps/s and teacher v5's first segment
at 3.6e4, with the GPU at 100%. The Triton kernel is about 6x faster on an
H100 (gpucheck 1030003). triton is now in the list, and `fir_env.sh` prints
which backend the cast will use.

## 2026-10-28 — neural_racer teacher v4 inside the car's cornering cap; observation spec v2; distillation and league rebuilt

teacher-v3 finished 99.1% of 320 fresh held-out races with no cornering cap,
but 36.2% when held to the car wrapper's 2.0 m/s² curvature cap (it trained
with the wrapper off and cornered at 4.6 m/s², 95th percentile). This
session builds training plan step 1 and the trainer/network plan's items
(`docs/superpowers/specs/2026-10-28-neural-racer-*.md`). Teacher v4 jobs are
submitted on Fir; nothing here is measured at scale yet.

### neural_racer (car side) — observation spec v2

- **`OBS_SPEC_VERSION` 1 → 2; ego 23 → 24 values**: `ego[23]` is the
  cornering cap the wrapper holds the speed to, over `A_LAT_SCALE` (5 m/s²).
  `build_observation` now requires `a_lat=`; the node passes its configured
  `max_lateral_accel`. A v1 model is refused by `model_store` (spec version
  check). FULL student 646,274 → 646,338 parameters.

### neural_racer_train

- **Cornering cap per race** (`Level.a_lat_cap`, `learn.A_LAT_RANGE` 2–5
  m/s², 40% of levels exactly 2.0, drawn last so a level's other draws are
  unchanged). The wrapper uses it (`SafetyTorch.apply_detailed(...,
  max_lateral_accel=)`); with the wrapper off, `sim.curvature_cap = True`
  still holds the speed to it (it needs only the steering). `curv_cut` (m/s
  the cap removed) is in `step()`'s output; `RewardConfig.curvature` (default
  0) charges it.
- **Privileged input 298 → 363**: the cornering cap, then a known-flag per
  track point (`PRIV_A_LAT`, `PRIV_VIS`), read by the track convolution as a
  fifth channel. `model.load_expanded` loads older checkpoints with zero
  weights on the new inputs (same outputs); `RunningNorm.set_columns`.
- **Teacher speed squash** `model.capped_squash`: speed spans 0 to
  min(v_cap, curvature cap at the steering the car will apply after the slew
  limit), so the cap never cuts it. Checkpoints carry `'squash'`
  (`teacher_squash(mode)`); absent means the old plain squash.
- **teacher.py**: `--cap-mode squash|penalty|off`, `--cap-penalty`,
  `--a-lat-range`, `--init-from RUN` (best actor + last critic, widened),
  `--init-std`, `--ema` (averaged actor, 0.99/update), `--anneal-frac/-kl`
  (KL target shrinks over the last 20%), `--mask-share` (item 2.3: hide the
  track points the LiDAR cannot see from the actor in that share of
  episodes' reactive-mode ticks; `sim.track_visibility`, `lidar.line_of_sight`,
  `model.mask_track`), `--rerace-top/-starts` (top 5 snapshots raced again on
  1000 fresh races on the GPU at the end; that sets best.pt), fused Adam.
  `--eval-starts` 4 → 8 plus `--eval-fresh` 8, `--eval-threads` 4 → 7.
  On SIGUSR1 the held-out race worker is terminated, not awaited: Python's
  exit waits for a running work item, which could outlast the 10-minute
  warning, end the job in TIMEOUT and cancel its chained segment. One race
  of the whole plan (560 races at the 2.0 cap) takes the CPU worker about
  14 min on Fir, so while training runs the worker races the main entry
  (and the masked one) only; when training ends, a race still on the CPU is
  dropped and the last snapshot races the whole plan on the GPU (before,
  the GPU sat idle up to ~30 min waiting on two CPU races).
  `teacher_eval.evaluate_file(device=)`. best.pt is written with
  `common.save_checkpoint` (the smoke run with `--eval-every 10` crashed:
  its first result came in before any save had made `checkpoints/`).
- **ppo.Update**: the teacher's update, shared (item 1.1): separate
  optimisers and clips, critic first, actor frozen during warm-up, KL-adaptive
  step measured from each minibatch's own forward pass before its step (no
  third forward; the first minibatch's KL is zero and is skipped);
  `ppo.accumulate` and `ppo.kl_config` moved here.
- **teacher_eval.py**: 320 races (fixed + fresh starts), at the car's 2.0
  cap; Wilson 95% interval (`finish_lo/hi`), best.pt chosen on its lower
  bound; results by v_cap bin and track; lateral-acceleration percentiles;
  curvature-cut share; `ref_ratio`, race time over the minimum-curvature
  line's at the same caps (`raceline_ref.py`, gate G5); students with the
  whole wrapper (`races(wrapper=True, episode_type=, mask=)`).
- **student_eval.py** (new): a student raced as the car runs it, per episode
  type, with the teacher and scripted gap_follow as references, on the
  procedural held-out tracks and the four real course maps.
- **realmaps.py** (new, step 3c): the real maps as pool tracks: resampled to
  5 cm (`circular_obs` is 1.3 cm), cropped to the free space, widths measured
  along the centreline normals; obstacles stay in the grid. On a real map a
  start that puts a car within 18 cm of a wall is re-drawn.
- **distill.py**: labels are the teacher's own (capped) commands; per-type
  held-out eval every `--eval-every` updates and `best.pt` by its lower
  bound; `PolicyNet.unroll` runs the encoders over a whole rollout and steps
  only the GRU (item 1.4; `--micro` envs per backward); `--width
  full|half|wide`, `--memory gru|none`, `--latent` (training-only head on the
  teacher's encoding, item 2.5), `--teacher-mask`, `--library`,
  `--a-lat-range`; crash, intervention and cut rates logged.
- **league_learner.py**: rebuilt on `ppo.Update` with `ValueNorm`, critic
  warm-up (`--critic-warmup` 20), `--critic-from` (a teacher's critic),
  critic normaliser updates, `--actor-lr` 2e-5, batched `unroll`, and
  kickstarting (`--teacher`, `--kickstart`, `--kickstart-frac`, item 1.7).
- **LiDAR as one Triton kernel** (`gpu_sim/lidar_triton.py`, item 1.3): each
  beam marches in registers and a block stops when no beam moved; car boxes
  in the same launch. `lidar.cast` uses it on a GPU with Triton
  (`NR_LIDAR=torch` forces the PyTorch cast, now `lidar.cast_reference`).
  Parity test `tests/test_lidar_triton.py` (GPU only). Not yet measured.
- **Run records** (item 1.8): every stage appends `run.jsonl` (args, commit,
  hash of the uncommitted changes) and writes `code-<job>.tar.gz` of them.
- **run.sh**: `teacher4 [squash|penalty|masked] [seeds]`, `distill-sweep`.
  `fir_train.sbatch`: no CPU referee workers for distill.
- Tests: `test_distill.py`, `test_league.py`, `test_lidar_triton.py`, and
  teacher v4 tests in `test_teacher.py`. On Fir CPU nodes everything passes
  except the known F1TENTH Gym imports (libgio).

## 2026-10-27 — neural_racer teacher-v3: a training regime that can finish 3 laps

The teacher-v1/v2 runs never produced a policy as good as the scripted driver
they started from (held-out: 65–98% of races crashed, 0–15% finished).
Measured causes: a behaviour-cloned start that crashed every race; a reward
and discount (5 s horizon) that tolerated a crash every ~1000 ticks over a
3600-tick race; 66% of training on 10 of 304 tracks; a teacher input missing
the commands in its actuator delay lines; obstacles it could not see; and
92% of each tick spent on a LiDAR cast nothing read. In a 30-minute H100
smoke run (2.6e8 steps) the held-out finish rate rose from 64% after the
warm start to 91% (crashes 36% → 9%); the scripted driver it clones finishes
71%. Not yet run at full scale.

### neural_racer_train

- **Sim without sensors** (`BatchedRaceSim(sensors=False)`): no LiDAR, IMU or
  observation history, for the teacher, which reads only `privileged()`. The
  cast was 92% of a tick: 17.6k → 232k steps/s at 8192 envs, 672k at 32768
  (H100). A reset now casts only the envs it resets (it re-cast all of them,
  218 ms a tick). The safety wrapper refuses to run without sensors.
- **Privileged input, 286 → 298 values** (`PRIV_DIM`; old checkpoints do not
  load): the commands still in the steering (4) and speed (3) delay lines —
  without them a memoryless teacher's state is not Markov — and 5 unobserved
  car parameters (steering gain, cornering stiffnesses, servo gain, yaw
  inertia). `PRIV_BASE` keeps the first 286 where the scripted drivers read them.
- **TeacherNet / Critic**: a 1-D convolution along the 64-point track preview,
  an MLP on the running-normalised rest (`RunningNorm`, saved in the
  checkpoint), LayerNorm fusion; the policy's last layer starts 100× small;
  the time-left input is masked (the episode limit is bootstrapped, a 3-lap
  race has none).
- **teacher.py**: DAgger warm start on scripted pursuit, loss on the squashed
  commands (behaviour cloning had left a clone that crashed every race);
  separate actor and critic optimisers and gradient clips (one joint clip
  shrank the actor's step to 20–51%); normalised returns (`ppo.ValueNorm`);
  KL-adaptive actor step from 2e-5; γ 0.995 → 0.998; wall cost 25 + 15(v/4)²
  (was 5 + 15(v/4)²), stall 40; 180 s training episodes. New flags:
  `--library`, `--gamma`, `--lam`, `--actor-lr`, `--wall`, `--wall-min`,
  `--max-seconds`, `--eval-every`, `--eval-starts`, `--eval-threads`;
  `--warm-start-steps` now 1e7. Each PPO minibatch is computed in pieces of
  at most 131k samples (`teacher.accumulate`, the same gradient): at 32768
  envs a whole minibatch, up to 524k samples, ran the H100's allocator out.
- **Held-out race during training** (`teacher_eval.py`, also a CLI): every
  `--eval-every` updates a CPU process races a snapshot on the 20 held-out
  tracks (obstacles removed), up to 300 s, and logs finish rate, finish rate
  where 3 laps are feasible, crash hazard per tick, laps and speed as
  `kind: eval` lines. The best is `checkpoints/best.pt`; distillation starts
  from it.
- **GAE** (`ppo.gae(rewards, values, next_values, terminal, done, ...)`): a
  step cut by the time limit bootstraps from the state it reached, not the
  next episode's first state. League learner updated to match, and its
  policy and critic gradients are clipped apart.
- **PLR**: scores are credited step by step to the level actually played
  (they went to whatever level the env held after the rollout, and episodes
  shorter than a rollout were never scored); temperature 0.3 → 1 (0.3 gave
  the top level 87% of score-weighted draws); `Runner(level_cap=...)` limits
  envs per level (default 1 per 1024); the sampling distribution is cached
  for 16 draws and a full buffer evicts its bottom 5% at once.
- **Tracks**: training tracks have no obstacles until after the league (team
  decision); `learn.TRAIN_TRACKS`. New `tracklib.py`: a pre-generated library
  (8192 tracks in 100 s on 32 cores) that fills the GPU pool at start-up (a
  run had reached only 304 tracks in 20 minutes). Pool eviction is one pass
  over the levels instead of one per slot.
- **Scripted pursuit** also brakes for bends up to 16 m ahead, and caps its
  bend acceleration at 60% of μg; it never drives faster than before
  (held-out, 160 races: 70.6% finished against 69.4%).
- `Runner.refresh_tracks` keeps 2 requests per worker in flight, and does
  nothing without workers (it generated tracks in-process every update).

### slurm

- New `fir_tracklib.sbatch` and `fir_run.sh tracklib`; `fir.conf` gains
  `NR_TEACHER_ENVS=32768` and `NR_TRACKLIB`, and `NR_TEACHER_RUN=teacher-v3`;
  the teacher stage starts no archive referees; `distill` starts from `best.pt`.
- `fir_train.sbatch` sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  (the PPO batch changes size every update, since only replayed levels train).

### Tests

- New `test_teacher.py` (23); GAE tests updated and a timeout-bootstrap test
  added in `test_training_units.py`.
- New `clean_sys_path` fixture (`conftest.py`). The gym modules' skip marks
  put `.sim/` (its own scipy and numpy) first on `sys.path` at collection,
  and spawned workers inherit it: the library test's pool never returned,
  and the full suite ran into its time limit.
- Full suite on Fir (CPU): 112 passed. The 9 tests that import the F1TENTH
  Gym fail on Fir's `libgio` (`g_source_set_static_name`), as before.

## 2026-10-24 — Gyro odometry for SLAM; every mapping run recorded and explained

For the first IMU mapping test (A/B, `docs/launchstart.md#testing-the-imu-in-mapping-ab`).
Built and checked off the car: **not yet driven on.**

### imu_bridge

- **Gyro odometry: new `/odom/imu`** (`nav_msgs/Odometry`, frame `odom_imu`),
  always published by `vehicle_state_node`. `vesc_to_odom`'s distance (its
  own pose change along its previous heading, so identical to `/odom`'s) with
  the heading from a gyro: BNO085 first, then the VESC's, then -- when neither
  is usable, a step spans a gyro gap, or the gyros disagree -- `vesc_to_odom`'s
  own heading change, which makes the fallback exactly `/odom`.
- **New `publish_odom_tf`** (default **false**): also publish `odom_imu ->
  odom`, chosen so `odom_imu -> base_link` is the gyro pose. `vesc_to_odom`
  keeps `odom -> base_link`; bringup is untouched. **Interlock:** the node
  watches `/tf` and stops that transform for good the moment anything else
  gives `odom` a parent (logged every 5 s).
- New knobs in `vehicle_state.yaml`: `heading_sources` (`['bno', 'vesc']`;
  `['none']` = steering model), `max_gyro_gap`, `max_heading_extrapolation`,
  `imu_odom_topic`, `imu_odom_frame`, `publish_odom_tf`, `odom_frame`,
  `base_frame`. `check_frames` refuses an `imu_odom_frame` that would loop or
  collide with `map`.
- `vehicle_state_launch.py publish_odom_tf:=`.
- Review fixes: `steering_topic:=/odom` is now refused; detecting a running
  BNO bridge no longer mistakes an editor with the file open for one;
  `bno:=1` means true. `wrap_angle(-pi)` returned `-pi` against its own
  `(-pi, pi]` contract (caught by a new test).
- Tests: `test_gyro_odometry.py` (46). 292 in the package; every new test was
  seen to fail against deliberately broken code.
- Checked on an isolated domain: a fake car on a circle through the real node
  (tf2 chain = `/odom/imu` to 1e-15, heading from the gyro not the model, the
  interlock stopped the transform one message after a fake `map -> odom`); and
  parked with the real LiDAR, real BNO and real `slam_toolbox` on `odom_imu`
  (map built, `map -> base_link` resolved, 0.009 deg of gyro drift in 45 s).

### racerbot_launch

- **`map_course gyro_odom:=true`** (default false): SLAM's `odom_frame`
  becomes `odom_imu` and `vehicle_state` publishes the transform, always
  together (`gyro_odom_settings`); refuses with `imu:=false`. `gap_follow`
  and the lap counter are unchanged (lap turning stays in `odom`).
- **`map_course` records a bag by default** (`record_bag:=true`).
- `slam_launch.py`: new `slam_odom_frame:=` (default `odom`).

### race_diagnostics

- `record_run.py` bags `/odom/imu`, `/pose`, `/slam_toolbox/graph_visualization`,
  `/map_metadata`, `/rosout`, `/diagnostics`, `/drive_intent`, `/initialpose`.
- **New `<run>/metadata/`**: git HEAD, branch, status and full uncommitted
  diff of `racerbot-ws` and `~/imu-calibration`; copies of `car.yaml` and the
  calibration JSONs; and **`snapshot_params`** (new executable, read-only:
  `list_parameters`/`get_parameters` only) writing every node's parameters
  15 s in, plus any node that appears in the next 2 minutes. A metadata
  failure is written down, never stops a run.

### tools/run_analysis

- **New `imu_report.py <run>`**: turn rate from `/odom` (steering model), the
  VESC gyro, `/odom/imu` and LiDAR scan matching against the BNO gyro; steering
  lag and turning gain fitted against the gyro and against scan matching;
  heading correction SLAM applied on top of `/odom` and `/odom/imu`. Ran on the
  car on a real (parked) bag in 13 s including scan matching. Tests:
  `tests/test_imu_report.py` (the bag round trip runs sourced).

### imu-calibration (`~/imu-calibration`, not committed)

- BNO085 `rpy_deg` `[90, 0, 90]` -> `[92.44, 5.22, 90.11]`: roll and pitch
  from 10 s parked on a level floor (gravity in car axes was -0.41 / -0.89 /
  9.71 m/s^2, now 0.00 / 0.00 / 9.76). Yaw not measured: forward-push check is
  step 1 of the A/B.

## 2026-10-24 — imu_bridge: fused speed and measured steering; IMU in `map_course`

### imu_bridge

- **New node `vehicle_state_node`**, support code (publishes no drive topic):
  - **`/odom/fused`** (`nav_msgs/Odometry`): `/odom` with the forward speed
    replaced by a Kalman fusion of the wheel speed and the BNO085's forward
    accelerometer (lever arm corrected for its front-axle mount). Wheel speeds
    the filter can't explain are skipped as slip for up to 0.5 s. **Always
    within `max_divergence` (0.3 m/s) of the wheel speed**, and exactly the
    wheel speed when the accelerometer is missing, stale or mounted wrong.
    One message per `/odom`, so it goes silent with `/odom`.
  - **`/steering/measured`** (`sensor_msgs/JointState`, joint `steering`,
    rad, + left, 50 Hz): `atan(wheelbase × yaw rate / speed)`, yaw rate the
    average of both gyros. `NaN` below 0.3 m/s, with no usable gyro, or when
    the gyros steadily disagree by > 0.15 rad/s.
  - Math in `imu_bridge/vehicle_state.py` (no rclpy); every knob in
    `config/vehicle_state.yaml`. `check_output_topics` refuses output names
    that look like drive topics or equal `/odom`.
- **New `vehicle_state_launch.py`**: car 1's whole IMU side in one launch.
  It starts imu-calibration's `tools/bno_i2c_bridge.py --pty /tmp/bno085
  --no-dyncal` (car 1's BNO085 is on the Jetson's I2C bus 7), or uses one
  already running, then `imu_bridge_node` and `vehicle_state_node`.
  `bno:=false` for the VESC IMU only.
- `imu_bridge_node` **restarts its BNO driver after 2 s without samples**:
  racer_imu's driver gave up for good when the bridge's port didn't exist yet.
- Tests: `test_vehicle_state.py` and `test_vehicle_state_config.py` (232 in
  the package, no ROS needed). Every new test was seen to fail against
  deliberately broken code: 40 breaks in the estimator, 12 in the wiring, the
  topic check and the YAML.
- Checked: a static topic check on an isolated domain (fake car on a circle:
  speed exact, angle = closed form), and the launch against the real BNO085
  on the bench (250 Hz, gravity check passed). **Not yet driven on.** The BNO
  read −0.39 m/s² forward and −0.89 sideways at rest (about 2° pitch, 5° roll,
  from the bench or the nominal mount angles): redo its mount step on a level
  floor.

### racerbot_launch

- **`map_course` runs the IMU side by default** (`imu:=true`). New
  **`fused_speed:=true`** (default **false**) gives `gap_follow`
  `odom_topic:=/odom/fused` for TTC braking, its acceleration ramp and its
  stale-odometry stop; it refuses to start with `imu:=false`. SLAM, the
  transforms and `auto_map_race_node` stay on `/odom`. The `map_course`
  wrapper adds `imu:=false` in the simulator. Tests: `test_map_course_imu.py`
  (needs ROS sourced) and two in `test_map_course.py`.
- `test_car1_hardware_config.py`: four new checks guard car 1's I2C BNO085
  setup against a port from car 2 (`docs/car1-vs-car2.md` has the new row).

### race_diagnostics

- `record_run.py` bags `/odom/fused` and `/steering/measured` too.

### imu-calibration (`~/imu-calibration`, separate repo, not committed)

- `config/car.yaml`: car 1's `wheelbase` 0.36 and `speed_to_erpm_gain`
  −4040; the BNO's `position` x 0.18 (front axle). The VESC entry is marked
  as still car 2's. Two of that repo's tests,
  `test_rezero_at_stops_beats_fixed_bias` and
  `test_lidar_heading_calibrates_gyro_without_stopping`, fail with car 1's
  `car.yaml` and pass with the committed one. The first was bisected to the
  BNO `rpy_deg: [90, 0, 90]` line from 2026-10-23, not to these edits. Not
  investigated further.

## 2026-10-23 — Car 1: third port from car 2 (to `b3ad3b9`)

### race_diagnostics

- `record_run.py` bags **`/sensors/imu/raw`** (the VESC's IMU) by default, so
  recorded runs can show what the gyro would do for odometry and SLAM. Nothing
  drives on it. Ported from car 2's `b3ad3b9`; no car 1 difference involved.

## 2026-10-23 — gap_follow: `max_speed:=` for one run; car 1 + car 2 test plan

### gap_follow

- **New launch argument `max_speed:=<m/s>`** on `gap_follow_launch.py`, e.g.
  `max_speed:=1.0` for a first floor run. It only lowers the speed: above
  `gap_follow.yaml`'s `max_speed`, zero, negative, `nan`/`inf` or not a number
  is refused and nothing starts. `min_speed`, `corner_speed` and
  `corner_speed_wide` are lowered to it when they are above it, because the
  node refuses to start otherwise. Empty (the default) changes nothing. Logic
  in `gap_follow/launch_speed.py` (no ROS), tests in
  `test/test_launch_speed.py`, including a real node that never commands above
  the cap and still stops on LB release. Checked with a real launch on a test
  domain: parameters as expected, out-of-range values refused.
- `docs/launchstart.md`: a step-by-step first run with car 1 and car 2
  together (separate ROS graphs, wheels off, then low speed in open space).

### Simulator, car 1's config with opponent awareness (2026-10-23)

`tools/f1tenth_sim/eval_gap_multi.py`, 2 cars, 5 tracks, 4 seeds each way:

- Leader 1.8, follower 3.5 m/s: **40/40, no wall or car-car contact**
  (closest 2.27 m), the same as car 2's published result.
- Leader 0.8, follower 1.0 m/s, corner and minimum speeds lowered as
  `max_speed:=` does: **no contact anywhere except `narrow_1p2`** (1.2 m wide),
  where 1–3 of 8 runs each way touched a wall. Not opponent awareness: with it
  off the rate is the same, and one car alone touched once in 8 reverse runs.
  Car 2's 0.349 rad steering limit does a little better there (1 of 8). The
  big tracks' "unfinished" runs were the 90 s limit at 0.8 m/s, not the car.
  **Open question:** gap_follow at walking pace in a 1.2 m corridor.

## 2026-10-22 — Car 1: second port from car 2 (to `38c35a0`)

A per-file three-way merge of car 2's `38c35a0` onto car 1, against car 2's
`7ec6ae6` (what the 2026-10-11 port took). The last three car 2 commits
(2026-10-23) make `map_course` refuse to start without `bringup_launch.py`
already running, instead of starting a second copy of the drivers that
fights the first over the VESC and LiDAR. The three sections directly below
this one were written in car 2's workspace; "this car" there means car 2.
Explicitly requested; car 2 is still not an upstream.

Taken from car 2: gap_follow's corridor rework and opponent awareness (off by
default), the pluggable steering laws (pure pursuit stays the default),
`map_course` one-command mapping and saved-map racing without RViz, the
calibration wizard's circle step and overrides, simulator track import and
`tools/run_analysis`, and their docs and tests.

**Not** taken — car 2's steering calibration (commit `d7456e7`):
`vesc.yaml` (`-0.78`/`0.5904`, `servo_max 0.95`) and `joy_teleop.yaml`
(full stick 0.436) are untouched. Car 1 keeps `-0.83`/`0.366`,
`0.05`–`0.75` and full stick 0.38.

**Behaviour change — car 1's autonomy steers harder.** Car 2 raised its limits
from its own circles; car 1 now does the same from **its own** 2026-10-21
circles (0.388 / 0.401 rad reached at the 0.38 rad full stick):

- `max_steering_angle` **0.26 → 0.33** rad in `gap_follow.yaml` and
  `pure_pursuit.yaml` (minimum radius 1.35 → 1.05 m).
- `profile_max_steering_angle` **0.26 → 0.30** rad in `auto_map_race.yaml`
  (1.35 → 1.16 m), below the controller as car 2's new
  `test_steering_limits_fit_servo.py` requires.
- `odom_calibration.yaml` `planner_max_steering_rad` 0.26 → 0.33.
- `gap_follow.yaml` `arc_guard_max_correction` 0.1745 → **0.165**. It must be
  at most half of `max_steering_angle`; car 2's value is half of *its* 0.349,
  and with car 1's 0.33 `gap_follow_node` refused to start.
- **Not yet validated on car 1:** wheels off the ground, then low speed.

Test fixed, not loosened: `test_arc_guard_blocked_fallback_reaches_the_guard`
measured the gap's request under the shipped steering limit, which car 1's
0.33 clips (the scene's gap sits at 20°), and its "freest moves" precondition
was met by steering slew from the previous command, not by the guard. It now
uses its own 0.5 rad limit and lets the command settle before reading each
mode; seen to fail with either fallback branch broken.

Known gaps:

- Both simulators' vehicle models now carry **car 2's** measured servo map and
  full-lock limits (+0.509 / −0.448 rad); before this port they carried car 2's
  older ones. They have never modelled car 1's steering. Car 1's controllers
  cap at 0.33 rad, inside both, so sim results are not limited by it, but a
  sim run is not evidence about car 1's linkage.
- Car 2's calibration-wizard and steering-curve tests use car 2's recorded
  sessions as fixtures; they test the maths, not car 1's values.

## 2026-10-22 — gap_follow opponent awareness; one-command mapping; launch cheat sheet

All of it is **simulator-validated only**. Bring it up wheels-off first, then at
low speed ([docs/gap-follow.md](docs/gap-follow.md#validation)).

### gap_follow

- **Opponent awareness** (`enable_opponent_awareness`, **off by default**;
  `gap_follow_launch.py opponent_awareness:=true`). New
  `gap_follow/opponent_tracking.py`, no ROS needed:
  - it finds free-standing, car-sized clusters and tracks them against this
    car's dead-reckoned motion (the steering modelled as the servo follows it);
  - behind a moving car it caps speed to `(gap − 0.5 m) / 1.0 s`;
  - it passes only on a straight, and only with 0.15 m of extra room around
    where the other car will be over the next 0.8 s.

  Safety checks still read the real scan. With it off, the commands are
  identical. New `opponent_*` parameters are all commented in
  `config/gap_follow.yaml`.
- Simulator, leader 1.8 m/s and the others 3.5 m/s, both directions:

  | | Layer on | Layer off |
  |---|---|---|
  | Two cars | 40/40 | 34/40 |
  | Three cars | 24/24 | 15/24 |
  | One car | 111/132 | 108/132 |

  There was no contact in any run with it on. It follows rather than
  overtakes on every track tried.

### racerbot_launch / pure_pursuit

- **`ros2 run racerbot_launch map_course`** (with `bringup_launch.py` already
  running; it never starts the drivers and refuses without them): `gap_follow` drives 3 laps, then
  the run saves the map, pose graph, racing line and `start_pose.yaml`, shuts
  down and prints the race commands. It is `auto_map_race_node` in the new
  `finish_after_mapping` mode, with `map_course_launch.py`.
- **`race_launch.py run_dir:=<dir> [steering_law:=...]`**: races a saved run.
  The new `seed_start_pose` node seeds the particle filter where mapping
  stopped the car, so no RViz is needed. `pure_pursuit` only starts once the
  filter has taken that pose.
- `localize_run_launch.py standalone:=true` publishes the filter's TF and
  `/map` when no SLAM is running.
- `tools/racerbot_sim/run_map_course_validation.py` runs the whole flow end to
  end over the simulator. On `indoor_wide`, mapping passed every run. Pure
  pursuit, LQR and MPC each raced 128–142 m with no contact. Stanley once
  swerved around a false "opponent" (a scan/map mismatch) and brushed a wall;
  that is the existing `pure_pursuit` overtake logic.
- `localize_run_launch.py standalone:=true` starts the particle filter 4 s
  after `map_server`. Started together, the filter got an empty map and
  crashed (1 race start in 8 in simulation).

### docs

- New [docs/launchstart.md](docs/launchstart.md): every mode as one
  copy-paste command.

## 2026-10-19 — gap_follow corridor rework; Stanley, LQR and MPC made usable

Both changes are **simulator-validated only**. Bring them up wheels-off first,
then at low speed ([docs/gap-follow.md](docs/gap-follow.md#validation),
[docs/steering-laws.md](docs/steering-laws.md#safety)).

### gap_follow

- **Swept-path clearance** (`enable_swept_clearance`, on): the braking speed
  ceiling uses how far the body can drive along its arc, not the nearest
  return in the ±30° cone (which read hallway side walls as obstacles ahead).
- **Arc guard** (`enable_arc_guard`, on): if the requested steering would
  sweep the body into something, use the nearest clear steering angle, at
  most `arc_guard_max_correction` (0.1745 rad) away.
- **Corner setup** (`enable_corner_setup`, on): move toward the outside wall
  before a corridor turn, so the car turns in late.
- **`max_lateral_accel` 1.0 → 2.0 m/s²** — simulator only, not yet
  validated on the floor.
- New parameters (all documented in `config/gap_follow.yaml`); startup
  refuses arc-guard/corner-setup authority above half of `max_steering_angle`.
- Simulator, 5 indoor layouts × 2 directions × 6 seeds: 36/60 passed with 24
  wall-contact runs → 60/60, none; `asb_10000` lap 24.6 → 18.9 s
  (`docs/gap-follow-sim-results.json`).
- Tools: `tools/f1tenth_sim/gap_mirror.py` (the simulator's copy of the
  per-scan pipeline, reads the YAML, held to the node to 1e-6 by
  `test_gap_mirror_parity.py`), `sweep_gap.py`, and `trace_gap.py` rebuilt on
  it. `run_validation.py`'s stale `gap_command` was replaced by it.

### pure_pursuit steering laws

- **Stanley formula corrected** (a change to spec 02): the separate
  `+atan(L·κ)` feedforward is gone (the front-axle heading error already
  carries it, so it was counted twice and held the car inside every bend),
  and the front axle is steered to the offset a rear-axle racing line implies.
  Spec tests 6/7 now check the kinematic answer, `atan(L/R)` on a circle.
- **New parameters:** `stanley.delay_compensation`, `lqr.delay_compensation`,
  `mpc.delay_compensation` (predict the pose ~0.1 s ahead before steering),
  `stanley.steering_time_constant`, `lqr.steering_time_constant` (output
  low-pass against localisation jitter).
- **Retuned defaults:** Stanley gain 5.0, softening 1.0; MPC `q_lateral` 4.0;
  plus the values above. Dataclasses, YAML and node defaults all agree
  (tested).
- MPC's returned plan is projected onto the exact angle and rate limits.
- The node logs `steering law: <name> <params>` at startup.
- Review fixes: MPC miss latch read a non-existent attribute; live lookahead
  and steering-rate tunes did not reach the law; `/drive_intent` could run the
  MPC solver; MPC's condensed dynamics were wrong at varying speed; the bound
  check was tighter than OSQP's accuracy.
- `tools/f1tenth_sim/run_validation.py` reads its limits from
  `pure_pursuit.yaml` (it had a stale 0.26 rad steering limit);
  `tools/f1tenth_sim/law_sweep.py` runs comparisons in parallel.
- **osqp 1.1.3 installed** on the car (`pip install --user
  --break-system-packages osqp==1.1.3`). MPC benchmark G0 passed: p99.9
  3.5 ms against an 8 ms deadline under load
  (`docs/mpc-benchmark-2026-10-19.json`).
- **Judging (seeds 12345–12349): Stanley, LQR and MPC all PROMOTE**, 15/15
  runs each, no contact, every margin under 5%
  (`docs/tracking-law-comparison.json`). This supersedes the REJECTs in the
  entry below. Pure pursuit stays the default.

## 2026-10-19 — Pluggable steering laws for the race controller

`pure_pursuit_node` can now steer with a choice of tracking law —
`pure_pursuit` (today's, the default), `stanley`, `lqr`, or `mpc` —
selected at launch with `steering_law`, without changing anything else the
node does. Every safety mechanism (LB deadman, pose-stale/frozen and
cross-track watchdogs, off-line recovery, opponent overtaking, the reactive
LiDAR net, steering slew and curvature speed cap) applies to every law,
because the law is swapped in *between* them rather than beside them. The
laws output steering only; speed stays with the profile and the node's caps.

### Behaviour changes

- **New parameter `steering_law`** (default `pure_pursuit`, launch-time only)
  plus per-law `stanley.*`, `lqr.*`, `mpc.*` gains and
  `tracking_curvature_smoothing`. All are refused at runtime (the law is built
  once at startup), and none are live-tunable from the dashboard. The launch
  file passes `steering_law:=...` through.
- **New failure states** in the decision log and `race_diagnostics`:
  `steering_law_failed` (a law that cannot produce a trustworthy command stops
  the car), `mpc_fallback` (a single-tick pure-pursuit fallback on an MPC miss),
  and `mpc_miss_latched` (MPC gave up after `max_consecutive_misses`; cleared
  by releasing LB).
- **`/drive_intent`** re-runs the *active* law (replays the MPC solver's
  predicted sequence rather than re-solving).

### Code

- `pure_pursuit/racing_math.py`: `estimate_signed_path_curvature`,
  `project_onto_path`, `interpolate_along` (all ROS-free, all unit-tested).
- `pure_pursuit/tracking_laws.py` (new, ROS-free): the `TrackingLaw` seam,
  `PurePursuitLaw` (today's steering moved unchanged, verified by a
  change-detector test against the pre-refactor computation), `StanleyLaw`,
  `LqrLaw` + `LqrGainTable`, `MpcLaw` + `MissPolicy`, and `make_law`.
- `tools/f1tenth_sim/run_validation.py`: `PathFollower` steers through the
  pluggable law; `--steering-law`, `--law-param`, and `--compare` run a
  candidate against pure pursuit over the same tracks × seeds. Refactor proof:
  identical `pure_solo` metrics before/after. The promotion rule is pure
  Python in `tools/f1tenth_sim/law_compare.py`, tested in
  `tools/f1tenth_sim/tests/` (`python3 -m pytest tools/f1tenth_sim/tests/ -v`).
- `tools/mpc_benchmark/benchmark_mpc.py` (new): the spec 04 G0 solve-time
  benchmark, ready for the osqp install decision.

### Judging results (spec 01)

Both `stanley` and `lqr` **REJECT** at the shipped starting gains on the
judging seeds (`docs/tracking-law-comparison.json`, command and analysis in
`docs/sim-validation.md`). Neither ships; pure pursuit remains the default and
the baseline. MPC is gated on spec 04 G0 (the osqp install decision and a
Jetson benchmark) before it can be judged. Specs:
`docs/superpowers/specs/2026-09-27-tracking-*.md`.

## 2026-10-11 — Car 1: port car 2's workspace, dashboard as `rb1`

Every section below this one was written in **car 2's** workspace
(sfu-racerbot/Racerbot-Car-2-Workspace) and arrived here in one port: a
three-way merge of car 2's `7ec6ae6` onto car 1, against car 2's `e16bda6`
(the last commit the two repos shared in content). Car 2 is still not an
upstream; this was a one-off, explicitly requested sync. "This car" in the
entries below means car 2.

Car 1's hardware differences were kept:

- **LiDAR:** USB-serial Hokuyo UST-30LX at `/dev/sensors/hokuyo`
  (`sensors.yaml` unchanged), not car 2's Ethernet UST-10LX.
- **Camera:** UVC webcam on `/dev/video0`; **no RealSense**. The
  `realsense-ros` submodule, `realsense_camera_launch.py` and
  `docs/realsense-camera.md` were not brought over. New
  `racerbot_launch/launch/webcam_launch.py` starts `usb_cam_stream` with car
  1's settings; `dashboard_test_launch.py` uses it.
- **VESC:** car 1's `3650.0` speed gain, `-1.1135`/`0.4` servo mapping and
  `0.05`–`0.75` servo range. Committed just before the port.
- **Odometry sign — behaviour change in how, not what.** Car 1 used to
  negate speed inside `vesc_to_odom.cpp`; car 2 negates
  `vesc_to_odom_node`'s `speed_to_erpm_gain` in `vesc.yaml` instead. The
  merge would have kept *both*, cancelling out and running `/odom`
  backwards. Now `vesc_to_odom.cpp` is upstream again and
  `vesc_to_odom_node` gets `speed_to_erpm_gain: -3650.0`. Same arithmetic
  as before. **Verify on the car:** drive forward and check
  `/odom` `twist.linear.x` is positive.

Car 1 takes on car 2's geometry (team decision, not measured on car 1):

- `wheelbase` **0.25 → 0.36** (`vesc.yaml`, controllers), LiDAR
  `base_link`→`laser` **0.27 → 0.26** in every bringup.
  **TODO(verify on car 1):** tape-measure both;
  `docs/hardware-reference.md#physical-dimensions-used-in-config` lists
  every file to change together.

Web dashboard:

- `src/web_dashboard` and `src/usb_cam_stream` are now the
  `src/web_dashboards` submodule (sfu-racerbot/web-dashboards); the old
  in-tree packages are gone.
- Car 1 is **`rb1`** on https://dashboard.sfuracerbot.ca. Its settings:
  `racerbot_launch/config/web_dashboard_rb1.yaml` (renamed from car 2's
  `_rb2`, camera switched to `device: /dev/video0`), checked by
  `racerbot_launch/test/test_web_dashboard_rb1_config.py`.

Known gaps:

- `racerbot_doctor`'s LiDAR and camera checks still describe car 2's
  hardware and will report false failures on car 1 (flagged in its README).
- `odom_calibration.yaml`'s fallbacks were set to car 1's values; any
  calibration from car 2's sessions does not apply to car 1.
- Racing lines and odom calibration from before this port were made under
  the old 0.25 m wheelbase and 0.27 m LiDAR offset — re-record/re-run.

## 2026-09-27 — The dashboard moves to sfu-racerbot/web-dashboards

`web_dashboard` and `usb_cam_stream` moved, with their git history
(`git subtree split` here, `git subtree add` there), into
**sfu-racerbot/web-dashboards** under `car/ros/`. That repo now holds the
site and everything a team needs on its car; this workspace consumes it as
the git submodule **`src/web_dashboards`** and keeps only car 2's settings.
Package names, `ros2 run` and `ros2 launch` commands are unchanged.

### Behaviour changes

- **Start the dashboard with `ros2 launch racerbot_launch dashboard_launch.py`.**
  The package now ships generic defaults, so a bare
  `ros2 launch web_dashboard web_dashboard_launch.py` allows **no** site
  (the site shows CAR OFFLINE), tunes and stops nothing, and lists no maps.
  Car 2's values are in the new **`racerbot_launch/config/web_dashboard_rb2.yaml`**,
  passed as `car_config:=`: `allowed_origins`, `laser_offset_x: 0.26`,
  `tuning_nodes`/`tuning_config_files`, `killable_nodes`, `map_roots`, and
  the RealSense stream's `image_topic`/`passthrough`. Values unchanged from
  before. `realsense_camera_launch.py`, `dashboard_test_launch.py` and
  `racerbot_sim`'s `dashboard:=true` use it.
- **The car serves no pages.** The old frontend (`web/`) and its tests were
  deleted — the site's `apps/simple` replaced them. `serve_static` defaults
  to false and is ignored (with a warning) if set; `http://<car-ip>:8080/`
  answers 404. Open https://dashboard.sfuracerbot.ca.
- **`drive_intent` is now optional for the dashboard** (it stays in this
  workspace): without it, nothing subscribes to `/drive_intent` and the
  node logs `drive intent: OFF` once.
- **`remote_check` takes `--site https://dashboard.sfuracerbot.ca --car rb2`.**
- **foxglove_bridge's config, launch and safety test moved into
  `web_dashboard`** (`ros2 launch web_dashboard foxglove_bridge_launch.py`);
  client publishing stays **off**. `racerbot_launch/launch/foxglove_bridge_launch.py`
  is now a **temporary forwarder**, kept with `tools/systemd/` only so the
  installed `foxglove-bridge` unit survives a reboot; both go once the unit is
  reinstalled from `src/web_dashboards/car/systemd/` (`docs/web-dashboard.md`).
- `usb_cam_stream`'s `realsense_stream_launch.py`/`realsense_stream.yaml`
  are gone (car-2 config, now in the rb2 YAML).
- `tools/web_dashboard/` moved to the submodule's `car/tools/`.
- Later the same day, after `foxglove-bridge.service` was reinstalled from
  the submodule and the site's `/rb2/check` passed every hop: the forwarder
  `racerbot_launch/launch/foxglove_bridge_launch.py` and `tools/systemd/`
  were **removed**. `ros2 launch web_dashboard foxglove_bridge_launch.py` is
  the bridge's launch. `racerbot_launch`'s test now fails if a bridge launch
  or config reappears in it.

### Tests

- New `racerbot_launch/test/test_web_dashboard_rb2_config.py`: every rb2 key
  is a parameter the node declares; `laser_offset_x` equals every bringup's
  `base_link`→`laser` x; the site origin equals the site's own
  `PUBLIC_ORIGIN`; no stoppable process or map folder would be silently
  dropped. Replaces `test_foxglove_bridge_config.py`, which moved with the
  bridge config.
- `gap_follow`/`pure_pursuit` `test_*_live_tuning.py`: the dashboard's
  `tuning.py` is imported from the submodule, and each node's **live**
  config now round-trips through its YAML writer — this workspace's half of
  the `live_tunable_spec` contract (the dashboard repo round-trips a snapshot).
- **Run ROS tests on an isolated domain** (`ROS_DOMAIN_ID=79
  ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST`, now in `CLAUDE.md`):
  `gap_follow`'s node tests published to the real `/drive`. Found the hard way
  while taking baseline counts for this move, with bringup up and the
  battery detached.
- **`gap_follow`'s node tests no longer publish to the real `/drive`.** Every
  `rclpy.init` in `src/gap_follow/test/` ends with `TEST_ONLY_DRIVE`
  (`-p drive_topic:=/test_only/drive`), as `pure_pursuit`'s already did.
  It must be **last**: measured on Jazzy, once any `-p` precedes
  `--params-file`, the YAML's node-keyed values beat every `-p`, so a
  remap placed first left `test_packaged_config_plus_mapping_overrides_actually_starts`
  on `/drive`. `test_no_test_here_builds_a_node_on_the_real_drive` checks
  every call (by AST), each fixture asserts its publisher's topic, and
  `test_test_only_drive_must_come_after_the_params_file` pins the ordering
  rule. The isolated domain stays: the tests still publish latched topics.

## 2026-09-27 (later) — Diagnosing the remote site from the car

The site could not connect although every service on the car was healthy.
Checked from the car: **rb2-bridge-origin.sfuracerbot.ca had no DNS record**
(NXDOMAIN on 1.1.1.1) although the tunnel routed it. Every hostname also
shows non-browser clients a Cloudflare bot challenge; this entry first
blamed that for the failure, which was **wrong** — the site's Worker got
HTTP 101 from the dashboard through the same zone. The site's check also
reports the bridge as 502 because it probes it with a plain request, which
foxglove_bridge hangs up on; a WebSocket upgrade with its subprotocol gets
101.

### web_dashboard

- **New `ros2 run web_dashboard remote_check`** (`remote_check.py`): local
  services, a relay handshake as the site makes it, tunnel routes from
  cloudflared's journal, public DNS, how the Cloudflare edge answers
  (Access vs tunnel down vs *unprotected*; a challenge is a warning), and the site
  connections that reached the dashboard. Read-only; prints a fix per
  failure. Verdicts tested in `test_remote_check.py` (mutation-checked).
- **The dashboard logs every connection open and close**, all roles, with
  user, address, duration and close code. Before, direct connections were
  not logged at all.
- Needs rebuilding: new entry point (`colcon build --packages-select
  web_dashboard`).

## 2026-09-27 — foxglove_bridge: client publishing switched off (safety fix)

**foxglove_bridge 3.5.0 does not enforce `client_topic_whitelist`.** The
2026-09-26 config relied on it to allow only `/initialpose`. Tested on an
isolated copy of the bridge (ROS domain 79, its own port, this YAML,
nothing connected to the car):

- with `clientPublish` on, a client advertised `/drive`, published one
  zero-speed message, and `ros2 topic echo /drive` **received it**;
- with `clientPublish` removed from `capabilities`, the bridge answered
  `Server does not support clientPublish capability` and nothing arrived.

A `/drive` published from a browser goes straight to `ackermann_mux` and
skips every driving node's LB deadman. So:

### racerbot_launch

- `config/foxglove_bridge.yaml`: **`clientPublish` removed from
  `capabilities`**. Nothing can be published through the bridge.
  `client_topic_whitelist` stays as a second layer for a future version that
  enforces it; it is not what protects the car.
- `test/test_foxglove_bridge_config.py`: the test that asserted
  `clientPublish` was **on** now asserts it is **off**, citing the
  measurement above. Seen to fail with `clientPublish` put back.
- Cost: Lichtblick can no longer send a "2D Pose Estimate"
  (`/initialpose`). Use RViz on the car's network for that.

### Checked live on the car (bringup + RealSense + dashboard + bridge, no control layer)

- Bridge listening on `127.0.0.1:8765` only; offers 34 topics, all
  compressed image topics, **no raw `Image`**; 83 services;
  `getParameters` returns values (391 across the graph); calling
  `/foxglove_bridge/list_parameters` returned 39 names.
- dashboard-rb2.sfuracerbot.ca is retired; mentions removed.

## 2026-09-27 — Particle-filter pose is the LiDAR's, now converted to the rear axle

**Bug fixed.** `particle_filter` ray-casts each scan from the particle's own
pose (`particle_filter.py`, `sensor_model`: `queries = proposal_dist`), so
`/pf/viz/inferred_pose` is the pose of the **LiDAR**, 0.26 m ahead of the
rear-axle `base_link` everything else assumes. Nothing converted it:

- `pure_pursuit_node` (saved-map racing) believed the car was 0.26 m ahead
  of itself, and its map ray caster added `laser_offset_x` on top, placing
  the expected scan 0.52 m forward of the rear axle.
- `auto_map_race_node` recorded the line from SLAM's real `base_link`, then
  at handover republished the filter's LiDAR pose on `/slam_pose` as if it
  were `base_link` (a 0.26 m jump), and seeded the filter with a
  `base_link` pose as if it were the LiDAR's.
- `waypoint_recorder_node` recorded the LiDAR's path.
- `web_dashboard` drew the car and scan 0.26 m forward in PF mode.

Changes:
- `racing_math`: new `laser_pose_to_base_link` / `base_link_pose_to_laser`.
- `pure_pursuit_node`, `waypoint_recorder_node`: **new parameter
  `pose_frame`** (`laser` default, matching the default
  `/pf/viz/inferred_pose`; or `base_link`); unknown values refuse to start.
  `waypoint_recorder.yaml` gains `laser_offset_x/y`.
- `auto_map_race_node`: converts the PF pose before republishing and the
  seed before sending; **new parameters `laser_offset_x/y`** in
  `auto_map_race.yaml`.
- `auto_map_race_launch.py`: sets `pose_frame: base_link` with
  `pose_topic: /slam_pose`.
- `web_dashboard`: **new parameter `laser_pose_topics`**
  (`[/pf/viz/inferred_pose]`).
- Docs: `localization.md#which-point-on-the-car-the-position-means`, param
  tables, the `hardware-reference.md` offset row.

Tests: 339 `pure_pursuit` (sourced), 607 `web_dashboard`, 53
`racerbot_launch`; every new test seen failing against 11 mutations. Two
existing tests changed, flagged in the commit: the auto_map_race handover
test asserted the unconverted pose (the bug), and the recorder status test
now declares its poses as `base_link`.

Old PF-recorded lines trace the LiDAR's path: on a closed loop the same line
shifted along itself, about 3 cm outside on a 1 m-radius corner
(0.26² / 2R), so still usable.

**Needs on-car validation:**
- [ ] `auto_map_race` handover: no forward jump in `/slam_pose` at the
  switch to `particle_filter` (plot `race_diagnostics`' pose around it).
- [ ] Saved-map `race_launch.py`: dashboard car and scan overlay line up
  with the map walls; map-mode opponent detection shows no phantom or
  missed opponents near walls.

Also: implementation specs for Stanley / LQR / linear MPC steering laws in
`docs/superpowers/specs/2026-09-27-tracking-*.md` (not implemented).

## 2026-09-26 — Car side of dashboard.sfuracerbot.ca, and foxglove_bridge

The dashboard frontend is moving to sfu-racerbot/web-dashboards, served at
https://dashboard.sfuracerbot.ca through the Cloudflare Tunnel. This is the
car's half of that contract. `web_dashboard` still creates **no ROS
publisher**. Verified: 603 `web_dashboard` tests, 53 bridge-config tests
and a mutation pass over every new test; a real `dashboard_node` driven
over real sockets by relay, control and direct clients on an isolated ROS
domain; `usb_cam_stream` fetched with the tunnel's headers.

**Needs on-car validation** (none of this has been through the real tunnel):
- [x] ~~dashboard-rb2.sfuracerbot.ca still connects~~ — retired on
  2026-09-27 (it was not working), so nothing to check.
- [ ] The remote site's relay and control sockets connect through
  `rb2-dash-origin`, get `hello` first, and the relay's writes are refused.
- [ ] foxglove_bridge: Lichtblick through `rb2-bridge-origin` lists topics,
  reads/sets a parameter and calls a service. (Publishing is now off
  entirely — see 2026-09-27.)
- [ ] Camera through `rb2-cam-origin`.
- [ ] A recorded run's bag contains `/parameter_events` entries for a
  change made from the dashboard and one made from Lichtblick.

### web_dashboard — who may connect, and as what

- **Behavior change: WebSocket origins are checked.** It used to accept
  every origin. Now: same-origin (every LAN/Tailscale/port-forward use),
  an exact match in the new **`allowed_origins`** parameter (ships
  `["https://dashboard.sfuracerbot.ca"]`), or no `Origin` header (scripts).
  Anything else is a 403. `origins.py`, tested in `test_origins.py`.
- **`hello` first on every connection**:
  `{"type":"hello","protocol_version":1}`. New `protocol.PROTOCOL_VERSION`;
  bump it on any incompatible wire change (rule in the package README).
- **Connection roles** from `X-Racerbot-Role`: `relay` (full telemetry,
  every write refused with a new `write_refused` message; unknown requests
  count as writes), `control` (write-panel state, stopwatch, and replies to
  its own actions only; no map/scan/batch), none (unchanged). Any other
  role value is a 400. Every write is logged with `X-Racerbot-User`, or
  `unknown (direct)`. `roles.py`, tested in `test_roles.py`.
- **New `serve_static`** (default `true`): `false` serves only `/ws` and
  404s every page.
- **`server.py`**: the Tornado handler and fan-out moved out of
  `dashboard_node.py`, so `test_server.py` runs them on a real socket
  without `rclpy`.
- **`web/` is deprecated**, a frozen fallback. New frontend work goes to
  sfu-racerbot/web-dashboards. Its tests stay until that repo ports them.

### racerbot_launch — foxglove_bridge

- New `config/foxglove_bridge.yaml` and `launch/foxglove_bridge_launch.py`:
  `127.0.0.1:8765`; topics, services and parameters (get and set) open;
  **client publishing limited to `^/initialpose$`** (**WRONG — superseded
  2026-09-27: 3.5.0 does not enforce that whitelist; publishing is now off
  entirely**), because a raw `/drive` publish would skip every driving
  node's LB deadman; raw camera/depth
  images hidden. Parameter names checked against foxglove_bridge 3.5.0.
- `exec_depend foxglove_bridge` (apt `ros-jazzy-foxglove-bridge`, **not
  installed yet**). First test in this package:
  `test/test_foxglove_bridge_config.py`.
- `tools/systemd/foxglove-bridge.service` + `foxglove-bridge.sh`, written
  but **not installed or enabled**. See `docs/foxglove-bridge.md`.

### race_diagnostics

- `/parameter_events` added to `record_run.py`'s default bag topics, so
  parameter changes from the dashboard, Lichtblick or a terminal land in
  each run's bag.

## 2026-08-24 — Measuring on the map, and three ways to clear one

Both features live entirely in `web_dashboard`. The package still creates
**no ROS publisher** — the SLAM reset is a service client, like the existing
live-tuning clients. Verified end to end against a running node: every
refusal path exercised over a real WebSocket, and one throwaway run
directory created and deleted for real. **Not yet exercised on the car with
`slam_toolbox` actually running** — the reset path was only observed
refusing when the service was absent.

### web_dashboard — a measuring tool on the map

- **Tap the map to measure.** Two taps give a point-to-point distance; more
  keep chaining, with each leg labelled and a running total. Dragging still
  pans and pinching still zooms — only a tap adds a point, judged on the
  *furthest* the pointer travelled rather than where it ended, so a drag
  that wanders and returns is not mistaken for a click.
- **The geometry is its own file.** `web/measure.js` touches no DOM, canvas
  or WebSocket, so `test/browser/measure_test.js` loads the real file under
  plain node and checks it against closed forms (79 checks).
- **`dashboard.js` gained `canvasToWorld` / `canvasToBody` /
  `canvasToActive`.** The inverse transform previously existed only inline
  inside `zoomAt`, in two copies; `zoomAt` now calls the extracted pair.
- **A measurement taken before localization is cleared when a pose
  arrives,** with a note saying why. Those points were relative to a car
  that has since moved, and redrawing them in map coordinates would put a
  confident-looking number in the wrong place.
- New sidebar section `measure`, a readout pinned over the map (the phone
  sheet starts closed), and <kbd>M</kbd> / <kbd>Esc</kbd> /
  <kbd>Backspace</kbd>. No new parameters.

### web_dashboard — clearing a map, three ways

- **clear view** — the browser forgets its copy and asks for a fresh
  keyframe. Touches nothing on the car. Both outcomes are diagnostic: the
  map returning at once means the car is still publishing it.
- **reset live SLAM** — calls `/slam_toolbox/reset`. **Refused while any
  driving node is running**, on the reasoning already recorded in
  `killable_nodes`: a controller with a frozen pose is more dangerous than
  one with no pose. `pause_new_measurements` is hard-coded `False` and
  deliberately not a parameter.
- **delete a saved run** — permanently removes a run directory under
  `map_roots`. Gated on typing the run's name exactly, plus a content
  digest that refuses a delete from a page listing that has gone stale.
- **The deletion unit is a whole run directory, never a file inside one.**
  `map.yaml` names its image relatively, the racing line was optimized
  against that grid, and the pose graph is the only route back to a map
  whose save raced. Deleting `map.pgm` while leaving `map.yaml` makes
  `map_server` fail to configure and `particle_filter` block in its
  constructor — so a partial delete is made impossible rather than
  discouraged.
- **`delete_run` does not `rmtree` the run.** It re-classifies every entry
  first, unlinks only what it recognised as run output, and finishes with
  `os.rmdir`, so an unexpected file aborts the delete instead of being
  swept up with it. `shutil` work runs on a worker thread, not the rclpy
  thread that serves telemetry.
- New `web_dashboard/mapstore.py` (ROS-free, 133 tests) and
  `proccontrol.DRIVING_CONTROLLERS`. `package.xml` gains
  `<exec_depend>slam_toolbox</exec_depend>`; the import is guarded so a
  machine without it still gets a working dashboard.
- **New parameters** in `config/web_dashboard.yaml`, all defaulting on:
  `enable_map_delete` (`true`), `map_roots`
  (`[~/.ros/racerbot_auto, ~/.ros/racerbot_sim/auto]`),
  `map_scan_interval_sec` (`10.0`), `enable_slam_reset` (`true`),
  `slam_reset_service` (`/slam_toolbox/reset`), `slam_reset_timeout_sec`
  (`10.0`).
- **Security posture is unchanged and now weaker in one direction:** this
  port has no authentication, and deleting a map is the first irreversible
  thing on it. `enable_map_delete: false` at a shared venue. The typed name
  guards against a mis-tap, not against someone hostile.

Tests: 232 → 400 in `web_dashboard`, still running with no ROS sourced.

## 2026-08-22 (later) — A racing line worth racing, and a filter to follow it with

Builds on the GPU work below: with the particle filter finally affordable,
the auto-map-race flow can hand localization over for the race, and the
optimizer that was always sitting in the repo can run inline. Validated in
`racerbot_sim`; **needs on-car validation**, and see the speed warning.

### pure_pursuit — raceline optimization is now part of the auto flow

- **`auto_map_race_node` optimizes the racing line before racing it.**
  `_write_profile` previously cleaned up the recorded lap and paced it, but
  never reshaped it — the car raced wherever `gap_follow` happened to drive.
  It now feeds that cleaned lap to `raceline_optimizer` as a *seed*,
  re-derives the minimum-curvature line inside the measured track width, and
  paces **that**. Curvature is re-measured from whichever line wins; pacing
  new geometry with old curvature would have thrown the gain away.
- **It only races the optimized line if that line is genuinely better.**
  Three gates, any failure falling back to the cleaned recording: wall
  clearance below `profile_wall_clearance` is a hard refusal; steering past
  the rack limit is refused only when *also* worse than the recording (which
  `prepare` itself merely warns about, so an absolute bar would have
  disabled the feature on exactly the tight courses it was tested on); and
  **estimated lap time must actually improve**.
- **That last gate matters more than it sounds.** Minimum curvature buys
  corner speed by using the full track width, which lengthens the path, so
  it only pays where curvature is the binding limit. Measured: a lumpy
  ~34m circuit went 13.70s → 12.69s, but this car's own 13.3m test course
  produced a 16.9m line a second per lap *slower* — correctly refused. A
  "not faster" log line is the feature working.
- New `optimize_*` parameters in `auto_map_race.yaml`, `optimize_raceline`
  defaulting **true**. Saves `raceline_optimized.csv` beside the raw and
  profiled files.

### pure_pursuit — the optimizer is 4x faster

- **`solve_lateral_offsets` forced `lsq_solver='lsmr'`.** Profiled on the
  Jetson: a 126m circuit spent 55s in `trf_linear`, of which 46s was 781
  `lsmr` calls and ~36 million Python-level matvec dispatches through
  scipy's `LinearOperator` — big-sparse machinery on a few hundred unknowns.
  Densifying and asking for the direct solve is **55.7s → 13.9s**, same line
  to within 0.9mm, identical mean curvature. New `DENSE_SOLVER_MAX_UNKNOWNS`
  (4000) keeps the old iterative path for anything genuinely large.
  This is what makes the step affordable inline, while the car is stopped.

### pure_pursuit + racerbot_launch — localization hands over for the race

- **The racing phase now uses the particle filter, not `slam_toolbox`.**
  Once the map is saved, `auto_map_race_node` spawns localization against
  it, seeds `/initialpose` from the pose SLAM already knows, waits for
  `pf_settle_poses` consecutive poses, then republishes *the filter's*
  estimate on `pose_topic`. `pure_pursuit_node` is not reconfigured — this
  node always owned that topic, so the handover changes the source, not the
  wiring. `slam_toolbox` measured 106 pose jumps >0.12m in a 136s lap.
- **Fails safe at every step**, each logging its reason and leaving the car
  on `slam_toolbox`: no saved map, a filter that will not start, one that
  never converges inside `pf_startup_timeout_sec`, or one that goes quiet
  later. Demotion is one-way for the rest of the run — flapping between two
  pose sources at speed is worse than either.
- **New `racerbot_launch/launch/localize_run_launch.py`.** `particle_filter`'s
  own `localize_launch.py` could not be used: it points `map_server` at the
  map packaged *inside the submodule*, and hardcodes `use_sim_time: True`,
  which on the real car waits on a `/clock` that never ticks. Being a
  submodule, neither is fixable in place.
- **The `/tf` remap in that file is load-bearing.** `particle_filter`
  publishes `map -> laser` unconditionally with no parameter to stop it,
  while `slam_toolbox` still publishes `map -> odom` and `laser` already
  descends from `odom`. Two parents for one frame is a broken TF tree. Its
  transforms go to a dead topic; only its PoseStamped output is consumed.
  The `/map_server/map` *service* is deliberately left unremapped — moving
  it silently breaks the pairing with `map_server`.
- **The spawned process now dies with its parent** (`PR_SET_PDEATHSIG`).
  Found by leaking: killing the sim validation run left `map_server` and
  the particle filter alive afterwards, holding the service name the next
  run's handover needs. `start_new_session=True` was the cause and is gone,
  so an ordinary Ctrl+C now reaches `ros2 launch` and shuts it down in order.

### pure_pursuit — profile speed caps raised

- **`profile_max_speed` 2.0 → 4.0, `profile_max_lateral_accel` 1.2 → 2.5,
  `profile_max_accel` 3.0 → 6.0**, matching what `pure_pursuit_node` already
  enforces online, so the profile is no longer the binding limit. Asked for
  deliberately, to race the optimized line as fast as the controller allows.
- **This contradicts a measured warning that is still in the file above it.**
  The 0.39-0.57m cross-track error recorded through a corner was measured at
  2.5-3.0 m/s — *inside* the new range — and on a 1.8m indoor corridor these
  values are expected to put the car into a wall. Use `supervisor_config:=`
  with a copy holding the old 2.0/1.2/3.0 for anywhere tight.

### Validation

- `racerbot_sim`, `indoor_wide`, full map-then-race: optimizer accepted the
  line (mean |curvature| 0.2454 → 0.1471, estimated lap **15.16s → 11.38s**,
  0.30m wall clearance) in 0.44s; handover completed after 20 poses; 48
  racing ticks at up to 4.00 m/s; **zero errors, zero collisions, the filter
  never demoted**.
- 274 `pure_pursuit` tests pass (10 new covering both features).

## 2026-08-22 — The GPU ray caster, which was never actually built

Came out of an audit of what the Jetson's GPU is worth to this workspace
(new `docs/gpu-acceleration.md`). Needs on-car validation — the library
under localization was replaced.

### range_libc (build change, not a submodule bump)

- **`range_libc` was installed without CUDA, while `particle_filter` asks
  for `range_method: 'rmgpu'`.** `PyRayMarchingGPU.__cinit__` prints a
  warning and then returns *without constructing anything*, leaving a null
  pointer; the config is accepted, the node starts, and the first scan
  kills the process with `Aborted (core dumped)`. `race_launch.py` includes
  `localize_launch.py`, so the documented race-day launch would have lost
  localization on its first scan. Latent only because the auto-map-then-race
  flow keeps `slam_toolbox` running through the racing phase instead.
- **Rebuilt with `WITH_CUDA=ON`** (`cd src/range_libc/pywrapper &&
  WITH_CUDA=ON python3 setup.py install --user`). `setup.py` already
  carried the correct `-arch=sm_87`. `range_libc.SHOULD_USE_CUDA` now
  reports `True`. No submodule source was modified; this is a build
  artifact, and `colcon build` does not touch it (no `package.xml`).
- **Measured: 4.8× on the particle filter's own sensor-update path** —
  28.86ms → 5.96ms at the shipped 4000 particles, against a 25ms scan
  period, so the CPU path did not fit. Raw ray casting is 10.7× (34.05ms →
  3.19ms for 240k rays), returning bit-identical distances to the CPU
  method (max difference exactly 0.0).
- **CPU range methods are unchanged by the rebuild** — verified identical
  output, so `pure_pursuit`'s map-subtraction opponent detection
  (`map_subtraction.py`, `PyRayMarching`) is unaffected.

### racerbot_launch

- **`race_launch.py` now refuses to launch on a GPU config that would
  fail**, via a new `_check_gpu_ray_caster()` run at launch-description
  generation time. It reads `particle_filter`'s `localize.yaml` the same
  way the file already reads `pure_pursuit.yaml`. Two checks, both only
  when `range_method` is `'rmgpu'`:
  - `range_libc` built without CUDA → raises with the rebuild command,
    instead of letting the node abort on its first scan.
  - `max_particles × rays > 262144` → raises with the computed limit.
    Above that threshold `range_libc`'s GPU ray caster returns **silently
    wrong ranges** rather than failing: measured, one particle over the
    limit made 100% of returned ranges wrong, with one line on stdout.
    Cause is two defects in `numpy_calc_range_angles` (RangeLib.h) — a
    `ceil` that oversizes the chunk past the `CHUNK_SIZE` device buffer,
    and an offset computed from `num_in_chunk` where it means
    `particles_per_iter`. Both upstream, both documented in
    `docs/gpu-acceleration.md`.
  - New module constants `LIDAR_BEAMS` (1081, matching
    `racerbot_sim/sim_bridge.py`) and `RANGE_LIBC_CHUNK_SIZE` (262144).
    No new ROS parameters.
- With the shipped `max_particles: 4000` and `angle_step: 18` (61 rays),
  the limit computes to 4297 particles — the current config passes with
  about 7% headroom.

### docs

- **New `docs/gpu-acceleration.md`** — the full audit: what is and isn't in
  this Jetson (no DLA, no PVA, no NVENC, no CUDA in OpenCV, all verified on
  the board), why the driving loops stay on the CPU (`gap_follow` 2.8% and
  `pure_pursuit` 4.3% of their 25ms budget, against a 36–61µs GPU round
  trip), and what the idle tensor cores would need to be worth using.
- **`docs/racing-autonomy.md` corrected.** It described `range_libc` as
  doing "fast GPU-accelerated ray casting", which was not true at the time
  it was written. It is true now, with the two conditions stated.

## 2026-08-19 — Making a mapping run finish, and finish quicker

Everything here came out of one run's logs (`~/.ros/log`, 2026-08-19
18:46). Needs on-car validation.

### gap_follow

- **`gap_follow_node` would not start under any mapping launch file.** The
  previous commit added `corner_speed_wide` and a startup check enforcing
  `corner_speed <= corner_speed_wide <= max_speed`. Every mapping launch
  file overrode `max_speed` to `1.0` and nothing else, leaving the
  packaged `corner_speed: 1.1` / `corner_speed_wide: 1.4` above it, so the
  node exited with `ValueError` 1.8s after launch. Nothing then published
  `/auto_map/drive`, the supervisor recorded no lap, and
  `pure_pursuit_node` sat in `waiting_for_profile` with the car
  motionless — three symptoms, one dead node.
- **New `gap_follow/speed_overrides.py`.** Pure Python, no rclpy. Given a
  requested speed cap it scales every parameter coupled to `max_speed` by
  the same factor and clamps the result into `[min_speed, max_speed]`,
  preserving the ordering the node validates.
- **The mapping speed cap is gone by default.** `mapping_max_speed` /
  `mapping_min_speed` now default to empty, meaning "no cap": the car maps
  at `gap_follow.yaml`'s tuned speeds and the sensed curvature/clearance
  caps do the limiting. On the 2026-08-19 run the forced 1.0m/s cap was
  the binding limit on **154 of 191** logged driving ticks — 81% of the
  run was the car obeying an override rather than anything it could see.
  Pass a number to cap it for an unfamiliar course; the coupled corner
  caps scale with it automatically.

### pure_pursuit

- **Lap turning is counted in the `odom` frame.** Absorbing a SLAM
  correction also discarded the car's real turning during that tick. The
  run absorbed **106 corrections in one 136s lap** and measured 335° for a
  genuine revolution, against a 300° gate. Odometry is never
  re-optimised, so the gate no longer depends on how busy the pose graph
  was. Falls back to map yaw when no `odom` transform exists. New
  parameter: `odom_frame` (default `odom`).
- **A lap that will not close now widens rather than running forever.**
  Past `closure_widen_after_revolutions` (new, 1.25) the proximity gate
  opens in proportion to the extra turning, up to `max_closure_distance`
  (new, 4.0m). A reactive controller does not repeat its line, and on the
  126m course this car maps each missed closure costs 2.3 minutes.
- **`LapRecorder.lap_points()` trims a recording to its final
  revolution**, so a closure that needed two laps still produces a
  one-lap racing line instead of two overlapping ones.
- **Lap progress is reported in plain terms.** The decision log now leads
  with `~34% round, ~83m to go (~90s at last lap's pace)`, and the
  supervisor prints an estimate of what lap 2 will cost when lap 1 closes.
  The operator on 2026-08-19 stopped a working run 47s into lap 2; the
  raw gate numbers gave no way to tell progress from a stall.
- **Stray-return speckle is cleaned out of the saved map and the copy the
  racing line is checked against.** New `occupancy_map.despeckle_grid()`
  removes small occupied blobs *only* where nothing unknown sits anywhere
  in their neighbourhood — a real object occludes and so keeps an unknown
  shadow behind it, while a stray beam is a direction the LiDAR swept
  clear on every other pass. Measured on this car's own saved map: 12
  phantom blobs in clear track removed, the 34 small ones against walls
  and the map edge deliberately kept. **Not** applied to the live `/map`
  used for opponent detection. New parameter:
  `map_despeckle_max_cells` (default 4).
- **New latched `/auto_map_race/controller` topic** naming the controller
  currently selected, for the dashboard. Read-only, published after the
  tick's drive command, disables itself rather than the node on error.
- **`pure_pursuit_node` broadcasts its active racing line** on a latched
  `/racing_line` topic, once per loaded profile and never from the
  control loop. Read-only; disables itself rather than the node on error.

### web_dashboard

- **The racing line is drawn on the map**, coloured by its own target
  speed (green fast, amber slow), from a new latched `/racing_line`
  topic that `pure_pursuit_node` publishes once per loaded profile.
  Its presence is the visual answer to "is pure pursuit racing yet" --
  it cannot appear until a profile is accepted. New parameters:
  `racing_line_topic` (both nodes) and `racing_line_max_points`.
- **The tuning panel says which node is driving.** During an
  `auto_map_race` run both controllers are online and tunable while only
  one drives. The driving one is sorted to the top and badged in green;
  the other is dimmed and badged "not driving". Every live-tune write of
  the 2026-08-19 run went to `pure_pursuit_node` while `gap_follow_node`
  drove the entire run. New parameter: `controller_topic`.

### racerbot_launch

- **New `config/slam_tracking.yaml`**, layered over the vendored
  `f1tenth_online_async.yaml` by `slam_launch.py` so the vendored file
  stays clean and an upstream sync cannot take our tuning with it.
  `minimum_time_interval` 0.5 → 0.2, `minimum_travel_distance` 0.5 →
  0.25, `minimum_travel_heading` 0.5 → 0.25, `throttle_scans` 1 → 2,
  `loop_search_maximum_distance` 3.0 → 8.0,
  `loop_match_minimum_chain_size` 10 → 20, `ceres_loss_function` None →
  HuberLoss. The loop-search radius is the likely reason the map was not
  closing: the course is 126m round and stock SLAM only looked 3m for a
  candidate. New launch argument: `slam_tracking_file`.
- **`auto_map_race_launch.py` starts `race_diagnostics` by default**
  (`diagnostics:=true`, `record_bag:=false`), so a run leaves pose-lag and
  watchdog numbers instead of only terminal scrollback.

### Hardware findings (no code change)

- **The VESC's onboard IMU reports all zeros.** Measured directly:
  `/sensors/imu/raw` publishes at a steady 50Hz with every gyro and
  accelerometer axis exactly `0.0`, and no gravity vector on any axis.
  Either the board has no IMU fitted or it is disabled in firmware
  (VESC Tool -> App Settings -> IMU). This matters because `/odom`'s
  heading is integrated from the *commanded* steering angle through a
  bicycle model, with nothing measuring actual rotation — a working gyro
  is the single biggest available improvement to it. Recorded in
  [hardware-reference.md](docs/hardware-reference.md) with a re-check
  procedure.

### docs

- **New [localization.md](docs/localization.md)** — where the position
  estimate comes from, why it arrived late, what was tuned, and a ranked
  plan for what would improve it next. Records that `vesc_to_odom`
  integrates *commanded* steering angle through a bicycle model, that the
  VESC IMU is dead, and why "fuse odom and LiDAR with an EKF" is not a
  change: SLAM already fuses them, and an EKF between them would
  double-count the odometry.

## 2026-08-08 — A dashboard the car can afford, and a camera worth watching

### web_dashboard

- **The map is sent as changes, not as a map.** `slam_toolbox` republishes
  its entire grid every `map_update_interval` for as long as it is mapping
  — which is exactly while somebody is driving — and at the levine map's
  2048×2048 that is 4MB a message, 819 kB/s, per open browser tab. New
  `mapstream.py` answers each grid with a *keyframe* (first sight, a
  resize, a new tab, or every `map_keyframe_sec`), a *patch* covering only
  the rectangle that changed, or nothing at all when the grid is identical
  to the last one. Measured: ~200 bytes where it used to be 4MB. Every
  frame carries a sequence number, and the browser applies a patch only if
  it is the exact successor of the last one — on a gap it waits for a
  keyframe rather than painting a map that is quietly wrong.
- **~155 WebSocket frames a second became ~40.** Pose, command, speed,
  intent, stopwatch and stats are collected into one `batch` frame at
  `telemetry_rate_hz`. A frame costs about the same however small it is, so
  the framing was the cost. Latest-wins per type — except `/drive_intent`,
  whose *state transitions* the browser builds its decision log from, so
  those are queued in order and every one survives (`batching.py`).
- Scans are uint16 millimetres rather than float32: half the bytes, for a
  difference below one screen pixel and below the LIDAR's own accuracy.
  Intent messages drop their commanded path while it is indistinguishable
  from the desired one.
- `protocol.map_cells`/`scan_ranges` now hand rclpy's `array.array`
  straight to the wire instead of unpacking 4.2 million ints into
  `struct.pack`: 178ms → 2.2ms per map message, byte-for-byte identical
  (the existing round-trip tests pass unedited, which is the proof).
- **Nothing happens at all when no browser is connected.** Every
  broadcasting callback checks first; a newly connected tab is caught up by
  `send_initial_state()` as before. Stats still sample on their timer and
  intent messages are still validated, since both are useful with no tab
  open.
- Measured over a live 60s mapping run: **914 kB/s → 61 kB/s** (7.1 →
  0.48 Mbit/s), zero dropped frames. Re-runnable:
  `tools/web_dashboard/bench_protocol.py`.
- **The sidebar can be used now.** The decision log had `overflow-y: auto`
  and was impossible to scroll, because `#overlay` was
  `pointer-events: none` so the wheel went past it to the canvas and
  zoomed the map. And the sidebar had no height bound while the page could
  not scroll, so on a laptop or phone its bottom — the decision log, the
  tuning button, "reset view" — was rendered off-screen and unreachable.
  It is now bounded and scrolls, with the mode banner and view controls
  pinned outside the scroll region.
- Sections are collapsible `<details>` that remember their state per
  browser, and a collapsed one still shows its headline value in its
  header. Feeds, vehicle and system moved to two compact columns with the
  long form in each row's tooltip. Scrollbars are styled, since the
  defaults are invisible on this theme.
- The browser coalesces repaints through `requestAnimationFrame` (pose
  alone used to force 40 full canvas repaints a second) and paints map
  cells through a 256-entry palette. A hidden tab now draws nothing.
- **New parameters:** `telemetry_rate_hz` (20.0), `map_compression`
  (true), `map_patching` (true), `map_keyframe_sec` (30.0),
  `scan_encoding` (`u16mm`), `scan_decimation` (1).
  **Changed default:** `stopwatch_update_rate_hz` 10.0 → 4.0, since the
  browser runs the clock between updates.

### usb_cam_stream

- **Two tiers.** `/stream` is a small preview (`preview_width`, default
  480) for the dashboard's camera inset; `/stream?tier=full` is the
  camera's own resolution for the recording view. The inset is at most 220
  CSS pixels wide and was being fed 1280×720 at 12–18 Mbit/s — about 34×
  more picture than it could show, competing with the dashboard's own
  telemetry for the same link. Each tier is encoded once and shared, and a
  tier nobody is watching is never encoded at all.
- **The blur was a second JPEG generation.** The camera is asked for
  MJPEG, OpenCV silently decodes it to BGR, and this node re-encoded it.
  With `passthrough` (default true) the camera's own JPEG is served
  untouched — sharper, and it removes a decode and an encode per frame.
  Probed at startup, falls back cleanly, logs which mode is active.
- Frames are pushed as soon as they exist instead of polled for on a
  `1/stream_fps` sleep, which used to add up to 66ms to every frame.
  Encoding moved to its own thread and always works on the newest frame,
  dropping any backlog rather than letting latency accumulate — in
  `image_topic` mode it used to run on the rclpy executor thread, stalling
  every other ROS callback behind a 720p encode.
- `image_topic` mode subscribes with sensor QoS at depth 1, not the default
  10: a ten-frame backlog is a third of a second of stale frames.
- Downscaling uses `INTER_AREA`, so the preview is legible rather than
  aliased.
- **New parameters:** `preview_width` (480), `preview_quality` (65),
  `full_width` (0), `full_quality` (90), `passthrough` (true).
  **Changed default:** `stream_fps` 15.0 → 30.0 (now a cap, not a poll
  rate). **Deprecated:** `jpeg_quality` — still honoured, applied to both
  tiers, warns.

### tools

- `tools/web_dashboard/bench_protocol.py` — packing speed, bytes on the
  wire, and WebSocket framing cost at real dimensions (2048² map, 1081
  beams), with thresholds so it fails on a regression.
- `tools/web_dashboard/check_wire_format.py` — drives `dashboard_node`'s
  real callbacks and checks what it puts on the wire.
- `tools/racerbot_sim/capture_dashboard.py` speaks the new protocol and
  reports kB/s per message type, keyframe/patch counts, and any patch it
  had to ignore.

## 2026-08-08 — The automatic map-to-race path, and a simulator that can see it

### racerbot_sim (new package)

- F1TENTH Gym behind the car's real topics. `gym_bridge_node` replaces
  `urg_node` and the whole VESC chain: it consumes `/ackermann_cmd` and
  publishes `/scan` (1081 beams over 270°, matching the Hokuyo UST-10LX),
  `/odom`, and the `odom->base_link` transform. `sim_joy_node` replaces the
  physical F710 with a synthetic LB hold. `sim_bringup_launch.py` mirrors
  `bringup_launch.py`, keeping the real `ackermann_mux` and the real static
  `base_link->laser` transform, and `sim_auto_map_race_launch.py` *includes*
  `racerbot_launch`'s own `auto_map_race_launch.py` rather than copying it,
  so what is validated is the launch file people run.
- Odometry is dead-reckoned from wheel speed and the *commanded* servo angle,
  exactly as `vesc_to_odom` does, so it drifts the way the car's does.
  Ground truth is published separately on `/sim/ground_truth_pose` in an
  unconnected frame, for scoring only.
- **Hard interlock against real hardware.** Neither node publishes anything
  while `vesc_driver_node`, `ackermann_to_vesc_node`, `vesc_to_odom_node`,
  `urg_node` or `joy` is on the ROS graph, re-checked continuously so
  bringing the car up after the simulator silences the simulator too. A
  synthetic `/joy` holding LB defeats the workspace safety policy by design,
  and a second `/scan` publisher is its own hazard.
- Room-sized generated tracks (`indoor_oval`, `indoor_tight`, `indoor_wide`),
  because the official 300m+ circuits make one automatic-mode run over twenty
  minutes long, and a validation nobody runs is how the automatic mode got
  into the state it was in.
- `tools/racerbot_sim/run_auto_map_validation.py` runs the whole composition
  solo, with a parked car, and in traffic, and exits non-zero if any scenario
  fails to reach racing or touches anything. All three pass
  (`docs/auto-map-sim-results.json`, seed 12345): 138.7m, 29.7m and 83.1m
  raced in ~80s each, zero wall contact and zero car-to-car contact.
  `tools/racerbot_sim/capture_dashboard.py` renders what the web dashboard is
  drawing as a PNG, from the terminal.
- Found along the way: in the pinned Gym revision, **wall collisions never
  fire** with this workspace's vehicle parameters, because `side_distances`
  computes to all zeros and the test reduces to "did a beam return under
  5mm" against a 0.05m `range_min`. `tools/f1tenth_sim/run_validation.py`'s
  "no collision" column has therefore always meant "not detected".
  `racerbot_sim` samples the padded body against the occupancy grid instead.

### web_dashboard

- **The map was not glitchy; the camera was.** The view auto-framed the map
  by re-deriving centre and zoom from *every* `/map` message, and
  slam_toolbox resizes and re-origins its grid constantly as the map grows,
  shrinking as often as growing. Measured over 130s of mapping: 27 map
  messages, **18 view disturbances, the picture sliding up to 3.6m and
  rescaling by up to 36%** while the map itself was fine. It now frames the
  map once and re-fits only when the map no longer fits on screen — the
  same run gives **2**, both of them the map genuinely growing.
- Map and scan headers now carry `bytes`, and the browser drops any binary
  frame whose length disagrees rather than painting it. The browser holds
  one "what does the next binary mean" slot, so a header that never got its
  binary made every later payload decode as the wrong type — a 1081-beam
  scan read as occupancy cells is 4324 bytes against an 80000-cell header,
  every read past the end undefined, every colour NaN, and the map paints
  as garbage rather than failing. `applyMap` also refuses a short payload
  outright, dropped frames are counted in the mode banner, and
  `_send_to_all` now drops a client whose pair it could not complete (it
  caught only `WebSocketClosedError` before) so that browser reconnects and
  resynchronises instead of decoding everything wrongly from then on.
  Across ~7,500 binary frames of validation **zero** failed the check, so
  this is a guard against an unobserved failure, not a fix for a seen one.
- `tools/racerbot_sim/capture_dashboard.py` renders what the dashboard is
  drawing as a PNG from the terminal, and doubles as its test instrument:
  `--seconds`/`--interval` watches a whole run, validates every frame, and
  exits non-zero if any failed. It also replays both view-fitting policies
  over the recorded map sequence, which is where the numbers above come
  from.

### pure_pursuit — the recorded racing line

- New `recorded_path.py`: trims a recording to one revolution, resamples it
  to uniform spacing, and low-passes the closed loop in space, discarding
  features shorter than the car's own 1.22m turning circle. Reports peak
  curvature against the steering rack, the seam heading error, and how far
  the cleanup moved the line off the one actually driven.
- `auto_map_race_node` now **refuses** to hand pure pursuit a line needing
  more than `profile_reject_ratio` (1.5) times the rack limit, or needing
  more than the rack has over more than `profile_reject_fraction` (0.25) of
  the lap, and says so with numbers. Every racing line this car had ever
  generated demanded more steering than the rack has on a third of its
  waypoints, peaking at 47° against a 14.9° limit.
- The supervisor now subscribes to `/map` and checks the finished line
  against SLAM's own occupancy grid, because filtering a recorded lap
  rounds its corners *inward*, toward the wall, and on a course whose
  corners are near the car's 1.22m turning circle that is the direction
  that hurts: a measured run finished with the line 0.05m from a wall --
  under the car's own 0.155m half-width -- while curvature, seam error and
  deviation all read healthy. Clearance outranks curvature when choosing
  the filter cutoff. The requirement is capped by the driven line's own
  clearance, because mapping with traffic paints the other cars into the
  grid -- the two-car scenario produced a recorded line measuring 0.00m of
  clearance on a lap the ego had just driven twice without touching
  anything, and refusing there would be refusing to believe the lap that
  happened. `OccupancyMap.from_grid_message` is new, and stays ROS-free by
  taking the message's fields rather than the message.
- **New parameters** in `auto_map_race.yaml`: `minimum_lap_turn_deg` (300),
  `max_pose_jump_m` (0.12), `profile_max_steering_angle`, `profile_wheelbase`,
  `map_save_retries`, `map_save_retry_delay_sec`,
  `profile_min_feature_wavelength`, `profile_curvature_margin`,
  `profile_max_deviation`, `profile_reject_ratio`, `profile_reject_fraction`,
  `map_topic`, `profile_wall_clearance` (0.30), and
  `profile_map_occupied_threshold`.
  **Removed**: `profile_smoothing_window` (replaced by the above).
- **Changed defaults**: `minimum_lap_distance` 20.0 → 5.0 (it was longer than
  the loop this car is driven on, so closure could not fire until the car had
  been round *twice*, and every recorded line was two overlapping laps);
  `profile_max_brake` 8.0 → 3.0 (it is a braking-authority assumption in the
  velocity profile, not a slew rate, and 8.0 was 2.7× what `gap_follow.yaml`
  will assume for the same car); `profile_max_speed` 4.0 → 2.0 and
  `profile_max_lateral_accel` 2.5 → 1.2 (that was the surveyed-track profile,
  applied to a course discovered thirty seconds earlier at 1 m/s). What sets
  that ceiling is width, not grip: the line comes from `gap_follow`, which
  drives 0.25-0.35m from a corner's wall, and pure pursuit's cross-track
  error through a corner near this car's turning circle measured 0.39-0.57m
  at 2.5-3.0 m/s. On a 1.8m indoor corridor those do not both fit, at any
  speed where the error exceeds the room.
- `closure_distance` 0.75 → 1.5. That gate existed to keep the start/finish
  seam short, because the seam used to be a naked chord from the last
  recorded point to the first; `recorded_path` now trims and resamples it
  instead, so the gate can be what it should always have been -- "did the
  car come back past its start". 0.75m was also simply unachievable for a
  reactive controller weaving around traffic: the two-car scenario drove
  413m and 3726 degrees of turning -- ten laps -- passing 6.5m wide of its
  start every time, without ever closing. The progress log now warns
  explicitly once two laps of turning have gone by with no closure, instead
  of leaving that to be inferred from two numbers.
- `LapRecorder` now absorbs SLAM pose corrections into the already-recorded
  path instead of recording them as geometry, and counts them in the progress
  log. Closure is gated on accumulated yaw, which does not need to know how
  big the course is.

### pure_pursuit — saving the map

- The occupancy-map save is retried (`map_save_retries`, 3, spaced
  `map_save_retry_delay_sec`). slam_toolbox runs nav2's `map_saver` inline
  and `map_saver` gives up after ~2s if no `/map` arrives, while `/map` is
  only republished every `map_update_interval` (5s) — so whether the save
  works is a race against when the request lands, and the same run fails or
  succeeds with nothing else different. A retry lands in a different part of
  the window; if all attempts fail the run still continues and says the pose
  graph can be turned back into a map with `deserialize_map`. The handover
  gate deliberately does not treat a save as settled while a retry is due.

### pure_pursuit — the reactive safety net

- The `emergency_obstacle` hard stop is no longer a latch. A stopped car
  cannot clear its own safety cone, so the stop, on its own, ended the run
  wherever it fired. It now crawls toward a genuine opening at
  `emergency_escape_speed` (0.25 m/s) when the whole body still has
  `emergency_escape_clearance` (0.10m) of room and a gap deeper than
  `emergency_escape_min_gap` (0.8m) exists. The contact and stale-scan stops
  still win outright; `emergency_escape_speed: 0.0` restores the old
  behaviour. Same reasoning as `gap_follow`'s `escape_creep_speed`.

### racerbot_launch

- `auto_map_race_launch.py` takes `supervisor_config:=`, so a course can have
  its own racing profile without editing the packaged config.

### Tests

- `test_overtakes_toward_the_*` asserted the *sign* of a steering command
  that back-to-back `control_loop()` calls left pinned within a thousandth
  of a radian of zero -- `max_steering_rate` permits about 0.0001 rad per
  microsecond-long tick, so the sign was scheduler noise, and a few extra
  microseconds of work per tick flipped it. They now give the slew limiter a
  realistic time step (the idiom already used by
  `test_shaped_speed_ramps_up_to_the_profiled_speed`) and assert the pass
  side as a comparison between passing left and passing right, which is what
  the claim actually is: the 0.35m offset applied 4m ahead is about five
  degrees of bearing, a nudge on top of the racing line's own curvature
  rather than a replacement for it, so on that fixture's track both passes
  come out positive.

## 2026-07-25 — Runtime autonomy decision logging

### gap_follow and pure_pursuit

- Both local autonomy nodes now print immediate `STOP [reason]` /
  `DRIVE [reason]` state transitions and one-second steady-state summaries in
  their launch terminals. The diagnostics include the measurements,
  thresholds, algorithm choice, and final steering/speed command that explain
  each decision.
- Pure pursuit reports deadman/profile/localization/cross-track/LIDAR stop
  gates; path target/lookahead/curvature/profile-speed choices; reactive
  overrides; and opponent/overtake gating. Gap follow reports deadman and scan
  failures, emergency/no-gap stops, chosen gap bearings, steering clipping,
  and steering-based speed scaling.
- New `decision_log_period_sec` parameter in both packages controls periodic
  summaries (`1.0` s default, `0.0` for transitions only). Gap follow also has
  a `scan_timeout_sec` diagnostic watchdog (`0.5` s) so a dead scan stream no
  longer causes an unexplained mux timeout. Driving and deadman behavior are
  unchanged.

### Automatic workflow, waypoint recorder, and camera stream

- `auto_map_race_node` now reports its final forwarded/stopped command and why:
  deadman state, missing/stale child commands, selected controller, lap number
  and closure measurements, profile loading, and transition hold time. Its new
  `decision_log_period_sec` defaults to `1.0` s.
- `waypoint_recorder_node` now reports missing/stale pose input and live
  recording progress (waypoint count, path length, sample spacing, and output
  file). New defaults are `pose_timeout_sec: 1.0` and
  `status_log_period_sec: 2.0`.
- `usb_cam_stream_node` now reports device/topic wait states, negotiated camera
  mode, frame health, loss/recovery, stale frames, and conversion/JPEG errors.
  New defaults are `frame_timeout_sec: 2.0` and
  `status_log_period_sec: 5.0`.

## 2026-07-21 — F1TENTH Gym validation and automatic map-to-race launch

**Simulator validation complete; physical validation still required.** The
workspace now has a pinned, local F1TENTH Gym setup and deterministic headless
runner. Gap follow and pure pursuit completed Spielberg, Silverstone, and
Brands Hatch without collisions; a two-car Spielberg run completed one pass
and a full independently measured lap with neither car colliding. See
`docs/simulator.md` and `docs/f1tenth-sim-results.json`.

### pure_pursuit

- Simulator-tuned lookahead is now `0.6 + 0.15*v`, capped at `1.5m`; the old
  2.0m lookahead at 4m/s cut corners and collided in the dynamics model.
- Velocity-profile defaults are now `v_max=4.0m/s`, `a_lat_max=2.5m/s²`, and
  five smoothing passes.
- Map subtraction is the default opponent detector and now gates general
  reactive avoidance too. Added `avoidance_fallback_trigger_distance: 0.7m`
  for operation before a map arrives; the map-aware trigger remains `1.5m`.
- Active overtakes no longer get overwritten by the generic 1m/s avoidance
  command. Emergency stopping and stale-scan stopping still always win.
- Opponent progress-rate tracking now wraps correctly at start/finish, and the
  simulator-validated lateral pass offset is `0.35m`.
- `pure_pursuit_node` can opt into a stopped waiting state and atomically load a
  generated `waypoints_file` at runtime. Normal launches retain fail-fast
  behavior for a missing profile.
- New `auto_map_race_node`, config, and top-level
  `auto_map_race_launch.py`: one command starts cautious autonomous mapping,
  detects/records closed laps, generates and loads the profile, saves the map
  and pose graph, then transitions to pure-pursuit racing while SLAM remains
  online. Existing mapping, recorder, profiler, and saved-map race modes remain.

### gap_follow and tooling

- Gap follow's disparity-extender pipeline is now documented explicitly and
  validated for collision-free laps on all three simulator tracks.
- Added `tools/f1tenth_sim/setup.sh`, the deterministic validation runner,
  pinned dependencies, simulator documentation, and a checked-in JSON report.
- The LB deadman requirement is unchanged and is enforced again by the new
  command-selector supervisor.

## 2026-07-09 — Algorithm review follow-up: safety fixes, profile quality, map-based opponent detection

Full review + implementation session over the driving algorithms
(`gap_follow`, `pure_pursuit`). 86 tests pass (73 plain-pytest unit
tests, 13 rclpy integration tests); both packages rebuilt with
`colcon build --symlink-install --packages-select pure_pursuit gap_follow`.

**⚠ Not yet validated on-car.** Everything below passed the test suites
only. Before racing it, run the standard ladder from
`docs/writing-your-own-node.md`: static topic check → wheels off the
ground with LB held → low speed on open floor. The deadman policy is
untouched: `enable_deadman` stays `true` everywhere.

### pure_pursuit — bug fixes

- **Recovers after a localization jump** (`pure_pursuit_node.py`).
  Previously, if the particle filter re-converged somewhere outside the
  ±40-waypoint nearest-search window (pose jump, re-seed, car
  repositioned), the cross-track watchdog stopped the car *permanently*
  until a node restart. Now the watchdog retries once with a full-line
  search and only stops if that also misses; it stays un-anchored while
  lost so a recovered pose re-locks cleanly.
- **Overtakes complete on ego progress, not detection freshness**
  (`pure_pursuit_node.py`, `OpponentTracker`). Previously an active pass
  was cancelled 1.0s (`opponent_lost_timeout_sec`) after the opponent
  left the forward detection cone — which it always does when you pull
  alongside — snapping the steering target back onto the racing line the
  opponent might still occupy. Now, mid-pass, the opponent's position is
  dead-reckoned forward at its tracked progress rate and the pass ends
  only once the ego car is `overtake_clear_margin` past that predicted
  position. New parameter: `overtake_max_blind_sec` (3.0) — the hard cap
  after which a pass with no re-detection is abandoned instead.
- **Sub-`range_min` LIDAR readings no longer count as obstacles** in
  `_closest_in_cone` (the hard-stop / avoidance-trigger check). They are
  the sensor's "invalid" encoding, not a real 4cm object.

### pure_pursuit — map-subtraction opponent detection (new, opt-in)

- New detection mode: ray-cast the scan the LIDAR *should* see from the
  current pose using the saved map (`range_libc`, the same library the
  particle filter uses — new module `pure_pursuit/map_subtraction.py`),
  and flag whatever is meaningfully shorter than the map predicts.
  Unlike the shape heuristic, this cannot be fooled by wall corners and
  works with an opponent right in front of a wall. Comparison/clustering
  logic is in `racing_math.py` (`dynamic_beam_mask`,
  `detect_dynamic_cluster`) so it stays unit-testable without range_libc.
- New parameters (`config/pure_pursuit.yaml`): `opponent_detection_mode`
  (**ships as `heuristic`** — flip to `map` only after on-car
  validation), `map_topic`, `map_beam_step`, `map_subtraction_margin`.
  With no map received yet, `map` mode falls back to the heuristic with
  a warning rather than racing blind.
- `package.xml` gains `<depend>nav_msgs</depend>` (OccupancyGrid
  subscription, transient-local QoS).

### pure_pursuit — velocity-profile quality

- **Waypoint smoothing before profiling** (`racing_math.smooth_path`,
  wired into `generate_velocity_profile` as `--smoothing-window`,
  default 3, `0` disables). Localization jitter on the recorded line
  reads as curvature and produced phantom braking zones — on realistic
  recording geometry the raw jittered line measures >2× the true
  curvature; smoothed recovers it within 10%. The smoothed line is what
  gets written to the output CSV.
- **Friction-ellipse coupling** in `compute_velocity_profile` (default
  on; `--no-friction-ellipse` restores the old behavior for
  comparison). Accel/brake budget is scaled by
  `sqrt(1 − (v²κ/a_lat_max)²)` so corner-entry/exit speeds stop
  assuming the tires can brake at full force mid-corner. Strictly more
  conservative than the old profile.
- **Regenerate any profiled racing lines** you care about — existing
  CSVs still load fine, but they were paced with the old math.

### gap_follow — invalid-beam handling + physical clearance model

- Scan-processing logic split out of the node into importable
  `gap_follow/gap_logic.py` with its own plain-pytest test dir
  (`src/gap_follow/test/`), mirroring pure_pursuit's `racing_math.py`
  pattern.
- **NaN/invalid beams are no longer 0.0m obstacles.** Previously a
  single NaN dropout became the "closest obstacle": spurious emergency
  stops on scan noise, and the safety bubble carved around a beam with
  nothing in it. Invalid beams are now excluded from the e-stop check
  but stay non-free for gap selection (never steer into a blind spot);
  `+inf` correctly counts as free space at `max_range`.
- **Disparity extender**: every sharp range jump (obstacle edge) is
  extended by half a car width at that edge's distance, so gap selection
  is clearance-aware at *every* edge, not just around the single closest
  point.
- **Width-based safety bubble**: the fixed 20° bubble
  (`bubble_angle_deg`, **parameter removed**) is replaced by
  `atan2(car_width/2 + safety_margin, closest_dist)` — a fixed angle was
  far too little clearance up close and wastefully much far away.
- **Gaps narrower than the car are discarded** outright instead of being
  eligible to win as the least-bad option.
- New parameters (`config/gap_follow.yaml`): `car_width` (0.30),
  `safety_margin` (0.10), `disparity_threshold` (0.4).

### Machine setup (not a repo change)

- The `range_libc` **Python** module was never installed on this Jetson
  — `colcon build` only builds the C++ lib, so even `particle_filter`
  could not import it. Installed user-level:
  `pip3 install --user --break-system-packages cython`, then
  `python3 setup.py install --user` in `src/range_libc/pywrapper`.
  A fresh OS/user setup must repeat this (it survives
  `rm -rf build install log`, but lives in `~/.local`, not the repo).
