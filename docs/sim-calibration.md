# Simulator calibration: making the simulated car drive like the real one

> **Who this is for:** anyone about to trust a simulator result for this car, or about to measure the car again on a new floor.
> **Read first:** [course-analysis.md](course-analysis.md), for how a recorded run is located against a map. [simulator.md](simulator.md) for what the simulator is.
> **You'll be able to:** select a measured floor, check it against the real car, and measure a new floor the same way.
> **Time:** 15 minutes to read. Re-running the validation takes about 10 minutes.

The simulator has measured **floor profiles**: named sets of numbers taken from real runs on one floor with one set of tyres. The 2026-10-30 profile laps within 1% of its validation run.

**2026-11-06:** the [new car 2 profile](#car-2-floor-of-2026-11-06) improves held-out steering and speed replay, but still wedges on course 06a in five of six seeds. The highlights and tables below describe 2026-10-30 unless labelled otherwise.

## Highlights

- **Laps within 1% on a run it never saw.** Real run `20261030-184826` lapped in 20.70 s. The simulator, driving that run's own launched settings, lapped in 20.56 s median over 24 laps (20.30-20.72). Before calibration it wedged at the bottom-right bend in 6 of 6 seeds.
- **Speed within 0.24 m/s, section by section.** Over 21 sections of 2.5 m, the RMS speed difference is 0.24 m/s, the worst 0.40 m/s.
- **The real stop, to 1 cm.** The real car stopped from 5.41 m/s (on its wheels) in 3.39 m, measured by the wall ahead. The simulator takes 3.38 m. It took 3.67 m before.
- **Grip as a curve, not one number.** The real car follows the request to about 2.5 m/s², then rolls off gradually. A [progressive grip model](#grip) reproduces that curve to 0.17 m/s² RMS, against 0.31 before.
- **Wall fragments that look like cars.** Real scans break walls into pieces that the opponent tracker (the part of `gap_follow` that spots other cars, [gap-follow.md](gap-follow.md)) mistakes for moving cars, 22.5 times per 100 s. The old simulator produced none. Now it produces 17.2.
- **Fit on some runs, judged on others.** Six runs were used to fit, six were held back, and the table below is scored on held-back runs.

**Limits:** one floor, one day, one car. Nothing here is proven until it is driven on the floor. What it still gets wrong is listed in [What it cannot reproduce](#what-it-cannot-reproduce).

### Why it exists

On 2026-10-30 the simulator said settings were fast and safe that the real car could not use, and wedged the car at a bend the real car drove every lap. Tuning against a simulator that disagrees with the floor tunes the simulator, not the car.

## Using it

**Any terminal, drivers not needed:**

```bash
source /opt/ros/jazzy/setup.bash && source ~/racerbot-ws/install/setup.bash
cd ~/racerbot-ws
python3 tools/f1tenth_sim/eval_gap.py --track course_2026_10_30_scans --floor sfu_course_2026_10_30 --seeds 6
```

**Working when:** each seed prints a line with a lap time near 20 s and no contact.

`--floor` works on `eval_gap.py`, `eval_gap_multi.py`, `sweep_gap.py`, `sweep2.py` and `trace_gap.py`. Without it you get the plain calibration, exactly as before.

Use the floor profile with the track it was measured on, `course_2026_10_30_scans`. Its grip belongs to that floor. On another track it describes a car on the wrong floor.

## What was measured

Every item below came from the runs of 2026-10-30. They use the [LiDAR](glossary.md#lidar) (the spinning laser that measures distance to walls) and [odometry](glossary.md#odometry--odom) (the car's own estimate of its speed and turning). Each run was located against `data/racerbot_auto/20261030-185036-fixed` with `loc.py`.

**The split.** Fit runs: `185036`, `184556`, `175728`, `180355`, `181616`, `183607`. Held-out runs: `184826`, `183134`, `181003`, `181819`, `184316`, `175124`. Both lists are in `sim_fidelity/calibration.py`, and a test checks they do not overlap.

| # | Item | Real car | Before | After |
|---|---|---|---|---|
| 1 | Grip: lateral achieved for 3.0-3.5 asked | 2.61 m/s² | 2.93 (sharp limit at 5.4) | 2.65 |
| 1 | Grip: whole curve, RMS against real | | 0.31 m/s² | 0.17 m/s² |
| 2 | Steering: yaw rate error, real commands replayed, held-out runs | | 0.233 rad/s | 0.166 rad/s |
| 2 | Steering: same, fit runs | | 0.198 rad/s | 0.141 rad/s |
| 2 | Steering trim (rack left of command) | 0.0105 rad | 0 | 0.0105 rad |
| 2 | Full stick 0.436 rad turns | 0.423 rad both sides | 0.436 | 0.423 |
| 3 | Command to speed: delay | 75 ms | 50 ms | 75 ms |
| 3 | Top speed, on the wheels | 5.39-5.44 m/s peaks | 5.04 | 5.23 |
| 3 | Speed, real commands replayed, held-out runs, RMS | | 0.73 m/s | 0.82 m/s (worse) |
| 4 | Stop from 5.41 m/s (wheels), by the wall | 3.39 m, 1.40 s | 3.67 m, 1.40 s | 3.38 m, 1.20 s |
| 4 | Stop from 3.10 m/s (wheels), by the wall | 1.31 m, 0.93 s | | 1.23 m, 0.72 s |
| 5 | Wheel speed over true speed | 1.033 | 1.000 | 1.033 |
| 5 | `/odom` yaw rate over gyro | 1.282 | 1.189 | 1.290 |
| 5 | `/odom` yaw leads the gyro by | 0.112 s | 0.162 s | 0.137 s |
| 6 | LiDAR noise, parked | 3 mm near, 10 mm past 4 m | 10 mm flat | 3 mm + 1.5 mm per m |
| 6 | Beams with no echo | 4 in 10 million | 0.2% | 0 |
| 6 | Beams hitting the car itself | 23, at -135 to -130° | none | the same 23 |
| 6 | Range jumps over 0.3 m, per scan | 7.8 | 3.3 | 10.6 |
| 6 | Lone returns, per scan | 12.0 | 2.7 | 6.1 |
| 8 | Map walls against real scans, held out | | +6.4 cm too close | +1.0 cm (contact walls), +2.4 cm (beam walls) |

"Before" is the plain calibration on the saved map `course_2026_10_30`. Rows 2 and 3 replay real `/drive` commands through the plant on an empty floor; the rest drive the held-out run's settings on the course.

Item 7, geometry and mass, has no error to measure. See [Measured and assumed](#measured-and-assumed).

## Did it drive like the car?

Scored on held-out run `20261030-184826`, a `gap_follow` lap at the 3.5 m/s cap. The simulator drove that run's own launched parameters, read from its `metadata/params`, with the code paths of that day (the four parameters in `OLD_CODE` set to what the code then did). Six seeds, 115 s each.

| Measure | Real | Before | After |
|---|---|---|---|
| Flying lap, median | 20.70 s | no lap | 20.56 s (24 laps) |
| Flying lap, range | 20.65-20.74 | | 20.30-20.72 |
| Seeds with wall contact | 0 | 0 of 6 | 0 of 6 |
| Seeds wedged at the bottom-right bend | 0 | 6 of 6 | 0 of 6 |
| Bend: share of time in `corner_fallback` | 0.31 | 0.08 | 0.48 |
| Bend: mode changes per pass | 2.7 | 1.0 | 3.5 |
| False moving cars per 100 s | 22.5 | 0.0 | 17.2 |
| Section speed, RMS against real | | | 0.24 m/s |
| Section speed, worst / bias | | | 0.40 / -0.05 m/s |
| Stop from 5.41 m/s | 3.39 m | 3.67 m | 3.38 m |

The fit runs score the same way: real laps 20.59-20.75 s.

**To check it yourself, any terminal:**

```bash
R=data/racerbot_runs/20261030; M=data/racerbot_auto/20261030-185036-fixed
python3 tools/run_analysis/loc.py $R-184826 --map $M
python3 tools/run_analysis/car_measure.py $R-184826 $R-183134 $R-181003 $R-181819 $R-184316 $R-175124 --json /tmp/held.json
python3 tools/f1tenth_sim/validate_calibration.py real $R-184826 --line $M/raceline_raw.csv \
    --measured /tmp/held.json --stop-m 3.39 --stop-s 1.40 --out /tmp/real.json
python3 tools/f1tenth_sim/validate_calibration.py sim --params-from $R-184826 --line $M/raceline_raw.csv \
    --out /tmp/before.json
python3 tools/f1tenth_sim/validate_calibration.py sim --params-from $R-184826 --line $M/raceline_raw.csv \
    --floor sfu_course_2026_10_30 --out /tmp/after.json
python3 tools/f1tenth_sim/validate_calibration.py table real=/tmp/real.json before=/tmp/before.json after=/tmp/after.json
```

`loc.py` needs each of the six held-out runs located before `car_measure.py` reads them; pass them all to `loc.py` the first time.

**Working when:** the last command prints the table above, give or take a few hundredths. The simulator side varies with the seed.

## What changed in the simulator

All of it lives in the floor profile `sfu_course_2026_10_30` in `tools/f1tenth_sim/sim_fidelity/calibration.py`, with a source for every number. The plain calibration is unchanged, and a test pins that.

### Grip

The old model was a sharp friction circle: the car turned exactly as asked up to 0.55 × 9.81 = 5.4 m/s², then not at all harder. The real car has no edge like that. It follows the request, then each extra unit asked buys less:

| Asked, m/s² | 2.25 | 3.25 | 4.5 | 5.5 | 7 | 10 |
|---|---|---|---|---|---|---|
| Real, achieved | 2.15 | 2.62 | 3.00 | 3.34 | 3.70 | 4.10 |

`ProgressiveGrip` in `sim_fidelity/grip.py` takes that table. It never delivers more than was asked, and a test checks that for the shipped profile.

The table in the profile is the real one times 1.15, because the simulator's tyres, which are stock F1TENTH ones, understeer by themselves on top of it. The 1.15 was fitted by replaying real commands against the gyro (1.15-1.2 equally good).

Braking does not cost cornering in this model. The real car showed almost none: asked 3-5 m/s², it achieved 2.73 while braking harder than 3 m/s² and 2.71 at steady speed.

### Steering

The trim and the full-stick angles above, with straight lines between. The command reaches the rack after two control ticks, 50 ms, then the servo closes on it at 15 per second.

<details>
<summary>Why not the measured linkage table (safe to skip)</summary>

Binning the rack angle (from yaw at low lateral acceleration) by the command gave a table that read 30% less rack at 0.1-0.2 rad. Replaying real commands through the plant rejected it: yaw rate error 0.24-0.31 rad/s with the table against 0.14 without.

Binning by a command that is lagged and noisy flattens the slope. Statisticians call it regression dilution.

The replay compares whole time series and does not suffer from it. That is why every steering and grip number here was chosen by replay.

`car_measure.py` still prints the table, because it is how anyone will first look at the linkage. Don't fit to it.

</details>

### Speed and braking

The VESC's speed loop is modelled as proportional each way, with a ceiling. Accelerating: gain 7 per second, at most 4 m/s². Braking: gain 7 per second, at most 4.5 m/s². Commands act after 75 ms.

The braking ceiling comes from a fit-run stop (run `180355`, 3.10 m/s, 1.31 m by the wall). It predicts the held-out 5.41 m/s stop of run `175124` at 3.44 m, against 3.39 measured.

### Odometry

`/odom` speed reads 1.033 times true speed, so the simulated `/odom` does too. The speed command is scaled by 4614 / 4447.98 / 1.033 = 1.004. That follows from `vesc.yaml` commanding at 4614 eRPM per m/s while `/odom` reads at 4447.98.

### LiDAR

Noise grows with range, no beam goes missing, and the 23 beams that see the car's own rear right corner read what they read on the real car.

### The map

The saved map's walls are three cells thick at the median, one to six, and stand about 6 cm proud of where the real scans land. `tools/run_analysis/scan_map.py` rebuilds them from the fit runs' scans into two layers:

- `course_2026_10_30_scans.png` is **what beams see.** It holds the cells beams end in, with the gaps the real walls have. The gaps make the fragments the tracker mistakes for cars.
- `course_2026_10_30_scans_contact.png` is **what the car hits.** The saved walls are peeled back to where beams end, never opened, so a car cannot drive through a gap a beam slipped through.

The plant judges contact on the second layer when a track has one.

## Measured and assumed

| Value | Status |
|---|---|
| Wheelbase 0.36 m, track width 0.30 m, LiDAR 0.26 m ahead of the rear axle | measured with a tape, 2026-08-24 |
| Steering limits, trim, full stick | measured: circles 2026-10-16, yaw on 2026-10-30 |
| Grip curve, steering delay, speed loop, braking, odometry, LiDAR | measured, this page |
| Overall length 0.535 m | Traxxas specification, not measured |
| Mass 3.74 kg, yaw inertia, centre of gravity height and position | **assumed**: F1TENTH Gym defaults, never weighed |
| Cornering stiffness | **assumed**: Gym defaults. The fitted grip curve absorbs their error on this floor, which is why the curve is a floor profile and not a fact about the tyres |

`python3 tools/f1tenth_sim/sim_fidelity/calibration.py` prints all 43 values with where each came from. Six are still Gym defaults.

## What it cannot reproduce

- **The soft end of a hard stop.** Real stops slow gently below about 1 m/s and take 1.40 s from 5.41 m/s. The model gets the distance (3.38 m) but stops in 1.20 s. A gain low enough to match the time makes ordinary driving brake too gently, and laps came out 5% fast.
- **The top speed overshoot.** The real wheels peak at 5.39-5.44 m/s, above the 5.23 the VESC limit should allow. The model stops at 5.23.
- **Open-loop speed.** Replaying real commands for 60 s with no controller, the speed error went from 0.73 to 0.82 m/s on held-out runs. In closed loop, the number that matters for lap times, sections are within 0.24 m/s.
- **Fewer false cars than real.** 17.2 per 100 s against 22.5. The rate swings from 16 to 41 between beam-layer thresholds of 25% and 28%, so treat it as good to about ±30%.
- **Different wall texture.** The simulator has more big range jumps than real scans (10.6 against 7.8 per scan) and half the lone returns (6.1 against 12.0). Real lone returns are mostly edge effects of the laser itself, not modelled.
- **A left/right grip difference.** At high asks the right side achieves 5-10% less than the left. The model is symmetric.
- **More bend flicker than real.** The simulator switches between `gap_follow` and `corner_fallback` 3.5 times per pass at the bottom-right bend, against 2.7.
- **LiDAR latency below one tick.** Real scans line up with the gyro to within 10 ms; the simulator delays them one 25 ms tick.
- **Anything about another floor.** The grip curve is this floor and these tyres on 2026-10-30.

## Car 2 floor of 2026-11-06

Select `--floor sfu_course_2026_11_06` for car 2's floor and tyres that day. This profile improves replay of real commands and completes every course 06b simulation, but **does not yet reproduce the reliable 06a hairpin turn**.

### Runs and measurement

The plant (the model of the car's response to commands) was fitted on three course 06a runs. Entire recordings were held out for validation:

| Role | Run | Controller recorded in `metadata/params` |
|---|---|---|
| Fit, 06a | `20261106-163633` | `gap_follow_node` |
| Fit, 06a | `20261106-164401` | `pure_pursuit_node` |
| Fit, 06a | `20261106-170803` | `gap_follow_node` |
| Held out, 06a | `20261106-171030` | `gap_follow_node` |
| Held out, 06b | `20261106-184753` | `gap_follow_node` through `auto_map_race_node`, mapping only |

**06b has only one held-out run and no fit run.** Grip, steering response, speed response, odometry and LiDAR noise describe the car on that floor, not its layout. Fitting those parameters on 06a therefore leaves 06b as an independent different-course check of the plant.

Run `185425` was excluded: it has pure-pursuit metadata but no LB-held samples or nonzero `/drive` speed, and its last bag reports truncation. Run `181002` used a different layout. No lap from `184753` was used to fit the plant.

`loc.py` located the five selected runs against the supplied rebuilt maps. `car_measure.py` measured only the three fit runs; the adjacent [`sfu_course_2026_11_06_measurement.json`](../tools/f1tenth_sim/sim_fidelity/sfu_course_2026_11_06_measurement.json) preserves its output.

| Measurement | Fit-only result | Profile uses |
|---|---|---|
| Grip, 2.5–3 asked | 2.441 m/s² achieved; left 2.496, right 1.650 | Symmetric curve, replay-selected scale 1.05 |
| Grip, 3–3.5 / 3.5–4 / 4–5 asked | 2.596 / 2.870 / 3.051 m/s² achieved | Measured medians at bin midpoints, scaled and capped at the request |
| Lateral grip, braking below −3 / steady ±0.5 m/s² | 2.779 / 2.647 m/s² | No longitudinal grip coupling |
| Wheel speed / scan-to-map true speed | 1.032345 | Same odometry scale |
| Straight steering trim | +0.006443 rad | Same trim |
| Simplified servo replay | 4 ticks, gain 15/s, rate 2.5 rad/s | Full plant replay selects 3 ticks; gain and rate retained |
| True-speed loop replay | Gains 7/s each way; acceleration 4, braking 3 m/s²; delay 4 ticks | Same speed loop; replay RMS 0.174 m/s |
| Hard stops | 2.63–2.66 m/s wheel speed: 0.80 s and 1.00–1.06 wheel-metres; 4.74 m/s: 1.375 s, 3.438 wheel-metres | Recorded evidence, not independent wall-distance measurements |

The whole-run `--lidar` report mixes several parked positions and gives unusably large noise estimates. A single stable window in fit run `163633`, timestamps 1794011831.129–1794011887.418, contains 2,250 scans. Its noise is 2.98 mm below 1 m and 11.87 mm beyond 4 m, with no rear-right self hits.

The window is the longest contiguous stretch below 0.01 m/s wheel speed, split whenever the median adjacent-scan range change exceeds 0.02 m. Applying `car_measure.scan_statistics` gives the committed [`stable_lidar` fixture](../tools/f1tenth_sim/sim_fidelity/sfu_course_2026_11_06_stable_lidar.json). The profile uses 3 mm base noise, the same sensor's earlier 1.5 mm/m range slope, zero dropouts and no self-hit beams.

### Fit and held-out replay

`validate_calibration.py fit-plant` replayed real commands against the gyro, using the first 60 s from the first trusted sample of each run. Selection used mean fit-run yaw-rate RMS only. The speed loop came from true-speed replay, before steering and grip selection.

A 90-candidate grid compared grip scales 0.95–1.35, linkage slopes 0.90–1.15 and delays 2–4 ticks. A 45-candidate refinement selected **grip scale 1.05, linkage slope 1.05 and 3 ticks of steering delay**. The [`replay` fixture](../tools/f1tenth_sim/sim_fidelity/sfu_course_2026_11_06_replay.json) records the axes, best candidates and scores.

The central linkage is `wheel_angle = 1.05 × command + 0.006443`, up to ±0.349 rad commands. It retains the October circle measurements at full stick and the servo endpoints. Its +0.349 command gives an **effective** wheel angle of 0.372893 rad in this plant, rather than the binned table's 0.344420 rad.

That effective linkage compensates the plant's remaining tyre dynamics. It is not a direct measurement of rack geometry. As with the October floor, the binned linkage was diagnostic evidence, not the fitting target.

| Replay measure | Before: 10-30 floor | After: 11-06 floor |
|---|---:|---:|
| Fit yaw-rate RMS | 0.1097 rad/s | 0.0993 rad/s |
| Fit speed RMS | 0.1993 m/s | 0.1806 m/s |
| Held-out mean yaw-rate RMS | 0.1213 rad/s | 0.1032 rad/s |
| Held-out mean speed RMS | 0.2259 m/s | 0.2027 m/s |
| 06a held-out yaw-rate RMS | 0.1345 rad/s | 0.1189 rad/s |
| 06a held-out speed RMS | 0.2786 m/s | 0.2365 m/s |
| 06b held-out yaw-rate RMS | 0.1081 rad/s | 0.0874 rad/s |
| 06b held-out speed RMS | 0.1732 m/s | 0.1689 m/s |

### Wall geometry and lap scores

The existing 06a scan track passed the wall check and was retained. Course 06b needed a new track, `car2_course_2026_11_06b_scans`, built with `scan_map.py` defaults from `184753`. Its centreline links to the original track's centreline.

**Wall geometry is the course's shape, not a plant parameter.** Building it from the held-out run does not contaminate the held-out check of the plant: grip, steering and speed response were fitted on 06a runs only. The 06b wall check itself is in-sample, so it is not an independent validation of map geometry.

The same check as `scan_map.py --check` used every 25th eligible scan and every fourth beam. Positive median error means the simulated wall is too close:

| Track | Contact-wall median error | Beam-wall median error |
|---|---:|---:|
| Existing 06a `_scans`, held-out `171030` | −0.3 cm | +1.8 cm |
| Original 06b, `184753` | +4.8 cm | +4.8 cm |
| New 06b `_scans`, `184753`, in-sample | +1.6 cm | +2.6 cm |

The new 06b layers reduce the bias, though they remain slightly above October's +1.0 / +2.4 cm reference. The original `car2_course_2026_11_06b` files are unchanged.

Both profiles drove each held-out run's own parameters with `--params-from --launched-code`, steering cap **0.349 rad**, six seeds 12345–12350, two workers and 115 s per seed. Before/after use identical tracks and starts. Lap times are flying laps, excluding each seed's first lap.

| Course and run | Real flying median | Before: 10-30 floor | After: 11-06 floor |
|---|---:|---:|---:|
| 06a, `171030`, speed cap 5.0 m/s | 11.50 s, 8 laps | No flying lap | 13.84 s, 16 laps, range 12.82–15.10 |
| 06a, seeds eventually wedged | | 6/6 | 5/6 |
| 06b, `184753`, speed cap 3.5 m/s | 9.14 s, 2 laps | 8.92 s, 66 laps | 9.33 s, 66 laps, range 9.20–9.54 |
| 06b, seeds eventually wedged | | 0/6 | 0/6 |
| Simulated seeds with body contact, each course | | 0/6 | 0/6 |

The current localisation and scoring give 11.505 s median for 06a, versus the earlier 11.58 s real-run reference. The rounded flying times are 11.97, 11.29, 11.57, 11.44, 11.62, 11.71, 11.37 and 11.39 s. Both references are substantially faster than the new simulation; neither was a fitting target.

**Hairpin verdict: partially improved, still not reproduced reliably.** Four seeds complete at least one lap at 0.349 rad, but only seed 12346 runs all 115 s without wedging.

Five seeds stop near x=9.37–9.68 m, y=−0.32 to −0.03 m on the hairpin approach. Completed laps are 20% slower than the current real median, so the successful laps alone must not be treated as a clean validation.

Contact comes from `FidelityPlant.chassis_contact_steps`, the harness's body-versus-wall check, never Gym's collision flag. The real-card tool's zero contact field is not an independent body-clearance measurement of the real car.

The 06a start is the supplied `9.43,1.28,101.5` past the hairpin. For 06b, the first located sample at or beyond one lap of progress in the longest LB hold is timestamp **1794019693.186**: **`0.5438070161,0.2136068187,349.25150385`** (x, y, yaw in degrees).

That 06b pose starts the first flying lap, found with `Course.project`, `Course.unwrap` and the run map's `raceline_raw.csv`. It is not the end-of-mapping `start_pose.yaml`. The simulator begins at rest at that pose and excludes its first lap.

### Reproduce the scores

**Dev-box terminal, from this repository's root folder; no drivers required.** The source runs and rebuilt maps live in the parent checkout on this box. These commands write only to scratch space; adjust `BASE` if those read-only recordings are elsewhere.

```bash
BASE=/home/bryan/racerbot-ws/.claude/worktrees/map-racer-faster-than-gap/data/rb2
R=$BASE/racerbot_runs; M=$BASE/racerbot_auto
O=.oc-runs/sim-floor-06-recheck
export RB_NO_QUEUE=1 RB_WS="$PWD"
rb python3 tools/run_analysis/loc.py "$R/20261106-163633" "$R/20261106-164401" \
    "$R/20261106-170803" "$R/20261106-171030" --map "$M/20261106-164002-rebuilt" --out "$O/loc" --workers 2
rb python3 tools/run_analysis/loc.py "$R/20261106-184753" \
    --map "$M/20261106-184753-rebuilt" --out "$O/loc" --workers 2
rb python3 tools/run_analysis/car_measure.py "$R/20261106-163633" "$R/20261106-164401" \
    "$R/20261106-170803" --loc "$O/loc" --json "$O/fit.json"
```

**Working when:** the measurement lists exactly the three fit runs and wheel-over-true near 1.0323. To replay a locked profile without another parameter search, use an empty grid; repeat with the 10-30 floor for the before column:

```bash
rb env RACERBOT_CAR=rb2 python3 tools/f1tenth_sim/validate_calibration.py fit-plant \
    "$R/20261106-171030" "$R/20261106-184753" --loc "$O/loc" \
    --floor sfu_course_2026_11_06 --grid '{}' --workers 2 --out "$O/replay.json"
rb python3 tools/f1tenth_sim/validate_calibration.py real "$R/20261106-171030" \
    --line "$M/20261106-164002-rebuilt/raceline_raw.csv" --loc "$O/loc" --out "$O/06a-real.json"
rb python3 tools/f1tenth_sim/validate_calibration.py real "$R/20261106-184753" \
    --line "$M/20261106-184753-rebuilt/raceline_raw.csv" --loc "$O/loc" --out "$O/06b-real.json"
rb env RACERBOT_CAR=rb2 python3 tools/f1tenth_sim/validate_calibration.py sim \
    --params-from "$R/20261106-171030" --launched-code --floor sfu_course_2026_11_06 \
    --track car2_course_2026_11_06a_scans --line "$M/20261106-164002-rebuilt/raceline_raw.csv" \
    --start 9.43,1.28,101.5 --seeds 6 --workers 2 --seconds 115 --out "$O/06a-after.json"
rb env RACERBOT_CAR=rb2 python3 tools/f1tenth_sim/validate_calibration.py sim \
    --params-from "$R/20261106-184753" --launched-code --floor sfu_course_2026_11_06 \
    --track car2_course_2026_11_06b_scans --line "$M/20261106-184753-rebuilt/raceline_raw.csv" \
    --start 0.543807016147227,0.21360681867323164,349.25150384970954 \
    --seeds 6 --workers 2 --seconds 115 --out "$O/06b-after.json"
rb python3 tools/f1tenth_sim/validate_calibration.py table \
    real="$O/06b-real.json" after="$O/06b-after.json"
```

**Working when:** replay prints yaw RMS near 0.103 rad/s and speed RMS near 0.203 m/s; the final table prints a 9.33 s simulated median. The 06a run should reproduce the reported wedges, not six clean seeds. Warnings for empty sector bins are expected; those bins are not used in these verdicts.

To rebuild 06b's layers in scratch space and repeat its in-sample wall check, use the same terminal:

```bash
rb python3 tools/run_analysis/scan_map.py "$R/20261106-184753" \
    --map "$M/20261106-184753-rebuilt" --loc "$O/loc" --out "$O/06b-walls" \
    --check "$R/20261106-184753"
```

**Working when:** contact and beam medians print +0.016 m and +0.026 m. The committed [`laps` fixture](../tools/f1tenth_sim/sim_fidelity/sfu_course_2026_11_06_laps.json) contains the scores, per-seed outcomes, start selection and wall checks. No raw bags or localisation arrays are committed.

### Limitations

- **06a still fails the intended hairpin validation.** Better open-loop replay has not made the closed-loop trajectory match. Steering and grip remain symmetric; the measured left/right difference is large in some bins. No parameter was retuned toward a lap time or a successful seed.
- **Effective linkage is model-dependent.** It absorbs remaining tyre-model error. Commands beyond ±0.349 rad use earlier circle measurements and are unvalidated on this day's floor.
- **Braking is fitted to true-speed replay, not a new wall-measured stop.** The tool's synthetic 5.41 m/s wheel-speed stop grows from 3.38 to 5.02 m with the new speed loop. Neither figure is a validation against a 6 November stop from that speed; the fit run's largest recorded stop starts at 4.74 m/s on the wheels.
- **06b has two real flying laps.** Its 66 simulated laps reuse one measured layout; this is a plant cross-check, not broad evidence across floors. Wall residuals remain centimetre-scale and were checked on the same run used to build the walls.
- **LiDAR texture and mounting effects remain approximate.** The noise slope is retained from the same sensor's October measurement. The 10-30 bend-flicker box does not cover these courses, so its empty flicker scores are omitted.
- **Mass, inertia and cornering stiffness retain the base calibration's assumptions.** This profile does not turn those defaults into measurements.

### Regression checks and deliberate breaks

`test_floor_2026_11_06.py` reconstructs the grip curve and linkage from recorded measurements and replay selection, checks every override's source, and drives the actual plant at a crawl. The new contact layer is checked against an independently loaded image distance transform.

| Test name after `test_november_` | Break observed to fail |
|---|---|
| `split_and_sources_exclude_both_validation_runs` | Added held-out `184753` to the fit tuple |
| `grip_reconstructs_the_measured_bins_and_replay_scale` | Made `with_floor` return the plain base; separately raised grip scale by 10% |
| `linkage_and_delay_follow_gyro_replay_not_binned_rack` | Made `with_floor` return the plain base |
| `speed_loop_and_odometry_follow_true_speed_measurement` | Same plain-base stub |
| `sensing_and_braking_coupling_have_measured_sources` | Same plain-base stub |
| `floor_reaches_the_plant_and_new_track_contact_layer` | Same plain-base stub; plant rack settled at 0.100 instead of 0.111443 rad |

Each break failed under both car selections in the tool test runner. All breaks were reverted before the final suite run.

**Dev-box terminal, repository root:** the full simulator suite compares the mirror with a real [node](glossary.md#node), a running ROS program, so `drive_intent` must be built in this checkout's overlay. `rb` sources that overlay and isolates the ROS graph.

```bash
RB_NO_QUEUE=1 RB_WS=$(pwd) rb colcon build --symlink-install --packages-select drive_intent
RB_NO_QUEUE=1 RB_WS=$(pwd) rb python3 -m pytest tools/f1tenth_sim/tests/ -q -p no:cacheprovider
```

**Working when:** the build finishes with one package and pytest reports **572 passed**, including all 12 new cases (six tests under two car selections). Without that local build, two existing node-parity cases fail to import `drive_intent`; no test was changed to bypass them.

## Measuring a new floor

1. Drive at least six runs on it, including some hard corners and one hard straight-line stop from over 3 m/s, and save a map.
2. Locate every run: `python3 tools/run_analysis/loc.py <runs> --map <map>`.
3. Split them: fit runs and held-out runs, with the stop in each if you can.
4. Measure: `python3 tools/run_analysis/car_measure.py <fit runs> --json fit.json`, and `--lidar` on a run where the car sat still for a while.
5. Rebuild the walls: `python3 tools/run_analysis/scan_map.py <fit runs> --map <map> --out tools/f1tenth_sim/maps/<name> --check <held-out runs>`.
6. Add a `Floor` to `FLOORS`, with a source for every number, and fit the grip scale and steering delay with `validate_calibration.py fit-plant`.
7. Score it on a held-out run with `validate_calibration.py`, as in [Did it drive like the car?](#did-it-drive-like-the-car).

<details>
<summary>The tools, one line each (safe to skip)</summary>

| Tool | Does |
|---|---|
| `tools/run_analysis/car_measure.py` | Wheel over-read, `/odom` yaw against the gyro, steering trim and lag, the grip curve by side, the speed loop, hard stops; `--lidar` for noise, dropouts and self hits |
| `tools/run_analysis/scan_map.py` | Rebuilds a map's walls from where real scans landed, in a beam layer and a contact layer |
| `tools/f1tenth_sim/validate_calibration.py real` | Scores located runs: laps, sections, bend flicker, false movers, scan texture |
| `validate_calibration.py sim` | Drives the simulator on a run's launched settings and scores it the same way |
| `validate_calibration.py fit-plant` | Ranks plant settings by replaying real commands against the gyro |
| `validate_calibration.py table` | Prints scored cards side by side |

</details>

## The 2026-11-11 race-day floor (car 2)

`sfu_course_2026_11_11_rb2` is measured from the race-day runs `20261111-152132`, `-135934` and `-142254` (`car_measure.py`, `grip_truth.py`, `fit-plant`); the evidence sits beside it as `sfu_course_2026_11_11_rb2_{measurement,replay}.json` and `sfu_course_2026_11_11_grip_rb2_vesc.json`. Three things differ from the 11-06 floor.

- **The grip curve falls past its peak.** On the gyro basis the car delivers 2.94 m/s^2 at 3.75 asked and 2.71 at 4.38. `ProgressiveGrip` refuses a falling table by default; the floor sets `grip_curve_falls=True`. The curve is the gyro-basis median per asked bin (bins of 10 or more samples, never above the request), times **1.15**, because the plant's own tyres understeer on top of it.

  **What the floor is calibrated on, and what it is not.** It is calibrated on flying-lap time and braking: at 1.15 the cap-on flying lap is 9.95 s against the real 10.05 s, braking lag and deceleration match (acceptance B), and the open-loop yaw RMS is better than the 11-06 floor. **The 152132 launch run-wide is NOT reproduced**: from s = 8.1 m with the online cap off the simulated car reaches 4.74 and 5.09 m/s at s = 10 and 11 against the real 4.28 and 4.75 and makes contact at s 11.7 to 12.0 in 6/6 seeds. That run is not a valid target either, because the real car's online cap was still binding at s = 8 to 9.5 (a steering of 0.10 rad gives a 3.33 m/s cap). The grip scale lies between 1.10 (flying lap 10.12 s) and 1.25 (best open-loop yaw RMS, flying lap 9.69 s); 1.15 is where the flying lap is closest to the car's without being slower than it. The six cells and the rule are in `sfu_course_2026_11_11_rb2_replay.json` (`closed_loop_selection`).

  **A measured acceleration table did not help.** `Calibration.accel_by_speed` (opt-in, no floor sets it) replaces the speed loop's acceleration ceiling by a table against wheel speed. `tools/run_analysis/accel_by_speed.py` measured rb2's from the three race-day runs: p75 of the wheel acceleration per speed bin, where the command is 0.5 m/s or more above the wheels, is non-monotone (4.26, 2.23, 3.26, 4.18, 4.58, 4.36 m/s^2 for bins 0-1 to 4-4.5; nothing above 4.5), and where the loop is saturated (command 1.0 m/s above) the capability is about 6 to 7 m/s^2 up to 4 m/s. The slow real gains at 3.25 to 4 s were command-limited. With that table the simulated launch from s = 8.1 m was faster, not slower: 4.87 and 5.10 m/s at s = 10 and 11 against the real 4.28 and 4.75, with contact at s about 11.5 in 6/6 seeds, so no floor uses it.
- **The speed loop** is `car_measure.py`'s fit on the three runs: accel gain 10, accel max 4, brake gain 5, brake max 4, 4 ticks.
- **Braking lag.** `tools/run_analysis/braking_lag.py` measures the delay that best aligns `/drive` speed with wheel speed on rising and on falling commands, and the deceleration in the hardest braking zones. On `152132` the real car gives 0.2375 s and 0.2375 s, and 2.98, 2.90, 2.79 m/s^2. `tests/test_floor_2026_11_11_rb2.py` replays the run's commands (fixture `sfu_course_2026_11_11_rb2_braking.npz`) and holds the floor within 0.05 s and 0.3 m/s^2 of those.

`tools/f1tenth_sim/start_replay.py` starts the map racer at a recorded pose (at rest, or already moving with `--start-s/--start-speed/--start-right/--start-heading-error`), prints speed and cross-track at chosen arc lengths per seed, and has `--online-lateral 99` (the online curve cap off), `--grip-scale` and `--grip-coupling`. `--check-profile-alone` drives a speed plan with the cap off from standing starts round the line and flying laps and exits 1 on any contact: the gate for a new plan.

## Speed on the calibrated floor

Run on 2026-10-31, after validation passed. Each row is 30 seeds in both directions: 60 laps from a standing start, on `course_2026_10_30_scans` with the floor profile.

"Time trial" is `aim_depth_bias 1.0` with opponent awareness off, as `time_trial:=true` sets.

| Settings | Clean laps | Lap | Closest body to wall |
|---|---|---|---|
| Shipped default (opponent awareness on) | 60/60 | 20.71 s | 9.1 cm |
| Time trial, lateral 3.0, steering 0.349 rad (shipped `time_trial:=true`) | 59/60 | 18.10 s | 1.4 cm |
| Time trial, lateral 3.5, steering 0.349 | 53/60 | 17.86 s | 0.4 cm |
| Time trial, lateral 3.0, steering 0.436 | 59/60 | 17.88 s | 4.3 cm |
| **Time trial, lateral 3.25, steering 0.436** | **60/60** | **17.74 s** | 4.2 cm |
| Time trial, lateral 3.5, steering 0.436 | 59/60 | 17.75 s | 3.3 cm |
| Time trial, lateral 3.75, steering 0.436 | 60/60 | 17.80 s | 3.5 cm |

A first pass at 6 seeds also ran lateral 4.0: 5 of 12 clean at steering 0.349, 9 of 12 at 0.436.

**What it says.** The steering limit is the lever. At 0.349 rad the car runs out of lock in the tight bends when allowed more lateral acceleration.

At 0.436 rad, full stick, lateral 3.25-3.75 all lap in about 17.75 s. Past that, the VESC's 5.04 m/s limit and the line cap the lap, not grip.

Opponent awareness costs about 0.5 s on one car (20.25 s without it, 6-seed pass), from the false cars.

**The fastest settings the measurements support:** time trial with `max_lateral_accel 3.25` and `max_steering_angle 0.436`, 17.74 s in the simulator against 20.6-20.9 s driven on 2026-10-30. Lateral 3.5 and 3.75 are no faster in the simulator, only closer to the wall.

**The risk, stated plainly.** The closest the body came to a wall in 60 laps was 4.2 cm, and the simulator puts walls within about 2 cm of the real ones. The real car may touch a wall. Steering 0.436 rad is full stick: the servo reaches it on both sides, but autonomy has never commanded it. These are not shipped defaults; the floor decides.

## Trying it on the floor

This needs `main` from 2026-11-01 or later built, because `time_trial:=true` arrived then. Put the car on a stand first, so the wheels are off the ground.

**Terminal 1, once:** update and build.

```bash
cd ~/racerbot-ws && git pull
source /opt/ros/jazzy/setup.bash && colcon build --symlink-install --packages-select gap_follow
```

**Working when:** colcon ends with `Summary: 1 package finished`.

**Terminal 1:** the hardware. Leave it running.

```bash
source /opt/ros/jazzy/setup.bash && source ~/racerbot-ws/install/setup.bash
ros2 launch f1tenth_stack bringup_launch.py
```

**Working when:** the log settles with no `[ERROR]` lines. `ros2 run racerbot_doctor doctor` in another terminal names anything that is not up.

**Terminal 2, step 1:** shipped time trial, capped at the speed the course was driven at on 2026-10-30.

```bash
source /opt/ros/jazzy/setup.bash && source ~/racerbot-ws/install/setup.bash
ros2 launch gap_follow gap_follow_launch.py time_trial:=true max_speed:=3.5
```

**Working when:** the wheels turn only while LB is held. Then on the floor, it laps.

**Terminal 2, step 2:** the fastest settings above, full speed. Stop step 1 with Ctrl+C first.

```bash
ros2 run gap_follow gap_follow_node --ros-args \
  --params-file $(ros2 pkg prefix --share gap_follow)/config/gap_follow.yaml \
  -p aim_depth_bias:=1.0 -p enable_opponent_awareness:=false \
  -p max_lateral_accel:=3.25 -p max_steering_angle:=0.436
```

**Working when:** the `gap_follow` [node](glossary.md#node) (the running program) logs that it is waiting for LB, and the car laps when LB is held.

**If it doesn't:** a refused parameter stops the node at start with the reason. Check the spelling of each `-p`.

Every run is recorded by bringup's `auto_record`. Compare it with `laps.py` from [course-analysis.md](course-analysis.md): the simulator says 17.74 s.

## Simulator numbers elsewhere that are now out of date

- [gap-follow.md](gap-follow.md), "Speed on the real course", Results: the course lap times (21.3-22.1 s new default, 18.8-19.3 s time trial) and the friction 0.55/0.42 rows were measured on the plain calibration. On the floor profile the same settings lap in 20.71 and 18.10 s (standing start).
- [../tools/f1tenth_sim/README.md](../tools/f1tenth_sim/README.md): the speed loop and friction fitted to run `20261019-200703` still describe that floor; the floor profile overrides them for this one.
- [sim-fidelity-audit.md](sim-fidelity-audit.md): findings F2-F4 (no friction limit, no ellipse) describe the plain calibration and still hold there.

## Safety

The simulator is not the floor. Bring any setting from this page to the car in the order in [writing-your-own-node.md](writing-your-own-node.md#testing-before-its-on-wheels):

1. Wheels off the ground first.
2. Then the floor, at low speed.
3. The driver holds LB the whole time, ready to let go. LB is the deadman button on the F710 controller: releasing it stops every autonomy [node](glossary.md#node) (a program that drives the car), because each one refuses to drive without a live hold. That is what stops the car when the simulator was wrong.
