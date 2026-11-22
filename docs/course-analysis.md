# Course analysis: true lap times and speed from a recorded run

> **Who this is for:** anyone who has driven a recorded run on a course that has a saved map, and wants to know how fast it really was and where the time went.
> **Read first:** [run-diagnostics.md](run-diagnostics.md) for where runs are recorded. Nothing else is needed: every command here works on a saved run with the car off.
> **You'll be able to:** get true lap times, compare two algorithms point by point, read the real grip curve and real stopping distances, and judge whether time is lost in the line or in the speed.
> **Time:** 10 minutes to read. The worked example takes under a minute to run.

## Highlights

- **True position, not wheel speed.** Each [LiDAR](glossary.md#lidar) [scan](glossary.md#scan) (one sweep of the laser) is matched against the saved map. On run `20261030-185036`, 98% of beams land within 10 cm of a mapped wall.
- **Lap times to a tenth of a second.** That run reads 20.9, 20.7, 20.6, 20.7 and 20.6 s on the 50.74 m course line.
- **Works for any algorithm.** It reads only the recording, so it compares `gap_follow` with Team A's nodes on equal terms.
- **Writes nothing into `data/`.** Results go to `~/.cache/racerbot/loc` unless you say otherwise. The example leaves the run's folder with the same 49 files.
- **Tells you what limits the car.** On that run a perfect speed profile on the path it drove would have lapped in 16.98 s at 3 m/s² of grip, against 20.63 s actual.

**Limits:** it needs a good map of the course, and a course that changed after the map was made will match badly. Tracker replay is open loop: the car's path stays as recorded.

### Why it exists

The wheels over-read true speed by about 4%, and more when they slip. `/odom`'s yaw comes from the steering command rather than from what the car did. So comparing algorithms from [odometry](glossary.md#odometry--odom) (the car's own estimate of its motion) flatters some of them. Measuring against a fixed map removes that.

## The tools

The bag reader uses ROS when available. Without ROS, it falls back to `rosbags.highlevel.AnyReader` on the bag directory; use the analysis environment with `rosbags==0.10.11` installed. A bag is the recording of a run's messages. Tools that only read cached poses (the car's map positions) do not need ROS either.

| Tool | Answers |
|---|---|
| `tools/run_analysis/loc.py` | Where was the car, scan by scan? **Run this first:** the next four read its results |
| `tools/run_analysis/laps.py` | What were the true lap times and best pace? |
| `tools/run_analysis/speed_gain.py` | How much of each commanded speed did the car really deliver, and which motor command gain would have delivered all of it? Car 1's 2026-11-02 recalibration rests on it ([report](rb1-lap-gap-2026-10-30.md)) |
| `tools/run_analysis/sectors.py` | How fast at each point on the course, run against run? |
| `tools/run_analysis/gripcurve.py` | How much lateral acceleration did the car achieve for what it was asked? |
| `tools/run_analysis/lap_theory.py` | What would a perfect speed profile lap in? |
| `tools/run_analysis/stops.py` | How far did real stops take, by the wall ahead? |
| `tools/run_analysis/clearance.py` | What was the run launched with, and how close did it come to things, from raw LiDAR? |
| `tools/run_analysis/oa_replay.py` | What would the opponent tracker make of this run? |
| `tools/run_analysis/simgrip.py` | The simulator's grip curve, to compare with `gripcurve.py` |
| `tools/f1tenth_sim/sweep2.py`, `summ2.py` | Compare settings over tracks, seeds, directions and floor frictions, then print the table |
| `tools/run_analysis/car_measure.py` | Steering, speed loop, stops, odometry and LiDAR numbers for calibrating the simulator; see [sim-calibration.md](sim-calibration.md) |
| `tools/run_analysis/scan_map.py` | Rebuilds a map's walls from where real scans landed |
| `tools/run_analysis/run_summary.py` | What drove in each run of a day, for how long, how fast, how hard; did `/drive` reach the VESC (the motor controller)? No `loc.py` needed |
| `tools/run_analysis/intent_report.py` | Why the car went the speed it did: which cap bound, every stall and how it ended, LB toggles, decisions in a time window or by course section |
| `tools/run_analysis/spot.py` | One spot of the course close up: the map, each run's path, a real scan, a simulated trace, and what the driving [node](glossary.md#node) decided there |
| `tools/f1tenth_sim/trace_deficit.py` | Where a simulated lap loses time against a perfect speed profile on its own path, and which cap bound |
| `tools/f1tenth_sim/trace_multi.py` | Two or more cars in the simulator, with a per-tick log of why one of them slowed |
| `tools/f1tenth_sim/pure_course.py` | `pure_pursuit` on a real course in the simulator, on a racing line (the path to follow) you give it |

## Worked example: run 20261030-185036

This is `gap_follow` mapping the course at its 3.5 m/s cap, on the map rebuilt from the same run.

**Any terminal, car off:**

```bash
source /opt/ros/jazzy/setup.bash && source ~/racerbot-ws/install/setup.bash
cd ~/racerbot-ws
R=data/racerbot_runs/20261030-185036
M=data/racerbot_auto/20261030-185036-fixed
python3 tools/run_analysis/loc.py $R --map $M
python3 tools/run_analysis/laps.py $R --line $M/raceline_raw.csv
```

**Working when:** it prints these lines. A figure well under 0.9 of beams within 10 cm means the map does not fit the run, so don't trust the laps.

```text
20261030-185036 2201 scans, 0.98 of beams within 10 cm, 1 global searches
course line 50.74 m
20261030-185036: 0.98 of beams within 10 cm
   hold  109.4 s,  266.9 m, laps [20.9, 20.7, 20.6, 20.7, 20.6], best pace 19.0 s/lap
```

"Best pace" is the fastest 25 m scaled up to a lap. It flatters the run, and it is there for holds too short to close a lap. Don't quote it as a lap time.

**Then ask where the time went:**

```bash
python3 tools/run_analysis/lap_theory.py --map $M --driven $R --from 50 --to 92 --lateral 3.0
```

**Working when:** it prints the actual pace and the perfect one, here 20.63 and 16.98 s per lap. That gap is about 3.6 s, lost to speed planning on that path rather than to the path itself.

**The rest of the set, on the same run:**

```bash
python3 tools/run_analysis/sectors.py $R --skip 30 --line $M/raceline_raw.csv
python3 tools/run_analysis/gripcurve.py $R
python3 tools/run_analysis/stops.py $R
python3 tools/run_analysis/clearance.py $R
python3 tools/run_analysis/oa_replay.py $R
```

**Working when:** `sectors.py` prints a speed per 2.5 m of course, and `oa_replay.py` reports 24 false movers in 109 s for this run on the current settings. It holds one car, so every "moving car" in that count is false.

`stops.py` prints nothing here, because this run contains no hard stop from above 2 m/s. Run `20261030-175124` has one.

**Asking why, on the same run:**

```bash
python3 tools/run_analysis/run_summary.py --date 20261030
python3 tools/run_analysis/intent_report.py data/racerbot_runs/20261030-175728
python3 tools/run_analysis/spot.py --map $M --at 7.0,-9.8 $R --scan --intent --out /tmp/bend.png
python3 tools/run_analysis/oa_replay.py $R --truth --map $M
```

**Working when:** `run_summary.py` prints a line per run of that day saying what drove it.

On run `175728`, `intent_report.py` lists stalls bound by `follow cap`, several ended by `LB released`: the false-car stalls of 2026-10-30.

`oa_replay.py --truth` reports how far each false "car" really moved and how far it sat from a mapped wall; on `185036` they sit on the walls.

`oa_replay.py --ego true` tells the tracker the car's true speed and the gyro's turn instead of the wheels and the steering command. If false cars vanish, ego motion made them. On 2026-10-30 they did not.

## Reading the results

**Grip is a curve, not a number.** The real car follows the request to about 2.5 m/s², then rolls off. Across all the runs of 2026-10-30, asking 3.25 got 2.6, asking 5.5 got 3.4, and asking 10 got about 4.2.

**Clearance is "close", not "touched".** `clearance.py` uses a body box whose overall length has never been measured. A reading of zero means a beam fell inside that box. Look at the peak acceleration, and ask whoever watched the run. Team A's runs on 2026-10-30 read zero here and did not touch a wall.

**Stops need straight running to be measured by the wall.** `stops.py` marks each stop `wall` or `wheel`. Wheel distance over-reads, so treat `wheel` stops as approximate.

## Comparing settings in the simulator

**Any terminal, drivers not needed:**

```bash
python3 tools/f1tenth_sim/sweep2.py --out /tmp/sweep.jsonl --track course_2026_10_30 --seeds 6 --friction 0.55,0.42 base: lat3:max_lateral_accel=3.0
python3 tools/f1tenth_sim/summ2.py /tmp/sweep.jsonl
```

**Working when:** `summ2.py` prints a line per candidate and friction under each track, then totals.

Use at least six seeds. On 2026-10-30 a three-seed sweep made one setting look like a clear winner, and six seeds overturned it.

`--friction` also works on `eval_gap.py`, `eval_gap_multi.py`, `sweep_gap.py` and `trace_gap.py`.

To see where a simulated lap loses time:

```bash
python3 tools/f1tenth_sim/trace_gap.py --track course_2026_10_30_scans --floor sfu_course_2026_10_30 --timeout 60 --out /tmp/trace.csv
python3 tools/f1tenth_sim/trace_deficit.py /tmp/trace.csv --lateral 3.0
```

**Working when:** it prints the lap time against a perfect profile on the same path (20.7 s against 16.4 s at 3 m/s² for the 3.5 m/s settings), then each 2.5 m with the time lost and the cap that bound.

For this course, use the measured floor instead of a friction number: `--track course_2026_10_30_scans --floor sfu_course_2026_10_30`. It reproduces the real grip curve, steering and braking ([sim-calibration.md](sim-calibration.md)).

<details>
<summary>How the localizer works (safe to skip)</summary>

Wheel speed and the gyro in the VESC (the motor controller) predict the next pose. A Gauss-Newton step then moves the pose to put the scan's points on the map's walls, using a smoothed distance field. Matching runs at 20 Hz while moving and 2 Hz when parked.

If fewer than 75% of beams fit for six scans in a row, a global search tries every free cell within 1.5 m of the course line, on a 0.25 m grid, at 18 headings. The result replaces the current pose only if it is clearly better. That handles a car carried to a new spot, and a wrong lock while parked.

Launches can still briefly confuse it, because wheelspin makes the prediction wrong. Those samples are filtered by match quality and by a speed sanity check.

</details>
