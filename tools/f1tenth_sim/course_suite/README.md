# Course suite: map racer against gap_follow on many courses

**For:** anyone changing car 2's map racer (the racing-line builder or `pure_pursuit`) who needs to know whether a change is faster everywhere, not just on one course. **Read first:** [docs/simulator.md](../../../docs/simulator.md). **Time:** about an hour for the small courses on 5 workers. The 23 full-size F1TENTH circuits need several hours, so run those on the pit PC.

## Highlights

- **50 courses:**
  - 8 of our recorded courses and maps
  - 7 indoor layouts
  - 23 real F1 and DTM circuits at 1:10 scale (the public [F1TENTH racetracks](https://github.com/f1tenth/f1tenth_racetracks))
  - 4 of those circuits shrunk to indoor size
  - 8 random indoor-sized loops
- **The real pipeline on each course:**
  - a "mapping lap" by gap_follow becomes the recorded line
  - the shipped builder (`python -m pure_pursuit.race_line`) builds the racing line from it
  - the map racer races that line
- **Fair opponent:** gap_follow runs in four configurations (default, time trial, time trial at 3.5 and 4.0 m/s), plus one at a raised top speed when `--vmax` is used. A course's bar is its **best** configuration that ran clean on at least half its seeds.
- **Contact never comes from gym's flag.** It is the simulator's own body-against-wall check ([docs/glossary.md#contact](../../../docs/glossary.md#contact)).

### Why it exists

A change that helps course 06a can hurt 06b, and on 2026-11-09 several did. The suite judges every change on all courses at once. The user's bar is an average margin across them.

## Using it

**Dev-box terminal, workspace root, simulator environment set up** ([setup.sh](../setup.sh)). No ROS nodes start. Run every command through `rb` with `RB_WS` set to this checkout.

```bash
# 1. Courses (downloads the GPL F1TENTH maps into .sim/course_suite/src; nothing is committed)
python3 tools/f1tenth_sim/course_suite/gen_tracks.py
# 2. gap_follow's current settings for car 2, as a stand-in "run" the benchmark reads
python3 tools/f1tenth_sim/course_suite/gf_params.py .sim/course_suite/gf_rb2 rb2
# 3. Everything for some courses (omit --tracks for all 50)
RACERBOT_CAR=rb2 WS=$PWD python3 tools/f1tenth_sim/course_suite/run.py /tmp/suite \
    --tracks car2_course_2026_11_06a_scans,indoor_oval,rand_1 --workers 5 --seeds 3
# 4. The verdict
python3 tools/f1tenth_sim/course_suite/table.py /tmp/suite
```

**It worked when** the table ends with a line like `24 tracks compared: median margin +5.4%, mean +6.1%, ..., map-racer runs with contact 0`. **If a course shows `build None`**, the builder refused it. That is expected for `narrow_1p0` (too narrow for the clearance rules) and `circular_obs` / `levine_obs` (obstacles across the line). It is not a loss.

Useful options:
- `--stage build,race` or `--stage race`: rerun only those stages. Lines already built are kept, so a re-race after a controller change is quick.
- `--vmax 7.0`: a raised VESC top speed for the builder, the map racer and an extra gap_follow time trial (`ttv`).
- `--gap-configs ttv`: run only that gap_follow configuration.
- Environment variables read by `race.py` and `builder.py` try out changes without editing the shipped code:
  - `LEAD` sets the speed lead.
  - `SPEED_FLOOR_GAIN` and the other physics overrides go through `SUITE_BUILDER=tools/f1tenth_sim/course_suite/builder.py`.

## What it cannot tell you

- **Simulator only.** Every course uses car 2's 2026-11-06 floor model ([docs/sim-calibration.md](../../../docs/sim-calibration.md)). Above 5.4 m/s, acceleration is an **estimate**: the motor's force is modelled as falling to zero at 16 m/s (`speed_loop_accel_fade_from` / `_no_load`). No car has been driven above 5.4 m/s.
- **Races start on the racing line** (`race.py` sets `RACELINE_SOURCE = "plan"`). On the real car, `auto_map_race` switches onto the line while driving. A standing start far off the line is not tested here.
- **gap_follow starts at the course's first centreline point,** not at a real run's start pose, so a course's numbers can differ from `docs/sim-calibration.md`.
