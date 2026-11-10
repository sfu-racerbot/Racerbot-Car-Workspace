# Recording and diagnosing a run

> **Who this is for:** anyone trying to work out why a run went wrong, after the fact.
> **Read first:** [operations.md](operations.md) — record the run first, diagnose it second.
> **You'll be able to:** find the run bringup recorded for you (or record one by hand), then analyze pipeline health, pose lag and watchdog stops.
> **Time:** about 10 minutes to locate a run and generate its first report.

How to capture everything needed to work out what a run did — and how to
hand that to a person or an AI agent so they can act on it.

This exists because of a debugging session on **2026-07-27** (car 2) in which four
separate faults stacked on top of each other, each one hiding the next.
Every recommendation below is here because its absence cost real time or a
real collision that night. The postmortem is at the end.

---

## Automatic recording

**Since 2026-10-26, every run of an algorithm is recorded for you.** Start
bringup the usual way and leave LB released while the recorder prepares a paused
**standby** under `~/racerbot-ws/data/<id>/racerbot_runs/` (`<id>` is `rb1` or `rb2`).
Starting an algorithm resumes that recording in the same directory.

Replace `<id>` in commands with the car ID, without angle brackets. You do not need the
recorder terminal from the [TL;DR](#tldr) below any more, unless you want
one for a run that is not an algorithm (manual driving, say).

### Highlights

- **On by default, in every bringup** (`bringup_launch.py` and its two
  alternates). Off with one flag: `record:=false`.
- **Records only while an algorithm runs.** Bringup alone, the dashboard,
  and manual driving with the gamepad record no messages. The idle standby
  already has its bag and diagnostic subscriptions ready.
- **Any algorithm, anyone's.** It watches for any [node](glossary.md#node)
  publishing `/drive` (or `/auto_map/drive`, `/auto_race/drive`,
  `neural_racer`'s shadow output, or the motor command topics). Our
  `gap_follow`, `pure_pursuit`, `auto_map_race` and `neural_racer` all count,
  and so do team A's `ftg`, `disparity_extender` and `reactive`.
- **Never two recorders at once.** If `record_run.py`, `map_course`,
  `auto_map_race_launch.py` or any `ros2 bag record` is already recording,
  it stays out of the way. If one starts mid-run, it stops its own recording
  and keeps what it has.
- **Same layout as a hand-started recording**, so `analyze_run.py` and
  `summarize_run` work on it unchanged. It adds which algorithm ran, team
  A's git commit when it was their code, and every ROS process's command line.
- **Cannot move the car.** It subscribes to `/joy` to read LB, queries the
  graph (the list of ROS nodes and their connections), and publishes only
  `/auto_record/start`, a directory-name signal for its diagnostic probe.

**Why standby is prepared before driving:** on 2026-11-09, six of car 2's
fourteen recorded runs had a roughly 0.35 s freeze of every ROS topic,
each within seconds of a recorder joining the graph. Two of the four map-racer bags
showed it, one right at the braking point before the corner where the car later
hit the wall.

The exact DDS mechanism is unknown, so automatic recording now
prepares its ROS processes while LB is released and resumes them when an
algorithm starts.

A [topic](glossary.md#topic) is a named ROS message stream; the freezes
affected all of these streams at once.

**LB gates recorder startup.** The [deadman](glossary.md#deadman) is the LB
button the driver must hold to enable movement. Standby preparation, a cold
start without a ready standby, and parameter-snapshot startup all wait while
fresh `/joy` messages say LB is held.

A ready standby can resume while LB is held
because its bag, probe subscriptions and resume client already exist.

#### Limitations

The parameter snapshot starts after **1.0 s of continuous LB release**.
If the driver presses LB within the few seconds `snapshot_params` takes, its
service clients may still join; it is started only after a full second of
release to make that unlikely.

If recording stops first, the snapshot waits up to **30.0 s after stop** for
that release interval. Without it, `auto_record.yaml` and `auto_record.log`
say `params_snapshot: skipped (LB held after stop)`.

At bringup shutdown,
a pending snapshot is skipped immediately if LB is held, with reason
`LB held at shutdown`; if released, it may run within the 15 s stop timeout.

Missing `/joy`, or no new message for more than **0.5 s**, counts as released
because the driving nodes also refuse to drive without fresh gamepad input.

The directory name is the time **standby was prepared**, which may be much
earlier than the drive. In `auto_record.yaml`, `standby_prepared` records that
time and `started` records the real recording start; an unused standby has no
`started` field. The directory is never renamed when recording starts.

At startup, `auto_record` stops orphaned `auto_` recorder processes left by
an earlier bringup: it logs each PID, sends SIGINT and allows up to 10 s before
SIGKILL. It excludes its own processes and descendants, and leaves unrelated
recorders alone.

An unused standby is stopped and its directory deleted on
shutdown or when another recorder appears; started recordings are kept.

**Why it exists:** on 2026-10-26 the races against team A (car 1) were run with
plain `gap_follow_launch.py`, and nothing recorded them. All that was left
was the node's one-line-a-second log, which cannot show what the car saw or
how smoothly it steered.

### Using it

**Terminal 1** — bringup, exactly as before:

```bash
source /opt/ros/jazzy/setup.bash && source ~/racerbot-ws/install/setup.bash
ros2 launch f1tenth_stack bringup_launch.py
```

**Working when:** among bringup's startup lines you see
`auto_record ready: records a run while any node publishes /drive, ...`.
With LB released and enough disk space, `paused standby prepared -> ...`
follows. This is preparation, not the start of a recorded run.

**Terminal 2** — any algorithm, for example:

```bash
source /opt/ros/jazzy/setup.bash && source ~/racerbot-ws/install/setup.bash
ros2 launch gap_follow gap_follow_launch.py
```

**Working when:** after the algorithm's start delay, Terminal 1 prints
`recording /gap_follow_node on /drive -> /home/.../racerbot-ws/data/<id>/racerbot_runs/<time>`.

**Then wait for the second line before you press LB:**
`recording live: /drive /joy /odom /scan are in the bag, <n> s after the algorithm appeared`.
From that line on, what the car does is in the recording.

With a ready standby, starting the recording resumes the existing bag and
signals the existing probe. If there is no ready standby and LB is held,
Terminal 1 says `waiting for LB release to start recording`; release LB and
wait for `recording live` before driving. Historical cold-start measurements
are under "How it decides, and every setting" below.

The recorder never holds the car back: LB works before that line appears. The
beginning of the run is just not recorded. If that happens the run says so
(see `activity` below), and `summarize_run` prints `STARTED LATE`.

When you stop the algorithm with `Ctrl+C`, about 3 seconds later Terminal 1
prints `saved run ... (<seconds> s, bag <size> MB)` and the command to
analyse it:

```bash
python3 ~/racerbot-ws/tools/run_analysis/analyze_run.py ~/racerbot-ws/data/<id>/racerbot_runs/<time>
```

**If it says** `another recorder is on the graph (...): not recording it`:
that is correct. Something else (a `record_run.py` you started, or the one
`map_course` brings) is already recording this run.

**To turn it off** for a session: `ros2 launch f1tenth_stack bringup_launch.py record:=false`.

**Disk use:** about 0.8 MB per second of driving (3 GB an hour), almost all
of it [LiDAR](glossary.md#lidar) scans. Only unused standbys are deleted automatically.
It refuses to prepare or start a recording
with 2 GB free or less, and says so in Terminal 1 and in `auto_record.log`.

**To see what it decided,** read `data/<id>/racerbot_runs/auto_record.log`. It keeps one
line for every start, stop, refusal and late start, across sessions. This is the
place to look when a run you expected is not there. The same lines are in
`~/.ros/log/python3_<pid>_*.log`, which is hard to find and is not saved with the runs.

### What a run directory holds, beyond the usual

Everything in [the table below](#what-gets-captured-and-why-each-piece-earns-its-place),
plus three files only automatic runs have:

| File | What it says |
|---|---|
| `auto_record.yaml` | Which node started the recording and on which topic, when it started and stopped, and why it stopped (`no algorithm publishing for 3.0 s`, `another recorder started: ...`, `bringup shut down`). See the fields below for how to tell whether it went well |
| `metadata/processes.txt` | The command line of every ROS process running when it started: which [launch file](glossary.md#launch-file), which executable, which params files. This is how you tell team A's code from ours later |
| `recorder.log` | `record_run.py`'s own output, for when a recording went wrong |

`metadata/git/` also holds `racerbot_a.txt` and `racerbot_a.diff`: team A's
commit and uncommitted edits. That holds whether the run was theirs or ours.

**How to tell whether a recording went well.** These are the `auto_record.yaml`
fields to check. `summarize_run` prints a `Recording problems` section if any of them is bad.

| Field | Good value | What a bad value means |
|---|---|---|
| `recorder_failures` | `[]` | These recording processes died. Check this one first: `recorder_exit_code` stays `0` even when a process under it dies |
| `bag_exit_code` | `0` | The bag did not close normally |
| `recorder_killed` | `false` | It did not stop within 15 s and was killed. The bag may need `ros2 bag reindex` |
| `bag_live_sec` | Under about 7 | Seconds from the algorithm appearing to the bag listening, as seen on the graph. Missing means it never got that far |
| `activity.started_late` | `false` | The car was already moving at the bag's first `/odom`, so **the start of the run is not in the bag** |
| `activity.moved` | `true` | The car never moved. `null` means the bag has no `/odom`, so it is unknown |
| `superseded` | absent | Another recorder replaced this one within 10 s, so this run is only a stub |
| `params_snapshot` | `started` or `started after stop` | `pending` means it is still waiting; `skipped (...)` gives the LB reason |

`activity` also has `first_odom_speed_mps`, `max_speed_mps` and `lb_held_fraction`.
It appears a few seconds after the run stops, because `run_activity` reads the closed bag at low priority.

`tools/save_runs.sh` lists the runs where the car never moved and the stubs before
committing. `tools/save_runs.sh --skip-idle` leaves those uncommitted. Nothing deletes them.

<details>
<summary><b>How it decides, and every setting</b> (click to expand). Skip unless you're changing it.</summary>

Bringup starts `auto_record` (`race_diagnostics/auto_record_node.py`). Every
0.25 s it lists every node on the graph, and the nodes that
[publish](glossary.md#publish--subscribe) on each trigger topic. It hands
that list to a small state machine, `race_diagnostics/auto_record.py`. That
file has no ROS in it and is tested in `test/test_auto_record.py`.

1. A publisher on a trigger topic is an **algorithm**, unless it is on the
   ignore list. `ackermann_mux` and `ackermann_to_vesc_node` are there
   because they publish the motor command topics all the time, as part of
   bringup. `rosbag2_player` is there because it is a bag being played back.
2. While idle and LB is released, it prepares exactly one paused bag and a
   silent probe with all subscriptions created. The probe writes no events
   until `/auto_record/start` names its directory.

   An algorithm that has been
   up for `start_delay_sec` (1.5 s) starts that recording, unless another
   recorder is on the graph. The delay gives a
   launch with its own recorder (`map_course`) time to start it.

   In one measurement, the recorder appeared about 1 s after the algorithm
   (car 1). If it arrives later, the recording stops and is
   labelled `superseded: true`.
3. **Another recorder** is any node named `race_diag_node` or
   `rosbag2_recorder*`. The recording `auto_record` starts is
   `record_run.py` run with `name_prefix:=auto_`, so its own nodes are
   `auto_race_diag_node` and `auto_rosbag2_recorder` and are never mistaken
   for someone else's.
4. With no algorithm for `stop_delay_sec` (3 s), it stops the recording:
   `SIGINT` to the bag first, then to `record_run.py` once settled. The next
   eligible poll prepares a new standby.
5. If a recording dies on its own, it does not start another until the
   algorithm has gone and come back, so a broken recorder cannot restart in
   a loop.

**Historical cold-start measurements, before standby (2026-10-31).** The old
automatic recorder started `ros2 bag record` first and on its own, and
`record_run.py` with `record_bag:=false` only once the bag was live (or after
5 s). Measured on a Jetson (car 2):

| How the bag is started | Start of recording to topics subscribed |
|---|---|
| Through `record_run.py`, as it was | 5.4 to 5.9 s |
| Beside `record_run.py`, both at once | Slower than alone: the two compete for the CPU |
| Alone | 1.4 to 2.6 s in that timing run; up to about 4 s in the test suite |

On car 2 on 2026-10-30, with the 3 s delay and the first row,
the bag began about 9 to 10 s after the launch command. Two runs
(`20261030-183134`, `-184149`) were already moving at 2.3 and 3.5 m/s at
their first `/odom`.

**How much of a run's start cold startup missed.** This is from the algorithm's first
message to the bag's first message, measured on a Jetson on 2026-10-31 with two small test topics (car 2):

| Recorder | Missed | Median | Runs |
|---|---|---|---|
| Through the launch file | 5.45 to 7.0 s | about 6.6 s | 4 |
| Separate bag startup | 2.95 to 6.9 s | about 4 s | 8 |

The typical run improved by about 2.5 s, **but the worst case barely moved**. The cost was the
1.5 s delay, about 1 s of `ros2 bag` startup, and 0.3 to 2.7 s for ROS discovery to
find the topics. So **wait for the `recording live` line if the start matters**.

**Most of that discovery time came from the tests' own settings (measured 2026-11-02).** The table above was measured with the test suite, which uses localhost-only discovery (`ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST`). Bringup on the cars uses Jazzy's default, `SUBNET`. Car 1, 10 attempts with each setting:

| Discovery setting | Missed | Over 5.0 s |
|---|---|---|
| `LOCALHOST`, as every test uses | 3.65 to 6.65 s, in two groups about 3 s apart | 6 of 10 |
| `SUBNET`, as bringup uses | 2.6 to 3.65 s | 0 of 10 |

With `LOCALHOST`, a newly started recorder is often not found until the next periodic discovery announcement, about 3 s later, so an attempt lands near 3.65 s or near 6.65 s. With `SUBNET`, the recorder subscribed about 0.1 s after it started in 9 of the 10 attempts.

The same split shows on a development machine (2.3 to 3.1 s against 8 of 10 over 5.0 s). It made car 1's timing test fail unless a retry happened to be fast.

So `test_the_bag_holds_the_run_from_soon_after_the_algorithm_appears` runs with `SUBNET` discovery. It first checks that a domain from 90 to 101 has no nodes on it, and uses that one; every other test stays on domain 79 with `LOCALHOST`. Its bound is unchanged: the fastest of four attempts must miss at most 5.0 s.

Standby now moves recorder discovery before the run. It keeps subscriptions
present while idle, including during manual driving, but its bag stays paused
and its probe writes nothing until an algorithm activates it.

Stopping is designed not to lose the bag:

- On bringup's `Ctrl+C`, `auto_record` closes the recording first (up to
  `stop_timeout_sec`, 15 s) and only then exits.
- The bag gets its stop signal at once. `record_run.py` gets its signal once
  `race_diag_node` has written `events.jsonl` (or after 8 s), because a `ros2 launch`
  signalled while still starting its processes can hang until it is killed.
- `ros2 launch` can also hang at exit with all its children gone. `record_run.py`
  therefore exits on its own 18 s after shutdown begins. A normal stop that hits this
  still reports `recorder_killed: true` after 15 s, though the bag is fine (`bag_exit_code: 0`).
- If `auto_record` is killed outright, `setpriv --pdeathsig` requests `SIGINT`
  for its child processes. Descendants have survived that signal on car 2;
  the next bringup's startup orphan cleanup stops any matching survivors.
- The recording runs in its own process session, so killing it as a last
  resort cannot touch bringup.

All settings are in `src/race_diagnostics/config/auto_record.yaml`, each
commented there. They are `trigger_topics`, `ignore_publishers`,
`other_recorders`, `start_delay_sec`, `stop_delay_sec`, `poll_period_sec`,
`record_bag`, `live_topics`, `topics` (empty means the same topic list as `record_run.py`),
`min_free_gb` and `stop_timeout_sec`.

The LB and snapshot settings are `deadman_button` (4), `joy_timeout_sec` (0.5 s),
`snapshot_release_sec` (1.0 s) and `snapshot_wait_after_stop_sec` (30.0 s).

Recordings go to `~/racerbot-ws/data/<id>/racerbot_runs`. Pass
`record_output_directory:=<dir>` to bringup to send them elsewhere.

**Tests.** These files cover it:

- `test_auto_record.py`: the decisions, no ROS.
- `test_auto_record_standby.py` and `test_orphans.py`: LB gates, deferred and
  skipped snapshots, standby deletion eligibility and orphan detection, no ROS.
- `test_run_activity.py`: what `activity` says about a run, no ROS.
- `test_save_runs.py`: `tools/save_runs.sh` against a throwaway git repository, no ROS.
- `test_bringup_auto_record.py`: every bringup includes the recorder, on by
  default and off with `record:=false`, and a missing `race_diagnostics`
  cannot stop bringup.
- `test_auto_record_node.py`: the real node on an isolated ROS domain. It
  records, closes the bag, defers to another recorder and survives being
  killed. It also judges how much of a run's start the bag misses, from the bag's
  own first message. It takes about two minutes.

The last one needs ROS sourced. **Any terminal:**

```bash
source /opt/ros/jazzy/setup.bash && source install/setup.bash
export ROS_DOMAIN_ID=79 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
python3 -m pytest src/race_diagnostics/test/ -v
```

**Working when:** every test passes and none is skipped. Unsourced, the
four node tests and two of the bringup tests skip, and say why.

</details>

## TL;DR

For a run that [automatic recording](#automatic-recording) does not cover, or to record by hand: three terminals. Start them in this order.

```bash
# --- Terminal 1: the recorder. Start FIRST, so nothing is missed. ---
source /opt/ros/jazzy/setup.bash && source ~/racerbot-ws/install/setup.bash
ros2 launch race_diagnostics record_run.py
# It prints the run directory it created, and the exact tee command to use.

# --- Terminal 2: the driving stack. Note the `| tee` (optional, see below). ---
source /opt/ros/jazzy/setup.bash && source ~/racerbot-ws/install/setup.bash
ros2 launch racerbot_launch auto_map_race_launch.py \
  2>&1 | tee ~/racerbot-ws/data/<id>/racerbot_runs/<the-dir-it-printed>/launch.log

# --- Terminal 3 (optional): live high-signal view of terminal 2 ---
ros2 run race_diagnostics filter_log ~/racerbot-ws/data/<id>/racerbot_runs/<dir>/launch.log
```

> **`auto_map_race_launch.py` already starts this for you.** It runs the probe by default (`diagnostics:=true`), so a one-command map-and-race run leaves a machine-readable record without a second terminal. The rosbag stays off there unless you pass `record_bag:=true`. Everything below still applies to any other workflow.

Afterwards:

```bash
ros2 run race_diagnostics summarize_run ~/racerbot-ws/data/<id>/racerbot_runs/<dir>
```

**`| tee` is no longer strictly required, but is still the simplest option.**
`summarize_run` can reconstruct an equivalent merged log on its own: ROS 2
launch already writes every node's complete stdout to a loose
`~/.ros/log/python3_<PID>_<epoch_ms>.log` file per node process, whether or
not anyone tees anything. When `launch.log` is missing, `summarize_run`
locates the per-node files that fall inside the run's own time window and
merges them back into one chronological stream automatically — see
`race_diagnostics/node_logs.py`.

This is a **different** file from ROS's own aggregated
`~/.ros/log/<run>/launch.log`, which is still close to useless and this
reconstruction does not rely on: during the 2026-07-27 session that
aggregated file held 48 lines of a run that printed several thousand.
`| tee` remains the most direct way to get one complete, already-ordered
log without depending on after-the-fact file-hunting, so keep using it
when you remember to — the fallback exists for when you didn't.

*(Housekeeping, not an action item: `~/.ros/log/` currently holds around
19,000 of these loose per-process log files on this machine, going back
months, and nothing prunes them.)*

---

## What gets captured, and why each piece earns its place

One timestamped directory per run under `~/racerbot-ws/data/<id>/racerbot_runs/`, where `<id>` is `rb1` or `rb2`. That folder is inside the git repo on purpose, so runs can be committed and pushed. After a session, run `tools/save_runs.sh` to do both ([data/README.md](../data/README.md)).

`metadata/car_id.txt` records the selected car, and `metadata/car.yaml` records `car_id` and `overlays:`, with each overlay's workspace-relative path and SHA-256 hash. If no car resolves, recording still succeeds: `car_id` is `null` and `car_id.txt` gives the reason. See [`run_metadata.py`](../src/race_diagnostics/race_diagnostics/run_metadata.py).

The bag is split into 50 MB files (`bag/bag_0.mcap`, `bag_1.mcap`, ...) because GitHub refuses any file over 100 MB. You never open those files one at a time: `ros2 bag play <run>/bag` and the analysis tools read the whole `bag/` folder as one recording.

| Artifact | Produced by | Answers |
|---|---|---|
| `launch.log` | your `tee` | What every node decided, tick by tick |
| `events.jsonl` | `race_diag_node` | Machine-readable pipeline state, **pose lag**, watchdog fires |
| `probe.log` | `race_diag_node` stdout | Same, human-readable |
| `system/proc.jsonl` | `system_log`, every second | CPU use, memory and thread count for each named ROS node process. CPU use of 100% means one full core. Was a node starved of CPU? |
| `system/tegrastats.log` | The Jetson's `tegrastats`, every second, when available on the car | CPU use per core, GPU use, memory and temperatures |
| `bag/` | `ros2 bag record` | Everything, replayable offline. Includes both odometries (`/odom`, `/odom/imu`), both IMUs, SLAM's `/pose` and pose graph, and every node's log on `/rosout` |
| `metadata/git/` | `record_run.py` at start | Which code ran: HEAD, branch, `git status` and the full uncommitted diff of `racerbot-ws` (which includes `tools/imu_calibration/`; a separate calibration clone passed as `racer_imu_root:=` is recorded as `imu-calibration`). Untracked files are listed, not copied |
| `metadata/files/` | `record_run.py` at start | Copies of imu-calibration's `car.yaml` and calibration JSONs: the IMU mounts and calibration, which no ROS parameter holds |
| `metadata/packages.txt` | `record_run.py` at start | The installed version of every `ros-jazzy-*` apt package, with `slam_toolbox` listed first. Git state cannot say which `slam_toolbox` ran, because it is apt-installed, not code in this repo — so this file is the answer to "did a package upgrade change SLAM" |
| `metadata/submodules.txt` | `record_run.py` at start | For each submodule in `.gitmodules` — team A's `src/racerbot_a` and B's `src/racerbot_b` included — the commit that was checked out, the commit the workspace itself records for it, and whether it is dirty. A `MOVED: <path>` line at the end names any submodule whose checked-out commit differs from what the workspace expects, which usually means someone advanced it without committing the change in the main repo |
| `metadata/params/` | `snapshot_params`, from 3 s in and every 2 s for the whole run | Each node's parameters in `ros2 param dump` YAML format (usable as `--params-file`). A node that starts later gets a file when it appears. A restarted node gets `<name>.<n>.yaml`. Namespaced nodes use `ns__name.yaml`; names that collide get `~2`, `~3`, and so on before any restart number. Later parameter changes are in the bag's `/parameter_events` |
| `map.pgm` / `map.yaml` | `slam_toolbox` save | The map as SLAM saw it |
| `raceline_*.csv` | `auto_map_race_node` | The line that was generated and driven. `raceline_raw.csv` is the unmodified recording and is written even when the cleanup *refuses* the run, so a refusal can be plotted against the map |

If `tegrastats` is missing, the recorder logs a warning and continues recording.

### Why the probe exists at all

Each node's own log says what *it* is doing. None of them can say why the
node upstream is silent. When SLAM was dead on 2026-07-27, every
downstream node logged a perfectly reasonable "waiting…" — which reads
identically to "healthy but idle". The probe watches the whole chain at
once:

```
/scan → slam_toolbox → /map + map→base_link TF → /slam_pose
      → auto_map_race_node → /drive
```

### Why pose lag is the single most valuable number

Localization staleness is **invisible in a topic's message rate**.
`auto_map_race_node` republishes SLAM's transform at a fixed 40 Hz
regardless of its age, so a frozen transform arrives exactly as punctually
as a live one. Only the header stamp reveals it.

This is not theoretical: it is what drove the car into a wall. Pure
pursuit spent a full second steering from a position the car had already
left, while its staleness watchdog — which measured *message arrival* —
saw nothing wrong.

The same number explains the dashboard's scan-not-lining-up-with-the-map
complaint. Overlay error ≈ **speed × pose lag**. Measured that night:
0.01 s parked, 0.31 s under load, **3.38 s peak** — 3 m of error at
1 m/s.

Rules of thumb:

| Worst pose lag | Meaning |
|---|---|
| < 0.15 s | healthy |
| 0.15–0.5 s | marginal; visible dashboard misalignment |
| > 0.5 s | `pure_pursuit` will hard-stop through the stall (`pose_timeout_sec`) |

### Why a rosbag

Because "what did the car actually see at the moment it hit the wall" is
not answerable from log text. The bag makes a run replayable:

```bash
ros2 bag play ~/racerbot-ws/data/<id>/racerbot_runs/<dir>/bag --clock
```

`/scan` is ~90% of the size. Drop it for a 10× smaller bag if you only
need pose/TF/command timing:

```bash
ros2 launch race_diagnostics record_run.py \
  topics:="/odom /slam_pose /drive /ackermann_cmd /tf /tf_static /map /joy /parameter_events"
```

**Parameter changes are in the bag too.** `/parameter_events` is recorded by default. Every node announces each change to one of its settings on that topic, so the bag holds every change made during the run: from the [web dashboard's tuning panel](../src/web_dashboards/car/docs/web-dashboard.md#live-parameter-tuning), from Lichtblick through [foxglove_bridge](foxglove-bridge.md), or from `ros2 param set` in a terminal. Without it, a bag can show the car driving differently halfway through a run with no record of why.

To read them back:

```bash
ros2 bag play ~/racerbot-ws/data/<id>/racerbot_runs/<dir>/bag --clock &
ros2 topic echo /parameter_events
```

Keep `/parameter_events` in the list if you pass your own `topics:=`, as above.

---

## Reading the results

```bash
ros2 run race_diagnostics summarize_run ~/racerbot-ws/data/<id>/racerbot_runs/<dir>
ros2 run race_diagnostics summarize_run ~/racerbot-ws/data/<id>/racerbot_runs/<dir> --json   # for an agent
```

It reports, in the order that matters:

1. **How far the run got** — SLAM active → lap closed → profile generated
   → profile loaded → pure pursuit driving. The first `[ NOT ]` is where
   to look.
2. **Localization health** — worst pose lag, frozen-pose samples, TF losses.
3. **Watchdog stops by reason** — which safety net fired, how often. Every
   hard-stop state either driving node can log is named here now, not just
   the six that were originally listed — see `run_events.py`'s
   `CRITICAL_PATTERNS`.
4. **Unclassified stops** — how many `STOP [...]` lines matched no known
   watchdog state. 0 is the expected value; a nonzero count means
   `CRITICAL_PATTERNS` has drifted out of sync with what the driving nodes
   actually log (exactly what happened to the lap-progress regex below,
   before 2026-08-25) and is itself a finding worth fixing. (This
   deliberately excludes `auto_map_race_node`'s own routine bookkeeping
   stops — `loading_profile`, `transition_hold`, and the rest of
   `supervisor_stop` — which fire on every ordinary run and would
   otherwise make this count permanently, uninformatively nonzero.)
5. **Which lap-closure gate is holding a lap open** — see below.
6. **Errors and node deaths**, de-duplicated.
7. **Missing artifacts** — itself a finding. No `map.pgm` means the
   occupancy save failed, which happened when slam_toolbox's executor was
   too busy to serve its own map subscription. `launch.log` itself is no
   longer fatal if missing — see the `| tee` note above.

Escape/recovery maneuvers (`emergency_escape`, and — once landed —
`ttc_escape`/`body_contact_escape`/`off_racing_line_recovery`) get their
own `escape_maneuver` count too: these publish a nonzero crawl speed and
log with the `DRIVE` prefix rather than `STOP`, so before 2026-08-25 they
were invisible to this whole classification scheme, not merely
miscategorized.

### The lap-closure gates

The supervisor declares a lap only when *all* of these pass at once, and
the log line reports every one as `value/limit`:

| Gate | Config key | Failure looks like |
|---|---|---|
| departed the start | `departure_distance` | never leaves the start area |
| distance travelled | `minimum_lap_distance` | loop shorter than the minimum can never close |
| turning done | `minimum_lap_turn_deg` | repeated reactive stops keep resetting progress before a full 300° accumulates — the actual blocker on every run in the 2026-08-25 session (car 2), and silently unreported by `summarize_run` before that day's regex fix |
| elapsed time | `minimum_lap_duration_sec` (4.0 s), then also `minimum_lap_duration_fraction` (0.5) of the previous lap | The first lap needs 4 s. Later laps need the larger of 4 s and half the previous lap's time, so the gate follows the course length. The former fixed 15 s blocked short, fast laps; the 300° turning gate still applies. |
| back near the start | `closure_distance` | passes the start but too wide |
| heading matches start | `closure_heading_deg` | comes back the "wrong way round" |

> **With `close_at_closest_approach: true`, the last two gates are checked at one point.** `auto_map_race.yaml` sets it.
>
> A lap does not close on the first tick inside the closure circle. That circle's radius is `closure_distance` until the gate widens after 1.25 laps (`closure_widen_after_revolutions`).
>
> It closes at the nearest point the car came to its start, once the car is 0.05 m farther away (`closest_approach_margin_m`) or has left the circle.
>
> The heading check is made at that nearest point. If the heading is off there, that point is dropped and the lap goes on. [operations.md](operations.md) shows the log line for a lap that closes this way.

`summarize_run` names the single blocking gate. On 2026-07-27 the car
drove **114 m without closing a lap** because heading error was 30.2°
against a 30.0° limit while every other gate had passed long before —
invisible unless you read that one field.

---

## SLAM report

After the car has stopped and the recording has finished, use this report to
check how [SLAM](glossary.md#slam) built the map. It reads the recorded bag and
does its analysis afterwards, so do not run it while the car is driving or
mapping.

In a terminal, from the workspace root, replace `RUN_DIR` with the path to the
recorded run. The first read needs ROS sourced; later reads use cached data in
the run's `analysis/` directory.

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
python3 tools/run_analysis/slam_report.py RUN_DIR
```

The command prints seven sections and writes `slam_report.txt`, `summary.json`
and available PNG plots under `RUN_DIR/analysis/slam_report/`. To count loop
closures, add `--log LOG_PATH` with the path to the launch terminal log. Without
that log, the loop-closure count is unknown. The command uses `launch.log` in
the run directory when it exists. Add `--refresh` to reread the bag, or
`--no-plots` to skip PNGs.

To compare two runs, replace `RUN_A`, `RUN_B` and `OUTPUT_DIR` with paths in
this terminal command:

```bash
python3 tools/run_analysis/slam_report.py --compare RUN_A RUN_B --out OUTPUT_DIR
```

The comparison prints the numbers side by side and writes overlay plots to the
output directory. It does not write into either run.

### Reading the seven sections

1. **Setup** compares SLAM's configured odometry frame with the frame in the
   recorded transforms. A mismatch is flagged. It also gives wheel
   [odometry](glossary.md#odometry--odom) distance and mean and maximum speed
   while moving faster than 0.3 m/s.
2. **Heading drift** shows how far two heading estimates drift apart after
   starting at the same difference. Smaller differences are generally better,
   but SLAM uses the odometry it was given, so agreement alone does not prove
   either estimate is correct.
3. **Corrections and loop closures** counts changes SLAM made to its map
   transform, the largest movement and turn, and jumps over 0.3 m and 1.5 m.
   A Ceres solve in the launch log is used as a proxy for a loop closure.
4. **Laps** lists each closed lap's distance, closest approach to its start,
   corrections absorbed and quality verdict. Laps before the last requested
   lap are not judged. A judged lap is marked failed when the supervisor logs
   a failure; it may be accepted anyway at the maximum lap count, which the
   report also says. A judged lap with no failure is marked passed only when a
   later message confirms that its racing profile was written. Without that
   evidence, its verdict is unknown. So each lap gets exactly one of "not
   judged", "passed", "failed", "failed (accepted at the ceiling)" or
   "unknown", always from what the supervisor logged.
5. **Map consistency** is the percentage of finite LiDAR endpoints that land
   on or within one grid cell of an occupied wall in the final map. A higher
   percentage means more of what the LiDAR saw lines up with those walls. It
   is shown for the whole run and for each lap window. Each lap's scans are
   placed using the poses SLAM published at that time, then compared with the
   final map. Later loop closures can move the map, so early laps may score
   lower even when the final map is not wrong. In car 1's 19:49 run, laps 1 and 2
   scored 19–27%, while lap 4 scored 55%. Compare corresponding laps between
   runs rather than treating an early-to-late difference as a map fault.
6. **Scan drops** counts dropped scan messages recorded on `/rosout`. The
   10-second bins also show slam_toolbox CPU use when a system log exists;
   100% means one full CPU core. A missing system log leaves CPU unknown.
7. **Wheel vs scan-matched distance** compares the wheel path with SLAM's
   map-frame path over the same time span. A ratio near 1 means their lengths
   agree, though it does not establish that either path is accurate.

For example, car 1's run `20261026-194913` used `odom_imu` in both the parameter
snapshot and the transforms. The final heading difference between `/odom/imu`
and SLAM was −16.44°, compared with 102.61° for `/odom` and SLAM. The report
found 32 scan drops, but that older recording has no system log, so CPU use
cannot be checked. Its map consistency was 34.1% overall, and three corrections
moved the estimated car position by more than 1.5 m. Those low agreement and
large corrections are reasons to investigate, rather than a healthy-map
verdict. Laps 3 and 4 failed their heading quality check at 29.5° and 29.2°
against a 15° limit; the run did not save a map.

### Limitations

The final map and SLAM poses are estimates, so consistency is not independent
proof of map accuracy. Missing inputs are labelled in the report. If SLAM did
not map the run, the report says so and still shows the other available data.

---

## Finding graph stalls

A graph stall is a moment when every ROS topic goes quiet at once. One slow topic means one slow sensor or node. All of them together means the whole ROS graph paused. A car at 5 m/s drives 1.75 m on its last command through a 0.35 s stall.

On 2026-11-09, six of car 2's fourteen recorded runs had such a stall. In each one, `/scan`, `/odom`, the [particle filter](glossary.md#particle-filter)'s pose (the car's position on a saved map) and `/drive` stopped together for about 0.35 s, within seconds of a recorder joining the ROS graph. A process with no ROS in it kept running throughout, so the machine itself was not frozen.

```bash
# Dev box or car, workspace root, ROS sourced (rb on the dev box). Reads the bag, writes nothing.
python3 tools/run_analysis/graph_stalls.py data/rb2/racerbot_runs/<time> [more runs ...]
```

**Working when:** it prints one line per run, `N graph stall(s) over /scan, /odom, ...`, followed by one line per stall giving when it started and how long it lasted. It exits 1 if any run had a stall, so a script can check a test run.

Only topics with at least two messages count, and only the stretch where all of them were publishing is judged. A topic that started late or stopped early cannot fake a stall. `--min-gap` (default 0.15 s) sets how long a silence must be; `--topics` replaces the list.

## Rebuilding a racing folder from a run

`map_course` does this for you after every run. Run it by hand for an older run, or one whose check was interrupted:

**Terminal: any, with ROS sourced. It starts no node and publishes nothing.**

```bash
python3 tools/run_analysis/rebuild_run.py data/<id>/racerbot_runs/<time>
```

**Working when:** the first line starts `GO:` or `NO-GO:`, followed by `Saved in:` and a folder under `data/<id>/racerbot_auto/`. **If it says `NOTHING TO REBUILD`:** the run has no bag, or SLAM never ran in it. Existing folders are never overwritten, so delete or rename one to redo it.

It takes the last `/map` in the bag and matches every scan against it, using odometry only to carry the pose between scans. That gives two scores:

- **Fit:** the share of beams ending on a wall. Low means the map does not match the world. Limits: 0.90 on average, and 0.70 over its worst stretch.
- **See-through:** the share of beams that pass through a mapped wall before they end, which a laser cannot do. High means walls are drawn twice. Limit: 15%.

Fit alone cannot catch a doubled map, because a beam ending on either copy of a wall still counts: car 2's run `20261030-184556` has a doubled map that fits at 0.97. Why these scores, and how they did against an independent reference: [the spec](superpowers/specs/2026-10-30-slam-reliability.md).

A map that passes gets `raceline_profiled.csv`, capped at 1.5 m/s (`--max-speed` changes it), with `raceline_2p5.csv` and `raceline_4p0.csv` beside it. A refused map gets no line, so `race_launch.py` refuses that folder too. The numbers are in `rebuild.json`.

The check starts from SLAM's own pose when that fits as well as the best place a search of the whole map finds. On a course that looks the same turned round, like the simulator's oval, the search alone can land on the mirror-image spot and call a good map NO-GO. `rebuild.json`'s `start_from` says which it used.

## When mapping fails: the tools, in order

These are the tools that found what was wrong on car 2 on 2026-10-30 and 31 ([the spec](superpowers/specs/2026-10-30-slam-reliability.md)). Use them in this order: each one answers the question the next one assumes.

**Terminal: any, with ROS sourced (`source /opt/ros/jazzy/setup.bash && source install/setup.bash`). None of them moves the car.**

**1. What was this recording?** Many "failed maps" are not mapping runs at all.

```bash
python3 tools/run_analysis/run_triage.py data/<id>/racerbot_runs/<time>
```

**Working when:** it prints the driving code that ran, every LB hold with its distance and top speed, and a last line: `usable`, `not a mapping run`, or `carried between holds`. On 30 Oct most of the runs that "failed" were another team's algorithm, or a few seconds of driving.

**2. Would its laps have closed?** This puts the run's recorded SLAM pose through the real lap counter, on each odometry.

```bash
python3 tools/run_analysis/lap_replay.py data/<id>/racerbot_runs/<time>
```

**Working when:** it prints one line per odometry frame, `counted in odom_imu 3 lap(s) [...] m`. Lap lengths that disagree, or many SLAM corrections in one frame only, mean the counter is reading the wrong odometry. It needs `/slam_pose`, so `map_course` runs and their replays only.

**3. What does the whole launch do with this recording now?** This runs the real `map_course` launch on the run's recorded sensors, parked, and says whether its laps close and its map passes. It plays the driver's real LB presses, so it refuses to run while any driver or SLAM is up, and runs on ROS domain 78, localhost only.

```bash
python3 tools/run_analysis/replay_map_course.py data/<id>/racerbot_runs/<time> --out /tmp/replays
python3 tools/run_analysis/replay_map_course.py data/<id>/racerbot_runs/<time> --out /tmp/replays imu:=true gyro_odom:=false
```

**Working when:** it ends with a JSON result, `laps_closed`, `map_go` and the reason, about as long as the run took. Exit 0 means the laps closed and the map passed. Launch arguments go after the run, as `map_course` takes them, so the same run can be tried with different settings. `--from-seconds` starts from a later stop.

**4. How good is a map, against a known-good one?** This scores a map's walls against a reference map of the same course, and SLAM's live pose against the run's own scans tracked on that reference. It doesn't depend on the SLAM under test.

```bash
python3 tools/run_analysis/slam_score.py --reference data/<id>/racerbot_auto/<good run>/map.yaml data/<id>/racerbot_runs/<time>
```

**Working when:** one JSON line per target. `walls` near 0.9 or above is a good map; a map with every wall drawn twice scores about half. `live_p95_m` is how far the live pose strayed. The old settings scored 0.48 walls and 3.2 m on run `184556`; the new profile scored 0.92 and 0.27 m.

**5. Can the tests behind a fix fail?** CLAUDE.md's Hard rule 1, rerunnable.

```bash
python3 tools/check_mutations.py tools/mutations/slam_reliability.json [--only map_localize]
```

**Working when:** it ends `N/N breaks caught`. Each break is applied to a temporary copy, so your files are never touched. A `MISSED` line names code no test would notice breaking. The full set of 140 takes about an hour.

<details>
<summary>What went wrong while finding the faults, so nobody repeats it (safe to skip)</summary>

- **Test what the operator runs.** The SLAM fix sat in a profile that only loaded with two flags, and replaying SLAM alone never showed the lap counter reading the wrong odometry. Running the whole launch on recordings, with no flags, found both.
- **Judge a SLAM setting over repeats, on more than one run.** Identical settings on identical data gave 0.32 and 0.64 of walls.
- **Line maps up from many starts.** One start understated a 0.86 map as 0.22.
- **Don't judge carrying from single scans placed on a map.** A false placement once made a car look carried 12 m. `run_triage.py` judges it from the car's own wheels and scans.
- **The lap counter must use the odometry SLAM maps on.** It tells SLAM corrections from driving by comparing the two.

</details>

## Replaying SLAM with the car parked

`tools/run_analysis/slam_replay.py` runs SLAM again on a recorded bag and
writes its results into a new run folder. Use it to compare settings or check
whether SLAM can reproduce the recorded poses without another drive.

**Highlights**

- Starts SLAM with the recording's parameter snapshot, plus optional profile overrides.
- Preserves the recorded odometry TF tree in forced mode and recomputes map corrections.
- Forces the recorded scan-admission sequence for stamp-by-stamp pose comparison.
- Saves a replay bag, logs, map files when available, and a `replay.json` summary.

**Stop bringup before replaying.** The command refuses while a
[VESC](glossary.md#vesc) motor-controller, LiDAR or joystick driver process is
alive. Simulator joystick processes also match this
check. Every replay child uses ROS domain 77 with localhost discovery, keeping
the replay's messages on a separate ROS graph from the car.

### Run an exact replay

In one terminal at the workspace root, replace `RUN_DIR` with the recorded run
folder. This example uses `odom_imu`, the wheel-and-gyro odometry used by car 1's run
`20261026-194913`. Use `--odom odom` for a recording mapped on wheel odometry.

```bash
RB_WS="$(git rev-parse --show-toplevel)" rb python3 tools/run_analysis/slam_replay.py \
  RUN_DIR --exact --odom odom_imu --label exact
```

The machine's `rb` wrapper loads the ROS environment. The replay prints its
output folder on success. Open that folder's `replay.json` to check `status`
and `pose_sequence_error`; a timeout prints the failed scan stamp and keeps the
logs. See the measured limitation below before trying the 19:49 recording.

Output defaults to
`data/<id>/racerbot_runs/RUN_NAME-replay-odom_imu-exact/` for the selected car ([`slam_replay.py`](../tools/run_analysis/slam_replay.py)). Existing output folders
are refused, so use a different `--label` for another attempt. Add
`--out-root /tmp/slam-replays` to put results in a scratch directory instead.
The source run is read without being modified.

Resolved output paths may enter an existing run only under its `analysis/` directory; a `bag/` or `metadata/` directory marks a run.

### Forced admission: the default exact mode

`--exact` selects `--exact-mode forced` by default. It starts `sync_slam_toolbox_node` and
uses the recorded `/pose` stamps to identify the scans the car processed.
These are the **planned scans**. It feeds them in recorded pose order,
waiting for each matching `/pose` before advancing: strict lockstep.

The feed sequence is:

1. The first planned scan, followed by its pose acknowledgement.
2. The three recorded scans immediately before the second planned scan.
   These are **sacrificial scans**: they consume startup callback counts 2–4
   and must produce no poses. If one produces a pose, replay fails with its stamp.
3. The remaining planned scans, each followed by its pose acknowledgement.

The startup rejection is hard-coded in
[`slam_toolbox_common.cpp:795`](https://github.com/SteveMacenski/slam_toolbox/blob/2.8.5/src/slam_toolbox_common.cpp#L795).
Forced mode also changes four admission gates, while retaining the recorded
scan-matching, map and solver settings. For the 19:49 recording:

| Admission parameter | Recorded value | Forced value |
|---|---:|---:|
| `throttle_scans` | 2 | 1 |
| `minimum_time_interval` | 0.2 s | 0.0 s |
| `minimum_travel_distance` | 0.25 m | 0.0 m |
| `minimum_travel_heading` | 0.25 rad | 0.0 rad |

This forces the already-known admission sequence rather than trying to
reproduce timing-dependent rejections. On the car, every planned scan passed
the travel and time filters measured from the same previous processed pose.

With identical inputs those scans pass again; opening the gates removes the
callback-count and timing dependency without changing their scan-matching
configuration.

`replay.json` records `exact.feed_mode: "forced"` and, under
`exact.admission_overrides`, each original snapshot value and its override.
The original value comes from the recording even if a profile changed it.

A [TF transform](glossary.md#tf--transform--frame) relates a child coordinate
frame to its parent.

Forced mode publishes the recorded `/tf` messages, keeping their order and
transform payloads. It removes every transform whose parent frame is `map`,
so the car's map corrections cannot replace the replay's new corrections.

If a message mixes map and non-map transforms, only the map entries are
removed. `/tf_static` is published with its recorded transforms as before.

The needed chain is derived from the configured `odom_frame` and `base_frame`,
using the recorded dynamic and static tree. For the 19:49 run this is
`odom_imu → odom → base_link → laser`. The final edge is a fixed, static mount; both earlier
edges need dynamic interpolation brackets.

Each needed dynamic pair gets a sample at or before the scan and the first
sample at or after it. Before a planned or sacrificial scan, the feeder
publishes a recorded-message prefix that covers all those brackets. It keeps
message order rather than replaying the pairs as separate streams.

This lets [TF](glossary.md#tf--transform--frame) perform the same interpolation
and composition as on the car. Forced mode starts no `odom_tf_relay.py` and
does not publish the odometry topic. `exact.tf_source` is `"recorded"`, and
`exact.needed_tf_pairs` lists the derived dynamic pairs.

`/clock` advances to the scan stamp, never to a future bracket's stamp.
`exact.lookahead_samples` counts newly sent TF transforms beyond that stamp
in forced mode, or newly sent odometry messages in all-scans mode. Already
sent brackets are reused.

Both exact modes log a planned pose missing after its 10-second pose timeout
and continue. **More than ten consecutive missing planned poses ends replay
as failed**, naming the stamps.

A later pose can resolve an earlier timeout
in the final accounting; `exact.timed_out_stamps` retains its history.

Forced mode refuses before starting if a pose's matching scan was not recorded
or scan stamps are duplicated. It also refuses if the tree has no needed chain
or a pair lacks a planned or sacrificial scan's bracket. Bracket errors
name the pair and scan stamp.

It also refuses when there are fewer than three recorded scans before the
second planned scan, naming that second stamp.

A missing processed scan means the recording cannot reproduce that run;
even the first pose has no exception. Both exact modes refuse `--rate`
and mode-related profile overrides.

### All-scans admission experiment

Use `all-scans` to investigate SLAM's timing-dependent admission behaviour.
It sends **every recorded scan in bag order**, keeps the recorded admission
parameters, and waits for poses only at planned checkpoints. Extra poses are
recorded without stopping. `exact.admission_overrides` is empty in this mode.

All-scans mode keeps the odometry relay, as does rate mode. This preserves the
round-1 experiment, including its composed-odometry interpolation, instead of
changing both admission and the TF source at once. Its `exact.tf_source` is
`"odom_relay"`.

In the same terminal, replace `RUN_DIR` as above and choose a fresh label:

```bash
RB_WS="$(git rev-parse --show-toplevel)" rb python3 tools/run_analysis/slam_replay.py \
  RUN_DIR --exact --exact-mode all-scans --odom odom_imu --label all-scans
```

Check that the summary says `exact.feed_mode: "all-scans"`. A failed summary
with missing checkpoints can be the experimental result; see the measured
19:49 outcome below. This mode does not guarantee the car's admission sequence,
because recorded header stamps do not reproduce transform arrival timing.

Unplanned scans outside odometry coverage are sent with the available odometry,
without complete brackets. SLAM's TF message filter decides their fate.
`exact.unbracketed_scans` counts these sends; forced mode sends no such scans.

For each covered all-scans input, the feeder sends odometry through the first
message at or after that scan, letting the relay build its transform.

The 19:49 bag has 98 scans before odometry starts and one after it ends, so
a complete all-scans feed would send 99 unbracketed scans. Planned scans still
require brackets in either mode.

Without `--exact`, the command uses asynchronous SLAM and plays all scans at
`--rate` (default `1.0`). A YAML file passed with `--profile` overrides the
recorded settings. Rate replay is useful for tuning, but an overloaded SLAM
node can drop different scans, so its pose sequence is not a tick-for-tick
reproduction.

### Keeping scan receipt observable

The feeder checks that each bracketed scan reached SLAM's scan callback before
sending another. This receipt check applies even when SLAM rejects the scan;
it does not wait for an unplanned pose. It prevents a burst of replay input
from overrunning SLAM's subscription or its TF message filter.

In slam_toolbox 2.8.5, the synchronous processing queue is an unbounded FIFO:
[`slam_toolbox_sync.hpp:54`](https://github.com/SteveMacenski/slam_toolbox/blob/2.8.5/include/slam_toolbox/slam_toolbox_sync.hpp#L54)
declares it, and
[`slam_toolbox_sync.cpp:47–53`](https://github.com/SteveMacenski/slam_toolbox/blob/2.8.5/src/slam_toolbox_sync.cpp#L47-L53)
removes entries in order. Exceeding ten entries emits a warning, not a drop.
The earlier
[TF message filter](https://github.com/SteveMacenski/slam_toolbox/blob/2.8.5/src/slam_toolbox_common.cpp#L472-L475)
is separate and bounded; this recording sets `scan_queue_size: 1`.

The scan publisher uses reliable `KEEP_ALL` history, so it has no fixed-length
outgoing queue that discards older scans.

For a covered scan, SLAM's
`map → odom` transform header acknowledges receipt of that scan's callback.
The check requires `restamp_tf: false` and a positive
`transform_publish_period`, as recorded in the 19:49 run.

In all-scans mode, unbracketed boundary scans cannot have this acknowledgement;
the feeder yields between sends so
SLAM can handle them through its TF filter.

### Measured result on car 1's 19:49 run

Car 1's run `20261026-194913` contains **3,826 scans and 315 poses**. Every pose stamp
matches a distinct recorded scan. The SLAM report's **314 updates** count
changes between successive `map → odom_imu` corrections; the initial
correction is identity. That count does not mean a pose lacks its scan.

#### Forced admission with recorded TF: round 3

**The 1 mm / 0.1° bar was met.** The `recorded-tf3` replay matched all **315
poses**, with none missing or extra. Maximum errors were
**1.171361 × 10⁻¹³ m** and **3.053332 × 10⁻¹³°**; `first_over_bar` is null.
It sent 318 scans: the 315 planned inputs and three sacrifices, which produced
no poses. All 318 had callback receipts. The map and pose graph were saved.

The dynamic pairs derived from the recording are `odom_imu → odom` and
`odom → base_link`; the final `base_link → laser` edge is static.

The replay
bag retained **9,475 non-map TF messages**, verified to have identical payloads
and order to the required source prefix. The source contains 9,487 such
messages in total; the remaining tail is beyond the last required bracket.

The filter removed **4,606 map-parented transforms** from the candidate stream.
There are **no mixed messages** in this source bag. Synthetic tests also check
that a mixed message retains its non-map transforms rather than being dropped
as a whole.

#### Comparison with round 2

Round 2 (`forced1`) used the odometry relay and matched all 315 stamps, but
failed the numerical bar: maximum errors were **2.246775 m** and **3.625845°**
at pose 59. Its first exceedance was pose **20**, stamp
**1793069383585897818 ns**, at **0.00998585 m** and **0.199962°**.

The comparison uses one-based pose indices and rounded values.

| Pose | Round 2 position (m) | Round 3 position (m) | Round 2 yaw (°) | Round 3 yaw (°) |
|---|---:|---:|---:|---:|
| 2 | 0.00004318 | 9.02e-15 | < 0.000001 | 2.54e-14 |
| 10 | 0.00012115 | 2.89e-14 | < 0.000001 | 5.09e-14 |
| 19 | 0.00000583 | 2.94e-14 | < 0.000001 | 0 |
| 20 | 0.00998585 | 2.86e-14 | 0.199962 | 2.54e-14 |
| 50 | 0.01999860 | 2.39e-14 | 0.399925 | 1.02e-13 |
| 59 | 2.24677468 | 8.14e-15 | 3.625845 | 2.54e-14 |
| 100 | 0.12057477 | 1.94e-14 | 0.586101 | 5.09e-14 |
| 315 | 0.72786463 | 8.10e-14 | 0.065573 | 1.53e-13 |

Round 2 interpolated the already-composed `/odom/imu` pose. The car's recorded
TF tree interpolates `odom_imu → odom` and `odom → base_link` separately, then
composes them. At pose 20, those odometry guesses differed by **0.00003035 m**,
with no measured yaw difference. Round 3 preserves the recorded tree instead.

Round 2's first yaw discrepancy was one fine-search angular step, **0.00349 rad**
(0.199962°). Karto passes this configured value as the angular resolution in
[`Mapper.cpp:624–628`](https://github.com/SteveMacenski/slam_toolbox/blob/2.8.5/lib/karto_sdk/src/Mapper.cpp#L624-L628).

The recorded-TF result removes that discrepancy and the later large errors.
This supports the interpolation difference as the cause of the earlier
scan-match divergence.

Ceres threading can also affect numerical reproducibility: reductions are not
bit-identical across thread counts. In 2.8.5,
[`ceres_solver.cpp:166`](https://github.com/SteveMacenski/slam_toolbox/blob/2.8.5/solvers/ceres_solver.cpp#L166)
hard-codes `num_threads = 50`; the replay log reports Ceres capping that to
**8**.

The plugin exposes no thread-count parameter, so a controlled one-thread
replay was not available. The car's effective thread count is unknown. No
thread-count change was needed to meet the bar with the recorded TF tree.

#### All-scans timing experiment

The all-scan replay with slam_toolbox **2.8.5** sent **973 scans**, then stopped
after **11 consecutive planned-pose timeouts**. Its bag contains **12 extra
poses and no matching pose stamps**.

The comparison reports `matched: 0`, `missing_in_replay: 315`,
`extra_in_replay: 12`, `max_position_m: 0.0`, `max_yaw_deg: 0.0`, and
`first_over_bar: null`. The 315 missing stamps include planned scans not
reached after the abort; eleven were actually attempted and timed out.

**The 1 mm / 0.1° reproduction bar was not met.** With no matched stamps,
the zero numeric errors and null `first_over_bar` provide no accuracy
measurement. The first missing planned stamp is **1793069360801010451 ns**.

Replay first diverges by admitting scan **1793069359721058241 ns** (zero-based
bag scan index 98), which produced no pose on the car. The car's first pose
instead used scan index 141, at **1793069360801010451 ns**. The car's `/rosout`
records TF-filter queue-full drops, including scan time **1793069360.650 s**,
before that first pose.

Transform arrival timing differs from header timing: the first recorded
`odom_imu → odom` transform has stamp **1793069359712544260 ns**, but the bag
received it **488.759 ms later**.

Replay provides recorded odometry brackets
before each covered scan, making transforms available earlier. The recorded
`throttle_scans: 2` then makes differences in callback admission affect later
scan choices.

The evidence points to input-admission differences, not solver
numerical error; no corresponding poses exist to compare numerically.

All **875 bracketed scans** sent had callback receipts, and **98 leading
unbracketed scans** were also sent. A full feed would send **99 unbracketed
scans**, including the final scan after odometry ends, but the consecutive
timeout guard stopped this attempt before that final scan.

Its `exact.lookahead_samples` count is **867**. The original output label was
`all-scans1`. A repeat with the explicit `--exact-mode all-scans` flag, labelled
`all-scans2`, reproduced these counts and retained all four recorded admission
values with `exact.admission_overrides: {}`.

The round-3 `all-scans3` smoke replay also started the relay and reached the
same eleven-missing guard, confirming that this experiment kept its earlier
TF source and admission behaviour.

### When map saving fails

Both modes try SLAM's map-saving service up to three times. If no `map.yaml`
appears, replay closes and flushes its bag, then writes `map.pgm` and
`map.yaml` from the bag's last `/map`. This converts the recorded occupancy
grid, whose cells mark free, occupied or unknown space, into map files.

`replay.json` distinguishes `map_saved: true` for a service save,
`map_saved: "from_bag"` for this fallback, and `map_saved: false` if neither
produced a map. A forced service failure on the 19:49 recording at rate `2.0`
produced `map_saved: "from_bag"`, a 272,967-byte `map.pgm` and a 166-byte
`map.yaml`.

If a replay fails before reaching its save step, the summary can
still report `map_saved: false` even when an earlier map exists in the bag.

---

## Safety notes for anyone (or anything) changing this code

Read [architecture.md](architecture.md) and
[writing-your-own-node.md](writing-your-own-node.md) first. The
non-negotiables:

- **Every node that can move the car requires the LB deadman held.** Never
  set `enable_deadman: false` — that is a policy change, not a tuning knob.
- **`race_diagnostics` only subscribes.** It publishes nothing and is
  therefore not driving code. Keep it that way; if it ever needs to
  publish, it becomes driving code and inherits the whole safety policy.
  `auto_record` goes further: it does not even subscribe, it only reads
  the graph. That is why it may run inside bringup, which must start
  nothing that can move the car.
- **Test order for driving changes, never skipped:** static topic check →
  wheels off the ground → floor, low speed, open space.
- Integration tests construct real nodes with the deadman disabled. They
  remap `drive_topic` to `/test_only/drive` for that reason. **Never
  remove that remap** — without it, `pytest` sends live commands to the
  VESC if the driver stack happens to be up.

---

## Prompt structure for an AI agent

For a fresh Claude Code session in this workspace. Fill the brackets. The
ordering is deliberate: constraints before task, evidence before
hypothesis, and an explicit statement of what is *not* yet known.

```markdown
## Context
ROS2 Jazzy F1TENTH car (Jetson Orin Nano, 8GB). This is a PHYSICAL robot
that can hurt itself or people. Read CLAUDE.md and docs/architecture.md
before proposing any change to code that publishes to /drive.

## What I ran
[e.g. ros2 launch racerbot_launch auto_map_race_launch.py, 2 mapping laps
then automatic handover to pure pursuit]

## What I expected
[e.g. car maps 2 laps, generates a raceline, then races it faster than the
1.0 m/s mapping cap]

## What actually happened
[e.g. handover completed, car accelerated, then hit a wall on the left]

## Recorded evidence
Run directory: ~/racerbot-ws/data/<id>/racerbot_runs/<dir>
- launch.log       full terminal output of every node
- events.jsonl     probe stream: pipeline state + pose lag
- bag/             replayable with `ros2 bag play <dir>/bag --clock`
- summarize_run output pasted below:

<paste `ros2 run race_diagnostics summarize_run <dir>` here>

## Your task
1. Read the artifacts above BEFORE forming a hypothesis. Quote the
   specific log lines or events.jsonl records that support your
   conclusion — do not infer from code alone.
2. Distinguish clearly between what the evidence PROVES and what it merely
   suggests. Say which is which.
3. Identify root cause. If several faults stacked, order them by which one
   masked the others.
4. Propose fixes. For each: what breaks if it's wrong, and how it would be
   verified.
5. Do NOT change safety watchdog behaviour, the deadman gate, or speed
   limits without flagging it explicitly and waiting for my confirmation.
6. Add a regression test for every behavioural fix. Tests that need no
   rclpy go in the pure-logic modules (racing_math.py, run_events.py,
   protocol.py); node tests must remap drive_topic to /test_only/drive.
7. Run the full suite and report the real result:
   `python3 -m pytest src/<pkg>/test/ -q`

## Constraints
- --symlink-install: edits to .py/launch/config take effect on next launch
  with no rebuild. Rebuild only after touching package.xml/setup.py/C++.
- Full rebuild on this Jetson needs --parallel-workers 1 (8GB RAM).
- Do not commit unless I ask.
```

### What makes this work

- **Evidence before hypothesis.** Asking for quoted log lines is what
  separates "the pose was frozen — here are two identical poses 1.0 s
  apart while odometry read 1.25 m/s" from a plausible-sounding guess.
- **Proved vs. suggested.** On 2026-07-27 the pose freeze was *proved* by
  the logs; the blocking map-save being its *cause* was only inferred. Both
  were worth acting on, but conflating them would have been wrong.
- **Ordering stacked faults.** Four faults were present at once. Reporting
  them flat would have been useless — what mattered was that the SLAM
  lifecycle bug made the other three unreachable.
- **Explicit safety veto.** An agent asked to "fix the car hitting things"
  will otherwise cheerfully relax a threshold that exists for a reason.

### Also worth telling the agent

- Which fix attempts already failed, and why. Nothing wastes an agent's
  time faster than re-deriving a dead end.
- Whether the car is physically undamaged. A bent steering linkage
  presents exactly like a control bug and no log will ever show it.
- That a monitor's *silence* is ambiguous: it can mean healthy, or that
  the monitor's filter never matched. Ask it to confirm the pipeline is
  alive rather than infer it from an absence of complaints.

---

## Postmortem: 2026-07-27

Car 2 was used in this session.

Four faults, each hiding the next. In the order they had to be peeled:

**1. `slam_toolbox` never started.** As of 2.x it is a lifecycle node that
does not configure itself; `slam_launch.py` started it as a plain `Node`,
so it sat in `unconfigured` forever — no `/map`, no `map` frame, no
`/slam_toolbox/save_map`. The tell was a *missing* log line: the
constructor's "Node using stack size" appeared, `on_configure`'s "Using
solver plugin" never did. Everything downstream waited forever and
reported it as normal idling.

**2. `AttributeError` at the handover.** `AsyncParameterClient` exposes
`services_are_ready()`, not `service_is_ready()`. The call sat in the code
path that only runs after the final mapping lap closes — unreachable while
fault 1 kept any lap from ever closing. It killed the supervisor at the
exact moment mapping succeeded.

**3. The collision.** Three things at once: the map/pose-graph save
blocked slam_toolbox's executor and froze the pose; pure pursuit's
staleness watchdog measured message *arrival*, so a faithfully
republished stale transform looked perfectly fresh; and its safety net was
a 60° forward-cone minimum, structurally blind to the wall the car was
already touching *beside* it. It logged `LIDAR clear` while in contact.

**4. A fix that made things worse.** The footprint check added for fault 3
was applied across the Hokuyo's full 270° sweep. The rearmost beams see
the car's own chassis, which is inside the footprint by construction, so
clearance read a permanent −0.110 m and the car could not move at all.
Negative clearance is physically impossible for a real obstacle — that is
what gave it away. `gap_follow` had always windowed to 180° before the
same computation; the maths was copied, the windowing was not.

### What the tooling in this document would have changed

- Faults 1 and 4 were both diagnosed from a **number that could not be
  real** (a missing log line; a negative distance). The probe surfaces
  both classes directly.
- Fault 3 needed pose lag, which nothing was measuring at the time. It had
  to be reconstructed by hand from two log lines that happened to be one
  second apart.
- No rosbag existed, so "what did the LiDAR see at the moment of impact"
  was permanently unanswerable.
- The 114 m of laps that never closed was one unread field in a line that
  scrolled past once a second.
