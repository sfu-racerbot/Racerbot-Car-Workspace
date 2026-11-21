# Score Neural Racer on recorded drives

> **Who this is for:** club members comparing a candidate driver with what the cars actually did.
> **Read first:** [glossary](glossary.md) for the car's signals; [run diagnostics](run-diagnostics.md) for how they are recorded.
> **You'll be able to:** build a dataset, compare steering by speed and traffic, and check whether the change helps a simulated car finish a lap.
> **Time:** about 15 minutes after cloning, plus downloads and simulation.

## Highlights

- Reads both cars' recordings on a laptop without starting ROS or connecting a car.
- Matches each laser scan (a sweep of distances) with the next eligible command, not a future sensor reading.
- Separates entire recording days for fitting and scoring. Both cars stay on the same side of the split.
- Compares a fixed neural network with a speed-dependent steering correction, before the same safety limits.

**Why it exists:** the existing driver learned in simulation. To find what to improve, first measure where its commands differ from recorded ones. Then test in a simulator where its commands actually move the car.

**Offline only.** This tool does not approve a model, launch a driver or change the live speed cap. Keep shadow mode (predictions without driving), the LB deadman (hold the left bumper to permit motion) and the required IMU health check in place.

## 1. Turn recordings into examples

**Terminal 1 — laptop, from the top folder of Racerbot-Car-Workspace.** Tested with Python 3.10. The scratch directory keeps large arrays out of git.

```bash
WORK=$(mktemp -d)
python3 -m venv "$WORK/venv"
source "$WORK/venv/bin/activate"
python -m pip install -r tools/neural_racer/requirements-real-runs.txt
NR=tools/neural_racer/real_runs.py
export OPENBLAS_NUM_THREADS=1
python "$NR" extract data "$WORK/dataset"
```

**Working when:** each line shows accepted examples / scans for one run. The final `manifest.json` lists every run, even one with no examples. If extraction fails partway, fix the input and choose a **new** output directory; existing outputs are never overwritten by extraction.

Each example freezes the scan and latest wheel-based speed, then waits up to 25 ms for a command from the mux (the selector between command sources). The command is a comparison target, **not** proof it was caused by this scan, an expert action, or a measurement of what the wheels did.

<details>
<summary>Dataset fields and rejection rules (optional)</summary>

There is one compressed NumPy archive per usable run. Its arrays share a row index:

| Field | Meaning |
|---|---|
| `time_ns`, `label_delay_ns` | Recorder's scan time and wait until the command, nanoseconds |
| `speed` | Most recent recorded forward odometry speed, m/s; negative means reverse |
| `ranges`, `geometry` | 1,081 distances in metres; start angle, angle step (radians), min and max range (metres) |
| `action` | Recorded mux steering angle (radians) and speed command (m/s) |
| `reset` | Clear prediction and scan history before this row |

The extractor reads the split MCAP files in metadata order. It requires LB held with a joystick sample less than 500 ms old, odometry less than 100 ms old, a scan no more than 75 ms old and the expected 270-degree field of view.

Header times must increase on each topic (one named stream of messages). Up to 5 ms of header lead is allowed for clock skew.

The target must arrive after the scan in recorder order, with a header no earlier than the scan's. A release cancels a pending pair. Missing pairs, rejected scans, long scan gaps and releases break history; safety stops also reset the policy in replay. Invalid laser returns stay in the scan for the deployed safety wrapper to judge.

The manifest records rejection counts, missing topics and IMU message counts. Unknown course or traffic stays `unknown`. Known labels come from the [course analysis](rb1-lap-gap-2026-10-30.md#the-runs) and [map catalog](../tools/f1tenth_sim/maps/catalog.yaml). A two-car run is a whole session, not a label for an opponent in each scan.

</details>

## 2. Compare predictions on different days

**Same terminal.** Use the exported weights from [the training run](https://github.com/sfu-racerbot/Racerbot-Car-Workspace/pull/91), without switching branches or installing them on a car:

```bash
git fetch origin refs/pull/91/head
MODEL_PATH=src/neural_racer/models/a0eef2af0a3757b6
git archive febf96df9e787832af4f84a177388c92bf376dca "$MODEL_PATH" | tar -x -C "$WORK"
MODEL="$WORK/$MODEL_PATH"
python "$NR" evaluate "$WORK/dataset" "$MODEL" "$WORK/score.json" \
  --cap 5 --fit-day 20261019 --fit-day 20261026
```

**Working when:** you see `base` (unchanged) and `adapted` results. Without `--fit-day`, all usable runs are scored with the base model only. If the weights are missing, check that the fetched history contains the exact export above. The 5 m/s cap is for comparison only, **not** a launch setting.

Replay uses the real observation builder and each car's safety wrapper. Predictions see their own previous commands, not recorded answers. No map, race state or camera features are supplied.

The recordings contain IMU (motion sensor) measurements but no `/imu/status` messages; replay marks IMU absent. A live driver with the default IMU requirement would refuse to drive in that state.

Open `groups` in the JSON scorecard for rows by run, course, traffic and measured speed. **MAE** means the average absolute difference from the recorded command, in radians for steering and m/s for speed. Lower is closer; it does not mean a safer or faster lap.

Fitting finds one steering multiplier per speed band, using only fit-day examples where the safety wrapper did not stop. Each multiplier stays between 0.5 and 1.5. It scales the network's steering **before** the wrapper limits the command; it never overrides a stop.

The scorecard also counts turn-direction agreement (recorded turns over 0.02 rad), wrapper interventions and stops. Its latency is the 99th percentile of base network forward time on this machine, not end-to-end control timing.

## 3. Check a candidate in a race

**Same terminal, optional.** For this step install the [pinned simulator](../tools/f1tenth_sim/README.md) and the CPU PyTorch dependency used by the [existing referee](../tools/neural_racer/README.md). On a Linux x86 laptop:

```bash
tools/f1tenth_sim/setup.sh
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1
RACERBOT_CAR=rb2 python "$NR" simulate "$MODEL" "$WORK/score.json" "$WORK/races.json" \
  --track course_2026_10_30 --grid behind-gap --seed 12345 --seed 12346 --seed 12347
```

**Working when:** each seed prints a base race and an adapted race. Look at `finish_time` (null means no finish), `progress` (metres), and `contact`. A hash mismatch means the scorecard and weights came from different models: rescore the intended weights. Use `solo` or `ahead-of-gap` for other starts.

## What the first comparison found

At workspace base `7b2003f`, 46,612 of 213,540 scans across 35 of 36 recordings could be paired. Gains fitted on October 19 and 26 were 1.116 below 1 m/s, 0.994 from 1 to below 3 m/s and 0.520 at or above 3 m/s. Unseen speed bands use gain 1. Speed predictions are not calibrated.

| October 30 validation slice | Examples | Base steering MAE (rad) | Adapted (rad) |
|---|---:|---:|---:|
| All | 30,331 | 0.0848 | 0.0812 |
| Below 1 m/s, excluding reverse | 2,907 | 0.1065 | 0.1102 |
| From 1 to below 3 m/s | 19,517 | 0.0899 | 0.0899 |
| At least 3 m/s | 7,897 | 0.0642 | 0.0491 |
| Confirmed two-car sessions | 7,717 | 0.0915 | 0.0881 |

Speed error stayed at **2.218 m/s**. Both versions triggered a safety limit on every scored October 30 tick. The change improved steering mostly at higher speeds, and worsened it at low speeds. October 30 did not supply fit examples, but it was used to compare design choices; it is not an untouched final test set.

**Neither version finished a lap in 12 trials each** with simulated rb2 and IMU masked. We used three seeds per start: the October course solo and behind gap-follow, the November race-day course solo, and Spielberg solo. Time limit: 60 s, except 150 s at Spielberg.

No lap-completion or opponent-handling improvement was established. Keep this calibration offline; test a fresh recording day and readiness to finish in simulation before considering any deployment review.
