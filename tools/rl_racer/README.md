# Paper-style RL racing, offline only

> **Who this is for:** someone testing a state-based racing policy without connecting to a car.
> **Read first:** [the paper](https://arxiv.org/abs/2504.02420), especially sections III–IV.
> **You'll be able to:** run a small training check, export an actor and measure simulated laps.
> **Time:** about 15 minutes after Python dependencies are installed; full training is not included.

This is a separate implementation inspired by *On learning racing policies with reinforcement learning*.
It learns control from vehicle state and known track geometry, not directly from sensor images or laser scans.

## Highlights

- NumPy vehicle simulation batches independent cars without ROS or hardware access.
- Separate actor and critic networks use the paper's 256–256 and 512–512 hidden layers.
- Training uses PPO (Proximal Policy Optimization), an algorithm that limits policy changes per update.
- Long runs require a reviewer sign-off tied to the exact training-config file hash.
- Exported actors run in NumPy without PyTorch; evaluation writes JSON lap metrics.
- A read-only FidelityPlant adapter provides an optional second-simulator check.

### Why it exists

This package (a self-contained set of Python modules) isolates the paper's state-based time-trial approach for review and experiments.
It does not import Neural Racer or replace it.
Neural Racer's end-to-end sensor-based policy is a different project with its own training and deployment process.

**Nothing here authorizes driving a physical car.** Reviewer sign-off only permits offline computation.
The files exported here are not approved car models, even after a signed training run.

## Install and test

Use one terminal, starting in the workspace root (the directory containing `src/` and `tools/`). Python 3.10 or later is required.
The requirements install a CPU build of PyTorch 2.5.1 and NumPy 1.26.4.

```bash
python3 -m venv .venv-f1tenth
.venv-f1tenth/bin/python -m pip install -r tools/f1tenth_sim/requirements.txt
.venv-f1tenth/bin/python -m pip install -r tools/rl_racer/requirements.txt pytest
export PYTHONPATH="$PWD/tools/rl_racer${PYTHONPATH:+:$PYTHONPATH}"
.venv-f1tenth/bin/python -m pytest tools/rl_racer/tests -q
```

**Working when:** pytest reports passing tests. The optional plant test skips if the pinned Gym checkout is unavailable.
For that checkout, follow [the simulator setup](../f1tenth_sim/README.md).

**If it doesn't:** check that the command uses `.venv-f1tenth/bin/python`, not the system Python.
No ROS environment needs to be sourced for this package's tests.

## Run only a smoke check

A smoke check is a deliberately tiny training run that checks the plumbing, not racing performance.
In the same terminal:

```bash
OMP_NUM_THREADS=1 .venv-f1tenth/bin/python -m rl_racer.train --smoke
```

**Working when:** the final JSON line reports `steps: 256` and a `run_dir` under `~/.cache/rl_racer/smoke/`.
Loss values should be finite. The run uses two cars, 32-step rollouts and two optimization epochs.

Smoke runs cannot choose `--run-dir`, even when a sign-off is supplied.
They always write to scratch storage, never an approved-model directory.

In the same terminal, set `RUN` to the exact directory printed above, then resume and export:

```bash
RUN="$HOME/.cache/rl_racer/smoke/run-REPLACE_WITH_PRINTED_SUFFIX"
.venv-f1tenth/bin/python -m rl_racer.train --smoke --resume "$RUN/latest.pt"
.venv-f1tenth/bin/python -m rl_racer.export "$RUN/latest.pt" "$RUN/actor.npz"
.venv-f1tenth/bin/python -m rl_racer.evaluate "$RUN/actor.npz" --max-steps 100 --output "$RUN/evaluation.json"
```

**Working when:** export prints the `.npz` path and evaluation prints JSON matching `evaluation.json`.
Resuming an already finished smoke run performs no additional updates.
An untrained or smoke-trained actor may complete zero laps; its lap time is then JSON `null`.

Only load `.pt` checkpoints you trust: they contain pickled optimizer and random-generator state.
NumPy exports use a non-pickled `.npz` format and contain no critic or optimizer.

## Review gate and full-run defaults

`config/train_paper.yaml` selects the track CSV, vehicle YAML, seed and device.
Paths inside that file are relative to its own directory.
`device: auto` chooses CUDA when the installed PyTorch build supports it; otherwise it uses the CPU.
Vehicle stepping remains NumPy/CPU even when network optimization uses CUDA.

The defaults are 400 cars, 1,024-step rollouts, 0.05 s control steps, minibatches of 1,024 and 120 million environment steps.
Discount is 0.99, and learning rate decreases linearly from 0.001 to 0.0001 across updates.
Unspecified paper settings are labeled **not in paper, our choice** in YAML.

**Runs above 200,000 environment steps are refused without reviewer sign-off.**
The owner must arrange genuine reviewer approval; do not fabricate an approval to bypass this gate.
The code checks the file's fields and hash, not the reviewer's identity or a cryptographic signature.

In the same terminal, obtain the exact file hash for the reviewer:

```bash
sha256sum tools/rl_racer/config/train_paper.yaml
```

**Working when:** it prints a 64-character hash. The reviewer places that value in a separate YAML file:

```yaml
approved_by: "Actual reviewer name"
date: "YYYY-MM-DD"
sha256: "Exact 64-character training-config hash"
```

After approval only, the reviewed command in the same terminal is:

```bash
.venv-f1tenth/bin/python -m rl_racer.train --config tools/rl_racer/config/train_paper.yaml --signoff /path/to/reviewer.yaml --run-dir /path/to/offline-run
```

**Working when:** the run writes `run.json`, `metrics.csv`, `metrics.jsonl` and `latest.pt`.
Missing fields, invalid dates or mismatched hashes cause `REFUSED` before output creation.
Changing even a comment in the training config requires new sign-off.

Checkpoints include network/optimizer state, environment state, random generators and hashes of track/vehicle files.

Resume uses the same command plus `--resume /path/to/offline-run/latest.pt`.
It refuses changed configurations or changed track/vehicle contents.

The sign-off binds the training YAML bytes; reviewers must also review the referenced assets, whose hashes are recorded separately.

`slurm/train_paper.sbatch` is an optional cluster job template; it has **not been submitted**.
It requires sign-off and run-directory arguments and resumes `latest.pt` when present.
Set `RL_RACER_PYTHON` to a prepared cluster interpreter; install a CUDA-compatible PyTorch 2.5.1 build there to use its requested GPU.
The local pinned CPU wheel does not use a GPU.

## Evaluate in the second simulator

In the same terminal, after setting up the pinned F1TENTH Gym checkout:

```bash
.venv-f1tenth/bin/python -m rl_racer.evaluate "$RUN/actor.npz" --fidelity --max-steps 100 --output "$RUN/fidelity.json"
```

**Working when:** JSON reports `backend: FidelityPlant` and documents its state mapping.
This mode only imports the existing plant and calibration; it never modifies them.
The track CSV must have matching Gym map files beside it.

The adapter uses the plant's estimated pose/speed and its simulated steering, yaw rate and slip angle.
Body velocities are reconstructed from speed and slip angle.

Measured wheel speed is unavailable, so it is estimated as longitudinal ground speed (zero longitudinal slip).

Previous commands are tracked locally. These substitutions limit the comparison's meaning.

Evaluation reports completed lap times, total off-track integral in metre-seconds and per-lap statistics.
An off-track reset abandons the partial lap; it cannot count as a completed lap.
The integral measures distance beyond the boundary over time, without an extra vehicle-body margin.

## Differences from the paper and work not done

- **Simulator:** simplified Magic Formula tyres, static axle loads and friction-ellipse coupling, not full MF6.1.
  Low-speed motion uses kinematic relaxation; there is no tyre identification dataset fitted here.
- **Parameters:** mass/inertia, axle weight distribution, longitudinal tyres, some actuator constants and normalization scales still need identification.
  Sources and provisional values are listed in `config/vehicle_rb1.yaml`.
- **Track:** the checked-in rb1 centerline replaces the paper's L-shaped course; width is a configurable full-width fallback.
  Track geometry uses smooth interpolation with chord-based arc lengths.
- **State:** simulation supplies state directly. There is no OptiTrack integration.
  On a real car, pose would come from localization (estimating position on a map), with additional velocity/actuator estimates.
- **Not done:** system identification from November recordings; a ROS inference node (the process that would exchange messages with the car); full training; real-world evaluation.

No claim of matching the paper's lap times or of safe physical deployment is made.
