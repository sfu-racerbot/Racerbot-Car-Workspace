# Training the neural racer on FIR

> **Who this is for:** whoever submits neural-racer training jobs to Fir.
> **Read first:** [the neural racer plan](../../../docs/superpowers/specs/2026-10-23-neural-racer-plan.md#a9-fir-and-gpu-efficiency) — why the jobs are shaped this way.
> **You'll be able to:** set up once, benchmark, train one stage, run the league, and check each job's GPU efficiency.

These scripts run the plan's training on the Alliance's **Fir** cluster at SFU. Each job uses one H100. The scripts are built to keep that GPU over 80% busy, and to put the CPU cores it leaves idle to work.

> **Status, 2026-10-23:** every stage exists, and the whole chain has run on the Jetson's CPU, including a full CPU dress rehearsal of the shakedown job. **Nothing has run on Fir or on a GPU yet**, which is why the shakedown job is step 2.

## Quick path

Everything below, condensed. Each step is one command, run on a **Fir login node** from your clone of the workspace.

| Step | Command | What it does | Then |
|---|---|---|---|
| 0 | edit `tools/neural_racer/slurm/fir.conf` | set `NR_ACCOUNT` to your `def-...` allocation | — |
| 1 | `tools/neural_racer/slurm/fir_run.sh setup` | referee gym, wheelhouse check, login venv, held-out maps | once |
| 2 | `tools/neural_racer/slurm/fir_run.sh shakedown` | 1 h GPU job: the tests, then every stage at tiny scale, then the export golden check | read `nr-shakedown-<job>.out`: every line must say PASS |
| 3 | `tools/neural_racer/slurm/fir_run.sh benchmark` | 30 min: how many environments fill an H100 | put `chosen.num_envs` into `fir.conf` as `NR_NUM_ENVS` |
| 4 | `tools/neural_racer/slurm/fir_run.sh teacher` | Stage 1, seeds 0 1 2, each a chain of `NR_SEGMENTS` segments | `fir_run.sh status` until `DONE` |
| 5 | `tools/neural_racer/slurm/fir_run.sh distill <best seed>` | Stage 2. **Re-run after Phase 0** puts car 1's measured IMU numbers into `gpu_sim/params.py` | wait for `DONE` |
| 6 | `tools/neural_racer/slurm/fir_run.sh league` | Stage 3: four roles; the referee workers score snapshots as they appear | days on a default allocation |
| 7 | `tools/neural_racer/slurm/fir_run.sh select` | picks the release from the referee's results | read `selection_report.json` |
| 8 | `tools/neural_racer/slurm/fir_run.sh export <winner>` | the folder the car loads, with both reports | copy to the car, run `approve_model` there |

- **Status:** `tools/neural_racer/slurm/fir_run.sh status` shows your jobs, each run's latest metrics, `DONE`/`FAILED` markers, GPU utilisation, and the benchmark's choice.
- **Dry runs:** prefix any step with `NR_DRY_RUN=1` to see the exact `sbatch` commands without submitting.
- **Continuing a run:** rerun the same step. Every stage resumes from its last checkpoint, so nothing restarts from scratch.

## Calibrated retraining (retrain_calibrated.yaml)

The 2026-11-22 retraining setup: the simulator built on one car's own calibration, randomization kept narrow around that car's fit, steering and motor lag modeled, four club SLAM courses in the training pool, the race-day course `car2_course_2026_11_11` held out, and more scripted opponents in the league. Why, and what the current model gets wrong on each club course: [the Neural Racer README](../README.md#club-map-diagnosis-and-the-calibrated-retraining-setup).

It runs the same pipeline as the quick path. Only steps 0, 4, 5 and 6 change. Nothing trains until someone signs the setup off.

**Before you start:** do the quick path's steps 0-3 (account, setup, shakedown, benchmark) once.

1. **Pick the car.** In `tools/neural_racer/slurm/fir.conf`, set `NR_CAR` to the car you are training for (`rb2` is car 2, which raced on 2026-11-11). Every job exports it as `RACERBOT_CAR`.
2. **Edit the setup.** In `tools/neural_racer/retrain_calibrated.yaml`, set `car:` to the same id. Train one car per run: each car's randomization is centred on its own fit.
3. **Sign it off.** Set `signed_off_by:` to the name of whoever approved this run (Vera, for the club). Leave it `null` and every training stage refuses to start.
4. **Point the scripts at it.** In `fir.conf`, set `NR_RETRAIN_CONFIG=tools/neural_racer/retrain_calibrated.yaml`.
5. **Check it,** on the login node:

   ```bash
   tools/neural_racer/slurm/fir_run.sh retrain-check
   ```

   **Working when:** it prints the car's friction (`mu`), its steering and speed delays, the four `club_tracks`, `pinned_slots: 4` and `signed_off: true`.

   **If it doesn't:** it names the problem. A held-out course in `club_tracks`, a car that does not match `NR_CAR`, or an unknown key are all refused.
6. **Train,** exactly as in the quick path. Use fresh run names in `fir.conf` (for example `teacher-cal-rb2`), so an older run is not resumed with different physics:

   ```bash
   NR_DRY_RUN=1 tools/neural_racer/slurm/fir_run.sh teacher   # every sbatch line must end in --retrain-config ...
   tools/neural_racer/slurm/fir_run.sh teacher
   tools/neural_racer/slurm/fir_run.sh distill <best seed>
   tools/neural_racer/slurm/fir_run.sh league
   ```

   `teacher`, `distill` and `league` check the sign-off again before submitting anything.
7. **Select and export** with the quick path's steps 7 and 8.
8. **Re-run the club-map diagnosis** on the export, from the workspace root on a login node. It uses the CPU referee, so it needs no GPU:

   ```bash
   RACERBOT_CAR=rb2 PYTHONPATH=tools/neural_racer ~/.venvs/neural_racer/bin/python \
       -m neural_racer_train.diagnose --model <export dir> --out "$SCRATCH/neural_racer/diag-<name>" --workers 8
   ```

   **Working when:** `report.md` shows 3/3 on every course, including the held-out `car2_course_2026_11_11`, and lap times at or under `gap_follow`'s.

   **If it doesn't:** the report names the course, the distance and the wrapper cap that stopped the car. Bring that back before changing the setup.

The export stays shadow-only until it passes `approve_model` on the car. Nothing here drives a car.

## One-time setup, on a Fir login node

Login nodes have internet access; compute nodes don't. So anything that downloads happens here, once.

**Terminal 1** — a Fir login node, from your clone of this workspace:

```bash
module load StdEnv/2023 python/3.11
tools/f1tenth_sim/setup.sh          # clones the pinned F1TENTH Gym into .sim/ for the CPU referee
avail_wheels torch numpy scipy PyYAML tensorboard   # confirm the wheelhouse carries them
```

**Working when:** `setup.sh` finishes without errors, and `avail_wheels` lists a version for each package.

**If it doesn't:** if a package is missing from the wheelhouse, download it into a directory of your own. Then pass that directory to jobs as `NR_WHEELS`:

```bash
pip download --no-deps -d "$PROJECT/nr-wheels" <package>
```

## Running

**Run these in order.** Each command is for a Fir login node, from the workspace root. Put your allocation's account name in place of `def-<pi>`.

**1. Measure throughput first.** Run this whenever the simulator or the network changes. The `benchmark` stage finds the environment count that fills the GPU.

```bash
sbatch --account=def-<pi> --time=00:30:00 tools/neural_racer/slurm/fir_train.sbatch benchmark bench-$(date +%F)
```

**Working when:** `$SCRATCH/neural_racer/runs/bench-<date>/solo/` contains the benchmark's report, and `gpu_util_summary_<job>.txt` reads 80% or more.

**2. A single stage**, for example Stage 1's teacher, as one seed:

```bash
sbatch --account=def-<pi> --export=ALL,NR_ROLE=seed0 \
    tools/neural_racer/slurm/fir_train.sbatch teacher teacher-v1
```

Resubmit the same command to continue it. It resumes from `$SCRATCH/neural_racer/runs/teacher-v1/seed0/checkpoints/`.

**3. The league:**

```bash
tools/neural_racer/slurm/fir_league_submit.sh --account def-<pi> --run league-v1 \
    --init "$SCRATCH/neural_racer/runs/distill-v1/solo/checkpoints/latest" --segments 6
```

Add `--dry-run` to print the `sbatch` commands without submitting anything.

**Working when:** `squeue -u $USER` shows four running jobs (one per role) and twenty pending ones, waiting on dependencies.

## Reading the results

Each role writes to `$SCRATCH/neural_racer/runs/<run>/<role>/`. All roles share `<run>/archive/`, so every role's snapshots become opponents for all the others.

| File | Meaning |
|---|---|
| `gpu_util_<job>.csv` | `nvidia-smi` sampled every 30 s |
| `gpu_util_summary_<job>.txt` | mean GPU utilisation for the whole job (what the Alliance sees) and after the first 10 minutes. It warns below 80% |
| `eval_<job>.log` | the CPU referee workers |
| `DONE` | the run used its whole budget. Later segments exit at once |
| `FAILED` | the trainer crashed. Later segments refuse to start until you fix the cause and delete this file |

After any job, `seff <jobid>` shows the scheduler's own CPU and memory efficiency.

## Why the job is shaped like this

- **One GPU, 12 cores, 64 GB.** One GPU's share of a Fir node is 12 cores and about 250 GB. Asking for more than the share is billed as more GPUs, and waits longer to start.
  - The network is too small for multi-GPU data parallelism to pay off. More GPUs means more league roles or seeds, each its own job.
- **Spare cores do real work.** The trainer keeps the simulator on the GPU, so it needs about one core. Of the other ten, three generate fresh tracks for the GPU's track pool (`NR_TRACKGEN_WORKERS`), and seven run the CPU referee (`tools/f1tenth_sim`) on archive snapshots. Without them, CPU efficiency would read about 8%.
- **12-hour segments, chained with `afterany`.** Shorter jobs start sooner. Slurm sends `SIGUSR1` 10 minutes before the limit; the script forwards it, and the trainer checkpoints and exits with code 85, which the script treats as success. So a pre-empted or timed-out segment never loses more than one checkpoint interval.

## The trainer contract

What every `neural_racer_train/<stage>.py` must do, so that these scripts work with it unchanged:

- Accept `--run-dir DIR --archive DIR --role NAME --cpu-workers N --resume`, plus its own options, and start N CPU track-generator processes (Phase 4 of the plan).
- With `--resume`, continue from `DIR/checkpoints/latest` if it exists.
- On `SIGUSR1`, write a checkpoint and exit **85**.
- Exit **0** only when the run's whole budget is done. Any other code is a crash.
- Never call `.item()` or otherwise synchronise with the GPU inside a rollout. Log every N updates. This is the single most common reason a GPU-resident trainer falls below 80%.

> **TODO(verify) before the first real submission:** the Fir wiki still listed access details as "to be determined" on 2026-10-23. Confirm the GPU request syntax (`--gpus-per-node=h100:1`), the maximum wall time, and whether the Python module is `python/3.11` or newer.
