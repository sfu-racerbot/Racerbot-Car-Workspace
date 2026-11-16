# Neural racer: making the trainer and the networks better

> **Who this is for:** whoever changes `tools/neural_racer/neural_racer_train/` or `src/neural_racer/`.
> **Read first:** [the training plan after teacher-v3](2026-10-28-neural-racer-training-next.md) (what to train next, and in what order), and [A4](2026-10-23-neural-racer-plan.md#a4-the-network) and [A8](2026-10-23-neural-racer-plan.md#a8-the-training-regime) of the main plan.
> **You'll be able to:** pick up any item below with its reason, its change, and the measurement that says it worked.

The training plan says *what* to train. This one is about the code that does the training and the networks it trains: making each stage correct, measured, and fast enough to run more than once. Every item keeps hard rule 1: a test that fails against a deliberate break.

## Where the code is, measured

| Stage | Code | Measured on Fir | What is missing |
|---|---|---|---|
| Teacher | `teacher.py` | 2×10⁹ steps in 2 h 55 min. A rollout runs at 830k steps/s, but the PPO update takes about 8 s of each 10.5 s update, so the total is about 200k steps/s | the update is 76% of wall time |
| Simulator with LiDAR (every student stage) | `gpu_sim/lidar.py` | 466 ms a tick at 8192 environments, about 17.6k steps/s. The LiDAR cast is 92% of the tick | a cast that stops each ray when it hits |
| Distillation | `distill.py` | never run on a GPU (it ran end to end on the Jetson's CPU) | no held-out eval and no `best.pt`; it logs only its loss; it runs the whole student once per timestep in its update |
| League | `league_learner.py` | never run on a GPU | none of the PPO fixes that made the teacher work (1.1) |

## Status, 2026-10-28 (evening)

Decisions taken as recommended: D10 (Triton), D11 (cap squash for the teacher only), D12 (masked teacher A/B), and D7 (the cap as an input).

| Item | Built | Measured |
|---|---|---|
| 1.1 shared PPO update, critic warm-up | `ppo.Update`; teacher and league use it | CPU tests only |
| 1.2 evaluator | `teacher_eval` (320 races, Wilson bounds, breakdowns, `ref_ratio`), `student_eval` (per episode type, real maps) | CPU tests only |
| 1.3 Triton LiDAR | `gpu_sim/lidar_triton.py` | GPU check job queued |
| 1.4 batched sequences | `PolicyNet.unroll` in distill and league | equals per-step to 1e-5 (test) |
| 1.5 faster update | fused Adam, KL from the minibatch's own forward | bf16 and `torch.compile` not done |
| 1.6 steadier snapshots | EMA actor, KL-target anneal, top-5 re-race | in the teacher v4 runs |
| 1.7 kickstarting | `league_learner --teacher` | CPU test only |
| 1.8 run records | `run.jsonl`, `code-<job>.tar.gz` | yes |
| 2.1 cap-aware squash | `model.capped_squash` | in the v4 runs (squash vs penalty) |
| 2.2 cap as an input | privileged column; observation spec v2 (`ego[23]`) | parity tests pass |
| 2.3 visibility mask | `track_visibility`, `mask_track`, `--mask-share`, `--teacher-mask` | teacher-v4-masked run queued |
| 2.5 latent head | `distill --latent` | in the sweep |
| 2.6 width and memory | `distill --width half/full/wide --memory none` | in the sweep |

## Part 1: the trainer

### 1.1 One PPO update for every stage

**Why.** The first CPU check of the teacher-v3 code (2026-10-27) measured a KL of 18 on the first actor update: one step undid the cloned driver. The fixes that then made teacher-v3 work live only in `teacher.py`. `league_learner.py` still:
- uses one Adam optimiser for both the policy and the critic
- regresses its critic on raw returns, not normalised ones
- starts a fresh `Critic()` behind a trained student, with no critic warm-up
- never updates its critic's input normaliser, which stays at mean 0, variance 1
- uses a fixed learning rate, with no KL-adaptive step

Started from the distilled student, the league risks the same first-update collapse.

**Change.** Move `teacher.py`'s update into `ppo.py` as one function, with a recurrent variant for the student.
- It brings separate optimisers and clips, `ValueNorm`, the KL-adaptive actor step, `accumulate` micro-batches and the normaliser updates.
- It also brings critic warm-up: the policy stays frozen for the first N updates.
- The teacher and the league learner both call it.

**Measured by:**
- In a 30-minute league smoke run from a distilled student, the first actor update's KL is under 0.05.
- The student's held-out results over its first 50 updates don't fall below its distilled start by more than the eval's own interval.

### 1.2 Evaluation inside every stage

**Why.**
- The training-time eval races the same 80 starts every time, and neighbouring teacher-v3 snapshots swung between 71% and 100% on it. A snapshot chosen on 80 races is partly luck.
- Distillation has no eval at all.
- The league's scoring depends on the CPU referee, which can't import the gym on Fir's compute nodes (step 3a of the training plan).
- The one-off diagnostic of 2026-10-28 (320 fresh races, results by speed cap and by track, lateral acceleration, races with and without the cornering cap) found the teacher's main problem. The 80-race eval could not have.

**Change.** One evaluator for every stage, reporting what the diagnostic reported, as `kind: eval` lines:
- **Races:** 320 or more, mixing fixed starts (comparable across snapshots) with fresh ones (honest about luck).
- **Settings:** the car's wrapper on for students, and sensors on.
- **Breakdowns:** by speed cap, cornering cap, episode type (reactive, map, handover) and track, including the four real course maps.
- **Lap time** against the optimized racing line from `raceline_optimizer.py`, paced at the same caps.
- **Wilson 95% intervals**, and `best.pt` chosen by the interval's lower bound.

The teacher's eval stays an asynchronous CPU process. A student's eval needs LiDAR, which is slow on the CPU, so it runs on the GPU between updates: 320 environments for a 300 s race is about 12,000 short ticks. Measure that cost, and keep it under 10% of wall time by choosing how often to evaluate.

**Measured by:** every stage's `metrics.jsonl` carries eval lines with intervals, and `distill.py` writes a `best.pt`.

### 1.3 LiDAR as one GPU kernel

**Why.** The cast runs 64 sphere-tracing iterations and 5 bisection steps for **every** beam.
- Each iteration is about ten whole-tensor operations, each reading and writing an (environments × cars × 1081) tensor in GPU memory.
- `torch.where` keeps marching rays that have already hit, so no ray stops early.
- That is about 700 kernel launches a tick, almost all of them memory traffic.

**Change.** A Triton kernel, one launch per tick:
- each ray marches in registers and stops when it hits or reaches the limit
- the bisection happens in registers
- the SDF is read as fp16
- the other cars' boxes are tested in the same kernel

The PyTorch version stays as the reference implementation.

**Measured by:**
- `test_scan_matches_brute_force` and the observation parity tests pass against the kernel.
- The profile's tick at 8192 environments falls to a quarter of today's 466 ms or less.
- Every 2× here is 2× for distillation and the league. At today's speed the league's 2×10¹⁰ steps are about 13 GPU-days.

**TODO(verify):** that Triton installs from the Alliance wheelhouse for the job's torch. The car never runs it.

### 1.4 Recurrent training over whole sequences

**Why.** `distill.py` and `league_learner.py` run the whole student once per timestep inside the update: 64 or 32 sequential forward and backward passes per minibatch. Each step also calls `if w.any()`, a GPU sync. Only the GRU has to be sequential.

**Change.**
- Encode all timesteps × environments in one batched pass: the scan, ego, preview, race and fusion layers.
- Run only the GRU's recurrence step by step, with `gru_cell` and episode-start resets, then the heads in one batched pass.
- Replace the per-step `if w.any()` syncs with masks.
- Inference is unchanged, so export parity and the opponents' `vmap` path aren't touched.

**Measured by:**
- A test showing losses and gradients equal to the per-step version within 1e-5.
- The update time of a distillation smoke run, before and after.

### 1.5 A faster PPO update

**Why.** It is 76% of the teacher's wall time. Each minibatch runs the critic's forward and backward pass, the actor's forward and backward pass, and a third actor forward just for the KL. All of it runs in fp32 (TF32 matmuls are already on, `common.configure_torch`), and two `float()` syncs make the CPU wait for the GPU.

**Change**, one at a time, each measured on its own:
- **Lower precision:** bf16 autocast for the networks. The main plan's hyperparameter table already says "bf16 network, fp32 physics".
- **Optimiser:** fused Adam.
- **KL estimate:** from the minibatch's own ratio before the step, or on a 64k-sample subset, instead of a third full forward.
- **Compilation:** `torch.compile` on the two encoders.

**Measured by:** teacher throughput of at least 400k steps/s at 32768 environments, with the held-out curve of a 30-minute smoke run matching teacher-v3's within its intervals.

### 1.6 Steadier snapshots

**Why.** On the same 80 races, updates 1050, 1100 and 1150 scored 95%, 71% and 96%. The adaptive step ranged from 1e-5 to 1.7e-4 while the policy's spread narrowed from 0.30 to 0.09.

**Change:**
- Keep an exponential moving average of the actor's weights (decay about 0.99 per update). That copy is the one evaluated, saved as `best.pt`, and handed to the next stage.
- Lower `lr_max` over the last 20% of a run.
- At the end, re-race the top 5 snapshots on 1000 or more races before choosing `best.pt`.

**Measured by:** the spread of the finish rate over the last 20 evals is smaller for the averaged weights than for the raw ones.

### 1.7 Kickstarting the league from the teacher

**Why.** A pretrained policy entering PPO is exactly where that first-update collapse happened. Under league pressure, a student can also forget the racing line it was taught.

**Change.**
- Add a distillation term to the league loss: the distance between the student's command and the teacher's command on the same state (Schmitt et al. 2018, "Kickstarting"). Its weight decays to zero over the first ~10% of the league.
- The teacher's input, the privileged state, is computed by the simulator anyway, so the term is cheap.
- It works alongside 1.1's critic warm-up.

**Measured by:** the student's held-out finish rate in the first 200 league updates never falls below its distilled start by more than the eval's interval.

### 1.8 Run records

**Why.** Fir runs uncommitted code, and runs are compared by searching their logs.

**Change:**
- Each stage writes `run.json` at start: its arguments, the git commit, and a hash of the uncommitted diff.
- A small `report` command plots or prints the eval curves of several runs side by side.

## Part 2: the networks

### 2.1 The teacher's speed output knows the cornering cap

**Why.** The teacher has to drive inside the car's 2.0 m/s² cornering cap (training plan, step 1). Held to the cap without training for it, teacher-v3 crashed in 64% of races. A penalty teaches it to avoid the cap. A squash that already respects it can't exceed the cap at all.

**Change.** The teacher's speed becomes `min(v_cap, sqrt(a_cap · L / |tan δ|)) · sigmoid(a1)`, where `δ` is the steering it commands in the same step.
- Distillation's labels stay physical commands, so `distill.py` is unchanged.
- The student keeps today's squash: the car's wrapper caps it anyway, and changing it would mean changing `policy_numpy.py` and the export golden test.

**Measured by:**
- A closed-form test that no command exceeds the cap.
- Two 1.5-hour fine-tunes from teacher-v3, this squash against the penalty version, compared on the 320-race eval.

### 2.2 The cornering cap as an input (decision D7)

**Change.**
- One more column in the privileged ego block. For the student, one scalar in the ego vector: observation spec version 2, with the node passing its configured `max_lateral_accel`.
- Training draws the cap from 2.0 to 5.0 m/s². The same policy is then right at today's cap and at any raised one.
- `test_observation_parity` and `test_export_golden` are updated to the new spec.

### 2.3 Labels the student can follow on lap 1

**Why.** On lap 1 (reactive mode) the student sees only its LiDAR. The teacher's 16 m preview includes track hidden behind walls. DAgger then pulls the student toward commands it can't work out from what it sees, and the usual result is a hedge between them.

**Change.**
- Give the teacher's track input a fifth channel: whether each of the 64 preview points is visible from the LiDAR. That takes one ray per point on the SDF, 64 rays a car, cheap next to the scan's 1081.
- Train teacher v4 with hidden points masked in the reactive share of episodes.
- Label reactive episodes with the masked teacher.

**Measured by:** a distillation A/B, against a student taught by the unmasked teacher, on reactive-mode lap time and crash rate.

### 2.4 Obstacles in the teacher's input (after the league)

**Why.** teacher-v1/v2 trained with obstacles they couldn't see, which was one of the measured causes of their crashes.

**Change.** A block holding the 4 nearest obstacles' corners in the car's frame, each with a present flag, read by the teacher's MLP branch. This is exact and needs no change to the track convolution. Training plan, step 6.

### 2.5 A training-only head: the student predicts the teacher's features

**Why.** A label of two numbers per state is thin supervision. The teacher's 256-value encoding of the same state says much more about what matters.

**Change.**
- A linear head from the student's GRU state that regresses the teacher encoder's output, with an MSE weight of about 0.1.
- It is never exported: `export.py` skips it, and the car's `policy_numpy.py` never reads it.

**Measured by:** a distillation A/B on the held-out student at equal steps.

### 2.6 The student's size and memory: measure, don't guess

| Question | How | Cost |
|---|---|---|
| Is FULL wide enough? | distil FULL (646k parameters, 3.4 ms a tick on the Jetson, 13.9 ms p99 with every core busy), HALF (2.23 ms p99 forward) and a 1.5× width, then compare on held-out | three supervised runs |
| Does the GRU earn its place? | distil with and without memory. The league's `main1_mlp` role asks the same question later and more expensively | two runs |
| Is 128 hidden units enough? | if the opponent-forecast loss stays high, try 256 | one run |

Ship the smallest student that matches the best. Every candidate has to keep the Jetson's 25 ms tick, measured with the full stack running (Phase 6's gate).

### 2.7 What stays as it is, and why

- **The convolution over the scan, not a transformer** (A4): the scan is a local 1-D signal, and the Jetson's CPU is the budget.
- **Direct steering-and-speed control**, with the main plan's plan B (offset from the optimized line, tracked by `pure_pursuit`) if Phase 10 shows it won't transfer.
- **The teacher is memoryless:** its input carries the actuator delay lines, so it is Markov.

## Order

| When | Items | Why then |
|---|---|---|
| First | 1.2 | every other item is judged by it |
| With teacher v4 (training plan, step 1) | 2.1, 2.2, 1.5, 1.6 | they change or speed up the teacher before it is re-run |
| Before distillation at scale | 1.3, 1.4, 2.3 | distillation is LiDAR-bound and runs the student once per timestep |
| During distillation | 2.5, 2.6 | supervised runs are the cheapest place for A/B tests |
| Before the league | 1.1, 1.7 | without them the league risks the same first-update collapse |
| Any time | 1.8 | small |
| After the league | 2.4 | obstacles come back then (team decision, 2026-10-27) |

## Decisions for you

1. **D10, Triton on Fir for the LiDAR kernel.** It is a training-only dependency; the car never sees it. Recommended: yes, after the TODO(verify) above.
2. **D11, the cap-aware speed squash.** Recommended: the teacher only. Putting it in the student too would change the car's `policy_numpy.py` and its export golden test, for no gain the wrapper doesn't already give.
3. **D12, the visibility-masked teacher (2.3).** Recommended: yes, as an A/B in distillation. The alternative is to leave lap 1 to the league's PPO, which is slower and less certain.
4. **D7, the cornering cap as an input,** is in the [training plan](2026-10-28-neural-racer-training-next.md#decisions-for-you). 2.2 is how it would be built.

## What is measured, and what isn't

- **Measured** (Fir, 2026-10-27/28): the teacher's rollout and update times, the LiDAR's share of a tick, the snapshot swings, and the cornering-cap result.
- **Not measured:**
  - Distillation or the league on a GPU.
  - Any speed-up promised above. Each item says how it will be measured.
