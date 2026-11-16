# Neural racer: training plan after teacher-v3

> **Who this is for:** whoever runs or changes neural-racer training.
> **Read first:** [the neural racer plan](2026-10-23-neural-racer-plan.md), especially [A2](2026-10-23-neural-racer-plan.md#a2-what-winning-means-and-how-well-know) (what "best" is measured by) and [A8](2026-10-23-neural-racer-plan.md#a8-the-training-regime) (the three stages).
> **You'll be able to:** see what teacher-v3 can and can't do, and run the next steps in order, each with a gate that says when it is done.

"The best possible algorithm for any given track" is the main plan's ship gates, met by the **student**: the network that drives the car from its LiDAR and IMU, inside the car's safety wrapper, on tracks no training stage has seen. The teacher is the means to that end. It never drives the car.

## Where we are, 2026-10-28

`teacher-v3` (seed 0, job 832150) used its whole 2×10⁹-step budget in 2 h 55 min on one H100, at 83% GPU utilisation. Obstacles were off, as decided on 2026-10-27. Its last snapshot is `checkpoints/best.pt` (update 1908).

**On 320 fresh held-out races** (20 held-out tracks × 16 starts that no training eval used; 3 laps; speed cap drawn from 1.0 to 6.37 m/s per race):

| Driver | Finished all 3 laps | Crashed | Lateral accel, 95th percentile | Ticks over 2.0 m/s² |
|---|---|---|---|---|
| Scripted pursuit (what the teacher started from) | 70.3% | 29.7% | 1.9 m/s² | 3.3% |
| **teacher-v3 `best.pt`** | **99.1%** | **0.9%** (2 of 20 tracks) | **4.6 m/s²** | **17.6%** |
| teacher-v3, held to the car's 2.0 m/s² cornering cap | 36.2% | 63.7% | 1.8 m/s² | 3.2% |
| Scripted pursuit, held to the same cap | 80.9% | 19.1% | 1.8 m/s² | 1.0% |

Three things follow from it:

1. **The teacher is excellent, but at a grip level the car is not allowed to use.** It trained with the safety wrapper off. The wrapper's cornering cap is `v ≤ sqrt(2.0 · L / |tan δ|)`, with wheelbase L = 0.36 m. Held to it, the teacher had its speed cut on 7.7% of ticks and crashed in 64% of races, and in every race with a speed cap of 4 m/s or more: its steering was learned for faster cornering. The main plan's rule (A6, "training uses the same cornering cap as the car") was not followed in Stage 1. Every later stage runs the wrapper, so this is the first thing to fix.
2. **The teacher has only ever driven alone.** Its opponent inputs were always zero, so its input normaliser learned a near-zero spread for them. In distillation's 1–4-car races, a real opponent would push those inputs to the normaliser's clip limit, and the teacher's labels there are untested.
3. **The evaluation needs to be larger, and needs a real lap-time reference.** The training-time eval races the same 80 starts every time. Neighbouring snapshots swung between 71% and 100% on it, so a snapshot chosen on 80 races is partly luck. And the teacher already beats the centreline-at-grip-limit time by about 4%: it is cutting corners on a racing line, so that bound can no longer say how close to the best it drives.

Throughput, measured on 2026-10-27: about 200k steps/s without LiDAR (the teacher). With LiDAR, needed by every student stage, a tick is 466 ms at 8192 environments, about 17.6k steps/s.

## Status, 2026-10-28 (evening)

- **Step 1** is built and its runs are queued on Fir:
  - `teacher-v4` seeds 0–2, fine-tuned from teacher-v3, 1e9 steps.
  - `teacher-v4-penalty` (item 2.1's A/B).
  - `teacher-v4-masked` (item 2.3).

  The cornering cap is drawn from 2.0 to 5.0 and given to the networks as an input (D7 as recommended).
- **3b** (Triton LiDAR), **3c** (real maps in the student evaluation) and **3d** (distillation evaluation and `best.pt`) are built.
- **3a** (the referee on Fir) is not done.
- **Step 4** will start as a small provisional sweep (D8) once teacher v4 passes T4:
  - every run races solo, so the solo teacher's labels hold;
  - one run races in traffic, to measure what skipping step 2 costs.

## The steps, in order

| Step | What | Needs | One H100, roughly |
|---|---|---|---|
| 1 | Teacher v4: drives within the car's limits | — | 1.5 h per seed |
| 2 | Teacher in traffic | step 1 | measure with a 30-min smoke first |
| 3 | Fixes the student stages need | — (in parallel with 1–2) | engineering |
| 4 | Distil into the student (Stage 2) | 2, 3d; G0 for the final run | ~5 h for 3×10⁸ steps |
| 5 | The league (Stage 3) | 3a, 3b, 4 | days (see 3b) |
| 6 | Obstacles back | 5 | a fine-tune of each stage |
| 7 | Selection and the car | 5 (then 6) | CPU, and the car |

### Step 1: teacher v4, inside the car's limits

**Goal:** the teacher-v3 result, with the cornering cap the car will enforce.

- **Apply the curvature cap in the sensor-free simulator.** It needs only the commanded steering, not the LiDAR, so it costs nothing. The teacher's actuation becomes `limits_only` plus the curvature cap. Count a cut as an intervention, with the wrapper's −0.5 penalty, so the policy learns not to need it.
  - The LiDAR braking cap still needs the scan. Leave it out of Stage 1 and measure how often it fires in Stage 2.
- **Make the cornering cap a per-race parameter** (decision D7 below). It would be drawn from 2.0 to 5.0 m/s² and given to the teacher as an input, the way `v_cap` already is.
  - The plan raises the car's cap only by measurement after Phase 10, and the simulated car held 4.6 m/s² at the 95th percentile. A policy that already knows each cap moves up that ladder without retraining.
- **Fine-tune from `teacher-v3/seed0/checkpoints/best.pt`.** No warm start is needed. About 1×10⁹ steps, which is 1.5 h at 200k steps/s. Run 3 seeds: on a default allocation, that is cheaper than one league role.
- **Evaluate on 320 fresh races,** not the fixed 80, and choose `best.pt` from that larger eval. Report each cap separately (2.0, and the top of the range).
- **Add the lap-time reference.** Gate G5 measures the teacher against the minimum-curvature line. `src/pure_pursuit/pure_pursuit/raceline_optimizer.py` already computes that line inside a track's width. Pace it at the same lateral-acceleration and speed caps, and report each track's time as a ratio to it.
- **Keep teacher-v3.** It is the reference for what the simulated car can do with no cap.

**Gate T4:** at a 2.0 m/s² cap, on 320 fresh held-out races: at least 98% finish 3 laps, at most 1% crash, curvature-cap cuts under 1% of ticks, and the median lap within 5% of the optimized line's time at the same caps (G5's measure, in the GPU simulator).

### Step 2: the teacher in traffic

**Goal:** labels that are right when there are other cars, because distillation and the league race 1 to 4 cars.

- **Fine-tune teacher v4 with 1–4 cars,** opponents from the scripted pool (`scripted.py`), and the A8 contact rewards and fault rule. Still no LiDAR for the teacher.
  - `gap_follow` reads the scan. Either cast it for those seats only, since `_scan(ids)` already takes a subset, or leave `gap_follow` out of this stage.
- **Measure throughput in a 30-minute smoke run first.** Four cars per environment cost more than one, and nothing has measured how much yet.

**Gate T4t:** in 200 held-out races of 2–4 cars against the scripted pool: zero at-fault contacts, at least 95% of races finished, and at least one pass per race on average against opponents slower than the teacher.

### Step 3: what the student stages need first

These are independent of steps 1 and 2. Do them in parallel.

**3a. The CPU referee on Fir compute nodes.** In the job environment, the 9 tests that import the F1TENTH Gym fail with `libgio-2.0.so.0: undefined symbol: g_source_set_static_name`. On login3 the gym imports cleanly, so the conflict is between the job's venv (after `module purge`) and the system glib. The league's referee scoring and Phase 9's selection both run on this referee, so it blocks step 5.

**3b. LiDAR speed.** The LiDAR cast is 92% of a tick. At 17.6k steps/s, the league's 2×10¹⁰-step budget is about 13 GPU-days, or about 3.3 days for each of its 4 roles.
- Every 2× in the cast is a 2× in Stages 2 and 3. Aim for 4× or more.
- The scan parity tests (`test_scan_matches_brute_force` and the observation parity tests) must still pass.

**3c. The real course maps in the GPU simulator's eval.** The four real maps have centreline CSVs and occupancy images in `tools/f1tenth_sim/maps/`: `room_2026_07_27`, `levine_blocked`, `levine_obs` and `circular_obs`. The CSVs hold `x, y` only, so take the widths from each map's distance transform.
- Today, "any track" is tested only on our own generator's style. That is the plan's "overfitting to the generator" risk.

**3d. Make distillation observable.**
- `distill.py` logs only its loss, and saves no `best.pt`. Add the held-out eval, by episode type (reactive, map, handover), and choose `best.pt` from it.
- Log crash, finish and wrapper-intervention rates.
- Point it at teacher v4 in traffic (step 2).

### Step 4: distil into the student (Stage 2)

- **G0 has not passed,** so the IMU noise, bias and delay in `gpu_sim/params.py` are still estimates. Phase 6 says the training run waits for G0.
  - Recommended (D8): run a provisional distillation now, to find the student's problems early, and re-run or fine-tune once Phase 0's measurements are in.
- **The wrapper is on,** exactly as on the car. Teacher v4's labels already respect its cornering cap, which is why step 1 comes first.
- **Expect a gap on lap 1.** In reactive mode the teacher's labels come from 16 m of true centreline, including round blind corners, while the student sees only what its LiDAR shows. Measure the student by episode type. The league's PPO, with its privileged critic, is what closes that gap.

**Gate S:**
- In map mode, the student finishes at least 95% as often as teacher v4, and its laps are within 10% of the teacher's time.
- In reactive mode, it is at least 10% faster than `gap_follow` (the A2 target).
- Wrapper interventions stay under 1% of ticks.
- Measured on the procedural held-out set and the real maps (3c).

### Step 5: the league (Stage 3)

As A8 describes it, with obstacles still off: four roles, PFSP over the archive with its floor, the scripted pool, and PLR with mutation. It needs 3a (referee scoring) and should wait for 3b (speed). Each role continues from the step 4 student.

### Step 6: obstacles back

The team decided on 2026-10-27 to keep obstacles off until after the league.
- teacher-v1 and v2 were never shown the obstacles in their training tracks; that was one of the measured causes of their crashes. So the teacher's privileged input needs an obstacle block first, for example the nearest obstacles' corners in the car's frame.
- Then fine-tune teacher v4, re-label, and fine-tune the league's students, with PLR bringing obstacles in gradually.

### Step 7: selection and the car

Unchanged from the main plan:
- Phase 9 selection on held-out tracks and the real maps, with the worst-case veto.
- Phase 7 shadow mode.
- Phase 10's floor ladder from 1.0 m/s.

The cornering cap rises only after measured runs on the car. With D7, the policy already knows the higher caps.

## Decisions for you

1. **D7, the cornering cap in training.**
   - **(a) Recommended:** draw it from 2.0 to 5.0 m/s² and give it to the network, as `v_cap` is. Raising it on the car later then needs no retraining. The cost is one scalar in the observation (spec version 2), in the node, and in the parity tests.
   - **(b)** Fix it at 2.0, matching the car today. No spec change, but every later raise means fine-tuning every stage.
2. **D8, distil before G0?**
   - **Recommended:** yes, a provisional run. GPU time is cheap next to finding the student's problems late. The run that ships is redone after G0.
3. **D9, seeds.** Recommended: 3 seeds of teacher v4, at about 1.5 h each.

## What is measured, and what isn't

- **Measured** (2026-10-27/28, Fir): everything in "Where we are", and the throughputs.
- **Not measured yet:**
  - Gate G5 in the CPU referee (it needs 3a, and a privileged-state adapter).
  - Any result on the real course maps.
  - Any multi-car result for a learned driver.
  - The student.
