# Steering laws: pure pursuit, Stanley, LQR, MPC, delay LQR and delay MPC

> **Who this is for:** anyone racing with `pure_pursuit` (the map-based race controller) who wants to try a different way of steering along the racing line, or tune one.
> **Read first:** [racing-autonomy.md](racing-autonomy.md#choosing-a-steering-law) — what the racing line is and what the race controller does around the steering.
> **You'll be able to:** pick a steering law at launch, explain what each one does, tune one in the simulator, and judge it fairly against pure pursuit.
> **Time:** 35 minutes to read. A tuning run in the simulator takes about a minute; a full judging run about 40 minutes.

The race controller follows a recorded racing line. Each tick (40 times a second) a *steering law* decides **how far to turn the wheels** to stay on that line. There are six to choose from at launch.

Car 2 defaults to `delay_lqr` through its tuning overlay; car 1 retains `pure_pursuit`. Delay LQR has simulator validation only, not physical-car validation; [results below](#delay-lqr-results).

## Contents

1. [Highlights](#highlights)
2. [What a steering law is, and what it isn't](#what-a-steering-law-is-and-what-it-isnt)
3. [Safety](#safety)
4. [Choosing a law at launch](#choosing-a-law-at-launch)
5. [The laws](#the-laws) — [pure pursuit](#pure-pursuit), [Stanley](#stanley), [LQR](#lqr), [MPC](#mpc), [delay LQR](#delay-lqr), [delay MPC](#delay-mpc)
6. [Delay compensation and smoothing](#delay-compensation-and-smoothing)
7. [How the laws were tuned](#how-the-laws-were-tuned)
8. [Tuning Stanley and LQR](#tuning-stanley-and-lqr) (and MPC)
9. [Judging a law](#judging-a-law)
10. [Installing osqp for MPC](#installing-osqp-for-mpc)
11. [Troubleshooting](#troubleshooting)

## Highlights

- **Six laws, one switch.** Choose `steering_law:=pure_pursuit | stanley | lqr | mpc | delay_lqr | delay_mpc` at launch. Car 2's default is delay LQR; deadman, watchdogs, safety net, overtaking and speed caps apply to every law.
- **Delay MPC is experimental.** It avoids body contact in the current comparison but triggers safety stops on 06b and fails the promotion rule. [Results and every configuration tried](#delay-mpc-results).
- **All three alternatives beat pure pursuit in the simulator** on 15 held-out runs each (3 circuits × 5 seeds), with no wall contact. None of the margins is big:
  - **Stanley:** 1.6% better cross-track, 2.8% slower laps.
  - **LQR:** 1.6% faster laps, 3.5% worse cross-track.
  - **MPC:** 1.5% faster laps, 0.6% worse cross-track.
- **Tuned and judged on different seeds**, so a law cannot pass by being fitted to the runs it is judged on. The promotion rule was never changed to make a law pass.
- **MPC solves in 1.9 ms typical, 3.5 ms at the 99.9th percentile** on the Jetson, with the full simulated race stack running beside it. The budget is 8 ms. A late solve falls back to pure pursuit for that one tick.
- **Laws only steer.** None of them can raise the speed. Speed always comes from the racing line's speed profile and the controller's caps.
- **Simulator only so far.** No alternative law has driven the real car. Car 2's shipped delay LQR path completed six seeds per course with zero contact and stop ticks ([results](#delay-lqr-results)).

## What a steering law is, and what it isn't

Every tick, the race controller works out where the car is (from localization: the particle filter matching LiDAR to the saved map) and which point on the racing line it is nearest to.

The steering law takes that and returns one number: the steering angle. That is all it does.

Everything around it is shared by all six laws and does not change when you switch:

- the **LB deadman** (no LB held → no driving),
- the **watchdogs** that stop the car if localization goes stale or the car ends up too far off the line,
- the **reactive safety net** that slows or stops for something close in the LiDAR,
- **overtaking**, which steers with pure-pursuit geometry whatever law is selected,
- the **steering slew limit** (how fast the wheels may turn) and the **cornering speed cap**.

So a bad steering law can make the car track the line worse or lap slower. It cannot switch off a safety layer.

## Safety

**Hold LB on the F710 the whole time the car moves. Let go and it stops.** This is true for every steering law; the deadman is in the controller, not the law.

If the deadman were turned off (`enable_deadman: false`), a law steering the wrong way would have nothing stopping the car short of someone catching it. That setting is never allowed on the car ([architecture.md](architecture.md#workspace-policy-the-lb-deadman-button-is-mandatory-for-every-node-that-can-move-the-car)).

**Stanley, LQR, MPC, delay LQR and delay MPC have only been driven in the simulator. The first time any of them runs on the car, do it in this order:**

1. Static checks with no driver stack running: follow [Testing before it is on wheels](writing-your-own-node.md#testing-before-its-on-wheels).
2. Wheels off the ground, full stack up, LB held: check steering direction and that releasing LB stops the wheels.
3. On the floor at low speed in open space. Use `raceline_floor_test.csv` for the capped 1.5 m/s line, or a lower dashboard `max_speed`.
4. Race speed only after that looks right.

The simulator's position noise and delay are estimates of this car's. Stanley and LQR respond strongly to both (see [Delay compensation and smoothing](#delay-compensation-and-smoothing)), so a law that is smooth in the simulator may weave on the car.

## Choosing a law at launch

**Terminal 2** — the race controller, on top of the hardware (terminal 1: `bringup_launch.py`) and localization:

```bash
ros2 launch pure_pursuit pure_pursuit_launch.py steering_law:=stanley
# or the race-day combination (localization + controller):
ros2 launch racerbot_launch race_launch.py steering_law:=lqr
```

Leave `steering_law` off to load the selected car's layered default: `delay_lqr` on car 2, from `src/racerbot_car/cars/rb2/tuning/pure_pursuit.yaml`; `pure_pursuit` on car 1, from the shared configuration.

**Working when:** the node's startup log has a line like `steering law: stanley StanleyParams(gain=5.0, ...)` with the values it loaded. Once you hold LB, the decision state on the dashboard (or in `/drive_intent`) reads `stanley`, `lqr`, `mpc` or `delay_lqr` while it is steering.

**If it doesn't:**

- An unknown name stops the node at startup with an error naming `steering_law`. That is deliberate: a typo must not quietly fall back to another law.
- `mpc` and `delay_mpc` without the `osqp` package installed also refuse to start. See [Installing osqp](#installing-osqp-for-mpc).

The law and its parameters are **launch-time only**. The dashboard cannot change them while driving, and the node refuses if anything tries. To change a law's gains, edit `pure_pursuit.yaml` and relaunch. (No rebuild needed — the workspace uses `--symlink-install`.)

## The laws

The first four answer the same question. They differ in *what error they look at* and *how far ahead they look*. [Delay LQR](#delay-lqr) and [delay MPC](#delay-mpc) also predict the steering chain's delays.

Two errors come up again and again:

- **Cross-track error** — how far the car is to the side of the line, in metres.
- **Heading error** — how crooked the car is relative to the line's direction, in radians.

### Pure pursuit

The shared default, car 1's default, and the baseline for the earlier law comparisons.

Pick a point on the line some distance ahead (the *lookahead*, 0.6–1.5 m depending on speed). Steer along the circle that passes through that point.

It is simple and hard to upset. Looking a metre ahead averages away a lot of position noise and hides the car's reaction delay.

The cost is that it cuts corners a little: it is always aiming at a point that is already round the bend. Its knobs are `min_lookahead`, `max_lookahead` and `lookahead_speed_gain` (these three *are* live-tunable).

### Stanley

Stanley was used on Stanford's DARPA Grand Challenge winner. It looks at the **front axle** instead of a point ahead:

> steering = heading error + a correction that pulls the front axle back to where it should be

The correction is `atan(gain × front-axle error / (speed + softening))`. It is strong at low speed and gentle at high speed. `softening` stops it from blowing up near standstill.

There is no lookahead to tune. That makes Stanley very direct, and also sensitive to noise in the position estimate.

**Two corrections to the textbook version are in this code.** Both come from the kinematic bicycle model (the standard simple model of a car: a rear wheel and a steered front wheel a wheelbase apart). Spec 02's formula had them wrong, and the car hit the wall on every Silverstone run because of it:

1. **No extra curvature term.** Spec 02 added `atan(wheelbase × curvature)` for bends.

   On a bend the heading error is already measured at the front axle's point on the line, where the line has turned further. So it already carries that steering, and adding the term counted it twice.

   In a 1 m-radius corner at 3 m/s that pushed the car about 0.6 m to the inside.
2. **The front axle should sit slightly outside the line in a bend.** Every racing line here is recorded at the *rear* axle (`base_link`). If the rear axle follows a circle of radius R, the front axle runs `√(R² + L²) − R` outside it (L = wheelbase, 0.36 m). Stanley now steers the front axle to that offset, not to zero.

A car with its rear axle exactly on a circle of radius R now gets exactly `atan(L/R)`, the steering that keeps it there. The unit tests check this against the formula worked out by hand (`test_stanley_test6/7` in `src/pure_pursuit/test/test_tracking_laws.py`).

Parameters: `stanley.gain` (5.0), `stanley.softening` (1.0 m/s), plus `stanley.delay_compensation` and `stanley.steering_time_constant` ([below](#delay-compensation-and-smoothing)).

### LQR

LQR stands for *linear-quadratic regulator*. It describes the car's error as two numbers — cross-track error `e` and heading error `θ` — and steers with a fixed rule:

> steering = steering the bend needs − K₁·e − K₂·θ

The two gains K₁ and K₂ are computed, not hand-picked. You say how much you care about each error (`lqr.q_lateral`, `lqr.q_heading`) against how much you mind steering hard (`lqr.r_steer`). LQR then finds the gains that best trade those off for a car moving at a given speed.

The best gains depend on speed, so they are precomputed for a range of speeds (up to `lqr.v_grid_max`, 6 m/s) and looked up each tick.

Like Stanley, it corrects the error *now* and has no lookahead, so the same delay and noise handling applies.

<details>
<summary><b>The model and the maths</b> — click to expand. Skip it unless you're changing <code>LqrGainTable</code>.</summary>

The error model is the kinematic bicycle linearised about the line, discretised at the control period dt:

```
e[k+1]     = e[k] + v·dt·θ[k]
θ[k+1]     = θ[k] + (v·dt/L)·(δ[k] − δ_ff)
```

with δ_ff = atan(L·κ) the steering the line's curvature κ needs.

The gain K(v) solves the discrete algebraic Riccati equation for Q = diag(q_lateral, q_heading), R = r_steer.

The table is built once at startup on a speed grid from 0.1 m/s to `v_grid_max` and interpolated. The tests check K against a direct Riccati iteration and check the closed loop is stable at every grid speed.

</details>

### MPC

MPC stands for *model predictive control*. Every tick it plans the next **second** of steering: 20 moves, 50 ms apart. The plan is chosen to keep the car on the line through the bends ahead.

It plans *knowing* the servo's limits: never past the maximum angle, and never turning faster than the servo can. It applies the first move, throws the rest away, and plans again next tick.

MPC and delay MPC see bends ahead and treat the steering rate as a hard limit rather than finding out afterwards.

Both MPC laws can run out of time. Each plan is a small optimisation problem, solved with the `osqp` library. If a solve is late or fails, that is a **miss**:

| What happens | What the car does | Decision state |
|---|---|---|
| One miss | Steers with pure pursuit for that one tick | `mpc_fallback` |
| `mpc.max_consecutive_misses` (3) in a row | **Stops**, and stays stopped until you release and re-press LB | `mpc_miss_latched` |

"Late" means slower than `mpc.deadline_sec` (8 ms). On the Jetson, with the whole simulated race stack running beside it, 4 solves in 20,000 were late, all isolated. None came close to three in a row ([mpc-benchmark-2026-10-19.json](mpc-benchmark-2026-10-19.json)).

Parameters:

- the plan: `mpc.dt`, `mpc.horizon_steps`;
- the weights: `mpc.q_lateral` (4.0), `mpc.q_heading`, `mpc.r_steer`, `mpc.r_rate` (smoothness);
- the miss rules: `mpc.deadline_sec`, `mpc.max_iter`, `mpc.max_consecutive_misses`;
- `mpc.delay_compensation`.

<details>
<summary><b>How the plan is solved</b> — click to expand. Skip it unless you're changing <code>MpcLaw</code>.</summary>

The model is LQR's error model, but time-varying. Each of the N = 20 steps gets the speed and curvature the line has at that point ahead, and the tan δ linearisation is carried along the horizon.

The states are eliminated ("condensed"), leaving a quadratic programme in the N steering moves only:

- the cost is tracking error, steering effort and steering change, plus LQR's Riccati cost at the end, so the short horizon behaves like an infinite one;
- each move has a box constraint for the angle limit, and each difference a constraint for the rate limit. The first move's rate is measured from the last command over one control tick (25 ms), the rest over `mpc.dt`.

OSQP is set up once with a fixed sparsity pattern and updated in place each tick, warm-started from the last plan.

OSQP meets constraints only to about 1e-5. So a solution up to 1e-4 rad past a bound counts as solved (anything further is a miss), and the plan that leaves the law is then moved onto the exact bounds, move by move.

</details>

### Delay LQR

**Car 2's default via its tuning overlay; simulator-validated only, not driven.** Select it explicitly with `steering_law:=delay_lqr`. With the updated builder, braking and car 2 calibration, it completed all 12 shipped-path runs with zero contact and stop ticks. See [Delay LQR results](#delay-lqr-results).

**Why it exists.** Pure pursuit hides the delay by aiming a metre ahead. Stanley and LQR use one fixed prediction, `delay_compensation`, that assumes the steering holds still.

Delay LQR models the steering chain itself: the commands the car has already sent, and how the servo follows them. On car 2 the simulator's model of that chain has two pure delays, which add up to 0.125 s, and a servo lag on top:

1. The position estimate is 2 ticks old. A [tick](glossary.md#tick) is one control cycle, 25 ms, so 40 a second.
2. The steering command waits 3 ticks (75 ms) before it is sent on to the motor controller.
3. The steering servo, the motor that turns the wheels, follows each command gradually. Its time constant is 1/15 s (67 ms): the time it takes to cover about 63% of the way to a new position.

Delay LQR predicts where the car will be when a command takes effect, and steers for that point.

**What it does, each tick:**

1. It keeps the steering commands it has sent, one per tick, and runs a model of the servo over them. The position estimate does not say where the wheels are, so this model fills that gap.
2. It moves the sensed pose forward along the same simple car model Stanley uses, for 5 ticks (125 ms), until a new command would take effect.
3. It finds the nearest point on the racing line to that predicted pose and reads how sharply the line bends there. From the bend it works out the steering the bend needs. That is the [feedforward](glossary.md#feedforward) term. It reads the bend one servo time constant further along the line, because the wheels lag the command.
4. It corrects the error the car will have at that point: the sideways distance from the line, the heading error, and the difference between the wheel angle and the feedforward. The gains come from [LQR](#lqr) and depend on speed, so they are looked up in a table built once at startup.
5. It holds the result to the same steering-rate and steering-angle limits as the other laws.

**Parameters.** They are set at launch, the same way as the other laws' ([choosing a law](#choosing-a-law-at-launch)). The defaults are in `src/pure_pursuit/config/pure_pursuit.yaml`.

| Parameter | Default | What it sets | Where the value comes from |
|---|---|---|---|
| `delay_lqr.q_lateral` | 1.0 | how much sideways error costs | tuned on the tuning seeds |
| `delay_lqr.q_heading` | 1.0 | how much heading error costs | tuned on the tuning seeds |
| `delay_lqr.r_steer` | 1.0 | how much steering effort costs | tuned on the tuning seeds |
| `delay_lqr.v_grid_max` | 6.0 m/s | the fastest speed in the gain table | design choice, above the 5 m/s racing speed |
| `delay_lqr.sensing_delay` | 0.05 s | age of the position estimate (2 ticks) | car 2's floor model, `pose_delay_ticks` |
| `delay_lqr.actuation_delay` | 0.075 s | wait before a command is sent (3 ticks) | car 2's floor model, `steering_delay_ticks` |
| `delay_lqr.servo_gain` | 15 per second | how fast the servo closes on a command | car 2's floor model, `steering_servo_gain` |
| `delay_lqr.servo_rate` | 2.5 rad/s | the fastest the servo turns | car 2's floor model, `steering_rate_max` |

The last four are physical values of car 2's steering chain, taken from the simulator's model of the 2026-11-06 floor (`tools/f1tenth_sim/sim_fidelity/calibration.py`). Car 1's steering chain has not been measured, so these numbers are not known to be right for car 1.

<details>
<summary><b>Why the weights are 1.0, and why the steering-rate limit matters</b> — click to expand. Skip it unless you are changing the weights or the gain table.</summary>

The gains come from an LQR design on three states: sideways error, heading error, and the wheel angle relative to the feedforward. For each speed on a grid, the gain is the solution of the discrete algebraic Riccati equation, a standard way to compute the best feedback gain for a linear system.

The steering-rate limit caps how fast the command may change: `max_steering_rate`, 1.0 rad/s, shared by every law. A gain that is best on paper can need faster steering than that.

In a closed-loop bench built on the simulator's delay chain, with no grip model, a sideways weight of 4 was unstable at a 1.0 rad/s limit and converged at 10 rad/s. So the shipped weights are 1.0. That bench is a scratch script and is not in the repository.

The tests compare the gain table with a separate Riccati iteration, and check that the closed loop is stable at six sample speeds.

</details>

**Checks.** `src/pure_pursuit/test/test_delay_lqr_law.py` has 32 tests. They check:

- the gain table, against an independent calculation;
- the delay counts, against the simulator's tick counts;
- the feedforward and servo response, against closed-form values;
- the steering limits, and the refusal of NaN input.

`test_tracking_law_integration.py` runs the law through the same deadman and stale-joystick paths as the other laws, and checks that every `delay_lqr.*` value reaches the law.

The law's tests were each seen to fail against a deliberate break of the code they cover.

### Delay MPC

**Experimental; it has not met the promotion rule.** Select it with `steering_law:=delay_mpc`. Car 2 continues to use delay LQR because delay MPC still triggers safety stops on 06b.

Delay MPC combines the command-history prediction of delay LQR with MPC's constrained steering plan. It predicts where the car will be when a command reaches the servo, then plans 32 commands at 25 ms intervals: 0.8 s of steering.

The prediction tracks lateral error, heading error and wheel angle. Each step samples the line's curvature and speed at the predicted arc position; the next position advances by that profile speed times the prediction interval.

The command-angle and command-rate bounds are hard constraints. An additional constraint keeps the predicted servo velocity within `servo_rate`, so the optimiser cannot assume the wheels will turn faster than the model allows.

Both delay laws use `DelayPredictor` for history, re-seeding after another path publishes a command, servo estimation, the settling-window check and pose rollout. Refactoring it left delay LQR's outputs exactly equal over a 500-tick comparison against `23b6a51`, including intervening commands.

Delay MPC uses the same miss handling as MPC: a miss gives a one-tick pure-pursuit fallback; three consecutive misses latch a zero-speed stop until LB is released. Both retain the existing diagnostic states `mpc_fallback` and `mpc_miss_latched`.

The shared `tracking_laws.DEADLINE_LAWS` definition controls fallback construction, [latch](glossary.md#latch) handling, timing reports and diagnostic plan replay in the node. The simulator uses it for fallback dispatch and result reporting too.

<details>
<summary><b>Parameters and prediction equations</b>: reference for tuning or changing the law; optional for running the comparison.</summary>

All `delay_mpc.*` parameters are launch-time only. Defaults agree in `DelayMpcParams`, the node declarations and `pure_pursuit.yaml`.

| Parameter suffix | Default | Meaning |
|---|---|---|
| `dt` | 0.025 s | prediction interval |
| `horizon_steps` | 32 | number of commands; allowed range 2–60 |
| `q_lateral` / `q_heading` | 4.0 / 4.0 | lateral and heading error weights |
| `r_steer` / `r_rate` | 1.0 / 0.0 | command offset and command-rate cost weights |
| `deadline_sec` | 0.008 s | whole-law wall-clock deadline, including QP assembly |
| `max_iter` | 4000 | OSQP iteration ceiling |
| `max_consecutive_misses` | 3 | consecutive misses that latch the stop |
| `sensing_delay` / `actuation_delay` | 0.05 / 0.075 s | car 2's delay chain |
| `servo_gain` / `servo_rate` | 15 /s / 2.5 rad/s | proportional response and physical slew ceiling |

With state `[e, theta, delta]`, reference steering `ff = atan(L*kappa)`, and command `u`:

```text
e_next     = e + v*dt*theta
theta_next = theta + v*dt/(L*cos(ff)^2) * (delta - ff)
delta_next = delta + servo_gain*dt*(u - delta)
```

The servo update is the Euler model used by `ServoLqrGainTable`. The solver constrains `|servo_gain*(u-delta)| <= servo_rate` instead of letting this linear update enter saturation.

Eliminating the predicted states leaves a QP in the commands, with tracking cost and a servo-model LQR terminal cost at the prediction interval. OSQP is constructed once with fixed sparsity and warm-started each tick, using absolute and relative tolerances of `1e-8`. The returned plan is projected onto exact command limits.

</details>

#### Limitations

`PathContext` supplies geometry, speed and curvature but no local body-clearance margins. This implementation therefore has no soft wall-room constraint. A distance from the line alone would invent clearance that the law has not been given.

The model is kinematic and does not predict the floor's grip-dependent understeer. At the first stop of a diagnostic 06b repeat, the predicted lateral error was −0.326 m and the heading error −0.253 rad. The next eight commands were all at 0.42 rad, while the reference needed 0.367 rad.

That observation establishes exhausted steering reserve at the stop. Grip and prediction mismatch are possible contributors; their separate effects have not been isolated. The existing safety layers remain binding.

Timing below comes from an x86-64 simulator host. It does not establish the Jetson Orin Nano's 40 Hz compute budget, and this law has no physical-car validation.

## Delay compensation and smoothing

These two settings are what made Stanley and LQR usable. Both are off when set to 0.

**Delay compensation.** A steering command does not act the moment it is computed:

- the position estimate is already a LiDAR scan ([glossary](glossary.md#scan)) or two old,
- the command waits a tick to be sent,
- the servo takes about 40 ms to get there.

That adds up to about 0.1 s in the simulator's model of this car (`tools/f1tenth_sim/sim_fidelity/calibration.py`).

Pure pursuit hides this by aiming a metre ahead. Stanley and LQR correct the error *now*. So they are always correcting an error the car has already moved past, and at speed that turns into a weave.

With `delay_compensation` set to d seconds, the law first predicts where the car will be d seconds from now, holding its current speed and steering, and steers from there. MPC uses it too, as the start of its plan.

**Smoothing** (`steering_time_constant`, Stanley and LQR only) is a light low-pass filter on the law's output. Each tick it moves part of the way from the last command toward the new one.

The reason: with the simulator's 2 cm of position noise, Stanley at its tuned gain would jitter about 0.04 rad a tick at 1 m/s, three times pure pursuit.

That jitter is not harmless. The controller's cornering speed cap reads sharp steering as a tight corner and slows the car. So jitter cost lap time directly: up to about 10 s a lap on Spielberg before the filter.

A time constant of 0.05 s (two ticks) removes the jitter, and the delay compensation absorbs the lag it adds.

| Parameter | Stanley | LQR | MPC | Range |
|---|---|---|---|---|
| `<law>.delay_compensation` | 0.10 s | 0.08 s | 0.05 s | 0 to 0.5 s |
| `<law>.steering_time_constant` | 0.05 s | 0.05 s | — | 0 to 0.5 s |

## How the laws were tuned

The defaults in `pure_pursuit.yaml` were chosen in the simulator on the **tuning seeds** (1000–1002) and then judged once on separate **judging seeds** (12345–12349).

| Law | Before tuning (tuning seeds) | What fixed it | Tuned values |
|---|---|---|---|
| Stanley | Hit the wall on every Silverstone run | the formula fix above, delay compensation, smoothing, higher gain | gain 5.0, softening 1.0, delay 0.10 s, smoothing 0.05 s |
| LQR | Weaved, then stopped by the safety net mid-lap | delay compensation, smoothing | delay 0.08 s, smoothing 0.05 s, weights unchanged |
| MPC | 8 of 9 runs passed; cross-track 5% worse | delay compensation, heavier cross-track weight | `q_lateral` 4.0, delay 0.05 s |

Along the way the harness itself was found to model the car with a stale steering limit (0.26 rad, not the 0.349 rad in `pure_pursuit.yaml`). It now reads every limit from that YAML, so the harness and the car cannot disagree.

## Tuning Stanley and LQR

The tuning loop runs in the no-ROS simulator ([simulator.md](simulator.md)). It never touches the car.

**Any terminal**, from `~/racerbot-ws`. Always on the **tuning** seeds:

```bash
python3 tools/f1tenth_sim/law_sweep.py --laws stanley --seed 1000 --repeat-seeds 3 \
    --law-param stanley.gain=4.0 --law-param stanley.delay_compensation=0.12
```

It runs pure pursuit and the candidate on the same 3 circuits × 3 seeds, three runs at a time, and prints:

- a row per law and circuit: runs passed, mean and worst cross-track, lap time, safety-net stops (`stops`) and slow-downs (`avoid`), wall contacts, compute time;
- a verdict line, `PROMOTE` or `REJECT`, with the reasons for a reject.

**Working when:** it finishes in about a minute with a verdict line. Add `--baseline-cache /tmp/pp_1000.json` to reuse pure pursuit's runs between tries.

What to look for:

- **`pass` below 3/3 or any `contact`** — the law is unsafe at these settings. That always rejects.
- **Lap time much slower than pure pursuit, with more `avoid` steps** — usually steering jitter or weave. Raise `delay_compensation` or `steering_time_constant` a little.
- **Cross-track worse** — raise the law's error gain (`stanley.gain`, `lqr.q_lateral`, `mpc.q_lateral`). Too high and lap time suffers.

When a setting looks good, write it into **both** places together:

1. the law's `...Params` dataclass at the top of its section in `src/pure_pursuit/pure_pursuit/tracking_laws.py`, and
2. `src/pure_pursuit/config/pure_pursuit.yaml`.

Then fix the node's `declare_parameter` defaults in `pure_pursuit_node.py` to match. Two tests fail if any of the three disagree (`test_yaml_mirrors_the_param_dataclasses` and `test_node_defaults_are_the_law_defaults`). That is what stops the simulator and the car running different gains.

## Judging a law

Tuning seeds tell you what to try. Only the judging run says whether a law is better, and it is run **once**, after tuning is finished. If you tune until the judging run passes, you have tuned on the judging seeds.

**Any terminal**:

```bash
python3 tools/f1tenth_sim/law_sweep.py --laws stanley lqr mpc --seed 12345 --repeat-seeds 5 \
    --output docs/tracking-law-comparison.json
```

(`run_validation.py --compare` gives the same verdict, one run at a time — see [sim-validation.md](sim-validation.md#comparing-steering-laws-head-to-head).)

**The promotion rule** (`tools/f1tenth_sim/law_compare.py`) needs all three of these:

1. **Safe on every run:** at least one lap, no contact, never more than 0.5 m off the line, no law failures, no MPC latch (the stop-until-LB-is-released state, [glossary](glossary.md#latch)).
2. **Strictly better than pure pursuit** on mean cross-track **or** mean lap time.
3. **No more than 5% worse** on the other one.

A REJECT means the law doesn't ship. It never means the rule gets relaxed.

**Results, 2026-10-19** (Jetson, simulator profile `car`, Spielberg / Silverstone / BrandsHatch × seeds 12345–12349; full rows in [tracking-law-comparison.json](tracking-law-comparison.json)):

| Law | Runs passed | Mean cross-track | Mean lap | Verdict |
|---|---|---|---|---|
| `pure_pursuit` | 15/15 | 0.0690 m | 147.6 s | (baseline) |
| `stanley` | 15/15 | 0.0679 m (−1.6%) | 151.7 s (+2.8%) | **PROMOTE** |
| `lqr` | 15/15 | 0.0714 m (+3.5%) | 145.2 s (−1.6%) | **PROMOTE** |
| `mpc` | 15/15 | 0.0694 m (+0.6%) | 145.3 s (−1.5%) | **PROMOTE** |

No run of any law touched a wall.

PROMOTE meant a law **could** be selected; the 2026-10-19 comparison did not change the default. Car 1 still uses pure pursuit. Car 2 now selects delay LQR based on the later combined changes and verification below. Every alternative still needs the physical test order under [Safety](#safety).

<details>
<summary><b>Why lap times differ at all, when laws only steer</b> — click to expand. Optional.</summary>

Two parts of the controller turn steering into speed:

- The **cornering speed cap** limits speed by the sharper of the requested and the actual steering. A law that steers more, or jitters, is slowed.
- The simulator harness's **safety net** drops to 1 m/s whenever something is within 0.7 m in the forward ±30° cone. It stops if something is within 0.4 m.

On Spielberg, pure pursuit itself spends about 30% of its ticks in that 1 m/s mode, because the shipped racing line runs close to the walls. A law that points the car's nose at a wall a little more often spends longer there.

That is most of the lap-time difference between the laws. It is also why Silverstone occasionally stalls a run: the harness has no way to restart after a 0.4 m stop. That is a property of the harness and the racing line, and it hits pure pursuit too (6 stop ticks on the judging seeds).

</details>

### Delay LQR results

**Current shipped path, verified 2026-11-08.** The CLI built both lines using car 2's layered configuration. Each line ran for four laps on six seeds (12345–12350), with floor `sfu_course_2026_11_06`, map safety, 0.42 rad steering, 5.0 m/s speed ceiling and 2.7 m/s² online lateral ceiling.

| Course | Flying median | Flying range, laps 2–4 | Contact ticks | Stop ticks | Max / mean cross-track |
|---|---|---|---|---|---|
| 06a | 11.42 s | 11.40–11.47 s | 0 | 0 | 0.2318 / 0.0897 m |
| 06b | 8.82 s | 8.75–8.87 s | 0 | 0 | 0.2922 / 0.0964 m |

All 12 runs completed. Each row pools 18 flying laps; cross-track gives the largest run maximum and the average run mean. The supplied `gap_follow` reference at its shipped 5 m/s ceiling is 11.77 / 8.91 s. The map racer is about 3% / 1% faster; `gap_follow` at 3.5 m/s remains about 3% faster on 06b alone.

The result depends on the braking-ramp fix, corner-scaled line clearance and car 2 calibration as well as the steering law. See [the builder verdicts, reproduction settings and probe comparison](racing-autonomy.md#how-the-map-racer-beats-gap_follow-on-car-2). This is simulator evidence, not a driven-car result.

**History.** The initial merged-code judging run contacted on all six delay LQR trials, using the older lines and 2.9 m/s² online ceiling. That result remains useful context, but it does not describe the current shipped path. The detailed record follows for comparison.

<details>
<summary><b>Initial experiment and failed judging runs</b>: optional history; skip this for the current settings.</summary>

**Historical conditions.** Every run in this record used the older settings below.

- **Floor and courses:** car 2's 2026-11-06 floor (`sfu_course_2026_11_06`), courses 06a and 06b.
- **Lines:** the minimum-time racing lines in `tools/f1tenth_sim/lines/`, driven as profiled lines, so the speeds come from the line. The `v29` in each file name matches car 2's 2.9 m/s² lateral limit. TODO(verify): the planner command that made the two files was not saved with them.
- **Controller limits:** steering cap 0.42 rad, top speed 5.0 m/s, online lateral ceiling 2.9 m/s², four laps, and a 70 s limit per run.

Run one command per course and seed, changing `--law` and `--seed` as needed:

```bash
RACERBOT_CAR=rb2 python3 tools/f1tenth_sim/pure_course.py \
    --track car2_course_2026_11_06a_scans \
    --floor sfu_course_2026_11_06 \
    --line tools/f1tenth_sim/lines/car2_2026_11_06a_mintime_v29.csv \
    --profiled --safety map --laps 4 \
    --steering-limit 0.42 --speed 5.0 --lateral 2.9 \
    --seed 12345 --timeout 70 --law delay_lqr
```

**Where:** a terminal at the workspace root, in the simulator environment that `tools/f1tenth_sim/setup.sh` sets up ([simulator README](../tools/f1tenth_sim/README.md)). No ROS node starts.

**Working when:** one line starts with `top 5.0 lateral 2.9:` and ends with the lap times. **If it doesn't:** a `ModuleNotFoundError` means the simulator's packages are not on this Python's path. Run `setup.sh` first.

**Reading the line.** Each run prints one line. Its fields mean:

- `lap` is the first lap's time. `lap times` lists every completed lap.
- `finished` is `True` only when all four laps are done.
- `contact` counts the ticks in which the car's body touched a wall. A run ends at the first one.
- `cross-track` is how far the car was from the line, in metres, as a maximum and a mean.
- `stop` counts the ticks in which the map safety layer held the car stopped.

Contact verdicts come from this body check. This repo does not use the simulator's own collision flag for verdicts ([glossary](glossary.md#contact)).

**Judging rule.** Settings are chosen on the tuning seeds (1000–1002). The judging seeds (12345–12347) decide.

For this law the rule comes from the task that added it. It needs zero contact on all six judged runs (three seeds on each course). It also needs a flying median lap faster than gap_follow on at least one course, without losing more than 3% on the other.

A *flying lap* is laps 2–4, because lap 1 starts from rest.

The judging seeds were run three times.

Once was before the merge. Once was after it, and that is the run that counts, because the merge changed the safety net. The third was a lateral-ceiling diagnostic that was not adopted.

**Judging runs, merged code.** The merge brings in origin/main with PR #33, the arc-aware Tier 1 check. This table is the result.

| Law | Course | Runs that contacted | Flying median (laps 2–4) | Max cross-track | Mean cross-track |
|---|---|---|---|---|---|
| `pure_pursuit` | 06a | 3 of 3 | none completed | 0.211–0.251 m | 0.111–0.122 m |
| `pure_pursuit` | 06b | 0 of 3 | 10.00 s (9 laps) | 0.459–0.471 m | 0.136–0.143 m |
| `delay_lqr` | 06a | 3 of 3 | none completed | 0.321–0.383 m | 0.093–0.105 m |
| `delay_lqr` | 06b | 3 of 3 | none completed | 0.243–0.296 m | 0.104–0.118 m |

The cross-track columns give the range over the three seeds.

**Historical result: not met.** The delay law contacted the wall in all six runs. Pure pursuit contacted in three, all on 06a. The default remained `pure_pursuit` at that stage. gap_follow was not re-measured on that merged code, so the lap-time half of the rule was not tested.

**What the merge changed.** Before the merge (origin/main 916d8af), the same judging runs gave:

- Pure pursuit: 06a contacted in all three runs, before and after the merge. On 06b it had no contact either time.
- Delay LQR: 06a contacted in all three runs. On 06b there was no contact, but the map safety layer held the car stopped for about 68 of the 70 seconds.

After that merge, the delay law's 06b runs contacted the wall after 29–34 ticks of safety stop. The initial investigation did not isolate which part of the merge caused the change.

**Tuning seeds, merged code.** This is context only and does not count toward the rule. Pure pursuit contacted in 3 of 6 runs, all on 06a. Delay LQR contacted in 4 of 6. One 06a run (seed 1000) finished four laps with no contact, with a flying median of 11.25 s over three laps.

**What the runs show.**

- **The contact is not specific to the delay law.** Pure pursuit contacted on 06a in all three judged runs, before and after the merge.
- **Changing the law's settings did not remove contact.** Every setting in the sweep left three to five contacts in six tuning runs, before the merge. The merged tuning run had four.
- **A lower cornering ceiling did not remove contact either.** With the online lateral ceiling at 2.5 m/s² instead of 2.9 (a diagnostic, not a proposed setting), pure pursuit contacted in five of six judged runs and the delay law in six of six. So the ceiling is not the whole cause.
- **The floor gives less lateral acceleration than the car asks for.** The floor's `grip_curve` (in `sfu_course_2026_11_06`, in `tools/f1tenth_sim/sim_fidelity/calibration.py`) delivers 2.56 m/s² when 2.75 is asked. At 2.9 asked it delivers about 2.6 m/s², and at 2.5 about 2.4 m/s² (both interpolated).
- **That was a likely part of the cause.** Car 2's calibrated limit was then 2.9 m/s² (`max_lateral_accel`), so the line's corners asked for more grip than the floor gave. The initial investigation did not isolate the rest of the cause.

**Limits of the initial investigation.** These describe that experiment, not the current verification above.

- The cause of the 06a contacts, for both laws, is not isolated.
- gap_follow was not re-measured on the merged code, so the lap-time half of the rule is still open.
- The brief's reference (11.77 s on 06a and 8.91 s on 06b, at 5.0 m/s over six seeds) has conditions that are not recorded in this repository.
- The committed simulator validation reports 13.84 s for 06a under other conditions, including a 0.349 rad steering cap ([simulator calibration](sim-calibration.md#car-2-floor-of-2026-11-06)). The two numbers do not describe the same condition.
- Car 1 has no numbers for this law.
- Nothing here has been driven on the car.

#### Every initial configuration tried

"Pre-merge" is origin/main 916d8af. "Merged" is origin/main 2fb626e with this branch. Contacts are counted over six runs (three seeds on each course).

| Run | Law and settings | Code | Seeds | Contacts | Notes |
|---|---|---|---|---|---|
| baseline | pure pursuit, shipped | pre-merge | judging | 3 | 06b flying median 9.92 s |
| baseline | pure pursuit, shipped | merged | judging | 3 | 06b flying median 10.00 s |
| tuning | pure pursuit, shipped | pre-merge | tuning | 3 | one 06b run stopped for 1,357 ticks |
| tuning | pure pursuit, shipped | merged | tuning | 3 | |
| tuning | Stanley, `delay_compensation` 0.125 s | pre-merge | tuning | 5 | one 06a run stopped for the rest of the run |
| tuning | LQR, `delay_compensation` 0.125 s | pre-merge | tuning | 3 | 06b runs stopped for the rest of the run |
| tuning | delay LQR, earlier revision (q 1) | pre-merge | tuning | 3 | exact settings not recorded |
| sweep | delay LQR, q 1, r 1, lead 0.067 s | pre-merge | tuning | 5 | lead was a parameter in an earlier revision |
| sweep | delay LQR, q 2, r 1, lead 0.067 s | pre-merge | tuning | 5 | as above |
| sweep | delay LQR, q 1, r 2, lead 0.067 s | pre-merge | tuning | 4 | one 06a run finished four laps |
| sweep | delay LQR, q 2, r 2, lead 0.067 s | pre-merge | tuning | 4 | one 06a run finished four laps |
| sweep | delay LQR, q 1, r 1, lead 0.125 s | pre-merge | tuning | 5 | 0.125 s cannot be set in the committed code |
| sweep | delay LQR, q 0.5, r 1, lead 0.067 s | pre-merge | tuning | 4 | as above |
| re-profiled | pure pursuit, line re-planned with brake 3.0 m/s² | pre-merge | tuning | 4 | line not committed |
| re-profiled | delay LQR, line re-planned with brake 3.0 m/s² | pre-merge | tuning | 5 | line not committed |
| shipped | delay LQR, committed settings | pre-merge | tuning | 5 | |
| shipped | delay LQR, committed settings | pre-merge | judging | 3 | 06b stopped for the rest of the run |
| shipped | pure pursuit | merged | judging | 3 | the comparison in the table above |
| shipped | delay LQR | merged | judging | 6 | the comparison in the table above |
| shipped | delay LQR | merged | tuning | 4 | |
| diagnostic | pure pursuit, online lateral 2.5 m/s² | merged | judging | 5 | not adopted |
| diagnostic | delay LQR, online lateral 2.5 m/s² | merged | judging | 6 | not adopted |

The sweep ran on an earlier revision in which the feedforward lead was an explicit parameter. The committed code derives it as 1/15 s, which is 0.067 s. The 0.067 s rows can therefore be repeated only to rounding, and the 0.125 s rows cannot be repeated with the committed code.

</details>

## Delay MPC results

**2026-11-08: rejected for promotion.** Delay MPC had zero body-contact ticks in all 55 runs: 24 tuning runs, 30 comparison runs and one diagnostic repeat. It failed the zero-stop requirement on 06b. Car 2's default remains `delay_lqr`; the six-seed promotion gate was not attempted after the three-seed comparison failed.

The comparison uses the supplied shipped lines, floor `sfu_course_2026_11_06`, car `rb2`, map safety, four requested laps, 0.42 rad steering and a 5 m/s speed ceiling. Seeds are 12345–12347. Flying laps are completed laps 2–4.

`S4` is the final experimental `delay_mpc` configuration: 32 × 25 ms steps, weights `q_lateral=4`, `q_heading=4`, `r_steer=1`, `r_rate=0`. It was selected on tuning seed 1000 before the comparison. The initial `S0` configuration used `q_heading=1`, `r_rate=0.001` and is included below as a failed alternative.

| Course | Law/config | Lateral ceiling (m/s²) | Four-lap runs | Flying median (range), s | Contact / stop ticks | Mean / max cross-track (m) | Timing p50 / p99 (ms) |
|---|---|---:|---:|---|---:|---|---|
| 06a | delay LQR | 2.7 | 3/3 | 11.425 (11.400–11.475) | 0 / 0 | 0.0897 / 0.2318 | 0.159 / 0.335 |
| 06a | delay MPC S4 | 2.7 | 3/3 | 11.525 (11.450–11.575) | 0 / 0 | 0.0919 / 0.2603 | 2.831 / 6.236 |
| 06a | delay MPC S4 | 2.9 | 3/3 | 11.425 (11.375–11.500) | 0 / 0 | 0.0974 / 0.2556 | 2.705 / 6.052 |
| 06a | delay MPC S4 | 3.2 | 3/3 | 11.350 (11.325–11.375) | 0 / 1 | 0.1080 / 0.2629 | 2.652 / 6.298 |
| 06b | delay LQR | 2.7 | 3/3 | 8.800 (8.750–8.875) | 0 / 0 | 0.0964 / 0.2862 | 0.167 / 0.338 |
| 06b | delay MPC S4 | 2.7 | 3/3 | 9.125 (8.925–9.275) | 0 / 77 | 0.1069 / 0.3340 | 2.706 / 6.123 |
| 06b | delay MPC S4 | 2.9 | 3/3 | 9.000 (8.875–9.200) | 0 / 94 | 0.1145 / 0.3455 | 2.860 / 6.853 |
| 06b | delay MPC S4 | 3.2 | 1/3 | 9.238 (8.975–9.675), 6 laps only | 0 / 3797 | 0.2260 / 0.3629 | 2.559 / 5.860 |
| 06a | delay MPC S0 | 2.7 | 3/3 | 11.425 (11.425–11.500) | 0 / 0 | 0.0822 / 0.2483 | 2.168 / 4.077 |
| 06b | delay MPC S0 | 2.7 | 1/3 | 9.000 (8.950–9.425), 5 laps only | 0 / 4043 | 0.2164 / 0.3549 | 2.306 / 4.076 |

Each complete row pools nine flying laps. Cross-track means average the three whole-run means; maxima are the largest run maximum. Timing reports the median of the three per-run p50 values and the largest per-run p99, **not pooled timing percentiles**. It includes QP assembly and any fallback work.

S4 had 12 isolated solver misses across its 18 comparison runs, each answered by fallback; no run latched a miss stop. Safety stops account for the stop column. Both stalled S4 runs at 3.2 m/s² reached the 70 s limit, completing two and three laps respectively.

The printed `finished True` field from `pure_course.py` only requires one lap. Use the number of returned lap times to check four-lap completion. Contact is always the harness's independent body check.

### Reproduce the comparison

**Any terminal at the workspace root**, with `rb` available and the simulator environment installed. The supplied CSV paths below are machine-local, read-only inputs. If they are unavailable, obtain the same shipped lines before comparing results.

```bash
ln -s /home/bryan/racerbot-ws/.sim .sim  # fresh worktree only
RB_NO_QUEUE=1 RB_WS=$(pwd) rb colcon build --symlink-install \
    --packages-select racerbot_car drive_intent pure_pursuit

for course in a b; do
    for law in delay_lqr delay_mpc; do
        lateral=2.7
        if [ "$law" = delay_mpc ]; then lateral=2.7,2.9,3.2; fi
        for seed in 12345 12346 12347; do
            RB_NO_QUEUE=1 RACERBOT_CAR=rb2 RB_WS=$(pwd) rb env RACERBOT_CAR=rb2 \
                python3 tools/f1tenth_sim/pure_course.py \
                --track car2_course_2026_11_06${course}_scans \
                --floor sfu_course_2026_11_06 \
                --line /home/bryan/.claude/jobs/8e0da722/tmp/${course}06/raceline_profiled.csv \
                --profiled --safety map --laps 4 --steering-limit 0.42 \
                --speed 5.0 --lateral "$lateral" --seed "$seed" --timeout 70 --law "$law"
        done
    done
done
```

**Working when:** each run prints contact, stop, cross-track and lap-time fields. This is a rejected candidate, so stops are expected on 06b. The loop uses one simulator process; never exceed two on the shared machine.

**If it fails:** a missing CSV is an unavailable input, not a law failure. A missing simulator import means the environment is incomplete; follow the [simulator setup](../tools/f1tenth_sim/README.md). Host load can change deadline misses and subsequent laps, so small differences from this record are possible.

To repeat S0, add `--set delay_mpc.q_heading=1 --set delay_mpc.r_rate=0.001` to a delay MPC command with `--lateral 2.7`.

<details>
<summary><b>Capture timing and the full result</b>: optional when reproducing the performance measurements.</summary>

**Any terminal at the workspace root.** This wraps the same `pure_course` call and prints the returned measurement dictionary, including per-tick law timing percentiles and miss counts.

```bash
RB_NO_QUEUE=1 RACERBOT_CAR=rb2 RB_WS=$(pwd) rb env RACERBOT_CAR=rb2 python3 - <<'PY'
import json
import sys
from pathlib import Path

root = Path.cwd()
sys.path[:0] = [str(root / 'tools/f1tenth_sim'), str(root / 'tools'),
                str(root / 'src/pure_pursuit')]
import pure_course

run = pure_course.rv.run_pure_solo
def report(*args, **kwargs):
    result = run(*args, **kwargs)
    print(json.dumps(result, indent=2))
    return result
pure_course.rv.run_pure_solo = report
pure_course.main([
    '--track', 'car2_course_2026_11_06b_scans',
    '--floor', 'sfu_course_2026_11_06',
    '--line', '/home/bryan/.claude/jobs/8e0da722/tmp/b06/raceline_profiled.csv',
    '--profiled', '--safety', 'map', '--laps', '4', '--steering-limit', '0.42',
    '--speed', '5.0', '--lateral', '2.7', '--seed', '12345', '--timeout', '70',
    '--law', 'delay_mpc'])
PY
```

**Working when:** the JSON includes `law_compute_ms_p50`, `law_compute_ms_p99`, `mpc_miss_steps` and `mpc_latched`. Repeat on the Jetson to measure its budget; the numbers here are host measurements.

</details>

### Every configuration tried

All tuning rows used seed 1000 on both courses at 2.7 m/s² unless stated otherwise. Values below are flying medians; full per-run lap times, cross-track and captured timing are in [the measurement record](delay-mpc-comparison-2026-11-08.json). Every tuning run completed four laps with zero contact.

| Config | Changes from S0 | 06a / 06b flying median (s) | 06a / 06b stop ticks |
|---|---|---|---|
| S0 | initial weights 4/1/1, rate cost 0.001, 32 steps | 11.425 / 9.025 | 0 / 16 |
| S0, lateral 2.9 | online ceiling only | 11.350 / 9.025 | 1 / 52 |
| S0, lateral 3.2 | online ceiling only | 11.250 / 9.200 | 0 / 56 |
| Q1 | lateral weight 1, rate cost 0 | 11.500 / 8.900 | 0 / 3 |
| Q8 | lateral weight 8, rate cost 0 | 11.550 / 9.050 | 0 / 1 |
| Q16 | lateral weight 16, rate cost 0 | 11.675 / 9.150 | 2 / 10 |
| H01 | heading weight 0.1, rate cost 0 | 11.500 / 9.075 | 0 / 1 |
| S4 | heading weight 4, rate cost 0 | 11.550 / 8.975 | 0 / 0 |
| H16 | heading weight 16, rate cost 0 | 11.625 / 9.000 | 7 / 2 |
| N8 | Q1 with 8 steps | 11.475 / 9.000 | 0 / 30 |
| N16 | Q1 with 16 steps | 11.500 / 8.900 | 0 / 4 |
| N24 | Q1 with 24 steps | 11.500 / 8.900 | 1 / 1 |

The comparison table lists every subsequent configuration: delay LQR at 2.7, S4 at all three ceilings, and S0 at 2.7. The diagnostic repeated S4, 06b, 2.7, seed 12345 once: zero contact, 24 stop ticks, flying laps 9.125 / 8.975 / 9.075 s.

### Test break table

The standard package pytest runner collects these tests. Each new or extended case was run against the stated deliberate implementation break, observed failing, then run green against restored code. Runtime mutations were confined to the test process.

| Test or cases | Deliberate break | Observed result |
|---|---|---|
| `test_straight_zero_and_signed_offset` | replace `steer` with constant zero | failed signed correction |
| `test_circle_equilibrium`, both directions | replace `steer` with zero | both failed feedforward |
| `test_step_demand_activates_rate_and_angle_constraints` | zero stub; separately remove QP rate bounds | failed; raw optimizer plan exposes missing bounds |
| `test_unconstrained_straight_matches_independent_servo_lqr` | increase QP lateral weight 10% | failed independent Riccati oracle |
| `test_action_initial_state_matches_closed_form_rollout` | zero steering stub bypassing rollout | failed missing action state |
| `test_servo_model_and_affine_curvature_match_euler_oracle` | negate lateral/heading dynamics coupling | failed Euler oracle |
| `test_reference_advances_at_profile_speeds` | sample a constant arc throughout the horizon | failed profile-speed recurrence |
| `test_nonfinite_input_is_a_law_error`, 15 cases | remove finite checks | all 15 failed |
| `test_invalid_parameters_are_refused`, 15 cases | no-op constructor | all 15 failed |
| `test_deadline_miss_discards_warm_start` | force a permissive deadline | failed expected miss |
| `test_yaml_and_factory_cover_every_parameter` | drop `delay_mpc.q_heading` in factory | failed parameter equality |
| `test_reset_forgets_plan_and_history` | no-op reset | failed retained warm start |
| `test_yaml_mirrors_the_param_dataclasses` | change dataclass heading default 10% | failed YAML equality |
| `test_lb_held_on_line_steers_toward_it[delay_mpc]` | constant-zero steering stub | failed steering direction |
| released / never-received / stale LB cases for delay MPC | bypass `_deadman_status` | all three failed published-speed assertions |
| miss fallback / repeated-miss latch / LB latch-clear cases for delay MPC | remove `delay_mpc` from `DEADLINE_LAWS` | all three failed |
| fallback live tune / intent solver exclusion cases for delay MPC | remove `delay_mpc` from `DEADLINE_LAWS` | both failed |
| `test_delay_mpc_success_records_timing_and_replays_plan` | remove `delay_mpc` from `DEADLINE_LAWS` | failed timing dispatch |
| `test_delay_mpc_parameters_reach_node_and_are_launch_only` | drop heading parameter from node collection | failed received-parameter equality |
| `test_delay_mpc_node_defaults_match_dataclass` | change node heading default 10% | failed defaults equality |

The initial rate-activity test also exposed a 0.000000565 rad interior error with OSQP's `1e-5` tolerances. The implementation now solves at `1e-8`; the test tolerance was retained.

### Suite verification

Run one suite per pytest invocation, with ROS sourced through `rb` and an isolated ROS domain. **Any terminal at the workspace root:**

```bash
RB_NO_QUEUE=1 RB_WS=$(pwd) rb env ROS_DOMAIN_ID=79 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST \
    python3 -m pytest src/pure_pursuit/test/ -q
RB_NO_QUEUE=1 RB_WS=$(pwd) rb env ROS_DOMAIN_ID=79 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST \
    python3 -m pytest tools/f1tenth_sim/tests/ -q
RB_NO_QUEUE=1 RB_WS=$(pwd) rb env ROS_DOMAIN_ID=79 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST \
    python3 -m pytest src/racerbot_car/test/ -q
RB_NO_QUEUE=1 RB_WS=$(pwd) rb env ROS_DOMAIN_ID=79 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST \
    python3 -m pytest src/racerbot_launch/test/ -q
```

**Working when:** the results match the table below. The pure pursuit suite takes about 12 minutes on this host because it rebuilds minimum-time lines.

| Suite | Result, 2026-11-08 |
|---|---|
| `src/pure_pursuit/test/` | 1029 passed, 11 existing skips, one known failure |
| `tools/f1tenth_sim/tests/` | 636 passed |
| `src/racerbot_car/test/` | 231 passed |
| `src/racerbot_launch/test/` | 312 passed |

The known failure is `test_raceline_optimizer.py::test_the_trust_region_bounds_a_single_pass`: maximum radius 5.250001622 m against a 5.250001 m bound. The test and optimiser were left intact.

**If collection fails:** initialise the pinned `particle_filter` and `web_dashboards` submodules and build the launch-test dependencies. This fresh worktree initially lacked `race_diagnostics`, `gap_follow` and dashboard packages. After building those and the launch packages, both configuration suites passed. No collection errors were bypassed.

## Installing osqp for MPC

MPC needs the `osqp` solver in the same Python that ROS uses. On the car it was installed with:

**Any terminal** on the car:

```bash
pip install --user --break-system-packages osqp==1.1.3
```

**Working when:** `python3 -c "import osqp; print(osqp.__version__)"` prints `1.1.3`.

**If it doesn't:** check that `numpy` stayed at 1.26.4 (`python3 -c "import numpy; print(numpy.__version__)"`). osqp 1.1.3 installs alongside the system numpy without upgrading it.

`--break-system-packages` is needed because Ubuntu 24.04 protects the system Python. `--user` keeps the package in your home directory, so it cannot break anything apt installed.

The other four laws never import osqp, so they work without it.

To re-measure the solve time on the Jetson, use `python3 tools/mpc_benchmark/benchmark_mpc.py --steps 20000`. The 2026-10-19 numbers are in [mpc-benchmark-2026-10-19.json](mpc-benchmark-2026-10-19.json).

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Node exits at startup naming `steering_law` | the name is misspelt | use one of `pure_pursuit`, `stanley`, `lqr`, `mpc`, `delay_lqr`, `delay_mpc` |
| `steering_law 'mpc' needs the 'osqp' package` | osqp not installed for this user's Python | [Installing osqp](#installing-osqp-for-mpc) |
| Decision state `mpc_miss_latched`, car stopped | three late or failed MPC solves in a row | release LB and press it again to re-arm. If it repeats, something else is loading the CPU: check `top`, and consider `pure_pursuit` for this session |
| A dashboard tune of `stanley.gain` (etc.) is refused | law parameters are launch-time only | edit `pure_pursuit.yaml` and relaunch |
| Weaving on the car with Stanley or LQR | more delay or position noise than the simulator models | raise `<law>.delay_compensation` by 0.02 s at a time, or the time constant; or go back to `pure_pursuit` |
| A unit test says the YAML and the dataclass disagree | a gain was changed in one place | change all three places ([Tuning](#tuning-stanley-and-lqr)) |

## See also

- [racing-autonomy.md](racing-autonomy.md) — the whole race pipeline the steering law sits inside.
- [sim-validation.md](sim-validation.md#comparing-steering-laws-head-to-head) — the comparison harness and its flags.
- [simulator.md](simulator.md) — the no-ROS simulator and its car fidelity profile.
- `docs/superpowers/specs/2026-09-27-tracking-0*.md` — the original design specs. They describe Stanley's curvature term the old way; this doc and the code are current.
