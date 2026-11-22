# Recorded-driving tyre fit

`fit_slam_tyres.py` fits the existing saturating Gym single-track tyre model
against trusted `slam_motion.py` tracks using `slam_rollout.py`. This follows
the recorded-driving motivation of [On-Track SysID](https://arxiv.org/abs/2411.17508),
not its axle-force identification algorithm. These are effective rollout
parameters, not independently identified physical tyre coefficients.

The fit opens **only** the November development runs hard-coded in `DEV`.
No October runs enter the fit. It holds mass, CG, yaw inertia, friction,
actuation, speed loop and longitudinal coupling fixed. It searches front/rear
Gym stiffness, front/rear force peaks and common shape, with 20 seeded coarse
candidates and at most 20 Nelder-Mead evaluations. Search uses up to six
evenly spaced H=2 windows per run; the best three distinct candidates are
ranked on **all** eligible H=2 development windows. This budget is not a
convergence or identifiability claim.

The minimized objective averages three equally weighted normalized errors:
yaw RMS / 0.30 rad/s, CG slip RMS / 0.10 rad and horizon position / 0.50 m.
Windows, rather than runs, are equally weighted. Rollouts construct fresh
seeded plants and retain the original command delays and speed controllers.

TyreModel normally bypasses the external steering grip clamp. `fit` tests
turning the existing friction-circle clamp back on; `ablate` additionally
tests ProgressiveGrip with a command/SLAM envelope derived **only from dev**.
The envelope is an approximate diagnostic, not an axle-force measurement.
Both clamp-on variants retain the same tyre parameters. The candidate keeps
the external clamp off to avoid applying measured saturation twice.

From the workspace root, after Gym and the ROS-free reader are installed:

```bash
export PYTHONPATH=.sim/python:.sim/f1tenth_gym:src/gap_follow:src/pure_pursuit:tools/f1tenth_sim:tools
export NUMBA_CACHE_DIR=.sim/numba-cache OPENBLAS_NUM_THREADS=1
PY=$HOME/.local/racerbot-analysis-venv/bin/python
OUT=$HOME/.cache/racerbot/tyre_fit
mkdir -p "$OUT"
for CAR in rb1 rb2; do
  "$PY" tools/f1tenth_sim/fit_slam_tyres.py fit --car "$CAR" --out "$OUT/${CAR}_frozen.json"
  "$PY" tools/f1tenth_sim/fit_slam_tyres.py ablate --frozen "$OUT/${CAR}_frozen.json" --out "$OUT/${CAR}_final_frozen.json"
done
# Do not inspect evaluation results or revise parameters during fitting.
for CAR in rb1 rb2; do
  "$PY" tools/f1tenth_sim/fit_slam_tyres.py evaluate --frozen "$OUT/${CAR}_final_frozen.json" --out "$OUT/${CAR}_evaluation.json"
done
```

Evaluation refuses a changed fingerprint, development manifest or protocol.
Before evaluation, acceptance requires yaw/slip/position means to strictly
improve with non-worse medians at **both H=2 and H=4**. Speed RMS mean and
median may worsen by at most 0.02 m/s. rb2 must pass November and the
independent October floor separately. rb1 has only one usable evaluation run,
20261116-095242. H=1 is diagnostic only. Historically published evaluation
runs are not newly untouched data.

The compact numerical record is `slam_tyres_2026_11_fit.json`; full frozen
search histories, window scores and logs remain under `~/.cache/racerbot/tyre_fit`.
**Both candidates were rejected on held-out rollouts. Neither car's default
calibration changed.** Retain these values as negative evidence, not approved
physical tyre measurements.
Profiles `slam_tyres_2026_11_rb1` and `slam_tyres_2026_11_rb2` are explicitly
selectable with `slam_rollout.py --floor`. Default promotion is a separate,
acceptance-gated decision, never performed automatically by the fitter.
Localisation's wheel/gyro prediction, differentiated SLAM noise, missing rack
measurements and unmeasured CG/inertia limit a tyre-only interpretation.
