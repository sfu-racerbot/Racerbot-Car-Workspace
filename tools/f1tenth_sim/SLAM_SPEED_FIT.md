# November SLAM speed-scale candidate

`fit_slam_speed.py freeze` uses **only November development loc tracks** and
`slam_motion`'s pooled sample-weighted median `abs(wheel_v) / ground_speed`.
Its mask is trusted, forward longitudinal velocity and ground speed >=0.5 m/s;
the same smoothing, jump margins and segmentation as Step 2 apply. No rollout
scores, evaluation tracks or October measurements enter the scale estimate.
The three specified development drives per car are recorded in the frozen JSON.

| Car | Moving samples | Wheel / ground | Command scale | Decision |
|---|---:|---:|---:|---|
| rb1 | 3257 | 1.0746581411238885 | 1.0595126187713653 | Rejected |
| rb2 | 5350 | 1.0261486728316354 | 1.0108904807240708 | Accepted on November |

## Semantics

As for existing floors, `odom_speed_scale` multiplies simulated true speed to
produce wheel odometry; **it does not slow the plant**. The physical command
conversion is separate:

```
speed_command_scale = command_eRPM_gain / abs(odom_eRPM_gain) / wheel_ground_ratio
```

The recorded `metadata/params/{ackermann_to_vesc_node,vesc_to_odom_node}.yaml`
gains are respectively **4600 / -4040** for all three rb1 dev runs and
**4614 / -4447.983786** for all three rb2 dev runs. All offsets are zero.
Inconsistent development gains or nonzero offsets are rejected, not averaged.
Thus a wheel ratio above one does **not** imply command scale below one.
The rb1 candidate actually speeds up the simulator and worsens the discrepancy.

`FidelityPlant.step` multiplies the delayed, clipped command by command scale;
its `OdomSensor` independently multiplies true speed by odom scale. Recorded
commands are not pre-scaled again; rollout scoring uses truth, not the sensor.
`racerbot_sim.sim_bridge.DeadReckonedOdometry.update` uses the same multiplicative
odom convention, including its dead-reckoned yaw/pose integration. That ROS
bridge uses raw Gym and explicit launch odom parameters, **not** FidelityPlant
or floor selection; the new factory default does not silently alter its launch
parameters. No bridge, topic contract, hardware config or physical-car gains
were changed. Bridge odometry tests and hardware guards are run separately.

## Frozen protocol and verdicts

Before evaluation, freeze the ratio, metadata gains, development manifest and
acceptance rule under a SHA256 fingerprint. Acceptance requires **both H=2 and
H=4 November evaluation** speed RMS means and medians to strictly improve;
yaw RMS, CG slip RMS and horizon position means may be at most 1.02 times the
baseline. October is a reported independent-floor check, not another gate.
All paired rollouts use the Step 3 fresh-plant, recorded-command procedure.

rb1 is rejected: speed, yaw and position worsen. Only `20261116-095242` has
usable rb1 evaluation windows. Its scale remains a labelled rejected floor.
rb2 passes November; these **two fields only** become its default calibration.
Its October speed RMS worsens slightly, so this is not a universal-floor claim.
Tyres, actuators, mass, CG and inertia remain at the default values for both.
The previous rb2 speed defaults are available as
`--floor pre_slam_speed_2026_11_rb2`; explicit candidates are
`--floor slam_speed_2026_11_rb1` and `--floor slam_speed_2026_11_rb2`.

`slam_speed_2026_11_fit.json` contains frozen measurement evidence, per-run and
pooled mean/median metrics and verdicts. Full per-window outputs remain cached.
The pre-change baseline in the script explicitly restores both speed scales
to 1, so rerunning after promotion does not compare the candidate with itself.

```bash
export PYTHONPATH=.sim/python:.sim/f1tenth_gym:src/gap_follow:src/pure_pursuit:tools/f1tenth_sim:tools
export NUMBA_CACHE_DIR=.sim/numba-cache
PY="$HOME/.local/racerbot-analysis-venv/bin/python"
OUT="$HOME/.cache/racerbot/speed_fit"
mkdir -p "$OUT"
for car in rb1 rb2; do
  "$PY" tools/f1tenth_sim/fit_slam_speed.py freeze --car "$car" --out "$OUT/${car}_frozen.json"
done
for car in rb1 rb2; do
  "$PY" tools/f1tenth_sim/fit_slam_speed.py evaluate --frozen "$OUT/${car}_frozen.json" --out "$OUT/${car}_evaluation.json"
done
```

The ratio is an effective measurement, not a tyre or gearing identification.
Localization still predicts with wheel/gyro before scan correction; trusted
matching does not guarantee independent ground truth. The rb2 speed gain is
small and October does not reproduce it. No physical or visual validation.
