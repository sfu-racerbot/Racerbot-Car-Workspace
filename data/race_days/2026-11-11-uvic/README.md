# UVic race day, 2026-11-11: which recordings are real racing

> **Who this is for:** anyone using the race-day recordings for the simulator, lap studies or the website.
> **Read first:** [finding the racing in a long recording](../../../docs/run-diagnostics.md#finding-the-racing-in-a-long-recording).
> **You will be able to:** pick out only the stretches where a car was really racing.

On race day both cars recorded the whole time an algorithm was running. So the
recordings also hold the car parked between races, short moves to the start line,
cars stuck against a wall, and people picking the cars up and carrying them. None of
that is racing, and it must not go into a fit of how the car drives.

[`segments.yaml`](segments.yaml) lists, for every race-day recording of both cars,
the stretches where the car was really driving, plus the times it was stuck with LB
held, handled by a person, or spinning its wheels in the air. It was written by:

```bash
# any terminal, from the workspace root (no ROS needed, only numpy, PyYAML and rosbags)
python3 tools/run_analysis/race_segments.py data/rb1/racerbot_runs/20261111-* \
    data/rb2/racerbot_runs/20261111-* --yaml data/race_days/2026-11-11-uvic/segments.yaml
```

Times in it are seconds from the recording's first `/scan`. Add `first_scan_epoch` to
get the time the bag uses. `clock` is the same stretch on the cars' own clocks
(Pacific, UTC-8). Don't edit the file by hand: change the tool and run it again.

## How a stretch is judged, in plain words

The tool looks at every half second of a recording and asks four questions, using
only the car's own sensors:

1. **Are the wheels turning at walking pace or faster?** (wheel speed 0.5 m/s or more)
2. **Is what the LiDAR sees changing?** If the wheels turn but the view stays the same,
   the car is in someone's hand or pinned against something: *spinning*, not driving.
3. **Is LB held?** Driving with LB held is the algorithm in charge.
4. **If the wheels are still, is the car moving or tilting anyway?** A changing view
   (the LiDAR ranges changing by 0.1 m/s or more), or the IMU feeling the car tip (8 degrees
   a second or more), means a person is *handling* it. A parked car tips under 2 degrees a second.

Driving half-seconds are joined into stretches. A pause of up to 3 s stays inside a
stretch (traffic, a short stop). A longer pause, or any handling or spinning, ends it.
A stretch counts as **race** when it lasts at least 10 s, covers at least one lap of
the 28.17 m course by the wheels, and averages at least 1.5 m/s. Anything shorter is a
**short** move: repositioning, a launch check, a nudge.

Lap counts below are wheel distance divided by 28.17 m. The wheels read 2 to 4% high on
these cars, so they are close, not exact.

## The races

The races were run from about 3:44 p.m. These two recordings hold them:
car 2's `rb2/racerbot_runs/20261111-154204` and car 1's `rb1/racerbot_runs/20261111-155542`.
The timing agrees with the club Discord: "WERE WINNING" was posted at 3:53 p.m.,
during car 2's 3:52 race, and "WE WON" at 4:02 p.m. The announcement says the win was
over 5 laps, which matches the length of most stretches here.

| Car | Clock | Recording time | Wheel distance | About |
|---|---|---|---|---|
| car 2 | 15:44:57 to 15:45:56 | 171.0 to 230.5 s | 150 m | 5.3 laps |
| car 2 | 15:52:33 to 15:54:06 | 627.5 to 720.5 s | 231 m | 8.2 laps |
| car 1 | 15:56:02 to 15:56:58 | 16.5 to 72.0 s | 143 m | 5.1 laps |
| car 1 | 15:57:01 to 15:57:12 | 75.0 to 86.5 s | 28 m | 1.0 lap, then stuck against a wall until 15:57:20 |
| car 2 | 15:56:13 to 15:57:13 | 847.0 to 907.5 s | 149 m | 5.3 laps |
| car 2 | 15:57:16 to 15:58:18 | 910.0 to 972.5 s | 155 m | 5.5 laps |
| car 1 | 15:57:20 to 15:58:24 | 94.5 to 158.0 s | 158 m | 5.6 laps |

From 15:56 both of our cars were on the track at the same time.

Everything else in those two recordings is not racing: car 2 sat parked or was carried
for about 16 of its 21 minutes, and car 1 was parked or carried for its last four and a half.

## Before the races

The other recordings, 13:54 to 15:41, are testing and practice with several
algorithms (our gap follow and pure pursuit, team A's ftg and disparity extender).
Their race stretches in `segments.yaml` are real driving at race speed and are good
data for the simulator. The recordings cannot tell whether any of them were the
official timed laps, so don't call them races.

## Using it

- **Simulator calibration, grip and lap studies:** use only `kind: race` stretches.
  Cut each run to `first_scan_epoch + start_s` to `first_scan_epoch + end_s`.
- **Never** use `stalled_with_lb`, `handled` or `spinning` spans as driving data. In
  those the wheels and the car's real motion disagree.
- A crash into a wall inside a stretch stays in it. The car was still being driven.
