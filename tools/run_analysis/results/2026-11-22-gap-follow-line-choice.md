# Gap-follow on the November SLAM course: keep the current controller

Car 2 was quicker on its November 11 map when it estimated the bend exit from two LiDAR cross-sections instead of aiming at a single point. **Reject the rule:** three previously passing floor controls contacted a wall, and a fourth timed out. This is a simulation finding, not a controller or settings change.

## Matched line-choice experiment

The candidate kept the chosen, inflated gap and the stock braking, stop, steering and speed limits. When that gap lay entirely to one side, two bounded cross-sections at one-third and two-thirds of the stock target range predicted the exit direction. The target could not cross a beam below 95% of the stock target's depth. Straight-ahead gaps, range-cap openings, launch, creep and opponent operation retained the original aim. The controller did not see the map or sector labels. Only this candidate was tested, without a parameter sweep.

| Car 2, documented scanner and November floor | Stock | Exit estimate | Difference |
|---|---:|---:|---:|
| Nine archived seeds, seconds per lap | 10.697 | 10.650 | −0.047 |
| Three fresh seeds (92501–92503), seconds per lap | 10.700 | 10.653 | −0.047 |
| Worst sampled chassis clearance over both sets | 0.112 m | 0.180 m | +0.068 m |
| Stop commands over both sets | 24 | 24 | 0 |

Both arms finished three laps per seed without contact. Eleven pairs got faster; one tied. The middle bend (50–62.5% of simulated scoring distance) saved 0.049 s/lap across all twelve seeds. The car entered faster (2.55 → 2.65 m/s), traveled 0.028 m less and exited **slower** (2.04 → 2.00 m/s). The rule changed 1,773 targets in that bend, none at 75–87.5%. The 0.178 s difference seen previously *between scanner geometries* is not a saving from this controller.

The 82 pairs include those twelve primary pairs, the 61 earlier timing/floor/Levine cases, and the nine Car 2 floor cases additionally tested on the documented geometry. Like-for-like stock results matched all 61 prior outcome records and all nine mapped command digests. Three standing-start-inclusive laps were required for timing and Spielberg controls, two laps within 90 seconds for stress and Levine. Contact takes precedence; no failed arm was used as a fast lap. These are simulation repeats, not independent physical trials.

| Previously passing case | Geometry | Floor / grip multiplier | Seed | New outcome |
|---|---|---|---:|---|
| Car 1, October 26 Car 1 layout | Pinned | Car / 0.85 | 12346 | Contact |
| Car 1, November 11 Car 2 layout | Pinned | November / 0.85 | 12347 | Contact |
| Car 1, October 30 layout | Pinned | November / 0.85 | 12345 | Contact |
| Car 2, November 6a scans layout | Pinned | November / 0.85 | 12347 | Timeout |

The archived synthetic/floor layouts above are regression controls, not newly decoded November 6 bags. Among jointly passing runs, seven pairs lost sampled clearance (two lost 0.050 m). One narrow-loop pair also added six stop commands. No previously passing case failed in the nine additional documented-floor pairs, but only one stock run there passed. The documented-mount narrow-loop mean regressed 0.081 s/lap. Spielberg commands were unchanged in all three positive controls. On Levine, the pinned nominal-grip timeout became a pass; the pinned low-grip timeout and both documented-geometry contacts remained. A rescued run does not offset a new failure.

## Earlier results preserved by this checkpoint

| Question | Supported conclusion |
|---|---|
| Did the simulator use the uploaded SLAM map? | The raster regenerated from Car 2 mapping run 151026's rebuild matches all 141,696 cells and the 293-point scoring centerline. Physical run 152132 used the rebuild for pure pursuit; it is not a gap-follow holdout. |
| Does the accepted wide-opening aim help here? | No: 27 mapped reference/open pairs across three car/mount groups had identical commands. No selected gap reached the 10 m cap. Four tight courses together had 108 identical-command pairs. |
| Where did stock lose time? | At 50–62.5% the documented scanner took 0.178 s/lap longer than pinned, mostly on a longer line; curvature limited every mid-bend command. A separate pinned, archived lower-centering comparison found 0.030 s/lap at 75–87.5%, mostly on turn-in. Neither comparison supports relaxing a safety threshold. |
| Did the Spielberg result depend on mount geometry? | On six matched Car 2 seeds the accepted open-aim rule saved 1.218 s/lap pinned and 1.719 s/lap documented; all Spielberg pairs finished without contact. The documented scanner is 0.26 m ahead of the rear axle (0.08 m from body center); pinned places it 0.26 m ahead of body center. |
| Why retain the original presets on tight tracks? | Five earlier presets failed the speed or safety criteria. Lower/higher centering, for example, turned seven Car 1 and six Car 2 passes into contacts in their respective full floor panels. Checking the measured turning arc saved 0.783 s/lap on Car 1's narrow loop but slowed the other eleven groups and turned three Car 2 passes into timeouts. No rejected rule was rerun here. |
| Are the Levine stalls all controller faults? | A scanner/body frame mismatch explained false clearance at 20/26 sampled stalled poses, not all contacts. On one matched seed a single speed reading at 1.625 s changed a steering correction and switched timeout to pass. These are diagnostic counterexamples, not a wheel-filter recommendation. |
| Does the uploaded Car 1 UVic race bag reproduce the reported steering? | On 148 scans from run 155542, maximum replay error was 0.016399 rad (published: 0.0164). Follow-cap episodes totaled 30.175 s (published: 30.18). Centering first reached zero at 86.233 s, not 86.01. This was a race with opponents, not an empty-course lap benchmark. |

## Verification and scope

From the workspace root, using the simulator dependencies and the uploaded mapping data:

```sh
PYTHONPATH=.sim/python python3 tools/run_analysis/verify_november_gap_map.py
```

This read-only check regenerates the raster, map YAML and scoring centerline in a temporary directory and compares them to the tracked simulator assets. It also checks the pure-pursuit run's recorded map and line paths. It does not run a controller. Per-tick simulation traces and the offline verdict checker are in the separately delivered experiment evidence; no raw or generated run data is added to this repository.

Simulation controller: [accepted workspace revision](https://github.com/sfu-racerbot/Racerbot-Car-Workspace/commit/691b427f4316d0e64490cca1ba9d4c3fd476b8de); pinned Gym revision: `bdaec1420c3b0f103858d289866d0d4e2e597c30`. [Implementation PR](https://github.com/sfu-racerbot/Racerbot-Car-Workspace/pull/95) and [benchmark guide](https://github.com/sfu-racerbot/racerbot-docs/pull/53) are unchanged. No real-car test or safety certification.
