# Project state (as of 2026-11-22, after the review batch)

The owner handed PR review and merging to Vera on 2026-11-22. Re-check every PR and branch
on GitHub before acting: this is a snapshot.

## Merged on 2026-11-22

Workspace:
- [#100](https://github.com/sfu-racerbot/Racerbot-Car-Workspace/pull/100) calibrated physics in the ROS bridge (opt-in, stock stays default), then [#101](https://github.com/sfu-racerbot/Racerbot-Car-Workspace/pull/101) Car 2's measured LiDAR mount.
- [#102](https://github.com/sfu-racerbot/Racerbot-Car-Workspace/pull/102) offline run player (replaces #92), [#103](https://github.com/sfu-racerbot/Racerbot-Car-Workspace/pull/103) read-only Lichtblick viewer, [#104](https://github.com/sfu-racerbot/Racerbot-Car-Workspace/pull/104) this handoff folder, [#105](https://github.com/sfu-racerbot/Racerbot-Car-Workspace/pull/105) offline RL racer package.
- [#107](https://github.com/sfu-racerbot/Racerbot-Car-Workspace/pull/107) (`ecc92ec`), [#108](https://github.com/sfu-racerbot/Racerbot-Car-Workspace/pull/108) SLAM-measured calibration (rb2 speed scales accepted, tyre fits rejected), [#109](https://github.com/sfu-racerbot/Racerbot-Car-Workspace/pull/109) Neural Racer diagnosis and calibrated retraining setup behind a sign-off gate (replaces #106).
- [#110](https://github.com/sfu-racerbot/Racerbot-Car-Workspace/pull/110) calibrated wall contact on contact-map tracks, [#111](https://github.com/sfu-racerbot/Racerbot-Car-Workspace/pull/111) turn-response study ([turn-response.md](turn-response.md)); both cars keep their defaults.

Docs: [56](https://github.com/sfu-racerbot/racerbot-docs/pull/56), [57](https://github.com/sfu-racerbot/racerbot-docs/pull/57), [58](https://github.com/sfu-racerbot/racerbot-docs/pull/58), [59](https://github.com/sfu-racerbot/racerbot-docs/pull/59), [64](https://github.com/sfu-racerbot/racerbot-docs/pull/64) and [65](https://github.com/sfu-racerbot/racerbot-docs/pull/65). Website: [#29](https://github.com/sfu-racerbot/racerbot-website/pull/29).

Closed as duplicates: workspace #92 and #106, docs 51 to 55.

## Still open

- [#91](https://github.com/sfu-racerbot/Racerbot-Car-Workspace/pull/91) Neural Racer shadow model, swept-path braking cap and flank creep. Needs its own review because it changes the safety wrapper.
- [#93](https://github.com/sfu-racerbot/Racerbot-Car-Workspace/pull/93), [#94](https://github.com/sfu-racerbot/Racerbot-Car-Workspace/pull/94) and [#95](https://github.com/sfu-racerbot/Racerbot-Car-Workspace/pull/95), and the docs that describe them: [60](https://github.com/sfu-racerbot/racerbot-docs/pull/60) (#93), [61](https://github.com/sfu-racerbot/racerbot-docs/pull/61) (#91 and #94) and [62](https://github.com/sfu-racerbot/racerbot-docs/pull/62) (#95). Merge each doc after its workspace PR.
- Follow-up: `slam_replay.py` should write `free_thresh` 0.196.
- Older open PRs not reviewed in this batch: workspace #46, #47, #55, #84, #96, #97, #98, #99; docs #40, #63.

The sections below are the coordinator's earlier snapshot, kept for its evidence notes.

## Held for the owner's batch review

- Gap Sprint source backups, no matching remote branches: gap-follow diagnostic checkpoint local e7290179ca7f4a846aa9989a32c027e0f0be2586, false-opening verdict local 481046850b970594c1c790e8621b6be92f004400. Line-choice verdict and evidence retained for review. Same attribution ask; no additional action.
- Source backups for blocked GitHub pushes: four pursuit patches and offline tools, benchmark extractor, BNO provenance check, MuJoCo reproduction. Existing remote head only for [Gazebo](https://github.com/sfu-racerbot/Racerbot-Car-Workspace/commit/6075ccd7e67cbec230dc3958654f12ea5b1af26c) and [MuJoCo](https://github.com/sfu-racerbot/Racerbot-Car-Workspace/commit/dd9ad869cb31ea983c5fb336314b1bc4854b09f9), not new checkpoints. No AutoDRIVE remote checkpoint.
- Neural Racer isolated shutdown review: worker reports 85 tests passed, independent subscriber saw final zero for both signals; removing publish failed both checks. Shadow-only, no hardware or approval change.
- Gap Sprint batch: reject the exit-line rule. Checked 82 paired trials; all 12 documented Car 2 SLAM-map pairs passed, averaging 0.047 s/lap faster, but three passing pinned stress controls became contacts and one a timeout. Mapped sector evidence. No controller change; diagnostics-only checkpoint pending.
- Pursuit batch: entry check verdict, evidence. 18 saved negative scenarios: 107 legacy attempts and two contacts versus zero of each with veto, matching following alone. One independent rolling case admitted an attempt, aborted 1.35 m behind, wall gap 0.450 m versus 0.636 m following alone. No moving pass or hardware evidence. Keep passing disabled.
- 2026-11-22 | Neural Racer ROS 2 | Isolated selection test: unapproved frozen policy produced 30 shadow stops, no commands to mux. Manual selection of gap follow regained exact 10/10 held and 10/10 release commands. This was a stationary fixture, no motors and no active learned-driver verification. Assigned observed teardown ordering error to existing owner; do not bypass approval.
- 2026-11-22 | Passing | Pass-opening evidence: representative moving rival detected but commit assumed profile speed; following limited 116/122 ticks, abort 2.76 m behind. A pulled-aside parked car left a viable line; following alone passed with 0.911 m gap, legacy passing hit a simulated wall. Not a moving overtake. Passing stays off; bounded conservative entry-check experiment, no relaxed safety gates.
- 2026-11-22 | Neural Racer | Selectable-driver test and traces: saved policy, rb2 151026-rebuilt, 2,400/2,400 safety-wrapped commands arrived unchanged and simulated release commanded zero. Only 3.33 m in 60 s; gap follow lapped in 16.325 s on the same start. Offline CPU wiring, no ROS graph or hardware. Continued isolated ROS software verification without training.
- 2026-11-22 | Gap follow | Mapped-sector evidence: the largest sector difference (+0.178 s) compared *two scanner geometries*, mainly different path lengths, not a measured improvement available on the car. Same pinned geometry vs previously rejected centering comparator: +0.030 s on turn-in, curvature limit active 364/368 ticks. Gap Sprint owns the distinct next line-choice test; F1 research informed.
- 2026-11-22 | Viewer | Native AutoDRIVE and Lichtblick teleop recording. Stock practice course with moving simulated vehicle and post-stop LiDAR, separate from physics-replay capabilities.
- 2026-11-22 | Pure pursuit | Mapped comparison and November 9 diagnosis: Nov 11 adaptive 10.07 s vs classic 10.39 s vs shipped 10.04 s. One Nov 9 contact against a 5 cm simulator-only obstacle; source map labels it free. No contact with recorded occupied cells in eight solo runs, not evidence of physical safety. Earlier braking not adopted; shipped steering unchanged, passing off. Race Pursuit now examining a prior attempted pass, not rerunning the suite.
- Batch review, Race Pursuit: Car 2 November evidence and offline reproduction scripts. Three of four Nov 9 bags recorded clearance stops in the same bend; the fourth began in motion and recorded avoidance. No proven physical impacts, opponent or root cause. In run 190202's 15-scan braking interval, max error in heading change vs scan-to-map estimate was 0.68° for integrated gyro vs 22.94° for steering odometry. Map fit is not surveyed ground truth; filter lag changes with timestamp pairing. Nov 11 run 152132 reproduced a 30.496 s right-flank clearance stop on the verified 151026 rebuild. Development/replay targets only, not held-out evidence or permission to loosen clearance. Reuse with Sim Fidelity at its next handoff; do not redefine locked splits or interrupt its benchmark.
- Latest verified Gym native Lichtblick bundle includes loopback configuration, launch instructions and verification; recording. Only built-in oval on x86-64; prior recording and files already sent.
- Head-to-head comparison: all 16 solo-opponent simulated races detected the opponent; zero leads, completed passes or recorded contacts. Wide course: planner on made five attempts across four runs and all timed out; planner off finished both moving-blocker runs, not the stalled-opponent runs. Narrow course: no attempts. Both arms used delay LQR because the planner requires it; **not a test of pure-pursuit overtaking** and separate from the bounded-resampling patch. Passing remains off by default; no hardware changes. Report (private link) ; pair records and traces (private link)
- Rejected wheel-speed filter, distinct from PR #95. In 104 held-out simulated Levine pairs, contacts 33→33 and stalls 34→17 hide six pass-to-contact regressions (all six passes with documented car 2 scanner geometry). No pass-to-stall. Preserved 3,155 sampled stop commands at identical states but changed trajectories. No code, limits or settings published. Verdict (private link) ; evidence (private link)
- Interactive Levine replay: offline simulation diagnostic, separate from five real logs and not part of the player PR. Opens on the 1.625 s speed fork. Controls, all displayed fields and ray transforms checked against archived traces; 0 external resource requests, no controller rerun. HTML (private link) ; recording (private link) ; checks (private link) and (private link) . Screenshots of split and terminal view on Run Visualizer.
- Car 1 second-date inventory: no qualifying independent day in permitted sources searched. Nov 11 excerpts lacked original wheel timestamps; 1/127 split-scan windows and 0/63 wall windows passed. Oct 30 remained excluded. No second-day estimate, model change or new physical run. If an existing independent-date bag becomes available, use the frozen checks on raw timestamped scans/wheels with the odometry gain and a distance reference. Inventory (private link)
- Car 1, Oct 26: over 260 qualifying nearly straight windows (26 s across five runs; not whole runs), wheels counted 48.52 m vs 45.91 m from scan matching, a 5.7% overread, speed RMSE 0.123 m/s. Same-day evidence with shared LiDAR systematic errors, not a universal calibration; keep wheel scale unchanged pending another date. Report (private link) ; reproduction (private link) . Independent replay of both Levine static-sign and closed-loop-contact results: (private link)
- Learned steering, local only: 6/9 valid solo starts finished (3 stalls, no contact, 3 margin entries), both frozen neural controls 3/9 (6 margin entries), gap-follow 9/9 (3 margin entries). All three additional finishes were on one generated polygon; all neural versions stalled on all three October 26 starts. Nine invalid starts per driver excluded from headline, classified separately. No traffic, real-car, or broader safety claim. Report (private link) ; evidence (private link)
- Race Pursuit's local bounded resampler fixed the x86 returned-path overshoot (5.250001622 m to 5.250000000 m) without relaxing the assertion; 1,361 passed / 1 skipped / 0 failed in the package. Single pass relative to reference only, no wall-clearance, multiple-pass, driving or overtaking guarantee. Not on published branches. Code (private link) ; docs (private link) ; evidence (private link)
- Sim Fidelity traced the Levine offset to car 2's measured 0.26 m rear-axle-to-LiDAR distance being applied from the collision-body center. Expected +0.08 m from center; simulated +0.26 m. Car 1 is assumed, not measured. Cross-check shared with Gap Sprint; geometry evidence only, not clearance to change a controller or run the car. https://github.com/sfu-racerbot/Racerbot-Car-Workspace/blob/691b427f4316d0e64490cca1ba9d4c3fd476b8de/docs/hardware-reference.md#L17-L31 and https://github.com/f1tenth/f1tenth_gym/blob/bdaec1420c3b0f103858d289866d0d4e2e597c30/f1tenth_gym/envs/simulator.py#L361-L394
- Levine one-seed simulation: substituting only the measured speed at 1.625 s (3.959 vs 4.162 m/s) changed the first steering correction and turned a 90 s timeout into two contact-free laps in 31.95 s. Separately, re-expressing the simulated LiDAR origin changed clearance sign at 20/26 stopped end poses; one closed-loop counterexample contacted a wall. No release change, hardware claim, or safe fix. Run Visualizer has both traces; Sim Fidelity has the frame question. Report (private link) ; replay (private link)
- Neural failure trace: both versions stalled after ~5.8 m on the October solo course when right clearance crossed the unchanged 20 cm margin. Replay confirmed all 24 previous results and added 12 more stalls on two other maps; gap-follow controls lapped both. No untouched labeled real day found. Spielberg starts already inside the margin. Report (private link) ; evidence (private link)
- Sim Fidelity did not change steering linkage gain: rb2 October holdouts (6 + 6 drives) showed only 0.3% and 4.0% yaw RMSE reductions against a 10% gate. No independent road-wheel-angle measurement. Replay: (private link)
- Race Pursuit's https://github.com/sfu-racerbot/Racerbot-Car-Workspace/compare/main...feature/adaptive-pursuit-racing and https://github.com/sfu-racerbot/racerbot-docs/compare/main...docs/adaptive-pursuit-simulation: PRs held for attribution. Simulated wide track 4 m/s: 14.03s vs frozen pure pursuit 16.74s but shipped controller 13.95s; masking detection 0/4 to 4/4, no paired-race passes, some faster Spielberg contacts. 1,048 sim and 188 focused Jazzy tests passed, no hardware. Trust-region post-resampling fix underway. Report (private link) and measurements (private link)
- Visualizer batch deliverables: (private link) ; (private link) ; (private link)
- Gap Sprint's five subsequent tighter-track variants were rejected; faster lap times regressed previously passing cases (rb1: 7 new contacts, rb2: 6 contacts and 5 stalls). Preset unchanged. Report: (private link) ; paired evidence: (private link)
- Neural Racer corrected scorecard for the batch: (private link)
- Docs onboarding v2 only: (private link) (checksum matched validation); (private link) ; (private link) . Guide independently exercises native Humble and a five-package read-only Jazzy simulator (50 scans, no drive publishers or pose shift). Both safety packages intact; full recursive build still fails on duplicate names, Humble course container rate limited. No PR pending supported attribution, no hardware or deployment. V1 withdrawn.

## Club-only checkpoint commits

Repository: https://github.com/sfu-racerbot/Racerbot-Car-Workspace. GitHub commit author, committer and entire message checked; all below are club-only. Isolated branch checkpoints, not merged releases.

| Checkpoint | Commit |
|---|---|
| `native_gym_viewer` | `329a93338fe6a888fd0ea1b5c367a0cf8d57130d` |
| `gap_follow_diagnostics_only` | `2cf2a9a0471ea961889c01026d56fe50dcba0018` |
| `gazebo_tools` | `fa068d7fb71cb54eddf1af94972be14479f53b7d` |
| `mujoco_tools` | `8f0ec8f643893cdbd37417394a013cb18dcec813` |
| `calibrated_ros_bridge` | `15678b2ea92ef525eaa2f7064ddd43adc8a153df` |
| `car2_measured_lidar_mount` | `ed0435827be542d4e2489e3ef1ec54e14d6a9f5b` |
| `car2_mount_docs_separate_repo` | https://github.com/sfu-racerbot/racerbot-docs/commit/e7efab3478311bbbd1aa7b4d94c99eff3da7fea6 |
| `car2_cornering_docs_separate_repo` | https://github.com/sfu-racerbot/racerbot-docs/commit/b99906475a9db133beb0e69e8cde95cdd60c528e |
| `neural_orderly_shutdown` | `8afc915b71e597f835c448b5170ae5d52ad26415` |
| `neural_mapped_driver_checks` | `cc27237761dbb61ad535b879714177598accf3bd` |
| `neural_november_replay` | `8b98d84da1d26ca5e240a5d75095c5c482bb17d6` |
| `pursuit_abort_following_experimental` | `9c79042b93dfd9d4cae064ecbe1a5dd4a34fb7c4` |
| `pursuit_abort_following_docs_separate_repo` | https://github.com/sfu-racerbot/racerbot-docs/commit/6acc3551cbc30a5076155c6a1db1aff4e9df6e8f |
| `autodrive_probes` | `3cfec3836cb8e7021501cff7f0e5019613d054f6` |
| `neural_october_diagnosis_script` | `d2da79d4c5189098a1e26c35e62bc5312ff1d3cc` |
| `pursuit_november_car2_replay` | `fff8c1e842261c3a8de8c8bb3504dacbbf478758` |
| `pursuit_bounded_raceline` | `f5355ae935f28551fe9e5e94e3ed3d3bbf65bfda` |
| `pursuit_bounded_raceline_docs_separate_repo` | https://github.com/sfu-racerbot/racerbot-docs/commit/a9e86d9d2991e443c8213d504829758db5b2cb20 |
| `neural_rl_frozen_parity` | `f3b36280559dfd664a31c3f09f46dfd35d76e49e` |
