# Speed-plan sweep, 2026-11-13 (Task 8 evidence; no calibration changed)

Floor `sfu_course_2026_11_11_rb2` (grip scale 1.15), map racer (`delay_lqr`, rb2 race-day config, map safety, online lateral cap at the shipped 3.1), 3 laps from a standing start, 3 seeds. Raw lines are the gap_follow laps of the earlier course suite (`/tmp/cmp/base`, made on the 11-06 floor), reused for every arm. Planner arms (lateral left = right / speed delay / grip coupling): C0 3.2 / 0.22 / 0, P1 2.95 / 0.22 / 0, P2 2.95 / 0.22 / 0.5, P3 2.95 / 0.22 / 1, P4 3.2 / 0.22 / 1, P5 2.95 / 0.30 / 1, P6 2.80 / 0.22 / 0.5. Lines are built with the damped-and-projected coupled profile of PR #55 (`gg-profile` 8264d68; arms with c > 0 were rebuilt after the first solver proved too conservative). Driver: `tools/f1tenth_sim/course_suite/plan_sweep.py`, `plan_build.py`.

**Verdict: no arm is eligible.** The profile-alone condition (cap off, race-day course, standing starts from 5 poses and 3 flying laps) fails for every arm including the control: all arms touch within 8 to 10 m of the first standing start. The ranking rule therefore selects nothing; the table below is the evidence. Mean flying-lap ratio to C0 (7 courses; mini_Oschersleben only exists for C0 and P1): P1 1.0005, P2 1.0062, P6 1.0086, P4 1.0182, P3 1.0256, P5 1.0327. P3 also has 2 contact and 2 unfinished runs (10_30).

Not done / owed: mini_Oschersleben for P2 to P6 (the coupled builds ran over an hour each, so they were stopped; C0 and P1 are in the table). The understeer-curve variant was run for C0 and P2 only because no arm ranked (not the rule's "two best"); it fails in this simulator (13 and 17 contact runs): the node's compensation asks for up to 3.75 m/s^2 while this floor's tyres (2.94 x 1.15 peak, falling) deliver less past the peak, and the math-level mirror of the node's compensation (`start_replay.understeer_compensation`) is not validated against the node tick for tick. Treat it as inconclusive.

Failed-pose standing start (s 0.00, 0.134 m right, +14.61 deg) with the cap on: C0 and P1 do not touch (right@14 about 0.22 m); P3, P4, P5 touch in 3/3 seeds within 2 m; P2 and P6 in 1 to 2 of 3 (c > 0 lines start differently). Real launch: 0.40 m right at s = 14 (cap not binding).

| arm | course | line | est lap s | median flying lap s | contact seeds | unfinished seeds | online cap binding | reactive ticks |
|---|---|---|---|---|---|---|---|---|
| C0 | car2_course_2026_11_11 | min_time | 9.40 | 9.96 (6) | 0/3 | 0/3 | 0.715 | 0 |
| C0 | car2_course_2026_11_06a_scans | min_time | 10.44 | 10.73 (6) | 0/3 | 0/3 | 0.603 | 0 |
| C0 | car2_course_2026_11_06b_scans | min_time | 7.98 | 8.23 (6) | 0/3 | 0/3 | 0.607 | 12 |
| C0 | course_2026_10_30_scans | min_time | 13.48 | 14.25 (6) | 0/3 | 0/3 | 0.635 | 0 |
| C0 | indoor_tight | min_time | 7.65 | 7.93 (6) | 0/3 | 0/3 | 0.648 | 0 |
| C0 | indoor_oval | min_time | 7.81 | 8.34 (6) | 0/3 | 0/3 | 0.709 | 0 |
| C0 | levine_blocked | min_time | 14.88 | 14.94 (6) | 0/3 | 0/3 | 0.387 | 0 |
| C0 | mini_Oschersleben | min_time | 39.60 | 41.71 (6) | 0/3 | 0/3 | 0.523 | 0 |
| P1 | car2_course_2026_11_11 | min_time | 9.77 | 10.05 (6) | 0/3 | 0/3 | 0.623 | 0 |
| P1 | car2_course_2026_11_06a_scans | min_time | 10.76 | 10.74 (6) | 0/3 | 0/3 | 0.533 | 0 |
| P1 | car2_course_2026_11_06b_scans | min_time | 8.18 | 8.15 (6) | 0/3 | 0/3 | 0.563 | 0 |
| P1 | course_2026_10_30_scans | min_time | 13.81 | 14.35 (6) | 0/3 | 0/3 | 0.586 | 0 |
| P1 | indoor_tight | min_time | 7.91 | 7.91 (6) | 0/3 | 0/3 | 0.565 | 0 |
| P1 | indoor_oval | min_time | 8.26 | 8.31 (6) | 0/3 | 0/3 | 0.582 | 0 |
| P1 | levine_blocked | min_time | 15.15 | 14.92 (6) | 0/3 | 0/3 | 0.360 | 0 |
| P1 | mini_Oschersleben | min_time | 40.36 | 41.76 (6) | 0/3 | 0/3 | 0.498 | 0 |
| P2 | car2_course_2026_11_11 | min_time | 9.87 | 10.18 (6) | 0/3 | 0/3 | 0.639 | 0 |
| P2 | car2_course_2026_11_06a_scans | min_time | 10.78 | 10.75 (6) | 0/3 | 0/3 | 0.520 | 0 |
| P2 | car2_course_2026_11_06b_scans | min_time | 8.40 | 8.21 (6) | 0/3 | 0/3 | 0.528 | 0 |
| P2 | course_2026_10_30_scans | min_time | 13.93 | 14.38 (6) | 0/3 | 0/3 | 0.581 | 0 |
| P2 | indoor_tight | min_time | 8.02 | 7.99 (6) | 0/3 | 0/3 | 0.529 | 0 |
| P2 | indoor_oval | min_time | 7.98 | 8.32 (6) | 0/3 | 0/3 | 0.640 | 0 |
| P2 | levine_blocked | min_time | 15.37 | 15.03 (6) | 0/3 | 0/3 | 0.335 | 0 |
| P2 | mini_Oschersleben | not built | | | | | | |
| P3 | car2_course_2026_11_11 | min_time | 10.32 | 10.23 (6) | 0/3 | 0/3 | 0.462 | 0 |
| P3 | car2_course_2026_11_06a_scans | min_time | 11.42 | 11.11 (6) | 0/3 | 0/3 | 0.379 | 0 |
| P3 | car2_course_2026_11_06b_scans | min_time | 8.87 | 8.39 (6) | 0/3 | 0/3 | 0.423 | 0 |
| P3 | course_2026_10_30_scans | min_time | 14.60 | 14.53 (2) | 2/3 | 2/3 | 0.427 | 0 |
| P3 | indoor_tight | min_time | 8.45 | 8.23 (6) | 0/3 | 0/3 | 0.368 | 0 |
| P3 | indoor_oval | min_time | 8.79 | 8.43 (6) | 0/3 | 0/3 | 0.334 | 0 |
| P3 | levine_blocked | min_time | 15.88 | 15.38 (6) | 0/3 | 0/3 | 0.297 | 0 |
| P3 | mini_Oschersleben | not built | | | | | | |
| P4 | car2_course_2026_11_11 | min_time | 9.96 | 10.18 (6) | 0/3 | 0/3 | 0.609 | 0 |
| P4 | car2_course_2026_11_06a_scans | min_time | 11.01 | 10.89 (6) | 0/3 | 0/3 | 0.441 | 0 |
| P4 | car2_course_2026_11_06b_scans | min_time | 8.53 | 8.36 (6) | 0/3 | 0/3 | 0.464 | 0 |
| P4 | course_2026_10_30_scans | min_time | 14.12 | 14.39 (6) | 0/3 | 0/3 | 0.521 | 0 |
| P4 | indoor_tight | min_time | 8.18 | 8.23 (6) | 0/3 | 0/3 | 0.491 | 0 |
| P4 | indoor_oval | min_time | 8.52 | 8.41 (6) | 0/3 | 0/3 | 0.456 | 0 |
| P4 | levine_blocked | min_time | 15.54 | 15.20 (6) | 0/3 | 0/3 | 0.304 | 0 |
| P4 | mini_Oschersleben | not built | | | | | | |
| P5 | car2_course_2026_11_11 | min_time | 10.53 | 10.28 (6) | 0/3 | 0/3 | 0.410 | 0 |
| P5 | car2_course_2026_11_06a_scans | min_time | 11.58 | 11.05 (6) | 0/3 | 0/3 | 0.314 | 0 |
| P5 | car2_course_2026_11_06b_scans | min_time | 9.24 | 8.51 (6) | 0/3 | 0/3 | 0.356 | 0 |
| P5 | course_2026_10_30_scans | min_time | 14.90 | 14.65 (6) | 0/3 | 0/3 | 0.368 | 0 |
| P5 | indoor_tight | min_time | 8.60 | 8.35 (6) | 0/3 | 0/3 | 0.322 | 0 |
| P5 | indoor_oval | min_time | 8.97 | 8.51 (6) | 0/3 | 0/3 | 0.281 | 0 |
| P5 | levine_blocked | min_time | 16.12 | 15.38 (6) | 0/3 | 0/3 | 0.227 | 0 |
| P5 | mini_Oschersleben | not built | | | | | | |
| P6 | car2_course_2026_11_11 | min_time | 10.11 | 10.14 (6) | 0/3 | 0/3 | 0.572 | 0 |
| P6 | car2_course_2026_11_06a_scans | min_time | 10.94 | 10.74 (6) | 0/3 | 0/3 | 0.485 | 0 |
| P6 | car2_course_2026_11_06b_scans | min_time | 8.56 | 8.26 (6) | 0/3 | 0/3 | 0.459 | 0 |
| P6 | course_2026_10_30_scans | min_time | 14.25 | 14.60 (6) | 0/3 | 0/3 | 0.561 | 20 |
| P6 | indoor_tight | min_time | 8.18 | 7.95 (6) | 0/3 | 0/3 | 0.453 | 0 |
| P6 | indoor_oval | min_time | 8.22 | 8.35 (6) | 0/3 | 0/3 | 0.524 | 0 |
| P6 | levine_blocked | min_time | 15.56 | 15.05 (6) | 0/3 | 0/3 | 0.316 | 0 |
| P6 | mini_Oschersleben | not built | | | | | | |
| C0+U | car2_course_2026_11_11 | min_time | 9.40 | nan (0) | 3/3 | 3/3 | 0.579 | 27 |
| C0+U | car2_course_2026_11_06a_scans | min_time | 10.44 | nan (0) | 3/3 | 3/3 | 0.663 | 15 |
| C0+U | car2_course_2026_11_06b_scans | min_time | 7.98 | nan (0) | 0/3 | 3/3 | 0.178 | 6931 |
| C0+U | course_2026_10_30_scans | min_time | 13.48 | nan (0) | 3/3 | 3/3 | 0.631 | 22 |
| C0+U | indoor_tight | min_time | 7.65 | 8.69 (4) | 1/3 | 1/3 | 0.559 | 135 |
| C0+U | indoor_oval | min_time | 7.81 | 8.39 (6) | 0/3 | 0/3 | 0.655 | 0 |
| C0+U | levine_blocked | min_time | 14.88 | nan (0) | 3/3 | 3/3 | 0.500 | 17 |
| C0+U | mini_Oschersleben | min_time | 39.60 | nan (0) | 0/0 | 0/0 | 0.000 | 0 |
| P2+U | car2_course_2026_11_11 | min_time | 9.87 | nan (0) | 3/3 | 3/3 | 0.552 | 17 |
| P2+U | car2_course_2026_11_06a_scans | min_time | 10.78 | nan (0) | 3/3 | 3/3 | 0.425 | 14 |
| P2+U | car2_course_2026_11_06b_scans | min_time | 8.40 | nan (0) | 3/3 | 3/3 | 0.290 | 52 |
| P2+U | course_2026_10_30_scans | min_time | 13.93 | nan (0) | 3/3 | 3/3 | 0.565 | 22 |
| P2+U | indoor_tight | min_time | 8.02 | 8.80 (1) | 3/3 | 3/3 | 0.430 | 152 |
| P2+U | indoor_oval | min_time | 7.98 | 8.30 (6) | 0/3 | 0/3 | 0.571 | 0 |
| P2+U | levine_blocked | min_time | 15.37 | 15.76 (2) | 2/3 | 2/3 | 0.432 | 6 |
| P2+U | mini_Oschersleben | not built | | | | | | |

| arm | eligible | mean flying-lap ratio to C0 | failed conditions |
|---|---|---|---|
| C0 | no | 1.0000 | profile-alone check not passed |
| P1 | no | 1.0005 | profile-alone check not passed |
| P2 | no | 1.0062 | a course C0 builds did not build; profile-alone check not passed |
| P3 | no | 1.0256 | a course C0 builds did not build; 2 contact run(s); 2 unfinished run(s); profile-alone check not passed |
| P4 | no | 1.0182 | a course C0 builds did not build; profile-alone check not passed |
| P5 | no | 1.0327 | a course C0 builds did not build; profile-alone check not passed |
| P6 | no | 1.0086 | a course C0 builds did not build; profile-alone check not passed |
| C0+U | no | nan | 13 contact run(s); 16 unfinished run(s); profile-alone check not passed |
| P2+U | no | nan | a course C0 builds did not build; 17 contact run(s); 17 unfinished run(s); profile-alone check not passed |

Ranking (eligible only, fastest first): none eligible

| arm | profile alone (race-day, cap off) | worst right / left offset, m | contacts (run, seed, s) | failed-pose start, cap on: contact seeds | right@14, m |
|---|---|---|---|---|---|
| C0 | FAIL | 1.01 / 0.45 | 18 of 18 runs; 3 flying laps after a s=0 s=7.6; s=0.0 s=7.6; s=11.2 s=15.3; s=16.7 s=20.4; s=16.7 s=20.8; | 0/3 | 0.22, 0.22, 0.23 |
| P1 | FAIL | 0.87 / 0.53 | 18 of 18 runs; 3 flying laps after a s=0 s=8.1; 3 flying laps after a s=0 s=8.3; s=0.0 s=8.1; s=0.0 s=8.3 | 0/3 | 0.22, 0.21, 0.21 |
| P2 | FAIL | 0.99 / 0.48 | 18 of 18 runs; 3 flying laps after a s=0 s=8.1; 3 flying laps after a s=0 s=8.2; s=0.0 s=8.1; s=0.0 s=8.2 | 2/3 | 0.20 |
| P3 | FAIL | 0.79 / 0.39 | 18 of 18 runs; 3 flying laps after a s=0 s=8.7; 3 flying laps after a s=0 s=8.8; 3 flying laps after a s= | 3/3 | - |
| P4 | FAIL | 0.98 / 0.42 | 18 of 18 runs; 3 flying laps after a s=0 s=8.1; 3 flying laps after a s=0 s=8.2; s=0.0 s=8.1; s=0.0 s=8.2 | 3/3 | - |
| P5 | FAIL | 0.62 / 0.42 | 18 of 18 runs; 3 flying laps after a s=0 s=9.0; 3 flying laps after a s=0 s=9.1; s=0.0 s=9.0; s=0.0 s=9.1 | 3/3 | - |
| P6 | FAIL | 0.85 / 0.46 | 18 of 18 runs; 3 flying laps after a s=0 s=8.4; 3 flying laps after a s=0 s=8.6; s=0.0 s=8.4; s=0.0 s=8.6 | 1/3 | 0.21, 0.20 |
