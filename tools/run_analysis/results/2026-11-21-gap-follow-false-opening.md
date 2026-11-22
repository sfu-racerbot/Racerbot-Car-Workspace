# Gap follow chose the true exit in four fixed-pose simulation checks

At workspace revision `7b2003f5db5b627e4326ca3423902dce5201778d`, both
time-trial profiles aimed through the true exit in left and right bends.
Each profile and direction was checked with and without a synthetic sensor
fault. The chosen aim and the steering and speed commands were unchanged
within each pair. All 40 local controller-mirror tests passed, including
the four paired checks.

The false opening did not survive filtering as a distinct candidate. This
checks a decision at one pose, not the ability to reject a competing gap
that remains selectable. It does not measure a trajectory, contact, or lap
time, explain the separate Spielberg result, or establish what a physical
sensor would see.
