# Waypoints

Put a track's CSV here as `<track>.csv` and select it with `track:=<track>`
(no `.csv`). Rows are `x,y[,anything else]` in the map frame, metres; lines
starting with `#` and non-numeric rows (an `x,y,speed` header) are skipped.

racerbot_a's CSVs (`src/racerbot_a/src/pure_pursuit_a/waypoints/`) use the same
format and can be copied here as-is. For a track mapped on this car, skip the
copy and point `waypoints_file:=` at the run's `raceline_profiled.csv` instead.
