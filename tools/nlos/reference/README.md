# NLOS reference prototype

Reference only. Slow on purpose (per-beam Python loops). Do not import from `src/`; tests may import it.

- `nlos_proto.py`: L-corner scene, one-bounce specular scan tracer, RANSAC panel fit, unfolding, panel-frame background grid, clustering.
- `run1.py W FOOTPRINT_K`: detection rate and surface error over a grid of hidden positions.
- `run2.py W FOOTPRINT_K SPEED [MAX_GHOST_PATH]`: ego travel gained over line of sight for a stopped car.
- `fitstats.py`: panel line-fit error statistics.
- `fa_kf.py`, `fa2.py`: false alarms on empty approaches against background coverage; Kalman filter error statistics.
- `results/`: raw outputs quoted in section 2.5 of the plan.

Run with any Python that has NumPy and SciPy, from this directory.
