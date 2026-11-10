"""race_line.py's CLI with car physics overridden from the environment (experiments only).

SPEED_DELAY=<s>, A_BRAKE=<m/s^2>, A_LAT=<m/s^2>: replace those CarPhysics fields after the car's config loads.
"""
import os
import sys
from dataclasses import replace

from pure_pursuit import race_line

orig = race_line.RaceLineConfig.from_params.__func__


def from_params(cls, values):
    cfg = orig(cls, values)
    phys = cfg.physics
    over = {}
    if os.environ.get("SPEED_DELAY"):
        over["speed_delay"] = float(os.environ["SPEED_DELAY"])
    if os.environ.get("A_BRAKE"):
        over["a_brake"] = float(os.environ["A_BRAKE"])
    if os.environ.get("A_LAT"):
        over["a_lat_left"] = over["a_lat_right"] = float(os.environ["A_LAT"])
    if os.environ.get("STEER_PLAN"):
        over["max_steering_angle"] = float(os.environ["STEER_PLAN"])
    if over:
        cfg = replace(cfg, physics=replace(phys, **over))
    top = {}
    if os.environ.get("BODY_FLOOR"):
        top["body_floor"] = float(os.environ["BODY_FLOOR"])
    if os.environ.get("BODY_TIGHT"):
        top["body_floor_tight"] = float(os.environ["BODY_TIGHT"])
    if os.environ.get("SPEED_FLOOR_GAIN"):
        top["speed_floor_gain"] = float(os.environ["SPEED_FLOOR_GAIN"])
    if os.environ.get("SPEED_FLOOR_FROM"):
        top["speed_floor_from"] = float(os.environ["SPEED_FLOOR_FROM"])
    if os.environ.get("WALL_MARGIN"):
        top["wall_margin"] = float(os.environ["WALL_MARGIN"])
    if os.environ.get("CONE_MARGIN"):
        top["cone_margin"] = float(os.environ["CONE_MARGIN"])
    if top:
        cfg = replace(cfg, **top)
    print("override", over, top, file=sys.stderr)
    return cfg


race_line.RaceLineConfig.from_params = classmethod(from_params)

if __name__ == "__main__":
    sys.exit(race_line.main())
