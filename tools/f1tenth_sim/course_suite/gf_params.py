"""Write a stand-in run folder holding car 2's CURRENT layered gap_follow parameters, so
validate_calibration.py sim --params-from <dir> --launched-code drives today's shipped gap_follow."""
import sys
from pathlib import Path

import yaml
from racerbot_car.layers import effective_config

out = Path(sys.argv[1])
car = sys.argv[2] if len(sys.argv) > 2 else "rb2"
p = effective_config("gap_follow.yaml", "gap_follow_node", car_id=car)
(out / "metadata" / "params").mkdir(parents=True, exist_ok=True)
(out / "metadata" / "params" / "gap_follow_node.yaml").write_text(
    yaml.safe_dump({"gap_follow_node": {"ros__parameters": p}}))
print(out, "max_speed", p.get("max_speed"), "max_steering_angle", p.get("max_steering_angle"),
      "max_lateral_accel", p.get("max_lateral_accel"), "aim_depth_bias", p.get("aim_depth_bias"),
      "opponent", p.get("enable_opponent_awareness"))
