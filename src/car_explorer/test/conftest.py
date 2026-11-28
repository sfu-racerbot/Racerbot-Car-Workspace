"""Make car_explorer importable without a colcon build (its logic needs no ROS)."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
