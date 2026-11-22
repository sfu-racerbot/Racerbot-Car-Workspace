"""Assemble stock Gazebo assets for an offline experiment (never writes into a run)."""

import argparse
import math
import subprocess
import tempfile
from pathlib import Path
from xml.etree import ElementTree as ET

import yaml


def assemble(sdf, car, map_start):
    world = sdf.find("world")
    world.set("name", "simple_world")
    x, y, yaw = 0, 0, 0
    if map_start is not None:
        try:
            pose = yaml.safe_load(map_start.read_text(encoding="utf-8"))
            x, y, yaw = (float(pose[k]) for k in ("x", "y", "yaw"))
        except (ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
            raise ValueError("expected numeric x, y and yaw in map start pose") from exc
        if not all(math.isfinite(v) for v in (x, y, yaw)):
            raise ValueError("nonfinite map start pose")
        ET.SubElement(ET.SubElement(world, "include"), "uri").text = "model://map"
    ET.SubElement(car, "pose").text = f"{x} {y} 0.08 0 0 {yaw}"
    world.append(car)


def main():
    from ament_index_python.packages import get_package_share_directory

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new world SDF, outside the workspace")
    parser.add_argument("--map-start", type=Path,
                        help="start_pose.yaml for the matching map2gazebo model://map")
    args = parser.parse_args()
    pkg = Path(get_package_share_directory("f1tenth_testbed"))
    vendor = Path(get_package_share_directory("gz_sim_vendor")).parents[1]
    floor = vendor / "opt/gz_sim_vendor/share/gz/gz-sim8/worlds/empty.sdf"
    sdf = ET.fromstring(subprocess.check_output(["gz", "sdf", "-p", str(floor)]))
    urdf = subprocess.check_output(["xacro", str(pkg / "urdf/f1tenth_testbed.urdf.xacro")])
    with tempfile.NamedTemporaryFile(suffix=".urdf") as f:
        f.write(urdf)
        f.flush()
        car = ET.fromstring(subprocess.check_output(["gz", "sdf", "-p", f.name])).find("model")
    assemble(sdf, car, args.map_start)
    with args.output.open("xb") as out:
        ET.ElementTree(sdf).write(out, encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    main()
