"""Assemble stock Gazebo assets for an offline experiment (never writes into a run)."""

import argparse
import subprocess
import tempfile
from pathlib import Path
from xml.etree import ElementTree as ET

import yaml
from ament_index_python.packages import get_package_share_directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new world SDF, outside the workspace")
    parser.add_argument("--map-start", type=Path,
                        help="start_pose.yaml for the matching map2gazebo model://map")
    args = parser.parse_args()
    pkg = Path(get_package_share_directory("f1tenth_testbed"))
    vendor = Path(get_package_share_directory("gz_sim_vendor")).parents[1]
    floor = vendor / "opt/gz_sim_vendor/share/gz/gz-sim8/worlds/empty.sdf"
    sdf = ET.fromstring(subprocess.check_output(["gz", "sdf", "-p", str(floor)]))
    world = sdf.find("world")
    world.set("name", "simple_world")
    urdf = subprocess.check_output(["xacro", str(pkg / "urdf/f1tenth_testbed.urdf.xacro")])
    with tempfile.NamedTemporaryFile(suffix=".urdf") as f:
        f.write(urdf)
        f.flush()
        car = ET.fromstring(subprocess.check_output(["gz", "sdf", "-p", f.name])).find("model")
    x, y, yaw = 0, 0, 0
    if args.map_start:
        pose = yaml.safe_load(args.map_start.read_text())
        x, y, yaw = (float(pose[k]) for k in ("x", "y", "yaw"))
        ET.SubElement(ET.SubElement(world, "include"), "uri").text = "model://map"
    ET.SubElement(car, "pose").text = f"{x} {y} 0.08 0 0 {yaw}"
    world.append(car)
    with args.output.open("xb") as out:
        ET.ElementTree(sdf).write(out, encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    main()
