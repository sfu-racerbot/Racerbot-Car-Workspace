import hashlib
from pathlib import Path

import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def car_limits(car_id=None):
    """This car's wheelbase and autonomy steering cap, from the same layered
    pure_pursuit.yaml the workspace's own pure_pursuit_node uses (shared
    config + src/racerbot_car/cars/<id>/calibration/pure_pursuit.yaml).
    """
    from racerbot_car.layers import effective_config, launch_car_id

    car_id = car_id or launch_car_id()
    values = effective_config("pure_pursuit.yaml", "pure_pursuit_node", car_id)
    return car_id, float(values["wheelbase"]), float(values["max_steering_angle"])


def laser_offset(car_id):
    """Where the LiDAR sits ahead of the rear axle on this car (m, x and y): the
    particle filter's pose is the LiDAR's, and pure pursuit's geometry is about
    the rear axle. Same values as the workspace's pure_pursuit.yaml."""
    from racerbot_car.layers import effective_config

    values = effective_config("pure_pursuit.yaml", "pure_pursuit_node", car_id)
    return float(values["laser_offset_x"]), float(values["laser_offset_y"])


def check_steering(max_steering, car_cap, car_id):
    """Refuse a steering limit above the car's calibrated autonomy cap."""
    if not 0.0 < max_steering <= car_cap:
        raise RuntimeError(
            f"max_steering:={max_steering} is outside (0, {car_cap}] rad, car {car_id}'s "
            f"autonomy cap (src/racerbot_car/cars/{car_id}/calibration/pure_pursuit.yaml). "
            "Refusing to start."
        )


def line_roots(car_id):
    """Where saved racing lines live for this car: the simulator's drawn lines
    (RViz panel) and mapping runs (map_course / auto_map_race)."""
    from racerbot_car.paths import data_dir

    return [data_dir("racerbot_sim", car_id) / "drawn_lines", data_dir("racerbot_auto", car_id)]


def latest_line(roots):
    """The newest <folder>/raceline_profiled.csv that has its map.yaml beside it."""
    found = [csv for root in map(Path, roots) if root.is_dir()
             for csv in root.glob("*/raceline_profiled.csv") if (csv.parent / "map.yaml").is_file()]
    if not found:
        raise RuntimeError(
            "track:=latest found no saved line (a folder with raceline_profiled.csv and "
            f"map.yaml) under {', '.join(map(str, roots))}. Refusing to start.")
    return max(found, key=lambda csv: csv.stat().st_mtime)


def map_identity(map_yaml):
    """What makes two map.yaml files the same map: the image's bytes, origin and resolution."""
    map_yaml = Path(map_yaml)
    meta = yaml.safe_load(map_yaml.read_text())
    image = Path(meta["image"])
    image = image if image.is_absolute() else map_yaml.parent / image
    return (hashlib.sha256(image.read_bytes()).hexdigest(),
            [round(float(v), 6) for v in meta["origin"]], round(float(meta["resolution"]), 9))


def check_same_map(line_map, localize_map):
    """Refuse a line saved on a different map from the one localization uses."""
    if map_identity(line_map) != map_identity(localize_map):
        raise RuntimeError(
            f"The line's map {line_map} is not the map localization uses ({localize_map}): "
            "its waypoints would be in the wrong place. Refusing to start.")


def generate_launch_description():
    car_id, wheelbase, car_cap = car_limits()

    odom_topic_arg = DeclareLaunchArgument(
        "odom_topic",
        default_value="/odom",
        description="Odometry topic (pose source in sim, wheel odom for speed when use_pf is true)",
    )

    use_pf_arg = DeclareLaunchArgument(
        "use_pf",
        default_value="true",
        description="true = pose from particle filter /pf/pose/odom (irl), false = pose from odom_topic (sim)",
    )

    enable_safety_arg = DeclareLaunchArgument(
        "enable_safety",
        default_value="true",
        description="Whether racerbot_a's safety_node (/safety/brake) gates this node. "
        "The node's own LB deadman stays on either way",
    )

    launch_safety_arg = DeclareLaunchArgument(
        "launch_safety",
        default_value="true",
        description="Also start racerbot_a's safety_node here, reading odom_topic for its iTTC speed",
    )

    enable_ttc_arg = DeclareLaunchArgument(
        "enable_ttc",
        default_value="true",
        description="Whether racerbot_a's safety_node also brakes on iTTC",
    )

    safety_timeout_arg = DeclareLaunchArgument(
        "safety_message_timeout_ms",
        default_value="50.0",
        description="safety_node's /joy timeout in ms (racerbot_a's default)",
    )

    track_arg = DeclareLaunchArgument(
        "track",
        default_value="",
        description="Name of the waypoint CSV (without .csv) in the waypoints folder, or "
        "'latest' for the newest saved line (drawn in the simulator, or a mapping run)",
    )

    map_yaml_arg = DeclareLaunchArgument(
        "map_yaml",
        default_value="",
        description="With track:=latest: the map.yaml localization runs on. The line is "
        "refused if it was saved on a different map",
    )

    waypoints_file_arg = DeclareLaunchArgument(
        "waypoints_file",
        default_value="",
        description="Absolute path to a waypoint CSV (x,y first), e.g. a mapping run's "
        "raceline_profiled.csv. Overrides track",
    )

    lookahead_base_arg = DeclareLaunchArgument(
        "lookahead_base",
        default_value="1.0",
        description="Base lookahead distance in meters",
    )

    lookahead_min_arg = DeclareLaunchArgument(
        "lookahead_min",
        default_value="0.8",
        description="Minimum lookahead distance in meters",
    )

    lookahead_max_arg = DeclareLaunchArgument(
        "lookahead_max",
        default_value="3.0",
        description="Maximum lookahead distance in meters",
    )

    lookahead_speed_gain_arg = DeclareLaunchArgument(
        "lookahead_speed_gain",
        default_value="0.1",
        description="Speed gain applied to lookahead distance",
    )

    max_steering_arg = DeclareLaunchArgument(
        "max_steering",
        default_value=str(car_cap),
        description=f"Maximum steering angle in radians (car {car_id}'s cap is {car_cap}; higher is refused)",
    )

    max_speed_arg = DeclareLaunchArgument(
        "max_speed",
        default_value="1.0",
        description="Maximum forward speed in m/s",
    )

    speed_args = [
        DeclareLaunchArgument(
            "speed_mode", default_value="steering",
            description="steering: pure_pursuit_a's rule (max_speed straight, min_speed at full "
            "lock). profile: plan the speed along the line -- corner speed from "
            "max_lateral_accel, braking before corners, max_accel on the way out"),
        DeclareLaunchArgument("medium_angle_speed", default_value="-1.0",
                              description="speed at half of max_steering, m/s; the steering rule "
                              "runs max_speed -> this -> min_speed. -1 = halfway (a straight line)"),
        DeclareLaunchArgument("max_lateral_accel", default_value="2.5",
                              description="profile: cornering grip, m/s^2 (v <= sqrt(a/curvature))"),
        DeclareLaunchArgument("max_accel", default_value="3.0",
                              description="profile: how fast speed may rise, m/s^2"),
        DeclareLaunchArgument("max_brake", default_value="3.0",
                              description="braking before corners (profile) and to the goal, m/s^2"),
        DeclareLaunchArgument("laps", default_value="0",
                              description="loop: stop at the start/finish line after this many laps (0 = never)"),
        DeclareLaunchArgument("goal_tolerance", default_value="0.3",
                              description="stop this close to the goal: the end of an open path, or the last lap's end (m)"),
        DeclareLaunchArgument("max_path_error", default_value="1.5",
                              description="stop if the car is further than this from the line (m)"),
        DeclareLaunchArgument("max_pose_age", default_value="0.25",
                              description="stop on a pose older than this (s), or unstamped"),
        DeclareLaunchArgument("compensate_latency", default_value="false",
                              description="aim from the pose predicted at now + actuation_delay"),
        DeclareLaunchArgument("actuation_delay", default_value="0.0",
                              description="command-to-wheels delay (s), 0 to 0.5"),
        DeclareLaunchArgument("max_steering_rate", default_value="1.0",
                              description="steering slew limit (rad/s), both directions"),
        DeclareLaunchArgument("lookahead_gain_straight", default_value="-1.0",
                              description="lookahead speed gain on straights (m per m/s); -1 = lookahead_speed_gain"),
        DeclareLaunchArgument("lookahead_gain_corner", default_value="-1.0",
                              description="lookahead speed gain in corners (m per m/s); -1 = lookahead_speed_gain"),
        DeclareLaunchArgument("lookahead_corner_curvature", default_value="0.67",
                              description="curvature (1/m) that counts as a full corner"),
        DeclareLaunchArgument("lookahead_preview", default_value="2.0",
                              description="how far ahead (m) to look for a corner"),
    ]

    min_speed_arg = DeclareLaunchArgument(
        "min_speed",
        default_value="0.5",
        description="Minimum forward speed in m/s",
    )

    def pure_pursuit(context):
        max_steering = float(LaunchConfiguration("max_steering").perform(context))
        check_steering(max_steering, car_cap, car_id)
        use_pf = LaunchConfiguration("use_pf").perform(context).lower() in ("true", "1", "yes")
        offset = laser_offset(car_id) if use_pf else (0.0, 0.0)
        track = LaunchConfiguration("track").perform(context)
        waypoints_file = LaunchConfiguration("waypoints_file").perform(context)
        localize_map = LaunchConfiguration("map_yaml").perform(context)
        notes = []
        if track == "latest":
            if waypoints_file:
                raise RuntimeError("Give track:=latest or waypoints_file, not both.")
            csv = latest_line(line_roots(car_id))
            line_map = csv.parent / "map.yaml"
            if localize_map:
                check_same_map(line_map, localize_map)
                checked = f"same map as {localize_map}"
            else:
                checked = "NOT checked against localization's map: pass map_yaml:= to check"
            notes.append(LogInfo(msg=f"track:=latest -> {csv} (its map: {line_map}; {checked})"))
            track, waypoints_file = "", str(csv)
        return notes + [
            Node(
                package="pure_pursuit_tech",
                executable="pure_pursuit_tech_node",
                name="pure_pursuit_tech_node",
                output="screen",
                parameters=[
                    {
                        "odom_topic": LaunchConfiguration("odom_topic"),
                        "use_pf": ParameterValue(LaunchConfiguration("use_pf"), value_type=bool),
                        "enable_safety": ParameterValue(
                            LaunchConfiguration("enable_safety"), value_type=bool
                        ),
                        "track": track,
                        "waypoints_file": waypoints_file,
                        "lookahead_base": ParameterValue(
                            LaunchConfiguration("lookahead_base"), value_type=float
                        ),
                        "lookahead_min": ParameterValue(
                            LaunchConfiguration("lookahead_min"), value_type=float
                        ),
                        "lookahead_max": ParameterValue(
                            LaunchConfiguration("lookahead_max"), value_type=float
                        ),
                        "lookahead_speed_gain": ParameterValue(
                            LaunchConfiguration("lookahead_speed_gain"), value_type=float
                        ),
                        "wheelbase": wheelbase,
                        # the PF's pose is the LiDAR's: shift it to the rear axle
                        "laser_offset_x": offset[0],
                        "laser_offset_y": offset[1],
                        "speed_mode": LaunchConfiguration("speed_mode"),
                        "laps": ParameterValue(LaunchConfiguration("laps"), value_type=int),
                        **{name: ParameterValue(LaunchConfiguration(name), value_type=float)
                           for name in ("medium_angle_speed", "max_lateral_accel", "max_accel", "max_brake",
                                        "goal_tolerance", "max_path_error", "max_pose_age",
                                        "actuation_delay", "max_steering_rate", "lookahead_gain_straight",
                                        "lookahead_gain_corner", "lookahead_corner_curvature", "lookahead_preview")},
                        "compensate_latency": ParameterValue(
                            LaunchConfiguration("compensate_latency"), value_type=bool
                        ),
                        "max_steering": max_steering,
                        "max_speed": ParameterValue(
                            LaunchConfiguration("max_speed"), value_type=float
                        ),
                        "min_speed": ParameterValue(
                            LaunchConfiguration("min_speed"), value_type=float
                        ),
                    }
                ],
            )
        ]

    # racerbot_a's safety_node. Its odometry input is the simulator's
    # /ego_racecar/odom, so it is remapped to odom_topic here: unremapped on
    # the car it would read speed 0 and its iTTC brake could never fire.
    safety_node = Node(
        package="safety_node",
        executable="safety_node",
        name="safety_node",
        output="screen",
        condition=IfCondition(LaunchConfiguration("launch_safety")),
        parameters=[
            {
                "deadman_button": 4,
                "message_timeout_duration": ParameterValue(
                    LaunchConfiguration("safety_message_timeout_ms"), value_type=float
                ),
                "enable_ttc": ParameterValue(LaunchConfiguration("enable_ttc"), value_type=bool),
            }
        ],
        remappings=[("/ego_racecar/odom", LaunchConfiguration("odom_topic"))],
    )

    return LaunchDescription(
        [
            odom_topic_arg,
            use_pf_arg,
            enable_safety_arg,
            launch_safety_arg,
            enable_ttc_arg,
            safety_timeout_arg,
            track_arg,
            map_yaml_arg,
            waypoints_file_arg,
            lookahead_base_arg,
            lookahead_min_arg,
            lookahead_max_arg,
            lookahead_speed_gain_arg,
            max_steering_arg,
            max_speed_arg,
            min_speed_arg,
            *speed_args,
            safety_node,
            # Launch pure pursuit
            OpaqueFunction(function=pure_pursuit),
        ]
    )
