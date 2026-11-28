"""Start the explorer beside an already running gap_follow_launch.py.

    ros2 launch car_explorer explore_launch.py

It changes gap_follow's max_speed and max_lateral_accel between stretches of
clean driving; it never drives. docs/explorer.md.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    config = os.path.join(get_package_share_directory('car_explorer'), 'config',
                          'car_explorer.yaml')
    margin = DeclareLaunchArgument(
        'wall_margin', default_value='0.03',
        description='m left after a predicted stop before the explorer backs off')
    return LaunchDescription([
        margin,
        Node(package='car_explorer', executable='explorer_node', name='car_explorer',
             output='screen',
             parameters=[config, {'wall_margin': LaunchConfiguration('wall_margin')}]),
    ])
