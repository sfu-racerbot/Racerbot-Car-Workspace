"""Process fixtures follow the command interfaces in the warm-recorder brief."""
from race_diagnostics.orphans import find_orphans


def test_matches_auto_launch_diag_snapshot_and_bag_only():
    commands = [
        'ros2 launch race_diagnostics record_run.py name_prefix:=auto_',
        '/install/race_diag_node --ros-args -r __node:=auto_race_diag_node',
        '/install/snapshot_params --ros-args -r __node:=auto_snapshot_params',
        'ros2 bag record --node-name auto_rosbag2_recorder -o /tmp/bag /scan',
        '/install/race_diag_node --ros-args -r __node:=race_diag_node',
        'ros2 bag record --node-name rosbag2_recorder /scan',
        'ros2 launch race_diagnostics record_run.py name_prefix:=auto_other',
        'unrelated --ros-args -r __node:=auto_race_diag_node_other',
    ]
    assert find_orphans([(i + 10, 1, cmd) for i, cmd in enumerate(commands)], ()) == [
        10, 11, 12, 13]
    assert find_orphans([], ()) == []


def test_own_roots_and_descendants_are_excluded_regardless_of_order():
    marker = 'ros2 launch race_diagnostics record_run.py name_prefix:=auto_'
    processes = [(13, 12, marker), (12, 11, marker), (11, 10, 'wrapper'),
                 (10, 1, marker), (20, 1, marker)]
    assert find_orphans(processes, {10}) == [20]
    assert find_orphans(processes, {10, 20}) == []
