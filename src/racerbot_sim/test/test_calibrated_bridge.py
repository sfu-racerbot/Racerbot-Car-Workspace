"""Compare the live producer's published state to a fresh headless Gym plant."""
import json
from pathlib import Path
import sys
import time

import numpy as np
import pytest

from conftest import require_isolation, require_test_drive_topic

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src/racerbot_sim'))


@pytest.mark.parametrize('car,floor', [('rb1', ''), ('rb2', ''),
                                       ('rb2', 'sfu_course_2026_11_11_rb2')])
def test_calibrated_ros_trace_matches_headless(car, floor, tmp_path):
    require_isolation()
    import rclpy
    from ackermann_msgs.msg import AckermannDriveStamped
    from rclpy.executors import SingleThreadedExecutor
    from std_msgs.msg import String
    from racerbot_sim.gym_bridge_node import GymBridgeNode

    topic = '/test_only/parity/ackermann_cmd'
    require_test_drive_topic(topic)
    args = ['--ros-args', '-p', f'car:={car}', '-p', 'physics:=calibrated',
            '-p', f'track_directory:={tmp_path}', '-p', f'drive_topic:={topic}']
    if floor:
        args.extend(['-p', f'floor:={floor}'])
    rclpy.init(args=args)
    producer = probe = None
    executor = SingleThreadedExecutor()
    trace = []
    try:
        probe = rclpy.create_node('calibration_probe', use_global_arguments=False)
        pub = probe.create_publisher(AckermannDriveStamped, topic, 10)
        probe.create_subscription(String, '/sim/status',
                                  lambda msg: trace.append(json.loads(msg.data)), 100)
        producer = GymBridgeNode()
        executor.add_node(producer)
        executor.add_node(probe)
        deadline = time.monotonic() + 30
        while len(trace) < 64 and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.1)
            if trace:
                n = len(trace)
                command = AckermannDriveStamped()
                command.drive.speed = 1.5 if n < 47 else 0.0
                command.drive.steering_angle = 0.2 if n < 22 else -0.18
                pub.publish(command)
    finally:
        executor.shutdown()
        if producer is not None:
            producer.bridge.close()
            producer.destroy_node()
        if probe is not None:
            probe.destroy_node()
        rclpy.shutdown()

    assert [row['steps'] for row in trace] == list(range(1, 65))
    assert all((row['car_profile'], row['floor'], row['physics']) ==
               (car, floor, 'calibrated') for row in trace)
    sys.path.insert(0, str(ROOT / 'tools/f1tenth_sim'))
    from eval_gap import make_plant
    from sim_fidelity import calibration_for_car, with_floor
    headless = make_plant(str(tmp_path / 'indoor_oval'), 12345,
                          with_floor(floor or None, calibration_for_car(car)))
    expected = []
    try:
        headless.reset(np.array([[0.0, -3.2, 0.0]]))
        headless.step([(0.0, 0.0)])  # Same stationary scan warm-up as the bridge.
        for row in trace:
            cmd = row['commanded']
            headless.step([(cmd['steering'], cmd['speed']) if cmd['fresh'] else (0.0, 0.0)])
            expected.append(headless.truth()[[0, 1, 4, 3, 5]].copy())
    finally:
        headless.close()
    actual = np.array([[row['ego'][key] for key in ('x', 'y', 'yaw', 'speed', 'yaw_rate')]
                       for row in trace])
    assert actual[:, 3].max() > 1.0 and actual[-1, 3] < 0.8
    assert actual[:, 4].min() < -0.1 and actual[:, 4].max() > 0.1
    errors = np.max(np.abs(actual - expected), axis=0)
    print(car, floor or 'base', '64 ticks; max |ROS - headless| [x,y,yaw,v,w]:', errors)
    # The ROS status rounds to four decimal places in metres, radians and seconds.
    np.testing.assert_allclose(actual, expected, atol=0.00005001, rtol=0)


@pytest.mark.parametrize('options,reason', [
    ({'physics': 'unknown'}, 'physics'), ({'floor': 'oops'}, 'calibrated'),
    ({'physics': 'calibrated', 'control_dt': 0.05}, '40 Hz'),
    ({'physics': 'calibrated', 'controlled_cars': 2}, 'single car'),
    ({'physics': 'calibrated', 'opponents': [object()]}, 'single car'),
])
def test_refuse_unsupported_physics(options, reason):
    from racerbot_sim.sim_bridge import SimConfig
    with pytest.raises(ValueError, match=reason):
        SimConfig(track_path='', centerline=[], **options)


def test_refuse_car_two_floor_on_car_one():
    from racerbot_sim.sim_bridge import SimBridge, SimConfig
    with pytest.raises(ValueError, match='not measured on rb1'):
        SimBridge(SimConfig(track_path='', centerline=[], car='rb1', physics='calibrated',
                            floor='sfu_course_2026_11_11_rb2'))
