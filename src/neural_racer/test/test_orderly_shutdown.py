"""An orderly signal must send its own stop, even when no more scans arrive."""
import os
import signal
import subprocess
import sys
import time

import pytest
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import LaserScan


@pytest.mark.parametrize('stop_signal', [signal.SIGINT, signal.SIGTERM])
def test_signal_sends_final_shadow_stop(stop_signal, tmp_path, monkeypatch):
    monkeypatch.setenv('ROS_LOCALHOST_ONLY', '1')
    monkeypatch.setenv('ROS_AUTOMATIC_DISCOVERY_RANGE', 'LOCALHOST')
    monkeypatch.setenv('ROS_DOMAIN_ID', '79')
    ns = f'/test_only/neural_shutdown_{os.getpid()}_{stop_signal.value}'
    rclpy.init(args=[])
    probe = rclpy.create_node('observer', namespace=ns)
    received = []
    probe.create_subscription(AckermannDriveStamped, ns + '/shadow', received.append, 10)
    scans = probe.create_publisher(LaserScan, ns + '/scan', 10)
    args = ['-r', '__ns:=' + ns, '-p', 'models_dir:=' + str(tmp_path), '-p', 'shadow:=true',
            '-p', 'shadow_topic:=' + ns + '/shadow', '-p', 'scan_topic:=' + ns + '/scan',
            '-p', 'drive_topic:=' + ns + '/unused_drive', '-p', 'intent_topic:=' + ns + '/intent']
    for param in ('joy_topic', 'odom_topic', 'imu_topic', 'imu_status_topic', 'pose_topic',
                  'path_topic', 'frozen_map_topic'):
        args += ['-p', param + ':=' + ns + '/' + param]
    child = subprocess.Popen(
        [sys.executable, '-m', 'neural_racer.neural_racer_node', '--ros-args', *args],
        env=dict(os.environ, PYTHONPATH=os.pathsep.join(
            [os.environ.get('PYTHONPATH', ''), *(str(p) for p in sys.path)])))

    def wait_for(condition):
        deadline = time.monotonic() + 8
        while not condition():
            assert time.monotonic() < deadline, f'missing discovery or stop; child exit={child.poll()}'
            rclpy.spin_once(probe, timeout_sec=0.02)

    try:
        wait_for(lambda: scans.get_subscription_count() and probe.count_publishers(ns + '/shadow'))
        scans.publish(LaserScan())  # no deadman, one expected stop; proves DDS is connected
        wait_for(lambda: len(received) == 1)
        before_signal = probe.get_clock().now().nanoseconds
        child.send_signal(stop_signal)
        wait_for(lambda: len(received) == 2)
        assert child.wait(timeout=6) == 0
        final = received[1]
        assert final.header.stamp.sec * 10**9 + final.header.stamp.nanosec >= before_signal
        assert (final.drive.steering_angle, final.drive.speed) == (0.0, 0.0)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
        probe.destroy_node()
        rclpy.shutdown()
