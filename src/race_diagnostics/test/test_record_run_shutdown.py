"""record_run.py stops on one SIGINT; runner: pytest src/race_diagnostics/test/ (sourced)."""
import os
from pathlib import Path
import signal
import subprocess
import time

import pytest

pytest.importorskip('rclpy', reason='requires ROS Jazzy and a built workspace')
if os.environ.get('ROS_DOMAIN_ID', '0') in ('', '0'):
    os.environ['ROS_DOMAIN_ID'] = '79'
os.environ['ROS_AUTOMATIC_DISCOVERY_RANGE'] = 'LOCALHOST'

# launch's own message when a child ignores SIGINT for sigterm_timeout
# (launch/actions/execute_local.py, __get_shutdown_timer_actions).
ESCALATION = "escalating to 'SIGTERM'"


def wait_for(predicate, timeout, period=0.2):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(period)
    return predicate()


def graph_nodes():
    done = subprocess.run(['ros2', 'node', 'list', '--no-daemon'],
                          capture_output=True, text=True, timeout=20.0)
    return done.stdout.split()


def group_members(group):
    done = subprocess.run(['ps', '-eo', 'pid=,pgid=,args='], capture_output=True, text=True)
    rows = [line.split(None, 2) for line in done.stdout.splitlines()]
    return [row for row in rows if len(row) == 3 and int(row[1]) == group]


def test_one_sigint_stops_every_recorder_process_without_escalation(tmp_path):
    """auto_record's parent-death signal is a single SIGINT to this launch.

    Every child, system_log included, must stop on it. A child that waits
    for launch's SIGTERM escalation leaves the stop to a timer, and a
    recording whose watcher was SIGKILLed then stayed up with nobody to
    stop it (test_auto_record_node's killed-watcher test, car 1, 2026-10-29).
    """
    run = tmp_path / 'run'
    run.mkdir()
    log_path = tmp_path / 'launch.log'
    with open(log_path, 'w') as log:
        proc = subprocess.Popen(
            ['ros2', 'launch', 'race_diagnostics', 'record_run.py',
             f'run_directory:={run}', 'record_bag:=false', 'name_prefix:=shutdown_test_'],
            stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            start_new_session=True)
    try:
        # Every child running, not starting: a SIGINT that lands between
        # rclpy.init() and spin() is a different race (see the PR notes).
        samples = run / 'system' / 'proc.jsonl'
        assert wait_for(lambda: samples.exists() and samples.stat().st_size > 0, timeout=30.0), \
            log_path.read_text()
        assert wait_for(lambda: 'STATUS:' in log_path.read_text(), timeout=30.0), \
            log_path.read_text()   # race_diag_node's status timer: it is spinning
        assert wait_for(lambda: '/shutdown_test_snapshot_params' in graph_nodes(), timeout=30.0)
        os.kill(proc.pid, signal.SIGINT)
        proc.wait(timeout=30.0)
        text = log_path.read_text()
        # oracle: launch logs ESCALATION exactly when a child outlives SIGINT.
        assert ESCALATION not in text, text
        # oracle: the session leader's group holds every process it started.
        assert wait_for(lambda: group_members(proc.pid) == [], timeout=5.0), \
            group_members(proc.pid)
        assert proc.returncode == 0, text
    finally:
        if proc.poll() is None or group_members(proc.pid):
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(timeout=10.0)


def test_system_log_is_launched_without_the_ros2_run_wrapper(tmp_path):
    """`ros2 run` swallows SIGINT and waits for a terminal to signal the
    group (ros2run/api/__init__.py run_executable), so launch's SIGINT,
    which goes to one pid, never reached system_log through it."""
    import importlib.util
    from launch import LaunchContext
    from launch.actions import ExecuteProcess
    from launch.utilities import perform_substitutions
    launch_file = Path(__file__).resolve().parents[1] / 'launch' / 'record_run.py'
    spec = importlib.util.spec_from_file_location('record_run_shutdown', launch_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    context = LaunchContext()
    context.launch_configurations.update({
        'run_directory': str(tmp_path / 'run'), 'record_bag': 'false',
        'name_prefix': 'p_', 'system_log': 'true',
        'topics': ' '.join(module.DEFAULT_TOPICS)})
    processes = [a for a in module._setup(context) if isinstance(a, ExecuteProcess)]
    names = [perform_substitutions(context, a.name) if a.name else None for a in processes]
    system = [a for a, name in zip(processes, names) if name == 'p_system_log']
    assert len(system) == 1, names
    system[0].process_description.prepare(context, system[0])
    cmd = system[0].process_description.final_cmd
    # oracle: the installed console script, per setup.py's entry point.
    assert os.path.basename(cmd[0]) == 'system_log', cmd
    assert Path(cmd[0]).parent.name == 'race_diagnostics', cmd
    assert cmd[1:] == ['--output', str(tmp_path / 'run' / 'system')], cmd


def test_snapshot_launch_is_optional_and_standby_reaches_probe(tmp_path):
    # Oracle: warm-recorder rule 1, evaluated as real launch actions.
    import importlib.util
    from launch import LaunchContext
    from launch_ros.actions import Node
    launch_file = Path(__file__).resolve().parents[1] / 'launch' / 'record_run.py'
    spec = importlib.util.spec_from_file_location('record_run_standby', launch_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    context = LaunchContext()
    context.launch_configurations.update({
        'run_directory': str(tmp_path / 'run'), 'record_bag': 'false',
        'name_prefix': 'p_', 'system_log': 'false', 'topics': '/scan',
        'snapshot': 'false', 'standby': 'true'})
    nodes = [a for a in module._setup(context) if isinstance(a, Node)]
    assert [a.node_executable for a in nodes] == ['race_diag_node']
    # The real node constructed with the launch parameters must stay silent.
    from launch_ros.utilities import evaluate_parameters
    params = evaluate_parameters(context, nodes[0]._Node__parameters)
    assert params[-1]['standby'] is True
    context.launch_configurations['snapshot'] = 'true'
    nodes = [a for a in module._setup(context) if isinstance(a, Node)]
    assert [a.node_executable for a in nodes] == ['race_diag_node', 'snapshot_params']


def test_standby_probe_ignores_other_directory_then_arms_first(tmp_path):
    # Oracle: rule 2. Real QoS/ROS messages, never any driving publisher.
    import json
    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from std_msgs.msg import String
    from sensor_msgs.msg import Joy
    from race_diagnostics.race_diag_node import RaceDiagNode
    run = tmp_path / 'run'
    rclpy.init(args=['--ros-args', '-p', f'output_directory:={run}',
                     '-p', 'standby:=true'])
    node = RaceDiagNode()
    sender = rclpy.create_node('standby_signal_test')
    pub = sender.create_publisher(String, '/auto_record/start', QoSProfile(
        depth=1, reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL))
    def spin_for(seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.05)
    try:
        node._joy(Joy(buttons=[0, 0, 0, 0, 1]))
        spin_for(1.1)
        assert not run.exists()
        pub.publish(String(data=str(tmp_path / 'other')))
        spin_for(1.1)
        assert not run.exists()
        pub.publish(String(data=str(run / '..' / 'run')))
        assert wait_for(lambda: (spin_for(0.1) or (run / 'events.jsonl').exists()), 5)
        records = [json.loads(line) for line in (run / 'events.jsonl').read_text().splitlines()]
        assert records[0]['category'] == 'armed'
        assert [r['category'] for r in records].count('armed') == 1
        assert node.counts['joy'] == 0  # standby data must not leak into the run
        pub.publish(String(data=str(run)))
        spin_for(0.2)
        assert (run / 'events.jsonl').read_text().count('"category": "armed"') == 1
    finally:
        node.close()
        node.destroy_node()
        sender.destroy_node()
        rclpy.shutdown()
