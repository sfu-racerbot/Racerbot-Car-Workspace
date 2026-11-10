"""
The real auto_record executable on a real ROS graph: it records while an
algorithm publishes, closes the bag when it goes, defers to another recorder,
and closes a recording in progress when bringup shuts down. Needs ROS
sourced and the workspace built:

    source /opt/ros/jazzy/setup.bash && source install/setup.bash
    export ROS_DOMAIN_ID=79 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
    python3 -m pytest src/race_diagnostics/test/test_auto_record_node.py -v

Never on the car's graph: if ROS_DOMAIN_ID is unset or 0 this file moves
itself to domain 79, localhost only, before any node starts. The one
exception is the bag-start timing test, which needs bringup's own subnet
discovery: it runs on a domain from 90 to 101 that it first finds empty
(subnet_domain), never 0 or 79. The fake
algorithm publishes on /test_only/drive, never /drive (CLAUDE.md, "Build,
test, run"), and the node under test is told to trigger on that topic.

Oracles: the rules in docs/run-diagnostics.md#automatic-recording, and the
bag itself, read back with rosbag2_py (a recording must contain the
messages the fake algorithm published). A few minutes on the Jetson.
"""
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import threading
import time

import pytest
import yaml

rclpy = pytest.importorskip(
    'rclpy', reason='needs ROS sourced (CLAUDE.md "Build, test, run"); runs '
                    'whenever rclpy imports')

if os.environ.get('ROS_DOMAIN_ID', '0') in ('', '0'):
    os.environ['ROS_DOMAIN_ID'] = '79'
os.environ['ROS_AUTOMATIC_DISCOVERY_RANGE'] = 'LOCALHOST'

from ackermann_msgs.msg import AckermannDriveStamped  # noqa: E402
from rclpy.executors import SingleThreadedExecutor  # noqa: E402
from rclpy.node import Node  # noqa: E402

TOPIC = '/test_only/drive'
CONFIG = Path(__file__).resolve().parents[1] / 'config' / 'auto_record.yaml'


@pytest.mark.parametrize('car', ['rb1', 'rb2'])
@pytest.mark.parametrize('explicit', [False, True])
def test_output_directory_resolves_per_car(tmp_path, monkeypatch, car, explicit):
    # Task 7: an empty default selects this car; an explicit path always wins.
    from race_diagnostics.auto_record_node import AutoRecordNode
    monkeypatch.setenv('RACERBOT_CAR', car)
    monkeypatch.setenv('RACERBOT_WS', str(tmp_path))
    # Also contain the legacy ~/ default when this test catches a regression.
    monkeypatch.setenv('HOME', str(tmp_path))
    requested = tmp_path / 'override'
    args = ['--ros-args']
    if explicit:
        monkeypatch.delenv('RACERBOT_CAR')
        args += ['-p', f'output_directory:={requested}']
    rclpy.init(args=args)
    node = None
    try:
        node = AutoRecordNode()
        assert node.output_directory == (requested if explicit else
                                          tmp_path / 'data' / car / 'racerbot_runs')
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


class Graph:
    """Fake algorithms and recorders, spun in a background thread."""

    def __init__(self):
        self.context = rclpy.Context()
        rclpy.init(context=self.context)
        self.executor = SingleThreadedExecutor(context=self.context)
        self.nodes = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def _spin(self):
        while not self._stop.is_set():
            self.executor.spin_once(timeout_sec=0.05)

    def add_algorithm(self, name='fake_algorithm'):
        node = Node(name, context=self.context)
        pub = node.create_publisher(AckermannDriveStamped, TOPIC, 10)
        node.sent = 0

        def tick():
            msg = AckermannDriveStamped()
            msg.header.stamp = node.get_clock().now().to_msg()
            pub.publish(msg)
            if node.sent == 0:
                node.first_sent_at = time.time()
            node.sent += 1
        node.create_timer(0.05, tick)
        self._add(name, node)
        return node

    def add_recorder(self, name='race_diag_node'):
        self._add(name, Node(name, context=self.context))

    def add_joy(self, held=True):
        from sensor_msgs.msg import Joy
        node = Node('test_joy', context=self.context)
        node.held = held
        pub = node.create_publisher(Joy, '/joy', 10)
        node.create_timer(0.02, lambda: pub.publish(
            Joy(buttons=[0, 0, 0, 0, int(node.held)])))
        self._add('test_joy', node)
        return node

    def _add(self, name, node):
        self.nodes[name] = node
        self.executor.add_node(node)

    def remove(self, name):
        node = self.nodes.pop(name)
        self.executor.remove_node(node)
        node.destroy_node()

    def close(self):
        self._stop.set()
        self._thread.join(timeout=5.0)
        for name in list(self.nodes):
            self.remove(name)
        rclpy.shutdown(context=self.context)


@pytest.fixture
def graph():
    g = Graph()
    yield g
    g.close()


def _run_watcher(tmp_path, *params):
    """The real auto_record node, triggering on TOPIC, with these parameters."""
    out = tmp_path / 'runs'
    cmd = ['ros2', 'run', 'race_diagnostics', 'auto_record', '--ros-args',
           '--params-file', str(CONFIG),
           '-p', f'output_directory:={out}',
           '-p', f"trigger_topics:=['{TOPIC}']",
           '-p', f"live_topics:=['{TOPIC}']",
           '-p', f'topics:={TOPIC} /rosout']
    for param in params:
        cmd += ['-p', param]
    log = open(tmp_path / 'watcher.log', 'w')
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)
    proc.out = out
    proc.log = tmp_path / 'watcher.log'
    proc.log_handle = log
    proc.groups = []   # recorder process groups seen, for cleanup
    time.sleep(3.0)    # the node is up and polling
    return proc


def _stop_watcher(proc):
    out, log = proc.out, proc.log_handle
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGINT)
        try:
            proc.wait(timeout=25.0)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
    log.close()
    # Last-resort cleanup only; every test asserts there were no leftovers.
    for group in proc.groups:
        try:
            os.killpg(group, signal.SIGKILL)
        except ProcessLookupError:
            pass
    for pid, _args in _processes_mentioning(out):
        os.kill(pid, signal.SIGKILL)


@pytest.fixture
def watcher(tmp_path):
    """Fast delays, so the tests about what happens next do not wait around."""
    proc = _run_watcher(tmp_path, 'start_delay_sec:=1.0', 'stop_delay_sec:=1.0',
                        'poll_period_sec:=0.2')
    yield proc
    _stop_watcher(proc)


@pytest.fixture
def shipped_watcher(tmp_path):
    """The delays bringup really uses (config/auto_record.yaml)."""
    proc = _run_watcher(tmp_path)
    yield proc
    _stop_watcher(proc)


# Domains for the one test that needs the cars' own discovery: never 0 (the cars' graph)
# or 79 (every other test), and each is checked to be empty before it is used.
SUBNET_TEST_DOMAINS = range(90, 102)
PROBE = 'auto_record_test_domain_probe'


def _domain_is_empty(domain):
    """No ROS node on `domain`, seen with subnet discovery for 3.5 s: longer than the
    3 s periodic discovery announcement, so a quiet participant still shows up."""
    context = rclpy.Context()
    rclpy.init(context=context, domain_id=domain)
    try:
        node = Node(PROBE, context=context)
        executor = SingleThreadedExecutor(context=context)
        executor.add_node(node)
        end = time.monotonic() + 3.5
        while time.monotonic() < end:
            executor.spin_once(timeout_sec=0.1)
        others = [name for name, _ns in node.get_node_names_and_namespaces() if name != PROBE]
        executor.remove_node(node)
        node.destroy_node()
        return not others
    finally:
        rclpy.shutdown(context=context)


@pytest.fixture
def subnet_domain(monkeypatch):
    """Bringup's discovery, on a domain nothing else uses.

    Bringup runs with Jazzy's default ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET. With the
    LOCALHOST setting every other test uses, a new recorder is often not discovered until
    the next periodic announcement, about 3 s later, which no car ever sees (car 1,
    2026-11-02, 10 attempts each: LOCALHOST missed 3.65 to 6.65 s, 6 over 5.0 s; SUBNET
    2.6 to 3.65 s). The fake algorithm still publishes only TOPIC."""
    monkeypatch.setenv('ROS_AUTOMATIC_DISCOVERY_RANGE', 'SUBNET')
    for domain in random.sample(list(SUBNET_TEST_DOMAINS), len(SUBNET_TEST_DOMAINS)):
        if _domain_is_empty(domain):
            monkeypatch.setenv('ROS_DOMAIN_ID', str(domain))
            return domain
    pytest.fail(f'every ROS domain in {SUBNET_TEST_DOMAINS.start}-{SUBNET_TEST_DOMAINS.stop - 1} '
                'already has nodes on this network; this test needs an empty one')


@pytest.fixture
def subnet_graph(subnet_domain):
    g = Graph()
    yield g
    g.close()


@pytest.fixture
def subnet_shipped_watcher(subnet_domain, tmp_path):
    """shipped_watcher, on subnet_domain."""
    proc = _run_watcher(tmp_path)
    yield proc
    _stop_watcher(proc)


def _ps(fields):
    done = subprocess.run(['ps', '-eo', fields], capture_output=True, text=True)
    return [line.split(None, fields.count(',')) for line in done.stdout.splitlines()[1:]]


def _processes_mentioning(text):
    return [(int(pid), args) for pid, args in _ps('pid,args')
            if str(text) in args and int(pid) != os.getpid()]


def recorder_group(watcher, run):
    """The process group of the recording of `run`. record_run.py is started
    in its own session, so its group holds every process it starts --
    including race_diag_node, whose command line never names the run."""
    for pid, _args in _processes_mentioning(f'run_directory:={run}'):
        group = os.getpgid(pid)
        watcher.groups.append(group)
        return group
    return None


def leftovers(run, group):
    """Every process still alive from the recording of `run`. run_activity is
    not one: it is meant to outlive the recording, for as long as it takes to
    read the bag. wait_for_activity() checks that it finishes."""
    alive = {pid for pid, args in _processes_mentioning(run)
             if 'race_diagnostics.run_activity' not in args}
    alive |= {int(pid) for pid, pgid in _ps('pid,pgid') if int(pgid) == group}
    return sorted(alive)


def runs(out):
    return [p for p in run_directories(out) if 'started' in info(p)]


def run_directories(out):
    return sorted(p for p in out.glob('2*') if p.is_dir()) if out.exists() else []


def standbys(out):
    return [p for p in run_directories(out) if 'started' not in info(p)]


def info(run):
    path = run / 'auto_record.yaml'
    return yaml.safe_load(path.read_text()) if path.exists() else {}


def wait_for(predicate, timeout, period=0.2):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(period)
    return predicate()


def _bag_reader(run, topic):
    import rosbag2_py
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(run / 'bag'), storage_id='mcap'),
                rosbag2_py.ConverterOptions('cdr', 'cdr'))
    reader.set_filter(rosbag2_py.StorageFilter(topics=[topic]))
    return reader


def bag_count(run, topic):
    reader = _bag_reader(run, topic)
    n = 0
    while reader.has_next():
        reader.read_next()
        n += 1
    return n


def first_stamp_in_bag(run, topic):
    """When the first message the bag holds was published, by its own header."""
    from rclpy.serialization import deserialize_message
    reader = _bag_reader(run, topic)
    _topic, data, _received = reader.read_next()
    stamp = deserialize_message(data, AckermannDriveStamped).header.stamp
    return stamp.sec + stamp.nanosec / 1e9


def wait_until_live(watcher, run):
    """The bag is subscribed to the topic, by the node's own graph check."""
    assert wait_for(lambda: 'bag_live_sec' in info(run), timeout=15.0), \
        (watcher.log.read_text(), (run / 'recorder.log').read_text())


def wait_until_recording(watcher, run):
    """Live, and record_run.py up as well: a recording fully in progress, rather
    than one being signalled while its processes are still importing."""
    wait_until_live(watcher, run)
    assert wait_for(lambda: (run / 'events.jsonl').exists(), timeout=20.0), \
        (run / 'recorder.log').read_text()
    time.sleep(1.0)


def wait_for_activity(run):
    assert wait_for(lambda: 'activity' in info(run), timeout=20.0), \
        (run / 'recorder.log').read_text()
    return info(run)['activity']


def decisions(watcher):
    path = watcher.out / 'auto_record.log'
    return path.read_text() if path.exists() else ''


def test_records_while_an_algorithm_runs_and_closes_the_bag_when_it_goes(graph, watcher):
    assert wait_for(lambda: len(standbys(watcher.out)) == 1, 10), watcher.log.read_text()
    standby = standbys(watcher.out)[0]
    assert runs(watcher.out) == [], 'activated with no algorithm running'
    assert not (standby / 'events.jsonl').exists()
    algo = graph.add_algorithm()
    assert wait_for(lambda: len(runs(watcher.out)) == 1, timeout=10.0), \
        watcher.log.read_text()
    run = runs(watcher.out)[0]
    assert run == standby, 'created a new recorder instead of activating standby'
    assert info(run)['algorithms'] == {TOPIC: ['/fake_algorithm']}
    wait_until_recording(watcher, run)
    time.sleep(3.0)    # 60 messages at 20 Hz
    group = recorder_group(watcher, run)
    assert group is not None, 'no record_run.py process for this run'
    graph.remove('fake_algorithm')
    sent = algo.sent   # read after removing it: the count can no longer grow

    assert wait_for(lambda: 'stopped' in info(run), timeout=25.0), watcher.log.read_text()
    done = info(run)
    assert done['stop_reason'].startswith('no algorithm publishing for')
    assert done['recorder_exit_code'] == 0 and done['recorder_killed'] is False
    assert done['bag_exit_code'] == 0
    assert done['recorder_failures'] == []
    assert 'superseded' not in done
    assert (run / 'bag' / 'metadata.yaml').exists(), 'bag was not closed cleanly'
    recorded = bag_count(run, TOPIC)
    # The bag starts a few seconds after the algorithm; it must hold most
    # of what was published after that, and never more than was sent.
    assert 40 <= recorded <= sent, (recorded, sent)
    assert (run / 'metadata' / 'git' / 'racerbot-ws.txt').exists()
    assert (run / 'events.jsonl').exists()
    assert leftovers(run, group) == [], 'recorder processes left running'
    assert len(runs(watcher.out)) == 1
    assert wait_for(lambda: len(standbys(watcher.out)) == 1, 15)
    assert standbys(watcher.out)[0] != run
    # No /odom in this bag: unknown, not "never moved".
    assert wait_for_activity(run)['moved'] is None
    text = decisions(watcher)
    assert 'recording /fake_algorithm' in text and 'saved run' in text


def test_defers_to_another_recorder_and_resumes_when_it_leaves(graph, watcher):
    graph.add_recorder('race_diag_node')        # e.g. a record_run.py started by hand
    time.sleep(1.0)
    graph.add_algorithm()
    time.sleep(5.0)                               # well past start_delay_sec
    assert runs(watcher.out) == [], 'recorded beside another recorder'
    assert 'another recorder is on the graph (/race_diag_node)' in watcher.log.read_text()
    assert 'another recorder is on the graph (/race_diag_node)' in decisions(watcher)

    graph.remove('race_diag_node')
    assert wait_for(lambda: len(runs(watcher.out)) == 1, timeout=10.0), \
        watcher.log.read_text()
    run = runs(watcher.out)[0]
    wait_until_recording(watcher, run)
    group = recorder_group(watcher, run)
    assert group is not None

    graph.add_recorder('rosbag2_recorder')       # a `ros2 bag record` appears mid-run
    assert wait_for(lambda: 'stopped' in info(run), timeout=25.0), watcher.log.read_text()
    assert info(run)['stop_reason'] == 'another recorder started: /rosbag2_recorder'
    assert info(run)['superseded'] is True
    assert (run / 'bag' / 'metadata.yaml').exists()
    time.sleep(3.0)
    assert len(runs(watcher.out)) == 1, 'restarted beside the other recorder'
    assert leftovers(run, group) == []


def test_bringup_shutdown_closes_a_recording_in_progress(graph, watcher):
    graph.add_algorithm()
    assert wait_for(lambda: len(runs(watcher.out)) == 1, timeout=10.0)
    run = runs(watcher.out)[0]
    wait_until_recording(watcher, run)
    group = recorder_group(watcher, run)
    assert group is not None
    os.killpg(watcher.pid, signal.SIGINT)        # bringup's Ctrl+C
    watcher.wait(timeout=25.0)
    done = info(run)
    assert done['stop_reason'] == 'bringup shut down'
    assert done['recorder_exit_code'] == 0
    assert (run / 'bag' / 'metadata.yaml').exists(), 'bag was not closed cleanly'
    assert bag_count(run, TOPIC) > 0
    assert leftovers(run, group) == [], 'recorder outlived bringup'
    assert 'Failed to publish log message' not in watcher.log.read_text()


def test_a_killed_watcher_does_not_leave_its_recorder_running(graph, watcher):
    graph.add_algorithm()
    assert wait_for(lambda: len(runs(watcher.out)) == 1, timeout=10.0)
    run = runs(watcher.out)[0]
    wait_until_recording(watcher, run)
    group = recorder_group(watcher, run)
    assert group is not None
    os.killpg(watcher.pid, signal.SIGKILL)       # no chance to stop anything
    watcher.wait(timeout=5.0)
    assert wait_for(lambda: leftovers(run, group) == [], timeout=25.0), \
        ([line for line in _ps('pid,pgid,args') if int(line[0]) in leftovers(run, group)],
         (run / 'recorder.log').read_text()[-3000:])
    assert (run / 'bag' / 'metadata.yaml').exists(), 'bag was not closed cleanly'


def test_the_domain_probe_refuses_a_domain_with_a_node_on_it(subnet_domain):
    # The subnet test's isolation rests on this check (A8: the refusal is the half that
    # matters): a node already on the domain, seen over subnet discovery, makes it not empty.
    g = Graph()
    try:
        g.add_recorder('someone_else_on_this_domain')
        assert not _domain_is_empty(subnet_domain)
    finally:
        g.close()


def test_the_bag_holds_the_run_from_soon_after_the_algorithm_appears(subnet_graph,
                                                                    subnet_shipped_watcher):
    # recorded measurement, 2026-10-31 on the Jetson, this same method: the old
    # path through record_run.py missed 5.45 to 7.0 s (4 runs); this one 2.95 to
    # 6.9 s (8 runs, median 4). ROS discovery only ever adds time and can add up
    # to 2.7 s, so one sample cannot tell them apart. The fastest attempt shows what
    # our own pipeline costs, and 5.0 s is under anything the old path ever did.
    # Judged on the bag's own first message against when the algorithm really
    # published its first, not on what the node reports.
    # Since 2026-11-02 with bringup's discovery (subnet_domain): those numbers were taken
    # with the tests' LOCALHOST discovery, whose ~3 s announcement wait put car 1's
    # attempts at 3.65 or 6.65 s and failed this test unless a retry was fast.
    graph, shipped_watcher = subnet_graph, subnet_shipped_watcher
    misses = []
    for attempt in range(4):
        algo = graph.add_algorithm()
        assert wait_for(lambda: len(runs(shipped_watcher.out)) == attempt + 1, timeout=10.0), \
            shipped_watcher.log.read_text()
        run = runs(shipped_watcher.out)[attempt]
        wait_until_live(shipped_watcher, run)
        time.sleep(2.0)
        graph.remove('fake_algorithm')
        assert wait_for(lambda: 'stopped' in info(run), timeout=25.0)
        misses.append(first_stamp_in_bag(run, TOPIC) - algo.first_sent_at)
        assert misses[-1] >= 0.0, 'the bag holds a message from before it was sent'
        if misses[-1] <= 5.0:
            return
    pytest.fail(f'the bag always missed over 5.0 s of the run: {misses}')


def test_the_log_says_when_the_bag_is_really_listening(graph, watcher):
    graph.add_algorithm()
    assert wait_for(lambda: len(runs(watcher.out)) == 1, timeout=10.0)
    run = runs(watcher.out)[0]
    wait_until_live(watcher, run)
    assert f'recording live: {TOPIC}' in watcher.log.read_text()
    # Not the last run's recorder lingering on the graph under the same name:
    # the bag this run started has opened its file.
    assert (run / 'bag').is_dir()
    # It was checked on the graph, so the recorder really is subscribed.
    assert 1.0 <= info(run)['bag_live_sec'] <= 15.0
    assert f'recording live: {TOPIC}' in decisions(watcher)


def test_a_bag_that_dies_stops_the_recording_and_is_reported(graph, watcher):
    graph.add_algorithm()
    assert wait_for(lambda: len(runs(watcher.out)) == 1, timeout=10.0)
    run = runs(watcher.out)[0]
    wait_until_live(watcher, run)
    bag = [pid for pid, args in _processes_mentioning(run / 'bag')
           if 'bag record' in args]
    assert bag, 'the bag is not its own process'
    os.kill(bag[0], signal.SIGKILL)
    assert wait_for(lambda: 'stopped' in info(run), timeout=25.0), watcher.log.read_text()
    done = info(run)
    assert done['stop_reason'] == 'the recording exited on its own'
    assert done['bag_exit_code'] != 0
    assert {'process': 'ros2 bag record', 'exit_code': done['bag_exit_code']} \
        in done['recorder_failures']
    # Rule 5 of the docs: a recorder that cannot run must not restart in a loop.
    time.sleep(2.0)
    assert len(runs(watcher.out)) == 1, 'restarted after the recording died'


def test_too_little_disk_refuses_and_leaves_a_record_of_why(tmp_path, graph):
    proc = _run_watcher(tmp_path, 'start_delay_sec:=1.0', 'poll_period_sec:=0.2',
                        'min_free_gb:=1000000000.0')
    try:
        graph.add_algorithm()
        assert wait_for(lambda: 'not recording' in decisions(proc), timeout=10.0), \
            proc.log.read_text()
        assert runs(proc.out) == [], 'recorded with too little disk'
        assert 'Delete old runs' in decisions(proc)
    finally:
        _stop_watcher(proc)


def test_record_run_stops_every_process_on_sigint(tmp_path):
    # recorded measurement: all 19 auto runs of 2026-10-30 logged system_log
    # "failed to terminate '5' seconds after receiving 'SIGINT'", then exit -15.
    run = tmp_path / 'run'
    log = open(tmp_path / 'recorder.log', 'w')
    proc = subprocess.Popen(['ros2', 'launch', 'race_diagnostics', 'record_run.py',
                             f'run_directory:={run}', 'name_prefix:=test_',
                             'record_bag:=false'],
                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        # Past their Python imports, where a signal is a plain KeyboardInterrupt.
        assert wait_for(lambda: (run / 'system' / 'proc.jsonl').exists()
                        and (run / 'system' / 'proc.jsonl').stat().st_size > 0
                        and (run / 'events.jsonl').exists()
                        and (run / 'metadata' / 'params').is_dir(), timeout=20.0)
        time.sleep(1.0)
        # Only the launch process, the way auto_record stops it.
        started = time.monotonic()
        proc.send_signal(signal.SIGINT)
        assert proc.wait(timeout=20.0) == 0
        text = (tmp_path / 'recorder.log').read_text()
        assert 'failed to terminate' not in text, text
        assert 'process has died' not in text, text
        assert time.monotonic() - started < 4.5, 'launch waited for its 5 s escalation'
    finally:
        log.close()
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()


class _Spin:
    def __init__(self, error):
        self.error = error

    def __call__(self, _node):
        raise self.error


def test_race_diag_ignores_the_error_rclpy_raises_as_it_shuts_down():
    # recorded measurement: run 20261030-181003 died with this RuntimeError from
    # take_message after SIGINT, with exit code 1 and a traceback.
    from race_diagnostics.race_diag_node import spin_until_shutdown
    error = RuntimeError("Unable to convert call argument '0' to Python object")
    spin_until_shutdown(None, spin=_Spin(error), ok=lambda: False)


def test_race_diag_still_dies_on_a_real_error_while_running():
    from race_diagnostics.race_diag_node import spin_until_shutdown
    with pytest.raises(RuntimeError, match='a real bug'):
        spin_until_shutdown(None, spin=_Spin(RuntimeError('a real bug')), ok=lambda: True)


@pytest.mark.parametrize('error', ['KeyboardInterrupt', 'ExternalShutdownException'])
def test_race_diag_treats_a_normal_shutdown_as_normal(error):
    from rclpy.executors import ExternalShutdownException
    from race_diagnostics.race_diag_node import spin_until_shutdown
    raised = {'KeyboardInterrupt': KeyboardInterrupt,
              'ExternalShutdownException': ExternalShutdownException}[error]()
    spin_until_shutdown(None, spin=_Spin(raised), ok=lambda: True)


def _write_bag(run, odom_speeds, lb):
    from nav_msgs.msg import Odometry
    from rclpy.serialization import serialize_message
    import rosbag2_py
    from sensor_msgs.msg import Joy
    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=str(run / 'bag'), storage_id='mcap'),
                rosbag2_py.ConverterOptions('cdr', 'cdr'))
    for topic_id, (name, kind) in enumerate((('/odom', 'nav_msgs/msg/Odometry'),
                                             ('/joy', 'sensor_msgs/msg/Joy'))):
        writer.create_topic(rosbag2_py.TopicMetadata(
            id=topic_id, name=name, type=kind, serialization_format='cdr'))
    t = 1
    for speed in odom_speeds:
        msg = Odometry()
        msg.twist.twist.linear.x = speed
        writer.write('/odom', serialize_message(msg), t)
        t += 1
    for held in lb:
        msg = Joy()
        msg.buttons = [0, 0, 0, 0, held, 0]
        writer.write('/joy', serialize_message(msg), t)
        t += 1
    del writer


def test_run_activity_reads_a_real_bag_into_auto_record_yaml(tmp_path):
    run = tmp_path / '20261030-184149'
    run.mkdir()
    (run / 'auto_record.yaml').write_text(yaml.safe_dump({'started': 'then'}))
    _write_bag(run, [3.5, -4.0, 1.0], [1, 0, 0, 0])
    done = subprocess.run([sys.executable, '-m', 'race_diagnostics.run_activity', str(run)],
                          capture_output=True, text=True, timeout=60.0)
    assert done.returncode == 0, done.stderr
    recorded = info(run)
    assert recorded['started'] == 'then'
    # oracle: the values written above; 1 of 4 /joy samples holds button 4.
    assert recorded['activity'] == {
        'first_odom_speed_mps': 3.5, 'max_speed_mps': 4.0, 'moved': True,
        'started_late': True, 'lb_held_fraction': 0.25}
    assert '20261030-184149' in (tmp_path / 'auto_record.log').read_text()


HUNG_LAUNCH = """
import importlib.util, sys, threading, time
spec = importlib.util.spec_from_file_location('record_run_hang', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.LAUNCH_EXIT_AFTER_SEC = 0.5
# What was seen: every child gone, the interpreter stuck joining a thread at exit.
threading.Thread(target=time.sleep, args=(60.0,)).start()
module._exit_even_if_launch_hangs(None, None)
"""


def test_a_launch_that_hangs_at_exit_still_goes_away(tmp_path):
    # recorded measurement: 3 of 14 runs of this file on 2026-10-31 left only
    # `ros2 launch` alive after its watcher was killed, in threading._shutdown,
    # with nobody left to stop it.
    launch = Path(__file__).resolve().parents[1] / 'launch' / 'record_run.py'
    done = subprocess.run([sys.executable, '-c', HUNG_LAUNCH, str(launch)],
                          capture_output=True, text=True, timeout=20.0)
    assert done.returncode == 0, done.stderr


def test_the_exit_timer_outlasts_the_escalation_of_launch_children():
    # oracle: launch's defaults, SIGINT -> 5 s -> SIGTERM -> 10 s -> SIGKILL
    # (seen in run recorder.logs). Exiting sooner would orphan a stuck child.
    import importlib.util
    launch = Path(__file__).resolve().parents[1] / 'launch' / 'record_run.py'
    spec = importlib.util.spec_from_file_location('record_run_timer', launch)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.LAUNCH_EXIT_AFTER_SEC > 5.0 + 10.0
    # Short of the 25 s the watcher-killed test allows for everything to go.
    assert module.LAUNCH_EXIT_AFTER_SEC <= 20.0


def test_record_run_arms_the_exit_timer_on_shutdown():
    import importlib.util
    from launch.actions import RegisterEventHandler
    from launch.event_handlers import OnShutdown
    launch = Path(__file__).resolve().parents[1] / 'launch' / 'record_run.py'
    spec = importlib.util.spec_from_file_location('record_run_wiring', launch)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    handlers = [action.event_handler for action in module.generate_launch_description().entities
                if isinstance(action, RegisterEventHandler)]
    assert [type(handler) for handler in handlers] == [OnShutdown]


def test_idle_bag_is_paused_then_resumes_same_run_and_signals_probe(tmp_path, graph):
    # Rule 1/3 oracle: real Resume/IsPaused services and the closed bag's timestamps.
    from rosbag2_interfaces.srv import IsPaused
    graph.add_algorithm('ackermann_mux')  # ignored publisher, supplying pre-start data
    watcher = _run_watcher(tmp_path, 'start_delay_sec:=0.0', 'poll_period_sec:=0.2')
    try:
        assert wait_for(lambda: len(standbys(watcher.out)) == 1, 10)
        run = standbys(watcher.out)[0]
        probe = Node('paused_bag_test', context=graph.context)
        executor = SingleThreadedExecutor(context=graph.context)
        executor.add_node(probe)
        client = probe.create_client(IsPaused, '/auto_rosbag2_recorder/is_paused')
        try:
            assert client.wait_for_service(timeout_sec=10)
            future = client.call_async(IsPaused.Request())
            executor.spin_until_future_complete(future, timeout_sec=5)
            assert future.result().paused is True
            assert not (run / 'events.jsonl').exists()
            # The bag has received data while paused; none may be retained.
            time.sleep(1)
            activation_after = time.time()
            graph.add_algorithm()
            assert wait_for(lambda: runs(watcher.out) == [run], 10)
            wait_until_recording(watcher, run)
            future = client.call_async(IsPaused.Request())
            executor.spin_until_future_complete(future, timeout_sec=5)
            assert future.result().paused is False
            import json
            events = [json.loads(s) for s in (run / 'events.jsonl').read_text().splitlines()]
            assert events[0]['category'] == 'armed'
            assert 'standby_prepared' in info(run) and 'started' in info(run)
        finally:
            executor.shutdown()
            probe.destroy_node()
        os.killpg(watcher.pid, signal.SIGINT)
        watcher.wait(timeout=25)
        assert bag_count(run, TOPIC) > 0
        assert first_stamp_in_bag(run, TOPIC) >= activation_after
    finally:
        _stop_watcher(watcher)


def test_held_lb_denies_standby_and_cold_start_on_real_joy(tmp_path, graph):
    # Rule 6 deny path: receive Joy through DDS before polling the real graph.
    from race_diagnostics.auto_record_node import AutoRecordNode
    graph.add_joy(True)
    graph.add_algorithm()
    rclpy.init(args=['--ros-args', '-p', f'output_directory:={tmp_path / "runs"}',
                     '-p', f"trigger_topics:=['{TOPIC}']", '-p', 'start_delay_sec:=0.0'])
    node = AutoRecordNode()
    for timer in node.timers:
        timer.cancel()
    try:
        deadline = time.monotonic() + 10
        while (not node._lb_held() or not node._graph()[0][TOPIC]) and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
        assert node._lb_held() and node._graph()[0][TOPIC]
        before_clients = list(node.clients)
        for _ in range(5):
            node._tick()
            rclpy.spin_once(node, timeout_sec=0.02)
        assert run_directories(node.output_directory) == []
        assert node._processes() == []
        assert list(node.clients) == before_clients
        assert (node.output_directory / 'auto_record.log').read_text().count(
            'waiting for LB release to start recording') == 1
    finally:
        node.shutdown('test complete')
        node.destroy_node()
        rclpy.shutdown()


def test_snapshot_waits_for_continuous_lb_release(tmp_path, graph):
    watcher = _run_watcher(tmp_path, 'start_delay_sec:=0.0', 'poll_period_sec:=0.1',
                           'snapshot_release_sec:=1.0')
    try:
        assert wait_for(lambda: len(standbys(watcher.out)) == 1, 10)
        run = standbys(watcher.out)[0]
        joy = graph.add_joy(True)
        time.sleep(3.5)  # discovered and delivering fresh LB before activation
        graph.add_algorithm()
        assert wait_for(lambda: runs(watcher.out) == [run], 10)
        wait_until_recording(watcher, run)
        assert info(run)['params_snapshot'] == 'pending'
        assert not (run / 'metadata' / 'params').exists()
        joy.held = False
        time.sleep(0.4)
        assert info(run)['params_snapshot'] == 'pending'
        joy.held = True
        time.sleep(0.4)
        assert info(run)['params_snapshot'] == 'pending'
        joy.held = False
        assert wait_for(lambda: info(run)['params_snapshot'] == 'started', 5)
        assert wait_for(lambda: (run / 'metadata' / 'params' / 'auto_record.yaml').exists(), 10)
    finally:
        _stop_watcher(watcher)


def test_unused_standby_is_removed_on_shutdown(tmp_path):
    watcher = _run_watcher(tmp_path, 'poll_period_sec:=0.1')
    try:
        assert wait_for(lambda: len(standbys(watcher.out)) == 1, 10)
        run = standbys(watcher.out)[0]
        os.killpg(watcher.pid, signal.SIGINT)
        watcher.wait(timeout=25)
        assert not run.exists()
        assert run_directories(watcher.out) == []
        assert _processes_mentioning(run) == []
    finally:
        _stop_watcher(watcher)


@pytest.mark.parametrize('shutdown', [False, True])
def test_pending_snapshot_records_held_skip_reason(tmp_path, graph, shutdown):
    watcher = _run_watcher(tmp_path, 'start_delay_sec:=0.0', 'stop_delay_sec:=0.1',
                           'poll_period_sec:=0.1', 'snapshot_wait_after_stop_sec:=0.5')
    try:
        assert wait_for(lambda: len(standbys(watcher.out)) == 1, 10)
        run = standbys(watcher.out)[0]
        graph.add_joy(True)
        time.sleep(3.5)
        graph.add_algorithm()
        assert wait_for(lambda: runs(watcher.out) == [run], 10)
        wait_until_recording(watcher, run)
        assert info(run)['params_snapshot'] == 'pending'
        if shutdown:
            os.killpg(watcher.pid, signal.SIGINT)
            watcher.wait(timeout=25)
            reason = 'skipped (LB held at shutdown)'
        else:
            graph.remove('fake_algorithm')
            reason = 'skipped (LB held after stop)'
        assert wait_for(lambda: info(run).get('params_snapshot') == reason, 10)
        assert reason in decisions(watcher)
        assert not (run / 'metadata' / 'params').exists()
    finally:
        _stop_watcher(watcher)


def test_startup_stops_orphan_auto_process_but_leaves_unrelated(tmp_path):
    orphan = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)',
                               '__node:=auto_snapshot_params'],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)',
                                  '__node:=race_diag_node'])
    watcher = None
    try:
        watcher = _run_watcher(tmp_path, 'poll_period_sec:=0.1')
        assert wait_for(lambda: orphan.poll() is not None, 15)
        assert unrelated.poll() is None
        assert f'stopping orphan auto recorder pid {orphan.pid}' in decisions(watcher)
        assert wait_for(lambda: len(standbys(watcher.out)) == 1, 10)
    finally:
        if watcher is not None:
            _stop_watcher(watcher)
        for proc in (orphan, unrelated):
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=5)
