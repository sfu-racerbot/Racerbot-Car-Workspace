"""
auto_record_node.py

Bringup's automatic run recorder. Every f1tenth_stack bringup starts it
(pass `record:=false` to leave it out). While an algorithm runs -- any node
publishing /drive or another trigger topic in config/auto_record.yaml, ours
or team A's -- it records into a new ~/racerbot-ws/data/<car>/racerbot_runs/<time>/,
the same layout a hand-started recording writes, and stops when the algorithm
goes. It never records beside another recorder.

It prepares a paused bag and a subscribed, silent probe while LB is released.
Starting a warm run resumes that bag and signals the existing probe. It
subscribes to /joy and publishes only /auto_record/start, never a drive topic.
The decisions and LB gates are in auto_record.py (tested without ROS).

Each run also gets auto_record.yaml (what started it, what stopped it, the
nodes on the graph), metadata/processes.txt (the command line of every ROS
process, which names the launch files and params files that ran) and
recorder.log (the recorders' own output). Afterwards run_activity.py adds what
the car did to auto_record.yaml. Every decision also goes in auto_record.log
beside the runs, including the refusals that leave no run behind.
"""

from datetime import datetime
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.signals import SignalHandlerOptions
from rosbag2_interfaces.srv import Resume
from sensor_msgs.msg import Joy
from std_msgs.msg import String
import yaml

from race_diagnostics import auto_record as ar
from race_diagnostics.orphans import find_orphans
from racerbot_car.paths import data_dir
from race_diagnostics.recording import bag_record_command, DEFAULT_TOPICS


# `ros2 launch` signalled while still starting its processes can hang until it is
# killed (seen 2026-10-31), which would wrongly mark the recording killed. So it
# is left to get going first: until race_diag_node, its first process, has written
# events.jsonl, or this long. The bag is stopped at once either way.
LAUNCH_SETTLE_SEC = 8.0


def _full(namespace, name):
    return namespace.rstrip('/') + '/' + name


def _dir_bytes(path):
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def _ros_processes():
    """Command lines of the ROS processes running now: which launch files,
    executables and params files make up this run."""
    try:
        done = subprocess.run(['ps', '-eo', 'pid,ppid,lstart,args', '--sort=start_time'],
                              capture_output=True, text=True, timeout=5.0)
    except (OSError, subprocess.TimeoutExpired) as e:
        return f'<ps failed: {e}>\n'
    lines = done.stdout.splitlines()
    keep = [line for line in lines[1:]
            if '--ros-args' in line or 'ros2 ' in line or '/opt/ros/' in line]
    return '\n'.join(lines[:1] + keep) + '\n'


class AutoRecordNode(Node):

    def __init__(self):
        super().__init__('auto_record')
        p = self.declare_parameter
        p('output_directory', '')
        p('trigger_topics', ['/drive'])
        p('ignore_publishers', ['ackermann_mux'])
        p('other_recorders', ['race_diag_node', 'rosbag2_recorder*'])
        p('start_delay_sec', 3.0)
        p('stop_delay_sec', 3.0)
        p('poll_period_sec', 0.5)
        p('record_bag', True)
        p('live_topics', ['/scan', '/odom', '/drive', '/joy'])
        p('topics', '')
        p('min_free_gb', 2.0)
        p('stop_timeout_sec', 15.0)
        p('deadman_button', 4)
        p('joy_timeout_sec', 0.5)
        p('snapshot_release_sec', 1.0)
        p('snapshot_wait_after_stop_sec', 30.0)

        def value(name):
            return self.get_parameter(name).value

        self.output_directory = (Path(value('output_directory')).expanduser()
                                 if value('output_directory') else data_dir('racerbot_runs'))
        self.record_bag = bool(value('record_bag'))
        self.topics = str(value('topics')).split() or list(DEFAULT_TOPICS)
        self.live_topics = [topic for topic in value('live_topics') if topic in self.topics]
        self.min_free_gb = float(value('min_free_gb'))
        self.stop_timeout_sec = float(value('stop_timeout_sec'))
        self.deadman_button = int(value('deadman_button'))
        self.joy_timeout_sec = float(value('joy_timeout_sec'))
        self.snapshot_release_sec = float(value('snapshot_release_sec'))
        self.snapshot_wait_after_stop_sec = float(value('snapshot_wait_after_stop_sec'))
        self.joy_buttons = []
        self.joy_received_at = None
        self.snapshots = []
        self.shutting_down = False
        self.standby = False
        self.run_started = False
        self.resume_client = None
        self.system = None
        self.start_publisher = self.create_publisher(String, '/auto_record/start', QoSProfile(
            depth=1, reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_subscription(Joy, '/joy', self._joy, QoSProfile(
            depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.settings = ar.settings_from_params(
            {name: value(name) for name in (
                'trigger_topics', 'ignore_publishers', 'other_recorders',
                'start_delay_sec', 'stop_delay_sec')},
            own_nodes=ar.recorder_node_names(ar.AUTO_PREFIX))
        self.machine = ar.AutoRecorder(self.settings)

        self.bag_node = ar.recorder_node_names(ar.AUTO_PREFIX)[2]
        self.child = None          # the record_run.py launch process
        self.bag = None            # the `ros2 bag record` process
        self.child_started_at = None
        self.child_signal_due = False
        self.run_dir = None
        self.run_info = None
        self.started_at = None     # monotonic
        self.stopping_since = None
        self.stop_reason = None
        self.live_pending = False
        self._last_status = None

        self._cleanup_orphans()
        self.create_timer(float(value('poll_period_sec')), self._tick)
        self._note(
            'info',
            f"auto_record ready: records a run while any node publishes "
            f"{', '.join(self.settings.trigger_topics)} (manual /teleop is not "
            f"recorded), into {self.output_directory}/<time>/; never beside "
            f"another recorder ({', '.join(self.settings.other_recorders)}). "
            f"Turn off with bringup record:=false.")

    # -------------------------------------------------------------- notes

    def _note(self, level, text):
        """Log a decision, and keep it where it stays with the runs. Node logs
        are ~/.ros/log/python3_<pid>_*.log, and a refusal leaves no run to
        explain it. Losing this line must never cost a recording."""
        # Two lines: rclpy refuses one call site logging at two severities.
        if level == 'error':
            self.get_logger().error(text)
        else:
            self.get_logger().info(text)
        try:
            self.output_directory.mkdir(parents=True, exist_ok=True)
            with open(self.output_directory / 'auto_record.log', 'a') as log:
                now = datetime.now().astimezone().isoformat(timespec='seconds')
                log.write(f'{now} {level.upper()} {text}\n')
        except OSError as e:
            self.get_logger().warn(f'could not write auto_record.log: {e}')

    # ------------------------------------------------------------ graph

    def _joy(self, msg):
        self.joy_buttons = list(msg.buttons)
        self.joy_received_at = time.monotonic()
        # Observe every repress, including one entirely between two graph polls.
        if self._lb_held():
            for job in self.snapshots:
                job['policy'].released_since = None

    def _lb_held(self):
        return ar.deadman_held(time.monotonic(), self.joy_buttons, self.joy_received_at,
                               self.deadman_button, self.joy_timeout_sec)

    def _cleanup_orphans(self):
        try:
            result = subprocess.run(['ps', '-eo', 'pid,ppid,args'], capture_output=True,
                                    text=True, timeout=5.0, check=True)
            processes = []
            for line in result.stdout.splitlines()[1:]:
                pid, ppid, args = line.split(None, 2)
                processes.append((int(pid), int(ppid), args))
            pids = find_orphans(processes, {os.getpid()})
            for pid in pids:
                self._note('info', f'stopping orphan auto recorder pid {pid}: '
                           + next(args for p, _, args in processes if p == pid))
                try:
                    os.kill(pid, signal.SIGINT)
                except ProcessLookupError:
                    pass
            deadline = time.monotonic() + 10.0
            while pids and time.monotonic() < deadline:
                survivors = []
                for pid in pids:
                    try:
                        stat = Path(f'/proc/{pid}/stat').read_text()
                        if stat.rsplit(')', 1)[1].split()[0] != 'Z':
                            survivors.append(pid)
                    except OSError:
                        pass
                pids = survivors
                if pids:
                    time.sleep(0.1)
            for pid in pids:
                self._note('error', f'killing orphan auto recorder pid {pid} after 10 s')
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        except (OSError, ValueError, subprocess.SubprocessError) as e:
            self._note('error', f'could not clean up orphan recorders: {e}')

    def _graph(self):
        publishers = {
            topic: [_full(info.node_namespace, info.node_name)
                    for info in self.get_publishers_info_by_topic(topic)]
            for topic in self.settings.trigger_topics}
        nodes = [_full(namespace, name)
                 for name, namespace in self.get_node_names_and_namespaces()]
        return publishers, nodes

    def _tick(self):
        if self.shutting_down:
            return
        self._poll_snapshots()
        if self.stopping_since is not None:
            publishers, nodes = self._graph()
            self._log_status(ar.Decision('', self.machine.state, '',
                                         ar.algorithms(publishers, self.settings),
                                         ar.other_recorders(nodes, self.settings)))
            self._wait_for_stop(block=False)
            return
        publishers, nodes = self._graph()
        running = self._processes()
        alive = bool(running) and all(proc.poll() is None for proc in running)
        others = ar.other_recorders(nodes, self.settings)
        if ar.discard_standby(self.standby, self.run_started, others):
            self._stop('another recorder appeared during standby')
            return
        if self.standby and not alive:
            self._stop('standby process exited')
            return
        if self.run_dir is None:
            self.output_directory.mkdir(parents=True, exist_ok=True)
            free_gb = shutil.disk_usage(self.output_directory).free / 1e9
            if ar.prepare_standby(self.machine.state, False, others, self._lb_held(),
                                  free_gb, self.min_free_gb):
                self._prepare()
                return
        ready = (self.standby and alive
                 and '/' + self.settings.own_nodes[0] in nodes
                 and (not self.record_bag or self.resume_client.service_is_ready()))
        # A partially started standby must settle before activation. In particular,
        # do not fall through to a second cold start while its imports are running.
        if self.standby and not ready:
            return
        decision = self.machine.update(time.monotonic(), publishers, nodes,
                                       recording_alive=alive, lb_held=self._lb_held(),
                                       standby_alive=ready)
        if decision.action == 'start':
            self._start(decision, nodes)
        elif decision.action == 'stop':
            self._stop(decision.reason)
        else:
            self._log_status(decision)
            self._check_live()

    def _processes(self):
        return [proc for proc in (self.bag, self.child, self.system) if proc is not None]

    def _check_live(self):
        """Say when the bag has really subscribed, read off the graph, so the
        driver knows a press of LB from now on is in the recording."""
        if self.bag is None or self.live_pending is False or self.bag.poll() is not None:
            return
        # The graph can still list the last run's recorder, under the same name
        # on the same topics, for a moment after it exits. Only this run's bag
        # process creates this directory, as it opens the bag.
        if not (self.run_dir / 'bag').exists():
            return
        published = [t for t in self.live_topics if self.count_publishers(t) > 0]
        subscribed = [t for t in self.live_topics
                      if any(sub.node_name == self.bag_node
                             for sub in self.get_subscriptions_info_by_topic(t))]
        if not subscribed or ar.waiting_on(published, subscribed):
            return
        self.live_pending = False
        after = time.monotonic() - self.machine.algorithm_seen_at
        self.run_info['bag_live_sec'] = round(after, 1)
        self._write_info()
        self._note('info', f"recording live: {' '.join(sorted(subscribed))} are in the bag, "
                           f'{after:.1f} s after the algorithm appeared')

    def _log_status(self, decision):
        # Only what someone at the bringup terminal needs to know about: an
        # algorithm that is NOT being recorded, and why.
        if decision.algorithms and decision.recorders:
            status = ('deferring', tuple(decision.recorders))
            text = (f"{ar.describe(decision.algorithms)} is running, but "
                    f"another recorder is on the graph "
                    f"({', '.join(decision.recorders)}): not recording it")
        elif decision.state == ar.HELD:
            status = ('held',)
            text = ('the last recording failed; the next algorithm started '
                    'will be recorded')
        elif decision.reason == 'waiting for LB release to start recording':
            status = ('lb_wait',)
            text = decision.reason
        else:
            status = None
            text = None
        if status != self._last_status and text:
            self._note('info', text)
        self._last_status = status

    # ---------------------------------------------------------- recording

    def _new_run_dir(self):
        stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        run_dir = self.output_directory / stamp
        n = 2
        while run_dir.exists():
            run_dir = self.output_directory / f'{stamp}-{n}'
            n += 1
        run_dir.mkdir(parents=True)
        return run_dir

    def _write_info(self):
        try:
            (self.run_dir / 'auto_record.yaml').write_text(
                yaml.safe_dump(self.run_info, sort_keys=False))
        except OSError as e:
            self.get_logger().warn(f'could not write auto_record.yaml: {e}')

    def _spawn(self, cmd):
        # Request SIGINT on parent death. Startup orphan cleanup also handles
        # descendants that survive a previous bringup despite this signal.
        if shutil.which('setpriv'):
            cmd = ['setpriv', '--pdeathsig', 'INT', '--'] + cmd
        with open(self.run_dir / 'recorder.log', 'a') as log:
            # Its own session: bringup's Ctrl+C reaches this node, which
            # stops the recording in order, and a last-resort kill of the
            # group can never touch bringup.
            return subprocess.Popen(
                cmd, stdout=log, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True)

    def _prepare(self):
        if not ar.may_join_graph(self._lb_held()):
            return
        self.run_dir = self._new_run_dir()
        self.run_started = False
        self.standby = True
        self.started_at = time.monotonic()
        self.run_info = {
            'standby_prepared': datetime.now().astimezone().isoformat(timespec='seconds')}
        self._write_info()
        try:
            if self.record_bag:
                self.resume_client = self.create_client(Resume, f'/{self.bag_node}/resume')
                self.bag = self._spawn(bag_record_command(
                    self.run_dir / 'bag', self.bag_node, self.topics, start_paused=True))
            self.child = self._spawn([
                'ros2', 'launch', 'race_diagnostics', 'record_run.py',
                f'run_directory:={self.run_dir}', f'name_prefix:={ar.AUTO_PREFIX}',
                'record_bag:=false', 'system_log:=false', 'snapshot:=false', 'standby:=true'])
            self.child_started_at = time.monotonic()
            self._note('info', f'paused standby prepared -> {self.run_dir}')
        except OSError as e:
            self._note('error', f'could not prepare standby: {e}')
            self._stop('standby preparation failed')

    def _poll_snapshots(self, shutdown=False):
        for job in list(self.snapshots):
            proc = job.get('proc')
            if proc is not None:
                if proc.poll() is not None:
                    self.snapshots.remove(job)
                elif time.monotonic() - job['spawned_at'] >= self.stop_timeout_sec:
                    os.killpg(proc.pid, signal.SIGINT)
                    try:
                        proc.wait(timeout=1.0)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait(timeout=5.0)
                    self.snapshots.remove(job)
                continue
            action = job['policy'].update(time.monotonic(), self._lb_held(),
                                          stopped=job['stopped'], shutdown=shutdown)
            if not action:
                continue
            if action == 'start':
                # Gate again at the actual process boundary.
                if not ar.may_join_graph(self._lb_held()):
                    job['policy'].done = False
                    continue
                try:
                    with open(job['path'] / 'recorder.log', 'a') as log:
                        cmd = ['ros2', 'run', 'race_diagnostics', 'snapshot_params',
                               '--output', str(job['path'] / 'metadata' / 'params'),
                               '--watch', '0', '--ros-args', '-r',
                               '__node:=' + ar.recorder_node_names(ar.AUTO_PREFIX)[1]]
                        if shutil.which('setpriv'):
                            cmd = ['setpriv', '--pdeathsig', 'INT', '--'] + cmd
                        job['proc'] = subprocess.Popen(
                            cmd, stdout=log, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, start_new_session=True)
                    job['spawned_at'] = time.monotonic()
                    action = 'started after stop' if job['stopped'] else 'started'
                except OSError as e:
                    action = f'failed ({e})'
            self._note('info', f"{job['path']}: params_snapshot: {action}")
            path = job['path'] / 'auto_record.yaml'
            data = yaml.safe_load(path.read_text())
            data['params_snapshot'] = action
            path.write_text(yaml.safe_dump(data, sort_keys=False))
            if self.run_dir == job['path']:
                self.run_info['params_snapshot'] = action
            if job.get('proc') is None:
                self.snapshots.remove(job)

    def _start(self, decision, nodes):
        if self.standby:
            if self.record_bag:
                self.resume_client.call_async(Resume.Request())
            self.start_publisher.publish(String(data=str(self.run_dir)))
            self.run_started = True
            self.standby = False
            self.started_at = time.monotonic()
            self.live_pending = self.record_bag
            self.run_info.update({
                'started_by': 'auto_record (bringup); turn off with record:=false',
                'started': datetime.now().astimezone().isoformat(timespec='seconds'),
                'algorithms': decision.algorithms, 'nodes': sorted(nodes),
                'params_snapshot': 'pending'})
            self._write_info()
            meta = self.run_dir / 'metadata'
            meta.mkdir(exist_ok=True)
            (meta / 'processes.txt').write_text(_ros_processes())
            self.system = self._spawn([sys.executable, '-m', 'race_diagnostics.system_log',
                                       '--output', str(self.run_dir / 'system')])
            self.snapshots.append({'path': self.run_dir, 'stopped': False,
                                   'policy': ar.PendingSnapshot(
                                       self.snapshot_release_sec,
                                       self.snapshot_wait_after_stop_sec)})
            self._note('info', f'recording {ar.describe(decision.algorithms)} -> {self.run_dir}')
            return
        if not ar.may_join_graph(self._lb_held()):
            return
        # No warm recorder: prepare the same paused processes while released;
        # activate them on a later poll once they are ready.
        self.machine.state = ar.WAITING
        self.output_directory.mkdir(parents=True, exist_ok=True)
        free_gb = shutil.disk_usage(self.output_directory).free / 1e9
        if free_gb <= self.min_free_gb:
            self._note('error', f'not recording: only {free_gb:.1f} GB free. Delete old runs.')
            self.machine.state = ar.HELD
            return
        self._prepare()
        return

    def _stop(self, reason):
        self.stop_reason = reason
        for job in self.snapshots:
            if job['path'] == self.run_dir:
                job['stopped'] = True
                if job['policy'].stopped_at is None:
                    job['policy'].stopped_at = time.monotonic()
        if self.system is not None and self.system.poll() is None:
            self.system.send_signal(signal.SIGINT)
        processes = self._processes()
        if not processes:
            if self.run_dir is not None:
                self._finish(killed=False)
            return
        running = [proc for proc in processes if proc.poll() is None]
        if running:
            self._note('info', f'stopping the recording: {reason}')
            if self.bag is not None and self.bag.poll() is None:
                self.bag.send_signal(signal.SIGINT)
            self.child_signal_due = self.child is not None and self.child.poll() is None
            self.stopping_since = time.monotonic()
            self._signal_launch_when_settled()
        else:
            self._finish(killed=False)

    def _signal_launch_when_settled(self):
        if not self.child_signal_due:
            return
        settled = ((self.run_dir / 'events.jsonl').exists()
                   or time.monotonic() - self.child_started_at >= LAUNCH_SETTLE_SEC)
        if settled:
            self.child_signal_due = False
            if self.child.poll() is None:
                self.child.send_signal(signal.SIGINT)

    def _wait_for_stop(self, block):
        end = self.stopping_since + self.stop_timeout_sec
        self._signal_launch_when_settled()
        while block and time.monotonic() < end and any(
                proc.poll() is None for proc in self._processes()):
            time.sleep(0.1)
            self._signal_launch_when_settled()
        running = [proc for proc in self._processes() if proc.poll() is None]
        if running:
            if time.monotonic() - self.stopping_since < self.stop_timeout_sec:
                return
            self._note(
                'error',
                f'the recording did not stop within {self.stop_timeout_sec:.0f} s; '
                f'killing it (the bag may need `ros2 bag reindex`)')
            for proc in running:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait(timeout=5.0)
            self._finish(killed=True)
            return
        self._finish(killed=False)

    def _failures(self):
        """Every recording process that failed, since launch's own exit code
        stays 0 when one of its children dies."""
        try:
            failures = ar.recorder_failures(
                (self.run_dir / 'recorder.log').read_text(errors='replace'))
        except OSError:
            failures = []
        if self.bag is not None and self.bag.returncode != 0:
            failures.append({'process': 'ros2 bag record', 'exit_code': self.bag.returncode})
        return failures

    def _start_activity_report(self):
        """What the car did during the run, read from the closed bag. Detached
        and at the lowest priority: it must never compete with the next run."""
        if not (self.run_dir / 'bag' / 'metadata.yaml').exists():
            return
        try:
            with open(self.run_dir / 'recorder.log', 'a') as log:
                subprocess.Popen(
                    ['nice', '-n', '19', sys.executable, '-m',
                     'race_diagnostics.run_activity', str(self.run_dir)],
                    stdout=log, stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL, start_new_session=True)
        except OSError as e:
            self.get_logger().warn(f'could not start run_activity: {e}')

    def _finish(self, killed):
        # An already running finite snapshot belongs to this run too. Do not
        # announce it saved while that process is still using its directory.
        if any(job['path'] == self.run_dir and job.get('proc') is not None
               and job['proc'].poll() is None for job in self.snapshots):
            if self.stopping_since is None:
                self.stopping_since = time.monotonic()
            return
        if self.standby and not self.run_started:
            # This path is exclusively the directory allocated by _prepare,
            # never a path supplied by another recorder or a started run.
            shutil.rmtree(self.run_dir)
            self._reset_run()
            return
        duration = time.monotonic() - self.started_at
        bag_bytes = _dir_bytes(self.run_dir / 'bag')
        self.run_info.update({
            'stopped': datetime.now().astimezone().isoformat(timespec='seconds'),
            'stop_reason': self.stop_reason,
            'duration_sec': round(duration, 1),
            'recorder_exit_code': self.child.returncode if self.child else None,
            'bag_exit_code': self.bag.returncode if self.bag else None,
            'recorder_failures': self._failures(),
            'recorder_killed': killed,
            'bag_bytes': bag_bytes,
        })
        if ar.is_superseded(self.stop_reason, duration):
            self.run_info['superseded'] = True
        self._write_info()
        self._start_activity_report()
        self._note(
            'info',
            f'saved run {self.run_dir} ({duration:.0f} s, bag '
            f'{bag_bytes / 1e6:.0f} MB): {self.stop_reason}. Analyse with: '
            f'python3 ~/racerbot-ws/tools/run_analysis/analyze_run.py {self.run_dir}')
        self._reset_run()

    def _reset_run(self):
        if self.resume_client is not None:
            self.destroy_client(self.resume_client)
            self.resume_client = None
        self.standby = False
        self.run_started = False
        self.live_pending = False
        self.system = None
        self.child = None
        self.bag = None
        self.child_signal_due = False
        self.run_dir = None
        self.run_info = None
        self.stopping_since = None

    def shutdown(self, reason):
        """Stop any recording in order before the node exits."""
        self.shutting_down = True
        deadline = time.monotonic() + self.stop_timeout_sec
        self._poll_snapshots(shutdown=True)
        if self._processes() and self.stopping_since is None:
            self._stop(reason)
        if self._processes():
            self._wait_for_stop(block=True)
        while self.snapshots and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            self._poll_snapshots(shutdown=True)
        for job in self.snapshots:
            proc = job.get('proc')
            if proc is not None and proc.poll() is None:
                os.killpg(proc.pid, signal.SIGINT)
                try:
                    proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=5.0)
        if self.run_dir is not None and self.stopping_since is not None:
            self._finish(killed=False)


def _interrupt(signum, frame):
    raise KeyboardInterrupt


def main(args=None):
    # Signals are handled here rather than by rclpy, whose handler shuts the
    # context down at once: the node must still be able to log while it
    # closes a recording in progress.
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    signal.signal(signal.SIGTERM, _interrupt)
    node = AutoRecordNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        # Bringup's launch follows its SIGINT with SIGTERM; neither may cut
        # short closing the bag. A SIGKILL still can, and then setpriv's
        # death signal closes it.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        node.shutdown('bringup shut down')
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
