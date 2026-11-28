"""The terminal menu. Step 1: watch only.

Start it on a car with tools/console/racerbot (it sources ROS and keeps the
menu in a tmux session). Screens: 1 Home, 2 Drive, 3 Runs, 4 Check,
5 Learn, 6 Logs, 7 ROS, 8 Workspace. Nothing in this step can start a
driving layer; see the design doc linked in docs/console.md.
"""
from __future__ import annotations

import os
from datetime import date
from pathlib import Path

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import ContentSwitcher, DataTable, Footer, RichLog, Static

from . import runs as runs_mod
from . import status
from .backend import Backend, Snapshot
from .jobs import JobRefused, JobRunner

STYLE = {status.Level.OK: 'green', status.Level.WARN: 'yellow',
         status.Level.FAIL: 'bold red', status.Level.OFF: 'dim'}
DOT = {status.Level.OK: '●', status.Level.WARN: '●', status.Level.FAIL: '●',
       status.Level.OFF: '○'}

SCREENS = ('home', 'drive', 'runs', 'check', 'learn', 'logs', 'ros', 'workspace')
TITLES = ('Home', 'Drive', 'Runs', 'Check', 'Learn', 'Logs', 'ROS', 'Workspace')


def shown(s: status.Shown, *, dot: bool = True) -> Text:
    return Text(f'{DOT[s.level]} {s.text}' if dot else s.text, style=STYLE[s.level])


def hardware_shown(layers) -> status.Shown:
    if layers.hardware == 'up':
        return status.Shown(status.Level.OK, 'HW up')
    if layers.hardware == 'partial':
        return status.Shown(status.Level.WARN, 'HW partial: ' + ', '.join(layers.missing))
    return status.Shown(status.Level.OFF, 'HW off')


def driving_shown(layers) -> status.Shown:
    if layers.simulator and layers.hardware != 'off':
        return status.Shown(status.Level.FAIL, 'SIMULATOR beside real drivers')
    if not layers.driving:
        return status.Shown(status.Level.OFF, 'DRIVING: none')
    if len(layers.driving) > 1:
        return status.Shown(status.Level.FAIL, 'DRIVING: ' + ' + '.join(layers.driving)
                            + ' (only one should run)')
    return status.Shown(status.Level.OK, 'DRIVING: ' + layers.driving[0])


def status_line(snap: Snapshot) -> Text:
    live = snap.live
    car = snap.car_id or 'car ?'
    parts = [Text(f' {car} ', style='bold reverse'),
             shown(hardware_shown(snap.layers)),
             shown(driving_shown(snap.layers), dot=False),
             Text('LB ') + shown(status.lb(live.buttons, live.joy_t, live.now)),
             shown(status.battery(live.volts if status.is_fresh(live.volts_t, live.now)
                                  else None))]
    line = Text(' │ ', style='dim').join(parts)
    return line


def car_panel(snap: Snapshot) -> Text:
    live, j = snap.live, snap.jetson
    from racerbot_doctor import car as dc
    out = Text()

    def row(*items):
        for i, item in enumerate(items):
            if i:
                out.append('   ')
            out.append_text(item)
        out.append('\n')

    volts = live.volts if status.is_fresh(live.volts_t, live.now) else None
    row(Text('Battery ', style='bold'), shown(status.battery(volts)))
    row(Text('Speed   ', style='bold'), shown(status.speed(live.speed, live.speed_t, live.now)))
    row(Text('Sensors ', style='bold'), shown(status.rate(live.scan_hz, dc.SCAN_MIN_HZ, 'LiDAR')),
        shown(status.rate(live.core_hz, dc.VESC_CORE_MIN_HZ, 'VESC')))
    row(Text('        '), shown(status.imu(live.imu_state)))
    row(Text('Control ', style='bold'), shown(status.joy_publishers(live.joy_publishers)),
        shown(status.rate(live.drive_hz, 1.0, '/drive')))
    row(Text('        '), shown(status.Shown(
        status.Level.OK if snap.layers.dashboard else status.Level.OFF, 'dashboard')))
    out.append('\n')
    cpu = '-' if j.cpu is None else f'{j.cpu:.0f}%'
    gpu = '-' if j.gpu is None else f'{j.gpu:.0f}%'
    mem = '-' if j.mem is None else f'{j.mem[0]:.1f}/{j.mem[1]:.1f} GB'
    temp = '-' if j.temp is None else f'{j.temp:.0f} °C'
    disk = '-' if j.disk_free is None else f'{j.disk_free:.0f} GB free'
    hot = j.temp is not None and j.temp >= 80
    out.append(f'Jetson  CPU {cpu}  GPU {gpu}  {temp}\n', style='bold red' if hot else 'bold')
    out.append(f'        RAM {mem}  disk {disk}\n', style='bold')
    if live.error:
        out.append(f'\n{live.error}\n', style='yellow')
    if snap.car_error:
        out.append(f'\n{snap.car_error}\n', style='yellow')
    return out


DRIVE_TEXT = """[b]Starting the car from the menu comes in the next step[/b], with the Get ready
checks, fast keys and presets. Until then, the usual commands (docs/launchstart.md):

  ros2 launch f1tenth_stack bringup_launch.py            hardware, never moves the car
  ros2 launch f1tenth_stack teleop_launch.py             manual, hold LB
  ros2 launch gap_follow gap_follow_launch.py max_speed:=1.0
  ros2 run racerbot_launch map_course
  ros2 launch racerbot_launch race_launch.py run_dir:=... profile:=raceline_floor_test.csv

The status bar above already shows which layer is running and whether LB is held."""

LEARN_TEXT = """[b]Learn: let the car learn its own physics[/b]

  [dim]Explorer laps          not built yet (waiting for car_explorer)[/dim]
  [dim]Race-day tune          not built yet (waiting for the floor refit)[/dim]
  [dim]Review fitted values   not built yet[/dim]

Each slot switches on when its part of the learning plan is merged."""

LATER_TEXT = {
    'ros': '[b]ROS[/b]: graph, live parameters, bags, restart one node and TF arrive in step 4.',
    'workspace': '[b]Workspace[/b]: git, build, tests, the A/B ladder and a shell arrive in step 4.',
}


def doctor_line_style(line: str) -> str:
    stripped = line.strip()
    for tag, style in (('FAIL', 'bold red'), ('WARN', 'yellow'), ('PASS', 'green'),
                       ('INFO', 'cyan'), ('SKIP', 'dim')):
        if stripped.startswith(tag):
            return style
    if ' FAIL' in line and line[:1].isdigit():
        return 'bold red' if not line.startswith('0 FAIL') else 'bold green'
    return ''


class Console(App):
    TITLE = 'racerbot'
    CSS = """
    #status { height: 1; background: $panel; }
    ContentSwitcher { height: 1fr; }
    .pane { height: 1fr; }
    #car { width: 1fr; border: round $secondary; padding: 0 1; }
    #session { width: 30; border: round $secondary; padding: 0 1; }
    #runs-table { height: 1fr; }
    RichLog { height: 1fr; border: round $secondary; }
    #run-hint { height: 1; color: $text-muted; }
    """

    BINDINGS = [
        *[Binding(str(i + 1), f'show("{name}")', TITLES[i], show=(i < 6),
                  key_display=str(i + 1)) for i, name in enumerate(SCREENS)],
        Binding('c', 'check', 'Check'),
        Binding('a', 'report("summary")', 'Summary', show=False),
        Binding('A', 'report("slam")', 'SLAM report', show=False),
        Binding('I', 'report("imu")', 'IMU report', show=False),
        Binding('f5', 'reload_runs', 'Reload runs', show=False),
        Binding('question_mark', 'help', 'Help'),
        Binding('q', 'quit_menu', 'Quit'),
        Binding('f10', 'quit_menu', 'Quit', show=False),
    ]

    def __init__(self, backend, *, log_dir: Path | None = None, runner: JobRunner | None = None,
                 refresh_sec: float = 0.25):
        super().__init__()
        self.backend = backend
        self.workspace = Path(backend.workspace)
        self.log_dir = log_dir or Path.home() / '.local/state/racerbot/console'
        self.runner = runner or JobRunner(self.workspace, self.log_dir)
        self.refresh_sec = refresh_sec
        self.runs: list[runs_mod.Run] = []
        self._shown_lines: dict[int, int] = {}
        self._quit_armed = False

    def compose(self) -> ComposeResult:
        yield Static(id='status')
        with ContentSwitcher(initial='home'):
            with Horizontal(id='home', classes='pane'):
                yield Static(id='car')
                yield Static(id='session')
            yield Static(DRIVE_TEXT, id='drive', classes='pane')
            with Vertical(id='runs', classes='pane'):
                yield DataTable(id='runs-table', cursor_type='row', zebra_stripes=True)
                yield Static('a summary   A SLAM report   I IMU report   F5 reload',
                             id='run-hint')
            yield RichLog(id='check', classes='pane', highlight=False, markup=False)
            yield Static(LEARN_TEXT, id='learn', classes='pane')
            yield RichLog(id='logs', classes='pane', highlight=False, markup=False)
            yield Static(LATER_TEXT['ros'], id='ros', classes='pane')
            yield Static(LATER_TEXT['workspace'], id='workspace', classes='pane')
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one('#runs-table', DataTable)
        table.add_columns('When', 'Kind', 'What', 'Moved', 'Length', 'Size', 'In git')
        self.query_one('#check', RichLog).write(
            Text('Press c to run the hardware check (racerbot_doctor). '
                 'It is read-only and safe any time.'))
        self.backend.start()
        self.action_reload_runs()
        self.tick()
        self._timer = self.set_interval(self.refresh_sec, self.tick)

    def on_unmount(self) -> None:
        if getattr(self, '_timer', None) is not None:
            self._timer.stop()
        self.backend.stop()

    # ---- refresh ----------------------------------------------------------
    def tick(self) -> None:
        if not self.is_running or not self.screen.query('#status'):
            return                          # shutting down: the widgets are gone
        snap = self.backend.snapshot()
        self.query_one('#status', Static).update(status_line(snap))
        self.query_one('#car', Static).update(car_panel(snap))
        self.query_one('#session', Static).update(self.session_text())
        self.pump_jobs()

    def session_text(self) -> Text:
        today = [r for r in self.runs if r.started and r.started.date() == date.today()]
        unsaved = [r for r in self.runs if r.in_git is False]
        out = Text('Session\n', style='bold')
        out.append(f'Runs today   {len(today)}\n')
        out.append(f'Not in git   {len(unsaved)}  '
                   f'{runs_mod.human_bytes(sum(r.size_bytes or 0 for r in unsaved))}\n',
                   style='yellow' if unsaved else '')
        if self.runs:
            r = self.runs[0]
            when = r.started.strftime('%b %d %H:%M') if r.started else r.path.name
            out.append(f'Last         {when} {r.what}\n')
        out.append('\nc hardware check   3 runs\n', style='dim')
        return out

    def pump_jobs(self) -> None:
        """Copy new output of every job into its log view."""
        logs = self.query_one('#logs', RichLog)
        check = self.query_one('#check', RichLog)
        for job in self.runner.jobs:
            lines = job.snapshot()
            done = self._shown_lines.get(id(job), 0)
            if done > len(lines):           # ring buffer wrapped
                done = 0
            for line in lines[done:]:
                styled = Text(line, style=doctor_line_style(line) if job.spec.key == 'doctor'
                              else '')
                logs.write(Text(f'[{job.spec.key}] ', style='dim') + styled)
                if job.spec.key == 'doctor':
                    check.write(styled)
            self._shown_lines[id(job)] = len(lines)

    # ---- actions ----------------------------------------------------------
    def action_show(self, name: str) -> None:
        self.query_one(ContentSwitcher).current = name
        self._quit_armed = False

    def action_check(self) -> None:
        if any(j.running and j.spec.key == 'doctor' for j in self.runner.jobs):
            self.notify('The hardware check is already running.')
            return
        self.action_show('check')
        check = self.query_one('#check', RichLog)
        check.clear()
        check.write(Text('$ ros2 run racerbot_doctor doctor --no-color', style='dim'))
        self.runner.start('doctor')

    def selected_run(self) -> runs_mod.Run | None:
        table = self.query_one('#runs-table', DataTable)
        if not self.runs or table.cursor_row is None:
            return None
        row = table.cursor_row
        return self.runs[row] if 0 <= row < len(self.runs) else None

    def action_report(self, key: str) -> None:
        if self.query_one(ContentSwitcher).current != 'runs':
            return
        run = self.selected_run()
        if run is None:
            self.notify('Pick a run first.', severity='warning')
            return
        try:
            self.runner.start(key, run.path)
        except JobRefused as exc:
            self.notify(str(exc), severity='error')
            return
        self.notify(f'Started {key} on {run.path.name}; output in 6 Logs.')

    def action_reload_runs(self) -> None:
        car = getattr(self.backend, 'car_id', None) or os.environ.get('RACERBOT_CAR')
        table = self.query_one('#runs-table', DataTable)
        table.clear()
        if not car:
            self.runs = []
            return
        tracked = runs_mod.tracked_folders(self.workspace, car)
        self.runs = runs_mod.list_runs(self.workspace, car, tracked=tracked)
        for r in self.runs:
            moved = {True: 'yes', False: 'no', None: '-'}[r.moved]
            git = {True: 'yes', False: 'no', None: '?'}[r.in_git]
            table.add_row(r.started.strftime('%b %d %H:%M') if r.started else r.path.name,
                          r.kind, r.what, moved, runs_mod.human_duration(r.duration_sec),
                          runs_mod.human_bytes(r.size_bytes), git)

    def action_help(self) -> None:
        self.notify('1-8 screens · c hardware check · on Runs: a summary, A SLAM, I IMU, '
                    'F5 reload · q quit. Nothing here can move the car.', timeout=8)

    def action_quit_menu(self) -> None:
        running = [j for j in self.runner.jobs if j.running]
        if running and not self._quit_armed:
            self._quit_armed = True
            self.notify(f'{len(running)} job(s) still running. Press q again to stop them '
                        'and quit.', severity='warning')
            return
        self.runner.stop_all()
        self.exit()


def main(argv=None) -> None:
    from racerbot_car import paths
    workspace = paths.workspace_root()
    Console(Backend(workspace)).run()


if __name__ == '__main__':
    main()
