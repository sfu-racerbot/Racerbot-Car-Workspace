"""The screens, driven headless with Textual's test pilot at 80x24, the
smallest browser terminal the menu must work in. Readings come from a
FakeBackend, so the status bar's rules are checked end to end: what /joy,
the battery and the process list say must be what the person sees.

Needs Textual, which lives in the console's own virtual environment
(docs/console.md#install); run these tests with that environment's python.
"""
import asyncio
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip('textual', reason='Textual is installed by tools/console/install.sh '
                    '(docs/console.md#install); runs once it is installed')

from racerbot_console.app import Console  # noqa: E402
from racerbot_console.backend import FakeBackend, Snapshot  # noqa: E402
from racerbot_console.jobs import JobRunner  # noqa: E402
from racerbot_console.layers import Layers  # noqa: E402
from racerbot_console.ros_reader import Live  # noqa: E402
from racerbot_console.sysinfo import Jetson  # noqa: E402

LB_UP = (0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0)
LB_DOWN = (0,) * 11
HW = Layers('up', (), (), False, True)


def snap(**live):
    base = dict(now=100.0, volts=11.7, volts_t=99.9, buttons=LB_DOWN, joy_t=99.95,
                joy_publishers=1, scan_hz=40.0, core_hz=50.0)
    base.update(live)
    return Snapshot('rb2', None, Live(**base), HW, Jetson(12.0, 3.0, (2.1, 7.4), 41.0, 38.0))


def run(app_snap, steps, *, tmp_path, popen=subprocess.Popen, layers=None):
    if layers is not None:
        app_snap = replace(app_snap, layers=layers)
    backend = FakeBackend(tmp_path, app_snap)
    backend.car_id = 'rb2'
    runner = JobRunner(tmp_path, tmp_path / 'logs', popen=popen)
    app = Console(backend, log_dir=tmp_path / 'logs', runner=runner, refresh_sec=0.05)

    async def go():
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            return await steps(app, pilot)
    return asyncio.run(go())


def text_of(app, selector):
    return str(app.query_one(selector).render())


def test_status_bar_shows_lb_held_only_while_joy_says_so(tmp_path):
    async def steps(app, pilot):
        return text_of(app, '#status')
    assert 'LB ● HELD' in run(snap(buttons=LB_UP), steps, tmp_path=tmp_path)
    released = run(snap(buttons=LB_DOWN), steps, tmp_path=tmp_path)
    assert 'released' in released and 'HELD' not in released
    stale = run(snap(buttons=LB_UP, joy_t=90.0), steps, tmp_path=tmp_path)   # /joy 10 s old
    assert 'controller silent' in stale and 'HELD' not in stale
    never = run(snap(buttons=None, joy_t=None), steps, tmp_path=tmp_path)
    assert 'no controller' in never and 'HELD' not in never


def test_status_bar_battery_hides_a_stale_reading(tmp_path):
    async def steps(app, pilot):
        return text_of(app, '#status')
    assert '11.70 V' in run(snap(), steps, tmp_path=tmp_path)
    assert 'no reading' in run(snap(volts_t=50.0), steps, tmp_path=tmp_path)


def test_two_driving_layers_are_flagged(tmp_path):
    async def steps(app, pilot):
        return text_of(app, '#status')
    out = run(snap(), steps, tmp_path=tmp_path,
              layers=Layers('up', (), ('gap follow', 'manual'), False, True))
    assert 'only one should run' in out


def test_simulator_beside_real_drivers_is_flagged(tmp_path):
    async def steps(app, pilot):
        return text_of(app, '#status')
    out = run(snap(), steps, tmp_path=tmp_path, layers=Layers('up', (), (), True, False))
    assert 'SIMULATOR beside real drivers' in out


def test_two_joy_publishers_show_on_home(tmp_path):
    async def steps(app, pilot):
        return text_of(app, '#car')
    assert 'cars share a network' in run(snap(joy_publishers=2), steps, tmp_path=tmp_path)


def test_number_keys_switch_screens(tmp_path):
    async def steps(app, pilot):
        seen = []
        for key in '12345678':
            await pilot.press(key)
            await pilot.pause()
            seen.append(app.query_one('ContentSwitcher').current)
        return seen
    assert run(snap(), steps, tmp_path=tmp_path) == [
        'home', 'drive', 'runs', 'check', 'learn', 'logs', 'ros', 'workspace']


def fake(script):
    def popen(argv, **kw):
        return subprocess.Popen([sys.executable, '-c', script], **kw)
    return popen


def test_c_runs_the_doctor_and_shows_its_output(tmp_path):
    async def steps(app, pilot):
        await pilot.press('c')
        for _ in range(100):
            await pilot.pause(0.05)
            if app.runner.jobs and not app.runner.jobs[0].running:
                break
        await pilot.pause(0.2)
        log = app.query_one('#check')
        return app.query_one('ContentSwitcher').current, [str(l.text) for l in log.lines]
    screen, lines = run(snap(), steps, tmp_path=tmp_path,
                        popen=fake('print("  FAIL  usb  no device"); print("1 FAIL, 0 WARN")'))
    assert screen == 'check'
    assert any('FAIL  usb  no device' in l for l in lines)
    assert any(l.startswith('1 FAIL') for l in lines)


def test_quit_asks_again_while_a_job_runs(tmp_path):
    async def steps(app, pilot):
        await pilot.press('c')
        await pilot.pause(0.2)
        await pilot.press('q')
        await pilot.pause()
        first = app.is_running and app.runner.jobs[0].running
        await pilot.press('q')
        await pilot.pause(0.5)
        return first, app.runner.jobs[0].running
    first, after = run(snap(), steps, tmp_path=tmp_path, popen=fake('import time; time.sleep(30)'))
    assert first is True and after is False


def test_runs_screen_lists_the_cars_runs(tmp_path):
    d = tmp_path / 'data/rb2/racerbot_runs/20261111-154204'
    d.mkdir(parents=True)
    (d / 'auto_record.yaml').write_text("algorithms: {/drive: [/gap_follow_node]}\n"
                                        "activity: {moved: true}\n")

    async def steps(app, pilot):
        await pilot.press('3')
        await pilot.pause()
        table = app.query_one('#runs-table')
        return table.row_count, [str(c) for c in table.get_row_at(0)]
    count, row = run(snap(), steps, tmp_path=tmp_path)
    assert count == 1 and row[2] == 'gap_follow' and row[3] == 'yes'


def test_a_report_needs_a_selected_run(tmp_path):
    async def steps(app, pilot):
        await pilot.press('3')
        await pilot.pause()
        await pilot.press('a')
        await pilot.pause()
        return len(app.runner.jobs)
    assert run(snap(), steps, tmp_path=tmp_path) == 0
