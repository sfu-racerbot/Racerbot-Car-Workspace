"""Oracles: warm-recorder brief rules 1, 4–8 and the stop/shutdown ruling."""
from race_diagnostics import auto_record as ar


def test_standby_requires_idle_space_release_and_no_other_recorder():
    eligible = (ar.IDLE, False, [], False, 3.0, 2.0)
    assert ar.prepare_standby(*eligible)
    for index, denied in [(0, ar.RECORDING), (0, ar.HELD), (1, True),
                          (2, ['/race_diag_node']), (3, True), (4, 2.0), (4, 1.9)]:
        args = list(eligible)
        args[index] = denied
        assert not ar.prepare_standby(*args), args
    assert not ar.may_join_graph(True)
    assert ar.may_join_graph(False)


def test_cold_start_waits_for_release_but_warm_start_can_resume_held():
    machine = ar.AutoRecorder(ar.Settings(('/test_only/drive',), start_delay_sec=0))
    pubs = {'/test_only/drive': ['/algorithm']}
    blocked = machine.update(0, pubs, [], lb_held=True)
    assert blocked.action == ''
    assert blocked.reason == 'waiting for LB release to start recording'
    assert machine.update(1, pubs, [], lb_held=False).action == 'start'
    warm = ar.AutoRecorder(machine.settings)
    assert warm.update(0, pubs, [], lb_held=True, standby_alive=True).action == 'start'


def test_snapshot_requires_continuous_release_including_boundary():
    pending = ar.PendingSnapshot(1.0)
    assert pending.update(0, False) == ''
    assert pending.update(0.999, False) == ''
    assert pending.update(1, True) == ''  # repress resets the whole interval
    assert pending.update(2, False) == ''
    assert pending.update(2.999, False) == ''
    assert pending.update(3, False) == 'start'
    assert pending.update(3.001, False) == ''  # exactly once
    past = ar.PendingSnapshot(1.0)
    assert past.update(4, False) == ''
    assert past.update(5.001, False) == 'start'


def test_joy_absent_stale_short_and_non_one_are_released():
    buttons = [0, 0, 0, 0, 1]
    assert ar.deadman_held(1.5, buttons, 1.0)
    assert not ar.deadman_held(1.501, buttons, 1.0)
    assert not ar.deadman_held(1, buttons, None)
    assert not ar.deadman_held(1, [], 1)
    assert not ar.deadman_held(1, [1], 1, button=4)
    assert not ar.deadman_held(1, buttons, 1, button=-1)
    assert not ar.deadman_held(1, [0, 0, 0, 0, 2], 1)
    assert ar.deadman_held(1, [1], 1, button=0)


def test_only_never_started_standby_can_be_discarded():
    assert ar.discard_standby(True, False, shutdown=True)
    assert ar.discard_standby(True, False, ['/race_diag_node'])
    assert not ar.discard_standby(True, True, shutdown=True)
    assert not ar.discard_standby(False, False, shutdown=True)
    assert not ar.discard_standby(True, False)


def test_snapshot_stop_wait_expires_without_bypassing_release():
    pending = ar.PendingSnapshot(1.0, 30.0)
    assert pending.update(10, True, stopped=True) == ''
    assert pending.update(39.999, True, stopped=True) == ''
    assert pending.update(40, True, stopped=True) == 'skipped (LB held after stop)'
    assert pending.update(41, False) == ''
    released = ar.PendingSnapshot(1.0, 30.0)
    assert released.update(10, True, stopped=True) == ''
    assert released.update(11, False, stopped=True) == ''
    assert released.update(12, False, stopped=True) == 'start'


def test_snapshot_shutdown_held_skips_and_released_still_debounces():
    held = ar.PendingSnapshot()
    assert held.update(1, True, shutdown=True) == 'skipped (LB held at shutdown)'
    assert held.update(2, False, shutdown=True) == ''
    released = ar.PendingSnapshot()
    assert released.update(0, False, shutdown=True) == ''
    assert released.update(1, False, shutdown=True) == 'start'
