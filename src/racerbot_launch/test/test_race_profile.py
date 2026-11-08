"""Launch description only: no ROS nodes start. Oracle: profile selection contract."""
import importlib.util
from pathlib import Path

from launch import LaunchContext
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch_ros.actions import Node
import pytest
import yaml


def _launch(tmp_path, profile=None):
    path = Path(__file__).resolve().parents[1] / 'launch/race_launch.py'
    spec = importlib.util.spec_from_file_location('race_profile_launch', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Filesystem boundary: a CPU particle-filter base avoids depending on a
    # host CUDA extension. The real launch/GPU configuration check still runs.
    pf = tmp_path / 'particle_filter'
    (pf / 'config').mkdir(parents=True, exist_ok=True)
    (pf / 'config/localize.yaml').write_text(
        'particle_filter:\n  ros__parameters:\n    range_method: rm\n')
    real_share = module.get_package_share_directory
    module.get_package_share_directory = lambda name: str(pf) if name == 'particle_filter' else real_share(name)
    ld = module.generate_launch_description()
    context = LaunchContext()
    for action in ld.entities:
        if isinstance(action, DeclareLaunchArgument):
            action.execute(context)
    context.launch_configurations.update(run_dir=str(tmp_path), seed_start='false')
    if profile is not None:
        context.launch_configurations['profile'] = profile
    return next(action for action in ld.entities if isinstance(action, OpaqueFunction)).execute(context), context


@pytest.mark.parametrize('profile', [None, 'raceline_floor_test.csv'])
def test_selected_profile_reaches_the_controller(tmp_path, profile):
    (tmp_path / 'map.yaml').touch()
    for name in ('raceline_profiled.csv', 'raceline_floor_test.csv'):
        (tmp_path / name).write_text('x,y,speed\n0,0,1\n')
    actions, context = _launch(tmp_path, profile)
    node, = [a for a in actions if isinstance(a, Node)]
    params = {}
    for entry in node._Node__parameters:
        if isinstance(entry, dict):
            for key, value in entry.items():
                name = ''.join(s.perform(context) for s in key)
                params[name] = (yaml.safe_load(''.join(s.perform(context) for s in value))
                                if isinstance(value, tuple) else value)
    # oracle: spec, default uses full profile; explicit floor-test selects only the capped CSV.
    assert params['waypoints_file'] == str(tmp_path / (profile or 'raceline_profiled.csv'))


def test_missing_selected_profile_is_named_even_when_default_exists(tmp_path):
    (tmp_path / 'map.yaml').touch()
    (tmp_path / 'raceline_profiled.csv').touch()
    # oracle: spec, refuse the missing file instead of falling back to a faster one.
    with pytest.raises(RuntimeError, match='missing .*raceline_floor_test.csv'):
        _launch(tmp_path, 'raceline_floor_test.csv')
