"""sysinfo.py against fake /proc and /sys trees. Oracles: proc(5) field
order for /proc/stat and /proc/meminfo; the Jetson GPU load file is tenths
of a percent; offline thermal zones read -256000."""
import pytest

from racerbot_console import sysinfo


def write(root, rel, text):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def test_cpu_percent_from_two_samples(tmp_path):
    meter = sysinfo.CpuMeter(str(tmp_path))
    # user nice system idle iowait irq softirq steal
    write(tmp_path, 'stat', 'cpu  100 0 100 700 100 0 0 0\ncpu0 1 1 1 1\n')
    assert meter.percent() is None                       # first sample has no delta
    write(tmp_path, 'stat', 'cpu  160 0 140 1000 200 0 0 0\n')
    # busy +100 (user +60, system +40) of total +500; idle +300 and iowait +100
    # both count as not busy -> 20 %
    assert meter.percent() == pytest.approx(20.0, abs=1e-9)


@pytest.mark.parametrize('text', [None, '', 'intr 1 2\n', 'cpu a b c d e\n', 'cpu 1 2\n'])
def test_cpu_times_malformed(text):
    assert sysinfo.parse_cpu_times(text) is None


def test_cpu_counter_going_backwards_is_not_a_reading(tmp_path):
    meter = sysinfo.CpuMeter(str(tmp_path))
    write(tmp_path, 'stat', 'cpu  500 0 0 500 0 0 0 0\n')
    meter.percent()
    write(tmp_path, 'stat', 'cpu  100 0 0 1000 0 0 0 0\n')   # total +100, busy -400
    assert meter.percent() is None


def test_meminfo_used_is_total_minus_available():
    text = 'MemTotal:  7400000 kB\nMemFree: 100000 kB\nMemAvailable: 5300000 kB\n'
    used, total = sysinfo.parse_meminfo(text)
    assert total == pytest.approx(7.4, abs=1e-9) and used == pytest.approx(2.1, abs=1e-9)


@pytest.mark.parametrize('text', [None, '', 'MemTotal: 10 kB\n', 'MemTotal: x kB\nMemAvailable: 1 kB\n'])
def test_meminfo_incomplete(text):
    assert sysinfo.parse_meminfo(text) is None


def test_hottest_zone_skips_offline_and_garbage(tmp_path):
    write(tmp_path, 'class/thermal/thermal_zone0/temp', '45500\n')
    write(tmp_path, 'class/thermal/thermal_zone1/temp', '-256000\n')   # offline zone
    write(tmp_path, 'class/thermal/thermal_zone2/temp', 'n/a\n')
    write(tmp_path, 'class/thermal/thermal_zone3/temp', '51250\n')
    assert sysinfo.hottest_c(str(tmp_path)) == pytest.approx(51.25, abs=1e-9)


@pytest.mark.parametrize('raw', ['-256000', '200000', '-41000'])
def test_a_zone_outside_minus_40_to_150_c_is_not_a_temperature(tmp_path, raw):
    write(tmp_path, 'class/thermal/thermal_zone0/temp', raw + '\n')
    assert sysinfo.hottest_c(str(tmp_path)) is None


def test_no_thermal_zones(tmp_path):
    assert sysinfo.hottest_c(str(tmp_path)) is None


@pytest.mark.parametrize('raw,want', [('0', 0.0), ('375', 37.5), ('1000', 100.0),
                                      ('1001', None), ('-1', None), ('x', None)])
def test_gpu_load_is_tenths_of_a_percent(tmp_path, raw, want):
    write(tmp_path, sysinfo.GPU_LOAD_PATHS[0], raw + '\n')
    assert sysinfo.gpu_percent(str(tmp_path)) == want


def test_gpu_falls_back_to_the_next_path(tmp_path):
    write(tmp_path, sysinfo.GPU_LOAD_PATHS[1], '200\n')
    assert sysinfo.gpu_percent(str(tmp_path)) == 20.0
    assert sysinfo.gpu_percent(str(tmp_path / 'none')) is None


def test_disk_free_of_a_missing_path():
    assert sysinfo.disk_free_gb('/nonexistent/racerbot') is None
