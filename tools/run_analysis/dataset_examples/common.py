"""
common.py -- what the five dataset examples share.

Read a run's bag (no ROS needed: bagread.py falls back to rosbags), put
signals on one time grid, find when the car was really driving, find jumps in
a pose estimate, and save plots and numbers. Nothing here writes into data/.

Times are bag receive times in seconds, as everywhere in race_diagnostics and
tools/run_analysis. Units are SI; IMU topics are the calibrated car-axis ones
(x forward, y left, z up).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
for _path in (ROOT / "tools", ROOT / "src" / "race_diagnostics", ROOT / "src" / "pure_pursuit"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from run_analysis.bagread import messages  # noqa: E402
from run_analysis.grip_truth import WHEELBASE  # noqa: E402,F401  0.36 m, docs/hardware-reference.md
from run_analysis.intent_report import parse as parse_intent  # noqa: E402
from race_diagnostics.dataset_classifier import load_rules  # noqa: E402
from race_diagnostics.dataset_classifier.metrics import lb_windows  # noqa: E402

LB_BUTTON = 4            # the F710's LB, the deadman (docs/architecture.md)
FS = 50.0                # Hz, the common grid: the VESC IMU's own rate (yaw_tracking.FS)
MAX_GAP_S = 0.1          # s, a grid point farther than this from a sample is NaN
JUMP_M = 0.3             # m, a pose step this much beyond what the wheels allow is a jump (slam_metrics.JUMP_M)
JUMP_RAD = math.radians(15.0)  # a heading step this much beyond what the gyro allows is a jump
JUMP_PAD_S = 0.5         # s, samples this close to a jump are left out
DEFAULT_OUT = Path("~/.cache/racerbot/dataset_examples").expanduser()

# Columns of each kind of message, as loaded.
COLUMNS = {
    "odometry": ("t", "x", "y", "yaw", "v", "wz"),
    "pose": ("t", "x", "y", "yaw"),
    "imu": ("t", "wx", "wy", "wz", "ax", "ay", "az"),
    "ackermann": ("t", "steer", "speed"),
    "joint": ("t", "position"),
    "joy": ("t", "lb"),
}


def _rules_measurement():
    return load_rules()["measurement"]


MOVING_SPEED = float(_rules_measurement()["moving_speed_mps"])   # 0.3 m/s, as the grades use
LB_MIN_HOLD_S = float(_rules_measurement()["lb_min_hold_s"])      # 0.5 s, as the grades use


# --------------------------------------------------------------------------- reading

def yaw_of(q) -> float:
    """Heading (rad) of a quaternion."""
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def _row(msg):
    """(kind, row) for one message, or (None, None) for a kind not read here."""
    if hasattr(msg, "twist"):
        p, tw = msg.pose.pose, msg.twist.twist
        return "odometry", (p.position.x, p.position.y, yaw_of(p.orientation), tw.linear.x, tw.angular.z)
    if hasattr(msg, "angular_velocity"):
        w, a = msg.angular_velocity, msg.linear_acceleration
        return "imu", (w.x, w.y, w.z, a.x, a.y, a.z)
    if hasattr(msg, "drive"):
        return "ackermann", (msg.drive.steering_angle, msg.drive.speed)
    if hasattr(msg, "buttons"):
        return "joy", (int(len(msg.buttons) > LB_BUTTON and bool(msg.buttons[LB_BUTTON])),)
    if hasattr(msg, "position") and hasattr(msg, "name"):
        return "joint", (float(msg.position[0]) if len(msg.position) else math.nan,)
    if hasattr(msg, "pose"):
        p = msg.pose.pose if hasattr(msg.pose, "pose") else msg.pose
        return "pose", (p.position.x, p.position.y, yaw_of(p.orientation))
    return None, None


@dataclass
class Run:
    """One recorded run: each topic read as a record array, plus parsed /drive_intent rows."""
    path: Path
    car: str
    run_id: str
    series: dict = field(default_factory=dict)
    frames: dict = field(default_factory=dict)
    intents: list = field(default_factory=list)
    rosout: list = field(default_factory=list)

    def has(self, topic) -> bool:
        return topic in self.series and len(self.series[topic]) > 0

    def __getitem__(self, topic):
        return self.series[topic]


def car_of(run_dir) -> str:
    """rb1 / rb2 from data/<car>/racerbot_runs/<run>, else metadata/car_id.txt, else unknown."""
    run_dir = Path(run_dir).resolve()
    if run_dir.parent.name == "racerbot_runs":
        return run_dir.parent.parent.name
    try:
        return (run_dir / "metadata" / "car_id.txt").read_text().strip() or "unknown"
    except OSError:
        return "unknown"


def load(run_dir, topics, rosout_from=()) -> Run:
    """Read ``topics`` from a run's bag into a Run. ``rosout_from``: node names
    whose /rosout lines are kept as (t, node, text)."""
    run_dir = Path(run_dir).expanduser()
    rows, kinds = {}, {}
    run = Run(path=run_dir, car=car_of(run_dir), run_id=run_dir.resolve().name)
    wanted = list(topics) + (["/rosout"] if rosout_from else [])
    for topic, t, msg in messages(run_dir, wanted):
        if topic == "/rosout":
            if msg.name in rosout_from:
                run.rosout.append((t, msg.name, msg.msg))
            continue
        if topic == "/drive_intent":
            row = parse_intent(msg.data)
            if row is not None:
                run.intents.append((t, row))
            continue
        if hasattr(msg, "header") and topic not in run.frames:
            run.frames[topic] = msg.header.frame_id
        kind, row = _row(msg)
        if kind is None:
            continue
        kinds[topic] = kind
        rows.setdefault(topic, []).append((t, *row))
    for topic, data in rows.items():
        run.series[topic] = np.rec.fromrecords(data, names=COLUMNS[kinds[topic]])
    return run


def node_params(run_dir, node) -> dict:
    """The parameters ``node`` reported when the run was recorded (metadata/params/<node>.yaml), else {}."""
    try:
        doc = yaml.safe_load((Path(run_dir) / "metadata" / "params" / f"{node}.yaml").read_text()) or {}
    except (OSError, yaml.YAMLError):
        return {}
    for value in doc.values():
        if isinstance(value, dict) and isinstance(value.get("ros__parameters"), dict):
            return value["ros__parameters"]
    return {}


def local_path(recorded) -> Path:
    """A path recorded on a car (/home/<user>/racerbot-ws/...) as the same file in this checkout."""
    text = str(recorded)
    marker = "/racerbot-ws/"
    return ROOT / text.split(marker, 1)[1] if marker in text else Path(text)


# --------------------------------------------------------------------------- time

def time_grid(t0: float, t1: float, fs: float = FS) -> np.ndarray:
    """Evenly spaced times from t0 to t1 inclusive-ish, at fs Hz."""
    if not (math.isfinite(t0) and math.isfinite(t1)) or t1 <= t0:
        return np.zeros(0)
    return t0 + np.arange(int(math.floor((t1 - t0) * fs)) + 1) / fs


def on_grid(grid, t, x, max_gap: float = MAX_GAP_S) -> np.ndarray:
    """x linearly interpolated at the grid times; NaN outside the samples or where
    the nearest sample is farther than max_gap (a silence is not invented)."""
    grid = np.asarray(grid, dtype=float)
    t, x = np.asarray(t, dtype=float), np.asarray(x, dtype=float)
    ok = np.isfinite(t) & np.isfinite(x)
    t, x = t[ok], x[ok]
    if len(t) == 0:
        return np.full(len(grid), np.nan)
    out = np.interp(grid, t, x)
    i = np.clip(np.searchsorted(t, grid), 1, max(len(t) - 1, 1))
    nearest = np.minimum(np.abs(grid - t[i - 1]), np.abs(grid - t[np.minimum(i, len(t) - 1)]))
    out[(grid < t[0]) | (grid > t[-1]) | (nearest > max_gap)] = np.nan
    return out


def unwrapped_on_grid(grid, t, angle, max_gap: float = MAX_GAP_S) -> np.ndarray:
    """An angle interpolated without jumping at +-pi."""
    return on_grid(grid, t, np.unwrap(np.asarray(angle, dtype=float)), max_gap)


def wrap(a):
    """Angle(s) into (-pi, pi]."""
    return (np.asarray(a, dtype=float) + math.pi) % (2.0 * math.pi) - math.pi


def moving_average(x, n: int) -> np.ndarray:
    """Centred mean over n samples of an evenly sampled series (NaN stays NaN)."""
    x = np.asarray(x, dtype=float)
    if n <= 1 or len(x) == 0:
        return x.copy()
    kernel = np.ones(n) / n
    ok = np.isfinite(x)
    num = np.convolve(np.where(ok, x, 0.0), kernel, mode="same")
    den = np.convolve(ok.astype(float), kernel, mode="same")
    out = np.where(den > 0.999, num / np.where(den > 0, den, 1.0), np.nan)
    return out


def derivative(x, fs: float, span_s: float = 0.2) -> np.ndarray:
    """Central difference over span_s seconds of an evenly sampled series."""
    x = np.asarray(x, dtype=float)
    k = max(1, int(round(span_s * fs / 2.0)))
    out = np.full(len(x), np.nan)
    if len(x) > 2 * k:
        out[k:-k] = (x[2 * k:] - x[:-2 * k]) * fs / (2 * k)
    return out


# --------------------------------------------------------------------------- driving

def lb_spans(run: Run) -> list:
    """[(press, release)] of LB holds, with the grades' own minimum hold."""
    if not run.has("/joy"):
        return []
    joy = run["/joy"]
    return lb_windows(joy.t, joy.lb, LB_MIN_HOLD_S)


def inside(t, spans) -> np.ndarray:
    """True where t falls inside any (start, end) span."""
    t = np.asarray(t, dtype=float)
    out = np.zeros(len(t), dtype=bool)
    for start, end in spans:
        out |= (t >= start) & (t <= end)
    return out


def intent_mask(run: Run, t, keep) -> np.ndarray:
    """True where the latest /drive_intent before t satisfies ``keep(row)``; all True
    when the run published no intent (the algorithm cannot then be split)."""
    t = np.asarray(t, dtype=float)
    if not run.intents:
        return np.ones(len(t), dtype=bool)
    it = np.array([s for s, _ in run.intents])
    ok = np.array([bool(keep(row)) for _, row in run.intents])
    i = np.searchsorted(it, t, side="right") - 1
    return np.where(i >= 0, ok[np.maximum(i, 0)], False)


# --------------------------------------------------------------------------- pose estimates

def pose_jumps(t, x, y, yaw, wheel_t, wheel_v, gyro_t=None, gyro_wz=None,
               tol_m: float = JUMP_M, tol_rad: float = JUMP_RAD) -> np.ndarray:
    """Times of the pose steps that moved farther than the wheels allow (by tol_m),
    or turned farther than the gyro allows (by tol_rad, when a gyro is given):
    relocalizations, loop closures, a particle filter snapping to another place."""
    t, x, y, yaw = (np.asarray(a, dtype=float) for a in (t, x, y, yaw))
    if len(t) < 2:
        return np.zeros(0)
    dt = np.diff(t)
    mid = (t[1:] + t[:-1]) / 2.0
    speed = np.abs(np.interp(mid, wheel_t, wheel_v)) if len(wheel_t) else np.zeros(len(mid))
    step = np.hypot(np.diff(x), np.diff(y))
    jump = step - speed * dt > tol_m
    if gyro_t is not None and len(gyro_t):
        turn = np.abs(wrap(np.diff(yaw)) - np.interp(mid, gyro_t, gyro_wz) * dt)
        jump |= turn > tol_rad
    return t[1:][jump]


MAP_POSES = ("/slam_pose", "/pf/viz/inferred_pose")   # SLAM's, then the particle filter's
# Which point of the car each map pose describes (src/pure_pursuit/config/pure_pursuit.yaml, pose_frame):
# slam_toolbox's /slam_pose is base_link; particle_filter ray-casts from the LiDAR, so its pose is the LiDAR's.
POSE_FRAMES = {"/slam_pose": "base_link", "/pf/viz/inferred_pose": "laser"}
LASER_X = 0.26           # m, base_link -> laser, docs/hardware-reference.md


def to_base_link(x, y, yaw, frame: str, offset_x: float = LASER_X, offset_y: float = 0.0):
    """Rear-axle (base_link) positions of poses of ``frame``: unchanged for base_link,
    moved back along the heading for laser (racing_math.laser_pose_to_base_link, the
    conversion pure_pursuit_node itself applies)."""
    from pure_pursuit import racing_math
    x, y, yaw = (np.asarray(a, dtype=float) for a in (x, y, yaw))
    if frame == "base_link":
        return x, y, yaw
    if frame != "laser":
        raise ValueError(f"pose frame {frame!r}: expected 'base_link' or 'laser'")
    out = np.array([racing_math.laser_pose_to_base_link(*p, offset_x, offset_y) for p in zip(x, y, yaw)]).reshape(-1, 3)
    return out[:, 0], out[:, 1], out[:, 2]


def map_pose(run: "Run"):
    """(topic, t, x, y, yaw) of the run's map pose moved to base_link, or None without one."""
    topic = next((t for t in MAP_POSES if run.has(t)), None)
    if topic is None:
        return None
    p = run[topic]
    return (topic, np.asarray(p.t, dtype=float), *to_base_link(p.x, p.y, p.yaw, POSE_FRAMES[topic]))
SPEED_SPAN_S = 0.4       # s, the centred span a pose is differentiated over
MIN_MAP_SPEED = 0.5      # m/s, below this a map speed is mostly pose noise


def forward_speed(x, y, yaw, fs: float = FS, span_s: float = SPEED_SPAN_S) -> np.ndarray:
    """Forward speed (m/s) of an evenly sampled pose: its velocity along its own heading.
    The same for the LiDAR's pose as for base_link's: both lie on the car's x axis."""
    vx, vy = derivative(x, fs, span_s), derivative(y, fs, span_s)
    return vx * np.cos(yaw) + vy * np.sin(yaw)


def map_speed(run: "Run", grid):
    """(topic, forward speed on the grid, jump times) from the run's map pose, NaN within
    a pose jump; (None, all-NaN, []) when the run has no map pose."""
    pose = map_pose(run)
    if pose is None or not run.has("/odom"):
        return None, np.full(len(grid), np.nan), np.zeros(0)
    topic, t, x, y, yaw = pose
    odom = run["/odom"]
    gyro = run["/imu/data"] if run.has("/imu/data") else None
    jumps = pose_jumps(t, x, y, yaw, odom.t, odom.v,
                       gyro.t if gyro is not None else None, gyro.wz if gyro is not None else None)
    v = forward_speed(on_grid(grid, t, x, 0.2), on_grid(grid, t, y, 0.2), unwrapped_on_grid(grid, t, yaw, 0.2))
    v[near(grid, jumps, JUMP_PAD_S + SPEED_SPAN_S / 2.0)] = np.nan
    return topic, v, jumps


AXIS_OK = 0.7            # |correlation| an accelerometer axis must reach with what it should measure


def accel_axes(ax, ay, forward, lateral, ok) -> dict:
    """Which car-axis accelerometer channel really carries the forward and the lateral
    acceleration, by correlation with independent estimates of each (the derivative of
    speed, and speed times the gyro's yaw rate). ``mapping_ok`` is True when ax follows
    forward and ay follows lateral, both with the right sign; otherwise ``problem``
    says what the channels follow instead (a mount rotation or sign error)."""
    ok = np.asarray(ok, dtype=bool)
    chans = {"ax": np.asarray(ax, dtype=float), "ay": np.asarray(ay, dtype=float)}
    refs = {"forward": np.asarray(forward, dtype=float), "lateral": np.asarray(lateral, dtype=float)}
    corr = {}
    for cn, cv in chans.items():
        for rn, rv in refs.items():
            m = ok & np.isfinite(cv) & np.isfinite(rv)
            corr[f"{cn}_vs_{rn}"] = (float(np.corrcoef(cv[m], rv[m])[0, 1])
                                     if m.sum() > 2 and np.ptp(cv[m]) > 0 and np.ptp(rv[m]) > 0 else math.nan)
    mapping_ok = corr["ax_vs_forward"] >= AXIS_OK and corr["ay_vs_lateral"] >= AXIS_OK
    follows = []
    for cn, want in (("ax", "forward"), ("ay", "lateral")):
        best = max(("forward", "lateral"), key=lambda rn: abs(corr[f"{cn}_vs_{rn}"])
                   if math.isfinite(corr[f"{cn}_vs_{rn}"]) else -1.0)
        r = corr[f"{cn}_vs_{best}"]
        if math.isfinite(r) and abs(r) >= AXIS_OK and not (best == want and r > 0):
            follows.append(f"{cn} follows {'-' if r < 0 else '+'}{best} (r {r:+.2f})")
    return dict(correlations=corr, mapping_ok=bool(mapping_ok),
                problem="; ".join(follows) if not mapping_ok else "")


def near(t, times, pad: float = JUMP_PAD_S) -> np.ndarray:
    """True where t is within pad seconds of any of times."""
    t = np.asarray(t, dtype=float)
    out = np.zeros(len(t), dtype=bool)
    for s in np.asarray(times, dtype=float):
        out |= np.abs(t - s) <= pad
    return out


FRAME_EXPLAINS = 0.5     # a rigid move that removes half the distance or more means two frames


def frame_verdict(distance_median_m: float, fit: dict) -> str:
    """'same frame' when no rotation and shift line two position tracks up better than they
    already are (the fit's residual is more than half the raw distance): what is left is the
    two estimates disagreeing. Otherwise 'frame offset', with the fitted rotation and shift."""
    if not (math.isfinite(distance_median_m) and math.isfinite(fit.get("residual_median_m", math.nan))):
        return "unknown"
    if distance_median_m <= 0 or fit["residual_median_m"] > FRAME_EXPLAINS * distance_median_m:
        return "same frame"
    return (f"frame offset: rotation {fit['rotation_deg']:.2f} deg, centre shift {fit['centre_shift_m']:.3f} m "
            f"explain most of the {distance_median_m:.3f} m")


def rigid_fit(a, b) -> dict:
    """The rotation and translation that best map points a onto points b (least squares,
    2-D Kabsch): rotation_deg; centre_shift_m, how far a's centre moves (a translation
    about the map origin would also count the rotation's swing, 0.17 m for 0.5 deg at
    20 m, so it is not reported); and residual_median_m, what no rigid move explains."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 2:
        return dict(points=int(len(a)), rotation_deg=math.nan, centre_shift_m=math.nan, residual_median_m=math.nan)
    ca, cb = a.mean(axis=0), b.mean(axis=0)
    h = (a - ca).T @ (b - cb)
    theta = math.atan2(h[0, 1] - h[1, 0], h[0, 0] + h[1, 1])
    r = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    residuals = np.hypot(*(b - cb - (a - ca) @ r.T).T)
    return dict(points=int(len(a)), rotation_deg=math.degrees(theta),
                centre_shift_m=float(np.hypot(*(cb - ca))), residual_median_m=float(np.median(residuals)))


# --------------------------------------------------------------------------- statistics

def finite(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x[np.isfinite(x)]


def stats(x) -> dict:
    """n, mean, RMS, p95 of |x|, max |x| of the finite values (NaN when there are none)."""
    x = finite(x)
    if not len(x):
        return dict(n=0, mean=math.nan, rms=math.nan, p95_abs=math.nan, max_abs=math.nan)
    return dict(n=int(len(x)), mean=float(np.mean(x)), rms=float(np.sqrt(np.mean(x * x))),
                p95_abs=float(np.percentile(np.abs(x), 95)), max_abs=float(np.max(np.abs(x))))


def line_fit(x, y) -> dict:
    """y = slope * x + intercept by least squares over finite pairs, with r (correlation)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3 or np.ptp(x) == 0:
        return dict(n=int(len(x)), slope=math.nan, intercept=math.nan, r=math.nan)
    slope, intercept = np.polyfit(x, y, 1)
    r = float(np.corrcoef(x, y)[0, 1]) if np.ptp(y) > 0 else math.nan
    return dict(n=int(len(x)), slope=float(slope), intercept=float(intercept), r=r)


# --------------------------------------------------------------------------- output

def out_dir(run: Run, out=DEFAULT_OUT) -> Path:
    """<out>/<car>-<run id>/, created. Refuses a folder inside data/: recordings stay untouched."""
    path = Path(out).expanduser().resolve() / f"{run.car}-{run.run_id}"
    data = (ROOT / "data").resolve()
    if path == data or data in path.parents:
        raise ValueError(f"{path}: results never go inside data/; choose another --out")
    path.mkdir(parents=True, exist_ok=True)
    return path


def pyplot():
    """matplotlib.pyplot with a file-only backend, or None when matplotlib is not installed."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    return plt


def save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    fig.clf()
    return path


def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return round(float(value), 4) if math.isfinite(value) else None
    return value


def write_summary(folder: Path, name: str, result: dict) -> Path:
    """Merge one example's numbers into <folder>/summary.json under ``name``."""
    path = Path(folder) / "summary.json"
    try:
        doc = json.loads(path.read_text())
    except (OSError, ValueError):
        doc = {}
    doc[name] = _plain(result)
    path.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return path


def fmt(value, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)
