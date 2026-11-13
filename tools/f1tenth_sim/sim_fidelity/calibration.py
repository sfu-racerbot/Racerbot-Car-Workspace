"""This car's physical parameters, each tagged with where it came from.

The point of this module is honesty about what is known. Stock F1TENTH Gym
ships one parameter set describing *a* small car; almost none of it was ever
checked against the vehicle in this workshop. Rather than silently inheriting
those numbers, every value here carries a :class:`Provenance` saying whether it
was measured on this car, derived from this car's committed configs, taken from
a datasheet for a part actually fitted, estimated from physics, or simply
inherited from upstream.

``python3 -m sim_fidelity.calibration`` prints the table, worst provenance
first. Anything still marked ``STOCK`` is a number nobody has justified.

Filling in a measurement means editing one line here. See
``docs/sim-fidelity-audit.md`` R1 for the procedures.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, fields, replace
from enum import Enum
import math
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src/racerbot_car'))
from racerbot_car import identity  # noqa: E402
from racerbot_car.identity import UnknownCarError, known_car_ids  # noqa: E402
from racerbot_car.layers import effective_config  # noqa: E402
from racerbot_car.paths import cars_dir  # noqa: E402
from racerbot_car.profile import load_profile  # noqa: E402

# Standard gravity, matching the figure used throughout the fidelity audit.
G = 9.81


class Provenance(Enum):
    """How much a number is worth, in descending order of trust."""

    MEASURED = "measured"     # measured on this physical car
    DERIVED = "derived"       # computed from this car's committed configs
    SPEC = "spec"             # datasheet for a part actually fitted
    ESTIMATED = "estimated"   # physics-based estimate, not this car
    STOCK = "stock"           # F1TENTH Gym default; describes another vehicle

    @property
    def rank(self) -> int:
        order = [
            Provenance.MEASURED,
            Provenance.DERIVED,
            Provenance.SPEC,
            Provenance.ESTIMATED,
            Provenance.STOCK,
        ]
        return order.index(self)


# --------------------------------------------------------------------------
# Configuration-derived fields are resolved at call time. Only simulator
# measurements and upstream constants remain fixed in this module.
# --------------------------------------------------------------------------

# VESC values are resolved by calibration_for_car(), never during import.

# Stock F1TENTH_VEHICLE_PARAMETERS, retained here so the fraction of the
# wheelbase ahead of the centre of gravity survives our wheelbase override.
STOCK_LF = 0.15875
STOCK_LR = 0.17145


# Car 2's full-lock wheel angles, retained for the car-2 floor linkage checks.
# MEASURED 2026-10-16 with tape-measured half circles
# (docs/hardware-reference.md): angle = atan(wheelbase / rear-axle radius).
# Not derived from the servo map above: the linkage is not linear, and
# inverting the affine servo map can claim angles past anything driven.
# Left: servo 0.178, radius 0.645 m (servo_min 0.15 itself was never driven).
# Right: servo_max 0.95, radius 0.750 m.
STEERING_MAX_LEFT_MEASURED = math.atan(0.36 / 0.645)    # +0.509 rad, 29.2 deg
STEERING_MAX_RIGHT_MEASURED = -math.atan(0.36 / 0.750)  # -0.448 rad, 25.6 deg

# CarCalibration's fields that come from the selected car's vesc.yaml (calibration_for_car);
# None in a constructed calibration means "the selected car's value".
VESC_FIELDS = ('steering_rate_max', 'speed_to_erpm_gain', 'steering_angle_to_servo_gain',
               'steering_angle_to_servo_offset', 'servo_min', 'servo_max', 'speed_max', 'speed_min')


@dataclass(frozen=True)
class CarCalibration:
    """Every physical constant the simulator needs for this car."""

    # --- Geometry -------------------------------------------------------
    wheelbase: float = 0.36
    # The padded safety envelope, used for clearance to walls.
    width: float = 0.31
    length: float = 0.58
    # The actual chassis. Contact between two cars has to use this: applying
    # the padded envelope to both bodies counts the safety margin twice and
    # reports a collision while roughly 7 mm of real air remains.
    body_width: float = 0.30
    body_length: float = 0.535
    lidar_offset_x: float = 0.26

    # --- Mass properties ------------------------------------------------
    mass: float = 3.74
    yaw_inertia: float = 0.04712
    cog_height: float = 0.074
    # Fraction of the wheelbase between the front axle and the centre of
    # gravity; 0.5 puts the CoG exactly at the wheelbase midpoint. Held at
    # 0.5 deliberately. The stock F1TENTH ratio is 0.4808 (STOCK_LF/STOCK_LR
    # below) and the audit rightly called 0.5 an assumption -- but 0.4808
    # belongs to a car with a 0.3302 m wheelbase, so it is no more a
    # measurement of this vehicle than 0.5 is. Swapping one unmeasured
    # number for another would move every result without improving fidelity.
    # Weigh each axle and put the real figure here; see R1.
    cog_front_fraction: float = 0.5

    # --- Tyres ----------------------------------------------------------
    cornering_stiffness_front: float = 4.718
    cornering_stiffness_rear: float = 5.4562
    # The friction coefficient between these tyres and the floor the car
    # actually runs on. Drives cornering, acceleration and braking limits
    # together, exactly as a friction circle says it should.
    surface_friction: float = 0.55
    # Share of the longitudinal demand that competes with cornering for grip.
    # See GripEnvelope.longitudinal_coupling for why this is not simply 1.
    longitudinal_grip_coupling: float = 1.0

    # --- Actuation ------------------------------------------------------
    steering_max_left: float = None
    steering_max_right: float = None
    car_id: InitVar[str | None] = None
    steering_rate_max: float = None
    speed_to_erpm_gain: float = None
    steering_angle_to_servo_gain: float = None
    steering_angle_to_servo_offset: float = None
    servo_min: float = None
    servo_max: float = None
    # Closed-loop proportional gain of the steering servo's own position
    # loop, 1/s. Must stay below 1/control_dt (40 for a 40 Hz loop) or the
    # discrete servo overshoots -- which is the bug this replaces.
    steering_servo_gain: float = 15.0
    # Whole-loop transport lag, in control ticks, from a command being
    # published to the actuator acting on it: ROS -> USB -> VESC firmware.
    steering_delay_ticks: int = 3
    speed_delay_ticks: int = 2
    # The VESC speed loop, as a proportional controller per direction with
    # an acceleration ceiling each way (plant._measured_speed_loop). Fitted
    # to run 20261019-200703 with tools/run_analysis/fit_sim.py: it brakes
    # far more gently than gym's stock loop (gain 13.7/s, up to mu*g).
    speed_loop_accel_gain: float = 5.0
    speed_loop_accel_max: float = 4.5
    speed_loop_brake_gain: float = 4.0
    speed_loop_brake_max: float = 4.0
    # Above this speed the acceleration ceiling falls linearly to zero at speed_loop_accel_no_load, as a
    # motor's force falls with speed (actuation.speed_loop_accel_ceiling). inf = no fall-off.
    speed_loop_accel_fade_from: float = math.inf
    speed_loop_accel_no_load: float = math.inf
    # Opt-in: ((wheel speed, acceleration ceiling), ...), linear between points, flat beyond. When set it replaces
    # speed_loop_accel_max and its fade-off as the loop's acceleration ceiling (actuation.accel_table_ceiling).
    accel_by_speed: tuple = ()

    # --- Speed envelope -------------------------------------------------
    speed_max: float = None
    speed_min: float = None

    # --- Sensing --------------------------------------------------------
    lidar_beams: int = 1081
    lidar_fov: float = math.radians(270.0)
    lidar_range_max: float = 25.0
    lidar_noise_std: float = 0.01
    lidar_dropout_rate: float = 0.002
    lidar_delay_ticks: int = 1
    pose_delay_ticks: int = 2
    pose_noise_xy: float = 0.02
    pose_noise_yaw: float = 0.01
    odom_speed_noise: float = 0.05

    # --- Floor models, off unless a floor profile sets them (see FLOORS) -
    # Command to effective wheel angle, ((command, angle), ...), ascending.
    steering_linkage: tuple | None = None
    # Lateral acceleration the rack asks for to what the tyres deliver,
    # ((asked, delivered), ...), ascending. None keeps GripEnvelope's sharp limit.
    grip_curve: tuple | None = None
    # True when that curve's delivered column falls past a peak (see ProgressiveGrip.falling).
    grip_curve_falls: bool = False
    # True speed the car settles at per unit of commanded speed.
    speed_command_scale: float = 1.0
    # What /odom reads per unit of true speed.
    odom_speed_scale: float = 1.0
    # Extra range noise per metre of range, on top of lidar_noise_std.
    lidar_noise_per_metre: float = 0.0
    # Beams that hit the car itself: ((beam index, range), ...).
    lidar_self_hits: tuple = ()
    # The floor profile these values came from; empty for the plain calibration.
    floor: str = ""

    def __post_init__(self, car_id):
        # Identity is provenance metadata, not a physical calibration/hash field.
        object.__setattr__(self, 'car_id', car_id)
        # Preserve CarCalibration(**overrides) for library users, resolving
        # omitted car fields at construction rather than module import. Only
        # these: the floor models' None means "off" (FLOORS), and resolving it
        # through calibration_for_car would recurse.
        missing = [name for name in (*VESC_FIELDS, 'steering_max_left',
                                    'steering_max_right', 'car_id')
                   if getattr(self, name) is None]
        if missing:
            selected = calibration_for_car(car_id)
            for name in missing:
                object.__setattr__(self, name, getattr(selected, name))

    # ------------------------------------------------------------------
    # Quantities the simulator needs that follow from the above
    # ------------------------------------------------------------------

    @property
    def lf(self) -> float:
        """Front axle to centre of gravity, metres."""
        return self.wheelbase * self.cog_front_fraction

    @property
    def lr(self) -> float:
        """Centre of gravity to rear axle, metres."""
        return self.wheelbase * (1.0 - self.cog_front_fraction)

    @property
    def friction_accel_limit(self) -> float:
        """Largest acceleration of any direction the floor can support.

        The friction circle's radius. This chassis is four-wheel drive
        (confirmed 2026-08-05), so the motor drives and brakes all four
        wheels through the transmission and the longitudinal extreme is mu*g
        just as the lateral one is -- a single number bounds acceleration,
        braking and cornering alike. On a rear-drive car it would not: rear
        axle load would cap braking near mu*g*lf/(L + mu*h), well under half
        this figure.
        """
        return self.surface_friction * G

    def vehicle_parameters(self, stock):
        """Stock gym :class:`VehicleParameters` updated to describe this car."""
        return stock.with_updates(
            mu=self.surface_friction,
            C_Sf=self.cornering_stiffness_front,
            C_Sr=self.cornering_stiffness_rear,
            lf=self.lf,
            lr=self.lr,
            h=self.cog_height,
            m=self.mass,
            I=self.yaw_inertia,
            s_min=self.steering_max_right,
            s_max=self.steering_max_left,
            sv_min=-self.steering_rate_max,
            sv_max=self.steering_rate_max,
            a_max=self.friction_accel_limit,
            # v_min/v_max are deliberately NOT overridden here. In gym they
            # are not only the speed envelope: pid_accl derives its
            # proportional gain from them as 10*a_max/v_max, so writing this
            # car's configured speed ceiling into v_max would quietly multiply
            # the plant's speed-loop gain by four. That gain is an upstream
            # invention, not a model of this car's VESC, so perturbing it is
            # not a fidelity improvement -- it just moves every result. The
            # real ceiling is enforced by FidelityPlant clamping the speed
            # command instead, which bounds the speed without touching the
            # gain. Measure the VESC's actual step response (R7) to replace
            # the whole arrangement with something justified.
            width=self.width,
            length=self.length,
            collision_body_center_x=self.wheelbase / 2.0,
        )


#: Where every field of :class:`CarCalibration` came from.
PROVENANCE: dict[str, tuple[Provenance, str]] = {
    "wheelbase": (
        Provenance.MEASURED,
        "tape-measured on this car 2026-08-24: 0.36 m front axle to rear "
        "axle, now in vesc.yaml. Traxxas publishes 0.324 for a stock "
        "74276-4; this car is not one",
    ),
    "width": (
        Provenance.DERIVED,
        "pure_pursuit.yaml car_width -- deliberately padded past the real body",
    ),
    "length": (
        Provenance.DERIVED,
        "pure_pursuit.yaml car_length -- deliberately padded past the real body",
    ),
    "body_width": (
        Provenance.MEASURED,
        "tape-measured on this car 2026-08-24: 0.30 m outer edge of tire "
        "to outer edge of tire, which is wider than the 0.281 m shell "
        "Traxxas publishes -- the tires are the widest part",
    ),
    "body_length": (
        Provenance.SPEC,
        "Traxxas 74276-4 published body length, per pure_pursuit.yaml",
    ),
    "lidar_offset_x": (
        Provenance.MEASURED,
        "tape-measured on this car 2026-08-24: 0.10 m behind the front "
        "axle, so 0.36 - 0.10 = 0.26 m ahead of the rear-axle base_link",
    ),
    "mass": (
        Provenance.STOCK,
        "gym default. NEVER WEIGHED -- put the car on a kitchen scale",
    ),
    "yaw_inertia": (
        Provenance.STOCK,
        "gym default. A uniform slab of this footprint would be 0.135 kg m^2, "
        "2.9x higher; a bifilar pendulum test would settle it",
    ),
    "cog_height": (Provenance.STOCK, "gym default. Never measured"),
    "cog_front_fraction": (
        Provenance.STOCK,
        "assumed midpoint. Weigh each axle separately to replace it -- this "
        "ratio IS the understeer/oversteer balance, and the traffic scenario "
        "is measurably sensitive to it",
    ),
    "cornering_stiffness_front": (
        Provenance.STOCK,
        "gym default, original F1TENTH tyres",
    ),
    "cornering_stiffness_rear": (
        Provenance.STOCK,
        "gym default, original F1TENTH tyres",
    ),
    "surface_friction": (
        Provenance.MEASURED,
        "fitted to car 2's run 20261019-200703 (data/rb2/racerbot_runs/, recorded on car 2; "
        "tools/run_analysis/fit_sim.py): 0.55 best matched the real yaw rate of 0.40/0.55/0.70/0.90; the car "
        "reached 91% of commanded curvature at 2.0-2.5 m/s^2 lateral and 78% "
        "above 2.5. Floor of that run only. Re-checked on car 1 on 2026-10-27 with "
        "tools/run_analysis/fit_sim.py (first LB hold): run 20261026-194514 yaw-rate "
        "RMS (rad/s) 0.169 at 0.45, 0.153 at 0.55 (minimum), 0.162 at 0.65; run "
        "20261026-194913 yaw-rate RMS 0.126-0.133 across 0.45-0.65, nearly flat. Summed over both "
        "runs 0.55 is lowest (0.283 against 0.295 at either end), so 0.55 stands for car 1",
    ),
    "longitudinal_grip_coupling": (
        Provenance.ESTIMATED,
        "how much of the friction circle the drivetrain can claim. Full "
        "coupling is the honest ellipse but bites far too often against "
        "upstream's all-or-nothing speed controller",
    ),
    "steering_max_left": (
        Provenance.MEASURED,
        "rb2: tape-measured full-lock half circles 2026-10-16, atan(0.36 / 0.645), "
        "servo 0.178 (servo_min 0.15 never driven); cars/rb2/car.yaml; "
        "racerbot_sim/sim_bridge.py and sim_fidelity/calibration.py",
    ),
    "steering_max_right": (
        Provenance.MEASURED,
        "rb2: tape-measured full-lock half circles 2026-10-16, -atan(0.36 / 0.750), "
        "servo_max 0.95; cars/rb2/car.yaml; racerbot_sim/sim_bridge.py and sim_fidelity/calibration.py",
    ),
    "steering_rate_max": (
        Provenance.DERIVED,
        "selected car's effective vesc.yaml throttle_interpolator max_servo_speed, rad/s",
    ),
    "steering_servo_gain": (
        Provenance.MEASURED,
        "fitted to run 20261019-200703 (tools/run_analysis/fit_sim.py): recorded /drive replayed open-loop into the plant, scored against LiDAR scan-matched speed and yaw rate. docs/run-analysis.md",
    ),
    "steering_delay_ticks": (
        Provenance.MEASURED,
        "fitted to run 20261019-200703 (tools/run_analysis/fit_sim.py): recorded /drive replayed open-loop into the plant, scored against LiDAR scan-matched speed and yaw rate. docs/run-analysis.md",
    ),
    "speed_delay_ticks": (
        Provenance.MEASURED,
        "fitted to run 20261019-200703 (tools/run_analysis/fit_sim.py): recorded /drive replayed open-loop into the plant, scored against LiDAR scan-matched speed and yaw rate. docs/run-analysis.md",
    ),
    "speed_loop_accel_gain": (Provenance.MEASURED, "fitted to run 20261019-200703 (tools/run_analysis/fit_sim.py): recorded /drive replayed open-loop into the plant, scored against LiDAR scan-matched speed and yaw rate. docs/run-analysis.md"),
    "speed_loop_accel_max": (Provenance.MEASURED, "fitted to run 20261019-200703 (tools/run_analysis/fit_sim.py): recorded /drive replayed open-loop into the plant, scored against LiDAR scan-matched speed and yaw rate. docs/run-analysis.md"),
    "speed_loop_brake_gain": (Provenance.MEASURED, "fitted to run 20261019-200703 (tools/run_analysis/fit_sim.py): recorded /drive replayed open-loop into the plant, scored against LiDAR scan-matched speed and yaw rate. docs/run-analysis.md"),
    "speed_loop_brake_max": (Provenance.MEASURED, "fitted to run 20261019-200703 (tools/run_analysis/fit_sim.py): recorded /drive replayed open-loop into the plant, scored against LiDAR scan-matched speed and yaw rate. docs/run-analysis.md"),
    "speed_loop_accel_fade_from": (Provenance.ESTIMATED, "inf by default (no fall-off); a floor profile sets it where the measured, traction-limited speed range ends"),
    "accel_by_speed": (Provenance.ESTIMATED, "empty: the loop's ceiling is speed_loop_accel_max with its fade-off. A floor may set a measured table (accel_by_speed.py). See FLOORS"),
    "speed_loop_accel_no_load": (Provenance.ESTIMATED, "inf by default; a floor profile sets an estimated motor no-load speed (actuation.speed_loop_accel_ceiling)"),
    "speed_max": (
        Provenance.DERIVED,
        "selected car's effective vesc.yaml speed_max / abs(speed_to_erpm_gain). Enforced "
        "by clamping the speed command, not through gym's v_max -- see "
        "vehicle_parameters()",
    ),
    "speed_min": (Provenance.DERIVED, "negative of the converted speed_max (symmetric envelope)"),
    "lidar_beams": (
        Provenance.SPEC,
        "Hokuyo UST-10LX: 1081 steps of 0.25 deg over 270 deg, and "
        "sensors.yaml asks for full resolution (cluster 1, skip 0). Confirm "
        "with: ros2 topic echo /scan --once --field angle_increment",
    ),
    "lidar_fov": (Provenance.SPEC, "Hokuyo UST-10LX 270 deg sweep"),
    "lidar_range_max": (
        Provenance.SPEC,
        "UST-10LX rated 10 m guaranteed / 30 m maximum; 25 m kept from the "
        "original harness",
    ),
    "lidar_noise_std": (
        Provenance.SPEC,
        "UST-10LX +/-40 mm accuracy under 10 m implies roughly 1 cm sigma",
    ),
    "lidar_dropout_rate": (
        Provenance.ESTIMATED,
        "fraction of beams returning no echo off dark or glancing surfaces",
    ),
    "lidar_delay_ticks": (
        Provenance.ESTIMATED,
        "one tick: the scan is up to one 40 Hz period stale when used",
    ),
    "pose_delay_ticks": (
        Provenance.ESTIMATED,
        "two ticks (50 ms) of particle-filter latency. Measure with "
        "race_diagnostics' pose-lag report",
    ),
    "pose_noise_xy": (
        Provenance.ESTIMATED,
        "2 cm particle-filter position jitter against a good map",
    ),
    "pose_noise_yaw": (
        Provenance.ESTIMATED,
        "0.01 rad (0.6 deg) particle-filter heading jitter",
    ),
    "odom_speed_noise": (
        Provenance.ESTIMATED,
        "VESC eRPM speed estimate; the 2026-07-27 log tracked commands "
        "within about 5%",
    ),
    "steering_linkage": (Provenance.ESTIMATED, "off: the plain calibration has no linkage model. See FLOORS"),
    "grip_curve": (Provenance.ESTIMATED, "off: the plain calibration has a sharp friction circle. See FLOORS"),
    "grip_curve_falls": (Provenance.DERIVED, "off: a grip curve must ascend in both columns. See FLOORS"),
    "speed_command_scale": (Provenance.DERIVED, "1: command taken as true speed. See FLOORS"),
    "odom_speed_scale": (Provenance.DERIVED, "1: odometry taken as true speed. See FLOORS"),
    "lidar_noise_per_metre": (Provenance.ESTIMATED, "0: flat noise. See FLOORS"),
    "lidar_self_hits": (Provenance.ESTIMATED, "none. See FLOORS"),
    "floor": (Provenance.DERIVED, "the floor profile's name, empty here"),
}


for _name in ('speed_to_erpm_gain', 'steering_angle_to_servo_gain',
              'steering_angle_to_servo_offset', 'servo_min', 'servo_max'):
    PROVENANCE[_name] = (Provenance.DERIVED, "selected car's effective vesc.yaml " + _name)


def selected_car(car_id: str | None = None, *, note: bool = True) -> str:
    """Resolve implicit identity strictly; allow known explicit offline targets."""
    directory = cars_dir()
    known = known_car_ids(directory)
    if car_id is None:
        return identity.current_car_id()
    if car_id not in known:
        raise UnknownCarError(f'unknown car {car_id!r}; pass --car or set RACERBOT_CAR')
    env = dict(os.environ)
    env.pop(identity.ENV_VAR, None)
    try:
        machine_car = identity.resolve_car_id(
            env, identity.read_serial(identity.SERIAL_PATH),
            identity.load_machines(directory / 'machines.yaml', known), known)
    except (UnknownCarError, OSError, ValueError):
        machine_car = None
    if note and machine_car and machine_car != car_id:
        print(f'Note: machine identity is {machine_car}; explicitly selected {car_id}.')
    return car_id


def add_car_argument(parser):
    parser.add_argument('--car', default=None,
                        help='car profile ID (default: $RACERBOT_CAR; required)')


def car_from_args(parser, args) -> str:
    try:
        return selected_car(args.car) if args.car is not None else selected_car()
    except UnknownCarError as exc:
        parser.error(str(exc))


def calibration_for_car(car_id: str | None = None) -> CarCalibration:
    """Build a fresh calibration from the selected car's effective VESC values."""
    car_id = selected_car(car_id, note=False)
    vesc = effective_config('vesc.yaml', '/**', car_id)
    throttle = effective_config('vesc.yaml', 'throttle_interpolator', car_id)
    profile = load_profile(car_id)
    speed_max = vesc['speed_max'] / abs(vesc['speed_to_erpm_gain'])
    return CarCalibration(
        car_id=car_id, steering_max_left=profile.full_lock_left_rad,
        steering_max_right=profile.full_lock_right_rad,
        **{key: vesc[key] for key in ('speed_to_erpm_gain',
           'steering_angle_to_servo_gain', 'steering_angle_to_servo_offset',
           'servo_min', 'servo_max')},
        speed_max=speed_max, speed_min=-speed_max,
        steering_rate_max=throttle['max_servo_speed'])


def __getattr__(name):
    # Compatibility for external probes. Internal callers use the factory.
    if name == 'CALIBRATION':
        return calibration_for_car()
    keys = {'SPEED_TO_ERPM_GAIN': 'speed_to_erpm_gain',
            'SPEED_MAX_ERPM': 'speed_max',
            'STEERING_ANGLE_TO_SERVO_GAIN': 'steering_angle_to_servo_gain',
            'STEERING_ANGLE_TO_SERVO_OFFSET': 'steering_angle_to_servo_offset',
            'SERVO_MIN': 'servo_min', 'SERVO_MAX': 'servo_max',
            'MAX_SERVO_SPEED': 'steering_rate_max'}
    if name in keys:
        cal = calibration_for_car()
        if name == 'SPEED_MAX_ERPM':
            return cal.speed_max * abs(cal.speed_to_erpm_gain)
        return getattr(cal, keys[name])
    raise AttributeError(name)


#: Fitted to these 2026-10-30 runs, located against data/rb2/racerbot_auto/20261030-185036-fixed.
SFU_COURSE_FIT_RUNS = ("20261030-185036", "20261030-184556", "20261030-175728",
                       "20261030-180355", "20261030-181616", "20261030-183607")
#: Held out of every fit, and used only to check it.
SFU_COURSE_HELD_OUT_RUNS = ("20261030-184826", "20261030-183134", "20261030-181003",
                            "20261030-181819", "20261030-184316", "20261030-175124")

SFU_COURSE_2026_11_06_FIT_RUNS = ("20261106-163633", "20261106-164401", "20261106-170803")
SFU_COURSE_2026_11_06_HELD_OUT_RUNS = ("20261106-171030", "20261106-184753")
SFU_COURSE_2026_11_11_RB2_FIT_RUNS = ("20261111-152132", "20261111-135934", "20261111-142254")


# Gyro-basis median delivered lateral acceleration per asked bin on car 2, 2026-11-11 (bins of 10 or more
# samples; grip_truth.py --json gyro_curve of runs 20261111-152132, -135934, -142254). Falls past its peak.
SFU_2026_11_11_RB2_GYRO_CURVE = (
                (0.0, 0.0),
                (0.24780840442007881, 0.24780840442007881),
                (0.7974440280734391, 0.7974440280734391),
                (1.2715557017298433, 1.2715557017298433),
                (1.767423331714832, 1.767423331714832),
                (2.2756500555012367, 2.2756500555012367),
                (2.7375241827844983, 2.7098528689465846),
                (3.217853000236789, 2.8809540690061866),
                (3.7081195959524607, 2.93841069556196),
                (4.3847966706921095, 2.7063512860611474),
                (5.279196169155968, 2.7998427458681205),
)
# Times which the plant's own tyres' understeer is made up in the plant. Calibrated on flying-lap time and braking;
# the 152132 launch run-wide is NOT reproduced; the grip scale lies between 1.10 (lap 10.12 s) and 1.25 (best
# open-loop yaw RMS). See FLOORS provenance and sfu_course_2026_11_11_rb2_replay.json closed_loop_selection.
SFU_2026_11_11_RB2_GRIP_SCALE = 1.15


def sfu_2026_11_11_rb2_grip_curve(scale: float) -> tuple:
    """The floor's grip table at ``scale``: the measured curve times the scale, never above the request."""
    return tuple((asked, min(asked, achieved * scale)) for asked, achieved in SFU_2026_11_11_RB2_GYRO_CURVE)


@dataclass(frozen=True)
class Floor:
    """One floor with one set of tyres, and how it was measured."""

    name: str
    description: str
    overrides: dict
    sources: dict
    fit_runs: tuple
    held_out_runs: tuple
    track: str = ""


FLOORS: dict[str, Floor] = {
    "sfu_course_2026_11_06": Floor(
        name="sfu_course_2026_11_06",
        description="Car 2 on the 2026-11-06 floor and tyres; fitted only on course 06a, "
                    "held out on 06a and 06b. See docs/sim-calibration.md and the adjacent measurement JSON.",
        track="car2_course_2026_11_06a_scans",
        fit_runs=SFU_COURSE_2026_11_06_FIT_RUNS,
        held_out_runs=SFU_COURSE_2026_11_06_HELD_OUT_RUNS,
        overrides=dict(
            floor="sfu_course_2026_11_06",
            # Slope 1.05 and trim from fit-only gyro replay, not the diluted binned table.
            # Retain the 2026-10-16 full-stick and endpoint circle measurements outside
            # the 0.349 rad range exercised by these runs; extrapolation is unvalidated.
            steering_linkage=((-0.461, -0.448), (-0.436, -0.423),
                              (-0.349, -0.36000714791229876), (0.0, 0.006442852087701211),
                              (0.349, 0.37289285208770123),
                              (0.436, 0.423), (0.528, 0.509)),
            # Independent stock-tyre understeer remains, so replay selects a 1.05 scale.
            grip_curve=tuple((asked, min(asked, achieved * 1.05)) for asked, achieved in (
                (0.0, 0.0), (2.0, 2.0), (2.25, 2.1196167301547333), (2.75, 2.441124084767721),
                (3.25, 2.5957683749493055), (3.75, 2.8700874781313486), (4.5, 3.051088703714249),
                (5.5, 3.4256766650211903), (7.0, 3.797103059968375), (10.0, 4.655680540688182))),
            steering_delay_ticks=3,
            steering_rate_max=2.5,
            steering_servo_gain=15.0,
            speed_loop_accel_gain=7.0,
            speed_loop_accel_max=4.0,
            speed_loop_brake_gain=7.0,
            speed_loop_brake_max=3.0,
            speed_loop_accel_fade_from=5.4,
            speed_loop_accel_no_load=16.0,
            speed_delay_ticks=4,
            odom_speed_scale=1.0323453317475084,
            speed_command_scale=4614.0 / 4447.983786 / 1.0323453317475084,
            longitudinal_grip_coupling=0.0,
            lidar_noise_std=0.003,
            lidar_noise_per_metre=0.0015,
            lidar_dropout_rate=0.0,
            lidar_self_hits=(),
        ),
        sources=dict(
            floor="2026-11-06 rb2 fit runs 163633, 164401, 170803; sfu_course_2026_11_06_split.json",
            steering_linkage="fit-run straight trim (sfu_course_2026_11_06_measurement.json); slope 1.05 chosen by "
                             "fit-plant gyro replay, not the binned linkage; endpoint circles 2026-10-16; "
                             "sfu_course_2026_11_06_replay.json",
            grip_curve="car_measure.py on 20261106-163633, -164401, -170803; scale 1.05 chosen by fit-plant "
                       "gyro replay, sfu_course_2026_11_06_replay.json",
            steering_delay_ticks="fit-plant gyro replay on 20261106-163633, -164401, -170803 selects 3 ticks; "
                                 "sfu_course_2026_11_06_replay.json",
            steering_rate_max="car_measure.py servo replay on 20261106-163633, -164401, -170803",
            steering_servo_gain="car_measure.py servo replay on 20261106-163633, -164401, -170803",
            speed_loop_accel_gain="fit-only true-speed replay, sfu_course_2026_11_06_measurement.json speed_loop",
            speed_loop_accel_max="fit-only true-speed replay, sfu_course_2026_11_06_measurement.json speed_loop",
            speed_loop_brake_gain="fit-only true-speed replay, sfu_course_2026_11_06_measurement.json speed_loop",
            speed_loop_brake_max="fit-only true-speed replay, sfu_course_2026_11_06_measurement.json speed_loop",
            speed_loop_accel_fade_from="ESTIMATED: the fitted 4 m/s^2 ceiling showed no fall-off up to the 5.4 m/s "
                                       "the car reached on 2026-11-06 (sim-calibration.md, top speed overshoot); "
                                       "above that, untested",
            speed_loop_accel_no_load="ESTIMATED, not measured: a DC motor's force falls linearly with speed to "
                                     "its no-load speed; ~16 m/s is the order expected for this motor and 3S "
                                     "battery through the 4614 eRPM per m/s gearing. Only matters above "
                                     "5.4 m/s (raised VESC speed_max experiments, 2026-11-09)",
            speed_delay_ticks="fit-only true-speed replay, sfu_course_2026_11_06_measurement.json speed_loop",
            odom_speed_scale="wheel / scan-to-map true speed, 20261106-163633, -164401, -170803",
            speed_command_scale="fit-run wheel over-read and metadata vesc gains 4614 / 4447.983786",
            longitudinal_grip_coupling="fit-run grip_by_long_accel: 2.779 braking below -3 vs 2.647 steady m/s^2",
            lidar_noise_std="20261106-163633 stable parked window; sfu_course_2026_11_06_stable_lidar.json",
            lidar_noise_per_metre="same Hokuyo's 2026-10-30 range slope, checked against 20261106-163633 stable window",
            lidar_dropout_rate="20261106-163633 moving_no_echo = 0; sfu_course_2026_11_06_lidar.json",
            lidar_self_hits="20261106-163633 stable parked window: no rear-right self hits",
        ),
    ),
    "sfu_course_2026_11_11_rb2": Floor(
        name="sfu_course_2026_11_11_rb2",
        description="Car 2 on the 2026-11-11 race-day course (UVic), measured from its race-day runs. The grip "
                    "curve FALLS past its peak (2.94 m/s^2 at 3.75 asked, 2.7 beyond). See docs/sim-calibration.md "
                    "and the adjacent sfu_course_2026_11_11_* JSON.",
        track="car2_course_2026_11_11",
        fit_runs=SFU_COURSE_2026_11_11_RB2_FIT_RUNS,
        held_out_runs=(),
        overrides=dict(
            floor="sfu_course_2026_11_11_rb2",
            steering_linkage=((-0.461, -0.448), (-0.436, -0.423), (-0.349, -0.3370065000246209), (0.0, 0.011993499975379084), (0.349, 0.36099349997537905), (0.436, 0.423), (0.528, 0.509)),
            # Gyro-basis median delivered lateral acceleration per asked bin, bins of 10 or more
            # samples, never above the request (grip_truth.py --json gyro_curve), times 1.15 (the plant's
            # own tyres understeer on top; chosen on the 152132 launch). Falls past the peak.
            grip_curve=sfu_2026_11_11_rb2_grip_curve(SFU_2026_11_11_RB2_GRIP_SCALE),
            grip_curve_falls=True,
            steering_delay_ticks=2,
            steering_rate_max=2.5,
            steering_servo_gain=15.0,
            speed_loop_accel_gain=10.0,
            speed_loop_accel_max=4.0,
            speed_loop_brake_gain=5.0,
            speed_loop_brake_max=4.0,
            speed_loop_accel_fade_from=5.4,
            speed_loop_accel_no_load=16.0,
            speed_delay_ticks=4,
            odom_speed_scale=1.0166595115220707,
            speed_command_scale=4614.0 / 4447.983786 / 1.0166595115220707,
            longitudinal_grip_coupling=0.0,
            lidar_noise_std=0.003,
            lidar_noise_per_metre=0.0015,
            lidar_dropout_rate=0.0,
            lidar_self_hits=(),
        ),
        sources=dict(
            floor="2026-11-11 rb2 runs 152132, 135934, 142254; sfu_course_2026_11_11_rb2_measurement.json",
            steering_linkage="central slope 1.0 and trim 0.011993 (car_measure.py steering_trim_rad) chosen by fit-plant "
                             "gyro replay on 20261111-152132 over slopes 0.95-1.1 and delays 2-4 ticks, "
                             "sfu_course_2026_11_11_rb2_replay.json; outer points retained from 2026-10-16 circles",
            grip_curve="grip_truth.py on 20261111-152132, -135934, -142254, /imu/vesc/data, steering window 0.10 rad, "
                       "gyro basis, bins of 10+ samples; sfu_course_2026_11_11_grip_rb2_vesc.json gyro_curve; times 1.15. "
                       "Calibrated on flying-lap time and braking; the 152132 launch run-wide is NOT reproduced; the grip scale "
                       "lies between 1.10 (lap 10.12 s) and 1.25 (best open-loop yaw RMS). At 1.15 the cap-on flying lap is 9.95 s "
                       "against the real 10.05 s, braking lag and deceleration match (acceptance B), and the open-loop yaw RMS is "
                       "better than the 11-06 floor. From s=8.1 m with the cap off the sim makes contact at s 11.7-12.0 in 6/6 "
                       "seeds (4.74 / 5.09 m/s at s=10 / 11 against the real 4.28 / 4.75), but the real car's cap was still "
                       "binding there, so that run is not comparable. Cells: sfu_course_2026_11_11_rb2_replay.json closed_loop_selection",
            grip_curve_falls="the same curve: 2.94 m/s^2 at 3.75 asked falling to 2.71 at 4.38",
            steering_delay_ticks="fit-plant gyro replay on 20261111-152132 selects 2 ticks by yaw RMS (0.1256 against 0.1333 at 3); sfu_course_2026_11_11_rb2_replay.json",
            steering_rate_max="car_measure.py servo replay on 20261111-152132, -135934, -142254",
            steering_servo_gain="car_measure.py servo replay on 20261111-152132, -135934, -142254",
            speed_loop_accel_gain="car_measure.py speed loop, sfu_course_2026_11_11_rb2_measurement.json speed_loop",
            speed_loop_accel_max="car_measure.py speed loop, sfu_course_2026_11_11_rb2_measurement.json speed_loop",
            speed_loop_brake_gain="car_measure.py speed loop, sfu_course_2026_11_11_rb2_measurement.json speed_loop",
            speed_loop_brake_max="car_measure.py speed loop, sfu_course_2026_11_11_rb2_measurement.json speed_loop",
            speed_loop_accel_fade_from="ESTIMATED, carried from sfu_course_2026_11_06: no fall-off seen up to 5.14 m/s",
            speed_loop_accel_no_load="ESTIMATED, carried from sfu_course_2026_11_06; not measured",
            speed_delay_ticks="car_measure.py speed loop, sfu_course_2026_11_11_rb2_measurement.json speed_loop",
            odom_speed_scale="wheel / scan-to-map true speed, 20261111-152132, -135934, -142254 (wheel_over_true)",
            speed_command_scale="wheel over-read of the three runs and the run's vesc gains 4614 / 4447.983786",
            longitudinal_grip_coupling="grip_by_long_accel: 3.079 braking below -3 vs 2.968 steady m/s^2, same runs",
            lidar_noise_std="CARRIED from sfu_course_2026_11_06_stable_lidar.json (same Hokuyo); not re-measured",
            lidar_noise_per_metre="CARRIED from sfu_course_2026_11_06 (same Hokuyo); not re-measured",
            lidar_dropout_rate="CARRIED from sfu_course_2026_11_06; not re-measured",
            lidar_self_hits="CARRIED from sfu_course_2026_11_06; not re-measured",
        ),
    ),
    "sfu_course_2026_10_30": Floor(
        name="sfu_course_2026_10_30",
        description="The 2026-10-30 course's floor and the tyres on car 2 that day. Measured "
                    "with tools/run_analysis/car_measure.py; see docs/sim-calibration.md.",
        track="course_2026_10_30_scans",
        fit_runs=SFU_COURSE_FIT_RUNS,
        held_out_runs=SFU_COURSE_HELD_OUT_RUNS,
        overrides=dict(
            floor="sfu_course_2026_10_30",
            # The rack sits 0.0105 rad left of the command (yaw on straights, fit runs). Full
            # stick, 0.436 rad, turns 0.423 both sides, and the servo limits reach 0.509 left and
            # 0.448 right (tape-measured circles, 2026-10-16, docs/hardware-reference.md).
            # Between those, straight lines: a table binned from yaw read 30% less rack at
            # 0.1-0.2 rad, and replaying real commands against the gyro rejected it (yaw error
            # 0.24-0.31 rad/s against 0.14). Binning by a lagged noisy command flattens it.
            # Servo limits as commands: (0.95 - 0.5904) / -0.78 = -0.461 right, (0.178 - 0.5904) / -0.78 = 0.528 left.
            steering_linkage=((-0.461, -0.448), (-0.436, -0.423), (0.0, 0.0105), (0.436, 0.423), (0.528, 0.509)),
            # The real asked-against-achieved curve of the fit runs, times 1.15, never above
            # what was asked. Times 1.15 because the plant's stock tyres understeer by themselves
            # on top of it; 1.15-1.2 fitted best by replay against the gyro.
            grip_curve=((0.0, 0.0), (2.0, 2.0), (2.25, 2.25), (2.75, 2.737), (3.25, 3.013), (3.75, 3.209),
                        (4.5, 3.45), (5.5, 3.841), (7.0, 4.255), (10.0, 4.715), (20.0, 5.175)),
            steering_delay_ticks=2,
            speed_loop_accel_gain=7.0,
            speed_loop_accel_max=4.0,
            # Braking gain from the whole-run replay; the ceiling from the wall-measured stop of
            # run 20261030-180355 (fit set): 3.10 m/s on the wheels to 0.29 in 1.31 m. This gives
            # 1.23 m there and 3.44 m on the held-out stop of run 20261030-175124 (measured 3.39).
            # Not reproduced: the real stops end with a soft tail under 1 m/s, so they take
            # 0.93 and 1.40 s against 0.72 and 1.21 here. Gain 3 matches those times too, but
            # then ordinary driving brakes too gently and laps come out 5% fast.
            speed_loop_brake_gain=7.0,
            speed_loop_brake_max=4.5,
            speed_delay_ticks=3,
            # vesc.yaml commands through 4614 eRPM per m/s and /odom reads through 4447.98.
            # With the measured wheel over-read: 4614 / 4447.98 / 1.033.
            speed_command_scale=1.004,
            odom_speed_scale=1.033,
            # Parked, run 20261030-185036: 3 mm near, 10 mm past 4 m. No echo: 4e-7 of beams.
            lidar_noise_std=0.003,
            lidar_noise_per_metre=0.0015,
            lidar_dropout_rate=0.0,
            # Parked, run 20261030-185036: the car's own rear right, -135 to -130 deg.
            lidar_self_hits=tuple(enumerate((
                .147, .140, .123, .108, .093, .085, .083, .086, .091, .127, .140, .155,
                .172, .195, .199, .204, .224, .247, .270, .285, .301, .338, .385))),
        ),
        sources=dict(
            steering_linkage="trim from yaw on straights, fit runs; full stick and limits from the circles "
                             "of 2026-10-16; chosen over a binned table by replay against the gyro",
            grip_curve="fit runs' asked-against-achieved curve (car_measure.py), scaled 1.15 by replay of "
                       "runs 20261030-185036, -184556, -181616, -183607 against the gyro (validate_calibration.py "
                       "fit-plant): yaw rate error 0.140 rad/s",
            steering_delay_ticks="real commands through the plant against the gyro, runs 20261030-185036 and "
                                 "-184556: yaw rate error 0.123 rad/s, lag within 6 ms. Gain 15 and the "
                                 "3.2 rad/s slew stand: slew does not bind from 2.5 up",
            speed_loop_accel_gain="true speed replay, fit runs",
            speed_loop_accel_max="true speed replay, fit runs",
            speed_loop_brake_gain="true speed replay, fit runs",
            speed_loop_brake_max="wall-measured stop of run 20261030-180355 (fit runs); checked on held-out "
                                 "run 20261030-175124",
            speed_delay_ticks="true speed replay, fit runs",
            speed_command_scale="vesc.yaml gains and the measured wheel over-read",
            odom_speed_scale="wheel against scan-to-map speed, fit runs (held out: the same 1.033)",
            lidar_noise_std="parked scans, run 20261030-185036",
            lidar_noise_per_metre="parked scans, run 20261030-185036",
            lidar_dropout_rate="driving scans, run 20261030-185036",
            lidar_self_hits="parked scans, run 20261030-185036",
        ),
    ),
}


def with_floor(name: str | None, base: CarCalibration | None = None) -> CarCalibration:
    """``base`` (default: the selected car's calibration, resolved now) with a named floor
    profile's values; None leaves it unchanged. The profiles were measured on one car (see
    each Floor's description): applying one to another car is a deliberate choice."""
    base = calibration_for_car() if base is None else base
    if name is None:
        return base
    if name not in FLOORS:
        raise ValueError(f"unknown floor {name!r}; known: {sorted(FLOORS)}")
    return replace(base, **FLOORS[name].overrides)


def with_friction(friction: float | None, base: CarCalibration | None = None) -> CarCalibration:
    """``base`` (default: the selected car's calibration, resolved now) on a floor of this
    friction; None leaves it unchanged."""
    base = calibration_for_car() if base is None else base
    if friction is None:
        return base
    if not (math.isfinite(friction) and friction > 0.0):
        raise ValueError(f"friction must be positive and finite, got {friction!r}")
    return replace(base, surface_friction=float(friction))


def describe(calibration: CarCalibration | None = None) -> list[tuple]:
    """(provenance, name, value, source) rows, least trustworthy first."""
    calibration = calibration or calibration_for_car()
    rows = []
    for field in fields(calibration):
        provenance, source = PROVENANCE[field.name]
        if calibration.car_id == 'rb1':
            if field.name == 'steering_max_left':
                provenance, source = (Provenance.MEASURED,
                    'rb1: 0.401 rad left, measured 2026-10-21 check circles; '
                    'docs/hardware-reference.md "Steering limits, 2026-10-22"; '
                    'full stick reaches servo_min (0.051 vs 0.05); cars/rb1/car.yaml')
            elif field.name == 'steering_max_right':
                provenance, source = (Provenance.DERIVED,
                    'rb1: -(0.75 - 0.366) / 0.83 rad, computed from the servo map, not measured; '
                    'cars/rb1/calibration/vesc.yaml at servo_max 0.75; cars/rb1/car.yaml')
        rows.append(
            (provenance, field.name, getattr(calibration, field.name), source)
        )
    rows.sort(key=lambda row: (-row[0].rank, row[1]))
    return rows


def unmeasured(calibration: CarCalibration | None = None) -> list[str]:
    """Fields still inherited from upstream rather than justified here."""
    return [
        name
        for provenance, name, _value, _source in describe(calibration)
        if provenance is Provenance.STOCK
    ]


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    add_car_argument(parser)
    args = parser.parse_args(argv)
    calibration = calibration_for_car(car_from_args(parser, args))
    rows = describe(calibration)
    width = max(len(name) for _p, name, _v, _s in rows)
    current = None
    for provenance, name, value, source in rows:
        if provenance is not current:
            print(f"\n[{provenance.value.upper()}]")
            current = provenance
        shown = f"{value:.5g}" if isinstance(value, float) else str(value)
        print(f"  {name:<{width}}  {shown:>10}   {source}")
    print(f"\n{len(unmeasured(calibration))} of {len(rows)} values are still gym stock.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
