#pragma once
// The pure pursuit law as plain functions: no ROS, no state.

#include <cstddef>
#include <optional>
#include <vector>

#include "path.hpp"

struct Pose
{
    double x;
    double y;
    double yaw;
};

/// The rear axle's pose from a sensor's pose, the sensor being `offset_x`
/// ahead of and `offset_y` left of the rear axle. The particle filter reports
/// the LiDAR's pose; pure pursuit's geometry is about the rear axle.
Pose rear_axle(const Pose& sensor, double offset_x, double offset_y);

/// A world point in the car frame: x ahead, y to the left.
void to_car_frame(const Pose& car, double wx, double wy, double& local_x, double& local_y);

/// Lookahead distance (m): base + gain * |speed|, clamped to [min, max].
double lookahead(double speed, double base, double gain, double min, double max);

/// Sharpest curvature from waypoint `from` to `distance` m ahead (a loop wraps,
/// an open path stops at its end). `curvature` is Path::curvature().
double max_curvature_ahead(const Path& path, const std::vector<double>& curvature, size_t from, double distance);

/// Lookahead speed gain: gain_straight at curvature 0, gain_corner at or above
/// corner_curvature, linear between. Non-finite curvature reads as a corner.
double blended_gain(double curvature, double gain_straight, double gain_corner, double corner_curvature);

struct Goal
{
    double local_x;   // ahead of the car (m)
    double local_y;   // left of the car (m)
    size_t index;     // waypoint it came from (the last one for a point past an open path's end)
    bool past_end;    // a point L ahead along an open path's last segment
};

/// The goal point: where the path crosses the circle of radius L, between the
/// last waypoint inside it and the first ahead of the car outside it (`index`).
/// That waypoint itself when none is inside or the crossing is behind the car.
/// A loop wraps round; an open path does not, and
/// near its end, when every waypoint left is closer than L, the goal is the
/// point L ahead along the last segment, so steering stays calm instead of
/// swinging at the final waypoint. None when nothing is ahead of the car.
std::optional<Goal> find_goal(const Path& path, size_t from, const Pose& car, double L);

/// The pose `dt` seconds on, at constant speed `v` and `steering` (kinematic
/// bicycle, exact arc). Unchanged for dt <= 0.
Pose predict(const Pose& p, double v, double steering, double wheelbase, double dt);

struct SteerCommand
{
    double time;       // s, when it was sent
    double steering;   // rad
};

/// The pose at t1 from a pose measured at t0, at speed v, steering as commanded:
/// each command (sorted by time) acts `delay` s after it was sent; 0 before the
/// first. Unchanged when t1 <= t0.
Pose predict_through(const Pose& p, double t0, double t1, double v, double wheelbase,
                     const std::vector<SteerCommand>& history, double delay);

/// Pure pursuit steering (rad) for a goal in the car frame: the arc through
/// it has curvature 2y/d^2; the bicycle model turns that into atan(wheelbase * k),
/// clamped to +-max_steering.
double steering_for(double local_x, double local_y, double wheelbase, double max_steering);

/// Speed for a steering angle: v_max straight, v_mid at half of max_steering,
/// v_min at full lock, straight lines between (pure_pursuit_a's rule when v_mid
/// is halfway). |steering| past max_steering, or non-finite, reads as full lock.
double steering_speed(double steering, double max_steering, double v_max, double v_mid, double v_min);

/// Speed that still stops by the goal: sqrt(2 * brake * (remaining - tolerance)),
/// 0 inside the tolerance; infinity when there is no goal.
double goal_ramp(double remaining, double tolerance, double brake);

/// Speed after an acceleration limit: at most `from + accel * dt` (dt clamped
/// to [0, 0.2] s). Slowing down is never limited.
double limit_acceleration(double target, double from, double accel, double dt);

/// `target` within +-rate*dt of `previous` (dt clamped to [0, max_dt]; non-finite = 0).
double limit_rate(double target, double previous, double rate, double dt, double max_dt = 0.1);
