#pragma once
// The pure pursuit law as plain functions: no ROS, no state.

#include <cstddef>
#include <optional>

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

struct Goal
{
    double local_x;   // ahead of the car (m)
    double local_y;   // left of the car (m)
    size_t index;     // waypoint it came from (the last one for a point past an open path's end)
    bool past_end;    // a point L ahead along an open path's last segment
};

/// The goal point: the first waypoint from `from` on that is ahead of the
/// car and at least L away. A loop wraps round; an open path does not, and
/// near its end, when every waypoint left is closer than L, the goal is the
/// point L ahead along the last segment, so steering stays calm instead of
/// swinging at the final waypoint. None when nothing is ahead of the car.
std::optional<Goal> find_goal(const Path& path, size_t from, const Pose& car, double L);

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
