#include "control.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

Pose rear_axle(const Pose& sensor, double offset_x, double offset_y)
{
    const double c = std::cos(sensor.yaw);
    const double s = std::sin(sensor.yaw);
    return {sensor.x - (c * offset_x - s * offset_y),
            sensor.y - (s * offset_x + c * offset_y),
            sensor.yaw};
}

void to_car_frame(const Pose& car, double wx, double wy, double& local_x, double& local_y)
{
    const double dx = wx - car.x;
    const double dy = wy - car.y;
    local_x = std::cos(car.yaw) * dx + std::sin(car.yaw) * dy;    // how far ahead of car
    local_y = -std::sin(car.yaw) * dx + std::cos(car.yaw) * dy;   // left (+) or right (-)
}

double lookahead(double speed, double base, double gain, double min, double max)
{
    return std::clamp(base + gain * std::abs(speed), min, max);
}

std::optional<Goal> find_goal(const Path& path, size_t from, const Pose& car, double L)
{
    const size_t n = path.size();
    const size_t steps = path.closed() ? n : n - from;
    bool have_prev = false;
    double px = 0.0, py = 0.0;   // previous waypoint, car frame
    for (size_t step = 0; step < steps; ++step) {
        const size_t i = path.closed() ? (from + step) % n : from + step;
        double lx, ly;
        to_car_frame(car, path[i].x, path[i].y, lx, ly);
        // waypoint must be IN FRONT of the car and AT LEAST L away
        if (lx > 0.0 && std::hypot(lx, ly) >= L) {
            // constraint L at the circle, effectively acts as a low pass filter
            if (have_prev && std::hypot(px, py) < L) {
                const double dx = lx - px;
                const double dy = ly - py;
                const double a = dx * dx + dy * dy;
                const double b = 2.0 * (px * dx + py * dy);
                const double c = px * px + py * py - L * L;   // < 0: prev is inside
                if (a > 1e-12) {
                    const double t = (-b + std::sqrt(b * b - 4.0 * a * c)) / (2.0 * a);
                    const double gx = px + t * dx;
                    // crossing behind the car: keep the waypoint
                    if (gx > 0.0) {
                        return Goal{gx, py + t * dy, i, false};
                    }
                }
            }
            return Goal{lx, ly, i, false};
        }
        px = lx;
        py = ly;
        have_prev = true;
    }
    if (!path.closed()) {
        const Waypoint& end = path[n - 1];
        const Waypoint& before = path[n - 2];
        const double seg = std::hypot(end.x - before.x, end.y - before.y);
        const double beyond = std::max(0.0, L - std::hypot(end.x - car.x, end.y - car.y));
        double lx, ly;
        to_car_frame(car, end.x + (end.x - before.x) / seg * beyond,
                     end.y + (end.y - before.y) / seg * beyond, lx, ly);
        if (lx > 0.0) {
            return Goal{lx, ly, n - 1, true};
        }
    }
    return std::nullopt;
}

double steering_for(double local_x, double local_y, double wheelbase, double max_steering)
{
    // local_x^2 + local_y^2 is the squared distance to the goal, so this uses
    // the actual distance to the waypoint rather than L
    const double curvature = 2.0 * local_y / (local_x * local_x + local_y * local_y);   // k = 2y/d^2
    return std::clamp(std::atan(wheelbase * curvature), -max_steering, max_steering);
}

double steering_speed(double steering, double max_steering, double v_max, double v_mid, double v_min)
{
    // how far toward full lock, 0 (straight) to 1 (full lock)
    const double a = std::isfinite(steering) ? std::min(std::abs(steering) / max_steering, 1.0) : 1.0;
    if (a <= 0.5) {
        return v_max + (v_mid - v_max) * (a / 0.5);       // straight -> half lock
    }
    return v_mid + (v_min - v_mid) * ((a - 0.5) / 0.5);   // half lock -> full lock
}

double goal_ramp(double remaining, double tolerance, double brake)
{
    if (!std::isfinite(remaining)) {
        return std::numeric_limits<double>::infinity();
    }
    return std::sqrt(2.0 * brake * std::max(0.0, remaining - tolerance));
}

double limit_acceleration(double target, double from, double accel, double dt)
{
    return std::min(target, from + accel * std::clamp(dt, 0.0, 0.2));
}
