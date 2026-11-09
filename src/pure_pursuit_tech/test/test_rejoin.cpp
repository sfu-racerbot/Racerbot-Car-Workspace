// Rejoining the line after a deviation: the real control law (Tracker,
// lookahead, find_goal, steering_for) driving a kinematic bicycle in closed
// loop. Runner: `colcon test --packages-select pure_pursuit_tech` (gtest
// target test_rejoin).
//
// The verdict never comes from the controller: the cross-track error is
// measured here, as the distance from the rear axle to the polyline.
//
// Oracle for small offsets: pure pursuit on a straight, linearized
// (heading psi, offset e, distance s, lookahead L; goal lateral offset in the
// car frame ~ -(e + L psi), curvature 2*ly/L^2):
//     e'' + (2/L) e' + (2/L^2) e = 0
// damping ratio 1/sqrt(2) (overshoot exp(-pi) = 4.3%), decay rate 1/L per
// metre, envelope |e(s)| <= sqrt(2) e0 exp(-s/L).

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <optional>
#include <vector>

#include "control.hpp"
#include "path.hpp"
#include "tracker.hpp"

namespace
{
constexpr double kWheelbase = 0.36;      // m, both cars (docs/hardware-reference.md)
constexpr double kMaxSteering = 0.33;    // rad, car 1's autonomy cap
constexpr double kBase = 1.0, kGain = 0.1, kMinL = 0.8, kMaxL = 3.0;   // the launch defaults

// Distance from (x, y) to the path's polyline (segments, not just waypoints).
double cross_track(const std::vector<Waypoint>& w, bool closed, double x, double y)
{
    double best = 1e9;
    const size_t n = w.size();
    for (size_t i = 0; i + (closed ? 0 : 1) < n; ++i) {
        const Waypoint& a = w[i];
        const Waypoint& b = w[(i + 1) % n];
        const double dx = b.x - a.x, dy = b.y - a.y;
        const double t = std::clamp(((x - a.x) * dx + (y - a.y) * dy) / (dx * dx + dy * dy), 0.0, 1.0);
        best = std::min(best, std::hypot(x - (a.x + t * dx), y - (a.y + t * dy)));
    }
    return best;
}

struct Trace
{
    std::vector<double> s;      // distance driven (m)
    std::vector<double> error;  // cross-track error (m)
    bool lost = false;          // the control law found no goal point
};

// Drive `distance` metres at constant `speed` from `start`, control at 40 Hz
// (the particle filter's rate), integration at 400 Hz.
Trace drive(const std::vector<Waypoint>& points, Pose start, double speed, double distance)
{
    const Path path(points, 0.5);
    Tracker tracker(path, 0, 1e9);
    Trace run;
    Pose car = start;
    double driven = 0.0, steering = 0.0;
    const double dt = 1.0 / 400.0;
    for (int tick = 0; driven < distance; ++tick) {
        if (tick % 10 == 0) {
            const auto where = tracker.update(car.x, car.y);
            const double L = lookahead(speed, kBase, kGain, kMinL, kMaxL);
            const auto goal = find_goal(path, where.closest, car, L);
            if (!goal) {
                run.lost = true;
                return run;
            }
            steering = steering_for(goal->local_x, goal->local_y, kWheelbase, kMaxSteering);
            run.s.push_back(driven);
            run.error.push_back(cross_track(points, path.closed(), car.x, car.y));
        }
        car.x += speed * std::cos(car.yaw) * dt;
        car.y += speed * std::sin(car.yaw) * dt;
        car.yaw += speed / kWheelbase * std::tan(steering) * dt;
        driven += speed * dt;
    }
    return run;
}

std::vector<Waypoint> straight(double length = 80.0, double step = 0.1)
{
    std::vector<Waypoint> w;
    for (int i = 0; i * step <= length; ++i) {
        w.push_back({i * step, 0.0});
    }
    return w;
}

std::vector<Waypoint> circle(double r, int n)
{
    std::vector<Waypoint> w;
    for (int i = 0; i < n; ++i) {
        const double a = 2.0 * M_PI * i / n;
        w.push_back({r * std::cos(a), r * std::sin(a)});
    }
    return w;
}

double max_after(const Trace& r, double from_s)
{
    double m = 0.0;
    for (size_t i = 0; i < r.s.size(); ++i) {
        if (r.s[i] >= from_s) {
            m = std::max(m, r.error[i]);
        }
    }
    return m;
}

double first_below(const Trace& r, double threshold)
{
    for (size_t i = 0; i < r.s.size(); ++i) {
        if (r.error[i] < threshold) {
            return r.s[i];
        }
    }
    return INFINITY;
}
}  // namespace

TEST(Rejoin, SmallOffsetOnAStraightFollowsTheLinearizedDynamics)
{
    // e0 = 0.1 m against L = 1.2 m at 2 m/s: e/L = 0.08, the linear regime.
    const double e0 = 0.1, v = 2.0, L = kBase + kGain * v;
    const Trace r = drive(straight(), {1.0, e0, 0.0}, v, 30.0);
    ASSERT_FALSE(r.lost);
    for (size_t i = 0; i < r.s.size(); ++i) {
        // oracle: the envelope sqrt(2) e0 exp(-s/L), plus 5 mm for the 0.1 m
        // waypoint grid (the goal is the first waypoint at or past L)
        EXPECT_LE(r.error[i], std::sqrt(2.0) * e0 * std::exp(-r.s[i] / L) + 0.005) << "at s=" << r.s[i];
    }
    // oracle: decays at 1/L per metre: below 2% of e0 by 4L (the 2% settling distance)
    EXPECT_LT(max_after(r, 4.0 * L * 1.25), 0.02 * e0 + 0.005);
}

TEST(Rejoin, LargeOffsetsOnAStraightNeverGrowAndRejoin)
{
    // Invariants rather than the linear oracle: steering saturates here.
    // Up to 1.4 m: just inside the node's default max_path_error (1.5 m).
    for (double e0 : {0.5, 1.0, 1.4}) {
        for (double v : {1.0, 2.0, 4.0}) {
            const Trace r = drive(straight(), {1.0, e0, 0.0}, v, 60.0);
            ASSERT_FALSE(r.lost) << "e0=" << e0 << " v=" << v;
            double peak = 0.0;
            for (double e : r.error) peak = std::max(peak, e);
            // invariant: pure pursuit never steers away from the line on a straight
            EXPECT_LE(peak, e0 + 1e-6) << "e0=" << e0 << " v=" << v;
            // rejoined (< 5 cm) and stays rejoined
            const double s_in = first_below(r, 0.05);
            EXPECT_LT(s_in, 60.0) << "never rejoined: e0=" << e0 << " v=" << v;
            EXPECT_LT(max_after(r, s_in + 5.0), 0.05) << "e0=" << e0 << " v=" << v;
            std::printf("straight e0=%.1f m v=%.0f m/s: within 5 cm after %.1f m, overshoot %.3f m\n",
                        e0, v, s_in, max_after(r, s_in + 0.5));
        }
    }
}

TEST(Rejoin, HeadingErrorOnAStraightIsCorrected)
{
    for (double deg : {15.0, 30.0, 60.0}) {
        const double psi = deg * M_PI / 180.0;
        const Trace r = drive(straight(), {1.0, 0.0, psi}, 2.0, 40.0);
        ASSERT_FALSE(r.lost) << deg;
        double peak = 0.0;
        for (double e : r.error) peak = std::max(peak, e);
        // measured, not asserted: how far it strays before turning back
        EXPECT_LT(first_below(r, 0.05), 40.0) << deg;
        EXPECT_LT(max_after(r, 25.0), 0.05) << deg;   // rejoined and stays there
        size_t at_peak = 0;
        while (r.error[at_peak] < peak) ++at_peak;
        double back = INFINITY;
        for (size_t i = at_peak; i < r.s.size(); ++i) {
            if (r.error[i] < 0.05) { back = r.s[i]; break; }
        }
        EXPECT_LT(back, 25.0) << deg;
        std::printf("straight heading %2.0f deg: strays %.2f m at %.1f m, back within 5 cm at %.1f m\n",
                    deg, peak, r.s[at_peak], back);
    }
}

TEST(Rejoin, OffsetsOnACircleRejoinFromInsideAndOutside)
{
    // A 3 m circle (curvature 0.33, inside the 0.95 /m the rack can do).
    // Steady state: a car on the circle facing along it pursues the circle
    // itself (any arc tangent to the heading through a point of the circle
    // is the circle), so the error must go to ~0.
    const auto c = circle(3.0, 300);
    for (double e0 : {-1.0, -0.5, 0.5, 1.0}) {    // negative: inside
        const double r0 = 3.0 + e0;
        const Trace r = drive(c, {r0, 0.0, M_PI / 2}, 2.0, 2 * 2 * M_PI * 3.0);
        ASSERT_FALSE(r.lost) << e0;
        double peak = 0.0;
        for (double e : r.error) peak = std::max(peak, e);
        EXPECT_LE(peak, std::abs(e0) + 0.05) << e0;
        // rejoined: the 300-gon itself sits within r (1 - cos(pi/300)) = 0.16 mm of the circle
        EXPECT_LT(max_after(r, 2 * M_PI * 3.0), 0.05) << e0;
        std::printf("circle r=3 e0=%+.1f m: within 5 cm after %.1f m, peak %.2f m\n",
                    e0, first_below(r, 0.05), peak);
    }
}

TEST(Rejoin, FacingAwayFindsNoGoal)
{
    // 120 degrees off on an open path: everything ahead of the car is behind
    // the search start. The node stops (test_pure_pursuit_tech_node.py); here
    // the law must say so rather than pick a point behind it.
    const Trace r = drive(straight(), {10.0, 0.5, 2.0 * M_PI / 3.0}, 2.0, 10.0);
    EXPECT_TRUE(r.lost);
}
