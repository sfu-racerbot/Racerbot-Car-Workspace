// Unit tests of pure_pursuit_tech's ROS-free core: path.hpp, tracker.hpp,
// control.hpp. Runner: `colcon test --packages-select pure_pursuit_tech`
// (ament_add_gtest target test_core). Every expected value is a closed form
// computed here, an invariant, or a cited rule; each test says which.

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "control.hpp"
#include "path.hpp"
#include "tracker.hpp"

namespace
{
constexpr double kInf = std::numeric_limits<double>::infinity();


std::vector<Waypoint> straight(int n = 201, double step = 0.1, double y = 0.0)
{
    std::vector<Waypoint> w;
    for (int i = 0; i < n; ++i) {
        w.push_back({step * i, y});
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

// closed form: perimeter of a regular n-gon inscribed in a circle of radius r
double polygon_perimeter(double r, int n) { return n * 2.0 * r * std::sin(M_PI / n); }

const SpeedLimits kLimits{4.0, 0.5, 2.5, 3.0, 3.0};
}  // namespace

// --- parse_waypoints_csv ------------------------------------------------------

TEST(ParseCsv, KeepsNumericRowsAndSkipsEverythingElse)
{
    std::istringstream in(
        "x,y,speed\n"          // header: not numbers
        "# a comment\n"
        "\n"
        "0,0,1.0\n"
        "1,0\n"
        "1,0\n"                // repeat of the previous point
        "nan,1\n"
        "inf,1\n"
        "2\n"                  // one column only
        "2.5abc,1\n"           // trailing junk
        " 3 , 1 \n");          // spaces around numbers are fine
    const auto w = parse_waypoints_csv(in);
    // oracle: the three valid, distinct rows above, in order
    ASSERT_EQ(w.size(), 3u);
    EXPECT_DOUBLE_EQ(w[0].x, 0.0);
    EXPECT_DOUBLE_EQ(w[1].x, 1.0);
    EXPECT_DOUBLE_EQ(w[2].x, 3.0);
    EXPECT_DOUBLE_EQ(w[2].y, 1.0);
}

TEST(ParseCsv, EmptyInputGivesNoPoints)
{
    std::istringstream in("");
    EXPECT_TRUE(parse_waypoints_csv(in).empty());
}

// --- Path: shape -----------------------------------------------------------------

TEST(Path, LoopOrOpenDependsOnTheGapAtTheClosureDistance)
{
    // 3-4-5 triangle corners; the gap from the last point to the first is 5 m.
    const std::vector<Waypoint> w{{0, 0}, {4, 0}, {4, 3}};
    EXPECT_TRUE(Path(w, 5.0).closed());          // exactly at the boundary: a loop
    EXPECT_FALSE(Path(w, 5.0 - 1e-9).closed());  // one step under: open
}

TEST(Path, LengthsAreTheSumOfSegments)
{
    const std::vector<Waypoint> w{{0, 0}, {4, 0}, {4, 3}};
    const Path open(w, 1.0);
    EXPECT_DOUBLE_EQ(open.length(), 7.0);        // 4 + 3
    EXPECT_DOUBLE_EQ(open.s(2), 7.0);
    const Path loop(w, 5.0);
    EXPECT_DOUBLE_EQ(loop.length(), 12.0);       // 4 + 3 + 5 back to the start
    EXPECT_DOUBLE_EQ(loop.segment(2), 5.0);
}

TEST(Path, ALoopWithItsFirstPointRepeatedDropsTheRepeat)
{
    auto w = circle(2.0, 50);
    w.push_back(w.front());
    const Path p(w, 0.5);
    EXPECT_TRUE(p.closed());
    EXPECT_EQ(p.size(), 50u);
    EXPECT_NEAR(p.length(), polygon_perimeter(2.0, 50), 1e-9);   // closed form
}

TEST(Path, RefusesTooFewPoints)
{
    EXPECT_THROW(Path({{0, 0}, {1, 0}}, 0.5), std::invalid_argument);
    EXPECT_THROW(Path({}, 0.5), std::invalid_argument);
}

TEST(Path, AdvanceWrapsOnALoopAndStopsAtTheEndOfAnOpenPath)
{
    const Path loop(circle(2.0, 10), 2.0);
    EXPECT_EQ(loop.advance(8, 3), 1u);
    const Path open(straight(11), 0.5);
    EXPECT_EQ(open.advance(8, 3), 10u);
}

// --- Path: curvature --------------------------------------------------------------

TEST(Path, CurvatureOfACircleIsOneOverItsRadius)
{
    // invariant: any three points on a circle lie on that circle, so the
    // three-point estimate is exact.
    const Path p(circle(2.0, 200), 0.5);
    for (double k : p.curvature()) {
        EXPECT_NEAR(k, 0.5, 1e-9);   // 1/m
    }
}

TEST(Path, CurvatureOfAStraightIsZeroIncludingItsEnds)
{
    const Path p(straight(), 0.5);
    for (double k : p.curvature()) {
        EXPECT_NEAR(k, 0.0, 1e-12);
    }
}

// --- Path: speed profile ------------------------------------------------------------

TEST(SpeedProfile, ACircleRunsAtTheCorneringLimit)
{
    // closed form: v = sqrt(a_lat / k) = sqrt(2.5 / 0.5)
    const Path p(circle(2.0, 200), 0.5);
    for (double v : p.speed_profile(kLimits, false)) {
        EXPECT_NEAR(v, std::sqrt(2.5 / 0.5), 1e-9);   // m/s
    }
}

TEST(SpeedProfile, ClampsToVmaxButTheCorneringLimitBeatsVmin)
{
    const Path p(circle(2.0, 200), 0.5);
    for (double v : p.speed_profile({4.0, 0.5, 1000.0, 3.0, 3.0}, false)) {
        EXPECT_DOUBLE_EQ(v, 4.0);    // grip to spare: v_max
    }
    // Almost no grip: the corner needs sqrt(0.01 / 0.5) = 0.141 m/s, far below
    // v_min = 3.0, and gets it. Spec: the cornering limit takes priority over
    // min_speed (user instruction 2026-11-08; README "speed_mode").
    for (double v : p.speed_profile({4.0, 3.0, 0.01, 3.0, 3.0}, false)) {
        EXPECT_NEAR(v, std::sqrt(0.01 / 0.5), 1e-9);    // closed form, m/s
    }
}

TEST(SpeedProfile, AnOpenPathBrakesToAStopAtItsEnd)
{
    const Path p(straight(), 0.5);                   // 20 m
    const auto v = p.speed_profile(kLimits, true);
    EXPECT_DOUBLE_EQ(v[200], 0.0);
    // closed form: v = sqrt(2 * a_brake * d) for the last stretch, v_max before it
    EXPECT_NEAR(v[190], std::sqrt(2.0 * 3.0 * 1.0), 1e-9);    // 1 m before the end
    EXPECT_NEAR(v[199], std::sqrt(2.0 * 3.0 * 0.1), 1e-9);
    EXPECT_DOUBLE_EQ(v[100], 4.0);
    // without stop_at_end the end is not special
    EXPECT_DOUBLE_EQ(p.speed_profile(kLimits, false)[200], 4.0);
}

TEST(SpeedProfile, NeverAsksForMoreBrakingAccelerationOrGripThanAllowed)
{
    // A straight into a 1 m-radius quarter turn and out on another straight.
    std::vector<Waypoint> w;
    for (int i = 0; i < 100; ++i) w.push_back({0.1 * i, 0.0});
    for (int i = 1; i <= 50; ++i) {
        const double a = M_PI / 2 * i / 50;
        w.push_back({9.9 + std::sin(a), 1.0 - std::cos(a)});
    }
    for (int i = 1; i <= 100; ++i) w.push_back({10.9, 1.0 + 0.1 * i});
    const Path p(w, 0.5);
    const auto k = p.curvature();
    const auto v = p.speed_profile(kLimits, false);
    bool braked = false, accelerated = false;
    for (size_t i = 0; i + 1 < v.size(); ++i) {
        const double ds = p.segment(i);
        // invariants: v_{i+1}^2 <= v_i^2 + 2 a_accel ds, and v_i^2 <= v_{i+1}^2 + 2 a_brake ds
        EXPECT_LE(v[i + 1] * v[i + 1], v[i] * v[i] + 2.0 * 3.0 * ds + 1e-9) << "accel at " << i;
        EXPECT_LE(v[i] * v[i], v[i + 1] * v[i + 1] + 2.0 * 3.0 * ds + 1e-9) << "brake at " << i;
        // invariant: the cornering limit, with no v_min floor above it
        if (k[i] > 1e-9) {
            EXPECT_LE(v[i], std::sqrt(2.5 / k[i]) + 1e-9) << "grip at " << i;
        }
        braked |= v[i + 1] < v[i] - 1e-6;
        accelerated |= v[i + 1] > v[i] + 1e-6;
    }
    // the corner really is slower than the straights, and the car speeds up after it
    EXPECT_NEAR(v[125], std::sqrt(2.5 / 1.0), 0.01);   // closed form, mid-corner, m/s
    EXPECT_DOUBLE_EQ(v[0], 4.0);
    EXPECT_TRUE(braked);
    EXPECT_TRUE(accelerated);
}

// A straight into a 1 m-radius quarter turn and out on another straight.
std::vector<Waypoint> straight_into_quarter_turn()
{
    std::vector<Waypoint> w;
    for (int i = 0; i < 100; ++i) w.push_back({0.1 * i, 0.0});
    for (int i = 1; i <= 50; ++i) {
        const double a = M_PI / 2 * i / 50;
        w.push_back({9.9 + std::sin(a), 1.0 - std::cos(a)});
    }
    for (int i = 1; i <= 100; ++i) w.push_back({10.9, 1.0 + 0.1 * i});
    return w;
}

// Worst breach of the friction ellipse over the path, in units of its budget
// (0 = inside it everywhere). Spec, as racing_math.compute_velocity_profile's
// docstring: each step's longitudinal acceleration may use only
// a_long_max * sqrt(1 - (v^2 k / a_lat)^2), evaluated at the neighbour the
// pass already knows (the next point when braking, the previous when speeding up).
double worst_ellipse_breach(const Path& p, const std::vector<double>& v, const SpeedLimits& lim)
{
    const auto k = p.curvature();
    const auto budget = [&](double a_max, double speed, double kappa) {
        const double lat = speed * speed * kappa / lim.a_lat;
        return a_max * std::sqrt(std::max(0.0, 1.0 - lat * lat));
    };
    double worst = 0.0;
    for (size_t i = 0; i + 1 < v.size(); ++i) {
        const double a = (v[i + 1] * v[i + 1] - v[i] * v[i]) / (2.0 * p.segment(i));   // m/s^2
        const double allowed = a > 0.0 ? budget(lim.a_accel, v[i], k[i])
                                       : budget(lim.a_brake, v[i + 1], k[i + 1]);
        worst = std::max(worst, std::abs(a) - allowed);
    }
    return worst;
}

TEST(SpeedProfile, BrakingAndAcceleratingShareGripWithCornering)
{
    const Path p(straight_into_quarter_turn(), 0.5);
    const auto v = p.speed_profile(kLimits, false);
    // invariant: inside the friction ellipse at every step (1e-9 m/s^2 round-off)
    EXPECT_LE(worst_ellipse_breach(p, v, kLimits), 1e-9);

    // The path really exercises it: with independent limits (the old profile)
    // the corner entry and exit break the ellipse by a clear margin.
    SpeedLimits independent = kLimits;
    independent.friction_ellipse = false;
    const auto old_v = p.speed_profile(independent, false);
    EXPECT_GT(worst_ellipse_breach(p, old_v, kLimits), 0.5);   // m/s^2

    // Sharing grip only ever lowers the plan, and does lower it somewhere.
    bool lower = false;
    for (size_t i = 0; i < v.size(); ++i) {
        EXPECT_LE(v[i], old_v[i] + 1e-12) << "faster at " << i;
        lower |= v[i] < old_v[i] - 1e-3;
    }
    EXPECT_TRUE(lower);
    // Still the cornering limit mid-corner (closed form sqrt(a_lat / k), k = 1 /m)
    // and still v_max at the start of the first straight.
    EXPECT_NEAR(v[125], std::sqrt(2.5 / 1.0), 0.01);
    EXPECT_DOUBLE_EQ(v[0], 4.0);
}

TEST(SpeedProfile, BrakingUsesExactlyTheEllipseBudgetLeftByCornering)
{
    // An open half circle, r = 4 m, braking to a stop at its end. Its cornering
    // limit sqrt(2.5 / 0.25) = 3.16 m/s is below v_max, so the braking zone
    // runs from 100% lateral load down to 0: every point of the ellipse's shape.
    std::vector<Waypoint> w;
    const int n = 252;                             // ~0.05 m apart
    for (int i = 0; i < n; ++i) {
        const double a = M_PI * i / (n - 1);
        w.push_back({4.0 * std::sin(a), 4.0 - 4.0 * std::cos(a)});
    }
    const Path p(w, 0.5);
    ASSERT_FALSE(p.closed());
    const auto k = p.curvature();
    const auto v = p.speed_profile(kLimits, true);
    // Where braking is what limits the speed (below the cornering limit, and
    // falling, so the acceleration pass is slack), each step must be the
    // equality v_i^2 = v_{i+1}^2 + 2 a_brake sqrt(1 - (v_{i+1}^2 k_{i+1} / a_lat)^2) ds:
    // the spec in racing_math.compute_velocity_profile's docstring.
    int checked = 0;
    for (size_t i = 0; i + 1 < v.size(); ++i) {
        if (v[i] >= std::sqrt(2.5 / 0.25) - 1e-6 || v[i + 1] >= v[i]) {
            continue;
        }
        const double lat = v[i + 1] * v[i + 1] * k[i + 1] / 2.5;
        const double expected =
            v[i + 1] * v[i + 1] + 2.0 * 3.0 * std::sqrt(std::max(0.0, 1.0 - lat * lat)) * p.segment(i);
        EXPECT_NEAR(v[i] * v[i], expected, 1e-9) << "at " << i << ", lateral load " << lat;   // m^2/s^2
        ++checked;
    }
    EXPECT_GT(checked, 20);   // the braking zone really was there (not a vacuous loop)
}

TEST(SpeedProfile, FrictionEllipseChangesNothingWhereThereIsNoCornering)
{
    // A straight has k = 0, so the full budget: the stop ramp is still the
    // closed form sqrt(2 * a_brake * d).
    const Path s(straight(), 0.5);
    const auto v = s.speed_profile(kLimits, true);
    EXPECT_NEAR(v[190], std::sqrt(2.0 * 3.0 * 1.0), 1e-9);   // m/s, 1 m before the end
    // A circle runs at one speed, so neither pass ever binds: unchanged.
    const Path c(circle(2.0, 200), 0.5);
    for (double x : c.speed_profile(kLimits, false)) {
        EXPECT_NEAR(x, std::sqrt(2.5 / 0.5), 1e-9);   // m/s
    }
}

// --- Path: nearest ---------------------------------------------------------------

TEST(Path, NearestNearOnlyLooksInsideItsWindow)
{
    const Path p(straight(), 0.5);
    EXPECT_EQ(p.nearest(15.04, 0.3).first, 150u);
    EXPECT_NEAR(p.nearest(15.04, 0.3).second, std::hypot(0.04, 0.3), 1e-9);
    // window [50 - 5, 50 + 5]: the best it can do is index 55
    EXPECT_EQ(p.nearest_near(15.0, 0.0, 50, 5, 5).first, 55u);
}

// --- Tracker -----------------------------------------------------------------------

TEST(Tracker, CountsDistanceDrivenForwardAndBackward)
{
    const Path p(straight(), 0.5);
    Tracker t(p, 0, 1.5);
    t.update(5.0, 0.1);
    EXPECT_EQ(t.closest(), 50u);
    EXPECT_DOUBLE_EQ(t.travelled(), 0.0);
    t.update(6.0, 0.1);
    EXPECT_NEAR(t.travelled(), 1.0, 1e-9);     // closed form: s(60) - s(50)
    t.update(5.5, -0.1);
    EXPECT_NEAR(t.travelled(), 0.5, 1e-9);     // backward counts negative
}

TEST(Tracker, AJumpIsNotCountedAsDriving)
{
    const Path p(straight(), 0.5);
    Tracker t(p, 0, 1.5);
    t.update(5.0, 0.0);
    const auto u = t.update(15.0, 0.0);        // 10 m: beyond the search window
    EXPECT_TRUE(u.jumped);
    EXPECT_EQ(u.closest, 150u);
    EXPECT_DOUBLE_EQ(t.travelled(), 0.0);
}

TEST(Tracker, ReportsHowFarTheCarIsFromThePath)
{
    const Path p(straight(), 0.5);
    Tracker t(p, 0, 1.5);
    EXPECT_NEAR(t.update(3.0, 2.5).off_path, 2.5, 1e-9);
}

TEST(Tracker, OpenPathRemainingIsDistanceToTheEnd)
{
    const Path p(straight(), 0.5);
    Tracker t(p, 3, 1.5);                      // laps mean nothing on an open path
    t.update(5.0, 0.0);
    EXPECT_NEAR(t.remaining(), 15.0, 1e-9);
    EXPECT_EQ(t.laps_done(), 0);
}

TEST(Tracker, LapsAcrossTheStartLine)
{
    const int n = 200;
    const Path p(circle(2.0, n), 0.5);
    const double lap = polygon_perimeter(2.0, n);
    Tracker t(p, 2, 1.5);
    // drive 1.5 laps, a waypoint at a time, starting at waypoint 190
    for (int step = 0; step <= 300; ++step) {
        const Waypoint& w = p[(190 + step) % n];
        t.update(w.x, w.y);
    }
    // closed form: 300 segments of one lap's 200
    EXPECT_NEAR(t.travelled(), 1.5 * lap, 1e-9);
    EXPECT_EQ(t.laps_done(), 1);
    EXPECT_NEAR(t.remaining(), 2.0 * lap - 1.5 * lap, 1e-9);
    t.restart_run();
    EXPECT_DOUBLE_EQ(t.travelled(), 0.0);
}

TEST(Tracker, ALoopWithoutLapsHasNoGoal)
{
    const Path p(circle(2.0, 100), 0.5);
    Tracker t(p, 0, 1.5);
    t.update(2.0, 0.0);
    EXPECT_EQ(t.remaining(), kInf);
}

// --- control ---------------------------------------------------------------------------

TEST(Control, RearAxleIsBehindTheSensorAlongTheHeading)
{
    // closed form: rear = sensor - R(yaw) * offset
    Pose r = rear_axle({1.0, 2.0, 0.0}, 0.26, 0.0);
    EXPECT_NEAR(r.x, 0.74, 1e-12);
    EXPECT_NEAR(r.y, 2.0, 1e-12);
    r = rear_axle({1.0, 2.0, M_PI / 2}, 0.26, 0.0);
    EXPECT_NEAR(r.x, 1.0, 1e-12);
    EXPECT_NEAR(r.y, 1.74, 1e-12);
    r = rear_axle({0.0, 0.0, 0.0}, 0.26, 0.05);      // sensor also 5 cm left
    EXPECT_NEAR(r.y, -0.05, 1e-12);
    EXPECT_NEAR(r.yaw, 0.0, 1e-12);
}

TEST(Control, CarFrameIsAheadAndLeft)
{
    double lx, ly;
    to_car_frame({1.0, 1.0, M_PI / 2}, 1.0, 3.0, lx, ly);   // facing +y, point 2 m in +y
    EXPECT_NEAR(lx, 2.0, 1e-12);
    EXPECT_NEAR(ly, 0.0, 1e-12);
    to_car_frame({0.0, 0.0, 0.0}, 0.0, 1.0, lx, ly);        // facing +x, point to the left
    EXPECT_NEAR(ly, 1.0, 1e-12);
}

TEST(Control, LookaheadGrowsWithSpeedInsideItsClamp)
{
    EXPECT_DOUBLE_EQ(lookahead(3.0, 1.0, 0.1, 0.8, 3.0), 1.3);
    EXPECT_DOUBLE_EQ(lookahead(-3.0, 1.0, 0.1, 0.8, 3.0), 1.3);   // reversing: |speed|
    EXPECT_DOUBLE_EQ(lookahead(0.0, 0.5, 0.1, 0.8, 3.0), 0.8);
    EXPECT_DOUBLE_EQ(lookahead(40.0, 1.0, 0.1, 0.8, 3.0), 3.0);
}

TEST(Control, GoalIsTheFirstWaypointAheadAndAtLeastLAway)
{
    const Path p(straight(201, 0.1, 0.2), 0.5);
    const auto g = find_goal(p, 0, {0.0, 0.0, 0.0}, 1.0);
    ASSERT_TRUE(g.has_value());
    // closed form: first x on the 0.1 m grid with x^2 + 0.2^2 >= 1 is x = 1.0
    EXPECT_EQ(g->index, 10u);
    EXPECT_NEAR(g->local_x, 1.0, 1e-9);
    EXPECT_NEAR(g->local_y, 0.2, 1e-12);
    EXPECT_FALSE(g->past_end);
}

TEST(Control, NearAnOpenPathsEndTheGoalIsPastItAlongTheLastSegment)
{
    const Path p(straight(), 0.5);                       // ends at x = 20
    const auto g = find_goal(p, 195, {19.5, 0.0, 0.0}, 1.0);
    ASSERT_TRUE(g.has_value());
    EXPECT_TRUE(g->past_end);
    EXPECT_EQ(g->index, 200u);
    // closed form: the end plus (L - 0.5) along +x, seen from x = 19.5
    EXPECT_NEAR(g->local_x, 1.0, 1e-9);
    EXPECT_NEAR(g->local_y, 0.0, 1e-12);
}

TEST(Control, AnOpenPathNeverWrapsToItsStart)
{
    const Path p(straight(), 0.5);
    // facing back toward the start: everything ahead of the car is before
    // the search's starting waypoint, and the path's end is behind it
    EXPECT_FALSE(find_goal(p, 195, {19.5, 0.0, M_PI}, 1.0).has_value());
}

TEST(Control, ALoopWrapsPastItsLastWaypoint)
{
    const Path p(circle(2.0, 100), 0.5);
    const Waypoint& w = p[98];
    const double heading = std::atan2(w.x, -w.y);   // tangent, counter-clockwise
    const auto g = find_goal(p, 98, {w.x, w.y, heading}, 1.0);
    ASSERT_TRUE(g.has_value());
    EXPECT_LT(g->index, 98u);                         // wrapped round to the start
}

TEST(Control, SteeringIsTheBicycleModelOfThePursuitArc)
{
    // closed form: atan(wheelbase * 2y/d^2)
    EXPECT_NEAR(steering_for(1.0, 0.2, 0.36, 0.33), std::atan(0.36 * 2 * 0.2 / 1.04), 1e-12);
    EXPECT_NEAR(steering_for(1.0, -0.2, 0.36, 0.33), -std::atan(0.36 * 2 * 0.2 / 1.04), 1e-12);
    EXPECT_DOUBLE_EQ(steering_for(0.9, 0.5, 0.36, 0.30), 0.30);     // atan(...) = 0.3275: clamped
    EXPECT_DOUBLE_EQ(steering_for(0.9, -0.5, 0.36, 0.30), -0.30);
}

TEST(Control, SteeringSpeedIsVmaxStraightAndVminAtFullLock)
{
    // v_mid halfway (2.75): pure_pursuit_a's straight line
    EXPECT_DOUBLE_EQ(steering_speed(0.0, 0.3, 4.0, 2.75, 1.5), 4.0);
    EXPECT_DOUBLE_EQ(steering_speed(-0.3, 0.3, 4.0, 2.75, 1.5), 1.5);
    EXPECT_DOUBLE_EQ(steering_speed(0.15, 0.3, 4.0, 2.75, 1.5), 2.75);
    // closed form of that line, v_max - (v_max - v_min) |s| / max, at angles off the knots
    for (double s : {0.03, -0.09, 0.2, 0.27}) {
        EXPECT_NEAR(steering_speed(s, 0.3, 4.0, 2.75, 1.5), 4.0 - 2.5 * std::abs(s) / 0.3, 1e-12) << s;
    }
}

TEST(Control, SteeringSpeedBendsAtHalfLock)
{
    // v_max 4.0 straight, v_mid 3.5 at half lock (0.15 rad), v_min 1.5 at full lock (0.3 rad)
    EXPECT_DOUBLE_EQ(steering_speed(0.0, 0.3, 4.0, 3.5, 1.5), 4.0);
    EXPECT_DOUBLE_EQ(steering_speed(0.15, 0.3, 4.0, 3.5, 1.5), 3.5);
    EXPECT_DOUBLE_EQ(steering_speed(-0.15, 0.3, 4.0, 3.5, 1.5), 3.5);   // either way
    EXPECT_DOUBLE_EQ(steering_speed(0.3, 0.3, 4.0, 3.5, 1.5), 1.5);
    // closed form: straight lines between the knots
    EXPECT_NEAR(steering_speed(0.075, 0.3, 4.0, 3.5, 1.5), (4.0 + 3.5) / 2, 1e-12);    // a quarter lock
    EXPECT_NEAR(steering_speed(-0.225, 0.3, 4.0, 3.5, 1.5), (3.5 + 1.5) / 2, 1e-12);   // three quarters
    // either side of the knot, so the bend is exactly at half lock
    EXPECT_NEAR(steering_speed(0.144, 0.3, 4.0, 3.5, 1.5), 4.0 - 0.5 * 0.96, 1e-12);   // 0.48 of lock
    EXPECT_NEAR(steering_speed(0.156, 0.3, 4.0, 3.5, 1.5), 3.5 - 2.0 * 0.04, 1e-12);   // 0.52 of lock
    // past full lock, or no sensible angle at all: the slowest point, never extrapolated
    EXPECT_DOUBLE_EQ(steering_speed(0.6, 0.3, 4.0, 3.5, 1.5), 1.5);
    EXPECT_DOUBLE_EQ(steering_speed(std::numeric_limits<double>::quiet_NaN(), 0.3, 4.0, 3.5, 1.5), 1.5);
    EXPECT_DOUBLE_EQ(steering_speed(-kInf, 0.3, 4.0, 3.5, 1.5), 1.5);
    // a bend the other way: drop early, hold through tight steering
    EXPECT_NEAR(steering_speed(0.075, 0.3, 4.0, 2.0, 1.5), (4.0 + 2.0) / 2, 1e-12);
    EXPECT_NEAR(steering_speed(0.225, 0.3, 4.0, 2.0, 1.5), (2.0 + 1.5) / 2, 1e-12);
}

TEST(Control, GoalRampStopsByTheGoal)
{
    EXPECT_EQ(goal_ramp(kInf, 0.3, 3.0), kInf);                   // no goal
    EXPECT_DOUBLE_EQ(goal_ramp(0.3, 0.3, 3.0), 0.0);             // exactly at the tolerance
    EXPECT_DOUBLE_EQ(goal_ramp(0.1, 0.3, 3.0), 0.0);             // inside it
    EXPECT_NEAR(goal_ramp(1.3, 0.3, 3.0), std::sqrt(2.0 * 3.0 * 1.0), 1e-12);   // closed form
}

TEST(Control, AccelerationIsLimitedButBrakingIsNot)
{
    EXPECT_NEAR(limit_acceleration(4.0, 1.0, 3.0, 0.1), 1.3, 1e-12);   // 1 + 3 * 0.1
    EXPECT_NEAR(limit_acceleration(4.0, 1.0, 3.0, 5.0), 1.6, 1e-12);   // dt clamped to 0.2 s
    EXPECT_DOUBLE_EQ(limit_acceleration(4.0, 1.0, 3.0, -1.0), 1.0);    // negative dt: no step
    EXPECT_DOUBLE_EQ(limit_acceleration(0.5, 3.0, 3.0, 0.01), 0.5);    // slowing: immediate
}
