#pragma once
// The racing line: waypoints, their geometry, and the speed planned along
// them. No ROS, so test/test_core.cpp exercises it directly.

#include <cstddef>
#include <istream>
#include <utility>
#include <vector>

struct Waypoint
{
    double x;
    double y;
};

/// Waypoints from a CSV stream: x, y as the first two columns of each row.
/// Skips empty lines, '#' comments, rows that are not numbers (an
/// "x,y,speed" header), non-finite values and repeats of the previous point.
std::vector<Waypoint> parse_waypoints_csv(std::istream& in);

struct SpeedLimits
{
    double v_max;    // m/s, never faster
    double v_min;    // m/s, NOT a floor: speed_profile lets the cornering limit go below it
    double a_lat;    // m/s^2, cornering: v <= sqrt(a_lat / |curvature|)
    double a_accel;  // m/s^2, how fast speed may rise along the path
    double a_brake;  // m/s^2, how fast speed may fall: makes the braking zones
    // true: a_accel and a_brake shrink while cornering (friction ellipse, see
    // speed_profile). false: the old independent limits, for comparisons.
    bool friction_ellipse = true;
};

class Path
{
public:
    /// A loop when the last point is within `closure_distance` of the first;
    /// otherwise an open path. A loop's repeated first point is dropped.
    /// Throws std::invalid_argument for fewer than 3 points or no length.
    Path(std::vector<Waypoint> points, double closure_distance);

    bool closed() const { return closed_; }
    size_t size() const { return points_.size(); }
    const Waypoint& operator[](size_t i) const { return points_[i]; }
    const Waypoint& back() const { return points_.back(); }
    /// One lap (loop), or start to end (open path), in metres.
    double length() const { return length_; }
    /// Distance along the path to waypoint i (s(0) = 0).
    double s(size_t i) const { return s_[i]; }
    /// Length of the segment from waypoint i to the next one.
    double segment(size_t i) const { return seg_[i]; }
    /// Index i steps on from `from`; on a loop it wraps, on an open path it stops at the end.
    size_t advance(size_t from, size_t steps) const;

    /// Unsigned curvature (1/m) at each waypoint, from the circle through the
    /// points about `chord` metres before and after it (neighbours would make
    /// waypoint noise into curvature spikes). Open-path endpoints read 0.
    std::vector<double> curvature(double chord = 0.5) const;

    /// Planned speed at each waypoint: the cornering limit (never raised to
    /// v_min: a corner that needs less than v_min gets less), then a braking
    /// pass (slow down *before* a corner) and an acceleration pass, both
    /// sharing grip with cornering through the friction ellipse. An open
    /// path with `stop_at_end` falls to 0 at its last point. Same method as
    /// the workspace's racing_math.compute_velocity_profile.
    std::vector<double> speed_profile(const SpeedLimits& limits, bool stop_at_end) const;

    /// Nearest waypoint to (x, y) and its distance, over the whole path.
    std::pair<size_t, double> nearest(double x, double y) const;
    /// Nearest waypoint within [center - behind, center + ahead] (wrapping on a loop).
    std::pair<size_t, double> nearest_near(double x, double y, size_t center,
                                           size_t behind, size_t ahead) const;

private:
    std::vector<Waypoint> points_;
    bool closed_;
    std::vector<double> seg_;
    std::vector<double> s_;
    double length_;
};
