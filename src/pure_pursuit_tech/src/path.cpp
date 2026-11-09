#include "path.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

std::vector<Waypoint> parse_waypoints_csv(std::istream& in)
{
    std::vector<Waypoint> points;
    std::string line;
    while (std::getline(in, line)) {
        // ignore empty lines or any starting with # (comments)
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::stringstream ss(line);
        std::string x_str, y_str;
        if (!std::getline(ss, x_str, ',') || !std::getline(ss, y_str, ',')) {
            continue;
        }
        try {
            size_t used_x = 0, used_y = 0;
            const Waypoint wp{std::stod(x_str, &used_x), std::stod(y_str, &used_y)};
            const bool whole = x_str.find_first_not_of(" \t\r", used_x) == std::string::npos &&
                               y_str.find_first_not_of(" \t\r", used_y) == std::string::npos;
            const bool repeat = !points.empty() &&
                                std::hypot(wp.x - points.back().x, wp.y - points.back().y) <= 1e-6;
            if (whole && std::isfinite(wp.x) && std::isfinite(wp.y) && !repeat) {
                points.push_back(wp);
            }
        } catch (const std::exception&) {
            // ignore rows that aren't numbers (e.g. an "x,y,speed" header)
        }
    }
    return points;
}

Path::Path(std::vector<Waypoint> points, double closure_distance) : points_(std::move(points))
{
    if (points_.size() < 3) {
        throw std::invalid_argument("a path needs at least 3 distinct waypoints");
    }

    //measure the map between last and first
    const double gap = std::hypot(points_.back().x - points_.front().x,
                                  points_.back().y - points_.front().y);
    closed_ = gap <= closure_distance;

    //remove a rpeeated endpoint
    if (closed_ && gap <= 1e-6) {
        points_.pop_back();   // a loop written with its first point repeated at the end
        if (points_.size() < 3) { //require enough points
            throw std::invalid_argument("a path needs at least 3 distinct waypoints");
        }
    }


    const size_t n = points_.size();
    //calculate each segment length
    seg_.assign(n, 0.0);

    for (size_t i = 0; i + 1 < n; ++i) {
        seg_[i] = std::hypot(
            points_[i + 1].x - points_[i].x, 
            points_[i + 1].y - points_[i].y);
    }

    if (closed_) {
        seg_[n - 1] = std::hypot(points_[0].x - points_[n - 1].x, points_[0].y - points_[n - 1].y);
    }
    s_.assign(n, 0.0);
    for (size_t i = 1; i < n; ++i) {
        s_[i] = s_[i - 1] + seg_[i - 1]; //calculate cumulative distance, s_[i] means distance along the path from the first waypoint to waypoint i
    }

    //total length
    length_ = closed_ ? s_[n - 1] + seg_[n - 1] : s_[n - 1];
    if (!(length_ > 0.0)) {
        throw std::invalid_argument("waypoint path has no length");
    }
}

size_t Path::advance(size_t from, size_t steps) const
{
    if (closed_) {
        return (from + steps) % points_.size();
    }
    return std::min(from + steps, points_.size() - 1);
}

std::vector<double> Path::curvature(double chord) const //measure how sharply the path bends
{
    const long n = static_cast<long>(points_.size());

    std::vector<double> k(points_.size(), 0.0); //returns one curvature

    const double mean = length_ / static_cast<double>(closed_ ? n : n - 1); //average waypoint

    const long step = std::max(1L, std::lround(chord / std::max(mean, 1e-6)));
    for (long i = 0; i < n; ++i) {
        long a = i - step;
        long b = i + step;
        if (closed_) {
            a = ((a % n) + n) % n;
            b = b % n;
        } else {
            a = std::max(0L, a); //never below index 0
            b = std::min(n - 1, b); //never above the last index
            if (a == i || b == i) {
                continue;   // an endpoint: no circle through it, call it straight
            }
        }
        const Waypoint& p = points_[static_cast<size_t>(a)]; //point before q
        const Waypoint& q = points_[static_cast<size_t>(i)]; //current point
        const Waypoint& r = points_[static_cast<size_t>(b)]; //point after q

        const double pq = std::hypot(q.x - p.x, q.y - p.y); 
        const double qr = std::hypot(r.x - q.x, r.y - q.y);
        const double pr = std::hypot(r.x - p.x, r.y - p.y);
        const double cross = (q.x - p.x) * (r.y - p.y) - (q.y - p.y) * (r.x - p.x);
        const double denom = pq * qr * pr;
        k[static_cast<size_t>(i)] = denom > 1e-12 ? std::abs(2.0 * cross / denom) : 0.0;
    }
    return k;
}

//assign planned speed to each waypoint
std::vector<double> Path::speed_profile(const SpeedLimits& lim, bool stop_at_end) const
{
    const std::vector<double> k = curvature(); //choose speed from curvature
    const size_t n = k.size();
    std::vector<double> v(n, lim.v_max); //node setting from load_path() in pp_tech_node
    for (size_t i = 0; i < n; ++i) {

        if (k[i] > 1e-9) {
            v[i] = std::min(lim.v_max, std::sqrt(lim.a_lat / k[i])); //set a speed on each points curvature
        }
        // No v_min floor: the cornering limit wins. A corner that needs
        // 0.4 m/s stays at 0.4 m/s even with min_speed 0.5.
    }
    if (!closed_ && stop_at_end) {
        v[n - 1] = 0.0;
    }
    // Friction ellipse: cornering uses grip too, so braking and accelerating
    // get only what is left of their budget,
    //   (a_long / a_long_max)^2 + (a_lat / a_lat)^2 <= 1
    //   a_long_available = a_long_max * sqrt(1 - (v^2 k / a_lat)^2),
    // taken at the neighbour whose speed each pass already knows. Full budget
    // on a straight, none at a corner already at its cornering limit. Same as
    // racing_math.compute_velocity_profile(friction_ellipse=True).
    const auto long_fraction = [&](double speed, double kappa) {
        if (!lim.friction_ellipse) {
            return 1.0;
        }
        const double lat = speed * speed * kappa / lim.a_lat;
        return std::sqrt(std::max(0.0, 1.0 - lat * lat));   // < 0 only by round-off
    };
    // A loop has no first point to start from: sweep it a few times so the
    // start/finish seam settles. Both passes only ever lower speeds.
    const int sweeps = closed_ ? 3 : 1;
    for (int pass = 0; pass < sweeps; ++pass) {
        for (size_t step = 0; step < n; ++step) {   // braking: backward
            const size_t i = n - 1 - step;
            if (i + 1 == n && !closed_) {
                continue;
            }
            const size_t next = (i + 1) % n;

            const double a_brake = lim.a_brake * long_fraction(v[next], k[next]);
            v[i] = std::min( //backword to start braking before corners
                v[i], 
                std::sqrt(v[next] * v[next] + 2.0 * a_brake * seg_[i])); //v^2 = u^2 +2as
        }

        for (size_t i = 0; i < n; ++i) {            // acceleration: forward
            if (i == 0 && !closed_) {
                continue;
            }
            const size_t prev = (i + n - 1) % n;

            const double a_accel = lim.a_accel * long_fraction(v[prev], k[prev]);
            v[i] = std::min( //foward to limit acceleration after corners
                v[i], 
                std::sqrt(v[prev] * v[prev] + 2.0 * a_accel * seg_[prev]));
        }
    }
    return v;
}

std::pair<size_t, double> Path::nearest(double x, double y) const
{
    size_t best = 0;
    double best_d = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < points_.size(); ++i) {
        const double d = std::hypot(points_[i].x - x, points_[i].y - y);
        if (d < best_d) {
            best_d = d;
            best = i;
        }
    }
    return {best, best_d};
}

std::pair<size_t, double> Path::nearest_near(double x, double y, size_t center,
                                             size_t behind, size_t ahead) const
{
    const long n = static_cast<long>(points_.size());
    size_t best = center;
    double best_d = std::numeric_limits<double>::infinity();
    for (long j = static_cast<long>(center) - static_cast<long>(behind);
         j <= static_cast<long>(center) + static_cast<long>(ahead); ++j) {
        long i = j;
        if (closed_) {
            i = ((j % n) + n) % n;
        } else if (j < 0 || j >= n) {
            continue;
        }
        const double d = std::hypot(points_[static_cast<size_t>(i)].x - x,
                                    points_[static_cast<size_t>(i)].y - y);
        if (d < best_d) {
            best_d = d;
            best = static_cast<size_t>(i);
        }
    }
    return {best, best_d};
}
