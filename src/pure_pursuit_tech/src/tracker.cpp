#include "tracker.hpp"

#include <cmath>

namespace
{
// Search near where the car was, more ahead than behind.
constexpr size_t kBehind = 20;
constexpr size_t kAhead = 60;
}  // namespace

Tracker::Tracker(const Path& path, int laps, double max_jump)
: path_(path), laps_(laps), max_jump_(max_jump)
{
}

Tracker::Update Tracker::update(double x, double y)
{
    std::pair<size_t, double> found;
    bool jumped = false;
    if (tracking_) {
        found = path_.nearest_near(x, y, closest_, kBehind, kAhead);
    }
    if (!tracking_ || found.second > max_jump_) {
        found = path_.nearest(x, y);
        jumped = tracking_;
    }
    if (tracking_ && !jumped) {
        double delta = path_.s(found.first) - path_.s(closest_);
        if (path_.closed()) {   // across the start/finish line
            if (delta > path_.length() / 2.0) {
                delta -= path_.length();
            } else if (delta < -path_.length() / 2.0) {
                delta += path_.length();
            }
        }
        travelled_ += delta;
    }
    closest_ = found.first;
    tracking_ = true;
    return {found.first, found.second, jumped};
}

int Tracker::laps_done() const
{
    if (!path_.closed()) {
        return 0;
    }
    return static_cast<int>(std::floor(travelled_ / path_.length()));
}

double Tracker::remaining() const
{
    if (!path_.closed()) {
        return path_.length() - path_.s(closest_);
    }
    if (laps_ > 0) {
        return laps_ * path_.length() - travelled_;
    }
    return std::numeric_limits<double>::infinity();
}
