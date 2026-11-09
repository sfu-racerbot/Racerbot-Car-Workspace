#pragma once
// Where the car is along the path: the nearest waypoint, the distance driven
// since the run started, laps, and what is left to the goal. No ROS.

#include <cstddef>
#include <limits>

#include "path.hpp"

class Tracker
{
public:
    /// `laps`: on a loop, the goal is the start/finish line after this many
    /// laps (0 = no goal). An open path's goal is always its last waypoint.
    /// `max_jump`: a nearest-waypoint search further than this from the car
    /// is retried over the whole path (start-up, or a localization jump).
    Tracker(const Path& path, int laps, double max_jump);

    struct Update
    {
        size_t closest;      // nearest waypoint
        double off_path;     // distance from the car to it (m)
        bool jumped;         // re-found by a whole-path search: not counted as driving
    };
    /// The car (rear axle) is at (x, y): move along the path and add the
    /// distance driven. A step backward counts negative.
    Update update(double x, double y);

    bool tracking() const { return tracking_; }
    size_t closest() const { return closest_; }
    double travelled() const { return travelled_; }
    /// Whole laps driven (loops; 0 for open paths).
    int laps_done() const;
    /// Metres left to the goal; infinity when there is none.
    double remaining() const;
    /// A new run from where the car is: distance and laps back to 0.
    void restart_run() { travelled_ = 0.0; }

private:
    const Path& path_;
    int laps_;
    double max_jump_;
    bool tracking_ = false;
    size_t closest_ = 0;
    double travelled_ = 0.0;
};
