#pragma once
// ROS glue for pure pursuit: settings, the two safety gates, topics. The
// logic lives in path.hpp, tracker.hpp and control.hpp (no ROS).

#include <memory>
#include <deque>
#include <optional>
#include <string>
#include <vector>

#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"
#include "control.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "path.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joy.hpp"
#include "std_msgs/msg/bool.hpp"
#include "tracker.hpp"

class PurePursuit : public rclcpp::Node
{
public:
    PurePursuit();

private:
    struct Settings
    {
        bool use_pf;                 // pose from the particle filter (car) or odom_topic (sim)
        double laser_offset_x;       // pose point ahead of the rear axle (m)
        double laser_offset_y;       // ... and left of it (m)
        int deadman_button;          // 4 = LB on the F710 in XInput mode
        double joy_timeout_sec;      // /joy older than this = released
        bool enable_safety;          // racerbot_a's safety_node gates this node
        double safety_timeout_sec;   // /safety/brake older than this = brake
        double lookahead_base;       // m
        double lookahead_min;        // m
        double lookahead_max;        // m
        double lookahead_speed_gain; // m per m/s
        double lookahead_gain_straight;    // m per m/s on straights (-1: lookahead_speed_gain)
        double lookahead_gain_corner;      // m per m/s in corners (-1: lookahead_speed_gain)
        double lookahead_corner_curvature; // 1/m: a full corner
        double lookahead_preview;          // m ahead to look for a corner
        double wheelbase;            // m
        double max_steering;         // rad
        double max_speed;            // m/s
        double min_speed;            // m/s
        double medium_angle_speed;   // m/s at half of max_steering (-1 given = halfway)
        bool profile_mode;           // speed_mode: profile (true) or steering (pure_pursuit_a's rule)
        double max_lateral_accel;    // m/s^2
        double max_accel;            // m/s^2
        double max_brake;            // m/s^2
        int laps;                    // loop: stop after this many laps (0 = never)
        double goal_tolerance;       // m
        double max_path_error;       // m
        double max_pose_age;         // s: older (or unstamped) poses stop the car
        bool compensate_latency;     // aim from the pose predicted at now + actuation_delay
        double actuation_delay;      // s: command to wheels
        double max_steering_rate;    // rad/s, both directions
    };
    Settings cfg_;

    rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr joy_sub_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr safety_brake_sub_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr obs_fallback_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr speed_sub_;
    rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr drive_pub_;

    // LB deadman (workspace policy, docs/architecture.md): checked here
    // whatever racerbot_a's safety_node says. It cannot be switched off.
    std::optional<rclcpp::Time> last_joy_time_;
    bool deadman_held_ = false;
    bool was_engaged_ = false;          // to see LB pressed again after a finished run

    // racerbot_a's safety_node (deadman + iTTC) on /safety/brake
    std::optional<rclcpp::Time> last_safety_time_;
    bool safety_brake_engaged_ = true;  // brake until safety_node says otherwise
    bool obs_fallback_engaged_ = false;

    std::unique_ptr<Path> path_;
    std::unique_ptr<Tracker> tracker_;
    std::vector<double> profile_;       // planned speed per waypoint (profile mode)
    std::vector<double> curvature_;     // per waypoint, for the lookahead gain
    int laps_logged_ = 0;
    bool finished_ = false;             // reached the goal: stopped until LB is pressed again

    double current_speed_ = 0.0;        // from odom_topic when use_pf
    double last_cmd_speed_ = 0.0;       // for the acceleration limit
    std::optional<rclcpp::Time> last_cmd_time_;
    std::deque<SteerCommand> steer_history_;   // recent commands, for latency compensation
    double last_steering_ = 0.0;        // for the steering rate limit
    std::optional<rclcpp::Time> last_steer_time_;

    Settings read_settings();
    void load_path(const std::string& file, double closure_distance);

    void joy_callback(const sensor_msgs::msg::Joy::ConstSharedPtr msg);
    void safety_brake_callback(const std_msgs::msg::Bool::ConstSharedPtr msg);
    void obs_fallback_callback(const std_msgs::msg::Bool::ConstSharedPtr msg);
    void speed_callback(nav_msgs::msg::Odometry::ConstSharedPtr msg);

    /// True only while LB is held on a /joy message younger than joy_timeout_sec.
    bool deadman_engaged(std::string& reason) const;
    /// True while racerbot_a's safety_node is braking, silent, or stale.
    bool safety_braking(std::string& reason) const;

    void remember_steering(double steering);
    /// Publishes speed 0, steering 0, and says why (throttled).
    void stop(const std::string& reason);
    /// Steering and speed for a goal point, then publishes the command.
    void drive_to_goal(const Goal& goal, double speed, double remaining);
    /// Every pose: gates, where on the path, goal point, drive.
    void odom_callback(nav_msgs::msg::Odometry::ConstSharedPtr msg);
};
