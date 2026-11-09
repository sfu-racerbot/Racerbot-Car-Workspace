#include "pure_pursuit_tech_node.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>

#include "ament_index_cpp/get_package_share_directory.hpp"

using std::placeholders::_1;

PurePursuit::PurePursuit() : Node("pure_pursuit_tech_node")
{
    // topics and the line
    const std::string odom_topic = declare_parameter("odom_topic", std::string("/odom"));
    const std::string pose_param = declare_parameter("pose_topic", std::string("/pf/pose/odom"));
    const std::string drive_topic = declare_parameter("drive_topic", std::string("/drive"));
    const std::string joy_topic = declare_parameter("joy_topic", std::string("/joy"));
    const std::string safety_topic = declare_parameter("safety_topic", std::string("/safety/brake"));
    const std::string obstacle_topic = declare_parameter("obstacle_topic", std::string("obstacle_ahead"));
    const std::string track = declare_parameter("track", std::string(""));
    const std::string waypoints_file = declare_parameter("waypoints_file", std::string(""));
    const double closure_distance = declare_parameter("loop_closure_distance", 0.5);
    cfg_ = read_settings();
    if (!(closure_distance >= 0.0)) {
        throw std::invalid_argument("loop_closure_distance must be >= 0");
    }
    RCLCPP_INFO(get_logger(), "Steering speed: %.2f m/s straight, %.2f at %.3f rad, %.2f at full lock %.3f rad",
                cfg_.max_speed, cfg_.medium_angle_speed, 0.5 * cfg_.max_steering, cfg_.min_speed,
                cfg_.max_steering);

    // LB deadman, checked here independently of racerbot_a's safety_node
    joy_sub_ = create_subscription<sensor_msgs::msg::Joy>(
        joy_topic, 10, std::bind(&PurePursuit::joy_callback, this, _1));
    // Safety: racerbot_a's safety_node
    safety_brake_sub_ = create_subscription<std_msgs::msg::Bool>(
        safety_topic, 10, std::bind(&PurePursuit::safety_brake_callback, this, _1));
    // Obstacle detection fallback: another node drives while this is true
    obs_fallback_sub_ = create_subscription<std_msgs::msg::Bool>(
        obstacle_topic, 10, std::bind(&PurePursuit::obs_fallback_callback, this, _1));

    // waypoints: an absolute file (a saved line) or a CSV in the package's waypoints folder
    std::string file = waypoints_file;
    if (file.empty()) {
        if (track.empty()) {
            throw std::invalid_argument("set track (waypoints/<track>.csv) or waypoints_file");
        }
        file = ament_index_cpp::get_package_share_directory("pure_pursuit_tech") +
               "/waypoints/" + track + ".csv";
    }
    load_path(file, closure_distance);

    // pose: the particle filter on the real car, odom_topic in the simulator
    const std::string pose_topic = cfg_.use_pf ? pose_param : odom_topic;
    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
        pose_topic, 10, std::bind(&PurePursuit::odom_callback, this, _1));
    // the PF usually doesn't fill in twist, so take speed from wheel odom
    if (cfg_.use_pf) {
        speed_sub_ = create_subscription<nav_msgs::msg::Odometry>(
            odom_topic, 10, std::bind(&PurePursuit::speed_callback, this, _1));
    }
    drive_pub_ = create_publisher<ackermann_msgs::msg::AckermannDriveStamped>(drive_topic, 10);

    RCLCPP_INFO(get_logger(),
                "Loaded %zu waypoints from %s: %s, %.1f m; speed_mode %s; pose %s, drive %s, safety %s",
                path_->size(), file.c_str(), path_->closed() ? "loop" : "open path", path_->length(),
                cfg_.profile_mode ? "profile" : "steering", pose_topic.c_str(), drive_topic.c_str(),
                cfg_.enable_safety ? safety_topic.c_str() : "off (LB deadman still on)");
}

PurePursuit::Settings PurePursuit::read_settings()
{
    Settings c;
    c.use_pf = declare_parameter("use_pf", true);
    c.laser_offset_x = declare_parameter("laser_offset_x", 0.0);
    c.laser_offset_y = declare_parameter("laser_offset_y", 0.0);
    c.deadman_button = static_cast<int>(declare_parameter("deadman_button", 4));
    c.joy_timeout_sec = declare_parameter("joy_timeout_sec", 0.5);
    c.enable_safety = declare_parameter("enable_safety", true);
    c.safety_timeout_sec = declare_parameter("safety_timeout_sec", 0.25);
    c.lookahead_base = declare_parameter("lookahead_base", 1.0);
    c.lookahead_min = declare_parameter("lookahead_min", 0.8);
    c.lookahead_max = declare_parameter("lookahead_max", 3.0);
    c.lookahead_speed_gain = declare_parameter("lookahead_speed_gain", 0.1);
    c.wheelbase = declare_parameter("wheelbase", 0.36);
    c.max_steering = declare_parameter("max_steering", 0.30);
    c.max_speed = declare_parameter("max_speed", 1.0);
    c.min_speed = declare_parameter("min_speed", 0.5);
    c.medium_angle_speed = declare_parameter("medium_angle_speed", -1.0);
    const std::string speed_mode = declare_parameter("speed_mode", std::string("steering"));
    c.max_lateral_accel = declare_parameter("max_lateral_accel", 2.5);
    c.max_accel = declare_parameter("max_accel", 3.0);
    c.max_brake = declare_parameter("max_brake", 3.0);
    c.laps = static_cast<int>(declare_parameter("laps", 0));
    c.goal_tolerance = declare_parameter("goal_tolerance", 0.3);
    c.max_path_error = declare_parameter("max_path_error", 1.5);

    // Refuse settings that would make the control law divide by zero or
    // command something nobody asked for.
    if (c.deadman_button < 0) {
        throw std::invalid_argument("deadman_button must be >= 0");
    }
    if (!(c.joy_timeout_sec > 0.0) || !(c.safety_timeout_sec > 0.0)) {
        throw std::invalid_argument("joy_timeout_sec and safety_timeout_sec must be > 0");
    }
    if (!(c.lookahead_min > 0.0) || !(c.lookahead_min <= c.lookahead_max)) {
        throw std::invalid_argument("need 0 < lookahead_min <= lookahead_max");
    }
    if (!(c.wheelbase > 0.0) || !(c.max_steering > 0.0)) {
        throw std::invalid_argument("wheelbase and max_steering must be > 0");
    }
    if (!(c.min_speed >= 0.0) || !(c.min_speed <= c.max_speed)) {
        throw std::invalid_argument("need 0 <= min_speed <= max_speed");
    }
    if (c.medium_angle_speed == -1.0) {
        c.medium_angle_speed = 0.5 * (c.max_speed + c.min_speed);   // today's straight line
    } else if (!(c.min_speed <= c.medium_angle_speed) || !(c.medium_angle_speed <= c.max_speed)) {
        throw std::invalid_argument("need min_speed <= medium_angle_speed <= max_speed (or -1 for halfway)");
    }
    if (speed_mode != "steering" && speed_mode != "profile") {
        throw std::invalid_argument("speed_mode must be 'steering' or 'profile'");
    }
    c.profile_mode = speed_mode == "profile";
    if (!(c.max_lateral_accel > 0.0) || !(c.max_accel > 0.0) || !(c.max_brake > 0.0)) {
        throw std::invalid_argument("max_lateral_accel, max_accel and max_brake must be > 0");
    }
    if (c.laps < 0 || !(c.goal_tolerance > 0.0) || !(c.max_path_error > 0.0)) {
        throw std::invalid_argument("need laps >= 0, goal_tolerance > 0 and max_path_error > 0");
    }
    if (!std::isfinite(c.laser_offset_x) || !std::isfinite(c.laser_offset_y)) {
        throw std::invalid_argument("laser_offset_x and laser_offset_y must be finite");
    }
    return c;
}

void PurePursuit::load_path(const std::string& file, double closure_distance)
{
    std::ifstream in(file);
    if (!in.is_open()) {
        throw std::runtime_error("Cannot open waypoint file: " + file);
    }
    std::vector<Waypoint> points = parse_waypoints_csv(in);
    if (points.size() < 3) {
        throw std::runtime_error("Need at least 3 distinct waypoints in " + file);
    }
    path_ = std::make_unique<Path>(std::move(points), closure_distance);
    tracker_ = std::make_unique<Tracker>(*path_, cfg_.laps, cfg_.max_path_error);
    if (cfg_.laps > 0 && !path_->closed()) {
        RCLCPP_WARN(get_logger(), "laps is set but the path is open: it stops at the end instead");
    }
    if (cfg_.profile_mode) {
        profile_ = path_->speed_profile(
            {cfg_.max_speed, cfg_.min_speed, cfg_.max_lateral_accel, cfg_.max_accel, cfg_.max_brake},
            /*stop_at_end=*/!path_->closed());
        const auto [lo, hi] = std::minmax_element(profile_.begin(), profile_.end());
        RCLCPP_INFO(get_logger(), "Speed profile: %.2f to %.2f m/s (lateral %.1f, accel %.1f, brake %.1f m/s^2)",
                    *lo, *hi, cfg_.max_lateral_accel, cfg_.max_accel, cfg_.max_brake);
    }
}

void PurePursuit::joy_callback(const sensor_msgs::msg::Joy::ConstSharedPtr msg)
{
    last_joy_time_ = now();
    // a message too short to hold the button counts as released
    deadman_held_ = static_cast<int>(msg->buttons.size()) > cfg_.deadman_button &&
                    msg->buttons[cfg_.deadman_button] != 0;
}

void PurePursuit::safety_brake_callback(const std_msgs::msg::Bool::ConstSharedPtr msg)
{
    last_safety_time_ = now();
    safety_brake_engaged_ = msg->data;
}

void PurePursuit::obs_fallback_callback(const std_msgs::msg::Bool::ConstSharedPtr msg)
{
    if (msg->data != obs_fallback_engaged_) {
        RCLCPP_INFO(get_logger(), "obstacle fallback %s", msg->data ? "engaged" : "disengaged");
    }
    obs_fallback_engaged_ = msg->data;
}

void PurePursuit::speed_callback(nav_msgs::msg::Odometry::ConstSharedPtr msg)
{
    current_speed_ = msg->twist.twist.linear.x;
}

bool PurePursuit::deadman_engaged(std::string& reason) const
{
    if (!last_joy_time_) {
        reason = "no /joy received yet";
        return false;
    }
    if ((now() - *last_joy_time_).seconds() >= cfg_.joy_timeout_sec) {
        reason = "/joy is stale";
        return false;
    }
    if (!deadman_held_) {
        reason = "LB not held";
        return false;
    }
    return true;
}

bool PurePursuit::safety_braking(std::string& reason) const
{
    if (!cfg_.enable_safety) {
        return false;
    }
    if (!last_safety_time_) {
        reason = "no /safety/brake received yet (is safety_node running?)";
        return true;
    }
    if ((now() - *last_safety_time_).seconds() >= cfg_.safety_timeout_sec) {
        reason = "/safety/brake is stale";
        return true;
    }
    if (safety_brake_engaged_) {
        reason = "safety_node is braking";
        return true;
    }
    return false;
}

void PurePursuit::stop(const std::string& reason)
{
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "Stopped: %s", reason.c_str());
    ackermann_msgs::msg::AckermannDriveStamped msg;
    msg.header.stamp = now();
    msg.header.frame_id = "base_link";
    msg.drive.speed = 0.0;
    msg.drive.steering_angle = 0.0;
    drive_pub_->publish(msg);
    last_cmd_speed_ = 0.0;
    last_cmd_time_ = now();
}

void PurePursuit::drive_to_goal(const Goal& goal, double speed, double remaining)
{
    const double steering = steering_for(goal.local_x, goal.local_y, cfg_.wheelbase, cfg_.max_steering);
    // pure_pursuit_a's rule: slow down with steering. In profile mode it is a
    // cap on top of the plan, for when the car is off the line and steering
    // harder than planned.
    double target = steering_speed(steering, cfg_.max_steering, cfg_.max_speed,
                                   cfg_.medium_angle_speed, cfg_.min_speed);
    if (cfg_.profile_mode) {
        target = std::min(target, profile_[tracker_->closest()]);   // corners, braking zones
    }
    target = std::min(target, goal_ramp(remaining, cfg_.goal_tolerance, cfg_.max_brake));
    if (cfg_.profile_mode) {
        // speed up no faster than max_accel; slowing down is never delayed
        const rclcpp::Time t = now();
        const double dt = last_cmd_time_ ? (t - *last_cmd_time_).seconds() : 0.0;
        target = limit_acceleration(target, std::max(last_cmd_speed_, std::abs(speed)), cfg_.max_accel, dt);
        last_cmd_time_ = t;
    }
    last_cmd_speed_ = target;

    ackermann_msgs::msg::AckermannDriveStamped msg;
    msg.header.stamp = now();
    msg.header.frame_id = "base_link";
    msg.drive.steering_angle = steering;
    msg.drive.speed = target;
    drive_pub_->publish(msg);
}

void PurePursuit::odom_callback(nav_msgs::msg::Odometry::ConstSharedPtr msg)
{
    std::string reason;
    const bool engaged = deadman_engaged(reason);
    if (engaged && !was_engaged_ && finished_) {
        // LB pressed again after a finished run: a new run from here
        finished_ = false;
        laps_logged_ = 0;
        tracker_->restart_run();
        RCLCPP_INFO(get_logger(), "LB pressed again: starting a new run");
    }
    was_engaged_ = engaged;


    //check permission to mmove
    if (!engaged || safety_braking(reason)) {
        stop(reason);
        return;
    }
    //fallback controller
    if (obs_fallback_sub_->get_publisher_count() > 0 && obs_fallback_engaged_) {
        return;   // the obstacle fallback node drives
    }


    //Extract position and yaw, shift sensor pose to the rear axle
    const auto& q = msg->pose.pose.orientation;
    const Pose sensor{msg->pose.pose.position.x, msg->pose.pose.position.y,
                      std::atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))};
    if (!std::isfinite(sensor.x) || !std::isfinite(sensor.y) || !std::isfinite(sensor.yaw)) {
        stop("the pose is not finite");
        return;
    }


    // position used for tracking and steering
    const Pose car = rear_axle(sensor, cfg_.laser_offset_x, cfg_.laser_offset_y);

    // where on the path: nearest waypoint, distance driven, what is left
    const Tracker::Update where = tracker_->update(car.x, car.y);
    //tracker returns the nearest way point


    if (where.off_path > cfg_.max_path_error) { //stop threshold
        stop("the car is " + std::to_string(where.off_path).substr(0, 4) + " m from the path (limit " +
             std::to_string(cfg_.max_path_error).substr(0, 4) + " m): localization lost or wrong line?");
        return;
    }

    //check laps and remaining distance
    if (tracker_->laps_done() > laps_logged_) {
        laps_logged_ = tracker_->laps_done();
        RCLCPP_INFO(get_logger(), "Lap %d done", laps_logged_);
    }


    const double remaining = tracker_->remaining();
    if (!finished_ && remaining <= cfg_.goal_tolerance) {
        finished_ = true;
        RCLCPP_INFO(get_logger(), "Reached the goal (%.2f m left): stopping. Release and press LB to go again.",
                    std::max(0.0, remaining));
    }
    if (finished_) {
        stop("at the goal; release and press LB to go again");
        return;
    }

    const double speed = cfg_.use_pf ? current_speed_ : msg->twist.twist.linear.x;
    //choose steering target
    const double L = lookahead(speed, cfg_.lookahead_base, cfg_.lookahead_speed_gain,
                               cfg_.lookahead_min, cfg_.lookahead_max);
    const std::optional<Goal> goal = find_goal(*path_, where.closest, car, L);
    if (!goal) {
        // e.g. the car faces away from the path: stop rather than leave the
        // last command standing until ackermann_mux times it out
        stop("no suitable forward waypoint");
        return;
    }
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000, "Goal %zu%s: local(%.2f, %.2f)",
                         goal->index, goal->past_end ? " (past the end)" : "", goal->local_x, goal->local_y);
    drive_to_goal(*goal, speed, remaining); //produce steering and speed
}

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PurePursuit>());
    rclcpp::shutdown();
    return 0;
}
