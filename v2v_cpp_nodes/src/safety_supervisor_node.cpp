/**
 * safety_supervisor_node (C++)
 * ─────────────────────────────
 * 20 Hz 전체 예외 상황 감시 노드.
 *
 * 우선순위:
 *   P1. LiDAR 근접 장애물
 *   P2. 인식 타임아웃
 *   P3. 경로 오류 (REVERSE_EXECUTE 중 path 미수신)
 *   P4. 제어 이상 (cmd_vel 이상값)
 */

#include <cmath>
#include <string>
#include <vector>
#include <algorithm>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/bool.hpp"
#include "std_msgs/msg/float32.hpp"
#include "std_msgs/msg/string.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"

class SafetySupervisorNode : public rclcpp::Node {
public:
    SafetySupervisorNode() : Node("safety_supervisor_node") {
        // ── 파라미터 ─────────────────────────────────────────────
        this->declare_parameter("supervisor_hz",         20.0);
        this->declare_parameter("lidar_min_dist_m",      0.15);
        this->declare_parameter("lidar_front_angle_deg", 60.0);
        this->declare_parameter("lidar_rear_angle_deg",  30.0);
        this->declare_parameter("lane_timeout_sec",      1.0);
        this->declare_parameter("path_error_hold_sec",   2.0);
        this->declare_parameter("max_safe_linear",       0.20);
        this->declare_parameter("max_safe_angular",      2.0);
        this->declare_parameter("startup_grace_sec",     3.0);

        supervisor_hz_       = this->get_parameter("supervisor_hz").as_double();
        lidar_min_dist_m_    = this->get_parameter("lidar_min_dist_m").as_double();
        lidar_front_angle_   = this->get_parameter("lidar_front_angle_deg").as_double() * M_PI / 180.0;
        lidar_rear_angle_    = this->get_parameter("lidar_rear_angle_deg").as_double() * M_PI / 180.0;
        lane_timeout_sec_    = this->get_parameter("lane_timeout_sec").as_double();
        path_error_hold_sec_ = this->get_parameter("path_error_hold_sec").as_double();
        max_safe_linear_     = this->get_parameter("max_safe_linear").as_double();
        max_safe_angular_    = this->get_parameter("max_safe_angular").as_double();
        startup_grace_sec_   = this->get_parameter("startup_grace_sec").as_double();
        startup_time_        = now_sec();

        // ── 구독 ────────────────────────────────────────────────
        sub_scan_  = create_subscription<sensor_msgs::msg::LaserScan>(
            "/scan", 10,
            [this](sensor_msgs::msg::LaserScan::SharedPtr m) {
                scan_ = m; last_scan_stamp_ = now_sec();
            });

        sub_state_ = create_subscription<std_msgs::msg::String>(
            "/vehicle_state", 10,
            [this](std_msgs::msg::String::SharedPtr m) { vehicle_state_ = m->data; });

        sub_path_rdy_ = create_subscription<std_msgs::msg::Bool>(
            "/reverse_path_ready", 10,
            [this](std_msgs::msg::Bool::SharedPtr m) { reverse_path_ready_ = m->data; });

        sub_cmd_ = create_subscription<geometry_msgs::msg::Twist>(
            "/cmd_vel", 10,
            [this](geometry_msgs::msg::Twist::SharedPtr m) { cmd_vel_ = *m; });

        sub_lane_ = create_subscription<std_msgs::msg::String>(
            "/lane_status_active", 10,
            [this](std_msgs::msg::String::SharedPtr m) {
                lane_status_ = m->data; last_lane_stamp_ = now_sec();
            });

        // ── 발행 ────────────────────────────────────────────────
        pub_stop_   = create_publisher<std_msgs::msg::Bool>("/safe_stop", 10);
        pub_status_ = create_publisher<std_msgs::msg::String>("/supervisor_status", 10);

        double dt = (supervisor_hz_ > 0) ? (1.0 / supervisor_hz_) : 0.05;
        timer_ = create_wall_timer(
            std::chrono::duration<double>(dt),
            std::bind(&SafetySupervisorNode::step, this));

        RCLCPP_INFO(get_logger(), "safety_supervisor_node (C++) 시작");
    }

private:
    double now_sec() const {
        return get_clock()->now().nanoseconds() * 1e-9;
    }
    bool in_startup_grace() const {
        return (now_sec() - startup_time_) < startup_grace_sec_;
    }
    bool is_fresh(double stamp, double timeout) const {
        return (stamp > 0) && (now_sec() - stamp <= timeout);
    }

    // ── P1: LiDAR 근접 ────────────────────────────────────────────
    std::string check_lidar() {
        if (!scan_ || !is_fresh(last_scan_stamp_, 0.5))
            return "";

        bool reversing = (vehicle_state_ == "REVERSE_EXECUTE");
        double center_angle = reversing ? M_PI : 0.0;
        double watch = reversing ? lidar_rear_angle_ : lidar_front_angle_;

        double min_r = 1e9;
        for (size_t i = 0; i < scan_->ranges.size(); ++i) {
            float r = scan_->ranges[i];
            if (!std::isfinite(r) || r <= 0.0f) continue;
            double angle = scan_->angle_min
                         + static_cast<double>(i) * scan_->angle_increment;
            double diff = std::atan2(std::sin(angle - center_angle),
                                     std::cos(angle - center_angle));
            if (std::abs(diff) <= watch)
                min_r = std::min(min_r, static_cast<double>(r));
        }
        if (min_r < lidar_min_dist_m_)
            return "P1_lidar_close=" + std::to_string(min_r);
        return "";
    }

    // ── P2: 인식 타임아웃 ───────────────────────────────────────────
    std::string check_perception() {
        if (vehicle_state_ == "SAFE_STOP" || vehicle_state_ == "WAIT_PASS")
            return "";
        if (in_startup_grace()) return "";
        if (!is_fresh(last_lane_stamp_, lane_timeout_sec_))
            return "P2_lane_timeout";
        return "";
    }

    // ── P3: 경로 오류 ──────────────────────────────────────────────
    std::string check_path_error() {
        if (vehicle_state_ != "REVERSE_EXECUTE") {
            path_error_start_ = 0.0;
            return "";
        }
        if (reverse_path_ready_) {
            path_error_start_ = 0.0;
            return "";
        }
        if (path_error_start_ <= 0.0)
            path_error_start_ = now_sec();
        if (now_sec() - path_error_start_ >= path_error_hold_sec_)
            return "P3_no_reverse_path";
        return "";
    }

    // ── P4: 제어 이상 ──────────────────────────────────────────────
    std::string check_control() {
        if (std::abs(cmd_vel_.linear.x) > max_safe_linear_ ||
            std::abs(cmd_vel_.angular.z) > max_safe_angular_)
            return "P4_control_anomaly";
        return "";
    }

    // ── 메인 루프 ───────────────────────────────────────────────────
    void step() {
        std::vector<std::string> reasons;
        auto push = [&](const std::string &s) {
            if (!s.empty()) reasons.push_back(s);
        };
        push(check_lidar());
        push(check_perception());
        push(check_path_error());
        push(check_control());

        bool stop = !reasons.empty();

        if (stop && !safe_stop_) {
            std::string all;
            for (auto &r : reasons) all += r + " | ";
            RCLCPP_WARN(get_logger(), "[Safety] SAFE_STOP: %s", all.c_str());
        }
        if (!stop && safe_stop_) {
            RCLCPP_INFO(get_logger(), "[Safety] 해제");
        }
        safe_stop_ = stop;

        std_msgs::msg::Bool b;
        b.data = safe_stop_;
        pub_stop_->publish(b);

        std_msgs::msg::String s;
        if (reasons.empty()) {
            s.data = "ok";
        } else {
            s.data = reasons.front();
        }
        pub_status_->publish(s);
    }

    // ── 파라미터 ─────────────────────────────────────────────────
    double supervisor_hz_, lidar_min_dist_m_;
    double lidar_front_angle_, lidar_rear_angle_;
    double lane_timeout_sec_, path_error_hold_sec_;
    double max_safe_linear_, max_safe_angular_;
    double startup_grace_sec_, startup_time_;

    // ── 상태 ─────────────────────────────────────────────────────
    sensor_msgs::msg::LaserScan::SharedPtr scan_;
    double last_scan_stamp_{0.0}, last_lane_stamp_{0.0};
    std::string vehicle_state_{"NORMAL_CENTER_DRIVE"};
    std::string lane_status_{"unknown"};
    bool reverse_path_ready_{false};
    geometry_msgs::msg::Twist cmd_vel_;
    bool safe_stop_{false};
    double path_error_start_{0.0};

    // ── ROS 인터페이스 ────────────────────────────────────────────
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr sub_scan_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr       sub_state_, sub_lane_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr         sub_path_rdy_;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr   sub_cmd_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr            pub_stop_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr          pub_status_;
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<SafetySupervisorNode>());
    rclcpp::shutdown();
    return 0;
}
