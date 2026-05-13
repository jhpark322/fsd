/**
 * chassis_control_node (C++)
 * ──────────────────────────
 * 20 Hz 통합 섀시 제어 노드.
 *
 * /control_mode 에 따라 세 가지 제어기를 전환:
 *   NORMAL_CENTER_DRIVE  → PID 중앙 주행
 *   KEEP_RIGHT_APPROACH  → PID 우측 오프셋 추종
 *   REENTER              → PID 중앙 (감속)
 *   REVERSE_EXECUTE      → Reverse Pure Pursuit
 *   WAIT_PASS / SAFE_STOP→ 즉시 정지
 *
 * 후진 목표 도달 시 /reverse_motion_done = True 를 발행해
 * v2v_decision_node가 WAIT_PASS로 전이할 수 있게 한다.
 */

#include <cmath>
#include <algorithm>
#include <string>
#include <optional>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/bool.hpp"
#include "std_msgs/msg/float32.hpp"
#include "std_msgs/msg/string.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "nav_msgs/msg/path.hpp"

static inline double clamp_val(double v, double lo, double hi) {
    return std::max(lo, std::min(hi, v));
}

static double quat_to_yaw(const geometry_msgs::msg::Quaternion &q) {
    double siny = 2.0 * (q.w * q.z + q.x * q.y);
    double cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
    return std::atan2(siny, cosy);
}

class ChassisControlNode : public rclcpp::Node {
public:
    ChassisControlNode() : Node("chassis_control_node") {
        // ── 파라미터 선언 ────────────────────────────────────────────
        this->declare_parameter("control_hz",          20.0);
        this->declare_parameter("kp_m",                0.22);
        this->declare_parameter("k_heading",           0.03);
        this->declare_parameter("steering_sign",      -1.0);
        this->declare_parameter("max_ang_z",           0.08);
        this->declare_parameter("nominal_speed",       0.028);
        this->declare_parameter("min_speed",           0.018);
        this->declare_parameter("slow_error_m",        0.10);
        this->declare_parameter("hard_stop_error_m",   0.25);
        this->declare_parameter("reverse_lookahead_m", 0.30);
        this->declare_parameter("reverse_speed",       0.018);
        this->declare_parameter("reverse_max_ang_z",   0.06);
        this->declare_parameter("reverse_goal_tol_m",  0.12);
        this->declare_parameter("reenter_speed_scale", 0.7);
        this->declare_parameter("lane_timeout_sec",    0.35);

        control_hz_          = this->get_parameter("control_hz").as_double();
        kp_m_                = this->get_parameter("kp_m").as_double();
        k_heading_           = this->get_parameter("k_heading").as_double();
        steering_sign_       = this->get_parameter("steering_sign").as_double();
        max_ang_z_           = this->get_parameter("max_ang_z").as_double();
        nominal_speed_       = this->get_parameter("nominal_speed").as_double();
        min_speed_           = this->get_parameter("min_speed").as_double();
        slow_error_m_        = this->get_parameter("slow_error_m").as_double();
        hard_stop_error_m_   = this->get_parameter("hard_stop_error_m").as_double();
        reverse_lookahead_m_ = this->get_parameter("reverse_lookahead_m").as_double();
        reverse_speed_       = this->get_parameter("reverse_speed").as_double();
        reverse_max_ang_z_   = this->get_parameter("reverse_max_ang_z").as_double();
        reverse_goal_tol_m_  = this->get_parameter("reverse_goal_tol_m").as_double();
        reenter_speed_scale_ = this->get_parameter("reenter_speed_scale").as_double();
        lane_timeout_sec_    = this->get_parameter("lane_timeout_sec").as_double();

        // 안전 가드
        if (reverse_lookahead_m_ < 1e-4) reverse_lookahead_m_ = 0.30;

        // ── 구독 ────────────────────────────────────────────────────
        sub_mode_    = create_subscription<std_msgs::msg::String>(
            "/control_mode", 10,
            [this](std_msgs::msg::String::SharedPtr m) { control_mode_ = m->data; });

        sub_stop_    = create_subscription<std_msgs::msg::Bool>(
            "/safe_stop", 10,
            [this](std_msgs::msg::Bool::SharedPtr m) { safe_stop_ = m->data; });

        sub_center_  = create_subscription<std_msgs::msg::Float32>(
            "/lane_error_center_m_active", 10,
            [this](std_msgs::msg::Float32::SharedPtr m) {
                center_err_ = m->data; last_lane_stamp_ = now_sec();
            });

        sub_right_   = create_subscription<std_msgs::msg::Float32>(
            "/lane_error_right_m_active", 10,
            [this](std_msgs::msg::Float32::SharedPtr m) {
                right_err_ = m->data; last_lane_stamp_ = now_sec();
            });

        sub_heading_ = create_subscription<std_msgs::msg::Float32>(
            "/lane_heading_error_active", 10,
            [this](std_msgs::msg::Float32::SharedPtr m) {
                double v = m->data;
                heading_err_ = std::isfinite(v) ? std::optional<double>(v) : std::nullopt;
            });

        sub_lane_    = create_subscription<std_msgs::msg::String>(
            "/lane_status_active", 10,
            [this](std_msgs::msg::String::SharedPtr m) {
                lane_status_ = m->data; last_lane_stamp_ = now_sec();
            });

        sub_odom_    = create_subscription<nav_msgs::msg::Odometry>(
            "/odom", 10,
            [this](nav_msgs::msg::Odometry::SharedPtr m) { odom_ = m; });

        sub_path_    = create_subscription<nav_msgs::msg::Path>(
            "/reverse_path", 10,
            [this](nav_msgs::msg::Path::SharedPtr m) {
                reverse_path_ = m;
                reverse_done_ = false;
                publish_reverse_done(false);
            });

        // ── 발행 ────────────────────────────────────────────────────
        pub_cmd_ = create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);
        pub_reverse_done_ = create_publisher<std_msgs::msg::Bool>("/reverse_motion_done", 10);

        double dt = (control_hz_ > 0) ? (1.0 / control_hz_) : 0.05;
        timer_ = create_wall_timer(
            std::chrono::duration<double>(dt),
            std::bind(&ChassisControlNode::step, this));

        RCLCPP_INFO(get_logger(), "chassis_control_node (C++) 시작");
    }

private:
    // ── 유틸 ────────────────────────────────────────────────────────
    double now_sec() const {
        return get_clock()->now().nanoseconds() * 1e-9;
    }
    bool lane_fresh() const {
        return (last_lane_stamp_ > 0) && (now_sec() - last_lane_stamp_ <= lane_timeout_sec_);
    }
    void publish_stop() {
        pub_cmd_->publish(geometry_msgs::msg::Twist());
    }
    void publish(double lx, double az) {
        geometry_msgs::msg::Twist tw;
        tw.linear.x = lx;
        tw.angular.z = az;
        pub_cmd_->publish(tw);
    }
    void publish_reverse_done(bool done) {
        std_msgs::msg::Bool msg;
        msg.data = done;
        pub_reverse_done_->publish(msg);
    }

    // ── PID 차선 추종 ───────────────────────────────────────────────
    double compute_speed(double abs_err) const {
        if (abs_err >= hard_stop_error_m_) return min_speed_;
        double alpha = clamp_val(abs_err / std::max(slow_error_m_, 1e-6), 0.0, 1.0);
        return nominal_speed_ - (nominal_speed_ - min_speed_) * alpha;
    }

    void pid_follow(std::optional<double> err_m, double speed_scale = 1.0) {
        if (!err_m.has_value() || !lane_fresh()) {
            publish_stop(); return;
        }
        double psi = heading_err_.value_or(0.0);
        double ang = steering_sign_ * (kp_m_ * err_m.value() + k_heading_ * psi);
        ang = clamp_val(ang, -max_ang_z_, max_ang_z_);
        double speed = compute_speed(std::abs(err_m.value())) * speed_scale;
        publish(speed, ang);
    }

    // ── Reverse Pure Pursuit ────────────────────────────────────────
    void reverse_pure_pursuit() {
        if (!reverse_path_ || reverse_path_->poses.empty() || !odom_) {
            publish_stop(); return;
        }

        double cx = odom_->pose.pose.position.x;
        double cy = odom_->pose.pose.position.y;
        double yaw = quat_to_yaw(odom_->pose.pose.orientation);

        // 목표 도달 확인
        auto &last = reverse_path_->poses.back();
        double gx = last.pose.position.x;
        double gy = last.pose.position.y;
        if (std::hypot(gx - cx, gy - cy) <= reverse_goal_tol_m_) {
            RCLCPP_INFO(get_logger(), "[Chassis] 후진 목표 도달");
            reverse_done_ = true;
            publish_reverse_done(true);
            publish_stop();
            return;
        }

        // Lookahead 점 선택
        double best_dist = 0.0;
        size_t best_idx = 0;
        for (size_t i = 0; i < reverse_path_->poses.size(); ++i) {
            double wx = reverse_path_->poses[i].pose.position.x;
            double wy = reverse_path_->poses[i].pose.position.y;
            double d = std::hypot(wx - cx, wy - cy);
            if (d <= reverse_lookahead_m_ && d >= best_dist) {
                best_dist = d;
                best_idx = i;
            }
        }
        double tx = reverse_path_->poses[best_idx].pose.position.x;
        double ty = reverse_path_->poses[best_idx].pose.position.y;

        // 후진 Pure Pursuit 조향
        double rear_yaw = yaw + M_PI;
        double dx = tx - cx;
        double dy = ty - cy;
        double angle_to_target = std::atan2(dy, dx);
        double alpha = std::atan2(
            std::sin(angle_to_target - rear_yaw),
            std::cos(angle_to_target - rear_yaw));

        double curvature = 2.0 * std::sin(alpha) / reverse_lookahead_m_;
        double ang = clamp_val(-curvature * reverse_speed_,
                               -reverse_max_ang_z_, reverse_max_ang_z_);
        publish(-reverse_speed_, ang);
    }

    // ── 메인 루프 ───────────────────────────────────────────────────
    void step() {
        if (safe_stop_) { publish_stop(); return; }

        auto mode = control_mode_;
        // 대문자 변환
        std::transform(mode.begin(), mode.end(), mode.begin(), ::toupper);

        if (mode == "SAFE_STOP" || mode == "WAIT_PASS" || mode == "PASS_BLOCKED") {
            publish_stop(); return;
        }
        if (mode == "NORMAL_CENTER_DRIVE") {
            pid_follow(center_err_, 1.0); return;
        }
        if (mode == "KEEP_RIGHT_APPROACH") {
            pid_follow(right_err_, 0.8); return;
        }
        if (mode == "REENTER") {
            pid_follow(center_err_, reenter_speed_scale_); return;
        }
        if (mode == "REVERSE_EXECUTE") {
            if (reverse_done_) {
                publish_reverse_done(true);
                publish_stop();
            } else {
                publish_reverse_done(false);
                reverse_pure_pursuit();
            }
            return;
        }
        publish_stop();
    }

    // ── 파라미터 ─────────────────────────────────────────────────
    double control_hz_, kp_m_, k_heading_, steering_sign_, max_ang_z_;
    double nominal_speed_, min_speed_, slow_error_m_, hard_stop_error_m_;
    double reverse_lookahead_m_, reverse_speed_, reverse_max_ang_z_, reverse_goal_tol_m_;
    double reenter_speed_scale_, lane_timeout_sec_;

    // ── 상태 ─────────────────────────────────────────────────────
    std::string control_mode_{"NORMAL_CENTER_DRIVE"};
    bool safe_stop_{false};
    std::string lane_status_{"unknown"};
    std::optional<double> center_err_, right_err_, heading_err_;
    double last_lane_stamp_{0.0};
    nav_msgs::msg::Odometry::SharedPtr odom_;
    nav_msgs::msg::Path::SharedPtr reverse_path_;
    bool reverse_done_{false};

    // ── ROS 인터페이스 ────────────────────────────────────────────
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr  sub_mode_, sub_lane_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr    sub_stop_;
    rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr sub_center_, sub_right_, sub_heading_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odom_;
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr     sub_path_;
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr   pub_cmd_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr          pub_reverse_done_;
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ChassisControlNode>());
    rclcpp::shutdown();
    return 0;
}
