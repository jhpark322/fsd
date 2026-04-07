#!/usr/bin/env python3
import math
from typing import Optional

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class LaneFollowControlNode(Node):
    def __init__(self) -> None:
        super().__init__('lane_follow_control_node')

        self.declare_parameter('center_error_m_topic', '/lane_error_center_m_active')
        self.declare_parameter('right_error_m_topic', '/lane_error_right_m_active')
        self.declare_parameter('heading_error_topic', '/lane_heading_error_active')
        self.declare_parameter('lane_status_topic', '/lane_status_active')
        self.declare_parameter('guidance_source_topic', '/lane_guidance_source')
        self.declare_parameter('control_mode_topic', '/control_mode')
        self.declare_parameter('safe_stop_topic', '/safe_stop')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')

        self.declare_parameter('control_hz', 20.0)

        self.declare_parameter('kp_m', 0.22)
        self.declare_parameter('k_heading', 0.03)
        self.declare_parameter('steering_sign', -1.0)
        self.declare_parameter('max_ang_z', 0.08)

        self.declare_parameter('nominal_speed', 0.028)
        self.declare_parameter('min_speed', 0.018)
        self.declare_parameter('slow_error_m', 0.10)
        self.declare_parameter('hard_stop_error_m', 0.25)
        self.declare_parameter('rotate_in_place_on_large_error', False)

        self.declare_parameter('lane_timeout_sec', 0.35)
        self.declare_parameter('heading_timeout_sec', 0.35)
        self.declare_parameter('stop_on_invalid_lane', True)
        self.declare_parameter('require_heading', False)
        self.declare_parameter('hold_last_heading_sec', 0.20)
        self.declare_parameter('default_follow_mode', 'center')

        self.declare_parameter('camera_forward_offset_m', 0.0)
        self.declare_parameter('lookahead_m', 0.20)
        self.declare_parameter('lookahead_gain', 0.0)
        self.declare_parameter('use_lookahead', False)

        # live/blend/memory source-specific behavior
        self.declare_parameter('live_center_deadband_m', 0.003)
        self.declare_parameter('live_heading_deadband_rad', 0.010)
        self.declare_parameter('blend_center_deadband_m', 0.006)
        self.declare_parameter('blend_heading_deadband_rad', 0.020)
        self.declare_parameter('memory_center_deadband_m', 0.012)
        self.declare_parameter('memory_heading_deadband_rad', 0.040)
        self.declare_parameter('blend_kp_scale', 0.80)
        self.declare_parameter('blend_k_heading_scale', 0.60)
        self.declare_parameter('blend_max_ang_z', 0.06)
        self.declare_parameter('blend_speed_limit', 0.023)
        self.declare_parameter('memory_kp_scale', 0.35)
        self.declare_parameter('memory_k_heading_scale', 0.20)
        self.declare_parameter('memory_max_ang_z', 0.04)
        self.declare_parameter('memory_speed_limit', 0.018)

        self.center_error_m_topic = str(self.get_parameter('center_error_m_topic').value)
        self.right_error_m_topic = str(self.get_parameter('right_error_m_topic').value)
        self.heading_error_topic = str(self.get_parameter('heading_error_topic').value)
        self.lane_status_topic = str(self.get_parameter('lane_status_topic').value)
        self.guidance_source_topic = str(self.get_parameter('guidance_source_topic').value)
        self.control_mode_topic = str(self.get_parameter('control_mode_topic').value)
        self.safe_stop_topic = str(self.get_parameter('safe_stop_topic').value)
        self.cmd_vel_topic = str(self.get_parameter('cmd_vel_topic').value)

        self.control_hz = float(self.get_parameter('control_hz').value)
        self.kp_m = float(self.get_parameter('kp_m').value)
        self.k_heading = float(self.get_parameter('k_heading').value)
        self.steering_sign = float(self.get_parameter('steering_sign').value)
        self.max_ang_z = float(self.get_parameter('max_ang_z').value)

        self.nominal_speed = float(self.get_parameter('nominal_speed').value)
        self.min_speed = float(self.get_parameter('min_speed').value)
        self.slow_error_m = float(self.get_parameter('slow_error_m').value)
        self.hard_stop_error_m = float(self.get_parameter('hard_stop_error_m').value)
        self.rotate_in_place_on_large_error = bool(self.get_parameter('rotate_in_place_on_large_error').value)

        self.lane_timeout_sec = float(self.get_parameter('lane_timeout_sec').value)
        self.heading_timeout_sec = float(self.get_parameter('heading_timeout_sec').value)
        self.stop_on_invalid_lane = bool(self.get_parameter('stop_on_invalid_lane').value)
        self.require_heading = bool(self.get_parameter('require_heading').value)
        self.hold_last_heading_sec = float(self.get_parameter('hold_last_heading_sec').value)
        self.default_follow_mode = str(self.get_parameter('default_follow_mode').value).strip().lower()

        self.camera_forward_offset_m = float(self.get_parameter('camera_forward_offset_m').value)
        self.lookahead_m = float(self.get_parameter('lookahead_m').value)
        self.lookahead_gain = float(self.get_parameter('lookahead_gain').value)
        self.use_lookahead = bool(self.get_parameter('use_lookahead').value)

        self.live_center_deadband_m = float(self.get_parameter('live_center_deadband_m').value)
        self.live_heading_deadband_rad = float(self.get_parameter('live_heading_deadband_rad').value)
        self.blend_center_deadband_m = float(self.get_parameter('blend_center_deadband_m').value)
        self.blend_heading_deadband_rad = float(self.get_parameter('blend_heading_deadband_rad').value)
        self.memory_center_deadband_m = float(self.get_parameter('memory_center_deadband_m').value)
        self.memory_heading_deadband_rad = float(self.get_parameter('memory_heading_deadband_rad').value)
        self.blend_kp_scale = float(self.get_parameter('blend_kp_scale').value)
        self.blend_k_heading_scale = float(self.get_parameter('blend_k_heading_scale').value)
        self.blend_max_ang_z = float(self.get_parameter('blend_max_ang_z').value)
        self.blend_speed_limit = float(self.get_parameter('blend_speed_limit').value)
        self.memory_kp_scale = float(self.get_parameter('memory_kp_scale').value)
        self.memory_k_heading_scale = float(self.get_parameter('memory_k_heading_scale').value)
        self.memory_max_ang_z = float(self.get_parameter('memory_max_ang_z').value)
        self.memory_speed_limit = float(self.get_parameter('memory_speed_limit').value)

        self.center_err_m: Optional[float] = None
        self.right_err_m: Optional[float] = None
        self.heading_err: Optional[float] = None

        self.last_good_heading_err: float = 0.0
        self.last_good_heading_stamp: Optional[float] = None

        self.lane_status: str = 'unknown'
        self.guidance_source: str = 'none'
        self.control_mode: str = 'NORMAL_CENTER_DRIVE'
        self.safe_stop: bool = False

        self.last_lane_stamp: Optional[float] = None
        self.last_heading_stamp: Optional[float] = None

        self.create_subscription(Float32, self.center_error_m_topic, self.center_cb, 10)
        self.create_subscription(Float32, self.right_error_m_topic, self.right_cb, 10)
        self.create_subscription(Float32, self.heading_error_topic, self.heading_cb, 10)
        self.create_subscription(String, self.lane_status_topic, self.status_cb, 10)
        self.create_subscription(String, self.guidance_source_topic, self.guidance_source_cb, 10)
        self.create_subscription(String, self.control_mode_topic, self.control_mode_cb, 10)
        self.create_subscription(Bool, self.safe_stop_topic, self.safe_stop_cb, 10)

        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)

        dt = 1.0 / self.control_hz if self.control_hz > 0 else 0.05
        self.timer = self.create_timer(dt, self.control_step)

    def now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def center_cb(self, msg: Float32) -> None:
        self.center_err_m = float(msg.data)
        self.last_lane_stamp = self.now_sec()

    def right_cb(self, msg: Float32) -> None:
        self.right_err_m = float(msg.data)
        self.last_lane_stamp = self.now_sec()

    def heading_cb(self, msg: Float32) -> None:
        v = float(msg.data)
        if math.isfinite(v):
            self.heading_err = v
            self.last_heading_stamp = self.now_sec()
            self.last_good_heading_err = v
            self.last_good_heading_stamp = self.last_heading_stamp
        else:
            self.heading_err = None
            self.last_heading_stamp = None

    def status_cb(self, msg: String) -> None:
        self.lane_status = str(msg.data).strip().lower()
        self.last_lane_stamp = self.now_sec()

    def guidance_source_cb(self, msg: String) -> None:
        self.guidance_source = str(msg.data).strip().lower()

    def control_mode_cb(self, msg: String) -> None:
        self.control_mode = str(msg.data).strip()

    def safe_stop_cb(self, msg: Bool) -> None:
        self.safe_stop = bool(msg.data)

    def publish_stop(self) -> None:
        self.cmd_pub.publish(Twist())

    def lane_fresh(self) -> bool:
        if self.last_lane_stamp is None:
            return False
        return (self.now_sec() - self.last_lane_stamp) <= self.lane_timeout_sec

    def heading_fresh(self) -> bool:
        if self.last_heading_stamp is None:
            return False
        if self.heading_err is None:
            return False
        return (self.now_sec() - self.last_heading_stamp) <= self.heading_timeout_sec

    def lane_valid_for_follow(self) -> bool:
        return self.lane_status == 'ok'

    def select_error_m(self) -> Optional[float]:
        mode = self.control_mode.strip().upper()
        if mode in ('STOP', 'SAFE_STOP', 'WAIT_PASS', 'PASS_BLOCKED'):
            return None
        if mode == 'KEEP_RIGHT_APPROACH':
            return self.right_err_m
        if mode == 'NORMAL_CENTER_DRIVE':
            return self.center_err_m
        if self.default_follow_mode == 'right':
            return self.right_err_m
        return self.center_err_m

    def compute_speed(self, abs_error_m: float) -> float:
        if abs_error_m >= self.hard_stop_error_m:
            return 0.0 if self.rotate_in_place_on_large_error else self.min_speed
        alpha = clamp(abs_error_m / max(self.slow_error_m, 1e-6), 0.0, 1.0)
        speed = self.nominal_speed - (self.nominal_speed - self.min_speed) * alpha
        return clamp(speed, self.min_speed, self.nominal_speed)

    def compensate_error(self, e_cam: float, psi: float) -> tuple[float, float]:
        e_base = e_cam - self.camera_forward_offset_m * psi
        if self.use_lookahead:
            e_ctrl = e_base + self.lookahead_gain * self.lookahead_m * psi
        else:
            e_ctrl = e_base
        return e_base, e_ctrl

    def resolve_heading(self) -> Optional[float]:
        if self.heading_fresh() and self.heading_err is not None:
            return float(self.heading_err)
        can_hold = (
            self.last_good_heading_stamp is not None and
            (self.now_sec() - self.last_good_heading_stamp) <= self.hold_last_heading_sec
        )
        if can_hold:
            return float(self.last_good_heading_err)
        if self.require_heading:
            return None
        return 0.0

    def source_params(self):
        src = self.guidance_source
        if src == 'memory':
            return (
                self.memory_center_deadband_m,
                self.memory_heading_deadband_rad,
                self.kp_m * self.memory_kp_scale,
                self.k_heading * self.memory_k_heading_scale,
                min(self.max_ang_z, self.memory_max_ang_z),
                self.memory_speed_limit,
            )
        if src == 'blend':
            return (
                self.blend_center_deadband_m,
                self.blend_heading_deadband_rad,
                self.kp_m * self.blend_kp_scale,
                self.k_heading * self.blend_k_heading_scale,
                min(self.max_ang_z, self.blend_max_ang_z),
                self.blend_speed_limit,
            )
        return (
            self.live_center_deadband_m,
            self.live_heading_deadband_rad,
            self.kp_m,
            self.k_heading,
            self.max_ang_z,
            self.nominal_speed,
        )

    def control_step(self) -> None:
        if self.safe_stop:
            self.publish_stop()
            return

        if self.control_mode.strip().upper() == 'PASS_BLOCKED':
            self.publish_stop()
            return

        e_cam = self.select_error_m()
        if e_cam is None:
            self.publish_stop()
            return

        if not self.lane_fresh():
            self.publish_stop()
            return

        if self.stop_on_invalid_lane and not self.lane_valid_for_follow():
            self.publish_stop()
            return

        psi = self.resolve_heading()
        if psi is None:
            self.publish_stop()
            return

        e_base, e_ctrl = self.compensate_error(e_cam, psi)
        dead_e, dead_h, kp_use, k_heading_use, max_ang_use, speed_limit = self.source_params()

        if abs(e_ctrl) < dead_e:
            e_ctrl = 0.0
        if abs(psi) < dead_h:
            psi = 0.0

        abs_err = abs(e_ctrl)
        speed = min(self.compute_speed(abs_err), speed_limit)

        ang = self.steering_sign * (kp_use * e_ctrl + k_heading_use * psi)
        ang = clamp(ang, -max_ang_use, max_ang_use)

        tw = Twist()
        if self.rotate_in_place_on_large_error and abs_err >= self.hard_stop_error_m:
            tw.linear.x = 0.0
            tw.angular.z = ang
        else:
            tw.linear.x = speed
            tw.angular.z = ang

        self.cmd_pub.publish(tw)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LaneFollowControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
