#!/usr/bin/env python3
from typing import Optional

import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, String, Bool


class LaneDecisionNode(Node):
    def __init__(self) -> None:
        super().__init__('lane_decision_node')

        self.declare_parameter('lane_width_m_topic', '/lane_width_m')
        self.declare_parameter('mem_width_m_topic', '/lane_mem_width_m')
        self.declare_parameter('lane_status_topic', '/lane_status_active')
        self.declare_parameter('guidance_source_topic', '/lane_guidance_source')

        self.declare_parameter('control_mode_topic', '/control_mode')
        self.declare_parameter('safe_stop_topic', '/safe_stop')
        self.declare_parameter('decision_status_topic', '/decision_status')

        self.declare_parameter('robot_width_m', 0.19)
        self.declare_parameter('width_margin_m', 0.00)
        self.declare_parameter('width_hysteresis_m', 0.01)
        self.declare_parameter('lane_timeout_sec', 0.5)
        self.declare_parameter('width_timeout_sec', 0.4)
        self.declare_parameter('hold_last_width_sec', 1.0)
        self.declare_parameter('decision_hz', 10.0)
        self.declare_parameter('stop_on_lane_lost', True)
        self.declare_parameter('min_valid_width_m', 0.05)

        self.lane_width_m_topic = str(self.get_parameter('lane_width_m_topic').value)
        self.mem_width_m_topic = str(self.get_parameter('mem_width_m_topic').value)
        self.lane_status_topic = str(self.get_parameter('lane_status_topic').value)
        self.guidance_source_topic = str(self.get_parameter('guidance_source_topic').value)
        self.control_mode_topic = str(self.get_parameter('control_mode_topic').value)
        self.safe_stop_topic = str(self.get_parameter('safe_stop_topic').value)
        self.decision_status_topic = str(self.get_parameter('decision_status_topic').value)

        self.robot_width_m = float(self.get_parameter('robot_width_m').value)
        self.width_margin_m = float(self.get_parameter('width_margin_m').value)
        self.width_hysteresis_m = float(self.get_parameter('width_hysteresis_m').value)
        self.lane_timeout_sec = float(self.get_parameter('lane_timeout_sec').value)
        self.width_timeout_sec = float(self.get_parameter('width_timeout_sec').value)
        self.hold_last_width_sec = float(self.get_parameter('hold_last_width_sec').value)
        self.decision_hz = float(self.get_parameter('decision_hz').value)
        self.stop_on_lane_lost = bool(self.get_parameter('stop_on_lane_lost').value)
        self.min_valid_width_m = float(self.get_parameter('min_valid_width_m').value)

        self.min_passable_width_m = self.robot_width_m + self.width_margin_m

        self.lane_width_m: Optional[float] = None
        self.mem_width_m: Optional[float] = None
        self.last_mem_width_stamp: Optional[float] = None
        self.lane_status: str = 'unknown'
        self.guidance_source: str = 'none'
        self.last_lane_status_stamp: Optional[float] = None
        self.last_width_stamp: Optional[float] = None

        self.last_good_width_m: Optional[float] = None
        self.last_good_width_stamp: Optional[float] = None

        self.current_mode: str = 'PASS_BLOCKED'
        self.current_safe_stop: bool = True

        self.create_subscription(Float32, self.lane_width_m_topic, self.lane_width_cb, 10)
        self.create_subscription(Float32, self.mem_width_m_topic, self.mem_width_cb, 10)
        self.create_subscription(String, self.lane_status_topic, self.lane_status_cb, 10)
        self.create_subscription(String, self.guidance_source_topic, self.guidance_source_cb, 10)

        self.control_mode_pub = self.create_publisher(String, self.control_mode_topic, 10)
        self.safe_stop_pub = self.create_publisher(Bool, self.safe_stop_topic, 10)
        self.decision_status_pub = self.create_publisher(String, self.decision_status_topic, 10)

        dt = 1.0 / self.decision_hz if self.decision_hz > 0.0 else 0.1
        self.timer = self.create_timer(dt, self.decision_step)

    def now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def lane_width_cb(self, msg: Float32) -> None:
        width = float(msg.data)
        self.lane_width_m = width
        self.last_width_stamp = self.now_sec()
        if math.isfinite(width) and width >= self.min_valid_width_m:
            self.last_good_width_m = width
            self.last_good_width_stamp = self.last_width_stamp

    def mem_width_cb(self, msg: Float32) -> None:
        width = float(msg.data)
        if math.isfinite(width) and width >= self.min_valid_width_m:
            self.mem_width_m = width
            self.last_mem_width_stamp = self.now_sec()

    def lane_status_cb(self, msg: String) -> None:
        self.lane_status = str(msg.data).strip().lower()
        self.last_lane_status_stamp = self.now_sec()

    def guidance_source_cb(self, msg: String) -> None:
        self.guidance_source = str(msg.data).strip().lower()

    def lane_status_fresh(self) -> bool:
        if self.last_lane_status_stamp is None:
            return False
        return (self.now_sec() - self.last_lane_status_stamp) <= self.lane_timeout_sec

    def width_fresh(self) -> bool:
        if self.last_width_stamp is None:
            return False
        return (self.now_sec() - self.last_width_stamp) <= self.width_timeout_sec

    def held_width_available(self) -> bool:
        if self.last_good_width_stamp is None or self.last_good_width_m is None:
            return False
        return (self.now_sec() - self.last_good_width_stamp) <= self.hold_last_width_sec

    def resolve_width(self) -> tuple[Optional[float], str]:
        if (
            self.lane_width_m is not None and
            self.width_fresh() and
            math.isfinite(self.lane_width_m) and
            self.lane_width_m >= self.min_valid_width_m
        ):
            return float(self.lane_width_m), 'live_width'
        if self.held_width_available():
            return float(self.last_good_width_m), 'held_width'
        # detection 실패 시 memory node가 기억한 도로 폭을 fallback으로 사용
        if (
            self.mem_width_m is not None and
            self.last_mem_width_stamp is not None and
            (self.now_sec() - self.last_mem_width_stamp) <= self.hold_last_width_sec
        ):
            return float(self.mem_width_m), 'mem_width'
        return None, 'no_width'

    def publish_decision(self, mode: str, safe_stop: bool, reason: str) -> None:
        msg_mode = String(); msg_mode.data = mode; self.control_mode_pub.publish(msg_mode)
        msg_stop = Bool(); msg_stop.data = bool(safe_stop); self.safe_stop_pub.publish(msg_stop)
        msg_status = String(); msg_status.data = reason; self.decision_status_pub.publish(msg_status)

    def decision_step(self) -> None:
        if not self.lane_status_fresh():
            self.current_mode = 'PASS_BLOCKED'
            self.current_safe_stop = True
            self.publish_decision('PASS_BLOCKED', True, 'lane_status_timeout')
            return

        if self.stop_on_lane_lost and self.lane_status != 'ok':
            self.current_mode = 'PASS_BLOCKED'
            self.current_safe_stop = True
            self.publish_decision('PASS_BLOCKED', True, f'lane_lost source={self.guidance_source}')
            return

        width, width_source = self.resolve_width()
        if width is None:
            self.current_mode = 'PASS_BLOCKED'
            self.current_safe_stop = True
            self.publish_decision('PASS_BLOCKED', True, f'no_valid_width source={self.guidance_source}')
            return

        open_threshold = self.min_passable_width_m + self.width_hysteresis_m
        close_threshold = self.min_passable_width_m

        if self.current_mode == 'NORMAL_CENTER_DRIVE':
            passable = width > close_threshold
        else:
            passable = width >= open_threshold

        if passable:
            self.current_mode = 'NORMAL_CENTER_DRIVE'
            self.current_safe_stop = False
            self.publish_decision(
                'NORMAL_CENTER_DRIVE',
                False,
                f'passable width={width:.3f}m width_src={width_source} guide_src={self.guidance_source} threshold={self.min_passable_width_m:.3f}m'
            )
        else:
            self.current_mode = 'PASS_BLOCKED'
            self.current_safe_stop = True
            self.publish_decision(
                'PASS_BLOCKED',
                True,
                f'blocked width={width:.3f}m width_src={width_source} guide_src={self.guidance_source} threshold={self.min_passable_width_m:.3f}m'
            )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LaneDecisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
