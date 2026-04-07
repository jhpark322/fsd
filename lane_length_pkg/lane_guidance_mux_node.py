#!/usr/bin/env python3
import math
from typing import Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String


class LaneGuidanceMuxNode(Node):
    def __init__(self) -> None:
        super().__init__('lane_guidance_mux_node')

        self.declare_parameter('live_center_topic', '/lane_error_center_m')
        self.declare_parameter('live_right_topic', '/lane_error_right_m')
        self.declare_parameter('live_heading_topic', '/lane_heading_error')
        self.declare_parameter('live_status_topic', '/lane_status')

        self.declare_parameter('mem_center_topic', '/lane_mem_error_center_m')
        self.declare_parameter('mem_right_topic', '/lane_mem_error_right_m')
        self.declare_parameter('mem_heading_topic', '/lane_mem_heading_error')
        self.declare_parameter('mem_valid_topic', '/lane_mem_valid')
        self.declare_parameter('mem_status_topic', '/lane_mem_status')

        self.declare_parameter('active_center_topic', '/lane_error_center_m_active')
        self.declare_parameter('active_right_topic', '/lane_error_right_m_active')
        self.declare_parameter('active_heading_topic', '/lane_heading_error_active')
        self.declare_parameter('active_status_topic', '/lane_status_active')
        self.declare_parameter('source_topic', '/lane_guidance_source')

        self.declare_parameter('publish_hz', 20.0)
        self.declare_parameter('live_timeout_sec', 0.35)
        self.declare_parameter('mem_timeout_sec', 1.20)
        self.declare_parameter('blend_when_both_valid', False)
        self.declare_parameter('live_weight', 0.95)
        self.declare_parameter('blend_center_diff_limit_m', 0.05)
        self.declare_parameter('blend_heading_diff_limit_rad', 0.20)

        self.live_center_topic = str(self.get_parameter('live_center_topic').value)
        self.live_right_topic = str(self.get_parameter('live_right_topic').value)
        self.live_heading_topic = str(self.get_parameter('live_heading_topic').value)
        self.live_status_topic = str(self.get_parameter('live_status_topic').value)
        self.mem_center_topic = str(self.get_parameter('mem_center_topic').value)
        self.mem_right_topic = str(self.get_parameter('mem_right_topic').value)
        self.mem_heading_topic = str(self.get_parameter('mem_heading_topic').value)
        self.mem_valid_topic = str(self.get_parameter('mem_valid_topic').value)
        self.mem_status_topic = str(self.get_parameter('mem_status_topic').value)
        self.active_center_topic = str(self.get_parameter('active_center_topic').value)
        self.active_right_topic = str(self.get_parameter('active_right_topic').value)
        self.active_heading_topic = str(self.get_parameter('active_heading_topic').value)
        self.active_status_topic = str(self.get_parameter('active_status_topic').value)
        self.source_topic = str(self.get_parameter('source_topic').value)
        self.publish_hz = float(self.get_parameter('publish_hz').value)
        self.live_timeout_sec = float(self.get_parameter('live_timeout_sec').value)
        self.mem_timeout_sec = float(self.get_parameter('mem_timeout_sec').value)
        self.blend_when_both_valid = bool(self.get_parameter('blend_when_both_valid').value)
        self.live_weight = float(self.get_parameter('live_weight').value)
        self.blend_center_diff_limit_m = float(self.get_parameter('blend_center_diff_limit_m').value)
        self.blend_heading_diff_limit_rad = float(self.get_parameter('blend_heading_diff_limit_rad').value)

        self.live_center: Optional[float] = None
        self.live_right: Optional[float] = None
        self.live_heading: Optional[float] = None
        self.live_status: str = 'lost'
        self.live_stamp: Optional[float] = None

        self.mem_center: Optional[float] = None
        self.mem_right: Optional[float] = None
        self.mem_heading: Optional[float] = None
        self.mem_valid: bool = False
        self.mem_status: str = 'lost'
        self.mem_stamp: Optional[float] = None

        self.create_subscription(Float32, self.live_center_topic, self.live_center_cb, 10)
        self.create_subscription(Float32, self.live_right_topic, self.live_right_cb, 10)
        self.create_subscription(Float32, self.live_heading_topic, self.live_heading_cb, 10)
        self.create_subscription(String, self.live_status_topic, self.live_status_cb, 10)
        self.create_subscription(Float32, self.mem_center_topic, self.mem_center_cb, 10)
        self.create_subscription(Float32, self.mem_right_topic, self.mem_right_cb, 10)
        self.create_subscription(Float32, self.mem_heading_topic, self.mem_heading_cb, 10)
        self.create_subscription(Bool, self.mem_valid_topic, self.mem_valid_cb, 10)
        self.create_subscription(String, self.mem_status_topic, self.mem_status_cb, 10)

        self.active_center_pub = self.create_publisher(Float32, self.active_center_topic, 10)
        self.active_right_pub = self.create_publisher(Float32, self.active_right_topic, 10)
        self.active_heading_pub = self.create_publisher(Float32, self.active_heading_topic, 10)
        self.active_status_pub = self.create_publisher(String, self.active_status_topic, 10)
        self.source_pub = self.create_publisher(String, self.source_topic, 10)

        dt = 1.0 / self.publish_hz if self.publish_hz > 0 else 0.05
        self.timer = self.create_timer(dt, self.step)

    def now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def is_fresh(self, stamp: Optional[float], timeout: float) -> bool:
        if stamp is None:
            return False
        return (self.now_sec() - stamp) <= timeout

    def live_center_cb(self, msg: Float32) -> None:
        self.live_center = float(msg.data)
        self.live_stamp = self.now_sec()

    def live_right_cb(self, msg: Float32) -> None:
        self.live_right = float(msg.data)
        self.live_stamp = self.now_sec()

    def live_heading_cb(self, msg: Float32) -> None:
        v = float(msg.data)
        self.live_heading = v if math.isfinite(v) else None
        self.live_stamp = self.now_sec()

    def live_status_cb(self, msg: String) -> None:
        self.live_status = str(msg.data).strip().lower()
        self.live_stamp = self.now_sec()

    def mem_center_cb(self, msg: Float32) -> None:
        self.mem_center = float(msg.data)
        self.mem_stamp = self.now_sec()

    def mem_right_cb(self, msg: Float32) -> None:
        self.mem_right = float(msg.data)
        self.mem_stamp = self.now_sec()

    def mem_heading_cb(self, msg: Float32) -> None:
        v = float(msg.data)
        self.mem_heading = v if math.isfinite(v) else None
        self.mem_stamp = self.now_sec()

    def mem_valid_cb(self, msg: Bool) -> None:
        self.mem_valid = bool(msg.data)
        self.mem_stamp = self.now_sec()

    def mem_status_cb(self, msg: String) -> None:
        self.mem_status = str(msg.data).strip().lower()
        self.mem_stamp = self.now_sec()

    def publish(self, center: Optional[float], right: Optional[float], heading: Optional[float], status: str, source: str) -> None:
        m = Float32(); m.data = float(center if center is not None else 0.0); self.active_center_pub.publish(m)
        m = Float32(); m.data = float(right if right is not None else 0.0); self.active_right_pub.publish(m)
        m = Float32(); m.data = float(heading if heading is not None else float('nan')); self.active_heading_pub.publish(m)
        s = String(); s.data = status; self.active_status_pub.publish(s)
        s2 = String(); s2.data = source; self.source_pub.publish(s2)

    def step(self) -> None:
        # heading은 optional: nan이어도 control_node의 hold/fallback이 처리하므로
        # center/right error만 있으면 live/mem을 valid로 판단한다.
        live_ok = (
            self.live_status == 'ok' and
            self.is_fresh(self.live_stamp, self.live_timeout_sec) and
            self.live_center is not None and
            self.live_right is not None
        )
        mem_ok = (
            self.mem_valid and
            self.mem_status == 'ok' and
            self.is_fresh(self.mem_stamp, self.mem_timeout_sec) and
            self.mem_center is not None and
            self.mem_right is not None
        )

        if live_ok and mem_ok and self.blend_when_both_valid:
            center_diff = abs(self.live_center - self.mem_center)
            lh = self.live_heading if self.live_heading is not None else 0.0
            mh = self.mem_heading if self.mem_heading is not None else 0.0
            heading_diff = abs(lh - mh)
            if center_diff <= self.blend_center_diff_limit_m and heading_diff <= self.blend_heading_diff_limit_rad:
                wl = max(0.0, min(1.0, self.live_weight))
                wm = 1.0 - wl
                center = wl * self.live_center + wm * self.mem_center
                right = wl * self.live_right + wm * self.mem_right
                heading = wl * lh + wm * mh
                self.publish(center, right, heading, 'ok', 'blend')
                return
            self.publish(self.live_center, self.live_right, self.live_heading, 'ok', 'live')
            return

        if live_ok:
            self.publish(self.live_center, self.live_right, self.live_heading, 'ok', 'live')
            return

        if mem_ok:
            self.publish(self.mem_center, self.mem_right, self.mem_heading, 'ok', 'memory')
            return

        self.publish(0.0, 0.0, None, 'lost', 'none')


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LaneGuidanceMuxNode()
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
