#!/usr/bin/env python3
import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Sequence, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String
from visualization_msgs.msg import Marker, MarkerArray


@dataclass
class LaneMapSample:
    x: float
    y: float
    width: float
    stamp: float


def quat_to_yaw(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class LaneMemoryNode(Node):
    def __init__(self) -> None:
        super().__init__('lane_memory_node')

        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('lane_status_topic', '/lane_status')
        self.declare_parameter('centerline_base_topic', '/lane_centerline_base_path')
        self.declare_parameter('left_boundary_base_topic', '/lane_left_boundary_base_path')
        self.declare_parameter('right_boundary_base_topic', '/lane_right_boundary_base_path')

        self.declare_parameter('sample_distance_m', 0.03)
        self.declare_parameter('max_samples', 700)
        self.declare_parameter('min_samples_for_valid', 8)
        self.declare_parameter('memory_timeout_sec', 1.2)
        self.declare_parameter('record_min_forward_m', 0.32)
        self.declare_parameter('record_max_forward_m', 1.40)
        self.declare_parameter('forward_window_min_m', -0.30)
        self.declare_parameter('forward_window_max_m', 1.60)
        self.declare_parameter('track_start_forward_m', 0.05)
        self.declare_parameter('cross_track_lookahead_m', 0.40)
        self.declare_parameter('goal_lookahead_m', 0.35)
        self.declare_parameter('heading_lookahead_m', 0.55)
        self.declare_parameter('endpoint_stop_distance_m', 0.10)
        self.declare_parameter('center_deadband_m', 0.008)
        self.declare_parameter('heading_deadband_rad', 0.030)
        self.declare_parameter('right_offset_ratio', 0.30)
        self.declare_parameter('publish_hz', 20.0)

        self.odom_topic = str(self.get_parameter('odom_topic').value)
        self.lane_status_topic = str(self.get_parameter('lane_status_topic').value)
        self.centerline_base_topic = str(self.get_parameter('centerline_base_topic').value)
        self.left_boundary_base_topic = str(self.get_parameter('left_boundary_base_topic').value)
        self.right_boundary_base_topic = str(self.get_parameter('right_boundary_base_topic').value)
        self.sample_distance_m = float(self.get_parameter('sample_distance_m').value)
        self.max_samples = int(self.get_parameter('max_samples').value)
        self.min_samples_for_valid = int(self.get_parameter('min_samples_for_valid').value)
        self.memory_timeout_sec = float(self.get_parameter('memory_timeout_sec').value)
        self.record_min_forward_m = float(self.get_parameter('record_min_forward_m').value)
        self.record_max_forward_m = float(self.get_parameter('record_max_forward_m').value)
        self.forward_window_min_m = float(self.get_parameter('forward_window_min_m').value)
        self.forward_window_max_m = float(self.get_parameter('forward_window_max_m').value)
        self.track_start_forward_m = float(self.get_parameter('track_start_forward_m').value)
        self.cross_track_lookahead_m = float(self.get_parameter('cross_track_lookahead_m').value)
        self.goal_lookahead_m = float(self.get_parameter('goal_lookahead_m').value)
        self.heading_lookahead_m = float(self.get_parameter('heading_lookahead_m').value)
        self.endpoint_stop_distance_m = float(self.get_parameter('endpoint_stop_distance_m').value)
        self.center_deadband_m = float(self.get_parameter('center_deadband_m').value)
        self.heading_deadband_rad = float(self.get_parameter('heading_deadband_rad').value)
        self.right_offset_ratio = float(self.get_parameter('right_offset_ratio').value)
        self.publish_hz = float(self.get_parameter('publish_hz').value)

        self.odom_msg: Optional[Odometry] = None
        self.lane_status: str = 'lost'
        self.centerline_base_path: Optional[Path] = None
        self.left_boundary_base_path: Optional[Path] = None
        self.right_boundary_base_path: Optional[Path] = None
        self.last_observed_stamp: Optional[float] = None

        self.path_buffer: Deque[LaneMapSample] = deque(maxlen=self.max_samples)

        self.create_subscription(Odometry, self.odom_topic, self.odom_cb, 10)
        self.create_subscription(String, self.lane_status_topic, self.status_cb, 10)
        self.create_subscription(Path, self.centerline_base_topic, self.centerline_cb, 10)
        self.create_subscription(Path, self.left_boundary_base_topic, self.left_boundary_cb, 10)
        self.create_subscription(Path, self.right_boundary_base_topic, self.right_boundary_cb, 10)

        self.mem_center_pub = self.create_publisher(Float32, '/lane_mem_error_center_m', 10)
        self.mem_right_pub = self.create_publisher(Float32, '/lane_mem_error_right_m', 10)
        self.mem_heading_pub = self.create_publisher(Float32, '/lane_mem_heading_error', 10)
        self.mem_valid_pub = self.create_publisher(Bool, '/lane_mem_valid', 10)
        self.mem_width_pub = self.create_publisher(Float32, '/lane_mem_width_m', 10)
        self.mem_status_pub = self.create_publisher(String, '/lane_mem_status', 10)
        self.mem_bridge_pub = self.create_publisher(Bool, '/lane_mem_bridge_ready', 10)
        self.mem_remaining_pub = self.create_publisher(Float32, '/lane_mem_remaining_m', 10)
        self.path_pub = self.create_publisher(Path, '/lane_memory_path', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/lane_memory_markers', 10)

        dt = 1.0 / self.publish_hz if self.publish_hz > 0 else 0.05
        self.timer = self.create_timer(dt, self.step)

    def now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def odom_cb(self, msg: Odometry) -> None:
        self.odom_msg = msg

    def status_cb(self, msg: String) -> None:
        self.lane_status = str(msg.data).strip().lower()

    def centerline_cb(self, msg: Path) -> None:
        self.centerline_base_path = msg
        self.last_observed_stamp = self.now_sec()

    def left_boundary_cb(self, msg: Path) -> None:
        self.left_boundary_base_path = msg
        self.last_observed_stamp = self.now_sec()

    def right_boundary_cb(self, msg: Path) -> None:
        self.right_boundary_base_path = msg
        self.last_observed_stamp = self.now_sec()

    def base_right_to_odom(self, xb: float, y_right: float) -> Tuple[float, float]:
        if self.odom_msg is None:
            raise RuntimeError('odom unavailable')
        p = self.odom_msg.pose.pose.position
        q = self.odom_msg.pose.pose.orientation
        yaw = quat_to_yaw(q)
        xo = p.x + math.cos(yaw) * xb + math.sin(yaw) * y_right
        yo = p.y + math.sin(yaw) * xb - math.cos(yaw) * y_right
        return xo, yo

    def odom_to_base_right(self, xo: float, yo: float) -> Tuple[float, float]:
        if self.odom_msg is None:
            raise RuntimeError('odom unavailable')
        p = self.odom_msg.pose.pose.position
        q = self.odom_msg.pose.pose.orientation
        yaw = quat_to_yaw(q)
        dx = xo - p.x
        dy = yo - p.y
        x_forward = math.cos(yaw) * dx + math.sin(yaw) * dy
        y_left = -math.sin(yaw) * dx + math.cos(yaw) * dy
        y_right = -y_left
        return x_forward, y_right

    def extract_xy(self, path: Optional[Path]) -> List[Tuple[float, float]]:
        if path is None:
            return []
        return [(float(ps.pose.position.x), float(ps.pose.position.y)) for ps in path.poses]

    def should_store(self, xo: float, yo: float) -> bool:
        if len(self.path_buffer) == 0:
            return True
        last = self.path_buffer[-1]
        return math.hypot(xo - last.x, yo - last.y) >= self.sample_distance_m

    def memory_fresh(self) -> bool:
        if self.last_observed_stamp is None:
            return False
        return (self.now_sec() - self.last_observed_stamp) <= self.memory_timeout_sec

    def store_current_observation(self) -> None:
        if self.odom_msg is None or self.lane_status != 'ok':
            return

        center_pts = self.extract_xy(self.centerline_base_path)
        left_pts = self.extract_xy(self.left_boundary_base_path)
        right_pts = self.extract_xy(self.right_boundary_base_path)
        n = min(len(center_pts), len(left_pts), len(right_pts))
        if n < 2:
            return

        for i in range(n):
            cx, cy = center_pts[i]
            lx, ly = left_pts[i]
            rx, ry = right_pts[i]
            if not (self.record_min_forward_m <= cx <= self.record_max_forward_m):
                continue
            width = float(ry - ly)
            if width <= 0.05:
                continue
            xo, yo = self.base_right_to_odom(cx, cy)
            if self.should_store(xo, yo):
                self.path_buffer.append(LaneMapSample(x=xo, y=yo, width=width, stamp=self.now_sec()))

    def current_curve_candidates(self) -> List[Tuple[float, float, float]]:
        out: List[Tuple[float, float, float]] = []
        if self.odom_msg is None:
            return out
        for s in self.path_buffer:
            xf, yr = self.odom_to_base_right(s.x, s.y)
            if self.forward_window_min_m <= xf <= self.forward_window_max_m:
                out.append((xf, yr, s.width))
        out.sort(key=lambda t: t[0])
        if len(out) < 2:
            return out

        # collapse almost-duplicate x samples by averaging
        merged: List[Tuple[float, float, float]] = []
        bucket = [out[0]]
        for cur in out[1:]:
            if abs(cur[0] - bucket[-1][0]) <= 0.005:
                bucket.append(cur)
            else:
                xs = np.array([b[0] for b in bucket], dtype=np.float32)
                ys = np.array([b[1] for b in bucket], dtype=np.float32)
                ws = np.array([b[2] for b in bucket], dtype=np.float32)
                merged.append((float(xs.mean()), float(ys.mean()), float(ws.mean())))
                bucket = [cur]
        xs = np.array([b[0] for b in bucket], dtype=np.float32)
        ys = np.array([b[1] for b in bucket], dtype=np.float32)
        ws = np.array([b[2] for b in bucket], dtype=np.float32)
        merged.append((float(xs.mean()), float(ys.mean()), float(ws.mean())))
        return merged

    def interpolate_triplet(self, curve: Sequence[Tuple[float, float, float]], xq: float) -> Optional[Tuple[float, float, float]]:
        if len(curve) == 0:
            return None
        if len(curve) == 1:
            x0, y0, w0 = curve[0]
            return xq, y0, w0

        for i in range(len(curve) - 1):
            x0, y0, w0 = curve[i]
            x1, y1, w1 = curve[i + 1]
            if x0 <= xq <= x1:
                if abs(x1 - x0) < 1e-6:
                    return xq, y0, w0
                a = (xq - x0) / (x1 - x0)
                yq = (1.0 - a) * y0 + a * y1
                wq = (1.0 - a) * w0 + a * w1
                return xq, float(yq), float(wq)
        return None

    def build_track(self, curve: Sequence[Tuple[float, float, float]]) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        if len(curve) == 0:
            return None

        # Important:
        # The detection node only publishes path points that are actually visible in front of the
        # robot, so the first remembered x is often around 0.35~0.45 m rather than near 0.0 m.
        # If we insist on interpolating at a fixed small track_start_forward_m (e.g. 0.05 m), the
        # start point becomes unavailable immediately after camera loss and memory guidance dies.
        # Instead, start from the requested tracking x when it lies inside the remembered curve,
        # otherwise start from the nearest available front point.
        if curve[0][0] <= self.track_start_forward_m <= curve[-1][0]:
            start = self.interpolate_triplet(curve, self.track_start_forward_m)
        else:
            start = curve[0]

        if start is None:
            return None

        start_x = float(start[0])
        pts: List[Tuple[float, float, float]] = [start]
        for x, y, w in curve:
            if x > start_x:
                pts.append((x, y, w))

        if len(pts) < 2:
            return None

        xs = np.array([p[0] for p in pts], dtype=np.float32)
        ys = np.array([p[1] for p in pts], dtype=np.float32)
        ws = np.array([p[2] for p in pts], dtype=np.float32)

        ds = np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2)
        ss = np.concatenate([np.array([0.0], dtype=np.float32), np.cumsum(ds, dtype=np.float32)])
        return xs, ys, ws, ss

    def point_at_arc(self, xs: np.ndarray, ys: np.ndarray, ws: np.ndarray, ss: np.ndarray, sq: float) -> Tuple[float, float, float]:
        sq = float(max(0.0, min(float(ss[-1]), sq)))
        x = float(np.interp(sq, ss, xs))
        y = float(np.interp(sq, ss, ys))
        w = float(np.interp(sq, ss, ws))
        return x, y, w

    def compute_memory_guidance(self) -> Tuple[bool, float, float, float, float, float]:
        # Important:
        # For camera-loss bridging, validity should be driven primarily by whether there is still a
        # remembered track AHEAD of the robot, not by a short freshness timeout. If we invalidate
        # memory immediately after the camera stops updating, the robot will stop even though a
        # perfectly usable remembered path is still available. Freshness is therefore used only as a
        # bootstrap guard when the buffer is still too small.
        if len(self.path_buffer) < self.min_samples_for_valid:
            if not self.memory_fresh():
                return False, 0.0, 0.0, float('nan'), 0.0, 0.0

        curve = self.current_curve_candidates()
        if len(curve) < 2:
            return False, 0.0, 0.0, float('nan'), 0.0, 0.0

        track = self.build_track(curve)
        if track is None:
            return False, 0.0, 0.0, float('nan'), 0.0, 0.0

        xs, ys, ws, ss = track
        remaining = float(ss[-1])
        if remaining < self.endpoint_stop_distance_m:
            return False, 0.0, 0.0, float('nan'), 0.0, remaining

        cross_x, cross_y, cross_w = self.point_at_arc(xs, ys, ws, ss, min(self.cross_track_lookahead_m, remaining))
        head_x, head_y, _ = self.point_at_arc(xs, ys, ws, ss, min(self.heading_lookahead_m, remaining))

        center_err = float(cross_y)
        heading_err = float(math.atan2(head_y - cross_y, max(1e-4, head_x - cross_x)))
        right_err = float(center_err + self.right_offset_ratio * cross_w)

        if abs(center_err) < self.center_deadband_m:
            center_err = 0.0
        if abs(heading_err) < self.heading_deadband_rad:
            heading_err = 0.0

        return True, center_err, right_err, heading_err, float(cross_w), remaining

    def build_path_msg(self) -> Path:
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = 'odom'
        for s in self.path_buffer:
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x = s.x
            ps.pose.position.y = s.y
            ps.pose.position.z = 0.0
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)
        return path

    def sample_heading(self, idx: int) -> float:
        if len(self.path_buffer) == 1:
            return 0.0
        i0 = max(0, idx - 1)
        i1 = min(len(self.path_buffer) - 1, idx + 1)
        if i0 == i1:
            return 0.0
        p0 = self.path_buffer[i0]
        p1 = self.path_buffer[i1]
        return math.atan2(p1.y - p0.y, p1.x - p0.x)

    def make_line_marker(self, ns: str, mid: int, rgba: Tuple[float, float, float, float], pts: List[Point]) -> Marker:
        mk = Marker()
        mk.header.frame_id = 'odom'
        mk.header.stamp = self.get_clock().now().to_msg()
        mk.ns = ns
        mk.id = mid
        mk.type = Marker.LINE_STRIP
        mk.action = Marker.ADD
        mk.pose.orientation.w = 1.0
        mk.scale.x = 0.02
        mk.color.r = float(rgba[0])
        mk.color.g = float(rgba[1])
        mk.color.b = float(rgba[2])
        mk.color.a = float(rgba[3])
        mk.points = pts
        return mk

    def build_marker_array(self) -> MarkerArray:
        arr = MarkerArray()
        if len(self.path_buffer) == 0:
            return arr

        center_pts: List[Point] = []
        left_pts: List[Point] = []
        right_pts: List[Point] = []
        for i, s in enumerate(self.path_buffer):
            hdg = self.sample_heading(i)
            right_unit_x = math.sin(hdg)
            right_unit_y = -math.cos(hdg)
            half_w = 0.5 * s.width
            center_pts.append(Point(x=float(s.x), y=float(s.y), z=0.02))
            left_pts.append(Point(x=float(s.x - half_w * right_unit_x), y=float(s.y - half_w * right_unit_y), z=0.01))
            right_pts.append(Point(x=float(s.x + half_w * right_unit_x), y=float(s.y + half_w * right_unit_y), z=0.01))

        arr.markers.append(self.make_line_marker('lane_memory', 0, (0.1, 1.0, 0.1, 1.0), center_pts))
        arr.markers.append(self.make_line_marker('lane_memory', 1, (0.2, 0.6, 1.0, 0.9), left_pts))
        arr.markers.append(self.make_line_marker('lane_memory', 2, (1.0, 0.3, 0.3, 0.9), right_pts))
        return arr

    def publish_memory(self, valid: bool, center_err: float, right_err: float, heading_err: float, width: float, remaining: float) -> None:
        msg = Float32(); msg.data = float(center_err); self.mem_center_pub.publish(msg)
        msg = Float32(); msg.data = float(right_err); self.mem_right_pub.publish(msg)
        msg = Float32(); msg.data = float(heading_err if valid else float('nan')); self.mem_heading_pub.publish(msg)
        msg = Float32(); msg.data = float(width); self.mem_width_pub.publish(msg)
        msg = Float32(); msg.data = float(remaining); self.mem_remaining_pub.publish(msg)
        b = Bool(); b.data = bool(valid); self.mem_valid_pub.publish(b)
        b2 = Bool(); b2.data = bool(valid and len(self.path_buffer) >= self.min_samples_for_valid); self.mem_bridge_pub.publish(b2)
        s = String(); s.data = 'ok' if valid else 'lost'; self.mem_status_pub.publish(s)
        self.path_pub.publish(self.build_path_msg())
        self.marker_pub.publish(self.build_marker_array())

    def step(self) -> None:
        self.store_current_observation()
        valid, center_err, right_err, heading_err, width, remaining = self.compute_memory_guidance()
        self.publish_memory(valid, center_err, right_err, heading_err, width, remaining)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LaneMemoryNode()
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
