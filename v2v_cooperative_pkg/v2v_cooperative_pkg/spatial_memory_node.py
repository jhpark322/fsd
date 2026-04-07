#!/usr/bin/env python3
"""
spatial_memory_node
───────────────────
V2V 협력 주행용 공간 기억 노드.

주행 중 pose + 도로 폭을 0.5m 간격으로 샘플링해 최근 10m 구간을 FIFO로 유지한다.
교착 상황에서 reverse_goal (후진해서 비켜줄 수 있는 목표 위치)을 계산한다.

입력 토픽:
  /odom        (nav_msgs/Odometry)   - 현재 위치·자세
  /lane_width  (std_msgs/Float32)    - 현재 도로 폭
  /vehicle_state (std_msgs/String)   - 현재 차량 상태 (DEADLOCK_DETECTED 시 goal 계산)

출력 토픽:
  /memory_status    String  ('ok' / 'insufficient' / 'no_goal')
  /reverse_goal     geometry_msgs/PoseStamped
  /reverse_goal_ready Bool
  /spatial_memory_markers  visualization_msgs/MarkerArray  (rviz 시각화)
"""

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

import rclpy
from geometry_msgs.msg import PoseStamped, Quaternion
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String
from visualization_msgs.msg import Marker, MarkerArray


@dataclass
class SpatialSample:
    x: float
    y: float
    yaw: float
    width: float
    stamp: float
    right_clearance: float = field(default=0.0)  # 우측 회피 가능 폭


def quat_to_yaw(q) -> float:
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


def yaw_to_quat(yaw: float) -> Quaternion:
    q = Quaternion()
    q.w = math.cos(yaw / 2.0)
    q.z = math.sin(yaw / 2.0)
    return q


class SpatialMemoryNode(Node):
    def __init__(self) -> None:
        super().__init__('spatial_memory_node')

        # ── 파라미터 ─────────────────────────────────────────────────────
        self.declare_parameter('sample_interval_m', 0.5)         # 저장 간격
        self.declare_parameter('max_memory_dist_m', 10.0)        # 최대 기억 거리
        self.declare_parameter('robot_width_m', 0.19)
        self.declare_parameter('min_right_clearance_m', 0.12)    # 양보 가능 판단 최소 우측 공간
        self.declare_parameter('goal_backward_offset_m', 0.10)   # 목표점 → 중심 후방 오프셋
        self.declare_parameter('publish_hz', 10.0)
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('lane_width_topic', '/lane_width')
        self.declare_parameter('vehicle_state_topic', '/vehicle_state')

        self.sample_interval_m   = float(self.get_parameter('sample_interval_m').value)
        self.max_memory_dist_m   = float(self.get_parameter('max_memory_dist_m').value)
        self.robot_width_m       = float(self.get_parameter('robot_width_m').value)
        self.min_right_clearance = float(self.get_parameter('min_right_clearance_m').value)
        self.goal_backward_offset = float(self.get_parameter('goal_backward_offset_m').value)
        self.publish_hz          = float(self.get_parameter('publish_hz').value)
        self.odom_topic          = str(self.get_parameter('odom_topic').value)
        self.lane_width_topic    = str(self.get_parameter('lane_width_topic').value)
        self.vehicle_state_topic = str(self.get_parameter('vehicle_state_topic').value)

        # ── 내부 상태 ────────────────────────────────────────────────────
        # FIFO 버퍼: 저장된 샘플 (가장 오래된 것 → 가장 최근 것)
        self.buffer: Deque[SpatialSample] = deque()
        self.total_dist: float = 0.0  # 버퍼 내 총 누적 거리

        self.odom_msg: Optional[Odometry] = None
        self.lane_width: Optional[float] = None
        self.vehicle_state: str = 'NORMAL_CENTER_DRIVE'

        self.reverse_goal: Optional[SpatialSample] = None
        self.goal_ready: bool = False

        # ── 구독 ─────────────────────────────────────────────────────────
        self.create_subscription(Odometry, self.odom_topic, self._cb_odom, 10)
        self.create_subscription(Float32, self.lane_width_topic, self._cb_width, 10)
        self.create_subscription(String, self.vehicle_state_topic, self._cb_state, 10)

        # ── 발행 ─────────────────────────────────────────────────────────
        self.pub_status = self.create_publisher(String,       '/memory_status',          10)
        self.pub_goal   = self.create_publisher(PoseStamped,  '/reverse_goal',           10)
        self.pub_ready  = self.create_publisher(Bool,         '/reverse_goal_ready',     10)
        self.pub_markers = self.create_publisher(MarkerArray, '/spatial_memory_markers', 10)

        dt = 1.0 / self.publish_hz if self.publish_hz > 0 else 0.1
        self.create_timer(dt, self._step)
        self.get_logger().info('spatial_memory_node 시작')

    # ── 콜백 ─────────────────────────────────────────────────────────────
    def _cb_odom(self, msg: Odometry) -> None:
        self.odom_msg = msg

    def _cb_width(self, msg: Float32) -> None:
        v = float(msg.data)
        if math.isfinite(v) and v > 0.0:
            self.lane_width = v

    def _cb_state(self, msg: String) -> None:
        prev = self.vehicle_state
        self.vehicle_state = str(msg.data).strip()
        # DEADLOCK 진입 시 reverse_goal 재계산
        if prev != 'DEADLOCK_DETECTED' and self.vehicle_state == 'DEADLOCK_DETECTED':
            self._compute_reverse_goal()

    # ── 샘플링 ──────────────────────────────────────────────────────────
    def _current_pose(self) -> Optional[Tuple[float, float, float]]:
        if self.odom_msg is None:
            return None
        p = self.odom_msg.pose.pose.position
        q = self.odom_msg.pose.pose.orientation
        return float(p.x), float(p.y), quat_to_yaw(q)

    def _should_sample(self, x: float, y: float) -> bool:
        if not self.buffer:
            return True
        last = self.buffer[-1]
        return math.hypot(x - last.x, y - last.y) >= self.sample_interval_m

    def _right_clearance(self, width: float) -> float:
        """우측으로 비켜줄 수 있는 공간 = (전체 폭 - 로봇 폭) / 2"""
        return max(0.0, (width - self.robot_width_m) / 2.0)

    def _prune_old_samples(self) -> None:
        """버퍼 총 길이가 max_memory_dist_m를 초과하면 오래된 샘플 제거"""
        while len(self.buffer) >= 2:
            oldest = self.buffer[0]
            second = self.buffer[1]
            seg_len = math.hypot(second.x - oldest.x, second.y - oldest.y)
            if self.total_dist - seg_len >= self.max_memory_dist_m:
                self.buffer.popleft()
                self.total_dist -= seg_len
            else:
                break

    def _try_sample(self) -> None:
        if self.odom_msg is None or self.lane_width is None:
            return
        pose = self._current_pose()
        if pose is None:
            return
        x, y, yaw = pose
        if not self._should_sample(x, y):
            return

        rc = self._right_clearance(self.lane_width)
        sample = SpatialSample(
            x=x, y=y, yaw=yaw,
            width=self.lane_width,
            stamp=self._now(),
            right_clearance=rc,
        )
        if self.buffer:
            last = self.buffer[-1]
            seg = math.hypot(x - last.x, y - last.y)
            self.total_dist += seg
        self.buffer.append(sample)
        self._prune_old_samples()

    # ── reverse_goal 계산 ──────────────────────────────────────────────
    def _compute_reverse_goal(self) -> None:
        """
        버퍼에서 우측 회피 공간이 충분한 가장 가까운 과거 지점을 reverse_goal로 선택.
        현재 위치에서 가장 최근 → 과거 방향으로 탐색.
        """
        if len(self.buffer) < 2:
            self.reverse_goal = None
            self.goal_ready = False
            self.get_logger().warn('[SpatialMemory] 샘플 부족 → reverse_goal 없음')
            return

        pose = self._current_pose()
        if pose is None:
            self.reverse_goal = None
            self.goal_ready = False
            return
        cur_x, cur_y, _ = pose

        # 버퍼를 최근→과거 순으로 탐색 (deque[-1]이 가장 최근)
        candidates: List[SpatialSample] = list(self.buffer)
        candidates_with_dist = [
            (math.hypot(s.x - cur_x, s.y - cur_y), s)
            for s in candidates
        ]
        # 가까운 것부터 정렬
        candidates_with_dist.sort(key=lambda t: t[0])

        for dist, sample in candidates_with_dist:
            if dist < 0.3:
                # 너무 가까운 것은 패스
                continue
            if sample.right_clearance >= self.min_right_clearance:
                self.reverse_goal = sample
                self.goal_ready = True
                self.get_logger().info(
                    f'[SpatialMemory] reverse_goal 선택: ({sample.x:.2f}, {sample.y:.2f}) '
                    f'우측여유={sample.right_clearance:.2f}m 거리={dist:.2f}m'
                )
                return

        self.reverse_goal = None
        self.goal_ready = False
        self.get_logger().warn('[SpatialMemory] 우측 회피 공간 충분한 지점 없음')

    # ── 발행 ─────────────────────────────────────────────────────────────
    def _publish_goal(self) -> None:
        ready = Bool(); ready.data = self.goal_ready
        self.pub_ready.publish(ready)

        if self.reverse_goal is None:
            return
        ps = PoseStamped()
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.header.frame_id = 'odom'
        ps.pose.position.x = self.reverse_goal.x
        ps.pose.position.y = self.reverse_goal.y
        ps.pose.position.z = 0.0
        ps.pose.orientation = yaw_to_quat(self.reverse_goal.yaw)
        self.pub_goal.publish(ps)

    def _publish_status(self) -> None:
        status = 'ok' if (len(self.buffer) >= 2) else 'insufficient'
        if self.goal_ready:
            status = 'ok'
        elif len(self.buffer) < 2:
            status = 'insufficient'
        else:
            status = 'no_goal'
        s = String(); s.data = status
        self.pub_status.publish(s)

    def _publish_markers(self) -> None:
        arr = MarkerArray()
        if not self.buffer:
            self.pub_markers.publish(arr)
            return

        # 경로 라인
        mk = Marker()
        mk.header.frame_id = 'odom'
        mk.header.stamp = self.get_clock().now().to_msg()
        mk.ns = 'spatial_memory'
        mk.id = 0
        mk.type = Marker.LINE_STRIP
        mk.action = Marker.ADD
        mk.pose.orientation.w = 1.0
        mk.scale.x = 0.03
        mk.color.r = 0.0; mk.color.g = 0.8; mk.color.b = 0.2; mk.color.a = 0.9

        from geometry_msgs.msg import Point as GPoint
        for s in self.buffer:
            p = GPoint(); p.x = s.x; p.y = s.y; p.z = 0.02
            mk.points.append(p)
        arr.markers.append(mk)

        # reverse_goal 마커
        if self.reverse_goal is not None:
            gk = Marker()
            gk.header.frame_id = 'odom'
            gk.header.stamp = self.get_clock().now().to_msg()
            gk.ns = 'reverse_goal'
            gk.id = 1
            gk.type = Marker.SPHERE
            gk.action = Marker.ADD
            gk.pose.position.x = self.reverse_goal.x
            gk.pose.position.y = self.reverse_goal.y
            gk.pose.position.z = 0.05
            gk.pose.orientation.w = 1.0
            gk.scale.x = 0.15; gk.scale.y = 0.15; gk.scale.z = 0.15
            gk.color.r = 1.0; gk.color.g = 0.2; gk.color.b = 0.0; gk.color.a = 1.0
            arr.markers.append(gk)

        self.pub_markers.publish(arr)

    # ── 메인 스텝 ────────────────────────────────────────────────────────
    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _step(self) -> None:
        self._try_sample()
        self._publish_status()
        self._publish_goal()
        self._publish_markers()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SpatialMemoryNode()
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
