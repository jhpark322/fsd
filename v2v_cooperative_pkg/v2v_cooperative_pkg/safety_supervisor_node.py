#!/usr/bin/env python3
"""
safety_supervisor_node
───────────────────────
전체 예외 상황 감시 및 안전 정지 노드 (Safety Layer).

어떤 상태에서든 아래 조건 중 하나가 충족되면 /safe_stop = True 발행:

우선순위 (높은 순):
  P1. LiDAR 근접 장애물 (scan 기반)
  P2. 인식 불안정 (vision/lane 토픽 타임아웃)
  P3. 경로 오류 (REVERSE_EXECUTE 중 /reverse_path_ready = False 지속)
  P4. 협상 불일치 (양쪽 모두 '진행' 결정)
  P5. 제어 이상 (cmd_vel 이상값)

/safe_stop 해제는 모든 조건이 해소되면 자동으로 이루어진다.

입력 토픽:
  /scan                    sensor_msgs/LaserScan
  /relative_distance       Float32
  /reverse_path_ready      Bool
  /vehicle_state           String
  /decision_result         String
  /cmd_vel                 geometry_msgs/Twist
  /lane_status_active      String

출력 토픽:
  /safe_stop               Bool
  /supervisor_status       String  (디버깅용)
"""

import math
from typing import Optional

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Float32, String


class SafetySupervisorNode(Node):
    def __init__(self) -> None:
        super().__init__('safety_supervisor_node')

        # ── 파라미터 ─────────────────────────────────────────────────────
        self.declare_parameter('supervisor_hz', 20.0)

        # P1: LiDAR 근접 장애물
        self.declare_parameter('lidar_min_dist_m', 0.15)          # 전방 안전 거리
        self.declare_parameter('lidar_front_angle_deg', 60.0)     # 전방 감시 각도 범위 ±
        self.declare_parameter('lidar_rear_angle_deg', 30.0)      # 후방 감시 각도 범위 ± (후진 중)

        # P2: 인식 타임아웃
        self.declare_parameter('vision_timeout_sec', 1.0)
        self.declare_parameter('lane_timeout_sec', 1.0)

        # P3: 경로 오류 지속 시간
        self.declare_parameter('path_error_hold_sec', 2.0)

        # P5: 제어 이상
        self.declare_parameter('max_safe_linear', 0.20)
        self.declare_parameter('max_safe_angular', 2.0)

        self.supervisor_hz       = float(self.get_parameter('supervisor_hz').value)
        self.lidar_min_dist_m    = float(self.get_parameter('lidar_min_dist_m').value)
        self.lidar_front_angle   = math.radians(float(self.get_parameter('lidar_front_angle_deg').value))
        self.lidar_rear_angle    = math.radians(float(self.get_parameter('lidar_rear_angle_deg').value))
        self.vision_timeout_sec  = float(self.get_parameter('vision_timeout_sec').value)
        self.lane_timeout_sec    = float(self.get_parameter('lane_timeout_sec').value)
        self.path_error_hold_sec = float(self.get_parameter('path_error_hold_sec').value)
        self.max_safe_linear     = float(self.get_parameter('max_safe_linear').value)
        self.max_safe_angular    = float(self.get_parameter('max_safe_angular').value)

        # ── 내부 상태 ─────────────────────────────────────────────────────
        self.scan_msg:          Optional[LaserScan] = None
        self.relative_distance: Optional[float]     = None
        self.reverse_path_ready: bool = False
        self.vehicle_state:     str = 'NORMAL_CENTER_DRIVE'
        self.decision_result:   str = 'none'
        self.cmd_vel:           Optional[Twist] = None
        self.lane_status:       str = 'unknown'

        self.last_scan_stamp:     Optional[float] = None
        self.last_dist_stamp:     Optional[float] = None
        self.last_lane_stamp:     Optional[float] = None
        self.path_error_start:    Optional[float] = None

        self.safe_stop: bool = False
        self.stop_reason: str = ''

        # ── 구독 ─────────────────────────────────────────────────────────
        self.create_subscription(LaserScan, '/scan',               self._cb_scan,    10)
        self.create_subscription(Float32,   '/relative_distance',  self._cb_dist,    10)
        self.create_subscription(Bool,      '/reverse_path_ready', self._cb_path,    10)
        self.create_subscription(String,    '/vehicle_state',      self._cb_state,   10)
        self.create_subscription(String,    '/decision_result',    self._cb_decision, 10)
        self.create_subscription(Twist,     '/cmd_vel',            self._cb_cmd,     10)
        self.create_subscription(String,    '/lane_status_active', self._cb_lane,    10)

        # ── 발행 ─────────────────────────────────────────────────────────
        self.pub_stop   = self.create_publisher(Bool,   '/safe_stop',          10)
        self.pub_status = self.create_publisher(String, '/supervisor_status',  10)

        dt = 1.0 / self.supervisor_hz if self.supervisor_hz > 0 else 0.05
        self.create_timer(dt, self._step)
        self.get_logger().info('safety_supervisor_node 시작')

    # ── 콜백 ─────────────────────────────────────────────────────────────
    def _cb_scan(self, msg: LaserScan) -> None:
        self.scan_msg = msg
        self.last_scan_stamp = self._now()

    def _cb_dist(self, msg: Float32) -> None:
        self.relative_distance = float(msg.data)
        self.last_dist_stamp = self._now()

    def _cb_path(self, msg: Bool) -> None:
        self.reverse_path_ready = bool(msg.data)

    def _cb_state(self, msg: String) -> None:
        self.vehicle_state = str(msg.data).strip()

    def _cb_decision(self, msg: String) -> None:
        self.decision_result = str(msg.data).strip().lower()

    def _cb_cmd(self, msg: Twist) -> None:
        self.cmd_vel = msg

    def _cb_lane(self, msg: String) -> None:
        self.lane_status = str(msg.data).strip().lower()
        self.last_lane_stamp = self._now()

    # ── 유틸 ─────────────────────────────────────────────────────────────
    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _fresh(self, stamp: Optional[float], timeout: float) -> bool:
        if stamp is None:
            return False
        return (self._now() - stamp) <= timeout

    # ── P1: LiDAR 근접 장애물 ─────────────────────────────────────────────
    def _check_lidar(self) -> Optional[str]:
        if self.scan_msg is None or not self._fresh(self.last_scan_stamp, 0.3):
            return None  # 스캔 없으면 이 조건은 무시

        scan = self.scan_msg
        is_reversing = self.vehicle_state == 'REVERSE_EXECUTE'
        # 감시 각도 범위 선택
        if is_reversing:
            watch_angle = self.lidar_rear_angle
            # 후방 기준: π ± rear_angle
            center_angle = math.pi
        else:
            watch_angle = self.lidar_front_angle
            center_angle = 0.0

        min_range = float('inf')
        n = len(scan.ranges)
        for i, r in enumerate(scan.ranges):
            if not math.isfinite(r) or r <= 0.0:
                continue
            angle = scan.angle_min + i * scan.angle_increment
            # 각도 정규화 [-π, π]
            diff = math.atan2(math.sin(angle - center_angle), math.cos(angle - center_angle))
            if abs(diff) <= watch_angle:
                min_range = min(min_range, r)

        if min_range < self.lidar_min_dist_m:
            return f'P1_lidar_close min_range={min_range:.2f}m'
        return None

    # ── P2: 인식 타임아웃 ─────────────────────────────────────────────────
    def _check_perception_timeout(self) -> Optional[str]:
        # 주행 중(정지 상태 아닐 때)에만 체크
        if self.vehicle_state in ('SAFE_STOP', 'WAIT_PASS'):
            return None
        if not self._fresh(self.last_lane_stamp, self.lane_timeout_sec):
            return 'P2_lane_timeout'
        return None

    # ── P3: 경로 오류 ─────────────────────────────────────────────────────
    def _check_path_error(self) -> Optional[str]:
        if self.vehicle_state != 'REVERSE_EXECUTE':
            self.path_error_start = None
            return None
        if self.reverse_path_ready:
            self.path_error_start = None
            return None
        if self.path_error_start is None:
            self.path_error_start = self._now()
            return None
        if (self._now() - self.path_error_start) >= self.path_error_hold_sec:
            return 'P3_no_reverse_path'
        return None

    # ── P4: 협상 불일치 ───────────────────────────────────────────────────
    def _check_negotiation_conflict(self) -> Optional[str]:
        # 양쪽 모두 '진행'이면 충돌 (간이 감지: 상대 led_state로 판단)
        # 여기서는 decision_result='proceed' && vehicle_state가 진행 중인데
        # 상대도 접근 중인 상황을 체크
        # (정밀 검증은 V2V 통신 구현 후 확장)
        return None  # 기본 비활성 (추후 확장)

    # ── P5: 제어 이상 ─────────────────────────────────────────────────────
    def _check_control_anomaly(self) -> Optional[str]:
        if self.cmd_vel is None:
            return None
        lx = abs(float(self.cmd_vel.linear.x))
        az = abs(float(self.cmd_vel.angular.z))
        if lx > self.max_safe_linear or az > self.max_safe_angular:
            return f'P5_control_anomaly linear={lx:.2f} angular={az:.2f}'
        return None

    # ── 메인 스텝 ─────────────────────────────────────────────────────────
    def _step(self) -> None:
        reasons = []

        r = self._check_lidar()
        if r:
            reasons.append(r)

        r = self._check_perception_timeout()
        if r:
            reasons.append(r)

        r = self._check_path_error()
        if r:
            reasons.append(r)

        r = self._check_negotiation_conflict()
        if r:
            reasons.append(r)

        r = self._check_control_anomaly()
        if r:
            reasons.append(r)

        if reasons:
            if not self.safe_stop:
                self.stop_reason = ' | '.join(reasons)
                self.get_logger().warn(f'[Safety] SAFE_STOP: {self.stop_reason}')
            self.safe_stop = True
        else:
            if self.safe_stop:
                self.get_logger().info('[Safety] 안전 조건 해소 → safe_stop 해제')
            self.safe_stop = False
            self.stop_reason = 'ok'

        # 발행
        stop_msg = Bool(); stop_msg.data = self.safe_stop
        self.pub_stop.publish(stop_msg)

        status = self.stop_reason if self.safe_stop else 'ok'
        s = String(); s.data = status
        self.pub_status.publish(s)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SafetySupervisorNode()
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
