#!/usr/bin/env python3
"""
chassis_control_node
─────────────────────
통합 섀시 제어 노드.

/control_mode 에 따라 세 가지 제어기를 전환한다.

  NORMAL_CENTER_DRIVE  → PID 중앙 주행 제어
  KEEP_RIGHT_APPROACH  → PID 우측 오프셋 추종
  REENTER              → PID 중앙 주행 제어 (감속)
  REVERSE_EXECUTE      → Reverse Pure Pursuit (후진 경로 추종)
  WAIT_PASS            → 정지
  SAFE_STOP            → 즉시 정지
  그 외               → 정지

후진 목표 도달 시 /reverse_motion_done = True 를 발행해
v2v_decision_node가 WAIT_PASS로 전이할 수 있게 한다.

입력 토픽:
  /control_mode            String
  /lane_error_center_m_active Float32
  /lane_error_right_m_active  Float32
  /lane_heading_error_active  Float32
  /lane_status_active      String
  /safe_stop               Bool
  /reverse_path            nav_msgs/Path   (reverse_path_planner_node)
  /odom                    nav_msgs/Odometry

출력 토픽:
  /cmd_vel   geometry_msgs/Twist
  /reverse_motion_done Bool
"""

import math
from typing import List, Optional, Tuple

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def quat_to_yaw(q) -> float:
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


class ChassisControlNode(Node):
    def __init__(self) -> None:
        super().__init__('chassis_control_node')

        # ── 파라미터 ─────────────────────────────────────────────────────
        self.declare_parameter('control_hz', 20.0)

        # PID 차선 추종 (중앙 / 우측)
        self.declare_parameter('kp_m',         0.22)
        self.declare_parameter('k_heading',     0.03)
        self.declare_parameter('steering_sign', -1.0)
        self.declare_parameter('max_ang_z',     0.08)
        self.declare_parameter('nominal_speed', 0.028)
        self.declare_parameter('min_speed',     0.018)
        self.declare_parameter('slow_error_m',  0.10)
        self.declare_parameter('hard_stop_error_m', 0.25)

        # 후진 Pure Pursuit
        self.declare_parameter('reverse_lookahead_m', 0.30)  # 후진 전방주시 거리
        self.declare_parameter('reverse_speed',       0.018) # 후진 속도 (양수값; linear.x는 음수)
        self.declare_parameter('reverse_max_ang_z',   0.06)
        self.declare_parameter('reverse_goal_tol_m',  0.12)  # 목표 도달 판정 거리

        # 재진입 감속
        self.declare_parameter('reenter_speed_scale', 0.7)

        # 타임아웃
        self.declare_parameter('lane_timeout_sec', 0.35)

        # ── 파라미터 읽기 ─────────────────────────────────────────────────
        self.control_hz   = float(self.get_parameter('control_hz').value)
        self.kp_m         = float(self.get_parameter('kp_m').value)
        self.k_heading    = float(self.get_parameter('k_heading').value)
        self.steering_sign = float(self.get_parameter('steering_sign').value)
        self.max_ang_z    = float(self.get_parameter('max_ang_z').value)
        self.nominal_speed = float(self.get_parameter('nominal_speed').value)
        self.min_speed    = float(self.get_parameter('min_speed').value)
        self.slow_error_m = float(self.get_parameter('slow_error_m').value)
        self.hard_stop_error_m = float(self.get_parameter('hard_stop_error_m').value)

        self.reverse_lookahead_m = float(self.get_parameter('reverse_lookahead_m').value)
        self.reverse_speed       = float(self.get_parameter('reverse_speed').value)
        self.reverse_max_ang_z   = float(self.get_parameter('reverse_max_ang_z').value)
        self.reverse_goal_tol_m  = float(self.get_parameter('reverse_goal_tol_m').value)

        self.reenter_speed_scale = float(self.get_parameter('reenter_speed_scale').value)
        self.lane_timeout_sec    = float(self.get_parameter('lane_timeout_sec').value)

        # ── 내부 상태 ─────────────────────────────────────────────────────
        self.control_mode:  str = 'NORMAL_CENTER_DRIVE'
        self.safe_stop:     bool = False
        self.lane_status:   str = 'unknown'
        self.center_err:    Optional[float] = None
        self.right_err:     Optional[float] = None
        self.heading_err:   Optional[float] = None
        self.last_lane_stamp: Optional[float] = None

        self.odom_msg: Optional[Odometry] = None
        self.reverse_path: Optional[Path] = None
        self.reverse_done: bool = False

        # ── 구독 ─────────────────────────────────────────────────────────
        self.create_subscription(String,  '/control_mode',              self._cb_mode,    10)
        self.create_subscription(Bool,    '/safe_stop',                 self._cb_stop,    10)
        self.create_subscription(Float32, '/lane_error_center_m_active', self._cb_center, 10)
        self.create_subscription(Float32, '/lane_error_right_m_active',  self._cb_right,  10)
        self.create_subscription(Float32, '/lane_heading_error_active',  self._cb_heading,10)
        self.create_subscription(String,  '/lane_status_active',         self._cb_lane,   10)
        self.create_subscription(Odometry, '/odom',                      self._cb_odom,   10)
        self.create_subscription(Path,    '/reverse_path',               self._cb_path,   10)

        # ── 발행 ─────────────────────────────────────────────────────────
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_reverse_done = self.create_publisher(Bool, '/reverse_motion_done', 10)

        dt = 1.0 / self.control_hz if self.control_hz > 0 else 0.05
        self.create_timer(dt, self._step)
        self.get_logger().info('chassis_control_node 시작')

    # ── 콜백 ─────────────────────────────────────────────────────────────
    def _cb_mode(self, msg: String) -> None:
        new_mode = str(msg.data).strip()
        if new_mode != self.control_mode:
            self.get_logger().info(f'[Chassis] 모드 전환: {self.control_mode} → {new_mode}')
            if new_mode == 'REVERSE_EXECUTE':
                self.reverse_done = False
                self._publish_reverse_done(False)
        self.control_mode = new_mode

    def _cb_stop(self, msg: Bool) -> None:
        self.safe_stop = bool(msg.data)

    def _cb_center(self, msg: Float32) -> None:
        self.center_err = float(msg.data)
        self.last_lane_stamp = self._now()

    def _cb_right(self, msg: Float32) -> None:
        self.right_err = float(msg.data)
        self.last_lane_stamp = self._now()

    def _cb_heading(self, msg: Float32) -> None:
        v = float(msg.data)
        self.heading_err = v if math.isfinite(v) else None

    def _cb_lane(self, msg: String) -> None:
        self.lane_status = str(msg.data).strip().lower()
        self.last_lane_stamp = self._now()

    def _cb_odom(self, msg: Odometry) -> None:
        self.odom_msg = msg

    def _cb_path(self, msg: Path) -> None:
        self.reverse_path = msg
        self.reverse_done = False
        self._publish_reverse_done(False)

    # ── 유틸 ─────────────────────────────────────────────────────────────
    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _lane_fresh(self) -> bool:
        if self.last_lane_stamp is None:
            return False
        return (self._now() - self.last_lane_stamp) <= self.lane_timeout_sec

    def _stop(self) -> None:
        self.pub_cmd.publish(Twist())

    def _publish(self, linear_x: float, angular_z: float) -> None:
        tw = Twist()
        tw.linear.x  = float(linear_x)
        tw.angular.z = float(angular_z)
        self.pub_cmd.publish(tw)

    def _publish_reverse_done(self, done: bool) -> None:
        msg = Bool()
        msg.data = bool(done)
        self.pub_reverse_done.publish(msg)

    # ── PID 차선 추종 ─────────────────────────────────────────────────────
    def _compute_speed(self, abs_err: float) -> float:
        if abs_err >= self.hard_stop_error_m:
            return self.min_speed
        alpha = clamp(abs_err / max(self.slow_error_m, 1e-6), 0.0, 1.0)
        return self.nominal_speed - (self.nominal_speed - self.min_speed) * alpha

    def _pid_follow(self, err_m: Optional[float], speed_scale: float = 1.0) -> None:
        if err_m is None or not self._lane_fresh():
            self._stop()
            return
        psi = self.heading_err if (self.heading_err is not None and math.isfinite(self.heading_err)) else 0.0
        ang = self.steering_sign * (self.kp_m * err_m + self.k_heading * psi)
        ang = clamp(ang, -self.max_ang_z, self.max_ang_z)
        speed = self._compute_speed(abs(err_m)) * speed_scale
        self._publish(speed, ang)

    # ── Reverse Pure Pursuit ─────────────────────────────────────────────
    def _current_pose(self) -> Optional[Tuple[float, float, float]]:
        if self.odom_msg is None:
            return None
        p = self.odom_msg.pose.pose.position
        q = self.odom_msg.pose.pose.orientation
        return float(p.x), float(p.y), quat_to_yaw(q)

    def _find_lookahead_point(
        self,
        poses: List,
        cx: float,
        cy: float,
    ) -> Optional[Tuple[float, float]]:
        """
        경로에서 lookahead 거리 내 가장 먼 waypoint 반환.
        (후진이므로 경로 끝 방향이 목표)
        """
        best_idx = 0
        best_dist = 0.0
        for i, ps in enumerate(poses):
            wx = float(ps.pose.position.x)
            wy = float(ps.pose.position.y)
            d = math.hypot(wx - cx, wy - cy)
            if d <= self.reverse_lookahead_m and d >= best_dist:
                best_dist = d
                best_idx = i

        # 경로 내 lookahead 점 없으면 가장 가까운 미래 점
        target_ps = poses[best_idx]
        return float(target_ps.pose.position.x), float(target_ps.pose.position.y)

    def _reverse_pure_pursuit(self) -> None:
        if self.reverse_path is None or len(self.reverse_path.poses) == 0:
            self._stop()
            return

        pose = self._current_pose()
        if pose is None:
            self._stop()
            return
        cx, cy, yaw = pose

        # 목표 도달 확인
        last_pose = self.reverse_path.poses[-1]
        gx = float(last_pose.pose.position.x)
        gy = float(last_pose.pose.position.y)
        dist_to_goal = math.hypot(gx - cx, gy - cy)

        if dist_to_goal <= self.reverse_goal_tol_m:
            self.get_logger().info('[Chassis] 후진 목표 도달')
            self.reverse_done = True
            self._publish_reverse_done(True)
            self._stop()
            return

        # Lookahead 점 선택
        target = self._find_lookahead_point(self.reverse_path.poses, cx, cy)
        if target is None:
            self._stop()
            return
        tx, ty = target

        # 후진 Pure Pursuit 조향각 계산
        # 후진이므로 차량 후방 방향 기준: yaw + π
        rear_yaw = yaw + math.pi
        dx = tx - cx
        dy = ty - cy

        # 목표 방향과 후방 방향 사이 각도 오차
        angle_to_target = math.atan2(dy, dx)
        alpha = angle_to_target - rear_yaw
        # 정규화 [-π, π]
        alpha = math.atan2(math.sin(alpha), math.cos(alpha))

        # Pure Pursuit 곡률: κ = 2*sin(α) / L
        L = self.reverse_lookahead_m
        curvature = 2.0 * math.sin(alpha) / L

        # 후진: linear.x 음수, 조향 부호 반전
        ang = clamp(-curvature * self.reverse_speed, -self.reverse_max_ang_z, self.reverse_max_ang_z)
        self._publish(-self.reverse_speed, ang)

    # ── 메인 스텝 ─────────────────────────────────────────────────────────
    def _step(self) -> None:
        if self.safe_stop:
            self._stop()
            return

        mode = self.control_mode.strip().upper()

        if mode in ('SAFE_STOP', 'WAIT_PASS', 'PASS_BLOCKED'):
            self._stop()
            return

        if mode == 'NORMAL_CENTER_DRIVE':
            self._pid_follow(self.center_err, speed_scale=1.0)
            return

        if mode == 'KEEP_RIGHT_APPROACH':
            self._pid_follow(self.right_err, speed_scale=0.8)
            return

        if mode == 'REENTER':
            self._pid_follow(self.center_err, speed_scale=self.reenter_speed_scale)
            return

        if mode == 'REVERSE_EXECUTE':
            if self.reverse_done:
                self._publish_reverse_done(True)
                self._stop()
            else:
                self._publish_reverse_done(False)
                self._reverse_pure_pursuit()
            return

        # 기본: 정지
        self._stop()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ChassisControlNode()
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
