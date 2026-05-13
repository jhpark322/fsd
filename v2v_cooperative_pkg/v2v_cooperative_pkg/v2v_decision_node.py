#!/usr/bin/env python3
"""
v2v_decision_node
─────────────────
전체 협력 양보 주행 상태 기계(State Machine).

상태 전이:
  NORMAL_CENTER_DRIVE
    → KEEP_RIGHT_APPROACH   (상대 차량 검출, relative_distance < approach_dist)
  KEEP_RIGHT_APPROACH
    → DEADLOCK_DETECTED     (교행 불가: relative_distance < deadlock_dist AND lane_width < passable)
    → NORMAL_CENTER_DRIVE   (상대 차량 사라짐)
  DEADLOCK_DETECTED
    → RPS_NEGOTIATION       (즉시)
  RPS_NEGOTIATION
    → SCORE_BASED_DECISION  (협상 실패: timeout / 거부 / 오류)
    → REVERSE_EXECUTE       (협상 승리 → 진행; 패배 → 양보 = REVERSE)
    → WAIT_PASS             (협상 승리 → 진행 차량은 WAIT_PASS 아님, 이동)
  SCORE_BASED_DECISION
    → REVERSE_EXECUTE       (내 점수 > 상대: 양보)
    → WAIT_PASS             (내 점수 < 상대: 대기)
  REVERSE_EXECUTE
    → WAIT_PASS             (후진 완료: reverse_goal 도달)
  WAIT_PASS
    → REENTER               (상대 차량 통과 완료: relative_distance > clear_dist)
  REENTER
    → NORMAL_CENTER_DRIVE   (재정렬 완료)
  어느 상태에서든 → SAFE_STOP (safety_supervisor /safe_stop)

입력 토픽:
  /lane_error_center_m_active   Float32
  /lane_error_right_m_active    Float32
  /lane_width                   Float32
  /relative_distance            Float32   (vision_perception_node)
  /led_state                    String    (vision_perception_node)
  /opponent_yield_score         Float32   (visual_v2v_perception_node)
  /memory_status                String    (spatial_memory_node)
  /reverse_goal_ready           Bool      (spatial_memory_node)
  /reverse_motion_done          Bool      (chassis_control_node)
  /rps_result                   String    ('win'/'lose'/'draw'/'none')
  /rps_timeout                  Bool
  /safe_stop                    Bool      (safety_supervisor_node)

출력 토픽:
  /vehicle_state    String
  /control_mode     String
  /negotiation_request  Bool
  /decision_result  String   ('yield'/'proceed'/'draw')
  /led_command      String
  /ego_yield_score  Float32
"""

import math
from typing import Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String


# ── 상태 정의 ──────────────────────────────────────────────────────────────
class State:
    NORMAL_CENTER_DRIVE  = 'NORMAL_CENTER_DRIVE'
    KEEP_RIGHT_APPROACH  = 'KEEP_RIGHT_APPROACH'
    DEADLOCK_DETECTED    = 'DEADLOCK_DETECTED'
    RPS_NEGOTIATION      = 'RPS_NEGOTIATION'
    SCORE_BASED_DECISION = 'SCORE_BASED_DECISION'
    REVERSE_EXECUTE      = 'REVERSE_EXECUTE'
    WAIT_PASS            = 'WAIT_PASS'
    REENTER              = 'REENTER'
    SAFE_STOP            = 'SAFE_STOP'


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class V2VDecisionNode(Node):
    def __init__(self) -> None:
        super().__init__('v2v_decision_node')

        # ── 파라미터 선언 ──────────────────────────────────────────────────
        self.declare_parameter('decision_hz', 10.0)
        self.declare_parameter('robot_width_m', 0.19)

        # 거리 임계값
        self.declare_parameter('approach_dist_m', 1.50)    # 상대 차량 검출 → 우측통행
        self.declare_parameter('deadlock_dist_m', 0.60)    # 교착 판단 거리
        self.declare_parameter('clear_dist_m', 1.20)       # 상대 통과 완료 판단 거리
        self.declare_parameter('passable_width_margin_m', 0.05)  # 교행 가능 폭 여유

        # 타임아웃
        self.declare_parameter('deadlock_confirm_sec', 0.8)   # 교착 연속 감지 시간
        self.declare_parameter('rps_timeout_sec', 10.0)       # 협상 최대 대기 시간
        self.declare_parameter('reenter_duration_sec', 2.0)   # 재진입 유지 시간
        self.declare_parameter('wait_pass_timeout_sec', 30.0) # 상대 통과 대기 최대 시간
        self.declare_parameter('sensor_timeout_sec', 0.5)

        # 점수 기반 판단 가중치
        self.declare_parameter('w_space', 0.30)
        self.declare_parameter('w_reverse', 0.25)
        self.declare_parameter('w_entry', 0.20)
        self.declare_parameter('w_wait', 0.10)
        self.declare_parameter('w_stability', 0.10)
        self.declare_parameter('w_dist', 0.05)

        # ── 파라미터 읽기 ──────────────────────────────────────────────────
        self.decision_hz          = float(self.get_parameter('decision_hz').value)
        self.robot_width_m        = float(self.get_parameter('robot_width_m').value)
        self.approach_dist_m      = float(self.get_parameter('approach_dist_m').value)
        self.deadlock_dist_m      = float(self.get_parameter('deadlock_dist_m').value)
        self.clear_dist_m         = float(self.get_parameter('clear_dist_m').value)
        self.passable_width_margin = float(self.get_parameter('passable_width_margin_m').value)
        self.deadlock_confirm_sec  = float(self.get_parameter('deadlock_confirm_sec').value)
        self.rps_timeout_sec       = float(self.get_parameter('rps_timeout_sec').value)
        self.reenter_duration_sec  = float(self.get_parameter('reenter_duration_sec').value)
        self.wait_pass_timeout_sec = float(self.get_parameter('wait_pass_timeout_sec').value)
        self.sensor_timeout_sec    = float(self.get_parameter('sensor_timeout_sec').value)
        self.w_space      = float(self.get_parameter('w_space').value)
        self.w_reverse    = float(self.get_parameter('w_reverse').value)
        self.w_entry      = float(self.get_parameter('w_entry').value)
        self.w_wait       = float(self.get_parameter('w_wait').value)
        self.w_stability  = float(self.get_parameter('w_stability').value)
        self.w_dist       = float(self.get_parameter('w_dist').value)

        # ── 내부 상태 변수 ─────────────────────────────────────────────────
        self.state: str = State.NORMAL_CENTER_DRIVE
        self.state_entry_time: float = self.now_sec()

        # 센서 입력값
        self.lane_error_center: Optional[float] = None
        self.lane_error_right: Optional[float] = None
        self.lane_width: Optional[float] = None
        self.relative_distance: Optional[float] = None
        self.led_state: str = 'unknown'
        self.opponent_yield_score: float = 0.0
        self.ego_yield_score: float = 0.0
        self.memory_status: str = 'lost'
        self.reverse_goal_ready: bool = False
        self.reverse_motion_done: bool = False
        self.rps_result: str = 'none'
        self.rps_timed_out: bool = False
        self.safe_stop_flag: bool = False

        self.last_distance_stamp: Optional[float] = None
        self.last_lane_stamp: Optional[float] = None

        # 교착 감지 연속 카운터
        self.deadlock_start_time: Optional[float] = None

        # 점수 기반 판단 결과
        self.decision_result: str = 'none'   # 'yield' / 'proceed'

        # 협상 요청 발행 여부
        self._rps_request_sent: bool = False

        # 후진 완료 여부는 chassis_control_node가 발행한다.

        # 대기 시간 누적 (점수 계산용)
        self.wait_accumulated_sec: float = 0.0

        # ── 구독 ───────────────────────────────────────────────────────────
        self.create_subscription(Float32, '/lane_error_center_m_active', self._cb_center, 10)
        self.create_subscription(Float32, '/lane_error_right_m_active',  self._cb_right,  10)
        self.create_subscription(Float32, '/lane_width',                  self._cb_width,  10)
        self.create_subscription(Float32, '/relative_distance',           self._cb_dist,   10)
        self.create_subscription(String,  '/led_state',                   self._cb_led_state, 10)
        self.create_subscription(Float32, '/opponent_yield_score',        self._cb_opponent_score, 10)
        self.create_subscription(String,  '/memory_status',               self._cb_mem_status, 10)
        self.create_subscription(Bool,    '/reverse_goal_ready',          self._cb_rev_goal, 10)
        self.create_subscription(Bool,    '/reverse_motion_done',         self._cb_reverse_done, 10)
        self.create_subscription(String,  '/rps_result',                  self._cb_rps_result, 10)
        self.create_subscription(Bool,    '/rps_timeout',                 self._cb_rps_timeout, 10)
        self.create_subscription(Bool,    '/safe_stop',                   self._cb_safe_stop, 10)

        # ── 발행 ───────────────────────────────────────────────────────────
        self.pub_state    = self.create_publisher(String, '/vehicle_state',       10)
        self.pub_mode     = self.create_publisher(String, '/control_mode',        10)
        self.pub_neg_req  = self.create_publisher(Bool,   '/negotiation_request', 10)
        self.pub_decision = self.create_publisher(String, '/decision_result',     10)
        self.pub_led      = self.create_publisher(String, '/led_command',         10)
        self.pub_ego_score = self.create_publisher(Float32, '/ego_yield_score',    10)

        dt = 1.0 / self.decision_hz if self.decision_hz > 0 else 0.1
        self.create_timer(dt, self._step)
        self.get_logger().info('v2v_decision_node 시작')

    # ── 유틸 ───────────────────────────────────────────────────────────────
    def now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def time_in_state(self) -> float:
        return self.now_sec() - self.state_entry_time

    def transition(self, new_state: str) -> None:
        self.get_logger().info(f'[FSM] {self.state} → {new_state}')
        self.state = new_state
        self.state_entry_time = self.now_sec()
        self._on_entry(new_state)

    def _on_entry(self, state: str) -> None:
        if state == State.RPS_NEGOTIATION:
            self._rps_request_sent = False
            self.rps_result = 'none'
            self.rps_timed_out = False
        elif state == State.SCORE_BASED_DECISION:
            self.decision_result = 'none'
        elif state == State.REVERSE_EXECUTE:
            self.reverse_motion_done = False
        elif state == State.NORMAL_CENTER_DRIVE:
            self.wait_accumulated_sec = 0.0
            self.deadlock_start_time = None
            self.decision_result = 'none'
            self.reverse_motion_done = False
        elif state == State.SAFE_STOP:
            pass

    # ── 콜백 ───────────────────────────────────────────────────────────────
    def _cb_center(self, msg: Float32) -> None:
        self.lane_error_center = float(msg.data)
        self.last_lane_stamp = self.now_sec()

    def _cb_right(self, msg: Float32) -> None:
        self.lane_error_right = float(msg.data)
        self.last_lane_stamp = self.now_sec()

    def _cb_width(self, msg: Float32) -> None:
        self.lane_width = float(msg.data)

    def _cb_dist(self, msg: Float32) -> None:
        self.relative_distance = float(msg.data)
        self.last_distance_stamp = self.now_sec()

    def _cb_led_state(self, msg: String) -> None:
        self.led_state = str(msg.data).strip()

    def _cb_opponent_score(self, msg: Float32) -> None:
        self.opponent_yield_score = clamp(float(msg.data), 0.0, 1.0)

    def _cb_mem_status(self, msg: String) -> None:
        self.memory_status = str(msg.data).strip()

    def _cb_rev_goal(self, msg: Bool) -> None:
        self.reverse_goal_ready = bool(msg.data)

    def _cb_reverse_done(self, msg: Bool) -> None:
        self.reverse_motion_done = bool(msg.data)

    def _cb_rps_result(self, msg: String) -> None:
        self.rps_result = str(msg.data).strip().lower()

    def _cb_rps_timeout(self, msg: Bool) -> None:
        self.rps_timed_out = bool(msg.data)

    def _cb_safe_stop(self, msg: Bool) -> None:
        self.safe_stop_flag = bool(msg.data)

    # ── 센서 신선도 ─────────────────────────────────────────────────────────
    def _distance_fresh(self) -> bool:
        if self.last_distance_stamp is None:
            return False
        return (self.now_sec() - self.last_distance_stamp) <= self.sensor_timeout_sec

    def _opponent_detected(self) -> bool:
        return (
            self._distance_fresh() and
            self.relative_distance is not None and
            math.isfinite(self.relative_distance) and
            self.relative_distance < self.approach_dist_m
        )

    def _deadlock_condition(self) -> bool:
        if not self._distance_fresh():
            return False
        if self.relative_distance is None:
            return False
        dist_close = self.relative_distance < self.deadlock_dist_m
        # 도로 폭이 교행 가능 한계 이하 (2*robot_width + margin)
        min_passable = 2.0 * self.robot_width_m + self.passable_width_margin
        width_narrow = (self.lane_width is not None and self.lane_width < min_passable)
        return dist_close and width_narrow

    def _opponent_cleared(self) -> bool:
        if not self._distance_fresh():
            return False
        return (
            self.relative_distance is not None and
            self.relative_distance > self.clear_dist_m
        )

    # ── Yield Score 계산 ────────────────────────────────────────────────────
    def _compute_yield_score(self) -> float:
        """
        점수가 높을수록 양보 가능성이 높음 (= 이 차량이 양보해야 함).
        각 sub-score는 0~1 정규화.
        """
        # S_space: 우측 회피 공간 (클수록 내가 비키기 쉬움 → 양보 점수 높음)
        s_space = 0.5
        if self.lane_width is not None and math.isfinite(self.lane_width):
            # 차로 폭 기준 0.3~0.8m 사이를 0~1로 매핑
            s_space = clamp((self.lane_width - 0.3) / 0.5, 0.0, 1.0)

        # S_reverse: spatial_memory에 reverse_goal 존재 여부
        s_reverse = 1.0 if self.reverse_goal_ready else 0.0

        # S_entry: 협소 구간 진입 정도 (접근 거리가 짧을수록 진입 많음 → 양보 불리)
        #          여기서는 간단히 0.5 고정 (진입 거리 토픽 추가 시 교체)
        s_entry = 0.5

        # S_wait: 대기 누적 시간 (길수록 기다린 쪽 → 진행 유리)
        s_wait = clamp(self.wait_accumulated_sec / 10.0, 0.0, 1.0)
        s_wait = 1.0 - s_wait  # 오래 기다렸으면 양보 점수 낮춤

        # S_stability: 차선 인식 안정성 (last_lane_stamp 기반)
        if self.last_lane_stamp is not None:
            age = self.now_sec() - self.last_lane_stamp
            s_stability = clamp(1.0 - age / self.sensor_timeout_sec, 0.0, 1.0)
        else:
            s_stability = 0.0

        # S_dist: 상대 차량과의 거리 (가까울수록 양보 압박 높음)
        s_dist = 0.5
        if self.relative_distance is not None and math.isfinite(self.relative_distance):
            s_dist = clamp(1.0 - self.relative_distance / self.approach_dist_m, 0.0, 1.0)

        score = (
            self.w_space     * s_space    +
            self.w_reverse   * s_reverse  +
            self.w_entry     * s_entry    +
            self.w_wait      * s_wait     +
            self.w_stability * s_stability+
            self.w_dist      * s_dist
        )
        self.ego_yield_score = float(score)
        self.get_logger().debug(
            f'[Score] space={s_space:.2f} rev={s_reverse:.2f} entry={s_entry:.2f} '
            f'wait={s_wait:.2f} stab={s_stability:.2f} dist={s_dist:.2f} → {score:.3f}'
        )
        return float(score)

    # ── 발행 헬퍼 ───────────────────────────────────────────────────────────
    def _publish_all(self, control_mode: str, led_cmd: str, decision: str = '') -> None:
        s = String(); s.data = self.state;   self.pub_state.publish(s)
        m = String(); m.data = control_mode; self.pub_mode.publish(m)
        l = String(); l.data = led_cmd;      self.pub_led.publish(l)
        if decision:
            d = String(); d.data = decision; self.pub_decision.publish(d)
        score_msg = Float32(); score_msg.data = self.ego_yield_score
        self.pub_ego_score.publish(score_msg)

    # ── 메인 스텝 ───────────────────────────────────────────────────────────
    def _step(self) -> None:
        # 최우선: safety_supervisor 강제 정지
        if self.safe_stop_flag and self.state != State.SAFE_STOP:
            self.transition(State.SAFE_STOP)

        # ── SAFE_STOP ──────────────────────────────────────────────────────
        if self.state == State.SAFE_STOP:
            self._publish_all('SAFE_STOP', 'RED_BLINK')
            if not self.safe_stop_flag:
                # 안전 해제 → 정상 복귀
                self.transition(State.NORMAL_CENTER_DRIVE)
            return

        # ── NORMAL_CENTER_DRIVE ────────────────────────────────────────────
        if self.state == State.NORMAL_CENTER_DRIVE:
            self._publish_all('NORMAL_CENTER_DRIVE', 'GREEN')
            if self._opponent_detected():
                self.transition(State.KEEP_RIGHT_APPROACH)
            return

        # ── KEEP_RIGHT_APPROACH ────────────────────────────────────────────
        if self.state == State.KEEP_RIGHT_APPROACH:
            self._publish_all('KEEP_RIGHT_APPROACH', 'YELLOW')
            if not self._opponent_detected():
                # 상대 사라짐 → 정상 복귀
                self.transition(State.NORMAL_CENTER_DRIVE)
                return
            if self._deadlock_condition():
                if self.deadlock_start_time is None:
                    self.deadlock_start_time = self.now_sec()
                elif (self.now_sec() - self.deadlock_start_time) >= self.deadlock_confirm_sec:
                    self.transition(State.DEADLOCK_DETECTED)
            else:
                self.deadlock_start_time = None
            return

        # ── DEADLOCK_DETECTED ──────────────────────────────────────────────
        if self.state == State.DEADLOCK_DETECTED:
            self._publish_all('KEEP_RIGHT_APPROACH', 'ORANGE_BLINK')
            # 즉시 협상으로 전이
            self.transition(State.RPS_NEGOTIATION)
            return

        # ── RPS_NEGOTIATION ────────────────────────────────────────────────
        if self.state == State.RPS_NEGOTIATION:
            self._publish_all('KEEP_RIGHT_APPROACH', 'BLUE_BLINK', 'negotiating')
            # 협상 요청 발행 (최초 1회)
            if not self._rps_request_sent:
                req = Bool(); req.data = True
                self.pub_neg_req.publish(req)
                self._rps_request_sent = True

            # 협상 완료 판단
            timeout_expired = self.time_in_state() >= self.rps_timeout_sec
            rps_failed = (
                self.rps_timed_out or
                timeout_expired or
                self.rps_result in ('none_response', 'error', 'refuse')
            )
            if rps_failed:
                self.get_logger().info('[RPS] 협상 실패 → 점수 기반 판단으로 전환')
                self.transition(State.SCORE_BASED_DECISION)
                return

            if self.rps_result == 'win':
                # 이겼으므로 진행 차량
                self.decision_result = 'proceed'
                self._publish_all('KEEP_RIGHT_APPROACH', 'GREEN_FLASH', 'proceed')
                self.transition(State.NORMAL_CENTER_DRIVE)
                return
            elif self.rps_result == 'lose':
                # 졌으므로 양보 차량
                self.decision_result = 'yield'
                self.transition(State.REVERSE_EXECUTE)
                return
            elif self.rps_result == 'draw':
                # 무승부 → 점수 기반
                self.transition(State.SCORE_BASED_DECISION)
                return
            return

        # ── SCORE_BASED_DECISION ───────────────────────────────────────────
        if self.state == State.SCORE_BASED_DECISION:
            my_score = self._compute_yield_score()
            self._publish_all('KEEP_RIGHT_APPROACH', 'PURPLE_BLINK', 'scoring')
            opponent_score = self.opponent_yield_score

            self.get_logger().info(
                f'[Score] 내 점수={my_score:.3f}, 상대 점수={opponent_score:.3f}'
            )
            if my_score >= opponent_score:
                self.decision_result = 'yield'
                self.get_logger().info('[Score] → 양보 결정 (REVERSE_EXECUTE)')
                self.transition(State.REVERSE_EXECUTE)
            else:
                self.decision_result = 'proceed'
                self.get_logger().info('[Score] → 진행 결정 (WAIT_PASS)')
                self.transition(State.WAIT_PASS)
            return

        # ── REVERSE_EXECUTE ────────────────────────────────────────────────
        if self.state == State.REVERSE_EXECUTE:
            self._publish_all('REVERSE_EXECUTE', 'RED', 'yield')
            if self.reverse_motion_done:
                self.transition(State.WAIT_PASS)
            return

        # ── WAIT_PASS ──────────────────────────────────────────────────────
        if self.state == State.WAIT_PASS:
            self.wait_accumulated_sec += 1.0 / self.decision_hz
            self._publish_all('WAIT_PASS', 'CYAN', self.decision_result)
            if self._opponent_cleared():
                self.transition(State.REENTER)
                return
            if self.time_in_state() >= self.wait_pass_timeout_sec:
                self.get_logger().warn('[WAIT_PASS] 대기 시간 초과 → SAFE_STOP')
                self.transition(State.SAFE_STOP)
            return

        # ── REENTER ────────────────────────────────────────────────────────
        if self.state == State.REENTER:
            self._publish_all('REENTER', 'GREEN_BLINK')
            if self.time_in_state() >= self.reenter_duration_sec:
                self.transition(State.NORMAL_CENTER_DRIVE)
            return


def main(args=None) -> None:
    rclpy.init(args=args)
    node = V2VDecisionNode()
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
