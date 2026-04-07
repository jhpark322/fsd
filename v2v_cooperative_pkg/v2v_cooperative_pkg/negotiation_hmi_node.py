#!/usr/bin/env python3
"""
negotiation_hmi_node
─────────────────────
가위바위보(Rock-Paper-Scissors) 기반 협상 HMI 노드.

동작 흐름:
  1. /negotiation_request = True 수신 시 협상 개시
  2. 키보드 입력 or 버튼으로 내 선택 수신 (r=가위, p=바위, s=보)
  3. 상대방 선택은 /led_state 토픽으로 읽어옴 ('rps_rock'/'rps_paper'/'rps_scissors')
  4. 결과 판정 후 /rps_result 발행 ('win'/'lose'/'draw'/'none_response'/'error'/'refuse')
  5. input_timeout_sec 내 입력 없으면 /rps_timeout = True

입력 토픽:
  /negotiation_request   Bool
  /led_state             String   (상대 차량 LED 상태 판독 결과)

출력 토픽:
  /rps_result            String
  /rps_timeout           Bool

선택 인터페이스:
  - stdin 키보드 입력 (비블로킹, 별도 스레드)
  - 향후 GPIO 버튼 또는 GUI로 교체 가능
"""

import select
import sys
import threading
from typing import Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String

# 가위바위보 판정 테이블: (내 선택, 상대 선택) → 결과
# r=가위, p=바위, s=보
_WIN_TABLE = {
    ('r', 'r'): 'draw',
    ('r', 'p'): 'lose',   # 가위 vs 바위 → 패
    ('r', 's'): 'win',    # 가위 vs 보   → 승
    ('p', 'r'): 'win',    # 바위 vs 가위 → 승
    ('p', 'p'): 'draw',
    ('p', 's'): 'lose',   # 바위 vs 보   → 패
    ('s', 'r'): 'lose',   # 보   vs 가위 → 패
    ('s', 'p'): 'win',    # 보   vs 바위 → 승
    ('s', 's'): 'draw',
}

_LED_TO_RPS = {
    'rps_rock':     'p',  # 바위
    'rps_paper':    's',  # 보
    'rps_scissors': 'r',  # 가위
}


class NegotiationHmiNode(Node):
    def __init__(self) -> None:
        super().__init__('negotiation_hmi_node')

        self.declare_parameter('input_timeout_sec', 8.0)   # 입력 대기 시간
        self.declare_parameter('result_hold_sec',   2.0)   # 결과 발행 유지 시간
        self.declare_parameter('hmi_hz',            10.0)

        self.input_timeout_sec = float(self.get_parameter('input_timeout_sec').value)
        self.result_hold_sec   = float(self.get_parameter('result_hold_sec').value)
        self.hmi_hz            = float(self.get_parameter('hmi_hz').value)

        # 상태
        self.active: bool = False
        self.negotiation_start: Optional[float] = None
        self.my_choice: Optional[str] = None        # 'r' / 'p' / 's'
        self.opponent_choice: Optional[str] = None  # 'r' / 'p' / 's'
        self.result_published: bool = False
        self.result_stamp: Optional[float] = None

        self.led_state: str = 'unknown'

        # 키보드 입력 큐 (스레드 안전)
        self._input_queue: list = []
        self._input_lock = threading.Lock()

        # 구독
        self.create_subscription(Bool,   '/negotiation_request', self._cb_request, 10)
        self.create_subscription(String, '/led_state',           self._cb_led,     10)

        # 발행
        self.pub_result  = self.create_publisher(String, '/rps_result',  10)
        self.pub_timeout = self.create_publisher(Bool,   '/rps_timeout', 10)

        # 키보드 스레드 시작
        self._kb_thread = threading.Thread(target=self._keyboard_reader, daemon=True)
        self._kb_thread.start()

        dt = 1.0 / self.hmi_hz if self.hmi_hz > 0 else 0.1
        self.create_timer(dt, self._step)
        self.get_logger().info('negotiation_hmi_node 시작. 협상 시 키보드: r=가위 p=바위 s=보 x=거부')

    # ── 콜백 ─────────────────────────────────────────────────────────────
    def _cb_request(self, msg: Bool) -> None:
        if bool(msg.data) and not self.active:
            self._start_negotiation()

    def _cb_led(self, msg: String) -> None:
        self.led_state = str(msg.data).strip().lower()

    # ── 협상 개시/종료 ────────────────────────────────────────────────────
    def _start_negotiation(self) -> None:
        self.active = True
        self.negotiation_start = self._now()
        self.my_choice = None
        self.opponent_choice = None
        self.result_published = False
        self.result_stamp = None
        self.get_logger().info(
            '[RPS] 협상 개시. 입력: r=가위, p=바위, s=보, x=거부 '
            f'({self.input_timeout_sec:.1f}초 이내)'
        )

    def _reset(self) -> None:
        self.active = False
        self.negotiation_start = None
        self.my_choice = None
        self.opponent_choice = None

    # ── 키보드 입력 스레드 ────────────────────────────────────────────────
    def _keyboard_reader(self) -> None:
        """stdin에서 비블로킹으로 한 글자씩 읽어 큐에 넣는다."""
        while True:
            try:
                # select로 stdin 대기 (0.1초 타임아웃)
                ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                if ready:
                    ch = sys.stdin.read(1).strip().lower()
                    if ch in ('r', 'p', 's', 'x'):
                        with self._input_lock:
                            self._input_queue.append(ch)
            except Exception:
                break

    def _get_input(self) -> Optional[str]:
        with self._input_lock:
            if self._input_queue:
                return self._input_queue.pop(0)
        return None

    # ── 상대방 선택 읽기 ─────────────────────────────────────────────────
    def _read_opponent_choice(self) -> Optional[str]:
        return _LED_TO_RPS.get(self.led_state, None)

    # ── 결과 판정 ────────────────────────────────────────────────────────
    def _judge(self, my: str, opp: str) -> str:
        return _WIN_TABLE.get((my, opp), 'error')

    def _publish_result(self, result: str) -> None:
        msg = String(); msg.data = result
        self.pub_result.publish(msg)
        self.get_logger().info(f'[RPS] 결과 발행: {result}')

    def _publish_timeout(self, val: bool) -> None:
        msg = Bool(); msg.data = val
        self.pub_timeout.publish(msg)

    # ── 메인 스텝 ─────────────────────────────────────────────────────────
    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _step(self) -> None:
        if not self.active:
            return

        elapsed = self._now() - (self.negotiation_start or self._now())

        # 결과 발행 후 hold 시간이 지나면 리셋
        if self.result_published and self.result_stamp is not None:
            if self._now() - self.result_stamp >= self.result_hold_sec:
                self._reset()
                return

        if self.result_published:
            return

        # 키보드 입력 확인
        ch = self._get_input()
        if ch == 'x':
            self._publish_result('refuse')
            self._publish_timeout(False)
            self.result_published = True
            self.result_stamp = self._now()
            return

        if ch in ('r', 'p', 's'):
            self.my_choice = ch
            self.get_logger().info(f'[RPS] 내 선택: {ch}')

        # 상대방 선택 읽기
        opp = self._read_opponent_choice()
        if opp is not None:
            self.opponent_choice = opp

        # 양쪽 선택이 모두 확인된 경우 판정
        if self.my_choice is not None and self.opponent_choice is not None:
            result = self._judge(self.my_choice, self.opponent_choice)
            self.get_logger().info(
                f'[RPS] 내={self.my_choice} 상대={self.opponent_choice} → {result}'
            )
            self._publish_result(result)
            self._publish_timeout(False)
            self.result_published = True
            self.result_stamp = self._now()
            return

        # 타임아웃 처리
        if elapsed >= self.input_timeout_sec:
            reason = 'none_response'
            if self.my_choice is not None and self.opponent_choice is None:
                reason = 'none_response'  # 상대 무응답
            elif self.my_choice is None:
                reason = 'none_response'  # 내 입력 없음
            self.get_logger().warn(f'[RPS] 타임아웃 ({elapsed:.1f}s) → {reason}')
            self._publish_result(reason)
            self._publish_timeout(True)
            self.result_published = True
            self.result_stamp = self._now()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = NegotiationHmiNode()
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
