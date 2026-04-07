#!/usr/bin/env python3
"""
led_interface_node
───────────────────
외부 LED 상태 표시 노드.

/led_command 토픽을 수신해 LED 5개를 상태에 따라 제어한다.

LED 색상 코드 → 상태 의미:
  GREEN         : NORMAL_CENTER_DRIVE (정상 중앙 주행)
  YELLOW        : KEEP_RIGHT_APPROACH (우측통행 접근)
  ORANGE_BLINK  : DEADLOCK_DETECTED   (교착 감지)
  BLUE_BLINK    : RPS_NEGOTIATION     (가위바위보 협상 중)
  PURPLE_BLINK  : SCORE_BASED_DECISION (점수 기반 판단)
  RED           : REVERSE_EXECUTE     (후진 양보)
  CYAN          : WAIT_PASS           (통과 대기)
  GREEN_BLINK   : REENTER             (재진입)
  RED_BLINK     : SAFE_STOP           (비상 정지)
  GREEN_FLASH   : 협상 승리 / 진행 결정

점등 개수: led_score_ratio (0~1) 에 비례해 1~5개 표시 (점수 우위 표현)

하드웨어 백엔드:
  - Jetson GPIO (Jetson Orin Nano)
  - RPi.GPIO 호환 (Raspberry Pi)
  - 실물 없을 시 터미널 로그로 대체 (DRY_RUN 모드)

입력 토픽:
  /led_command      String
  /led_score        Float32  (자차 점수, 0~1 정규화)

출력 토픽:
  /led_status_feedback  String  (현재 LED 상태 피드백)
"""

import math
import time
import threading
from typing import List, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, String

# LED 핀 번호 (BCM 기준, Jetson Orin Nano GPIO 번호로 교체 가능)
_DEFAULT_PIN_MAP = [18, 23, 24, 25, 12]  # LED 1~5번 핀

# 색상 RGB 근사 (NeoPixel 미사용, 단색 LED이므로 on/off 제어)
# 색상별 점등 패턴 [LED1, LED2, LED3, LED4, LED5]
_COLOR_PATTERNS = {
    'GREEN':          [True,  True,  True,  True,  True ],
    'YELLOW':         [True,  True,  True,  False, False],
    'ORANGE_BLINK':   [True,  True,  True,  False, False],  # blink 처리
    'BLUE_BLINK':     [False, True,  True,  True,  False],
    'PURPLE_BLINK':   [True,  False, True,  False, True ],
    'RED':            [True,  False, False, False, False],
    'RED_BLINK':      [True,  False, False, False, False],
    'CYAN':           [False, True,  False, True,  False],
    'GREEN_BLINK':    [True,  True,  True,  True,  True ],
    'GREEN_FLASH':    [True,  True,  True,  True,  True ],
    'OFF':            [False, False, False, False, False],
}

_BLINK_COMMANDS = {
    'ORANGE_BLINK', 'BLUE_BLINK', 'PURPLE_BLINK', 'RED_BLINK',
    'GREEN_BLINK', 'GREEN_FLASH',
}


class LedInterfaceNode(Node):
    def __init__(self) -> None:
        super().__init__('led_interface_node')

        # ── 파라미터 ─────────────────────────────────────────────────────
        self.declare_parameter('led_hz', 10.0)
        self.declare_parameter('blink_period_sec', 0.4)  # blink 주기
        self.declare_parameter('dry_run', True)          # True=실물 없이 터미널 출력
        self.declare_parameter('gpio_backend', 'jetson') # 'jetson' / 'rpi'
        self.declare_parameter('pin_map', _DEFAULT_PIN_MAP)

        self.led_hz         = float(self.get_parameter('led_hz').value)
        self.blink_period   = float(self.get_parameter('blink_period_sec').value)
        self.dry_run        = bool(self.get_parameter('dry_run').value)
        self.gpio_backend   = str(self.get_parameter('gpio_backend').value)
        self.pin_map: List[int] = list(self.get_parameter('pin_map').value)

        # ── GPIO 초기화 ───────────────────────────────────────────────────
        self._gpio = None
        if not self.dry_run:
            self._init_gpio()

        # ── 내부 상태 ─────────────────────────────────────────────────────
        self.current_command: str = 'OFF'
        self.led_score: float = 0.0
        self._blink_state: bool = False
        self._blink_timer: float = 0.0
        self._last_pattern: List[bool] = [False] * 5

        # ── 구독 ─────────────────────────────────────────────────────────
        self.create_subscription(String,  '/led_command', self._cb_command, 10)
        self.create_subscription(Float32, '/led_score',   self._cb_score,  10)

        # ── 발행 ─────────────────────────────────────────────────────────
        self.pub_feedback = self.create_publisher(String, '/led_status_feedback', 10)

        dt = 1.0 / self.led_hz if self.led_hz > 0 else 0.1
        self.create_timer(dt, self._step)
        self.get_logger().info(
            f'led_interface_node 시작 (dry_run={self.dry_run}, backend={self.gpio_backend})'
        )

    # ── GPIO 초기화 ───────────────────────────────────────────────────────
    def _init_gpio(self) -> None:
        try:
            if self.gpio_backend == 'jetson':
                import Jetson.GPIO as GPIO   # type: ignore
            else:
                import RPi.GPIO as GPIO      # type: ignore
            GPIO.setmode(GPIO.BCM)
            for pin in self.pin_map:
                GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)
            self._gpio = GPIO
            self.get_logger().info(f'GPIO 초기화 완료: 핀={self.pin_map}')
        except ImportError as e:
            self.get_logger().warn(f'GPIO 라이브러리 없음 → dry_run 강제 활성: {e}')
            self.dry_run = True
        except Exception as e:
            self.get_logger().error(f'GPIO 초기화 실패 → dry_run 강제 활성: {e}')
            self.dry_run = True

    # ── 콜백 ─────────────────────────────────────────────────────────────
    def _cb_command(self, msg: String) -> None:
        self.current_command = str(msg.data).strip().upper()

    def _cb_score(self, msg: Float32) -> None:
        v = float(msg.data)
        if math.isfinite(v):
            self.led_score = max(0.0, min(1.0, v))

    # ── LED 출력 ──────────────────────────────────────────────────────────
    def _score_to_count(self) -> int:
        """점수(0~1)를 점등 LED 개수(1~5)로 변환."""
        return max(1, min(5, round(self.led_score * 4) + 1))

    def _apply_score_to_pattern(self, pattern: List[bool]) -> List[bool]:
        """점수 기반 판단 상태에서는 점등 개수로 점수 표현."""
        if self.current_command == 'SCORE_BASED_DECISION':
            count = self._score_to_count()
            return [i < count for i in range(5)]
        return pattern

    def _set_leds(self, states: List[bool]) -> None:
        if states == self._last_pattern:
            return
        self._last_pattern = list(states)

        if self.dry_run:
            icons = ['●' if s else '○' for s in states]
            self.get_logger().debug(f'[LED] {" ".join(icons)}  ({self.current_command})')
            return

        if self._gpio is None:
            return
        GPIO = self._gpio
        for pin, state in zip(self.pin_map, states):
            GPIO.output(pin, GPIO.HIGH if state else GPIO.LOW)

    def _all_off(self) -> None:
        self._set_leds([False] * 5)

    # ── blink 처리 ────────────────────────────────────────────────────────
    def _update_blink(self, dt: float) -> bool:
        """현재 blink 위상 반환. True=켜짐, False=꺼짐."""
        self._blink_timer += dt
        if self._blink_timer >= self.blink_period / 2.0:
            self._blink_timer = 0.0
            self._blink_state = not self._blink_state
        return self._blink_state

    # ── 메인 스텝 ─────────────────────────────────────────────────────────
    def _step(self) -> None:
        dt = 1.0 / self.led_hz if self.led_hz > 0 else 0.1
        cmd = self.current_command

        base_pattern = _COLOR_PATTERNS.get(cmd, _COLOR_PATTERNS['OFF'])
        base_pattern = self._apply_score_to_pattern(base_pattern)

        if cmd in _BLINK_COMMANDS:
            on = self._update_blink(dt)
            pattern = base_pattern if on else [False] * 5
        else:
            pattern = base_pattern

        self._set_leds(pattern)

        # 피드백 발행
        on_count = sum(pattern)
        fb = String()
        fb.data = f'{cmd} leds={on_count} score={self.led_score:.2f}'
        self.pub_feedback.publish(fb)

    # ── 소멸자 ───────────────────────────────────────────────────────────
    def destroy_node(self) -> None:
        self._all_off()
        if self._gpio is not None:
            try:
                self._gpio.cleanup()
            except Exception:
                pass
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LedInterfaceNode()
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
