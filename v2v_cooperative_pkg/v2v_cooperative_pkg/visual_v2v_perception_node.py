#!/usr/bin/env python3
"""
visual_v2v_perception_node
--------------------------
Camera based Visual V2V perception node.

The design documents describe a lightweight perception split:
  upper ROI  -> opponent vehicle / LED panel
  lower ROI  -> lane pipeline handled by lane_length_pkg

This node fills the missing upper-ROI side. It reads the LED strip as a
4-bit on/off pattern, not by color. While the peer vehicle is approaching,
all four LEDs stay on as a beacon. Once the beacon is close enough, the
decision layer can stop and switch the LEDs to state/score patterns.

Inputs:
  /image_raw                  sensor_msgs/Image

Outputs:
  /relative_distance          std_msgs/Float32
  /led_state                  std_msgs/String
  /led_pattern                std_msgs/String
  /opponent_yield_score       std_msgs/Float32
  /visual_v2v_status          std_msgs/String
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, String


@dataclass
class LedBlob:
    cx: float
    cy: float
    area: float
    bbox: Tuple[int, int, int, int]


SCORE_PATTERNS: Dict[str, float] = {
    '0100': 0.25,
    '1000': 0.50,
    '1100': 0.75,
    '1111': 1.00,
}


PATTERN_TO_STATE: Dict[str, str] = {
    '1111': 'vehicle_beacon',
    '0101': 'keep_right',
    '1010': 'deadlock',
    '0010': 'rps_request',
    '1001': 'yield',
    '0110': 'wait_pass',
    '0011': 'reenter',
    '0001': 'safe_stop',
    '1011': 'proceed',
    '1101': 'rps_rock',
    '0111': 'rps_paper',
    '1110': 'rps_scissors',
}


class VisualV2VPerceptionNode(Node):
    def __init__(self) -> None:
        super().__init__('visual_v2v_perception_node')

        self.declare_parameter('image_topic', '/image_raw')
        self.declare_parameter('upper_roi_ratio', 0.45)
        self.declare_parameter('min_blob_area_px', 30.0)
        self.declare_parameter('max_blob_area_px', 5000.0)
        self.declare_parameter('min_led_brightness', 130)
        self.declare_parameter('led_slot_count', 4)
        self.declare_parameter('panel_real_width_m', 0.12)
        self.declare_parameter('camera_fx_px', 700.0)
        self.declare_parameter('distance_min_m', 0.15)
        self.declare_parameter('distance_max_m', 3.0)
        self.declare_parameter('stale_timeout_sec', 0.7)
        self.declare_parameter('publish_debug_image', False)

        self.image_topic = str(self.get_parameter('image_topic').value)
        self.upper_roi_ratio = float(self.get_parameter('upper_roi_ratio').value)
        self.min_blob_area_px = float(self.get_parameter('min_blob_area_px').value)
        self.max_blob_area_px = float(self.get_parameter('max_blob_area_px').value)
        self.min_led_brightness = int(self.get_parameter('min_led_brightness').value)
        self.led_slot_count = max(1, int(self.get_parameter('led_slot_count').value))
        self.panel_real_width_m = float(self.get_parameter('panel_real_width_m').value)
        self.camera_fx_px = float(self.get_parameter('camera_fx_px').value)
        self.distance_min_m = float(self.get_parameter('distance_min_m').value)
        self.distance_max_m = float(self.get_parameter('distance_max_m').value)
        self.stale_timeout_sec = float(self.get_parameter('stale_timeout_sec').value)
        self.publish_debug_image = bool(self.get_parameter('publish_debug_image').value)

        self.bridge = CvBridge()
        self.last_detection_stamp: Optional[float] = None
        self.last_distance: Optional[float] = None
        self.last_state = 'unknown'
        self.last_pattern = '00000'
        self.last_score = 0.0
        self.panel_bounds: Optional[Tuple[float, float]] = None

        self.create_subscription(Image, self.image_topic, self._cb_image, 10)
        self.pub_distance = self.create_publisher(Float32, '/relative_distance', 10)
        self.pub_led_state = self.create_publisher(String, '/led_state', 10)
        self.pub_led_pattern = self.create_publisher(String, '/led_pattern', 10)
        self.pub_score = self.create_publisher(Float32, '/opponent_yield_score', 10)
        self.pub_status = self.create_publisher(String, '/visual_v2v_status', 10)
        self.pub_debug = self.create_publisher(Image, '/visual_v2v/debug_image', 1)

        self.create_timer(0.1, self._publish_status)
        self.get_logger().info('visual_v2v_perception_node started')

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _cb_image(self, msg: Image) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().warn(f'image conversion failed: {exc}')
            return

        h, w = frame.shape[:2]
        roi_h = max(1, int(h * self.upper_roi_ratio))
        upper = frame[:roi_h, :]

        blobs = self._detect_led_blobs(upper)
        panel = self._select_panel(blobs)
        if panel:
            distance = self._estimate_distance(panel)
            pattern = self._panel_to_pattern(panel)
            state, score = self._decode_pattern(pattern)
            self.last_detection_stamp = self._now()
            self.last_distance = distance
            self.last_state = state
            self.last_pattern = pattern
            self.last_score = score
            self._publish_detection(distance, state, score)

        if self.publish_debug_image:
            debug = self._draw_debug(frame, blobs, panel)
            self.pub_debug.publish(self.bridge.cv2_to_imgmsg(debug, encoding='bgr8'))

    def _detect_led_blobs(self, bgr: np.ndarray) -> List[LedBlob]:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, self.min_led_brightness, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        blobs: List[LedBlob] = []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.min_blob_area_px or area > self.max_blob_area_px:
                continue
            x, y, bw, bh = cv2.boundingRect(contour)
            if bw <= 0 or bh <= 0:
                continue
            aspect = bw / float(bh)
            if aspect < 0.25 or aspect > 4.0:
                continue
            blobs.append(LedBlob(x + bw / 2.0, y + bh / 2.0, area, (x, y, bw, bh)))
        return blobs

    def _select_panel(self, blobs: List[LedBlob]) -> List[LedBlob]:
        if not blobs:
            return []
        # A 4-LED strip appears as horizontally aligned bright blobs.
        blobs = sorted(blobs, key=lambda b: b.cx)
        best: List[LedBlob] = []
        for blob in blobs:
            cluster = [b for b in blobs if abs(b.cy - blob.cy) < 35.0]
            if len(cluster) > len(best):
                best = cluster
        return sorted(best[:self.led_slot_count], key=lambda b: b.cx)

    def _estimate_distance(self, panel: List[LedBlob]) -> float:
        if len(panel) < 2:
            return self.distance_max_m
        detected_min = min(b.bbox[0] for b in panel)
        detected_max = max(b.bbox[0] + b.bbox[2] for b in panel)
        if self.panel_bounds is not None and len(panel) < self.led_slot_count:
            min_x, max_x = self.panel_bounds
        else:
            min_x, max_x = float(detected_min), float(detected_max)
        panel_width_px = max(1.0, float(max_x - min_x))
        distance = self.panel_real_width_m * self.camera_fx_px / panel_width_px
        return max(self.distance_min_m, min(self.distance_max_m, distance))

    def _panel_to_pattern(self, panel: List[LedBlob]) -> str:
        if not panel:
            return '0' * self.led_slot_count

        detected_min = min(b.bbox[0] for b in panel)
        detected_max = max(b.bbox[0] + b.bbox[2] for b in panel)
        if len(panel) >= self.led_slot_count:
            self.panel_bounds = (float(detected_min), float(detected_max))

        if self.panel_bounds is not None and len(panel) < self.led_slot_count:
            min_x, max_x = self.panel_bounds
        else:
            min_x, max_x = float(detected_min), float(detected_max)

        width = max(1.0, float(max_x - min_x))
        slot_w = width / float(self.led_slot_count)
        slots = [False] * self.led_slot_count
        for blob in panel:
            idx = int((blob.cx - min_x) / slot_w)
            idx = max(0, min(self.led_slot_count - 1, idx))
            slots[idx] = True
        return ''.join('1' if on else '0' for on in slots)

    def _decode_pattern(self, pattern: str) -> Tuple[str, float]:
        state = PATTERN_TO_STATE.get(pattern, 'unknown')
        score = SCORE_PATTERNS.get(pattern, 0.0)
        if pattern in SCORE_PATTERNS and state == 'unknown':
            state = 'score_based'
        return state, score

    def _publish_detection(self, distance: float, state: str, score: float) -> None:
        d = Float32()
        d.data = float(distance)
        self.pub_distance.publish(d)

        s = String()
        s.data = state
        self.pub_led_state.publish(s)

        p = String()
        p.data = self.last_pattern
        self.pub_led_pattern.publish(p)

        sc = Float32()
        sc.data = float(score)
        self.pub_score.publish(sc)

    def _publish_status(self) -> None:
        fresh = (
            self.last_detection_stamp is not None and
            (self._now() - self.last_detection_stamp) <= self.stale_timeout_sec
        )
        msg = String()
        if fresh:
            msg.data = (
                f'ok state={self.last_state} pattern={self.last_pattern} distance={self.last_distance:.2f} '
                f'opponent_score={self.last_score:.2f}'
            )
        else:
            msg.data = 'stale'
        self.pub_status.publish(msg)

    def _draw_debug(
        self,
        frame: np.ndarray,
        blobs: List[LedBlob],
        panel: List[LedBlob],
    ) -> np.ndarray:
        debug = frame.copy()
        panel_ids = {id(b) for b in panel}
        for blob in blobs:
            x, y, w, h = blob.bbox
            color = (0, 255, 0) if id(blob) in panel_ids else (80, 80, 80)
            cv2.rectangle(debug, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                debug,
                'on',
                (x, max(12, y - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
                cv2.LINE_AA,
            )
        return debug


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VisualV2VPerceptionNode()
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
