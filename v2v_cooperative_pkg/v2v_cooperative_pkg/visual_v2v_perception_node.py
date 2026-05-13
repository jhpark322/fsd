#!/usr/bin/env python3
"""
visual_v2v_perception_node
--------------------------
Camera based Visual V2V perception node.

The design documents describe a lightweight perception split:
  upper ROI  -> opponent vehicle / LED panel
  lower ROI  -> lane pipeline handled by lane_length_pkg

This node fills the missing upper-ROI side. It keeps the implementation
classical and inspectable so it can run before a YOLO model is available:
it detects bright LED blobs by HSV color masks, groups them as a panel,
estimates the opponent distance from panel size, and publishes the decoded
state plus score ratio for the decision node.

Inputs:
  /image_raw                  sensor_msgs/Image

Outputs:
  /relative_distance          std_msgs/Float32
  /led_state                  std_msgs/String
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
    color: str
    bbox: Tuple[int, int, int, int]


class VisualV2VPerceptionNode(Node):
    def __init__(self) -> None:
        super().__init__('visual_v2v_perception_node')

        self.declare_parameter('image_topic', '/image_raw')
        self.declare_parameter('upper_roi_ratio', 0.45)
        self.declare_parameter('min_blob_area_px', 30.0)
        self.declare_parameter('max_blob_area_px', 5000.0)
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
        self.last_score = 0.0

        self.create_subscription(Image, self.image_topic, self._cb_image, 10)
        self.pub_distance = self.create_publisher(Float32, '/relative_distance', 10)
        self.pub_led_state = self.create_publisher(String, '/led_state', 10)
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
            state, score = self._decode_panel(panel)
            self.last_detection_stamp = self._now()
            self.last_distance = distance
            self.last_state = state
            self.last_score = score
            self._publish_detection(distance, state, score)

        if self.publish_debug_image:
            debug = self._draw_debug(frame, blobs, panel)
            self.pub_debug.publish(self.bridge.cv2_to_imgmsg(debug, encoding='bgr8'))

    def _detect_led_blobs(self, bgr: np.ndarray) -> List[LedBlob]:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        masks: Dict[str, np.ndarray] = {
            'red': cv2.bitwise_or(
                cv2.inRange(hsv, np.array([0, 80, 100]), np.array([10, 255, 255])),
                cv2.inRange(hsv, np.array([170, 80, 100]), np.array([180, 255, 255])),
            ),
            'green': cv2.inRange(hsv, np.array([40, 60, 80]), np.array([85, 255, 255])),
            'blue': cv2.inRange(hsv, np.array([95, 60, 80]), np.array([135, 255, 255])),
            'yellow': cv2.inRange(hsv, np.array([18, 80, 100]), np.array([38, 255, 255])),
            'cyan': cv2.inRange(hsv, np.array([80, 60, 80]), np.array([100, 255, 255])),
            'purple': cv2.inRange(hsv, np.array([135, 50, 80]), np.array([165, 255, 255])),
        }

        kernel = np.ones((3, 3), np.uint8)
        blobs: List[LedBlob] = []
        for color, mask in masks.items():
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
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
                blobs.append(LedBlob(x + bw / 2.0, y + bh / 2.0, area, color, (x, y, bw, bh)))
        return blobs

    def _select_panel(self, blobs: List[LedBlob]) -> List[LedBlob]:
        if not blobs:
            return []
        # A 5-LED strip appears as horizontally aligned bright blobs.
        blobs = sorted(blobs, key=lambda b: b.cx)
        best: List[LedBlob] = []
        for blob in blobs:
            cluster = [b for b in blobs if abs(b.cy - blob.cy) < 35.0]
            if len(cluster) > len(best):
                best = cluster
        return sorted(best[:5], key=lambda b: b.cx)

    def _estimate_distance(self, panel: List[LedBlob]) -> float:
        if len(panel) < 2:
            return self.distance_max_m
        min_x = min(b.bbox[0] for b in panel)
        max_x = max(b.bbox[0] + b.bbox[2] for b in panel)
        panel_width_px = max(1.0, float(max_x - min_x))
        distance = self.panel_real_width_m * self.camera_fx_px / panel_width_px
        return max(self.distance_min_m, min(self.distance_max_m, distance))

    def _decode_panel(self, panel: List[LedBlob]) -> Tuple[str, float]:
        if not panel:
            return 'unknown', 0.0
        counts: Dict[str, int] = {}
        for blob in panel:
            counts[blob.color] = counts.get(blob.color, 0) + 1
        dominant = max(counts.items(), key=lambda item: item[1])[0]
        score = max(0.0, min(1.0, (len(panel) - 1) / 4.0))
        state_map = {
            'green': 'proceed',
            'yellow': 'keep_right',
            'red': 'yield',
            'blue': 'rps_request',
            'purple': 'score_based',
            'cyan': 'wait_pass',
        }
        return state_map.get(dominant, dominant), score

    def _publish_detection(self, distance: float, state: str, score: float) -> None:
        d = Float32()
        d.data = float(distance)
        self.pub_distance.publish(d)

        s = String()
        s.data = state
        self.pub_led_state.publish(s)

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
                f'ok state={self.last_state} distance={self.last_distance:.2f} '
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
                blob.color,
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
