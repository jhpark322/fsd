#!/usr/bin/env python3
import math
import os
from typing import Optional, Tuple

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, PoseStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, String


class LaneDetectionNode(Node):
    def __init__(self):
        super().__init__('lane_detection_node')

        # ROS / runtime params
        self.declare_parameter('image_topic', '/image_raw')
        self.declare_parameter('process_fps', 10.0)
        self.declare_parameter('show', True)
        self.declare_parameter('show_debug_view', True)
        self.declare_parameter('publish_debug_image', False)

        # Lane extraction params
        self.declare_parameter('lane_roi_top_ratio', 0.45)
        self.declare_parameter('blur_kernel', 5)
        self.declare_parameter('white_l_min', 135)
        self.declare_parameter('white_s_max', 200)
        self.declare_parameter('white_gray_min', 150)
        self.declare_parameter('yellow_h_min', 12)
        self.declare_parameter('yellow_h_max', 42)
        self.declare_parameter('yellow_s_min', 65)
        self.declare_parameter('yellow_v_min', 70)
        self.declare_parameter('min_component_area', 12)

        # Sliding window / fit params
        self.declare_parameter('nwindows', 6)
        self.declare_parameter('margin', 70)
        self.declare_parameter('minpix', 8)
        self.declare_parameter('min_lane_pixels', 40)
        self.declare_parameter('smooth_alpha', 0.08)
        self.declare_parameter('lane_full_refresh', 5)
        self.declare_parameter('max_lane_miss', 4)
        self.declare_parameter('right_offset_ratio', 0.30)
        self.declare_parameter('follow_mode', 'center')

        # Lane sanity check params (detection-warp pixel domain)
        self.declare_parameter('min_lane_width_px', 100.0)
        self.declare_parameter('max_lane_width_px', 900.0)
        self.declare_parameter('max_lane_width_variation_px', 260.0)
        self.declare_parameter('max_fit_slope_diff', 1.00)
        self.declare_parameter('enforce_image_side_sanity', False)
        self.declare_parameter('left_max_position_ratio', 0.70)
        self.declare_parameter('right_min_position_ratio', 0.30)

        # Perspective transform ratios for detection/debug view only
        self.declare_parameter('src_top_y_ratio', 0.54)
        self.declare_parameter('src_top_half_width_ratio', 0.34)
        self.declare_parameter('src_bottom_margin_ratio', 0.00)
        self.declare_parameter('dst_half_width_ratio', 0.42)

        # Metric homography
        self.declare_parameter('use_metric_homography', True)
        pkg_share = get_package_share_directory('lane_length_pkg')
        default_h = os.path.join(pkg_share, 'config', 'H.npy')
        default_hinv = os.path.join(pkg_share, 'config', 'Hinv.npy')
        self.declare_parameter('homography_path', default_h)
        self.declare_parameter('homography_inv_path', default_hinv)
        self.declare_parameter('bev_width_px', 120)
        self.declare_parameter('bev_height_px', 220)
        self.declare_parameter('px_per_m', 100.0)

        # Base-link sampling params
        self.declare_parameter('camera_to_base_m', 0.08)
        self.declare_parameter('bottom_visible_from_camera_m', 0.30)
        self.declare_parameter('stop_forward_m', 0.46)
        self.declare_parameter('steer_forward_m', 0.42)
        self.declare_parameter('heading_lookahead_m', 0.10)
        self.declare_parameter('max_forward_extrapolation_m', 0.10)
        self.declare_parameter('curve_sample_count', 160)
        self.declare_parameter('lateral_zero_bias_m', 0.0)
        self.declare_parameter('preview_path_step_m', 0.04)
        self.declare_parameter('preview_path_max_forward_m', 1.20)
        self.declare_parameter('status_requires_current_detection', True)

        self.image_topic = str(self.get_parameter('image_topic').value)
        self.process_fps = float(self.get_parameter('process_fps').value)
        self.show = bool(self.get_parameter('show').value)
        self.show_debug_view = bool(self.get_parameter('show_debug_view').value)
        self.publish_debug_image = bool(self.get_parameter('publish_debug_image').value)

        self.lane_roi_top_ratio = float(self.get_parameter('lane_roi_top_ratio').value)
        self.blur_kernel = int(self.get_parameter('blur_kernel').value)
        if self.blur_kernel % 2 == 0:
            self.blur_kernel += 1

        self.white_l_min = int(self.get_parameter('white_l_min').value)
        self.white_s_max = int(self.get_parameter('white_s_max').value)
        self.white_gray_min = int(self.get_parameter('white_gray_min').value)
        self.yellow_h_min = int(self.get_parameter('yellow_h_min').value)
        self.yellow_h_max = int(self.get_parameter('yellow_h_max').value)
        self.yellow_s_min = int(self.get_parameter('yellow_s_min').value)
        self.yellow_v_min = int(self.get_parameter('yellow_v_min').value)
        self.min_component_area = int(self.get_parameter('min_component_area').value)

        self.nwindows = int(self.get_parameter('nwindows').value)
        self.margin = int(self.get_parameter('margin').value)
        self.minpix = int(self.get_parameter('minpix').value)
        self.min_lane_pixels = int(self.get_parameter('min_lane_pixels').value)
        self.smooth_alpha = float(self.get_parameter('smooth_alpha').value)
        self.lane_full_refresh = max(1, int(self.get_parameter('lane_full_refresh').value))
        self.max_lane_miss = int(self.get_parameter('max_lane_miss').value)
        self.right_offset_ratio = float(self.get_parameter('right_offset_ratio').value)
        self.follow_mode = str(self.get_parameter('follow_mode').value).strip().lower()
        if self.follow_mode not in ('center', 'right'):
            self.follow_mode = 'center'

        self.min_lane_width_px = float(self.get_parameter('min_lane_width_px').value)
        self.max_lane_width_px = float(self.get_parameter('max_lane_width_px').value)
        self.max_lane_width_variation_px = float(self.get_parameter('max_lane_width_variation_px').value)
        self.max_fit_slope_diff = float(self.get_parameter('max_fit_slope_diff').value)
        self.enforce_image_side_sanity = bool(self.get_parameter('enforce_image_side_sanity').value)
        self.left_max_position_ratio = float(self.get_parameter('left_max_position_ratio').value)
        self.right_min_position_ratio = float(self.get_parameter('right_min_position_ratio').value)

        self.src_top_y_ratio = float(self.get_parameter('src_top_y_ratio').value)
        self.src_top_half_width_ratio = float(self.get_parameter('src_top_half_width_ratio').value)
        self.src_bottom_margin_ratio = float(self.get_parameter('src_bottom_margin_ratio').value)
        self.dst_half_width_ratio = float(self.get_parameter('dst_half_width_ratio').value)

        self.use_metric_homography = bool(self.get_parameter('use_metric_homography').value)
        self.homography_path = str(self.get_parameter('homography_path').value)
        self.homography_inv_path = str(self.get_parameter('homography_inv_path').value)
        self.bev_width_px = int(self.get_parameter('bev_width_px').value)
        self.bev_height_px = int(self.get_parameter('bev_height_px').value)
        self.px_per_m = float(self.get_parameter('px_per_m').value)

        self.camera_to_base_m = float(self.get_parameter('camera_to_base_m').value)
        self.bottom_visible_from_camera_m = float(self.get_parameter('bottom_visible_from_camera_m').value)
        self.visible_bottom_base_m = self.camera_to_base_m + self.bottom_visible_from_camera_m
        self.stop_forward_m = float(self.get_parameter('stop_forward_m').value)
        self.steer_forward_m = float(self.get_parameter('steer_forward_m').value)
        self.heading_lookahead_m = float(self.get_parameter('heading_lookahead_m').value)
        self.max_forward_extrapolation_m = float(self.get_parameter('max_forward_extrapolation_m').value)
        self.curve_sample_count = max(40, int(self.get_parameter('curve_sample_count').value))
        self.lateral_zero_bias_m = float(self.get_parameter('lateral_zero_bias_m').value)
        self.preview_path_step_m = float(self.get_parameter('preview_path_step_m').value)
        self.preview_path_max_forward_m = float(self.get_parameter('preview_path_max_forward_m').value)
        self.status_requires_current_detection = bool(self.get_parameter('status_requires_current_detection').value)
        self.forward_query_margin_m = 0.02

        self._sanitize_forward_queries()

        self.bridge = CvBridge()
        self.latest_frame: Optional[np.ndarray] = None
        self.width: Optional[int] = None
        self.height: Optional[int] = None

        self.M_detect = None
        self.Minv_detect = None
        self.M_metric = None
        self.Minv_metric = None

        self.prev_left_fit: Optional[np.ndarray] = None
        self.prev_right_fit: Optional[np.ndarray] = None
        self.left_miss_count = 0
        self.right_miss_count = 0
        self.frame_index = 0
        self.current_detection_valid = False

        self.center_error_pub = self.create_publisher(Float32, '/lane_error_center', 10)
        self.right_error_pub = self.create_publisher(Float32, '/lane_error_right', 10)
        self.follow_error_pub = self.create_publisher(Float32, '/lane_follow_error_px', 10)
        self.heading_error_pub = self.create_publisher(Float32, '/lane_heading_error', 10)
        self.lane_width_pub = self.create_publisher(Float32, '/lane_width', 10)
        self.target_x_pub = self.create_publisher(Float32, '/lane_target_x_px', 10)
        self.target_y_pub = self.create_publisher(Float32, '/lane_target_y_px', 10)
        self.image_center_x_pub = self.create_publisher(Float32, '/lane_image_center_x_px', 10)
        self.status_pub = self.create_publisher(String, '/lane_status', 10)

        self.lane_width_m_pub = self.create_publisher(Float32, '/lane_width_m', 10)
        self.lane_width_steer_m_pub = self.create_publisher(Float32, '/lane_width_steer_m', 10)
        self.center_error_m_pub = self.create_publisher(Float32, '/lane_error_center_m', 10)
        self.right_error_m_pub = self.create_publisher(Float32, '/lane_error_right_m', 10)
        self.follow_error_m_pub = self.create_publisher(Float32, '/lane_follow_error_m', 10)
        self.stop_forward_m_pub = self.create_publisher(Float32, '/lane_stop_forward_m', 10)
        self.steer_forward_m_pub = self.create_publisher(Float32, '/lane_steer_forward_m', 10)
        self.center_base_pt_pub = self.create_publisher(PointStamped, '/lane_center_base_pt', 10)
        self.centerline_base_path_pub = self.create_publisher(Path, '/lane_centerline_base_path', 10)
        self.left_boundary_base_path_pub = self.create_publisher(Path, '/lane_left_boundary_base_path', 10)
        self.right_boundary_base_path_pub = self.create_publisher(Path, '/lane_right_boundary_base_path', 10)

        if self.publish_debug_image:
            self.debug_image_pub = self.create_publisher(Image, '/lane_debug_image', 10)
            self.binary_image_pub = self.create_publisher(Image, '/lane_binary_image', 10)
            self.warped_image_pub = self.create_publisher(Image, '/lane_warped_image', 10)
        else:
            self.debug_image_pub = None
            self.binary_image_pub = None
            self.warped_image_pub = None

        self.create_subscription(Image, self.image_topic, self.image_callback, qos_profile_sensor_data)
        timer_period = 1.0 / self.process_fps if self.process_fps > 0.0 else 0.1
        self.timer = self.create_timer(timer_period, self.process_frame)

        self.get_logger().info(
            f'lane_detection_node started | topic={self.image_topic} fps={self.process_fps:.1f} '
            f'follow_mode={self.follow_mode} metric={self.use_metric_homography} '
            f'steer_x={self.steer_forward_m:.2f} vis_bottom={self.visible_bottom_base_m:.2f}'
        )

    def _sanitize_forward_queries(self):
        min_valid_forward_m = self.visible_bottom_base_m + self.forward_query_margin_m
        if self.steer_forward_m < min_valid_forward_m:
            self.steer_forward_m = min_valid_forward_m
        if self.stop_forward_m < min_valid_forward_m:
            self.stop_forward_m = min_valid_forward_m
        if self.stop_forward_m < self.steer_forward_m:
            self.stop_forward_m = self.steer_forward_m

    def metric_lateral_to_base_lateral(self, lateral_metric_m):
        return np.asarray(lateral_metric_m, dtype=np.float32) - float(self.lateral_zero_bias_m)

    def base_lateral_to_metric_lateral(self, lateral_base_m):
        return np.asarray(lateral_base_m, dtype=np.float32) + float(self.lateral_zero_bias_m)

    def image_callback(self, msg: Image):
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge failed: {e}')

    def init_frame_resources(self, frame: np.ndarray):
        self.height, self.width = frame.shape[:2]
        self.M_detect, self.Minv_detect = self.get_perspective_matrices(self.width, self.height)
        if self.use_metric_homography:
            self.M_metric = np.load(self.homography_path).astype(np.float32)
            self.Minv_metric = np.load(self.homography_inv_path).astype(np.float32)
        else:
            self.M_metric = None
            self.Minv_metric = None

    def get_perspective_matrices(self, width: int, height: int):
        src = np.float32([
            [0.5 - self.src_top_half_width_ratio, self.src_top_y_ratio],
            [0.5 + self.src_top_half_width_ratio, self.src_top_y_ratio],
            [1.0 - self.src_bottom_margin_ratio, 0.95],
            [self.src_bottom_margin_ratio, 0.95],
        ])
        dst = np.float32([
            [0.5 - self.dst_half_width_ratio, 0.00],
            [0.5 + self.dst_half_width_ratio, 0.00],
            [0.5 + self.dst_half_width_ratio, 1.00],
            [0.5 - self.dst_half_width_ratio, 1.00],
        ])
        src[:, 0] *= width
        src[:, 1] *= height
        dst[:, 0] *= width
        dst[:, 1] *= height
        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)
        return M, Minv

    def remove_small_components(self, binary: np.ndarray) -> np.ndarray:
        if self.min_component_area <= 0:
            return binary
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        filtered = np.zeros_like(binary)
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area >= self.min_component_area:
                filtered[labels == label] = 255
        return filtered

    def threshold_lane(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        roi_y = int(h * self.lane_roi_top_ratio)
        roi = frame[roi_y:, :]

        blur = cv2.GaussianBlur(roi, (self.blur_kernel, self.blur_kernel), 0)
        hls = cv2.cvtColor(blur, cv2.COLOR_BGR2HLS)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

        white_mask_hls = cv2.inRange(
            hls,
            np.array([0, self.white_l_min, 0], dtype=np.uint8),
            np.array([255, 255, self.white_s_max], dtype=np.uint8),
        )
        white_mask_gray = cv2.inRange(gray, self.white_gray_min, 255)
        white_mask = cv2.bitwise_and(white_mask_hls, white_mask_gray)

        yellow_mask = cv2.inRange(
            hsv,
            np.array([self.yellow_h_min, self.yellow_s_min, self.yellow_v_min], dtype=np.uint8),
            np.array([self.yellow_h_max, 255, 255], dtype=np.uint8),
        )
        color_mask = cv2.bitwise_or(white_mask, yellow_mask)

        mask = np.zeros_like(color_mask)
        rh, rw = color_mask.shape[:2]
        poly = np.array([[
            (int(rw * 0.00), rh - 1),
            (int(rw * 0.05), int(rh * 0.03)),
            (int(rw * 0.95), int(rh * 0.03)),
            (int(rw * 1.00), rh - 1),
        ]], dtype=np.int32)
        cv2.fillPoly(mask, poly, 255)
        color_mask = cv2.bitwise_and(color_mask, mask)

        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 11))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary_roi = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel_close)
        binary_roi = cv2.morphologyEx(binary_roi, cv2.MORPH_OPEN, kernel_open)
        binary_roi = self.remove_small_components(binary_roi)

        binary = np.zeros((h, w), dtype=np.uint8)
        binary[roi_y:, :] = binary_roi
        return binary

    def warp_binary(self, binary: np.ndarray) -> np.ndarray:
        return cv2.warpPerspective(binary, self.M_detect, (self.width, self.height), flags=cv2.INTER_NEAREST)

    def find_lane_base(self, binary_warped: np.ndarray) -> Tuple[int, int]:
        y_start = int(binary_warped.shape[0] * 0.55)
        histogram = np.sum(binary_warped[y_start:, :], axis=0)
        midpoint = histogram.shape[0] // 2
        leftx_base = int(np.argmax(histogram[:midpoint]))
        rightx_base = int(np.argmax(histogram[midpoint:]) + midpoint)
        return leftx_base, rightx_base

    def sliding_window_search(self, binary_warped: np.ndarray):
        leftx_base, rightx_base = self.find_lane_base(binary_warped)
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        window_height = int(binary_warped.shape[0] / self.nwindows)
        leftx_current = leftx_base
        rightx_current = rightx_base
        left_lane_inds = []
        right_lane_inds = []

        for window in range(self.nwindows):
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - self.margin
            win_xleft_high = leftx_current + self.margin
            win_xright_low = rightx_current - self.margin
            win_xright_high = rightx_current + self.margin

            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            if len(good_left_inds) > self.minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > self.minpix:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))

        left_lane_inds = np.concatenate(left_lane_inds) if len(left_lane_inds) > 0 else np.array([], dtype=np.int64)
        right_lane_inds = np.concatenate(right_lane_inds) if len(right_lane_inds) > 0 else np.array([], dtype=np.int64)

        leftx = nonzerox[left_lane_inds] if len(left_lane_inds) > 0 else np.array([])
        lefty = nonzeroy[left_lane_inds] if len(left_lane_inds) > 0 else np.array([])
        rightx = nonzerox[right_lane_inds] if len(right_lane_inds) > 0 else np.array([])
        righty = nonzeroy[right_lane_inds] if len(right_lane_inds) > 0 else np.array([])
        return leftx, lefty, rightx, righty

    def search_around_poly(self, binary_warped: np.ndarray, fit: Optional[np.ndarray]):
        if fit is None:
            return np.array([]), np.array([])
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        lane_x = fit[0] * (nonzeroy ** 2) + fit[1] * nonzeroy + fit[2]
        lane_inds = ((nonzerox > (lane_x - self.margin)) & (nonzerox < (lane_x + self.margin)))
        return nonzerox[lane_inds], nonzeroy[lane_inds]

    def fit_quadratic(self, x: np.ndarray, y: np.ndarray) -> Optional[np.ndarray]:
        if x is None or y is None or len(x) < self.min_lane_pixels or len(y) < self.min_lane_pixels:
            return None
        try:
            return np.polyfit(y, x, 2)
        except Exception:
            return None

    def smooth_fit(self, prev_fit: Optional[np.ndarray], new_fit: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if new_fit is None:
            return prev_fit
        if prev_fit is None:
            return new_fit
        return self.smooth_alpha * new_fit + (1.0 - self.smooth_alpha) * prev_fit

    def x_at_y(self, fit: Optional[np.ndarray], y: float) -> Optional[float]:
        if fit is None:
            return None
        return float(fit[0] * y * y + fit[1] * y + fit[2])

    def validate_lane_pair(self, left_fit: Optional[np.ndarray], right_fit: Optional[np.ndarray]) -> bool:
        if left_fit is None or right_fit is None:
            return False

        y_samples = np.array([self.height * 0.60, self.height * 0.75, self.height * 0.90], dtype=np.float32)

        widths = []
        for y in y_samples:
            lx = self.x_at_y(left_fit, float(y))
            rx = self.x_at_y(right_fit, float(y))
            if lx is None or rx is None or rx <= lx:
                return False
            widths.append(rx - lx)

        widths = np.array(widths, dtype=np.float32)
        if np.any(widths < self.min_lane_width_px) or np.any(widths > self.max_lane_width_px):
            return False
        if float(np.max(widths) - np.min(widths)) > self.max_lane_width_variation_px:
            return False
        if abs(float(left_fit[1] - right_fit[1])) > self.max_fit_slope_diff:
            return False

        y_check = float(self.height * 0.90)
        lx = self.x_at_y(left_fit, y_check)
        rx = self.x_at_y(right_fit, y_check)
        if lx is None or rx is None:
            return False

        if self.enforce_image_side_sanity:
            if lx > self.width * self.left_max_position_ratio:
                return False
            if rx < self.width * self.right_min_position_ratio:
                return False
        return True

    def detect_warped_points_to_image(self, pts_xy: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts_xy, dtype=np.float32).reshape(-1, 1, 2)
        img_pts = cv2.perspectiveTransform(pts, self.Minv_detect).reshape(-1, 2)
        return img_pts

    def image_points_to_metric_bev(self, pts_xy: np.ndarray) -> Optional[np.ndarray]:
        if not self.use_metric_homography or self.M_metric is None:
            return None
        pts = np.asarray(pts_xy, dtype=np.float32).reshape(-1, 1, 2)
        bev_pts = cv2.perspectiveTransform(pts, self.M_metric).reshape(-1, 2)
        return bev_pts

    def relative_metric_error_from_image_point(self, img_pt_xy: np.ndarray) -> Optional[float]:
        if self.M_detect is None or self.Minv_detect is None or self.M_metric is None:
            return None

        img_pt = np.asarray(img_pt_xy, dtype=np.float32).reshape(1, 1, 2)
        warped_pt = cv2.perspectiveTransform(img_pt, self.M_detect).reshape(2)
        ego_warped_pt = np.array([self.width * 0.5, warped_pt[1]], dtype=np.float32).reshape(1, 1, 2)
        ego_img_pt = cv2.perspectiveTransform(ego_warped_pt, self.Minv_detect).reshape(2)

        bev_pts = self.image_points_to_metric_bev(np.stack([img_pt.reshape(2), ego_img_pt], axis=0))
        if bev_pts is None or bev_pts.shape[0] != 2:
            return None

        target_bev, ego_bev = bev_pts
        return float((target_bev[0] - ego_bev[0]) / self.px_per_m)

    def base_points_to_image(self, lateral_m: np.ndarray, forward_m: np.ndarray) -> Optional[np.ndarray]:
        if self.Minv_metric is None:
            return None
        lateral_m = np.asarray(lateral_m, dtype=np.float32)
        forward_m = np.asarray(forward_m, dtype=np.float32)
        metric_lateral_m = self.base_lateral_to_metric_lateral(lateral_m)
        bev_x = (self.bev_width_px * 0.5) + metric_lateral_m * self.px_per_m
        cam_forward_m = forward_m - self.camera_to_base_m
        bev_y = (self.bev_height_px - 1.0) - (cam_forward_m - self.bottom_visible_from_camera_m) * self.px_per_m
        bev_pts = np.stack([bev_x, bev_y], axis=1).astype(np.float32).reshape(-1, 1, 2)
        img_pts = cv2.perspectiveTransform(bev_pts, self.Minv_metric).reshape(-1, 2)
        return img_pts

    def fit_curve_to_base(self, fit: Optional[np.ndarray]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if fit is None or self.M_metric is None:
            return None

        ploty = np.linspace(0.0, float(self.height - 1), self.curve_sample_count)
        plotx = fit[0] * ploty ** 2 + fit[1] * ploty + fit[2]
        warped_pts = np.stack([plotx, ploty], axis=1).astype(np.float32)

        img_pts = self.detect_warped_points_to_image(warped_pts)
        bev_pts = self.image_points_to_metric_bev(img_pts)
        if bev_pts is None or bev_pts.shape[0] < 4:
            return None

        lateral_metric_m = (bev_pts[:, 0] - (self.bev_width_px * 0.5)) / self.px_per_m
        lateral_base_m = self.metric_lateral_to_base_lateral(lateral_metric_m)
        cam_forward_m = self.bottom_visible_from_camera_m + (((self.bev_height_px - 1.0) - bev_pts[:, 1]) / self.px_per_m)
        base_forward_m = cam_forward_m + self.camera_to_base_m

        finite = np.isfinite(base_forward_m) & np.isfinite(lateral_base_m)
        base_forward_m = base_forward_m[finite]
        lateral_base_m = lateral_base_m[finite]
        if base_forward_m.size < 4:
            return None

        order = np.argsort(base_forward_m)
        base_forward_m = base_forward_m[order]
        lateral_base_m = lateral_base_m[order]

        _, unique_idx = np.unique(np.round(base_forward_m, 4), return_index=True)
        base_forward_m = base_forward_m[unique_idx]
        lateral_base_m = lateral_base_m[unique_idx]
        if base_forward_m.size < 4:
            return None
        return base_forward_m, lateral_base_m

    def sample_lateral_at_forward(self, curve: Optional[Tuple[np.ndarray, np.ndarray]], forward_query_m: float) -> Optional[float]:
        if curve is None:
            return None
        x, y = curve
        if x.size < 4:
            return None

        x_min = float(x[0])
        x_max = float(x[-1])

        if x_min <= forward_query_m <= x_max:
            return float(np.interp(forward_query_m, x, y))

        if forward_query_m < x_min:
            if (x_min - forward_query_m) > self.max_forward_extrapolation_m:
                return None
            k = min(8, x.size)
            coeff = np.polyfit(x[:k], y[:k], 1)
            return float(np.polyval(coeff, forward_query_m))

        if forward_query_m > x_max:
            if (forward_query_m - x_max) > self.max_forward_extrapolation_m:
                return None
            k = min(8, x.size)
            coeff = np.polyfit(x[-k:], y[-k:], 1)
            return float(np.polyval(coeff, forward_query_m))

        return None

    def sample_lane_base(self):
        left_curve = self.fit_curve_to_base(self.prev_left_fit)
        right_curve = self.fit_curve_to_base(self.prev_right_fit)

        if left_curve is None or right_curve is None:
            return {
                'lane_ok': False,
                'lane_width_stop_m': 0.0,
                'lane_width_steer_m': 0.0,
                'center_error_m': 0.0,
                'right_error_m': 0.0,
                'follow_error_m': 0.0,
                'heading_error': float('nan'),
                'target_forward_m': self.steer_forward_m,
                'center_lateral_m': None,
                'right_target_lateral_m': None,
                'left_stop_m': None,
                'right_stop_m': None,
                'center_img_pt': None,
                'right_target_img_pt': None,
                'target_img_pt': None,
                'left_curve': left_curve,
                'right_curve': right_curve,
            }

        left_stop = self.sample_lateral_at_forward(left_curve, self.stop_forward_m)
        right_stop = self.sample_lateral_at_forward(right_curve, self.stop_forward_m)
        left_steer = self.sample_lateral_at_forward(left_curve, self.steer_forward_m)
        right_steer = self.sample_lateral_at_forward(right_curve, self.steer_forward_m)

        lane_width_stop_m = 0.0
        lane_width_steer_m = 0.0
        if None not in (left_stop, right_stop) and right_stop > left_stop:
            lane_width_stop_m = float(right_stop - left_stop)
        if None in (left_steer, right_steer) or right_steer <= left_steer:
            return {
                'lane_ok': False,
                'lane_width_stop_m': lane_width_stop_m,
                'lane_width_steer_m': 0.0,
                'center_error_m': 0.0,
                'right_error_m': 0.0,
                'follow_error_m': 0.0,
                'heading_error': float('nan'),
                'target_forward_m': self.steer_forward_m,
                'center_lateral_m': None,
                'right_target_lateral_m': None,
                'left_stop_m': left_stop,
                'right_stop_m': right_stop,
                'center_img_pt': None,
                'right_target_img_pt': None,
                'target_img_pt': None,
                'left_curve': left_curve,
                'right_curve': right_curve,
            }

        lane_width_steer_m = float(right_steer - left_steer)
        center_lateral_m = float(0.5 * (left_steer + right_steer))
        right_target_lateral_m = float(center_lateral_m + self.right_offset_ratio * lane_width_steer_m)

        img_pts = self.base_points_to_image(
            np.array([center_lateral_m, right_target_lateral_m], dtype=np.float32),
            np.array([self.steer_forward_m, self.steer_forward_m], dtype=np.float32),
        )
        if img_pts is None or img_pts.shape[0] != 2:
            return {
                'lane_ok': False,
                'lane_width_stop_m': lane_width_stop_m,
                'lane_width_steer_m': lane_width_steer_m,
                'center_error_m': 0.0,
                'right_error_m': 0.0,
                'follow_error_m': 0.0,
                'heading_error': float('nan'),
                'target_forward_m': self.steer_forward_m,
                'center_lateral_m': center_lateral_m,
                'right_target_lateral_m': right_target_lateral_m,
                'left_stop_m': left_stop,
                'right_stop_m': right_stop,
                'center_img_pt': None,
                'right_target_img_pt': None,
                'target_img_pt': None,
                'left_curve': left_curve,
                'right_curve': right_curve,
            }

        center_img_pt = img_pts[0]
        right_target_img_pt = img_pts[1]
        if self.follow_mode == 'right':
            target_img_pt = right_target_img_pt
        else:
            target_img_pt = center_img_pt

        center_error_m = self.relative_metric_error_from_image_point(center_img_pt) or 0.0
        right_error_m = self.relative_metric_error_from_image_point(right_target_img_pt) or 0.0
        follow_error_m = right_error_m if self.follow_mode == 'right' else center_error_m

        heading_x0 = self.steer_forward_m
        heading_x1 = self.steer_forward_m + self.heading_lookahead_m
        left_h0 = self.sample_lateral_at_forward(left_curve, heading_x0)
        right_h0 = self.sample_lateral_at_forward(right_curve, heading_x0)
        left_h1 = self.sample_lateral_at_forward(left_curve, heading_x1)
        right_h1 = self.sample_lateral_at_forward(right_curve, heading_x1)
        heading_error = float('nan')
        if None not in (left_h0, right_h0, left_h1, right_h1):
            center_h0 = 0.5 * (left_h0 + right_h0)
            center_h1 = 0.5 * (left_h1 + right_h1)
            heading_error = float(math.atan2(center_h1 - center_h0, self.heading_lookahead_m))

        return {
            'lane_ok': True,
            'lane_width_stop_m': lane_width_stop_m,
            'lane_width_steer_m': lane_width_steer_m,
            'center_error_m': float(center_error_m),
            'right_error_m': float(right_error_m),
            'follow_error_m': float(follow_error_m),
            'heading_error': heading_error,
            'target_forward_m': self.steer_forward_m,
            'center_lateral_m': center_lateral_m,
            'right_target_lateral_m': right_target_lateral_m,
            'left_stop_m': left_stop,
            'right_stop_m': right_stop,
            'center_img_pt': center_img_pt,
            'right_target_img_pt': right_target_img_pt,
            'target_img_pt': target_img_pt,
            'left_curve': left_curve,
            'right_curve': right_curve,
        }

    def update_lane_fits(self, binary_warped: np.ndarray) -> bool:
        use_full_search = (
            (self.prev_left_fit is None) or
            (self.prev_right_fit is None) or
            (self.frame_index % self.lane_full_refresh == 0)
        )

        if use_full_search:
            leftx, lefty, rightx, righty = self.sliding_window_search(binary_warped)
        else:
            leftx, lefty = self.search_around_poly(binary_warped, self.prev_left_fit)
            rightx, righty = self.search_around_poly(binary_warped, self.prev_right_fit)

        left_fit_raw = self.fit_quadratic(leftx, lefty)
        right_fit_raw = self.fit_quadratic(rightx, righty)
        valid_now = self.validate_lane_pair(left_fit_raw, right_fit_raw)

        if valid_now:
            self.prev_left_fit = self.smooth_fit(self.prev_left_fit, left_fit_raw)
            self.prev_right_fit = self.smooth_fit(self.prev_right_fit, right_fit_raw)
            self.left_miss_count = 0
            self.right_miss_count = 0
        else:
            self.left_miss_count += 1
            self.right_miss_count += 1
            if self.left_miss_count > self.max_lane_miss:
                self.prev_left_fit = None
            if self.right_miss_count > self.max_lane_miss:
                self.prev_right_fit = None

        self.current_detection_valid = valid_now
        return valid_now

    def publish_scalar(self, pub, value: float):
        msg = Float32()
        msg.data = float(value)
        pub.publish(msg)

    def publish_center_base_point(self, center_lateral_m: Optional[float], lane_ok: bool):
        if (not lane_ok) or center_lateral_m is None:
            return
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'lane_base'
        msg.point.x = float(self.steer_forward_m)
        msg.point.y = float(center_lateral_m)
        msg.point.z = 0.0
        self.center_base_pt_pub.publish(msg)

    def build_base_path_msg(self, pts_xy: np.ndarray, stamp=None, frame_id: str = 'lane_base') -> Path:
        path = Path()
        path.header.stamp = stamp if stamp is not None else self.get_clock().now().to_msg()
        path.header.frame_id = frame_id
        pts_xy = np.asarray(pts_xy, dtype=np.float32).reshape(-1, 2)
        for x_m, y_right_m in pts_xy:
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x = float(x_m)
            ps.pose.position.y = float(y_right_m)
            ps.pose.position.z = 0.0
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)
        return path

    def publish_visible_base_paths(self, lane: dict):
        left_curve = lane.get('left_curve')
        right_curve = lane.get('right_curve')
        if left_curve is None or right_curve is None:
            return

        lx, ly = left_curve
        rx, ry = right_curve
        if lx.size < 4 or rx.size < 4:
            return

        x_min = max(float(lx[0]), float(rx[0]), self.visible_bottom_base_m + self.forward_query_margin_m)
        x_max = min(float(lx[-1]), float(rx[-1]), self.preview_path_max_forward_m)
        if x_max <= x_min:
            return

        xs = np.arange(x_min, x_max + 1e-6, max(0.02, self.preview_path_step_m), dtype=np.float32)
        left_pts = []
        right_pts = []
        center_pts = []

        # lateral_base_m is BEV-center-relative, but lane errors are ego-center-relative.
        # Compute the constant lateral offset of ego in the metric BEV so path y values
        # match the same reference frame as the published error signals.
        center_lateral_m = lane.get('center_lateral_m')
        center_error_m = lane.get('center_error_m')
        if center_lateral_m is not None and center_error_m is not None:
            ego_lateral_offset_m = float(center_lateral_m) - float(center_error_m)
        else:
            ego_lateral_offset_m = 0.0

        for x in xs:
            l = self.sample_lateral_at_forward(left_curve, float(x))
            r = self.sample_lateral_at_forward(right_curve, float(x))
            if l is None or r is None or r <= l:
                continue
            c = 0.5 * (l + r)
            left_pts.append((float(x), float(l) - ego_lateral_offset_m))
            right_pts.append((float(x), float(r) - ego_lateral_offset_m))
            center_pts.append((float(x), float(c) - ego_lateral_offset_m))

        if len(center_pts) < 2:
            return

        stamp = self.get_clock().now().to_msg()
        self.centerline_base_path_pub.publish(self.build_base_path_msg(np.asarray(center_pts, dtype=np.float32), stamp))
        self.left_boundary_base_path_pub.publish(self.build_base_path_msg(np.asarray(left_pts, dtype=np.float32), stamp))
        self.right_boundary_base_path_pub.publish(self.build_base_path_msg(np.asarray(right_pts, dtype=np.float32), stamp))

    def draw_lane_overlay(self, frame: np.ndarray, left_fit: Optional[np.ndarray], right_fit: Optional[np.ndarray]):
        out = frame.copy()
        if left_fit is None or right_fit is None:
            return out

        ploty = np.linspace(0, self.height - 1, self.height)
        leftx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        rightx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        leftx = np.clip(leftx, 0, self.width - 1)
        rightx = np.clip(rightx, 0, self.width - 1)

        left_pts_warped = np.vstack([leftx, ploty]).T.astype(np.float32)
        right_pts_warped = np.vstack([rightx, ploty]).T.astype(np.float32)

        lane_poly_warped = np.vstack([left_pts_warped, right_pts_warped[::-1]]).astype(np.int32)
        overlay_warped = np.zeros_like(frame)
        cv2.fillPoly(overlay_warped, [lane_poly_warped], (0, 150, 0))
        overlay_img = cv2.warpPerspective(overlay_warped, self.Minv_detect, (self.width, self.height), flags=cv2.INTER_LINEAR)

        out = cv2.addWeighted(out, 1.0, overlay_img, 0.35, 0)

        left_pts_img = self.detect_warped_points_to_image(left_pts_warped)
        right_pts_img = self.detect_warped_points_to_image(right_pts_warped)
        cv2.polylines(out, [np.round(left_pts_img).astype(np.int32).reshape(-1, 1, 2)], False, (255, 0, 0), 3)
        cv2.polylines(out, [np.round(right_pts_img).astype(np.int32).reshape(-1, 1, 2)], False, (0, 0, 255), 3)
        return out

    def build_debug_view(self, original: np.ndarray, result: np.ndarray, binary: np.ndarray, warped: np.ndarray):
        binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        warped_bgr = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
        top = np.hstack([original, result])
        bottom = np.hstack([binary_bgr, warped_bgr])
        return np.vstack([top, bottom])

    def process_frame(self):
        if self.latest_frame is None:
            return

        frame = self.latest_frame.copy()
        if self.width is None or self.height is None:
            self.init_frame_resources(frame)

        self.frame_index += 1
        binary = self.threshold_lane(frame)
        warped = self.warp_binary(binary)
        self.update_lane_fits(warped)
        lane = self.sample_lane_base()

        current_lane_ok = bool(lane['lane_ok']) and (self.current_detection_valid or (not self.status_requires_current_detection))
        result = self.draw_lane_overlay(frame, self.prev_left_fit, self.prev_right_fit)

        target_img_x = 0.0
        target_img_y = 0.0
        center_error_px = 0.0
        right_error_px = 0.0
        follow_error_px = 0.0
        lane_width_px = 0.0

        if current_lane_ok and lane['target_img_pt'] is not None:
            image_center_x = self.width * 0.5
            center_pt_img = np.round(lane['center_img_pt']).astype(np.int32) if lane['center_img_pt'] is not None else None
            right_pt_img = np.round(lane['right_target_img_pt']).astype(np.int32) if lane['right_target_img_pt'] is not None else None
            target_pt_img = np.round(lane['target_img_pt']).astype(np.int32)

            if center_pt_img is not None:
                center_error_px = float(center_pt_img[0] - image_center_x)
                cv2.circle(result, tuple(center_pt_img), 6, (255, 255, 0), -1)
            if right_pt_img is not None:
                right_error_px = float(right_pt_img[0] - image_center_x)
                cv2.circle(result, tuple(right_pt_img), 6, (255, 255, 255), -1)

            follow_error_px = float(target_pt_img[0] - image_center_x)
            target_img_x = float(target_pt_img[0])
            target_img_y = float(target_pt_img[1])

            cv2.circle(result, tuple(target_pt_img), 8, (0, 255, 255), -1)
            ego_bottom_img = (int(round(image_center_x)), self.height - 1)
            cv2.line(result, ego_bottom_img, tuple(target_pt_img), (0, 255, 255), 2)

        lane_width_stop_m = float(lane['lane_width_stop_m']) if current_lane_ok else 0.0
        lane_width_steer_m = float(lane['lane_width_steer_m']) if current_lane_ok else 0.0
        center_error_m = float(lane['center_error_m']) if current_lane_ok else 0.0
        right_error_m = float(lane['right_error_m']) if current_lane_ok else 0.0
        follow_error_m = float(lane['follow_error_m']) if current_lane_ok else 0.0
        heading_error = float(lane['heading_error']) if current_lane_ok and math.isfinite(float(lane['heading_error'])) else float('nan')

        if current_lane_ok and None not in (lane['left_stop_m'], lane['right_stop_m']):
            stop_img = self.base_points_to_image(
                np.array([lane['left_stop_m'], lane['right_stop_m']], dtype=np.float32),
                np.array([self.stop_forward_m, self.stop_forward_m], dtype=np.float32),
            )
            if stop_img is not None and stop_img.shape[0] == 2:
                left_stop_img = np.round(stop_img[0]).astype(np.int32)
                right_stop_img = np.round(stop_img[1]).astype(np.int32)
                lane_width_px = float(abs(right_stop_img[0] - left_stop_img[0]))
                cv2.line(result, tuple(left_stop_img), tuple(right_stop_img), (255, 0, 255), 2)

        self.publish_scalar(self.center_error_pub, center_error_px)
        self.publish_scalar(self.right_error_pub, right_error_px)
        self.publish_scalar(self.follow_error_pub, follow_error_px)
        self.publish_scalar(self.heading_error_pub, heading_error)
        self.publish_scalar(self.lane_width_pub, lane_width_px)
        self.publish_scalar(self.target_x_pub, target_img_x)
        self.publish_scalar(self.target_y_pub, target_img_y)
        self.publish_scalar(self.image_center_x_pub, float(self.width * 0.5))

        self.publish_scalar(self.lane_width_m_pub, lane_width_stop_m)
        self.publish_scalar(self.lane_width_steer_m_pub, lane_width_steer_m)
        self.publish_scalar(self.center_error_m_pub, center_error_m)
        self.publish_scalar(self.right_error_m_pub, right_error_m)
        self.publish_scalar(self.follow_error_m_pub, follow_error_m)
        self.publish_scalar(self.stop_forward_m_pub, self.stop_forward_m)
        self.publish_scalar(self.steer_forward_m_pub, self.steer_forward_m)
        self.publish_center_base_point(lane.get('center_lateral_m'), current_lane_ok)

        if current_lane_ok:
            self.publish_visible_base_paths(lane)

        status = String()
        status.data = 'ok' if current_lane_ok else 'lost'
        self.status_pub.publish(status)

        if self.publish_debug_image:
            self.debug_image_pub.publish(self.bridge.cv2_to_imgmsg(result, encoding='bgr8'))
            self.binary_image_pub.publish(self.bridge.cv2_to_imgmsg(binary, encoding='mono8'))
            self.warped_image_pub.publish(self.bridge.cv2_to_imgmsg(warped, encoding='mono8'))

        if self.show:
            if self.show_debug_view:
                debug_view = self.build_debug_view(frame, result, binary, warped)
                cv2.imshow('lane_debug_view', debug_view)
            else:
                cv2.imshow('lane_detection_result', result)
            cv2.waitKey(1)

    def destroy_node(self):
        try:
            if self.show:
                cv2.destroyAllWindows()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LaneDetectionNode()
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
