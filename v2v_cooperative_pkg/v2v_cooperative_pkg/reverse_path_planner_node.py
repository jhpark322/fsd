#!/usr/bin/env python3
"""
reverse_path_planner_node
──────────────────────────
A* 기반 후진 회피 경로 생성 노드.

현재 위치에서 reverse_goal까지 점유 격자 지도(OccupancyGrid) 위에서
A* 탐색으로 전역 경로를 생성한다.
경로 방향은 후진이므로 waypoint 순서가 역방향이 된다.

입력 토픽:
  /map            nav_msgs/OccupancyGrid  (slam_toolbox)
  /odom           nav_msgs/Odometry
  /reverse_goal   geometry_msgs/PoseStamped  (spatial_memory_node)
  /vehicle_state  std_msgs/String

출력 토픽:
  /reverse_path   nav_msgs/Path
  /reverse_path_ready Bool
"""

import heapq
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy.node import Node
from std_msgs.msg import Bool, String


def quat_to_yaw(q) -> float:
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


# ── A* 구현 ───────────────────────────────────────────────────────────────
def _heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def astar(
    grid: np.ndarray,          # (rows, cols), True=장애물
    start: Tuple[int, int],
    goal: Tuple[int, int],
    robot_radius_cells: int = 2,
) -> Optional[List[Tuple[int, int]]]:
    """
    A* 경로 탐색. 반환값: 셀 좌표 리스트(start→goal) 또는 None.
    robot_radius_cells: 로봇 크기 고려한 팽창 반경.
    """
    rows, cols = grid.shape

    # 장애물 팽창 (Minkowski sum 근사)
    from scipy.ndimage import binary_dilation  # type: ignore
    structure = np.ones((2 * robot_radius_cells + 1,) * 2, dtype=bool)
    inflated = binary_dilation(grid, structure=structure)

    def is_free(r: int, c: int) -> bool:
        return 0 <= r < rows and 0 <= c < cols and not inflated[r, c]

    if not is_free(*start) or not is_free(*goal):
        return None

    open_heap: List[Tuple[float, int, int]] = []
    heapq.heappush(open_heap, (0.0, start[0], start[1]))

    came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
    g_score: Dict[Tuple[int, int], float] = {start: 0.0}

    neighbors_8 = [(-1, 0), (1, 0), (0, -1), (0, 1),
                   (-1, -1), (-1, 1), (1, -1), (1, 1)]

    while open_heap:
        _, r, c = heapq.heappop(open_heap)
        cur = (r, c)
        if cur == goal:
            # 경로 재구성
            path: List[Tuple[int, int]] = []
            node: Optional[Tuple[int, int]] = goal
            while node is not None:
                path.append(node)
                node = came_from[node]
            path.reverse()
            return path

        for dr, dc in neighbors_8:
            nr, nc = r + dr, c + dc
            nb = (nr, nc)
            if not is_free(nr, nc):
                continue
            move_cost = math.hypot(dr, dc)
            tentative_g = g_score[cur] + move_cost
            if nb not in g_score or tentative_g < g_score[nb]:
                g_score[nb] = tentative_g
                f = tentative_g + _heuristic(nb, goal)
                heapq.heappush(open_heap, (f, nr, nc))
                came_from[nb] = cur

    return None  # 경로 없음


class ReversePathPlannerNode(Node):
    def __init__(self) -> None:
        super().__init__('reverse_path_planner_node')

        self.declare_parameter('planner_hz',         2.0)
        self.declare_parameter('robot_width_m',      0.19)
        self.declare_parameter('path_simplify_dist', 0.1)  # 경로 단순화 간격 (m)
        self.declare_parameter('occ_threshold',      50)   # 점유 격자 점유 임계값

        self.planner_hz        = float(self.get_parameter('planner_hz').value)
        self.robot_width_m     = float(self.get_parameter('robot_width_m').value)
        self.path_simplify_dist = float(self.get_parameter('path_simplify_dist').value)
        self.occ_threshold     = int(self.get_parameter('occ_threshold').value)

        # 내부 상태
        self.map_msg:     Optional[OccupancyGrid] = None
        self.odom_msg:    Optional[Odometry]      = None
        self.goal_msg:    Optional[PoseStamped]   = None
        self.vehicle_state: str = 'NORMAL_CENTER_DRIVE'

        self.last_path:   Optional[Path] = None
        self.path_ready:  bool = False

        # 구독
        self.create_subscription(OccupancyGrid, '/map',           self._cb_map,   1)
        self.create_subscription(Odometry,      '/odom',          self._cb_odom,  10)
        self.create_subscription(PoseStamped,   '/reverse_goal',  self._cb_goal,  10)
        self.create_subscription(String,        '/vehicle_state', self._cb_state, 10)

        # 발행
        self.pub_path  = self.create_publisher(Path, '/reverse_path',       1)
        self.pub_ready = self.create_publisher(Bool, '/reverse_path_ready', 10)

        dt = 1.0 / self.planner_hz if self.planner_hz > 0 else 0.5
        self.create_timer(dt, self._step)
        self.get_logger().info('reverse_path_planner_node 시작')

    # ── 콜백 ─────────────────────────────────────────────────────────────
    def _cb_map(self, msg: OccupancyGrid) -> None:
        self.map_msg = msg

    def _cb_odom(self, msg: Odometry) -> None:
        self.odom_msg = msg

    def _cb_goal(self, msg: PoseStamped) -> None:
        self.goal_msg = msg

    def _cb_state(self, msg: String) -> None:
        prev = self.vehicle_state
        self.vehicle_state = str(msg.data).strip()
        if prev != 'REVERSE_EXECUTE' and self.vehicle_state == 'REVERSE_EXECUTE':
            # REVERSE_EXECUTE 진입 시 경로 재계산
            self._plan()

    # ── 좌표 변환 ─────────────────────────────────────────────────────────
    def _world_to_cell(self, wx: float, wy: float) -> Optional[Tuple[int, int]]:
        if self.map_msg is None:
            return None
        info = self.map_msg.info
        ox = info.origin.position.x
        oy = info.origin.position.y
        res = info.resolution
        col = int((wx - ox) / res)
        row = int((wy - oy) / res)
        if 0 <= row < info.height and 0 <= col < info.width:
            return row, col
        return None

    def _cell_to_world(self, row: int, col: int) -> Tuple[float, float]:
        info = self.map_msg.info
        ox = info.origin.position.x
        oy = info.origin.position.y
        res = info.resolution
        wx = ox + (col + 0.5) * res
        wy = oy + (row + 0.5) * res
        return wx, wy

    def _build_grid(self) -> Optional[np.ndarray]:
        if self.map_msg is None:
            return None
        info = self.map_msg.info
        data = np.array(self.map_msg.data, dtype=np.int8).reshape(info.height, info.width)
        # 점유(>=threshold) 또는 미지(-1)를 장애물로 처리
        grid = (data >= self.occ_threshold) | (data < 0)
        return grid

    # ── 경로 단순화 ──────────────────────────────────────────────────────
    def _simplify_path(self, pts: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        if len(pts) < 2:
            return pts
        result = [pts[0]]
        for p in pts[1:]:
            if math.hypot(p[0] - result[-1][0], p[1] - result[-1][1]) >= self.path_simplify_dist:
                result.append(p)
        if result[-1] != pts[-1]:
            result.append(pts[-1])
        return result

    # ── A* 경로 계획 ─────────────────────────────────────────────────────
    def _plan(self) -> None:
        if self.map_msg is None or self.odom_msg is None or self.goal_msg is None:
            self.path_ready = False
            return

        # 시작 = 현재 위치
        sx = float(self.odom_msg.pose.pose.position.x)
        sy = float(self.odom_msg.pose.pose.position.y)
        # 목표 = reverse_goal
        gx = float(self.goal_msg.pose.position.x)
        gy = float(self.goal_msg.pose.position.y)

        start_cell = self._world_to_cell(sx, sy)
        goal_cell  = self._world_to_cell(gx, gy)

        if start_cell is None or goal_cell is None:
            self.get_logger().warn('[Planner] start/goal 셀 변환 실패')
            self.path_ready = False
            return

        grid = self._build_grid()
        if grid is None:
            self.path_ready = False
            return

        res = self.map_msg.info.resolution
        robot_radius_cells = max(1, int(math.ceil(self.robot_width_m / 2.0 / res)))

        self.get_logger().info(
            f'[Planner] A* 탐색: ({start_cell}) → ({goal_cell})'
        )
        try:
            cell_path = astar(grid, start_cell, goal_cell, robot_radius_cells)
        except Exception as e:
            self.get_logger().error(f'[Planner] A* 오류: {e}')
            self.path_ready = False
            return

        if cell_path is None:
            self.get_logger().warn('[Planner] 경로 없음')
            self.path_ready = False
            return

        # 셀 → 월드 좌표
        world_pts = [self._cell_to_world(r, c) for r, c in cell_path]
        world_pts = self._simplify_path(world_pts)

        # nav_msgs/Path 생성
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = 'map'
        for wx, wy in world_pts:
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x = wx
            ps.pose.position.y = wy
            ps.pose.position.z = 0.0
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)

        self.last_path = path
        self.path_ready = True
        self.get_logger().info(f'[Planner] 경로 생성 완료: {len(world_pts)} 웨이포인트')

    # ── 메인 스텝 ─────────────────────────────────────────────────────────
    def _step(self) -> None:
        # REVERSE_EXECUTE 상태에서만 경로 발행 (매 스텝 재계획은 하지 않음)
        ready_msg = Bool(); ready_msg.data = self.path_ready
        self.pub_ready.publish(ready_msg)

        if self.path_ready and self.last_path is not None:
            self.last_path.header.stamp = self.get_clock().now().to_msg()
            self.pub_path.publish(self.last_path)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ReversePathPlannerNode()
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
