/**
 * reverse_path_planner_node (C++)
 * ─────────────────────────────────
 * A* 기반 후진 회피 경로 생성.
 *
 * 점유 격자 지도(OccupancyGrid) 위에서 현재 위치 → reverse_goal 까지
 * A* 탐색으로 경로를 생성한다.
 */

#include <cmath>
#include <vector>
#include <queue>
#include <unordered_map>
#include <algorithm>
#include <string>
#include <functional>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/bool.hpp"
#include "std_msgs/msg/string.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "nav_msgs/msg/path.hpp"

// ── A* 구현 ─────────────────────────────────────────────────────────────
struct Cell {
    int r, c;
    bool operator==(const Cell &o) const { return r == o.r && c == o.c; }
};

struct CellHash {
    size_t operator()(const Cell &c) const {
        return std::hash<int>()(c.r) ^ (std::hash<int>()(c.c) << 16);
    }
};

static double heuristic(Cell a, Cell b) {
    return std::hypot(a.r - b.r, a.c - b.c);
}

struct AStarEntry {
    double f;
    Cell cell;
    bool operator>(const AStarEntry &o) const { return f > o.f; }
};

static std::vector<Cell> astar(
    const std::vector<bool> &occupied,
    int rows, int cols,
    Cell start, Cell goal,
    int robot_radius_cells)
{
    // ── 장애물 팽창 ─────────────────────────────────────────────────
    std::vector<bool> inflated(rows * cols, false);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (!occupied[r * cols + c]) continue;
            for (int dr = -robot_radius_cells; dr <= robot_radius_cells; ++dr) {
                for (int dc = -robot_radius_cells; dc <= robot_radius_cells; ++dc) {
                    int nr = r + dr, nc = c + dc;
                    if (nr >= 0 && nr < rows && nc >= 0 && nc < cols)
                        inflated[nr * cols + nc] = true;
                }
            }
        }
    }

    auto is_free = [&](int r, int c) -> bool {
        return r >= 0 && r < rows && c >= 0 && c < cols && !inflated[r * cols + c];
    };

    if (!is_free(start.r, start.c) || !is_free(goal.r, goal.c))
        return {};

    std::priority_queue<AStarEntry, std::vector<AStarEntry>, std::greater<>> open;
    std::unordered_map<Cell, Cell, CellHash> came_from;
    std::unordered_map<Cell, double, CellHash> g_score;

    open.push({0.0, start});
    g_score[start] = 0.0;
    came_from[start] = {-1, -1};  // sentinel

    static const int dx[] = {-1, 1, 0, 0, -1, -1, 1, 1};
    static const int dy[] = {0, 0, -1, 1, -1, 1, -1, 1};

    while (!open.empty()) {
        auto cur = open.top().cell;
        open.pop();

        if (cur == goal) {
            std::vector<Cell> path;
            Cell node = goal;
            while (!(node.r == -1 && node.c == -1)) {
                path.push_back(node);
                node = came_from[node];
            }
            std::reverse(path.begin(), path.end());
            return path;
        }

        for (int i = 0; i < 8; ++i) {
            Cell nb{cur.r + dx[i], cur.c + dy[i]};
            if (!is_free(nb.r, nb.c)) continue;
            double move_cost = std::hypot(dx[i], dy[i]);
            double tentative_g = g_score[cur] + move_cost;
            auto it = g_score.find(nb);
            if (it == g_score.end() || tentative_g < it->second) {
                g_score[nb] = tentative_g;
                double f = tentative_g + heuristic(nb, goal);
                open.push({f, nb});
                came_from[nb] = cur;
            }
        }
    }
    return {};  // 경로 없음
}


// ── 노드 ────────────────────────────────────────────────────────────────
static double quat_to_yaw(const geometry_msgs::msg::Quaternion &q) {
    double siny = 2.0 * (q.w * q.z + q.x * q.y);
    double cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
    return std::atan2(siny, cosy);
}

class ReversePathPlannerNode : public rclcpp::Node {
public:
    ReversePathPlannerNode() : Node("reverse_path_planner_node") {
        this->declare_parameter("planner_hz",         2.0);
        this->declare_parameter("robot_width_m",      0.19);
        this->declare_parameter("path_simplify_dist", 0.1);
        this->declare_parameter("occ_threshold",      50);

        planner_hz_        = this->get_parameter("planner_hz").as_double();
        robot_width_m_     = this->get_parameter("robot_width_m").as_double();
        path_simplify_dist_= this->get_parameter("path_simplify_dist").as_double();
        occ_threshold_     = this->get_parameter("occ_threshold").as_int();

        sub_map_  = create_subscription<nav_msgs::msg::OccupancyGrid>(
            "/map", 1,
            [this](nav_msgs::msg::OccupancyGrid::SharedPtr m) {
                map_ = m;
                plan_if_reversing();
            });

        sub_odom_ = create_subscription<nav_msgs::msg::Odometry>(
            "/odom", 10,
            [this](nav_msgs::msg::Odometry::SharedPtr m) { odom_ = m; });

        sub_goal_ = create_subscription<geometry_msgs::msg::PoseStamped>(
            "/reverse_goal", 10,
            [this](geometry_msgs::msg::PoseStamped::SharedPtr m) {
                goal_ = m;
                path_ready_ = false;
                plan_if_reversing();
            });

        sub_state_ = create_subscription<std_msgs::msg::String>(
            "/vehicle_state", 10,
            [this](std_msgs::msg::String::SharedPtr m) {
                auto prev = vehicle_state_;
                vehicle_state_ = m->data;
                if (prev != "REVERSE_EXECUTE" && vehicle_state_ == "REVERSE_EXECUTE")
                    plan();
                if (vehicle_state_ != "REVERSE_EXECUTE")
                    path_ready_ = false;
            });

        pub_path_  = create_publisher<nav_msgs::msg::Path>("/reverse_path", 1);
        pub_ready_ = create_publisher<std_msgs::msg::Bool>("/reverse_path_ready", 10);

        double dt = (planner_hz_ > 0) ? (1.0 / planner_hz_) : 0.5;
        timer_ = create_wall_timer(
            std::chrono::duration<double>(dt),
            std::bind(&ReversePathPlannerNode::step, this));

        RCLCPP_INFO(get_logger(), "reverse_path_planner_node (C++) 시작");
    }

private:
    void plan_if_reversing() {
        if (vehicle_state_ == "REVERSE_EXECUTE")
            plan();
    }

    bool world_to_cell(double wx, double wy, int &row, int &col) const {
        if (!map_) return false;
        auto &info = map_->info;
        double ox = info.origin.position.x;
        double oy = info.origin.position.y;
        double res = info.resolution;
        if (res < 1e-6) return false;
        col = static_cast<int>((wx - ox) / res);
        row = static_cast<int>((wy - oy) / res);
        return row >= 0 && row < static_cast<int>(info.height) &&
               col >= 0 && col < static_cast<int>(info.width);
    }

    std::pair<double, double> cell_to_world(int row, int col) const {
        auto &info = map_->info;
        double ox = info.origin.position.x;
        double oy = info.origin.position.y;
        double res = info.resolution;
        return {ox + (col + 0.5) * res, oy + (row + 0.5) * res};
    }

    void plan() {
        if (!map_ || !odom_ || !goal_) {
            path_ready_ = false;
            return;
        }

        double sx = odom_->pose.pose.position.x;
        double sy = odom_->pose.pose.position.y;
        double gx = goal_->pose.position.x;
        double gy = goal_->pose.position.y;

        int sr, sc, gr, gc;
        if (!world_to_cell(sx, sy, sr, sc) || !world_to_cell(gx, gy, gr, gc)) {
            RCLCPP_WARN(get_logger(), "[Planner] 셀 변환 실패");
            path_ready_ = false;
            return;
        }

        auto &info = map_->info;
        int rows = info.height;
        int cols = info.width;

        // 점유 격자 생성
        std::vector<bool> occ(rows * cols);
        for (int i = 0; i < rows * cols; ++i) {
            int8_t v = map_->data[i];
            occ[i] = (v >= occ_threshold_) || (v < 0);
        }

        double res = info.resolution;
        int radius = std::max(1, static_cast<int>(std::ceil(robot_width_m_ / 2.0 / res)));

        RCLCPP_INFO(get_logger(), "[Planner] A*: (%d,%d) → (%d,%d)", sr, sc, gr, gc);

        auto cell_path = astar(occ, rows, cols, {sr, sc}, {gr, gc}, radius);
        if (cell_path.empty()) {
            RCLCPP_WARN(get_logger(), "[Planner] 경로 없음");
            path_ready_ = false;
            return;
        }

        // 경로 단순화 + Path 메시지 생성
        nav_msgs::msg::Path path;
        path.header.stamp = now();
        path.header.frame_id = "map";

        double last_wx = -1e9, last_wy = -1e9;
        for (auto &c : cell_path) {
            auto [wx, wy] = cell_to_world(c.r, c.c);
            if (std::hypot(wx - last_wx, wy - last_wy) < path_simplify_dist_)
                continue;
            geometry_msgs::msg::PoseStamped ps;
            ps.header = path.header;
            ps.pose.position.x = wx;
            ps.pose.position.y = wy;
            ps.pose.orientation.w = 1.0;
            path.poses.push_back(ps);
            last_wx = wx; last_wy = wy;
        }
        // 마지막 점 보장
        auto [fwx, fwy] = cell_to_world(cell_path.back().r, cell_path.back().c);
        if (std::hypot(fwx - last_wx, fwy - last_wy) > 0.01) {
            geometry_msgs::msg::PoseStamped ps;
            ps.header = path.header;
            ps.pose.position.x = fwx;
            ps.pose.position.y = fwy;
            ps.pose.orientation.w = 1.0;
            path.poses.push_back(ps);
        }

        last_path_ = path;
        path_ready_ = true;
        RCLCPP_INFO(get_logger(), "[Planner] 경로 생성: %zu 웨이포인트",
                     path.poses.size());
    }

    void step() {
        std_msgs::msg::Bool b;
        b.data = path_ready_;
        pub_ready_->publish(b);

        if (path_ready_) {
            last_path_.header.stamp = now();
            pub_path_->publish(last_path_);
        }
    }

    // ── 파라미터 ─────────────────────────────────────────────────
    double planner_hz_, robot_width_m_, path_simplify_dist_;
    int occ_threshold_;

    // ── 상태 ─────────────────────────────────────────────────────
    nav_msgs::msg::OccupancyGrid::SharedPtr map_;
    nav_msgs::msg::Odometry::SharedPtr odom_;
    geometry_msgs::msg::PoseStamped::SharedPtr goal_;
    std::string vehicle_state_{"NORMAL_CENTER_DRIVE"};
    nav_msgs::msg::Path last_path_;
    bool path_ready_{false};

    // ── ROS ───────────────────────────────────────────────────────
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr sub_map_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odom_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr sub_goal_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr sub_state_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_path_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr pub_ready_;
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ReversePathPlannerNode>());
    rclcpp::shutdown();
    return 0;
}
