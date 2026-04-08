cmake_minimum_required(VERSION 3.8)
project(v2v_cpp_nodes)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -O2)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

# ── chassis_control_node ──────────────────────────────────
add_executable(chassis_control_node src/chassis_control_node.cpp)
ament_target_dependencies(chassis_control_node
  rclcpp std_msgs geometry_msgs nav_msgs)

# ── safety_supervisor_node ────────────────────────────────
add_executable(safety_supervisor_node src/safety_supervisor_node.cpp)
ament_target_dependencies(safety_supervisor_node
  rclcpp std_msgs geometry_msgs sensor_msgs)

# ── reverse_path_planner_node ─────────────────────────────
add_executable(reverse_path_planner_node src/reverse_path_planner_node.cpp)
ament_target_dependencies(reverse_path_planner_node
  rclcpp std_msgs geometry_msgs nav_msgs)

install(TARGETS
  chassis_control_node
  safety_supervisor_node
  reverse_path_planner_node
  DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
