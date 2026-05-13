# Visual V2V 기반 협력 양보 주행 시스템

협소한 도로에서 두 대의 TurtleBot3가 서로 마주쳤을 때, HD Map 없이 주변을 인식하고, 상대 차량과 협상하여 양보·교행·재진입까지 수행하는 협력 주행 시스템.

---

## 전체 시스템 흐름

```
평상시 중앙 주행  →  상대 차량 검출  →  우측통행 모드 전환
                                          │
                          교행 가능 ←─────┤
                                          │ 교행 불가
                                          ▼
                                    교착 상태 감지
                                          │
                          ┌───────────────┤
                          ▼               ▼
                    가위바위보 협상    점수 기반 자동 판단
                          │               │
                          └───────┬───────┘
                                  │
                    ┌─────────────┼──────────────┐
                    ▼                            ▼
              양보 차량                      진행 차량
            (후진 회피)                    (통과 대기)
                    │                            │
                    ▼                            │
              상대 통과 완료 ◄────────────────────┘
                    │
                    ▼
                재진입 → 중앙 주행 복귀
```

---

## 패키지 구성

### 1. `lane_length_pkg` — 차선 인식 및 기본 추종 (Python)

| 노드 | 역할 |
|------|------|
| `lane_detection_node` | 카메라 영상 → 차선 검출, IPM, 슬라이딩 윈도우, 오차 계산 |
| `lane_memory_node` | odom 좌표계에 차선 경로 기록, 카메라 loss 시 기억 경로 재생 |
| `lane_guidance_mux_node` | live/memory 신호 선택 또는 블렌딩 |
| `lane_decision_node` | 차선 폭 기반 통과 가능 여부 판단 |
| `lane_follow_control_node` | PD 제어 기반 `cmd_vel` 생성 |

### 2. `v2v_cooperative_pkg` — 협력 양보 주행 핵심 로직 (Python)

| 노드 | 역할 |
|------|------|
| `v2v_decision_node` | **9상태 FSM** — 중앙주행→우측통행→교착→협상→점수판단→후진→대기→재진입→안전정지 |
| `visual_v2v_perception_node` | 상단 ROI 기반 상대 LED 패널/상태/점수 인식 + 상대 거리 추정 |
| `spatial_memory_node` | 0.5m 간격 10m FIFO 공간 기억 + reverse_goal (후진 비켜줄 지점) 계산 |
| `negotiation_hmi_node` | 가위바위보 협상 HMI (키보드 r/p/s/x 입력 + LED 상태 판독 + 타임아웃) |
| `led_interface_node` | LED 5개 상태별 색상 패턴 + 점수 기반 점등 개수 표현 (Jetson GPIO/RPi.GPIO) |

### 3. `v2v_cpp_nodes` — 성능 핵심 노드 (C++)

| 노드 | 역할 | C++ 전환 이유 |
|------|------|--------------|
| `chassis_control_node` | PID 중앙/우측 추종 + Reverse Pure Pursuit 통합 20Hz 제어 | 실시간 제어 지연 최소화 |
| `safety_supervisor_node` | LiDAR/인식/경로/제어 이상 5단계 우선순위 20Hz 감시 | 안전 크리티컬 결정론적 타이밍 |
| `reverse_path_planner_node` | A* 기반 점유 격자 후진 회피 경로 생성 | 연산 집약적 탐색 최적화 |

> `v2v_cooperative_pkg`에 Python fallback 버전도 유지되어 있습니다 (`_py` 접미사).

---

## 노드 구현 기준

이 저장소는 노드 성격에 따라 C++와 Python을 나눠 사용한다.

| 구현 | 적용 노드 | 이유 |
|------|-----------|------|
| C++ | `chassis_control_node`, `safety_supervisor_node`, `reverse_path_planner_node` | 20Hz 제어, 안전 정지, A* 경로 탐색처럼 지연과 반복 연산이 실제 주행 안정성에 직접 영향을 주는 부분 |
| Python | `v2v_decision_node`, `visual_v2v_perception_node`, `spatial_memory_node`, `negotiation_hmi_node`, `led_interface_node` | 상태 정책, HMI, LED 프로토콜, HSV/ROI 기반 프로토타입 인식처럼 실험 중 파라미터와 로직을 자주 바꾸는 부분 |
| Python fallback | `*_node_py` | C++ 노드 빌드가 어려운 환경에서 같은 토픽 계약으로 기능 확인 |

실차 주행 기본 조합은 C++ 제어/안전/경로 노드와 Python 판단/인지/HMI 노드를 함께 실행하는 것이다.

---

## 상태 기계 (State Machine)

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  NORMAL_CENTER_DRIVE ──(상대 검출)──► KEEP_RIGHT_APPROACH│
│         ▲                                    │          │
│         │                          ┌─────────┤          │
│         │               (상대 사라짐)│  (교착 확인)      │
│         │                          │         ▼          │
│      REENTER                       │  DEADLOCK_DETECTED │
│         ▲                          │         │          │
│         │                          │         ▼          │
│  (재정렬 완료)                      │  RPS_NEGOTIATION   │
│         │                          │    │       │       │
│      WAIT_PASS ◄──(후진 완료 신호)┐  │  (승리)  (실패)   │
│         │                      │   │    │       │       │
│  (상대 통과)                    │   │    │       ▼       │
│                                │   │    │  SCORE_BASED  │
│                                │   │    │  _DECISION    │
│                                │   │    │    │     │    │
│                     REVERSE_   │   │    │ (양보) (진행) │
│                     EXECUTE ◄──┼───┼────┘    │     │    │
│                                    │         │     │    │
│                                    └─────────┘     │    │
│                                                    │    │
│  ※ 어느 상태에서든 → SAFE_STOP (안전 감시 발동)      │    │
│                      SAFE_STOP → NORMAL (해제 시)    │    │
└─────────────────────────────────────────────────────────┘
```

---

## Yield Score 계산

점수가 **높은 차량이 양보**, 낮은 차량이 진행.

```
Yield Score = 0.30×S_space + 0.25×S_reverse + 0.20×S_entry
            + 0.10×S_wait  + 0.10×S_stability + 0.05×S_dist
```

| 요소 | 가중치 | 의미 |
|------|--------|------|
| `S_space` | 0.30 | 우측 회피 공간 (클수록 비키기 쉬움) |
| `S_reverse` | 0.25 | 후진 가능 여부 (reverse_goal 존재) |
| `S_entry` | 0.20 | 협소 구간 진입 정도 |
| `S_wait` | 0.10 | 대기 누적 시간 (오래 기다릴수록 진행 유리) |
| `S_stability` | 0.10 | 차선 인식 안정성 |
| `S_dist` | 0.05 | 상대 차량과의 거리 |

---

## 토픽 흐름

```
[카메라] → lane_detection_node → lane_memory_node
    │
    └──► visual_v2v_perception_node ──► /relative_distance, /led_state, /opponent_yield_score
                │                      │
                └──► lane_guidance_mux_node
                              │
                              ├─► v2v_decision_node ──┬──► negotiation_hmi_node
                              │         │             ├──► led_interface_node (/ego_yield_score)
                              │         │             └──► spatial_memory_node
                              │         │
                              │         └──► reverse_path_planner_node (C++)
                              │                        │
                              └──► chassis_control_node (C++) ──► /cmd_vel, /reverse_motion_done
                                           │
[LiDAR] ──► safety_supervisor_node (C++) ──┘
[SLAM]  ──► reverse_path_planner_node
```

---

## 빌드

```bash
cd ~/fsd_ws

# Python 패키지
colcon build --packages-select lane_length_pkg v2v_cooperative_pkg

# C++ 패키지
colcon build --packages-select v2v_cpp_nodes

source install/setup.bash
```

---

## 실행

### 1단계: 로봇 기본 bring-up

```bash
export TURTLEBOT3_MODEL=burger
ros2 launch turtlebot3_bringup robot.launch.py
```

### 2단계: 카메라

```bash
ros2 run usb_cam usb_cam_node_exe --ros-args \
  -p pixel_format:=yuyv \
  -p camera_info_url:="file:///path/to/default_cam.yaml"
```

### 3단계: SLAM

```bash
ros2 launch slam_toolbox online_async_launch.py
```

### 4단계: 차선 인식 + 기본 추종

```bash
ros2 run lane_length_pkg lane_detection_node
ros2 run lane_length_pkg lane_memory_node
ros2 run lane_length_pkg lane_guidance_mux_node
```

### 5단계: V2V 협력 양보 시스템

```bash
# Python 노드
ros2 run v2v_cooperative_pkg v2v_decision_node
ros2 run v2v_cooperative_pkg visual_v2v_perception_node
ros2 run v2v_cooperative_pkg spatial_memory_node
ros2 run v2v_cooperative_pkg negotiation_hmi_node
ros2 run v2v_cooperative_pkg led_interface_node

# C++ 노드 (성능 핵심)
ros2 run v2v_cpp_nodes chassis_control_node
ros2 run v2v_cpp_nodes safety_supervisor_node
ros2 run v2v_cpp_nodes reverse_path_planner_node
```

---

## 하드웨어 구성

| 항목 | 사양 |
|------|------|
| 주행 플랫폼 | TurtleBot3 Burger |
| 연산 장치 | Jetson Orin Nano |
| 카메라 | Logitech C920e (1280×720, 30fps) |
| LiDAR | 2D LiDAR (360°) |
| LED | 일반 LED 5개 직렬 (상태 색상 + 점수 점등 개수) |
| 하위 제어 | OpenCR |

---

## 의존성

| 분류 | 패키지 |
|------|--------|
| ROS2 | `rclpy`, `rclcpp`, `cv_bridge`, `std_msgs`, `geometry_msgs`, `nav_msgs`, `sensor_msgs`, `visualization_msgs` |
| 비전 | `opencv-python`, `numpy`, `scipy` |
| SLAM | `slam_toolbox`, `navigation2` |
| 도구 | `rosbag2`, `rviz2` |

---

## 정량 목표

| 지표 | 목표 |
|------|------|
| 비전 파이프라인 처리 속도 | 15 FPS 이상 |
| 차선 중심 추종 횡방향 오차 | 0.15m 이하 |
| 상대 차량/LED 인식 정확도 | 90% 이상 |
| 우측통행 전환 반응 시간 | 1.0초 이내 |
| 후진 회피 성공률 | 80% 이상 |
| 전체 협력 주행 시나리오 성공률 | 80% 이상 |
