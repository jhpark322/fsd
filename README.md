# lane_length_pkg

ROS 2 기반 차선 인식 및 차선 추종 패키지입니다.  
카메라 영상으로 차선을 검출하고, 메모리 기반 폴백을 포함한 5개의 노드가 협력하여 로봇을 차선 중앙으로 주행시킵니다.

---

## 노드 구성 개요

```
[카메라]
   │
   ▼
lane_detection_node          ← 차선 검출, 오차 계산
   │  /lane_error_center_m
   │  /lane_error_right_m
   │  /lane_heading_error
   │  /lane_status
   │  /lane_width_m
   │
   ├──────────────────────► lane_memory_node     ← 차선 경로 기억
   │                              │ /lane_mem_error_center_m
   │                              │ /lane_mem_error_right_m
   │                              │ /lane_mem_heading_error
   │                              │ /lane_mem_valid
   │                              │ /lane_mem_width_m
   │                              │
   └──────────────────────────────▼
                        lane_guidance_mux_node   ← live/memory 신호 선택
                               │ /lane_error_center_m_active
                               │ /lane_error_right_m_active
                               │ /lane_heading_error_active
                               │ /lane_status_active
                               │ /lane_guidance_source
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
         lane_decision_node     lane_follow_control_node
         (통과 가능 판단)          (속도/조향 명령 출력)
               │                         │
               │ /control_mode           │ /cmd_vel
               │ /safe_stop              │
               └─────────────────────────┘
```

---

## 노드별 상세 설명

### 1. `lane_detection_node`

**역할**: 카메라 이미지에서 차선을 검출하고, 로봇이 차선 중앙/우측에서 얼마나 벗어났는지(오차)를 계산합니다.

**주요 동작**:
- 입력 이미지에서 흰색/노란색 차선 픽셀을 추출 (HLS/HSV 컬러 필터링)
- BEV(Bird's Eye View) 호모그래피 변환으로 탑뷰 이미지 생성
- 슬라이딩 윈도우 알고리즘으로 좌/우 차선 위치 추적 및 2차 다항식 피팅
- 메트릭 호모그래피(`H.npy`, `Hinv.npy`)로 픽셀→미터 변환
- 검출된 차선으로부터 center error, right error, heading error 계산
- 차선 폭(미터) 및 차선 경계 path 퍼블리시

**Subscribe**:
| 토픽 | 타입 | 설명 |
|------|------|------|
| `/image_raw` | `sensor_msgs/Image` | 카메라 원본 이미지 |

**Publish**:
| 토픽 | 타입 | 설명 |
|------|------|------|
| `/lane_error_center_m` | `Float32` | 차선 중앙으로부터의 횡방향 오차 (m) |
| `/lane_error_right_m` | `Float32` | 우측 차선 기준 횡방향 오차 (m) |
| `/lane_heading_error` | `Float32` | 차선 방향과의 각도 오차 (rad) |
| `/lane_status` | `String` | 차선 검출 상태 (`ok` / `lost`) |
| `/lane_width_m` | `Float32` | 현재 차선 폭 (m) |
| `/lane_centerline_base_path` | `nav_msgs/Path` | 차선 중심선 경로 (base_link 기준) |
| `/lane_left_boundary_base_path` | `nav_msgs/Path` | 좌측 차선 경계 경로 |
| `/lane_right_boundary_base_path` | `nav_msgs/Path` | 우측 차선 경계 경로 |

**주요 파라미터**:
| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `image_topic` | `/image_raw` | 입력 이미지 토픽 |
| `process_fps` | `10.0` | 이미지 처리 주기 (Hz) |
| `use_metric_homography` | `True` | 미터 단위 호모그래피 사용 여부 |
| `px_per_m` | `100.0` | BEV 이미지의 픽셀/미터 비율 |
| `nwindows` | `6` | 슬라이딩 윈도우 개수 |
| `smooth_alpha` | `0.08` | 차선 피팅 EMA 스무딩 계수 |
| `follow_mode` | `center` | 추종 모드 (`center` / `right`) |

---

### 2. `lane_memory_node`

**역할**: 차선이 정상적으로 보이는 동안 차선 경로를 odom 좌표계에 지속적으로 기록하고, 카메라가 차선을 잃었을 때 기억된 경로를 재생하여 guidance 오차를 제공합니다.

**주요 동작**:
- `/odom`으로 현재 로봇 위치를 추적
- 차선 상태가 `ok`인 동안 centerline/boundary path를 `LaneMapSample` 버퍼에 저장 (최대 700개 샘플, 3cm 간격)
- 기억된 경로에서 현재 로봇 위치 앞의 포인트들을 추출하여 cross-track 오차 및 heading 오차 계산
- 경로 끝에 도달하면(`endpoint_stop_distance_m` 이하) 유효성을 `False`로 전환
- RViz용 마커(중심선/좌우 경계) 퍼블리시

**Subscribe**:
| 토픽 | 타입 | 설명 |
|------|------|------|
| `/odom` | `nav_msgs/Odometry` | 로봇 위치/자세 |
| `/lane_status` | `String` | 현재 차선 검출 상태 |
| `/lane_centerline_base_path` | `nav_msgs/Path` | 검출된 중심선 경로 |
| `/lane_left_boundary_base_path` | `nav_msgs/Path` | 검출된 좌측 경계 |
| `/lane_right_boundary_base_path` | `nav_msgs/Path` | 검출된 우측 경계 |

**Publish**:
| 토픽 | 타입 | 설명 |
|------|------|------|
| `/lane_mem_error_center_m` | `Float32` | 기억 경로 기준 중앙 오차 (m) |
| `/lane_mem_error_right_m` | `Float32` | 기억 경로 기준 우측 오차 (m) |
| `/lane_mem_heading_error` | `Float32` | 기억 경로 기준 heading 오차 (rad) |
| `/lane_mem_valid` | `Bool` | 메모리 guidance 유효 여부 |
| `/lane_mem_width_m` | `Float32` | 기억된 차선 폭 (m) |
| `/lane_mem_status` | `String` | 메모리 상태 (`ok` / `lost`) |
| `/lane_mem_bridge_ready` | `Bool` | 최소 샘플 이상 수집되어 브릿지 준비됨 |
| `/lane_mem_remaining_m` | `Float32` | 기억 경로의 남은 거리 (m) |
| `/lane_memory_path` | `nav_msgs/Path` | 기억된 전체 경로 (odom 기준) |
| `/lane_memory_markers` | `visualization_msgs/MarkerArray` | RViz 시각화용 마커 |

**주요 파라미터**:
| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `sample_distance_m` | `0.03` | 샘플 저장 최소 이동 거리 (m) |
| `max_samples` | `700` | 최대 저장 샘플 수 |
| `cross_track_lookahead_m` | `0.40` | 횡방향 오차 계산용 전방 예측 거리 (m) |
| `heading_lookahead_m` | `0.55` | heading 오차 계산용 전방 예측 거리 (m) |
| `endpoint_stop_distance_m` | `0.10` | 경로 끝 판정 거리 (m) |
| `right_offset_ratio` | `0.30` | 우측 오차 = center_err + ratio × width |

---

### 3. `lane_guidance_mux_node`

**역할**: live 검출 신호와 memory 신호 중 어느 것을 사용할지 선택(또는 블렌딩)하여 하류 노드들에 단일 active guidance 신호를 제공합니다.

**주요 동작**:
- live가 유효하면(`lane_status == 'ok'` + 타임아웃 내) → **live** 선택
- live가 없고 memory가 유효하면 → **memory** 선택
- 둘 다 유효하고 `blend_when_both_valid=True`이면, 오차 차이가 허용 범위 내일 때 **가중 평균 블렌딩** (기본 live 95% : memory 5%)
- 소스 출처를 `/lane_guidance_source`로 퍼블리시 (`live` / `memory` / `blend` / `none`)

**Subscribe**:
| 토픽 | 타입 | 설명 |
|------|------|------|
| `/lane_error_center_m` | `Float32` | live 중앙 오차 |
| `/lane_error_right_m` | `Float32` | live 우측 오차 |
| `/lane_heading_error` | `Float32` | live heading 오차 |
| `/lane_status` | `String` | live 차선 상태 |
| `/lane_mem_error_center_m` | `Float32` | memory 중앙 오차 |
| `/lane_mem_error_right_m` | `Float32` | memory 우측 오차 |
| `/lane_mem_heading_error` | `Float32` | memory heading 오차 |
| `/lane_mem_valid` | `Bool` | memory 유효성 |
| `/lane_mem_status` | `String` | memory 상태 |

**Publish**:
| 토픽 | 타입 | 설명 |
|------|------|------|
| `/lane_error_center_m_active` | `Float32` | 선택된 중앙 오차 |
| `/lane_error_right_m_active` | `Float32` | 선택된 우측 오차 |
| `/lane_heading_error_active` | `Float32` | 선택된 heading 오차 |
| `/lane_status_active` | `String` | 선택된 차선 상태 |
| `/lane_guidance_source` | `String` | 현재 guidance 소스 (`live`/`memory`/`blend`/`none`) |

**주요 파라미터**:
| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `publish_hz` | `20.0` | 퍼블리시 주기 (Hz) |
| `live_timeout_sec` | `0.35` | live 신호 유효 시간 (s) |
| `mem_timeout_sec` | `1.20` | memory 신호 유효 시간 (s) |
| `blend_when_both_valid` | `False` | 둘 다 유효할 때 블렌딩 사용 여부 |
| `live_weight` | `0.95` | 블렌딩 시 live 가중치 |

---

### 4. `lane_decision_node`

**역할**: 현재 차선 폭을 로봇 폭과 비교하여 통과 가능 여부를 판단하고, 제어 모드와 안전 정지 명령을 퍼블리시합니다.

**주요 동작**:
- 차선 폭을 `live_width` → `held_width` → `mem_width` 순서로 폴백하여 결정
- `차선폭 >= 로봇폭 + margin + hysteresis` → **NORMAL_CENTER_DRIVE** (주행 허용)
- `차선폭 < 로봇폭 + margin` → **PASS_BLOCKED** (정지)
- 히스테리시스 적용으로 모드 채터링 방지
- 차선 상태 타임아웃, 차선 lost, 유효 폭 없음 시 모두 PASS_BLOCKED로 전환

**Subscribe**:
| 토픽 | 타입 | 설명 |
|------|------|------|
| `/lane_width_m` | `Float32` | 현재 검출된 차선 폭 |
| `/lane_mem_width_m` | `Float32` | 기억된 차선 폭 |
| `/lane_status_active` | `String` | active 차선 상태 |
| `/lane_guidance_source` | `String` | 현재 guidance 소스 |

**Publish**:
| 토픽 | 타입 | 설명 |
|------|------|------|
| `/control_mode` | `String` | 제어 모드 (`NORMAL_CENTER_DRIVE` / `PASS_BLOCKED`) |
| `/safe_stop` | `Bool` | 안전 정지 플래그 |
| `/decision_status` | `String` | 결정 이유 로그 문자열 |

**주요 파라미터**:
| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `robot_width_m` | `0.19` | 로봇 폭 (m) |
| `width_margin_m` | `0.00` | 통과 판정 여유 폭 (m) |
| `width_hysteresis_m` | `0.01` | 히스테리시스 폭 (m) |
| `lane_timeout_sec` | `0.5` | 차선 상태 유효 시간 (s) |
| `hold_last_width_sec` | `1.0` | 마지막 유효 폭 유지 시간 (s) |
| `stop_on_lane_lost` | `True` | 차선 lost 시 정지 여부 |
| `decision_hz` | `10.0` | 판단 주기 (Hz) |

---

### 5. `lane_follow_control_node`

**역할**: active guidance 오차(중앙/우측 오차, heading 오차)를 입력받아 PD 제어로 `cmd_vel`을 계산하고 퍼블리시합니다.

**주요 동작**:
- `safe_stop=True` 또는 `control_mode=PASS_BLOCKED`이면 즉시 정지 명령 출력
- guidance source(`live` / `blend` / `memory`)에 따라 서로 다른 제어 게인과 deadband 적용
  - **live**: 표준 게인, 좁은 deadband
  - **blend**: 게인 80%, heading 60%, 속도 제한 23mm/s
  - **memory**: 게인 35%, heading 20%, 속도 제한 18mm/s (보수적 추종)
- 오차가 클수록 속도 감소 (slow_error_m 이상부터 선형 감속, hard_stop_error_m 이상이면 최저속 또는 제자리 회전)
- lookahead 보상 옵션: 카메라-baselink 오프셋 및 전방 예측 오차 보정

**Subscribe**:
| 토픽 | 타입 | 설명 |
|------|------|------|
| `/lane_error_center_m_active` | `Float32` | 중앙 횡방향 오차 (m) |
| `/lane_error_right_m_active` | `Float32` | 우측 횡방향 오차 (m) |
| `/lane_heading_error_active` | `Float32` | heading 오차 (rad) |
| `/lane_status_active` | `String` | 차선 상태 |
| `/lane_guidance_source` | `String` | guidance 소스 |
| `/control_mode` | `String` | 제어 모드 |
| `/safe_stop` | `Bool` | 안전 정지 플래그 |

**Publish**:
| 토픽 | 타입 | 설명 |
|------|------|------|
| `/cmd_vel` | `geometry_msgs/Twist` | 로봇 속도 명령 |

**주요 파라미터**:
| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `control_hz` | `20.0` | 제어 주기 (Hz) |
| `kp_m` | `0.22` | 횡방향 오차 P 게인 |
| `k_heading` | `0.03` | heading 오차 게인 |
| `nominal_speed` | `0.028` | 기본 전진 속도 (m/s) |
| `min_speed` | `0.018` | 최소 전진 속도 (m/s) |
| `hard_stop_error_m` | `0.25` | 이 오차 이상이면 최저속/제자리 회전 (m) |
| `memory_kp_scale` | `0.35` | memory 소스 시 게인 스케일 |
| `default_follow_mode` | `center` | 기본 추종 기준 (`center` / `right`) |

---

## 실행 방법

### 빌드

```bash
cd ~/fsd_ws
colcon build --packages-select lane_length_pkg
source install/setup.bash
```

### 1. TurtleBot3 로봇 실행

```bash
source /opt/ros/humble/setup.bash
source ~/turtlebot3_ws/install/setup.bash
export TURTLEBOT3_MODEL=burger
ros2 launch turtlebot3_bringup robot.launch.py
```

### 2. 카메라 노드 실행

```bash
ros2 run usb_cam usb_cam_node_exe --ros-args \
  -p pixel_format:=yuyv \
  -p camera_info_url:="file:///home/jhp/fsd_ws/src/lane_length_pkg/config/default_cam.yaml"
```

### 3. 차선 추종 노드 실행

```bash
ros2 run lane_length_pkg lane_detection_node
ros2 run lane_length_pkg lane_memory_node
ros2 run lane_length_pkg lane_guidance_mux_node
ros2 run lane_length_pkg lane_decision_node
ros2 run lane_length_pkg lane_follow_control_node
```

---

## 의존성

- ROS 2 (Humble 이상)
- `rclpy`, `cv_bridge`, `sensor_msgs`, `geometry_msgs`, `nav_msgs`, `std_msgs`, `visualization_msgs`
- `opencv-python`, `numpy`
