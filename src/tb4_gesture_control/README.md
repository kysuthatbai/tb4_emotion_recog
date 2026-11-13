# tb4_gesture_control: điều khiển TurtleBot 4 bằng cử chỉ tay

Webcam laptop → MediaPipe Gesture Recognizer → `/detected_gesture` → máy trạng thái → `/cmd_vel` → TurtleBot 4 trong Gazebo.

```
 webcam ──▶ gesture_recognition_node ──/detected_gesture (String, ~15 Hz)──▶ robot_behavior_node ──/cmd_vel──▶ TurtleBot 4
               (OpenCV + MediaPipe,                                          (máy trạng thái, vòng kín
                lọc nhiễu, phát hiện Wave)  ◀──/robot_behavior_state──        theo /odom, kiểm tra /scan)
```

Mục lục:
- [0. Chuẩn bị (làm một lần)](#0-chuẩn-bị-làm-một-lần)
- [Tóm tắt lệnh](#tóm-tắt-lệnh)
- [Phần 1 – Simulation và webcam](#phần-1--simulation-và-webcam)
- [Phần 2 – Nhận diện cử chỉ tay](#phần-2--nhận-diện-cử-chỉ-tay)
- [Phần 3 – Điều khiển hành vi TurtleBot 4](#phần-3--điều-khiển-hành-vi-turtlebot-4)
- [Kịch bản demo](#kịch-bản-demo)
- [Dừng và dọn dẹp](#dừng-và-dọn-dẹp)
- [Thuật toán](#thuật-toán)
- [Kết quả kiểm thử](#kết-quả-kiểm-thử)
- [Tham số](#tham-số)
- [Xử lý sự cố](#xử-lý-sự-cố)

---

## 0. Chuẩn bị (làm một lần)

```bash
# Cài MediaPipe. Giữ numpy<2 và OpenCV 4.9 để không làm hỏng cv_bridge / ultralytics đang dùng
pip3 install --user mediapipe "numpy<2" "opencv-contrib-python==4.9.0.80"

# Build package
cd ~/tb4_ws
colcon build --symlink-install --packages-select tb4_gesture_control
```

**Mọi terminal mới** đều phải source trước khi chạy lệnh ROS:

```bash
source ~/tb4_ws/install/setup.bash
```

Kiểm tra cài đặt:

```bash
python3 -c "import mediapipe, numpy, cv2; print(mediapipe.__version__, numpy.__version__, cv2.__version__)"
ros2 pkg executables tb4_gesture_control
```

Kết quả mong đợi: in ra phiên bản (numpy phải là 1.x), và đủ 5 lệnh: `cmd_vel_check`, `gesture_recognition_node`, `prepare_robot`, `robot_behavior_node`, `webcam_view`.

Mô hình `models/gesture_recognizer.task` (8 MB) đã nằm sẵn trong package, không cần tải thêm.

---

## Tóm tắt lệnh

Mỗi dòng chạy trong **một terminal riêng**, theo đúng thứ tự (nhớ `source ~/tb4_ws/install/setup.bash` trước).

| Phần | Terminal | Lệnh |
|---|---|---|
| 1 | T1 | `ros2 launch tb4_gesture_control sim.launch.py` |
| 1 | T2 | `ros2 run tb4_gesture_control cmd_vel_check` |
| 1 | T3 | `ros2 run tb4_gesture_control webcam_view` |
| 2 | T2 | `ros2 run tb4_gesture_control gesture_recognition_node` |
| 2 | T3 | `ros2 topic echo /detected_gesture` |
| 3 | T1 | `ros2 launch tb4_gesture_control sim.launch.py` (để chạy tiếp từ Phần 1) |
| 3 | T2 | `ros2 run tb4_gesture_control gesture_recognition_node` |
| 3 | T3 | `ros2 run tb4_gesture_control robot_behavior_node` |
| 3 (gộp) | T2 | `ros2 launch tb4_gesture_control gesture_control.launch.py` (thay cho 2 lệnh trên) |

---

## Phần 1 – Simulation và webcam

### 1.1. Khởi chạy TurtleBot 4 trong Gazebo

**Terminal 1**

```bash
source ~/tb4_ws/install/setup.bash
ros2 launch tb4_gesture_control sim.launch.py
```

Kết quả mong đợi:
- Gazebo mở sân trống 8 × 8 m, có tường bao và một vòng tròn xanh đánh dấu vùng hoạt động.
- Robot tự undock (lùi khoảng 0.35 m và quay khoảng 180°), rồi tiến thêm 0.6 m ra giữa sân.
- Terminal in ra dòng: `[prepare_robot]: Sẵn sàng tại odom (-0.94, ...). Có thể bắt đầu điều khiển bằng cử chỉ`

Chờ thấy dòng này rồi mới chạy các bước tiếp theo. Mất khoảng 30–40 s.

Biến thể:

```bash
ros2 launch tb4_gesture_control sim.launch.py rviz:=true        # mở thêm RViz
ros2 launch tb4_gesture_control sim.launch.py model:=lite       # TurtleBot 4 Lite
ros2 launch tb4_gesture_control sim.launch.py prepare:=false    # không tự undock (robot vẫn đứng ở dock)
```

### 1.2. Kiểm tra robot nhận lệnh vận tốc qua `/cmd_vel`

**Terminal 2**

```bash
source ~/tb4_ws/install/setup.bash

# Topic /cmd_vel tồn tại và có subscriber (bridge sang Gazebo)
ros2 topic list | grep cmd_vel
ros2 topic info /cmd_vel

# Kiểm tra tự động: xoay trái, xoay phải, tiến, lùi; đo dịch chuyển thật trên /odom
ros2 run tb4_gesture_control cmd_vel_check
```

Kết quả mong đợi của `cmd_vel_check`:

```
[cmd_vel_check] /cmd_vel có 1 subscriber
  xoay trai  lệnh v=+0.00 w=+0.80 trong 2 s -> dịch 0.000 m, quay  +70.3 độ  [ĐẠT]
  xoay phai  lệnh v=+0.00 w=-0.80 trong 2 s -> dịch 0.000 m, quay  -75.6 độ  [ĐẠT]
  tien       lệnh v=+0.15 w=+0.00 trong 2 s -> dịch 0.273 m, quay   +0.0 độ  [ĐẠT]
  lui        lệnh v=-0.15 w=+0.00 trong 2 s -> dịch 0.170 m, quay   -0.0 độ  [ĐẠT]
[cmd_vel_check] Robot nhận lệnh /cmd_vel bình thường
```

(Quãng lùi ngắn hơn quãng tiến là do giới hạn lùi mặc định của Create 3. `robot_behavior_node` sẽ tự tắt giới hạn này ở Phần 3.)

Kiểm tra thủ công (tùy chọn):

```bash
# Robot quay tại chỗ; Ctrl+C để ngừng gửi
ros2 topic pub -r 10 /cmd_vel geometry_msgs/msg/Twist "{angular: {z: 0.5}}"

# Robot tiến chậm; Ctrl+C để ngừng gửi
ros2 topic pub -r 10 /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.1}}"

# Gửi lệnh dừng
ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist "{}"

# Xem vận tốc thật của robot
ros2 topic echo /odom --field twist.twist
```

### 1.3. Đọc và hiển thị webcam bằng OpenCV

**Terminal 3** (không cần sim)

```bash
source ~/tb4_ws/install/setup.bash
ros2 run tb4_gesture_control webcam_view          # /dev/video0
ros2 run tb4_gesture_control webcam_view 1        # nếu webcam là /dev/video1
```

Kết quả mong đợi: cửa sổ hiện hình webcam đã lật gương, góc trên có độ phân giải và FPS (khoảng 640x480, 20 fps). Nhấn `q` hoặc `Esc` để thoát.

Xem các webcam có trên máy:

```bash
ls /dev/video*
v4l2-ctl --list-devices
```

---

## Phần 2 – Nhận diện cử chỉ tay

Phần này **không cần sim**: chỉ cần webcam.

### 2.1. Chạy node nhận diện

**Terminal 1**

```bash
source ~/tb4_ws/install/setup.bash
ros2 run tb4_gesture_control gesture_recognition_node
```

Kết quả mong đợi:
- Cửa sổ `gesture_recognition_node` hiện hình webcam kèm khung xương bàn tay.
- Thanh trên cùng hiện tên cử chỉ và độ tin cậy (ví dụ `Victory  93%`), dòng `raw:` là nhãn thô từng khung, cùng FPS.
- Bên trái là bảng cử chỉ → hành vi; dòng dưới cùng hiện trạng thái robot (`ROBOT: ...`).
- Terminal in mỗi khi có cử chỉ mới, ví dụ `Cử chỉ: Thumb_Up (0.73)`.
- Nhấn `q` / `Esc` hoặc Ctrl+C để thoát. Khi thoát, node in bảng tổng kết số lần nhận được từng cử chỉ.

Tùy chọn:

```bash
ros2 run tb4_gesture_control gesture_recognition_node --ros-args -p camera_index:=1     # webcam khác
ros2 run tb4_gesture_control gesture_recognition_node --ros-args -p min_score:=0.7      # khắt khe hơn
ros2 run tb4_gesture_control gesture_recognition_node --ros-args -p show_window:=false  # không mở cửa sổ
```

### 2.2. Xem kết quả publish trên `/detected_gesture`

**Terminal 2**

```bash
source ~/tb4_ws/install/setup.bash
ros2 topic echo /detected_gesture          # in nhãn: None, Victory, Thumb_Up, ...
ros2 topic hz /detected_gesture            # khoảng 15 Hz (publish mọi khung, kể cả None)
ros2 topic info /detected_gesture          # kiểu std_msgs/msg/String
```

### 2.3. Thử 8 cử chỉ

Đưa **một bàn tay** vào giữa khung hình, cách webcam khoảng 0.5–1 m, rồi giữ mỗi cử chỉ khoảng 1 s:

| Cử chỉ | Nhãn publish |
|---|---|
| Vẫy tay trái–phải (bàn tay xòe) | `Wave` |
| Giơ 1 ngón trỏ | `Pointing_Up` |
| Giơ 2 ngón (chữ V) | `Victory` |
| Giơ ngón cái lên | `Thumb_Up` |
| Xòe bàn tay, giữ yên | `Open_Palm` |
| Nắm tay | `Closed_Fist` |
| Ngón cái chúc xuống | `Thumb_Down` |
| Ngón cái + trỏ + út (I love you) | `ILoveYou` |

### 2.4. Chạy unit test bộ lọc nhiễu (tùy chọn)

```bash
cd ~/tb4_ws
python3 -m pytest src/tb4_gesture_control/test -q          # mong đợi: 8 passed
```

---

## Phần 3 – Điều khiển hành vi TurtleBot 4

### 3.1. Chạy từng node riêng (dễ xem log)

**Terminal 1**: sim (giống Phần 1), chờ dòng `Sẵn sàng tại odom ...`

```bash
source ~/tb4_ws/install/setup.bash
ros2 launch tb4_gesture_control sim.launch.py
```

**Terminal 2**: node nhận diện cử chỉ

```bash
source ~/tb4_ws/install/setup.bash
ros2 run tb4_gesture_control gesture_recognition_node
```

**Terminal 3**: node điều khiển hành vi

```bash
source ~/tb4_ws/install/setup.bash
ros2 run tb4_gesture_control robot_behavior_node
```

Kết quả mong đợi ở terminal 3:

```
[robot_behavior_node]: Sẵn sàng. Chờ cử chỉ trên /detected_gesture
[robot_behavior_node]: Đã nhận tín hiệu cử chỉ
[robot_behavior_node]: motion_control: safety_override = backup_only (cho phép lùi xa)
```

**Terminal 4** (tùy chọn): theo dõi trạng thái và lệnh vận tốc

```bash
source ~/tb4_ws/install/setup.bash
ros2 topic echo /robot_behavior_state      # WAITING / SELECTED / RUNNING / STOPPED ...
ros2 topic echo /cmd_vel
```

### 3.2. Hoặc chạy gộp 2 node bằng launch

Thay cho terminal 2 và 3 ở trên (sim vẫn chạy ở terminal 1):

```bash
source ~/tb4_ws/install/setup.bash
ros2 launch tb4_gesture_control gesture_control.launch.py
ros2 launch tb4_gesture_control gesture_control.launch.py camera_index:=1                 # webcam khác
ros2 launch tb4_gesture_control gesture_control.launch.py params:=/đường/dẫn/khác.yaml     # tham số riêng
```

Tham số mặc định nằm ở `config/gesture_control.yaml`.

### 3.3. Bảng cử chỉ → hành vi

| Cử chỉ | Hành vi robot | Log ở `robot_behavior_node` |
|---|---|---|
| Vẫy tay (`Wave`) | Xoay trái–phải ±35° hai lần rồi về hướng cũ (chạy ngay) | `[Wave] chào lại: xoay trái-phải` |
| 1 ngón (`Pointing_Up`) | **Chọn** tiến 0.4 m – lùi 0.4 m, lặp 4 lần | `[Pointing_Up] chọn FORWARD_BACK. Giơ Thumb_Up để xác nhận` |
| 2 ngón (`Victory`) | **Chọn** quỹ đạo tròn bán kính 0.5 m | `[Victory] chọn CIRCLE. Giơ Thumb_Up để xác nhận` |
| Ngón cái (`Thumb_Up`) | **Xác nhận** và thực hiện hành vi đã chọn | `[Thumb_Up] xác nhận -> thực hiện ...` |
| Xòe tay (`Open_Palm`) | Dừng ngay, vẫn giữ lựa chọn (Thumb_Up để chạy lại) | `[Open_Palm] DỪNG NGAY ...` |
| Nắm tay (`Closed_Fist`) | Hủy hành vi và lựa chọn, về trạng thái chờ | `[Closed_Fist] hủy ... -> trạng thái chờ` |

Muốn vẫy tay cũng phải xác nhận bằng ngón cái:

```bash
ros2 run tb4_gesture_control robot_behavior_node --ros-args -p wave_requires_confirm:=true
```

### 3.4. Kiểm tra các yêu cầu an toàn

```bash
# Dừng khi mất tín hiệu: Ctrl+C terminal 2 (node nhận diện) trong lúc robot đang chạy
#   -> terminal 3 in "MẤT TÍN HIỆU CỬ CHỈ > 1.0 s -> dừng robot", robot đứng yên sau ~1 s

# Dừng khi tắt node hành vi: Ctrl+C terminal 3
#   -> in "Tắt node: gửi lệnh dừng robot"

# Dừng khi gặp vật cản: cho robot tiến rất xa để chạy thẳng vào tường
ros2 run tb4_gesture_control robot_behavior_node --ros-args -p forward_distance:=8.0
#   -> chọn 1 ngón, xác nhận ngón cái; robot tự dừng trước tường với log "vat can 0.35 m"
```

---

## Kịch bản demo

1. Terminal 1: `ros2 launch tb4_gesture_control sim.launch.py`, chờ `Sẵn sàng tại odom`.
2. Terminal 2: `ros2 launch tb4_gesture_control gesture_control.launch.py`.
3. Trước webcam, lần lượt:
   1. **Vẫy tay**: robot xoay trái–phải chào lại.
   2. **1 ngón**: chọn tiến–lùi (cửa sổ hiện `SELECTED FORWARD_BACK`). **Ngón cái**: robot tiến–lùi 4 lần.
   3. **2 ngón**: chọn vòng tròn. **Ngón cái**: robot chạy vòng tròn.
   4. Giữa chừng **xòe tay**: robot dừng ngay. **Ngón cái**: chạy lại vòng tròn từ đầu.
   5. **Nắm tay**: hủy, robot về trạng thái chờ.
   6. Giơ **ngón cái** khi chưa chọn gì: robot không chạy (chứng minh cần chọn trước rồi mới xác nhận).
   7. Ctrl+C terminal 2 khi robot đang chạy: robot dừng do mất tín hiệu.

Mẹo: hạ tay xuống (ra khỏi khung hình) giữa hai cử chỉ. Mỗi lần giữ một cử chỉ chỉ phát đúng một lệnh.

---

## Dừng và dọn dẹp

- Ctrl+C ở từng terminal (node hành vi luôn gửi lệnh dừng robot trước khi thoát).
- Nếu Gazebo còn chạy ngầm sau khi tắt launch:

```bash
pkill -f "ign gazebo"
pkill -f parameter_bridge
```

---

## Thuật toán

### Cấu trúc package

| File | Vai trò |
|---|---|
| `launch/sim.launch.py` | Gazebo + sân `gesture_arena` + spawn robot + `prepare_robot` |
| `worlds/gesture_arena.sdf` | Sân trống 8 × 8 m, tường cao 0.5 m (LiDAR thấy được) |
| `tb4_gesture_control/prepare_robot.py` | Undock rồi tiến 0.6 m ra giữa sân (chạy một lần) |
| `tb4_gesture_control/cmd_vel_check.py` | Gửi lệnh `/cmd_vel`, đo `/odom`, in ĐẠT / KHÔNG ĐẠT |
| `tb4_gesture_control/webcam_view.py` | Đọc và hiển thị webcam bằng OpenCV |
| `tb4_gesture_control/gesture_recognition_node.py` | Nhận diện cử chỉ, publish `/detected_gesture` |
| `tb4_gesture_control/robot_behavior_node.py` | Máy trạng thái hành vi, publish `/cmd_vel` |
| `tb4_gesture_control/gestures.py` | Tên cử chỉ dùng chung |
| `launch/gesture_control.launch.py`, `config/gesture_control.yaml` | Chạy gộp 2 node cùng tham số |
| `test/test_gesture_filter.py` | Unit test bộ lọc nhiễu / phát hiện Wave |

### Vì sao cần `prepare_robot`

Robot spawn ở trạng thái đang sạc, dock nằm ngay trước mặt (+0.157 m). Dock cao khoảng 10 cm, thấp hơn mặt quét LiDAR (khoảng 19 cm), nên LiDAR không thấy dock. Nếu chạy quỹ đạo tròn ngay tại chỗ thì robot quét ra sau và đụng dock. `prepare_robot` gọi action `/undock` (robot lùi 0.35 m, quay khoảng 180°) rồi tiến thêm 0.6 m.

### `gesture_recognition_node` (mỗi khung)

1. `cv2.VideoCapture` đọc ảnh, lật gương, chuyển BGR sang RGB.
2. MediaPipe `GestureRecognizer` (chế độ VIDEO) trả về nhãn, độ tin cậy và 21 landmark bàn tay.
3. Bộ lọc nhiễu `GestureFilter`:
   - nhãn có độ tin cậy dưới `min_score` (0.6) bị coi là `None`;
   - bỏ phiếu trên 7 khung gần nhất, cần ít nhất 5 phiếu (khoảng 0.3 s) thì nhãn mới ổn định;
   - `Open_Palm` chỉ hợp lệ khi bàn tay đứng yên (dịch dưới 5% bề rộng khung trong 0.5 s) và không ở ngay sau một cú vẫy, nên vẫy tay không bị hiểu thành lệnh dừng.
4. Phát hiện **Wave** (MediaPipe không có sẵn nhãn này): theo dõi tọa độ x của tâm bàn tay khi tay xòe, đếm số lần đổi chiều bằng zigzag có ngưỡng trễ (mỗi nhịp lệch ít nhất 6% bề rộng khung). Đổi chiều từ 2 lần trở lên trong 1.5 s, và tay vẫn đang chuyển động, thì nhận là `Wave`.
5. Publish nhãn ổn định lên `/detected_gesture` **ở mọi khung**, kể cả `None`, làm nhịp tim cho `robot_behavior_node`. Khi webcam hỏng, node ngừng publish và tự thử mở lại camera mỗi 2 s.

### `robot_behavior_node`

Máy trạng thái:

```
IDLE ──Pointing_Up/Victory──▶ SELECTED ──Thumb_Up──▶ EXECUTING ──xong──▶ IDLE
  ▲ ◀──────Closed_Fist──────────┘                       │ Open_Palm
  └────────Closed_Fist──────── STOPPED ◀────────────────┘  (Thumb_Up: chạy lại)
Mọi trạng thái: mất /detected_gesture > 1 s → dừng robot, về IDLE.
```

Các yêu cầu được đáp ứng thế nào:
- **Chỉ chạy sau khi xác nhận**: `Pointing_Up` và `Victory` chỉ chọn hành vi; chỉ `Thumb_Up` ở trạng thái SELECTED/STOPPED mới cho robot chạy.
- **Không dùng bàn phím**: mọi lệnh đi qua cử chỉ; phím `q` chỉ để đóng cửa sổ.
- **Chống nhiễu kích hoạt liên tục**:
  - lớp 1 trong node nhận diện: ngưỡng độ tin cậy và bỏ phiếu 5/7 khung;
  - lớp 2 trong node hành vi: cử chỉ phải giữ liên tục 0.5 s (Open_Palm chỉ 0.15 s); mỗi lần giữ chỉ phát **một** lệnh; hai lệnh cách nhau ít nhất 1 s; đang chạy thì bỏ qua lệnh chọn hành vi mới.
- **Dừng khi có tín hiệu dừng hoặc mất tín hiệu**: Open_Palm dừng ngay; không có `/detected_gesture` quá 1 s thì dừng; Ctrl+C node hành vi cũng gửi lệnh dừng.
- **Không va chạm**: hành vi chạy vòng kín theo `/odom` (dừng chậm dần gần đích). LiDAR kiểm tra quạt ±34° theo hướng chạy, vật cản gần hơn 0.35 m thì dừng (góc robot = góc scan + π/2 vì `rplidar_joint` xoay yaw π/2 trong URDF). Mỗi đoạn chạy có deadline, và robot dừng khi mất `/odom`.
- **Lùi được 0.4 m**: Create 3 mặc định chặn lùi xa (khoảng 0.15 m, vì phía sau không có cảm biến cliff). Node đặt `motion_control safety_override = backup_only` lúc khởi động. Trên robot thật cần cân nhắc vì thao tác này tắt bảo vệ cliff khi lùi.

---

## Kết quả kiểm thử

Sim headless (RTF 0.78), cử chỉ giả lập trên `/detected_gesture`, đối chiếu với pose thật trong Gazebo.

| Kịch bản | Kết quả |
|---|---|
| Thumb_Up khi chưa chọn hành vi | Bỏ qua, có cảnh báo |
| Pointing_Up nhấp nháy 0.2 s × 5 | Không kích hoạt |
| Giữ Thumb_Up 4 s | Chỉ 1 lệnh |
| Tiến–lùi × 4 | Về lại điểm xuất phát, lệch < 1 cm |
| Vòng tròn r = 0.5 m | Khép kín: lệch 8 mm, 1° |
| Open_Palm giữa vòng tròn | Đứng yên sau 0.37–0.46 s; Thumb_Up chạy lại được |
| Wave | Xoay ±35° × 2, về hướng cũ lệch khoảng 1° |
| Mất tín hiệu khi đang chạy | Đứng yên sau 1.26 s |
| Chạy thẳng về phía tường | Dừng do LiDAR thấy vật cản ở 0.35 m, cách tường khoảng 11 cm |
| Va chạm (`/hazard_detection` BUMP) | 0 |
| Webcam thật + 2 node | `/detected_gesture` khoảng 15 Hz; không ai giơ tay thì không có cử chỉ giả; tắt node nhận diện thì robot báo mất tín hiệu và dừng |
| Ảnh mẫu MediaPipe qua bộ lọc | Pointing_Up 0.81, Thumb_Down 0.82, Thumb_Up 0.73, Victory 0.93; ổn định sau 5–6 khung |
| Unit test `GestureFilter` (8 ca) | Đạt |

Ghi chú: trong sim, odom đo dư khoảng 6% góc khi **quay tại chỗ** do trượt bánh (cung tròn và đường thẳng thì chính xác). Hành vi chào gồm các góc trái/phải đối xứng nên sai số triệt tiêu.

---

## Tham số

Sửa trong `config/gesture_control.yaml` (dùng với launch), hoặc truyền `--ros-args -p tên:=giá_trị` khi chạy bằng `ros2 run`.

| Node | Tham số | Mặc định | Ý nghĩa |
|---|---|---|---|
| gesture_recognition_node | `camera_index` | 0 | Số N của `/dev/videoN` |
| | `min_score` | 0.6 | Độ tin cậy tối thiểu |
| | `stable_window` / `stable_count` | 7 / 5 | Bỏ phiếu: cần 5 trong 7 khung |
| | `wave_min_amplitude` / `wave_min_swings` | 0.06 / 2 | Biên độ mỗi nhịp vẫy / số lần đổi chiều |
| | `show_window` | true | Mở cửa sổ webcam |
| robot_behavior_node | `hold_time` / `stop_hold_time` | 0.5 / 0.15 s | Thời gian giữ cử chỉ để thành lệnh |
| | `cooldown` / `signal_timeout` | 1.0 / 1.0 s | Khoảng cách giữa 2 lệnh / ngưỡng mất tín hiệu |
| | `forward_distance` / `forward_back_cycles` | 0.4 m / 4 | Tiến–lùi |
| | `circle_radius` / `circle_speed` | 0.5 m / 0.15 m/s | Vòng tròn |
| | `greet_angle` / `greet_swings` | 0.6 rad / 2 | Chào |
| | `safety_distance` | 0.35 m | Khoảng cách dừng theo LiDAR |
| | `wave_requires_confirm` | false | Wave cũng phải xác nhận bằng Thumb_Up |
| | `disable_backup_limit` | true | Tắt giới hạn lùi của Create 3 |

---

## Xử lý sự cố

| Hiện tượng | Cách xử lý |
|---|---|
| `Package 'tb4_gesture_control' not found` | Chưa source: `source ~/tb4_ws/install/setup.bash`. Nếu vẫn lỗi sau khi build: `package.xml` không được chứa `--` trong comment XML (nếu có, colcon build như package Python thường) |
| `ModuleNotFoundError: mediapipe` | Chạy lại lệnh `pip3 install` ở mục 0 |
| Webcam không mở | Thử `camera_index:=1`; kiểm tra `ls /dev/video*`, `webcam_view` |
| Không có `/scan` khi dùng world riêng | Phải truyền `world:=<tên>` cho `turtlebot4_spawn.launch.py` (vì bridge LiDAR/camera ghép tên world vào topic). `sim.launch.py` đã làm việc này |
| Robot không lùi, log `Reached backup limit!` | Chưa chạy `robot_behavior_node`, hoặc đặt tay: `ros2 param set /motion_control safety_override backup_only` |
| Robot không phản ứng với cử chỉ | Xem `ros2 topic echo /detected_gesture` có ra đúng nhãn không; giữ cử chỉ ít nhất 0.5 s; nếu đang chạy hành vi thì phải xòe tay / nắm tay trước |
| Nhận nhầm khi ánh sáng yếu | Tăng `min_score` (0.7) hoặc `stable_count` (6) |
| Vẫy tay không nhận | Vẫy rộng hơn, hoặc giảm `wave_min_amplitude` (0.04) |
| Robot chọn/xác nhận được nhưng không tiến/lùi, log `vat can 0.16 m` hoặc `LiDAR loi: toan range_min`; camera trong Gazebo màu đen | Gazebo đang render bằng GPU Intel nên LiDAR trả toàn 0.164 m. Tắt sim, rồi chạy lại `ros2 launch tb4_gesture_control sim.launch.py` (mặc định `nvidia:=true`). Kiểm tra: `ros2 topic echo /scan --once` phải có giá trị vài mét |
