# tb4_emotion_control: điều khiển TurtleBot 4 bằng biểu cảm khuôn mặt

Webcam laptop → DeepFace (phát hiện mặt YuNet + CNN cảm xúc FER-2013) → `/detected_emotion` → quỹ đạo tương ứng → `/cmd_vel` → TurtleBot 4 trong Gazebo.

```
 webcam ──▶ emotion_recognition_node ──/detected_emotion (String, ~30 Hz)──▶ emotion_behavior_node ──/cmd_vel──▶ TurtleBot 4
               (OpenCV + DeepFace,                                           (biểu cảm -> quỹ đạo, vòng kín
                lọc nhiễu theo thời gian)  ◀──/emotion_behavior_state──        theo /odom, kiểm tra /scan)
               └─▶ cửa sổ OpenCV + /emotion/image_annotated
```

| Biểu cảm | Hành động của TurtleBot | Cách chạy |
|----------|-------------------------|-----------|
| Happy (Vui vẻ) | Quỹ đạo cong sang **trái**, v = 0.15 m/s, R = 0.5 m | liên tục, chừng nào còn giữ biểu cảm |
| Sad (Buồn) | Quỹ đạo cong sang **phải**, v = 0.15 m/s, R = 0.5 m | liên tục, chừng nào còn giữ biểu cảm |
| Neutral (Bình thường) | **Dừng tại chỗ**, hủy mọi chuyển động | ngay khi thấy (0.3 s) |
| Angry (Tức giận) | Lùi nhẹ 0.2 m | một lần |
| Surprise (Ngạc nhiên) | Xoay một vòng 360° tại chỗ | một lần |
| Fear (Sợ hãi) | Tiến nhẹ 0.25 m | một lần |

Mục lục:
- [0. Chuẩn bị (làm một lần)](#0-chuẩn-bị-làm-một-lần)
- [1. Chạy](#1-chạy)
- [2. Thử không cần webcam](#2-thử-không-cần-webcam)
- [Thuật toán](#thuật-toán)
- [Kết quả kiểm thử](#kết-quả-kiểm-thử)
- [Tham số](#tham-số)
- [Xử lý sự cố](#xử-lý-sự-cố)

---

## 0. Chuẩn bị (làm một lần)

```bash
# DeepFace bản mới chạy được bằng onnxruntime, KHÔNG cần TensorFlow.
# --no-deps: tránh kéo opencv-contrib-python-headless (ghi đè cv2 -> cv2.imshow hỏng) và retina-face (kéo TensorFlow)
pip3 install --user --no-deps "deepface==0.0.105"
pip3 install --user gdown pandas onnxruntime "numpy<2"

# Build (package dùng lại sân gesture_arena + prepare_robot của tb4_gesture_control)
cd ~/tb4_ws
colcon build --symlink-install --packages-select tb4_gesture_control tb4_emotion_control
source ~/tb4_ws/install/setup.bash
```

Kiểm tra:

```bash
python3 -c "import deepface, cv2, numpy; print(deepface.__version__, cv2.__version__, numpy.__version__)"
ros2 pkg executables tb4_emotion_control      # emotion_behavior_node, emotion_recognition_node
```

Trọng số (YuNet 0.2 MB, CNN cảm xúc 6 MB) tự tải về `~/tb4_ws/models/.deepface/weights/` ở lần chạy đầu
(cần mạng tới github.com). Trên máy này đã tải sẵn.

---

## 1. Chạy

Mỗi dòng một terminal, theo thứ tự (nhớ `source ~/tb4_ws/install/setup.bash`):

| Terminal | Việc | Lệnh |
|----------|------|------|
| 1 | Gazebo + robot, tự undock và tiến ra giữa sân | `ros2 launch tb4_emotion_control sim.launch.py` |
| 2 | Nhận diện biểu cảm + điều khiển robot | `ros2 launch tb4_emotion_control emotion_control.launch.py` |

Đợi terminal 1 in `Sẵn sàng tại odom (...)` (khoảng 30 s) rồi bật terminal 2. Cửa sổ `emotion_recognition_node` hiện:
- khung mặt (màu theo biểu cảm) và nhãn của khung hiện tại;
- thanh trên: biểu cảm **ổn định** (sau lọc nhiễu) và hành động tương ứng;
- bảng bên phải: xác suất 7 lớp (thanh màu = đã làm mượt, vạch trắng = khung hiện tại), bảng biểu cảm → hành động;
- thanh dưới: trạng thái robot (`IDLE`, `RUNNING CURVE_LEFT ...`, `SAFETY STOP ...`).

Biến thể:

```bash
ros2 launch tb4_emotion_control emotion_control.launch.py camera_index:=1     # webcam ngoài /dev/video1
ros2 launch tb4_emotion_control sim.launch.py rviz:=true                       # RViz: thêm Image /emotion/image_annotated
```

Mẹo khi làm biểu cảm: ngồi đủ sáng, mặt nhìn thẳng camera, **làm rõ / hơi cường điệu** biểu cảm (Sad: cụp mắt, trễ khóe miệng;
Surprise: mở to mắt và miệng; Fear: mắt mở to, miệng kéo ngang; Angry: nhíu mày, mím môi). Giữ biểu cảm ~1 s để lệnh được nhận.
Muốn robot dừng: trở về mặt bình thường hoặc rời khỏi camera.

Dừng: phím `q` trên cửa sổ hoặc Ctrl+C ở terminal 2 (robot nhận lệnh dừng trước khi node tắt), rồi Ctrl+C terminal 1.

---

## 2. Thử không cần webcam

```bash
# a) Chỉ phần điều khiển: giả lập biểu cảm bằng tay (robot cong trái khi đang publish happy)
ros2 run tb4_emotion_control emotion_behavior_node
ros2 topic pub -r 30 /detected_emotion std_msgs/msg/String "data: happy"     # Ctrl+C -> mất tín hiệu -> robot dừng

# b) Nhận diện từ thư mục ảnh (mỗi ảnh 4 s, lặp lại) hoặc file video thay cho webcam
ros2 launch tb4_emotion_control emotion_control.launch.py input:=/thư_mục/ảnh
ros2 launch tb4_emotion_control emotion_control.launch.py input:=/đường/dẫn/video.mp4

# c) Kiểm thử đơn vị (bộ lọc + quỹ đạo, không cần ROS / DeepFace)
cd ~/tb4_ws && python3 -m pytest src/tb4_emotion_control/test -q
```

---

## Thuật toán

### Nhận diện ([emotion_recognition_node.py](tb4_emotion_control/emotion_recognition_node.py), [emotion_detector.py](tb4_emotion_control/emotion_detector.py))

Mỗi khung (timer 30 Hz):
1. Đọc webcam (V4L2, 640×480), lật gương. Webcam mất hình 15 khung liên tiếp → ngừng publish (robot tự dừng) và thử mở lại mỗi 2 s.
2. `DeepFace.analyze(actions=['emotion'], detector_backend='yunet')`:
   YuNet (ONNX) tìm mặt → căn thẳng theo hai mắt → ảnh xám 48×48 → CNN cảm xúc → 7 xác suất
   (angry, disgust, fear, happy, sad, surprise, neutral). Ép backend `onnx` (`DEEPFACE_BACKEND_ENGINE`) nên không cần TensorFlow.
3. Bỏ "mặt giả": khi không có mặt DeepFace vẫn trả kết quả cho cả khung với `face_confidence = 0` → loại mặt có
   độ tin cậy < 0.5 hoặc nhỏ hơn 40 px. Nhiều mặt → chọn mặt **lớn nhất** (người gần camera nhất).
4. **Lọc theo thời gian** ([emotion_filter.py](tb4_emotion_control/emotion_filter.py)), vì mô hình FER-2013 dao động mạnh giữa các khung:
   - EMA véc-tơ xác suất theo thời gian: `alpha = 1 − exp(−dt / tau)`, tau = 0.3 s (không phụ thuộc FPS);
   - ứng viên = lớp lớn nhất nếu `p ≥ 0.45` và hơn lớp nhì `≥ 0.10`, ngược lại `unknown`;
   - ứng viên phải giữ nguyên `stable_time = 0.4 s` mới thành nhãn ổn định;
   - mất mặt < 0.5 s: giữ nhãn cũ; lâu hơn: `no_face` và xóa EMA.
5. Publish nhãn ổn định lên `/detected_emotion` **mỗi khung** (kể cả `no_face`, `unknown`) → làm nhịp tim cho node điều khiển.
6. Vẽ kết quả → cửa sổ OpenCV và `/emotion/image_annotated`.

### Điều khiển ([emotion_behavior_node.py](tb4_emotion_control/emotion_behavior_node.py), [motions.py](tb4_emotion_control/motions.py))

Vòng điều khiển 20 Hz:
- **Ra lệnh**: biểu cảm phải giữ `hold_time = 0.3 s`. Biểu cảm điều khiển mới khác hành vi đang chạy → thay thế ngay.
  Hành vi một lần (angry / surprise / fear) chỉ chạy một lần cho mỗi lần giữ biểu cảm; muốn lặp lại thì đổi biểu cảm rồi làm lại.
- **Dừng**: `neutral` (hoặc `no_face`) giữ 0.3 s → dừng tại chỗ, hủy mọi hành vi. Quỹ đạo cong còn dừng khi biểu cảm
  biến mất (`unknown` / `disgust`) quá `release_time = 0.5 s`.
- **Quỹ đạo vòng kín theo /odom**:
  - cong (happy / sad): `v = 0.15 m/s`, `w = ±v / R` (R = 0.5 m) → cung tròn bán kính 0.5 m, chạy liên tục;
  - thẳng (angry / fear): dừng khi quãng đường odom ≥ mục tiêu;
  - xoay (surprise): góc **tích lũy** `Σ wrap(yaw_k − yaw_{k−1})` ≥ 2π (đi qua ±π không bị lỗi);
  - gần đích chạy chậm dần `k = clamp(còn_lại / vùng, 0.3, 1)` (vùng 0.08 m / 0.3 rad) để ít vọt lố.
- **An toàn**: mất `/detected_emotion` > 1 s → dừng; LiDAR thấy vật cản < 0.35 m trong quạt ±0.6 rad theo hướng chạy
  (góc robot = góc scan + π/2) → dừng; mất `/odom` > 0.5 s; một đoạn chạy quá 3× thời gian dự kiến + 5 s.
  Lúc khởi động đặt `motion_control safety_override = backup_only` vì Create 3 mặc định chỉ cho lùi ~0.15 m.
  Khi tắt node (Ctrl+C / SIGTERM) gửi `Twist()` 0 trước khi đóng context ROS.

---

## Kết quả kiểm thử

Kiểm thử ngày 2026-11-13 trên Gazebo Fortress headless (sân `gesture_arena`, RTF ~0.8, GPU RTX 3050), DeepFace 0.0.105 (ONNX, CPU).
Chuyển động thật đo bằng `/sim_ground_truth_pose` của Gazebo, quy về hệ tọa độ robot lúc bắt đầu mỗi pha.

**Kiểm thử đơn vị**: `pytest src/tb4_emotion_control/test` → 17/17 đạt (bộ lọc: ổn định sau 0.4 s, không nhảy khi nhiễu 1 khung,
mất mặt chớp nhoáng giữ nhãn, mất lâu → `no_face`; quỹ đạo: cong đúng chiều, R = 0.5 m, lùi / tiến / xoay đúng mục tiêu).

**Nhận diện** (ảnh khuôn mặt mẫu đưa vào qua `input:=` thay webcam, khung 640×480): 30 khung/s, DeepFace 21–27 ms/khung.

| Ảnh | Nhãn ổn định | Độ tin cậy |
|-----|--------------|-----------:|
| mặt bình thường | neutral | 94% |
| cười | happy | 88–100% |
| buồn | sad | 51–52% |
| nhíu mày | angry | 49–51% |
| há miệng, mở to mắt | surprise | 87–98% |
| sợ hãi (ảnh nhiều mặt, chọn mặt lớn nhất) | fear | 83–92% |
| không có mặt | no_face | – |
| mặt mơ hồ (sad 35% / neutral 34%) | unknown (không điều khiển) | – |

**End-to-end** (ảnh → DeepFace → bộ lọc → `emotion_behavior_node` → Gazebo, mỗi ảnh 9 s):

| Nhãn | Tiến (m) | Sang trái (m) | Quay (°) | Hành vi |
|------|---------:|--------------:|---------:|---------|
| neutral | 0.000 | 0.000 | 0.0 | đứng yên |
| happy | 0.470 | 0.760 | +115.4 | CURVE_LEFT |
| sad | 0.617 | −0.624 | −100.4 | CURVE_RIGHT |
| angry | −0.194 | 0.000 | 0.0 | BACK_UP |
| surprise | 0.000 | 0.000 | 360.0 | SPIN |
| fear | 0.249 | 0.000 | 0.0 | FORWARD |
| no_face | 0.000 | 0.000 | 0.0 | đứng yên |

**Quy tắc điều khiển** (phát nhãn trực tiếp lên `/detected_emotion`, đo bằng ground truth):

| Tình huống | Kết quả |
|------------|---------|
| happy / sad giữ 6 s | cung tròn bán kính đo được 0.52 m (mục tiêu 0.5 m), trái / phải đúng chiều |
| neutral khi đang cong | dừng hẳn sau 0.59 s (0.3 s giữ + giảm tốc), trôi thêm 5 cm |
| biểu cảm biến mất (unknown) khi đang cong | dừng sau 0.75 s (`release_time` 0.5 s) |
| angry / fear / surprise | lùi 0.199 m, tiến 0.249 m, quay 360.0° |
| neutral khi đang xoay (sau 1.8 s) | hủy xoay, dừng tại chỗ |
| happy → sad → fear liên tiếp | mỗi biểu cảm mới thay thế ngay hành vi cũ |
| giữ fear sau khi đã tiến xong | không lặp lại (một lần cho mỗi lần giữ) |
| ngừng publish `/detected_emotion` | dừng sau 1.2 s ("MẤT TÍN HIỆU") |
| Ctrl+C node điều khiển | gửi lệnh dừng, vận tốc thật về 0 |

Lưu ý khi đo: odom bánh xe của Gazebo báo **dư ~6% khi quay tại chỗ** (odom 360° = thật 339.6°), nên
`config/emotion_control.yaml` đặt `spin_odom_gain: 1.06`. Robot thật (odom có IMU) dùng `1.0`. Quỹ đạo cong và thẳng không bị lệch này.

Chưa kiểm thử: biểu cảm thật của người trước webcam (đã kiểm tra webcam `/dev/video0` mở được 640×480).
Ảnh mẫu dùng để kiểm thử (ảnh trong bộ test của DeepFace và ảnh Wikimedia Commons) không đưa vào package.

---

## Tham số

Đầy đủ trong [config/emotion_control.yaml](config/emotion_control.yaml). Hay chỉnh:

| Tham số | Mặc định | Ý nghĩa |
|---------|----------|---------|
| `detector` | `yunet` | `opencv` (Haar cascade, không cần tải trọng số, kém hơn) |
| `min_conf`, `min_margin` | 0.45, 0.10 | hạ nếu biểu cảm khó (sad, fear) hay ra `unknown` |
| `stable_time`, `tau` | 0.4, 0.3 s | tăng nếu nhãn nhảy lung tung, giảm để phản hồi nhanh hơn |
| `hold_time`, `stop_hold_time` | 0.3 s | thời gian giữ biểu cảm để thành lệnh / để dừng |
| `curve_speed`, `curve_radius` | 0.15 m/s, 0.5 m | quỹ đạo cong |
| `back_distance`, `forward_distance` | 0.2, 0.25 m | lùi / tiến nhẹ |
| `spin_speed`, `spin_angle_deg` | 1.2 rad/s, 360° | xoay |
| `stop_on_no_face` | true | false: rời camera không làm dừng quỹ đạo |

---

## Xử lý sự cố

| Hiện tượng | Nguyên nhân / cách sửa |
|------------|------------------------|
| `ModuleNotFoundError: deepface` | chưa cài, xem mục 0 |
| `cv2.imshow ... not implemented` | lỡ cài `opencv-*-headless`: `pip3 uninstall opencv-contrib-python-headless opencv-python-headless` rồi `pip3 install --user --force-reinstall --no-deps opencv-contrib-python==4.9.0.80` |
| Lần đầu chạy treo/lỗi tải trọng số | cần mạng tới github.com; hoặc chạy `-p detector:=opencv` (Haar có sẵn, vẫn cần trọng số cảm xúc 6 MB) |
| Sad luôn ra `neutral` / `unknown` | điểm yếu đã biết của mô hình FER-2013 (sad dễ nhầm neutral/fear): làm biểu cảm rõ hơn, đủ sáng, hoặc hạ `min_margin:=0.05` |
| Robot không lùi đủ 0.2 m (`Reached backup limit`) | service `motion_control/set_parameters` chưa sẵn sàng; node tự thử lại mỗi 1 s |
| Robot không tiến/lùi, log "vat can 0.16 m" / "LiDAR loi" | Gazebo đang render bằng GPU Intel: chạy `sim.launch.py nvidia:=true` |
| Cửa sổ ghi "NO EMOTION SIGNAL" | `emotion_recognition_node` không chạy / webcam mất hình |
