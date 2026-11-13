"""
Nhận diện biểu cảm khuôn mặt từ webcam bằng DeepFace và hiển thị kết quả theo thời gian thực.

Luồng xử lý (mỗi khung, timer ~30 Hz):
  webcam (OpenCV V4L2) / file video / thư mục ảnh  -> (webcam: lật gương)
    -> EmotionDetector (DeepFace: YuNet tìm mặt -> căn mặt -> CNN 48x48 -> 7 xác suất)
    -> chọn khuôn mặt LỚN NHẤT (người đứng gần camera nhất là người điều khiển)
    -> EmotionFilter: EMA xác suất theo thời gian, ngưỡng tin cậy + khoảng cách với lớp nhì,
       giữ ổn định stable_time giây, mất mặt > no_face_time -> no_face
    -> publish nhãn ổn định lên /detected_emotion (std_msgs/String) MỖI khung, kể cả no_face / unknown
       => emotion_behavior_node dùng làm nhịp tim: mất nhịp > signal_timeout thì dừng robot
    -> vẽ khung mặt, nhãn, biểu đồ xác suất 7 lớp, bảng biểu cảm -> hành động, trạng thái robot
       (từ /emotion_behavior_state) -> cửa sổ OpenCV và topic /emotion/image_annotated (xem trên RViz)

Dùng:
  ros2 run tb4_emotion_control emotion_recognition_node
  ros2 run tb4_emotion_control emotion_recognition_node --ros-args -p camera_index:=1
  ros2 run tb4_emotion_control emotion_recognition_node --ros-args -p input:=/thư_mục/ảnh   # không cần webcam
Thoát: phím q / Esc trên cửa sổ, hoặc Ctrl+C.
"""

from collections import Counter
import glob
import os
import signal
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import Image
from std_msgs.msg import String

from tb4_emotion_control import emotions as E
from tb4_emotion_control.emotion_detector import EmotionDetector
from tb4_emotion_control.emotion_filter import EmotionFilter, FILTER_PARAMS

WINDOW = 'emotion_recognition_node'
PANEL_W = 290
IMAGE_EXT = ('.jpg', '.jpeg', '.png', '.bmp')


def letterbox(img, w, h):
    """Thu/phóng giữ tỷ lệ vào khung w x h, viền đen."""
    ih, iw = img.shape[:2]
    s = min(w / iw, h / ih)
    nw, nh = max(int(iw * s), 1), max(int(ih * s), 1)
    out = np.zeros((h, w, 3), np.uint8)
    x, y = (w - nw) // 2, (h - nh) // 2
    out[y:y + nh, x:x + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return out


class FrameSource:
    """Webcam (input = ''), file video (lặp lại) hoặc thư mục ảnh (mỗi ảnh image_period giây, lặp lại)."""

    def __init__(self, spec, cam_index, size, image_period, log):
        self.size, self.log = size, log
        self.cap, self.images, self.video = None, None, None
        self.spec = os.path.expanduser(spec) if spec else ''
        self.cam_index = cam_index
        self.image_period = image_period
        self.t0 = time.monotonic()
        self.current = ''
        if not self.spec:
            self.open_camera()
        elif os.path.isdir(self.spec):
            files = sorted(f for f in glob.glob(os.path.join(self.spec, '*')) if f.lower().endswith(IMAGE_EXT))
            if not files:
                raise FileNotFoundError(f'Thư mục {self.spec} không có ảnh')
            self.images = [(os.path.basename(f), letterbox(cv2.imread(f), *size)) for f in files]
            log(f'Đọc {len(files)} ảnh từ {self.spec}, mỗi ảnh {image_period:.1f} s')
        else:
            self.video = self.spec
            self.cap = cv2.VideoCapture(self.spec)
            if not self.cap.isOpened():
                raise FileNotFoundError(f'Không mở được video {self.spec}')
            log(f'Đọc video {self.spec}')

    @property
    def is_camera(self):
        return not self.spec

    def open_camera(self):
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.cam_index, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if self.cap.isOpened():
            self.log(f'Đã mở webcam /dev/video{self.cam_index}')
        else:
            self.log(f'Không mở được webcam /dev/video{self.cam_index}')

    def read(self):
        if self.images is not None:
            i = int((time.monotonic() - self.t0) / self.image_period) % len(self.images)
            self.current, img = self.images[i]
            return True, img.copy()
        ok, frame = self.cap.read()
        if not ok and self.video:                     # hết video -> quay lại đầu
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        if ok and self.video:
            frame = letterbox(frame, *self.size)
        return ok, frame

    def release(self):
        if self.cap is not None:
            self.cap.release()


class EmotionRecognitionNode(Node):

    def __init__(self):
        super().__init__('emotion_recognition_node')
        self.declare_parameter('camera_index', 0)
        self.declare_parameter('frame_width', 640)
        self.declare_parameter('frame_height', 480)
        self.declare_parameter('mirror', True)             # lật gương webcam cho tự nhiên
        self.declare_parameter('input', '')                # '' = webcam | file video | thư mục ảnh
        self.declare_parameter('image_period', 4.0)        # s mỗi ảnh khi input là thư mục
        self.declare_parameter('detector', 'yunet')        # yunet | opencv | centerface | ...
        self.declare_parameter('deepface_home', '~/tb4_ws/models')
        self.declare_parameter('min_face_conf', 0.5)
        self.declare_parameter('min_face_size', 40)        # px, bỏ mặt quá nhỏ / quá xa
        for name, default in FILTER_PARAMS.items():
            self.declare_parameter(name, default)
        self.declare_parameter('rate', 30.0)               # Hz tối đa
        self.declare_parameter('emotion_topic', '/detected_emotion')
        self.declare_parameter('state_topic', '/emotion_behavior_state')
        self.declare_parameter('image_topic', '/emotion/image_annotated')
        self.declare_parameter('publish_image', True)
        self.declare_parameter('show_window', True)

        p = lambda name: self.get_parameter(name).value  # noqa: E731
        self.mirror = p('mirror')
        self.show_window = p('show_window') and bool(os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'))
        if p('show_window') and not self.show_window:
            self.get_logger().warn('Không có màn hình (DISPLAY), tắt cửa sổ; xem ảnh trên ' + p('image_topic'))
        self.publish_image = p('publish_image')

        t0 = time.monotonic()
        self.detector = EmotionDetector(p('detector'), p('deepface_home'), p('min_face_conf'),
                                        p('min_face_size'), log=self.get_logger().warn)
        self.get_logger().info(f'DeepFace (onnx) + detector {self.detector.detector}: nạp '
                               f'{time.monotonic() - t0:.1f} s')
        self.efilter = EmotionFilter(**{name: p(name) for name in FILTER_PARAMS})
        self.source = FrameSource(p('input'), p('camera_index'), (p('frame_width'), p('frame_height')),
                                  p('image_period'), self.get_logger().info)

        self.pub = self.create_publisher(String, p('emotion_topic'), 10)
        self.image_pub = self.create_publisher(Image, p('image_topic'), 2)
        self.create_subscription(String, p('state_topic'), self.on_state, 10)

        self.robot_state = '(emotion_behavior_node chua chay)'
        self.published = E.NO_FACE
        self.counts = Counter()      # số lần mỗi biểu cảm ổn định xuất hiện (theo sườn lên)
        self.fail_count, self.next_reopen = 0, 0.0
        self.fps, self.infer_ms, self.last_step = 0.0, 0.0, None
        self.create_timer(1.0 / p('rate'), self.step)
        self.get_logger().info(f'Publish {p("emotion_topic")} | q/Esc để thoát')

    def on_state(self, msg):
        self.robot_state = msg.data

    # ---------- vòng xử lý ----------
    def step(self):
        ok, frame = self.source.read()
        now = time.monotonic()
        if not ok:
            # Không publish: emotion_behavior_node thấy mất tín hiệu và dừng robot
            self.fail_count += 1
            if self.fail_count == 15:
                self.get_logger().error('Mất hình từ webcam, ngừng publish; sẽ thử mở lại')
            if self.fail_count >= 15 and self.source.is_camera and now >= self.next_reopen:
                self.next_reopen = now + 2.0
                self.source.open_camera()
            return
        if self.fail_count >= 15:
            self.get_logger().info('Webcam hoạt động trở lại')
        self.fail_count = 0
        if self.mirror and self.source.is_camera:
            frame = cv2.flip(frame, 1)

        t = time.perf_counter()
        try:
            faces = self.detector.analyze(frame)
        except Exception as e:  # noqa: BLE001  lỗi một khung không được làm chết node
            self.get_logger().error(f'Lỗi DeepFace: {e}', throttle_duration_sec=5.0)
            faces = []
        ms = (time.perf_counter() - t) * 1000
        self.infer_ms = ms if not self.infer_ms else 0.9 * self.infer_ms + 0.1 * ms

        main = faces[0] if faces else None
        stable, conf, smooth = self.efilter.update(main.probs if main else None, now)
        self.pub.publish(String(data=stable))
        if stable != self.published:
            if stable in E.MODEL_LABELS:
                self.counts[stable] += 1
            src = f' [{self.source.current}]' if self.source.current else ''
            self.get_logger().info(f'Biểu cảm: {stable} ({E.VIETNAMESE[stable]}) {conf * 100:.0f}%'
                                   f' -> {E.ACTIONS.get(stable, "khong dieu khien")}{src}')
            self.published = stable

        if self.last_step is not None and now > self.last_step:
            inst = 1.0 / (now - self.last_step)
            self.fps = 0.9 * self.fps + 0.1 * inst if self.fps else inst
        self.last_step = now

        if not (self.show_window or self.publish_image):
            return
        canvas = self.draw(frame, faces, stable, conf, smooth)
        if self.publish_image and self.image_pub.get_subscription_count() > 0:
            msg = Image(height=canvas.shape[0], width=canvas.shape[1], encoding='bgr8',
                        step=canvas.shape[1] * 3, data=canvas.tobytes())
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'webcam'
            self.image_pub.publish(msg)
        if self.show_window:
            cv2.imshow(WINDOW, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                raise KeyboardInterrupt

    # ---------- hiển thị ----------
    def draw(self, frame, faces, stable, conf, smooth):
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i, f in enumerate(faces):
            x, y, bw, bh = f.box
            top = max(f.probs, key=f.probs.get)
            color = E.COLORS[top] if i == 0 else (150, 150, 150)
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 3 if i == 0 else 1)
            label = f'{top} {f.probs[top] * 100:.0f}%'
            (tw, th), _ = cv2.getTextSize(label, font, 0.6, 2)
            ty = y - 8 if y - th - 12 > 0 else y + bh + th + 8
            cv2.rectangle(frame, (x, ty - th - 6), (x + tw + 8, ty + 6), color, -1)
            cv2.putText(frame, label, (x + 4, ty), font, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

        # thanh trên: nhãn ổn định -> hành động
        color = E.COLORS.get(stable, (200, 200, 200))
        cv2.rectangle(frame, (0, 0), (w, 46), (30, 30, 30), -1)
        title = stable.upper() + (f' {conf * 100:.0f}%' if stable in E.MODEL_LABELS else '')
        cv2.putText(frame, title, (10, 33), font, 1.0, color, 2, cv2.LINE_AA)
        if stable in E.ACTIONS:
            cv2.putText(frame, '-> ' + E.ACTIONS[stable], (w - 250, 31), font, 0.65, color, 2, cv2.LINE_AA)
        # thanh dưới: trạng thái robot
        cv2.rectangle(frame, (0, h - 28), (w, h), (30, 30, 30), -1)
        cv2.putText(frame, 'ROBOT: ' + self.robot_state[:80], (8, h - 9), font, 0.48, (0, 255, 255), 1,
                    cv2.LINE_AA)

        # bảng bên phải: xác suất (EMA = thanh đặc, khung hiện tại = vạch trắng), bảng hành động
        panel = np.full((h, PANEL_W, 3), 40, np.uint8)
        cv2.putText(panel, 'probabilities (smoothed)', (10, 24), font, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
        raw = faces[0].probs if faces else None
        y = 44
        for lb in E.MODEL_LABELS:
            p = smooth.get(lb, 0.0) if smooth else 0.0
            c = E.COLORS[lb]
            cv2.putText(panel, lb, (10, y + 13), font, 0.45, c if lb == stable else (220, 220, 220), 1, cv2.LINE_AA)
            x0, x1 = 85, PANEL_W - 50
            cv2.rectangle(panel, (x0, y + 2), (x1, y + 16), (70, 70, 70), -1)
            cv2.rectangle(panel, (x0, y + 2), (x0 + int((x1 - x0) * p), y + 16), c, -1)
            if raw:
                rx = x0 + int((x1 - x0) * raw.get(lb, 0.0))
                cv2.line(panel, (rx, y), (rx, y + 18), (255, 255, 255), 2)
            cv2.putText(panel, f'{p * 100:3.0f}%', (x1 + 6, y + 14), font, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
            y += 24
        y += 14
        cv2.putText(panel, 'emotion -> robot action', (10, y), font, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
        y += 8
        for lb in E.CONTROL:
            y += 22
            active = lb == stable
            c = E.COLORS[lb] if active else (200, 200, 200)
            if active:
                cv2.rectangle(panel, (4, y - 15), (PANEL_W - 4, y + 6), (70, 70, 70), -1)
            cv2.putText(panel, f'{lb:9s}', (10, y), font, 0.45, c, 1, cv2.LINE_AA)
            cv2.putText(panel, E.ACTIONS[lb], (100, y), font, 0.45, c, 1, cv2.LINE_AA)
        info = f'{len(faces)} face | {self.infer_ms:.0f} ms | {self.fps:.1f} fps'
        cv2.putText(panel, info, (10, h - 12), font, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
        if self.source.current:
            cv2.putText(panel, self.source.current[:34], (10, h - 34), font, 0.42, (180, 180, 180), 1,
                        cv2.LINE_AA)
        return np.hstack([frame, panel])

    def close(self):
        self.source.release()

    def summary(self):
        if not self.counts:
            return '  (chưa nhận được biểu cảm nào)'
        return '\n'.join(f'  {lb:9s} {n} lần' for lb, n in self.counts.most_common())


def _interrupt(*_):
    raise KeyboardInterrupt


def main():
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    signal.signal(signal.SIGINT, _interrupt)
    signal.signal(signal.SIGTERM, _interrupt)
    node = None
    try:
        node = EmotionRecognitionNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if node is not None:
            print('\n[emotion_recognition_node] TỔNG KẾT số lần nhận diện\n' + node.summary())
            node.close()
            node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
