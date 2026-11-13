"""
Nhận diện cử chỉ tay từ webcam laptop bằng MediaPipe Gesture Recognizer.

Luồng xử lý (mỗi khung, timer ~30 Hz):
  webcam (OpenCV VideoCapture) -> lật gương -> MediaPipe GestureRecognizer (chế độ VIDEO)
    -> nhãn thô + độ tin cậy + 21 landmark bàn tay
    -> lọc nhiễu:
         * độ tin cậy < min_score              -> coi như None
         * Open_Palm khi bàn tay đang di chuyển -> coi như None (tránh vẫy tay bị hiểu thành "dừng")
         * bỏ phiếu đa số trên stable_window khung gần nhất, cần >= stable_count phiếu
    -> phát hiện Wave: tâm bàn tay đổi chiều trái/phải >= wave_min_swings lần trong wave_window giây
    -> publish nhãn ổn định lên /detected_gesture (std_msgs/String) MỖI khung,
       kể cả "None" => robot_behavior_node dùng làm nhịp tim, mất nhịp thì dừng robot
    -> vẽ landmark, tên cử chỉ, độ tin cậy, trạng thái robot (/robot_behavior_state) lên cửa sổ

Cử chỉ nhận được (8): Wave, Pointing_Up, Victory, Thumb_Up, Open_Palm, Closed_Fist, Thumb_Down, ILoveYou.

Dùng:
  ros2 run tb4_gesture_control gesture_recognition_node
  ros2 run tb4_gesture_control gesture_recognition_node --ros-args -p camera_index:=1 -p min_score:=0.7
Thoát: phím q / Esc trên cửa sổ webcam, hoặc Ctrl+C.
"""

from collections import Counter, deque
import os
import signal
import time

from ament_index_python.packages import get_package_share_directory
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import String

from tb4_gesture_control import gestures as G

WINDOW = 'gesture_recognition_node'
HAND_CONNECTIONS = [(c.start, c.end) for c in vision.HandLandmarksConnections.HAND_CONNECTIONS]
COLORS = {  # BGR
    G.WAVE: (255, 160, 0), G.POINTING_UP: (0, 200, 255), G.VICTORY: (255, 0, 200),
    G.THUMB_UP: (0, 220, 0), G.OPEN_PALM: (0, 0, 255), G.CLOSED_FIST: (0, 120, 255),
}


def default_model_path():
    return os.path.join(get_package_share_directory('tb4_gesture_control'), 'models',
                        'gesture_recognizer.task')


class WaveDetector:
    """Đếm số lần tâm bàn tay đổi chiều trái/phải (zigzag có ngưỡng trễ) trong cửa sổ thời gian."""

    def __init__(self, window, min_amplitude, min_swings):
        self.window = window
        self.amp = min_amplitude
        self.min_swings = min_swings
        self.samples = deque()  # (t, x chuẩn hóa 0..1, độ tin cậy Open_Palm của khung)

    def reset(self):
        self.samples.clear()

    def update(self, t, x, score):
        self.samples.append((t, x, score))
        while self.samples and t - self.samples[0][0] > self.window:
            self.samples.popleft()

    def swings(self):
        xs = [x for _, x, _ in self.samples]
        if len(xs) < 3:
            return 0
        lo = hi = xs[0]
        direction, ext, swings = 0, xs[0], 0
        for x in xs[1:]:
            if direction == 0:                  # chưa rõ chiều: chờ lệch đủ ngưỡng so với min/max
                lo, hi = min(lo, x), max(hi, x)
                if x - lo >= self.amp:
                    direction, ext = 1, x
                elif hi - x >= self.amp:
                    direction, ext = -1, x
            elif direction == 1:
                if x > ext:
                    ext = x
                elif ext - x >= self.amp:
                    direction, ext, swings = -1, x, swings + 1
            else:
                if x < ext:
                    ext = x
                elif x - ext >= self.amp:
                    direction, ext, swings = 1, x, swings + 1
        return swings

    def detected(self):
        return self.swings() >= self.min_swings

    def confidence(self):
        """Trung bình độ tin cậy Open_Palm trong cửa sổ (khung mờ do chuyển động tính 0)."""
        return sum(s for _, _, s in self.samples) / max(len(self.samples), 1)

    def spread(self, t, horizon):
        """Biên độ dịch chuyển x (max - min) trong horizon giây gần nhất."""
        xs = [x for ts, x, _ in self.samples if t - ts <= horizon]
        return max(xs) - min(xs) if len(xs) >= 2 else 0.0


# Tham số lọc nhiễu (tên = tên tham số ROS của node) và giá trị mặc định
FILTER_PARAMS = {
    'min_score': 0.6,             # độ tin cậy tối thiểu của nhãn MediaPipe
    'stable_window': 7,           # số khung bỏ phiếu
    'stable_count': 5,            # số phiếu tối thiểu để nhãn được coi là ổn định
    'wave_window': 1.5,           # s
    'wave_min_amplitude': 0.06,   # biên độ tối thiểu mỗi nhịp vẫy, theo bề rộng khung
    'wave_min_swings': 2,         # số lần đổi chiều (2 = trái-phải-trái)
    'wave_hold': 0.5,             # s giữ nhãn Wave sau lần phát hiện cuối (nối các nhịp vẫy chậm)
    'still_threshold': 0.05,      # Open_Palm chỉ hợp lệ khi tay dịch < ngưỡng này ...
    'still_horizon': 0.5,         # ... trong chừng ấy giây gần nhất
    'post_wave_palm_block': 0.5,  # s bỏ qua Open_Palm sau khi hết vẫy
}


class GestureFilter:
    """Lọc nhiễu nhãn thô từng khung của MediaPipe và phát hiện Wave. Không phụ thuộc ROS."""

    def __init__(self, **kw):
        cfg = {**FILTER_PARAMS, **kw}
        self.min_score = cfg['min_score']
        self.stable_count = cfg['stable_count']
        self.wave_hold = cfg['wave_hold']
        self.still_threshold = cfg['still_threshold']
        self.still_horizon = cfg['still_horizon']
        self.post_wave_palm_block = cfg['post_wave_palm_block']
        self.history = deque(maxlen=cfg['stable_window'])
        self.wave = WaveDetector(cfg['wave_window'], cfg['wave_min_amplitude'], cfg['wave_min_swings'])
        self.wave_until = 0.0
        self.palm_block_until = 0.0

    def update(self, raw, score, cx, now):
        """raw/score: nhãn và độ tin cậy của khung; cx: x tâm bàn tay (None nếu không thấy tay).

        Trả về (nhãn ổn định để publish, độ tin cậy của nó).
        """
        if cx is None:
            self.wave.reset()
        elif raw in (G.OPEN_PALM, G.NONE):
            self.wave.update(now, cx, score if raw == G.OPEN_PALM else 0.0)
        else:                                 # đang làm cử chỉ khác -> không phải vẫy tay
            self.wave.reset()

        # Wave: đủ số nhịp trong cửa sổ VÀ tay vẫn đang chuyển động (dừng vẫy là hết Wave ngay)
        if self.wave.detected() and self.wave.spread(now, self.still_horizon) > self.still_threshold:
            self.wave_until = now + self.wave_hold
            # tay vừa vẫy xong thường dừng lại vẫn xòe -> đừng hiểu nhầm thành Open_Palm (dừng)
            self.palm_block_until = self.wave_until + self.post_wave_palm_block
        if now < self.wave_until:
            self.history.clear()              # sau khi vẫy xong phải bỏ phiếu lại từ đầu
            return G.WAVE, self.wave.confidence()

        label = raw if score >= self.min_score else G.NONE
        if label == G.OPEN_PALM and (now < self.palm_block_until or
                                     self.wave.spread(now, self.still_horizon) > self.still_threshold):
            label = G.NONE                    # tay xòe nhưng đang/vừa chuyển động: là vẫy, không phải dừng
        self.history.append((label, score))
        best, votes = Counter(lb for lb, _ in self.history).most_common(1)[0]
        if best == G.NONE or votes < self.stable_count:
            return G.NONE, 0.0
        scores = [s for lb, s in self.history if lb == best]
        return best, sum(scores) / len(scores)


class GestureRecognitionNode(Node):

    def __init__(self):
        super().__init__('gesture_recognition_node')
        self.declare_parameter('camera_index', 0)
        self.declare_parameter('frame_width', 640)
        self.declare_parameter('frame_height', 480)
        self.declare_parameter('mirror', True)            # lật gương cho tự nhiên khi nhìn màn hình
        self.declare_parameter('model_path', '')          # '' = models/gesture_recognizer.task trong package
        self.declare_parameter('num_hands', 1)
        for name, default in FILTER_PARAMS.items():
            self.declare_parameter(name, default)
        self.declare_parameter('gesture_topic', '/detected_gesture')
        self.declare_parameter('state_topic', '/robot_behavior_state')
        self.declare_parameter('show_window', True)

        p = lambda name: self.get_parameter(name).value  # noqa: E731
        self.mirror = p('mirror')
        self.show_window = p('show_window')
        self.cam_index = p('camera_index')
        self.cam_size = (p('frame_width'), p('frame_height'))

        model = os.path.expanduser(p('model_path') or default_model_path())
        if not os.path.isfile(model):
            raise FileNotFoundError(f'Không thấy mô hình MediaPipe: {model}')
        self.recognizer = vision.GestureRecognizer.create_from_options(
            vision.GestureRecognizerOptions(
                base_options=BaseOptions(model_asset_path=model),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=p('num_hands')))

        self.gfilter = GestureFilter(**{name: p(name) for name in FILTER_PARAMS})
        self.published = G.NONE
        self.robot_state = '(robot_behavior_node chưa chạy)'
        self.t0 = time.monotonic()
        self.last_ts_ms = -1
        self.last_step = None
        self.fps = 0.0
        self.counts = Counter()   # số lần mỗi cử chỉ ổn định xuất hiện (tính theo sườn lên)

        self.pub = self.create_publisher(String, p('gesture_topic'), 10)
        self.create_subscription(String, p('state_topic'), self.on_state, 10)

        self.cap = None
        self.fail_count = 0
        self.next_reopen = 0.0
        self.open_camera()
        self.create_timer(1.0 / 30.0, self.step)
        self.get_logger().info(f'Mô hình: {model} | publish {p("gesture_topic")} | q/Esc để thoát')

    # ---------- camera ----------
    def open_camera(self):
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.cam_index, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_size[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_size[1])
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if self.cap.isOpened():
            self.get_logger().info(f'Đã mở webcam /dev/video{self.cam_index}')
        else:
            self.get_logger().error(f'Không mở được webcam /dev/video{self.cam_index}')

    def on_state(self, msg):
        self.robot_state = msg.data

    # ---------- vòng xử lý ----------
    def step(self):
        ok, frame = self.cap.read() if self.cap is not None else (False, None)
        now = time.monotonic()
        if not ok:
            # Không publish gì: robot_behavior_node sẽ thấy mất tín hiệu và dừng robot
            self.fail_count += 1
            if self.fail_count == 15:
                self.get_logger().error('Mất hình từ webcam, ngừng publish; sẽ thử mở lại')
            if self.fail_count >= 15 and now >= self.next_reopen:
                self.next_reopen = now + 2.0
                self.open_camera()
            return
        if self.fail_count >= 15:
            self.get_logger().info('Webcam hoạt động trở lại')
        self.fail_count = 0

        if self.mirror:
            frame = cv2.flip(frame, 1)
        raw, score, landmarks, hand = self.recognize(frame, now)
        cx = sum(lm.x for lm in landmarks) / len(landmarks) if landmarks else None
        stable, conf = self.gfilter.update(raw, score, cx, now)

        self.pub.publish(String(data=stable))
        if stable != self.published:
            if stable != G.NONE:
                self.counts[stable] += 1
                self.get_logger().info(f'Cử chỉ: {stable} ({conf:.2f})')
            self.published = stable

        if self.last_step is not None and now > self.last_step:
            inst = 1.0 / (now - self.last_step)
            self.fps = 0.9 * self.fps + 0.1 * inst if self.fps else inst
        self.last_step = now
        if self.show_window:
            self.draw(frame, raw, score, landmarks, hand, stable, conf)
            cv2.imshow(WINDOW, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                raise KeyboardInterrupt

    def recognize(self, frame, now):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ts_ms = max(int((now - self.t0) * 1000), self.last_ts_ms + 1)  # VIDEO mode cần timestamp tăng dần
        self.last_ts_ms = ts_ms
        res = self.recognizer.recognize_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                                                  ts_ms)
        if not res.hand_landmarks:
            return G.NONE, 0.0, None, ''
        top = res.gestures[0][0] if res.gestures and res.gestures[0] else None
        label = top.category_name if top and top.category_name else G.NONE
        hand = res.handedness[0][0].category_name if res.handedness else ''
        return label, (top.score if top else 0.0), res.hand_landmarks[0], hand

    # ---------- hiển thị ----------
    def draw(self, img, raw, score, landmarks, hand, stable, conf):
        h, w = img.shape[:2]
        if landmarks is not None:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
            for a, b in HAND_CONNECTIONS:
                cv2.line(img, pts[a], pts[b], (255, 255, 255), 2)
            for pt in pts:
                cv2.circle(img, pt, 4, (0, 140, 255), -1)

        color = COLORS.get(stable, (200, 200, 200))
        cv2.rectangle(img, (0, 0), (w, 78), (30, 30, 30), -1)
        title = f'{stable}  {conf * 100:.0f}%' if stable != G.NONE else 'No gesture'
        cv2.putText(img, title, (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3, cv2.LINE_AA)
        swings = self.gfilter.wave.swings()
        detail = f'raw: {raw} {score:.2f}  {hand}   swings: {swings}   {self.fps:4.1f} fps'
        cv2.putText(img, detail, (10, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
        if stable in G.ACTIONS:
            cv2.putText(img, '-> ' + G.ACTIONS[stable], (w - 290, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        color, 2, cv2.LINE_AA)

        # bảng cử chỉ -> hành vi
        y = 100
        for g, act in G.ACTIONS.items():
            cv2.putText(img, f'{g}: {act}', (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        COLORS[g] if g == stable else (240, 240, 240), 1, cv2.LINE_AA)
            y += 18

        cv2.rectangle(img, (0, h - 30), (w, h), (30, 30, 30), -1)
        cv2.putText(img, 'ROBOT: ' + self.robot_state, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 255), 1, cv2.LINE_AA)

    def close(self):
        self.recognizer.close()
        if self.cap is not None:
            self.cap.release()

    def summary(self):
        if not self.counts:
            return '  (chưa nhận được cử chỉ nào)'
        return '\n'.join(f'  {g:12s} {n} lần' for g, n in self.counts.most_common())


def _interrupt(*_):
    raise KeyboardInterrupt


def main():
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    signal.signal(signal.SIGINT, _interrupt)
    signal.signal(signal.SIGTERM, _interrupt)
    node = GestureRecognitionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # launch gửi SIGINT rồi SIGTERM: bỏ qua tín hiệu tiếp theo để dọn dẹp xong (close của MediaPipe chặn)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        print('\n[gesture_recognition_node] TỔNG KẾT số lần nhận diện\n' + node.summary())
        node.close()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
