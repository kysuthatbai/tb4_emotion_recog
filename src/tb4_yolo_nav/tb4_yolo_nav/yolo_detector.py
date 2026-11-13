"""
Nhận diện vật thể bằng YOLO trên ảnh camera RGB của TurtleBot 4.

Luồng xử lý:
  /oakd/rgb/preview/image_raw (sensor_msgs/Image)
    -> callback chỉ giữ khung mới nhất (không dồn hàng đợi khi suy luận chậm hơn camera)
    -> timer `rate` Hz: cv_bridge -> ảnh OpenCV BGR -> YOLO predict
    -> vẽ bounding box + tên class + độ tự tin
    -> publish /yolo/image_annotated (xem trên RViz), /yolo/detections (vision_msgs)
    -> in danh sách vật thể ra terminal (khi danh sách đổi hoặc mỗi print_period giây)
    -> (tùy chọn) hiện cửa sổ OpenCV

Mô hình:
  - YOLO-World (tên file có "-world", ví dụ yolov8s-worldv2.pt): open-vocabulary, danh sách lớp
    lấy từ tham số `classes` -> nhận được cả "fire extinguisher" (không có trong COCO).
    Lần đầu set_classes cần gói CLIP (git+https://github.com/ultralytics/CLIP.git).
  - YOLO COCO thường (yolov8n.pt, yolo11n.pt ...): `classes` dùng để lọc các lớp COCO.

Dùng:
  ros2 run tb4_yolo_nav yolo_detector
  ros2 run tb4_yolo_nav yolo_detector --ros-args -p show_window:=true
  ros2 run tb4_yolo_nav yolo_detector --ros-args -p model:=yolo11n.pt
"""

import os
import signal
import time

from cv_bridge import CvBridge
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, qos_profile_sensor_data, QoSProfile
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import Image
from std_msgs.msg import String
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

DEFAULT_MODEL = os.path.join(os.path.expanduser('~'), 'tb4_ws', 'models', 'yolo',
                             'yolov8s-worldv2.pt')


def color_of(cls_id):
    """Màu BGR cố định cho mỗi lớp."""
    rng = np.random.default_rng(cls_id * 7 + 3)
    return tuple(int(c) for c in rng.integers(60, 256, 3))


class YoloDetector(Node):

    def __init__(self):
        super().__init__('yolo_detector')
        self.declare_parameter('model', DEFAULT_MODEL)
        self.declare_parameter('classes', ['person', 'chair', 'fire extinguisher'])
        self.declare_parameter('conf', 0.25)
        self.declare_parameter('iou', 0.5)
        self.declare_parameter('imgsz', 640)
        self.declare_parameter('device', '')  # '' = tự chọn (GPU nếu có), 'cpu', '0' ...
        self.declare_parameter('image_topic', '/oakd/rgb/preview/image_raw')
        self.declare_parameter('status_topic', '/goal_navigator/status')
        self.declare_parameter('rate', 10.0)          # Hz suy luận tối đa
        self.declare_parameter('print_period', 1.0)   # s, chu kỳ in lại khi kết quả không đổi
        self.declare_parameter('display_scale', 2.0)  # phóng ảnh 320x240 cho dễ nhìn
        self.declare_parameter('show_window', False)

        p = lambda name: self.get_parameter(name).value  # noqa: E731
        self.conf, self.iou, self.imgsz = p('conf'), p('iou'), p('imgsz')
        self.device = p('device') or None
        self.scale = max(float(p('display_scale')), 1.0)
        self.print_period = p('print_period')
        self.show_window = p('show_window')

        self.model, self.class_filter = self._load_model(os.path.expanduser(p('model')),
                                                         list(p('classes')))
        self.names = self.model.names

        self.bridge = CvBridge()
        self.latest = None       # khung ảnh mới nhất chưa xử lý
        self.nav_status = ''     # trạng thái từ goal_navigator, ví dụ "Goal 2: arrived"
        self.last_print = (0.0, None)
        self.stats = {}          # tên lớp -> [số khung thấy, conf lớn nhất, nơi thấy]
        self.frames = 0
        self.infer_ms = 0.0

        self.create_subscription(Image, p('image_topic'), self._image_cb, qos_profile_sensor_data)
        self.create_subscription(
            String, p('status_topic'), self._status_cb,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.image_pub = self.create_publisher(Image, '/yolo/image_annotated', 10)
        self.det_pub = self.create_publisher(Detection2DArray, '/yolo/detections', 10)
        self.create_timer(1.0 / p('rate'), self._process)
        self.get_logger().info(f'Đang nhận ảnh từ {p("image_topic")}, publish '
                               '/yolo/image_annotated và /yolo/detections')

    def _load_model(self, path, classes):
        from ultralytics import YOLO  # import chậm (~2 s), để sau khi node đã khởi tạo

        if os.sep in path and not os.path.isfile(path):
            # Không có file: ultralytics tự tải trọng số chính thức về đúng đường dẫn này
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.get_logger().warn(f'Chưa có {path}, ultralytics sẽ tải về (cần mạng)')
        model = YOLO(path)
        class_filter = None
        if '-world' in os.path.basename(path) and classes:
            model.set_classes(classes)  # mã hóa tên lớp bằng CLIP
            self.get_logger().info(f'YOLO-World, lớp: {classes}')
        elif classes:
            ids = {n: i for i, n in model.names.items()}
            missing = [c for c in classes if c not in ids]
            if missing:
                self.get_logger().warn(f'Mô hình không có lớp {missing} (chỉ YOLO-World nhận '
                                       'lớp tùy ý), bỏ qua')
            class_filter = [ids[c] for c in classes if c in ids] or None
            self.get_logger().info(f'YOLO {os.path.basename(path)}, lọc lớp: '
                                   f'{[model.names[i] for i in class_filter or []] or "tất cả"}')
        # Khởi động trước để khung đầu tiên không bị chậm
        model.predict(np.zeros((240, 320, 3), np.uint8), imgsz=self.imgsz, device=self.device,
                      verbose=False)
        return model, class_filter

    def _image_cb(self, msg):
        self.latest = msg

    def _status_cb(self, msg):
        self.nav_status = msg.data

    # ---------- Suy luận ----------

    def _process(self):
        msg, self.latest = self.latest, None
        if msg is None:
            return
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        t0 = time.perf_counter()
        result = self.model.predict(frame, conf=self.conf, iou=self.iou, imgsz=self.imgsz,
                                    classes=self.class_filter, device=self.device,
                                    verbose=False)[0]
        ms = (time.perf_counter() - t0) * 1000
        self.infer_ms = ms if self.frames == 0 else 0.9 * self.infer_ms + 0.1 * ms
        self.frames += 1

        boxes = result.boxes
        dets = [(int(c), float(s), b) for c, s, b in zip(boxes.cls.tolist(), boxes.conf.tolist(),
                                                         boxes.xyxy.tolist())]
        dets.sort(key=lambda d: -d[1])

        annotated = self.draw(frame, dets)
        out = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
        out.header = msg.header
        self.image_pub.publish(out)
        self.det_pub.publish(self.to_msg(msg.header, dets))
        self.report(dets)

        if self.show_window:
            cv2.imshow('YOLO - TurtleBot 4', annotated)
            cv2.waitKey(1)

    def draw(self, frame, dets):
        s = self.scale
        img = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR) \
            if s != 1.0 else frame.copy()
        for cls_id, score, (x1, y1, x2, y2) in dets:
            color = color_of(cls_id)
            p1, p2 = (int(x1 * s), int(y1 * s)), (int(x2 * s), int(y2 * s))
            cv2.rectangle(img, p1, p2, color, 2)
            label = f'{self.names[cls_id]} {score:.2f}'
            (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            ty = p1[1] - 4 if p1[1] - th - 8 > 0 else p1[1] + th + 6
            cv2.rectangle(img, (p1[0], ty - th - 4), (p1[0] + tw + 6, ty + base - 2), color, -1)
            cv2.putText(img, label, (p1[0] + 3, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 0, 0), 1, cv2.LINE_AA)
        hud = f'{len(dets)} obj | {self.infer_ms:.0f} ms | {self.nav_status}'
        (tw, th), _ = cv2.getTextSize(hud, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y = img.shape[0] - 10
        cv2.rectangle(img, (4, y - th - 6), (14 + tw, y + 6), (0, 0, 0), -1)  # nền tối cho dễ đọc
        cv2.putText(img, hud, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        return img

    def to_msg(self, header, dets):
        arr = Detection2DArray(header=header)
        for cls_id, score, (x1, y1, x2, y2) in dets:
            d = Detection2D(header=header)
            d.bbox.center.position.x = (x1 + x2) / 2
            d.bbox.center.position.y = (y1 + y2) / 2
            d.bbox.size_x, d.bbox.size_y = x2 - x1, y2 - y1
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = self.names[cls_id]
            hyp.hypothesis.score = score
            d.results.append(hyp)
            arr.detections.append(d)
        return arr

    def report(self, dets):
        where = self.nav_status.split(':')[0] if self.nav_status else '-'
        for cls_id, score, _ in dets:
            st = self.stats.setdefault(self.names[cls_id], [0, 0.0, set()])
            st[0] += 1
            st[1] = max(st[1], score)
            st[2].add(where)

        key = tuple(sorted({self.names[c] for c, _, _ in dets}))
        now = time.monotonic()
        if key == self.last_print[1] and now - self.last_print[0] < self.print_period:
            return
        self.last_print = (now, key)
        if not dets:
            self.get_logger().info(f'[{self.nav_status or "-"}] không thấy vật thể')
            return
        items = ' | '.join(
            f'{self.names[c]} {s:.2f} @({(x1 + x2) / 2:.0f},{(y1 + y2) / 2:.0f}) '
            f'{x2 - x1:.0f}x{y2 - y1:.0f}px' for c, s, (x1, y1, x2, y2) in dets)
        self.get_logger().info(f'[{self.nav_status or "-"}] {len(dets)} vật: {items}')

    def summary(self):
        if not self.stats:
            return f'{self.frames} khung, không phát hiện vật thể nào.'
        lines = [f'{self.frames} khung, suy luận TB {self.infer_ms:.1f} ms/khung',
                 f"{'lớp':20s} {'số khung':>8s} {'conf max':>8s}  thấy tại"]
        for name, (n, best, where) in sorted(self.stats.items(), key=lambda kv: -kv[1][0]):
            lines.append(f'{name:20s} {n:8d} {best:8.2f}  {", ".join(sorted(where))}')
        return '\n'.join(lines)


def _interrupt(*_):
    raise KeyboardInterrupt


def main():
    # Ctrl+C và SIGTERM (khi launch dừng) đều thành KeyboardInterrupt -> in tổng kết rồi thoát gọn
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    signal.signal(signal.SIGTERM, _interrupt)
    node = YoloDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        print('\n[yolo_detector] TỔNG KẾT\n' + node.summary())
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.try_shutdown()
