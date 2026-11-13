"""Phát hiện khuôn mặt + nhận diện biểu cảm bằng DeepFace (không phụ thuộc ROS).

DeepFace.analyze(actions=['emotion']):
  1. detector_backend tìm khuôn mặt (mặc định YuNet, ONNX, ~10 ms / khung 640x480; dự phòng 'opencv' = Haar cascade)
  2. căn thẳng mặt theo 2 mắt (align), cắt vùng mặt
  3. ảnh xám 48x48 -> CNN cảm xúc (huấn luyện trên FER-2013) -> 7 xác suất
     angry, disgust, fear, happy, sad, surprise, neutral

Ép DeepFace chạy bằng onnxruntime (DEEPFACE_BACKEND_ENGINE=onnx): không cần TensorFlow, nhẹ, chạy CPU.
Trọng số tự tải lần đầu về <deepface_home>/.deepface/weights (mặc định ~/tb4_ws/models).

Lưu ý: với enforce_detection=False, khi KHÔNG thấy mặt DeepFace vẫn trả về kết quả cho cả khung hình với
face_confidence = 0 -> kết quả vô nghĩa, phải loại bằng ngưỡng min_face_conf.
"""

from dataclasses import dataclass
import os


@dataclass
class Face:
    box: tuple          # (x, y, w, h) pixel
    score: float        # độ tin cậy phát hiện mặt
    probs: dict         # nhãn -> xác suất 0..1


class EmotionDetector:

    def __init__(self, detector='yunet', deepface_home='~/tb4_ws/models', min_face_conf=0.5,
                 min_face_size=40, align=True, log=print):
        os.environ.setdefault('DEEPFACE_BACKEND_ENGINE', 'onnx')
        os.environ.setdefault('DEEPFACE_LOG_LEVEL', '30')       # chỉ in cảnh báo trở lên
        if deepface_home:
            os.environ['DEEPFACE_HOME'] = os.path.expanduser(deepface_home)
        from deepface import DeepFace  # import sau khi đặt biến môi trường
        self.df = DeepFace
        self.min_face_conf = min_face_conf
        self.min_face_size = min_face_size
        self.align = align
        self.detector = detector
        try:
            self.warmup()
        except Exception as e:  # noqa: BLE001  thiếu trọng số YuNet (không có mạng) -> Haar cascade có sẵn
            if detector == 'opencv':
                raise
            log(f'Không dùng được detector {detector} ({e}); chuyển sang opencv (Haar cascade)')
            self.detector = 'opencv'
            self.warmup()

    def warmup(self):
        import numpy as np
        self.analyze(np.zeros((480, 640, 3), np.uint8))  # nạp (và tải nếu cần) trọng số

    def analyze(self, frame):
        """frame BGR -> danh sách Face (đã lọc mặt giả), sắp theo diện tích giảm dần."""
        results = self.df.analyze(frame, actions=['emotion'], detector_backend=self.detector,
                                  enforce_detection=False, align=self.align, silent=True)
        faces = []
        for r in results:
            reg = r.get('region') or {}
            box = (int(reg.get('x', 0)), int(reg.get('y', 0)), int(reg.get('w', 0)), int(reg.get('h', 0)))
            score = float(r.get('face_confidence') or 0.0)
            if score < self.min_face_conf or min(box[2], box[3]) < self.min_face_size:
                continue
            probs = {k: float(v) / 100.0 for k, v in r['emotion'].items()}
            faces.append(Face(box, score, probs))
        faces.sort(key=lambda f: -f.box[2] * f.box[3])
        return faces
