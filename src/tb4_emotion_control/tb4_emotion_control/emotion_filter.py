"""Lọc nhiễu theo thời gian cho kết quả nhận diện biểu cảm từng khung. Không phụ thuộc ROS.

Mô hình FER-2013 dao động mạnh giữa các khung (ví dụ neutral <-> sad khi chớp mắt), nên:
  1. Làm mượt véc-tơ xác suất 7 lớp bằng trung bình trượt mũ (EMA) theo THỜI GIAN:
         alpha = 1 - exp(-dt / tau)        (không phụ thuộc FPS của camera / tốc độ suy luận)
  2. Ứng viên = lớp có xác suất làm mượt lớn nhất, chỉ chấp nhận khi
         p_max >= min_conf  và  p_max - p_nhì >= min_margin,   ngược lại UNKNOWN
  3. Chống dao động (debounce): ứng viên phải giữ nguyên liên tục stable_time giây mới thành nhãn ổn định.
  4. Mất mặt: chớp nhoáng (< no_face_time) giữ nguyên nhãn cũ; lâu hơn -> NO_FACE và xóa EMA
     (lần thấy mặt sau bắt đầu lại từ đầu, không bị biểu cảm cũ kéo theo).
"""

import math

from tb4_emotion_control import emotions as E

FILTER_PARAMS = {
    'tau': 0.3,            # s, hằng số thời gian EMA
    'min_conf': 0.45,      # xác suất tối thiểu của lớp thắng
    'min_margin': 0.10,    # chênh lệch tối thiểu với lớp thứ hai
    'stable_time': 0.4,    # s ứng viên phải giữ nguyên
    'no_face_time': 0.5,   # s không thấy mặt -> NO_FACE
}


class EmotionFilter:

    def __init__(self, **kw):
        cfg = {**FILTER_PARAMS, **kw}
        self.tau = cfg['tau']
        self.min_conf = cfg['min_conf']
        self.min_margin = cfg['min_margin']
        self.stable_time = cfg['stable_time']
        self.no_face_time = cfg['no_face_time']
        self.ema = None             # dict nhãn -> xác suất làm mượt
        self.last_t = None          # thời điểm mẫu có mặt gần nhất
        self.last_face = -math.inf
        self.candidate = E.NO_FACE
        self.pending, self.pending_since = E.NO_FACE, 0.0
        self.stable = E.NO_FACE

    def update(self, probs, now):
        """probs: dict nhãn -> xác suất 0..1 của khuôn mặt chính, None nếu không thấy mặt.

        Trả về (nhãn ổn định, độ tin cậy của nhãn đó, dict xác suất làm mượt hoặc None).
        """
        if probs is None:
            if now - self.last_face >= self.no_face_time:
                self.ema, self.last_t = None, None
                self.candidate = E.NO_FACE
            # mất mặt chớp nhoáng: giữ ứng viên cũ
        else:
            self.last_face = now
            total = sum(probs.get(k, 0.0) for k in E.MODEL_LABELS) or 1.0
            p = {k: probs.get(k, 0.0) / total for k in E.MODEL_LABELS}
            if self.ema is None:
                self.ema = p
            else:
                a = 1.0 - math.exp(-max(now - self.last_t, 0.0) / self.tau)
                self.ema = {k: (1 - a) * self.ema[k] + a * p[k] for k in E.MODEL_LABELS}
            self.last_t = now
            ranked = sorted(self.ema.items(), key=lambda kv: -kv[1])
            (top, p1), (_, p2) = ranked[0], ranked[1]
            self.candidate = top if p1 >= self.min_conf and p1 - p2 >= self.min_margin else E.UNKNOWN

        if self.candidate != self.pending:
            self.pending, self.pending_since = self.candidate, now
        # NO_FACE đã chờ no_face_time rồi, không cần chờ thêm
        wait = 0.0 if self.pending == E.NO_FACE else self.stable_time
        if self.pending != self.stable and now - self.pending_since >= wait:
            self.stable = self.pending
        conf = self.ema.get(self.stable, 0.0) if self.ema else 0.0
        return self.stable, conf, (dict(self.ema) if self.ema else None)
