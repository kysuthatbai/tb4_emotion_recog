"""Quỹ đạo của từng biểu cảm và bộ theo dõi tiến độ vòng kín theo odom. Không phụ thuộc ROS.

Mỗi hành vi là một chuỗi đoạn (Segment):
  line : đi thẳng với vận tốc v (âm = lùi) tới khi quãng đường odom >= goal (m)
  turn : quay tại chỗ với vận tốc góc w tới khi góc tích lũy >= goal (rad)
  hold : chạy (v, w) không giới hạn, LIÊN TỤC chừng nào biểu cảm còn được giữ (quỹ đạo cong)

  Happy    -> CURVE_LEFT  : hold, v = curve_speed, w = +curve_speed / curve_radius
  Sad      -> CURVE_RIGHT : hold, v = curve_speed, w = -curve_speed / curve_radius
  Angry    -> BACK_UP     : line lùi back_distance
  Surprise -> SPIN        : turn spin_angle_deg (mặc định 360°)
  Fear     -> FORWARD     : line tiến forward_distance
  Neutral  -> dừng (không có quỹ đạo, xử lý trong emotion_behavior_node)
"""

from collections import namedtuple
import math

from tb4_emotion_control import emotions as E

Segment = namedtuple('Segment', 'kind v w goal label')
Behavior = namedtuple('Behavior', 'name segments continuous')

BEHAVIOR_PARAMS = {
    'curve_speed': 0.15,       # m/s
    'curve_radius': 0.5,       # m (w = v / r = 0.3 rad/s)
    'back_distance': 0.2,      # m
    'back_speed': 0.1,         # m/s
    'forward_distance': 0.25,  # m
    'forward_speed': 0.12,     # m/s
    'spin_speed': 1.2,         # rad/s
    'spin_angle_deg': 360.0,
    # góc odom / góc thật khi quay TẠI CHỖ. Gazebo: odom bánh xe báo dư ~6% (đo bằng /sim_ground_truth_pose:
    # odom 360° = thật 339.6°) -> 1.06. Robot thật (odom có IMU) -> 1.0
    'spin_odom_gain': 1.0,
}


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def build_behaviors(cfg=None):
    c = {**BEHAVIOR_PARAMS, **(cfg or {})}
    v, r = c['curve_speed'], c['curve_radius']
    return {
        E.HAPPY: Behavior('CURVE_LEFT', [Segment('hold', v, v / r, None, f'cong trai r={r:.2f} m')], True),
        E.SAD: Behavior('CURVE_RIGHT', [Segment('hold', v, -v / r, None, f'cong phai r={r:.2f} m')], True),
        E.ANGRY: Behavior('BACK_UP', [Segment('line', -c['back_speed'], 0.0, c['back_distance'],
                                              f'lui {c["back_distance"]:.2f} m')], False),
        E.SURPRISE: Behavior('SPIN', [Segment('turn', 0.0, c['spin_speed'],
                                              math.radians(c['spin_angle_deg']) * c['spin_odom_gain'],
                                              f'xoay {c["spin_angle_deg"]:.0f} do')], False),
        E.FEAR: Behavior('FORWARD', [Segment('line', c['forward_speed'], 0.0, c['forward_distance'],
                                             f'tien {c["forward_distance"]:.2f} m')], False),
    }


class SegmentRun:
    """Theo dõi một đoạn đang chạy. pose = (x, y, yaw) từ odom."""

    LINE_ZONE, TURN_ZONE, MIN_SCALE = 0.08, 0.3, 0.4   # vùng giảm tốc gần đích (m / rad), hệ số tối thiểu
    LINE_TOL, TURN_TOL = 0.003, 0.01                     # dung sai coi như tới đích (m / rad)

    def __init__(self, seg, pose, now, time_factor=3.0, time_slack=5.0):
        self.seg = seg
        self.x0, self.y0, self.yaw_prev = pose
        self.yaw_acc = 0.0      # góc quay tích lũy (có dấu), cộng dồn wrap(dyaw) nên qua ±pi không lỗi
        self.dist = 0.0         # quãng đường thẳng từ điểm bắt đầu
        self.t0 = now
        if seg.kind == 'hold':
            self.deadline = math.inf
        else:
            speed = abs(seg.v) if seg.kind == 'line' else abs(seg.w)
            # deadline rộng vì sim có thể chạy chậm hơn thời gian thực
            self.deadline = now + seg.goal / max(speed, 1e-3) * time_factor + time_slack

    def update(self, pose):
        """Cập nhật theo pose mới, trả về tiến độ theo đơn vị của goal."""
        x, y, yaw = pose
        self.yaw_acc += wrap(yaw - self.yaw_prev)
        self.yaw_prev = yaw
        self.dist = math.hypot(x - self.x0, y - self.y0)
        return self.progress()

    def progress(self):
        if self.seg.kind == 'line':
            return self.dist
        if self.seg.kind == 'turn':
            return self.yaw_acc * math.copysign(1.0, self.seg.w)
        return abs(self.yaw_acc)

    def finished(self):
        if self.seg.kind == 'hold':
            return False
        tol = self.LINE_TOL if self.seg.kind == 'line' else self.TURN_TOL
        return self.progress() >= self.seg.goal - tol

    def command(self):
        """(v, w) cho khung điều khiển hiện tại: chậm dần khi gần đích để ít vọt lố."""
        s = self.seg
        if s.kind == 'hold':
            return s.v, s.w
        zone = self.LINE_ZONE if s.kind == 'line' else self.TURN_ZONE
        k = min(max((s.goal - self.progress()) / zone, self.MIN_SCALE), 1.0)
        return s.v * k, s.w * k
