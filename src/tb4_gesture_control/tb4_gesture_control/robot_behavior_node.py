"""
Điều khiển hành vi TurtleBot 4 theo cử chỉ tay nhận từ /detected_gesture.

Máy trạng thái:
                 Pointing_Up / Victory                Thumb_Up
     IDLE ─────────────────────────────▶ SELECTED ─────────────▶ EXECUTING ──(xong)──▶ IDLE
      ▲  ◀──────────── Closed_Fist ──────────┘                    │   │
      │                                                Open_Palm  │   │ Wave (chào lại, chạy ngay)
      └──────────── Closed_Fist ─────────── STOPPED ◀─────────────┘   │
                                            │  Thumb_Up: chạy lại hành vi đã chọn
  Mọi trạng thái: mất tín hiệu cử chỉ > signal_timeout -> dừng robot, về IDLE.

Hành vi (vòng kín theo /odom, mỗi hành vi là chuỗi đoạn line / turn / arc / pause):
  FORWARD_BACK : tiến forward_distance rồi lùi lại, lặp forward_back_cycles lần
  CIRCLE       : cung tròn bán kính circle_radius, đủ 360 độ, quay về điểm xuất phát
  GREETING     : xoay trái-phải greet_angle quanh hướng ban đầu greet_swings lần rồi trả về hướng cũ

Chống nhiễu (ngoài bộ lọc bỏ phiếu ở gesture_recognition_node):
  - một cử chỉ phải giữ liên tục hold_time giây mới thành lệnh (Open_Palm: stop_hold_time, ngắn hơn)
  - mỗi lần giữ cử chỉ chỉ phát đúng 1 lệnh; muốn lặp phải đổi sang cử chỉ khác/None rồi làm lại
  - giữa 2 lệnh cách nhau tối thiểu cooldown giây (trừ lệnh dừng)
  - khi đang EXECUTING bỏ qua lệnh chọn hành vi mới: phải Open_Palm / Closed_Fist trước
An toàn: dừng khi LiDAR thấy vật cản gần hơn safety_distance theo hướng đang chạy,
         khi mất /odom, khi một đoạn chạy quá thời gian dự kiến.

Dùng:
  ros2 run tb4_gesture_control robot_behavior_node
  ros2 run tb4_gesture_control robot_behavior_node --ros-args -p forward_distance:=0.3
"""

from collections import namedtuple
import math
import signal
import time

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

from tb4_gesture_control import gestures as G

IDLE, SELECTED, EXECUTING, STOPPED = 'IDLE', 'SELECTED', 'EXECUTING', 'STOPPED'
FORWARD_BACK, CIRCLE, GREETING = 'FORWARD_BACK', 'CIRCLE', 'GREETING'
SELECT_BY = {G.POINTING_UP: FORWARD_BACK, G.VICTORY: CIRCLE}

# kind: 'line' (goal = m), 'turn' / 'arc' (goal = rad), 'pause' (goal = s)
Segment = namedtuple('Segment', 'kind v w goal label')


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class RobotBehaviorNode(Node):

    def __init__(self):
        super().__init__('robot_behavior_node')
        self.declare_parameter('gesture_topic', '/detected_gesture')
        self.declare_parameter('state_topic', '/robot_behavior_state')
        self.declare_parameter('cmd_vel_topic', 'cmd_vel')
        self.declare_parameter('odom_topic', 'odom')
        self.declare_parameter('scan_topic', 'scan')
        self.declare_parameter('control_rate', 20.0)       # Hz
        # chống nhiễu
        self.declare_parameter('hold_time', 0.5)           # s giữ cử chỉ để thành lệnh
        self.declare_parameter('stop_hold_time', 0.15)     # s, riêng Open_Palm (dừng phải nhanh)
        self.declare_parameter('cooldown', 1.0)            # s tối thiểu giữa 2 lệnh
        self.declare_parameter('signal_timeout', 1.0)      # s không có /detected_gesture -> dừng
        self.declare_parameter('wave_requires_confirm', False)  # True: Wave chỉ chọn, cần Thumb_Up
        # hành vi
        self.declare_parameter('linear_speed', 0.2)        # m/s
        self.declare_parameter('forward_distance', 0.4)    # m mỗi lần tiến/lùi
        self.declare_parameter('forward_back_cycles', 4)
        self.declare_parameter('circle_radius', 0.5)       # m
        self.declare_parameter('circle_speed', 0.15)       # m/s
        self.declare_parameter('greet_angle', 0.6)         # rad (~35 độ) mỗi bên
        self.declare_parameter('greet_speed', 1.2)         # rad/s
        self.declare_parameter('greet_swings', 2)
        self.declare_parameter('segment_pause', 0.4)       # s nghỉ giữa 2 lần đổi chiều
        # an toàn
        self.declare_parameter('safety_distance', 0.35)    # m, tính từ LiDAR
        self.declare_parameter('safety_half_angle', 0.6)   # rad, nửa góc quạt kiểm tra
        # rplidar_joint xoay yaw = pi/2 so với robot (turtlebot4_description) => góc robot = góc scan + pi/2
        self.declare_parameter('scan_yaw_offset', math.pi / 2)
        # Create 3 mặc định chỉ cho lùi ~0.15 m (phía sau không có cảm biến cliff) -> tiến-lùi 0.4 m bị chặn.
        # True: đặt motion_control safety_override = backup_only lúc khởi động
        self.declare_parameter('disable_backup_limit', True)

        p = lambda name: self.get_parameter(name).value  # noqa: E731
        self.hold_time, self.stop_hold_time = p('hold_time'), p('stop_hold_time')
        self.cooldown, self.signal_timeout = p('cooldown'), p('signal_timeout')
        self.wave_requires_confirm = p('wave_requires_confirm')
        self.safety_distance, self.safety_half_angle = p('safety_distance'), p('safety_half_angle')
        self.scan_yaw_offset = p('scan_yaw_offset')
        self.plans = self.build_plans(p)

        self.cmd_pub = self.create_publisher(Twist, p('cmd_vel_topic'), 10)
        self.state_pub = self.create_publisher(String, p('state_topic'), 10)
        self.create_subscription(String, p('gesture_topic'), self.on_gesture, 10)
        self.create_subscription(Odometry, p('odom_topic'), self.on_odom, qos_profile_sensor_data)
        self.create_subscription(LaserScan, p('scan_topic'), self.on_scan, qos_profile_sensor_data)

        self.state = IDLE
        self.selected = None          # hành vi đang chọn chờ xác nhận
        self.running = None           # hành vi đang chạy
        self.note = ''                # lý do dừng gần nhất
        self.plan, self.idx, self.seg = [], 0, None
        # cử chỉ
        self.cur_gesture, self.cur_since, self.fired = G.NONE, 0.0, False
        self.last_cmd_time = -1e9
        self.last_msg_time = None
        self.signal_lost = True
        # cảm biến
        self.odom, self.odom_time = None, 0.0
        self.scan, self.scan_time = None, 0.0
        self.zero_until = 0.0
        self.last_state_pub = 0.0
        self.warned_no_scan = False
        self.warned_broken_scan = False

        self.create_timer(1.0 / p('control_rate'), self.tick)
        if p('disable_backup_limit'):
            self.param_client = self.create_client(SetParameters, 'motion_control/set_parameters')
            self.param_future = None
            self.param_timer = self.create_timer(1.0, self.set_backup_override)
        self.get_logger().info('Sẵn sàng. Chờ cử chỉ trên ' + p('gesture_topic'))

    def set_backup_override(self):
        """Gọi (lặp lại tới khi thành công) service set_parameters của motion_control."""
        if self.param_future is not None:
            if not self.param_future.done():
                return
            res = self.param_future.result()
            if res is not None and res.results and res.results[0].successful:
                self.get_logger().info('motion_control: safety_override = backup_only (cho phép lùi xa)')
                self.param_timer.cancel()
                return
            self.get_logger().warn('Không đặt được safety_override, thử lại')
        if not self.param_client.service_is_ready():
            return
        value = ParameterValue(type=ParameterType.PARAMETER_STRING, string_value='backup_only')
        self.param_future = self.param_client.call_async(
            SetParameters.Request(parameters=[Parameter(name='safety_override', value=value)]))

    def build_plans(self, p):
        v, d, pause = p('linear_speed'), p('forward_distance'), p('segment_pause')
        n = p('forward_back_cycles')
        fb = []
        for i in range(1, n + 1):
            fb += [Segment('line', v, 0.0, d, f'tien {i}/{n}'), Segment('pause', 0, 0, pause, ''),
                   Segment('line', -v, 0.0, d, f'lui {i}/{n}'), Segment('pause', 0, 0, pause, '')]

        cv, r = p('circle_speed'), p('circle_radius')
        circle = [Segment('arc', cv, cv / r, 2 * math.pi, f'vong tron r={r:.2f} m')]

        a, w, k = p('greet_angle'), p('greet_speed'), p('greet_swings')
        greet = [Segment('turn', 0, w, a, 'chao: trai')]
        sign = -1
        for _ in range(2 * k - 1):
            greet += [Segment('turn', 0, sign * w, 2 * a, 'chao: ' + ('phai' if sign < 0 else 'trai'))]
            sign = -sign
        greet += [Segment('turn', 0, sign * w, a, 'chao: ve giua')]
        return {FORWARD_BACK: fb, CIRCLE: circle, GREETING: greet}

    # ---------- đầu vào ----------
    def on_odom(self, msg):
        self.odom, self.odom_time = msg, time.monotonic()

    def on_scan(self, msg):
        self.scan, self.scan_time = msg, time.monotonic()
        broken = self.scan_broken(msg)
        if broken and not self.warned_broken_scan:
            self.get_logger().error(
                f'/scan gần như toàn {msg.range_min:.3f} m (range_min): Gazebo đang render bằng GPU Intel. '
                'Chạy lại sim bằng GPU NVIDIA (sim.launch.py nvidia:=true). Robot sẽ không tiến/lùi.')
        self.warned_broken_scan = broken

    def on_gesture(self, msg):
        now = time.monotonic()
        self.last_msg_time = now
        g = msg.data
        if g != self.cur_gesture:
            self.cur_gesture, self.cur_since, self.fired = g, now, False
        if g == G.NONE or g not in G.ACTIONS or self.fired:
            return
        if now - self.cur_since < (self.stop_hold_time if g == G.OPEN_PALM else self.hold_time):
            return
        if g != G.OPEN_PALM and now - self.last_cmd_time < self.cooldown:
            return                    # chưa hết cooldown; vẫn giữ cử chỉ thì lần gọi sau sẽ phát lệnh
        self.fired = True
        self.last_cmd_time = now
        self.on_command(g)

    # ---------- xử lý lệnh ----------
    def on_command(self, g):
        log = self.get_logger().info
        if g == G.OPEN_PALM:
            if self.state == EXECUTING:
                self.halt(STOPPED, 'Open_Palm')
                log(f'[Open_Palm] DỪNG NGAY. Thumb_Up để chạy lại {self.selected}, Closed_Fist để hủy')
            else:
                self.publish_zero()
                log('[Open_Palm] dừng (robot đang đứng yên)')
        elif g == G.CLOSED_FIST:
            was = self.running or self.selected
            self.selected = None
            self.halt(IDLE, '')
            log(f'[Closed_Fist] hủy {was or "(không có gì)"} -> trạng thái chờ')
        elif self.state == EXECUTING:
            self.get_logger().warn(f'[{g}] bỏ qua: đang chạy {self.running}. '
                                   'Open_Palm để dừng hoặc Closed_Fist để hủy trước')
        elif g in SELECT_BY:
            self.selected = SELECT_BY[g]
            self.state = SELECTED
            log(f'[{g}] chọn {self.selected}. Giơ Thumb_Up để xác nhận')
        elif g == G.WAVE:
            if self.wave_requires_confirm:
                self.selected, self.state = GREETING, SELECTED
                log('[Wave] chọn GREETING. Giơ Thumb_Up để xác nhận')
            else:
                self.selected = None
                log('[Wave] chào lại: xoay trái-phải')
                self.start(GREETING)
        elif g == G.THUMB_UP:
            if self.state in (SELECTED, STOPPED) and self.selected:
                log(f'[Thumb_Up] xác nhận -> thực hiện {self.selected}')
                self.start(self.selected)
            else:
                self.get_logger().warn('[Thumb_Up] chưa chọn hành vi (Pointing_Up / Victory)')
        self.publish_state(force=True)

    def start(self, behavior):
        if self.odom is None or time.monotonic() - self.odom_time > 0.5:
            self.get_logger().error('Không có /odom, không thể thực hiện hành vi')
            return
        self.running, self.plan, self.idx, self.seg = behavior, self.plans[behavior], 0, None
        self.state, self.note = EXECUTING, ''

    def halt(self, state, note):
        self.publish_zero()
        self.running, self.plan, self.seg = None, [], None
        self.state, self.note = state, note

    def publish_zero(self):
        self.cmd_pub.publish(Twist())
        self.zero_until = time.monotonic() + 0.5   # gửi lặp lại 0.5 s cho chắc

    # ---------- vòng điều khiển ----------
    def tick(self):
        now = time.monotonic()
        lost = self.last_msg_time is None or now - self.last_msg_time > self.signal_timeout
        if lost and not self.signal_lost:
            self.get_logger().error(f'MẤT TÍN HIỆU CỬ CHỈ > {self.signal_timeout:.1f} s -> dừng robot')
            if self.state != IDLE:
                self.selected = None
                self.halt(IDLE, 'mat tin hieu')
            self.publish_zero()
        elif not lost and self.signal_lost:
            self.get_logger().info('Đã nhận tín hiệu cử chỉ')
        self.signal_lost = lost

        if self.state == EXECUTING:
            self.step_plan(now)
        elif now < self.zero_until:
            self.cmd_pub.publish(Twist())
        self.publish_state()

    def step_plan(self, now):
        if self.seg is None:
            self.begin_segment(now)
        s = self.seg
        if s['kind'] != 'pause' and now - self.odom_time > 0.5:
            self.abort('mat odom')
            return
        progress = self.progress(s)
        if progress >= s['goal'] or (s['kind'] == 'pause' and now - s['t0'] >= s['goal']):
            self.idx += 1
            if self.idx >= len(self.plan):
                done = self.running
                self.selected = None
                self.halt(IDLE, '')
                self.get_logger().info(f'Hoàn thành {done} -> trạng thái chờ')
                return
            self.begin_segment(now)
            s = self.seg
            progress = 0.0
        if now > s['deadline']:
            self.abort(f'qua thoi gian doan "{s["label"]}"')
            return

        cmd = Twist()
        if s['kind'] != 'pause':
            remain = s['goal'] - progress
            zone = 0.12 if s['kind'] == 'line' else 0.3
            k = min(max(remain / zone, 0.3), 1.0)     # chạy chậm dần khi gần đích để ít vọt lố
            cmd.linear.x, cmd.angular.z = s['v'] * k, s['w'] * k
            if cmd.linear.x != 0.0:
                if self.scan_broken(self.scan):
                    self.abort('LiDAR loi: toan range_min, chay Gazebo bang GPU NVIDIA')
                    return
                blocked = self.obstacle(cmd.linear.x)
                if blocked is not None:
                    self.abort(f'vat can {blocked:.2f} m')
                    return
        self.cmd_pub.publish(cmd)

    def begin_segment(self, now):
        seg = self.plan[self.idx]
        speed = abs(seg.v) if seg.kind == 'line' else abs(seg.w)
        expected = seg.goal if seg.kind == 'pause' else seg.goal / max(speed, 1e-3)
        p, q = self.odom.pose.pose.position, self.odom.pose.pose.orientation
        self.seg = {'kind': seg.kind, 'v': seg.v, 'w': seg.w, 'goal': seg.goal, 'label': seg.label,
                    # deadline rộng: sim có thể chạy chậm hơn thời gian thực
                    't0': now, 'deadline': now + expected * 3.0 + 5.0,
                    'x0': p.x, 'y0': p.y, 'yaw_prev': yaw_of(q), 'yaw_acc': 0.0}
        if seg.label:
            self.get_logger().info(f'  {self.running}: {seg.label}')

    def progress(self, s):
        if s['kind'] == 'pause':
            return -1.0
        p, q = self.odom.pose.pose.position, self.odom.pose.pose.orientation
        if s['kind'] == 'line':
            return math.hypot(p.x - s['x0'], p.y - s['y0'])
        yaw = yaw_of(q)
        s['yaw_acc'] += wrap(yaw - s['yaw_prev'])
        s['yaw_prev'] = yaw
        return s['yaw_acc'] * math.copysign(1.0, s['w'])

    def obstacle(self, v):
        """Khoảng cách vật cản gần nhất trong quạt theo hướng chạy (None nếu thông thoáng)."""
        if self.scan is None or time.monotonic() - self.scan_time > 1.0:
            if not self.warned_no_scan:
                self.get_logger().warn('Không có /scan: chạy không có kiểm tra vật cản')
                self.warned_no_scan = True
            return None
        sc = self.scan
        heading = 0.0 if v > 0 else math.pi
        nearest = None
        for i, r in enumerate(sc.ranges):
            if not math.isfinite(r) or r <= sc.range_min + 0.01:   # chạm ngưỡng đo tối thiểu = không hợp lệ
                continue
            a = sc.angle_min + i * sc.angle_increment + self.scan_yaw_offset
            if abs(wrap(a - heading)) <= self.safety_half_angle and r < self.safety_distance:
                nearest = r if nearest is None else min(nearest, r)
        return nearest

    @staticmethod
    def scan_broken(sc):
        """True nếu quá nửa số tia bằng range_min: Gazebo đang render LiDAR bằng GPU Intel."""
        if sc is None or not sc.ranges:
            return False
        low = sum(1 for r in sc.ranges if math.isfinite(r) and r <= sc.range_min + 0.01)
        return low > 0.5 * len(sc.ranges)

    def abort(self, reason):
        name = self.running
        self.halt(STOPPED, reason)
        self.get_logger().error(f'Dừng {name}: {reason}. Thumb_Up để chạy lại, Closed_Fist để hủy')

    # ---------- trạng thái cho cửa sổ webcam ----------
    def describe(self):
        if self.signal_lost:
            return 'NO GESTURE SIGNAL - robot stopped'
        if self.state == IDLE:
            return 'WAITING | 1 finger: forward-back, 2 fingers: circle, wave: greet'
        if self.state == SELECTED:
            return f'SELECTED {self.selected} -> Thumb_Up to run, Fist to cancel'
        if self.state == EXECUTING:
            label = self.seg['label'] if self.seg and self.seg['label'] else ''
            return f'RUNNING {self.running} {label} | Open_Palm = stop'
        return f'STOPPED ({self.note}) | Thumb_Up: rerun {self.selected or "-"}, Fist: cancel'

    def publish_state(self, force=False):
        now = time.monotonic()
        if force or now - self.last_state_pub >= 0.2:
            self.state_pub.publish(String(data=self.describe()))
            self.last_state_pub = now


def _interrupt(*_):
    raise KeyboardInterrupt


def main():
    # Ctrl+C / SIGTERM: tự xử lý để kịp gửi lệnh dừng trước khi context ROS bị tắt
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    signal.signal(signal.SIGINT, _interrupt)
    signal.signal(signal.SIGTERM, _interrupt)
    node = RobotBehaviorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)   # tín hiệu thứ 2 không được chặn lệnh dừng
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        node.get_logger().info('Tắt node: gửi lệnh dừng robot')
        for _ in range(3):
            node.cmd_pub.publish(Twist())
            time.sleep(0.05)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
