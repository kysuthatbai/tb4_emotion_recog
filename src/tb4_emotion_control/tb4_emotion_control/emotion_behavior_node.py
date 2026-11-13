"""
Điều khiển TurtleBot 4 theo biểu cảm khuôn mặt nhận từ /detected_emotion, qua topic /cmd_vel.

  Biểu cảm          Hành động                                   Kiểu
  happy  (Vui vẻ)   quỹ đạo cong sang TRÁI  (v=0.15, R=0.5 m)    liên tục, chừng nào còn giữ biểu cảm
  sad    (Buồn)     quỹ đạo cong sang PHẢI  (v=0.15, R=0.5 m)    liên tục, chừng nào còn giữ biểu cảm
  neutral           DỪNG tại chỗ (hủy mọi chuyển động)           ngay lập tức
  angry  (Tức giận) lùi nhẹ 0.2 m                               một lần
  surprise          xoay một vòng 360° tại chỗ                  một lần
  fear   (Sợ hãi)   tiến nhẹ 0.25 m                             một lần

Quy tắc ra lệnh (vòng điều khiển control_rate Hz):
  - biểu cảm phải giữ liên tục hold_time giây mới thành lệnh (neutral: stop_hold_time)
  - biểu cảm điều khiển MỚI (khác hành vi đang chạy) thay thế ngay hành vi cũ
  - hành vi một lần chỉ chạy một lần cho mỗi lần giữ biểu cảm; muốn lặp lại phải đổi biểu cảm rồi làm lại
  - quỹ đạo cong dừng khi biểu cảm không còn (unknown / disgust) quá release_time giây,
    hoặc ngay khi thấy neutral; không thấy mặt (no_face) cũng dừng nếu stop_on_no_face
An toàn: mất /detected_emotion > signal_timeout -> dừng; LiDAR thấy vật cản gần hơn safety_distance theo
hướng chạy -> dừng; mất /odom; một đoạn chạy quá thời gian dự kiến.
Mọi chuyển động chạy vòng kín theo /odom (quãng đường / góc tích lũy), chậm dần khi gần đích.

Dùng:
  ros2 run tb4_emotion_control emotion_behavior_node
  ros2 topic pub -r 10 /detected_emotion std_msgs/msg/String "data: happy"     # thử không cần webcam
"""

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

from tb4_emotion_control import emotions as E
from tb4_emotion_control.motions import BEHAVIOR_PARAMS, build_behaviors, SegmentRun, wrap, yaw_of

IDLE, RUNNING, STOPPED = 'IDLE', 'RUNNING', 'STOPPED'


class EmotionBehaviorNode(Node):

    def __init__(self):
        super().__init__('emotion_behavior_node')
        self.declare_parameter('emotion_topic', '/detected_emotion')
        self.declare_parameter('state_topic', '/emotion_behavior_state')
        self.declare_parameter('cmd_vel_topic', 'cmd_vel')
        self.declare_parameter('odom_topic', 'odom')
        self.declare_parameter('scan_topic', 'scan')
        self.declare_parameter('control_rate', 20.0)        # Hz
        # chống nhiễu
        self.declare_parameter('hold_time', 0.3)            # s giữ biểu cảm để thành lệnh
        self.declare_parameter('stop_hold_time', 0.3)       # s riêng cho neutral / no_face
        self.declare_parameter('release_time', 0.5)         # s quỹ đạo cong chạy tiếp khi biểu cảm chập chờn
        self.declare_parameter('signal_timeout', 1.0)       # s không có /detected_emotion -> dừng
        self.declare_parameter('stop_on_no_face', True)
        # quỹ đạo
        for name, default in BEHAVIOR_PARAMS.items():
            self.declare_parameter(name, default)
        # an toàn
        self.declare_parameter('safety_distance', 0.35)     # m, tính từ LiDAR
        self.declare_parameter('safety_half_angle', 0.6)    # rad, nửa góc quạt kiểm tra
        self.declare_parameter('scan_yaw_offset', math.pi / 2)  # rplidar_joint xoay pi/2 so với base_link
        # Create 3 mặc định chỉ cho lùi ~0.15 m -> lùi 0.2 m bị chặn; True: safety_override = backup_only
        self.declare_parameter('disable_backup_limit', True)

        p = lambda name: self.get_parameter(name).value  # noqa: E731
        self.hold_time, self.stop_hold_time = p('hold_time'), p('stop_hold_time')
        self.release_time, self.signal_timeout = p('release_time'), p('signal_timeout')
        self.stop_on_no_face = p('stop_on_no_face')
        self.safety_distance, self.safety_half_angle = p('safety_distance'), p('safety_half_angle')
        self.scan_yaw_offset = p('scan_yaw_offset')
        self.behaviors = build_behaviors({name: p(name) for name in BEHAVIOR_PARAMS})

        self.cmd_pub = self.create_publisher(Twist, p('cmd_vel_topic'), 10)
        self.state_pub = self.create_publisher(String, p('state_topic'), 10)
        self.create_subscription(String, p('emotion_topic'), self.on_emotion, 10)
        self.create_subscription(Odometry, p('odom_topic'), self.on_odom, qos_profile_sensor_data)
        self.create_subscription(LaserScan, p('scan_topic'), self.on_scan, qos_profile_sensor_data)

        self.state, self.note = IDLE, ''
        self.running, self.run_emotion = None, None      # Behavior đang chạy và biểu cảm đã kích hoạt nó
        self.idx, self.seg = 0, None                      # đoạn hiện tại (SegmentRun)
        self.turned, self.path = 0.0, 0.0                 # thống kê hành vi đang chạy (theo odom)
        self.cur_emotion, self.cur_since = E.NO_FACE, time.monotonic()
        self.consumed = None          # biểu cảm một lần đã thực hiện trong lần giữ hiện tại
        self.last_msg_time = None
        self.signal_lost = True
        self.odom, self.odom_time, self.last_xy = None, 0.0, None
        self.scan, self.scan_time = None, 0.0
        self.zero_until = 0.0
        self.last_state_pub = 0.0
        self.warned_no_scan = self.warned_broken_scan = False

        self.create_timer(1.0 / p('control_rate'), self.tick)
        if p('disable_backup_limit'):
            self.param_client = self.create_client(SetParameters, 'motion_control/set_parameters')
            self.param_future = None
            self.param_timer = self.create_timer(1.0, self.set_backup_override)
        self.get_logger().info('Sẵn sàng. Chờ biểu cảm trên ' + p('emotion_topic'))

    def set_backup_override(self):
        """Gọi (lặp lại tới khi thành công) service set_parameters của motion_control (Create 3)."""
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

    # ---------- đầu vào ----------
    def on_emotion(self, msg):
        now = time.monotonic()
        self.last_msg_time = now
        if msg.data != self.cur_emotion:
            self.cur_emotion, self.cur_since, self.consumed = msg.data, now, None

    def on_odom(self, msg):
        self.odom, self.odom_time = msg, time.monotonic()
        p = msg.pose.pose.position
        if self.running is not None and self.last_xy is not None:
            self.path += math.hypot(p.x - self.last_xy[0], p.y - self.last_xy[1])
        self.last_xy = (p.x, p.y)

    def on_scan(self, msg):
        self.scan, self.scan_time = msg, time.monotonic()
        broken = self.scan_broken(msg)
        if broken and not self.warned_broken_scan:
            self.get_logger().error(
                f'/scan gần như toàn {msg.range_min:.3f} m (range_min): Gazebo đang render bằng GPU Intel. '
                'Chạy lại sim bằng GPU NVIDIA (nvidia:=true). Robot sẽ không tiến/lùi.')
        self.warned_broken_scan = broken

    def pose(self):
        p, q = self.odom.pose.pose.position, self.odom.pose.pose.orientation
        return p.x, p.y, yaw_of(q)

    # ---------- quyết định ----------
    def decide(self, now):
        e, held = self.cur_emotion, now - self.cur_since
        if e == E.NEUTRAL or (e == E.NO_FACE and self.stop_on_no_face):
            if held >= self.stop_hold_time and self.running is not None:
                self.get_logger().info(f'[{e}] DỪNG tại chỗ (hủy {self.running.name})')
                self.halt(IDLE, '')
            return
        if e in self.behaviors and held >= self.hold_time:
            if e == self.run_emotion and self.running is not None:
                return                                   # đang chạy đúng hành vi này
            if e == self.consumed:
                return                                   # hành vi một lần đã chạy trong lần giữ này
            self.start(e)
            return
        if (self.running is not None and self.running.continuous and e != self.run_emotion
                and held >= self.release_time):
            name, was = self.running.name, self.run_emotion
            self.get_logger().info(f'[{e}] hết biểu cảm {was} -> dừng {name}')
            self.halt(IDLE, '')

    def start(self, emotion):
        if self.odom is None or time.monotonic() - self.odom_time > 0.5:
            self.get_logger().error('Không có /odom, không thể thực hiện hành vi', throttle_duration_sec=2.0)
            return
        prev = self.running.name if self.running else None
        b = self.behaviors[emotion]
        self.running, self.run_emotion, self.idx, self.seg = b, emotion, 0, None
        self.turned, self.path, self.last_xy = 0.0, 0.0, None
        self.state, self.note = RUNNING, ''
        self.get_logger().info(f'[{emotion}] {E.VIETNAMESE[emotion]} -> {b.name}'
                               + (f' (thay {prev})' if prev else ''))

    def halt(self, state, note):
        if self.running is not None:
            self.get_logger().info(f'  {self.running.name}: đã quay {math.degrees(self.turned + self.seg_turn()):.0f}°'
                                   f', đi {self.path:.2f} m (odom)')
            self.consumed = self.run_emotion if not self.running.continuous else self.consumed
        self.publish_zero()
        self.running, self.run_emotion, self.seg = None, None, None
        self.state, self.note = state, note

    def seg_turn(self):
        return self.seg.yaw_acc if self.seg is not None else 0.0

    def publish_zero(self):
        self.cmd_pub.publish(Twist())
        self.zero_until = time.monotonic() + 0.5   # gửi lặp lại 0.5 s cho chắc

    # ---------- vòng điều khiển ----------
    def tick(self):
        now = time.monotonic()
        lost = self.last_msg_time is None or now - self.last_msg_time > self.signal_timeout
        if lost and not self.signal_lost:
            self.get_logger().error(f'MẤT TÍN HIỆU BIỂU CẢM > {self.signal_timeout:.1f} s -> dừng robot')
            if self.running is not None:
                self.halt(IDLE, 'mat tin hieu')
            self.publish_zero()
        elif not lost and self.signal_lost:
            self.get_logger().info('Đã nhận tín hiệu biểu cảm')
        self.signal_lost = lost

        if not lost:
            self.decide(now)
        if self.running is not None:
            self.step(now)
        elif now < self.zero_until:
            self.cmd_pub.publish(Twist())
        self.publish_state()

    def step(self, now):
        if now - self.odom_time > 0.5:
            self.abort('mat odom')
            return
        if self.seg is None:
            self.seg = SegmentRun(self.running.segments[self.idx], self.pose(), now)
        self.seg.update(self.pose())
        if self.seg.finished():
            self.turned += self.seg.yaw_acc
            self.idx += 1
            if self.idx >= len(self.running.segments):
                name, emotion = self.running.name, self.run_emotion
                self.seg = None
                self.halt(IDLE, '')
                self.consumed = emotion
                self.get_logger().info(f'Hoàn thành {name} -> chờ biểu cảm tiếp theo')
                return
            self.seg = SegmentRun(self.running.segments[self.idx], self.pose(), now)
        if now > self.seg.deadline:
            self.abort(f'qua thoi gian "{self.seg.seg.label}"')
            return
        v, w = self.seg.command()
        if v != 0.0:
            if self.scan_broken(self.scan):
                self.abort('LiDAR loi: toan range_min, chay Gazebo bang GPU NVIDIA')
                return
            blocked = self.obstacle(v)
            if blocked is not None:
                self.abort(f'vat can {blocked:.2f} m')
                return
        cmd = Twist()
        cmd.linear.x, cmd.angular.z = v, w
        self.cmd_pub.publish(cmd)

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
        name, emotion = self.running.name, self.run_emotion
        self.halt(STOPPED, reason)
        self.consumed = emotion          # không tự chạy lại khi vẫn giữ biểu cảm (tránh lao vào vật cản)
        self.get_logger().error(f'Dừng {name}: {reason}. Đổi biểu cảm rồi làm lại để chạy tiếp')

    # ---------- trạng thái cho cửa sổ camera ----------
    def describe(self):
        if self.signal_lost:
            return 'NO EMOTION SIGNAL - robot stopped'
        if self.state == RUNNING and self.running is not None:
            label = self.seg.seg.label if self.seg else ''
            return f'RUNNING {self.running.name} ({label}) | neutral = stop'
        if self.state == STOPPED:
            return f'SAFETY STOP ({self.note}) | change expression to continue'
        return f'IDLE (face: {self.cur_emotion}) | show an expression'

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
    node = EmotionBehaviorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        node.get_logger().info('Tắt node: gửi lệnh dừng robot')
        for _ in range(3):
            node.cmd_pub.publish(Twist())
            time.sleep(0.05)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
