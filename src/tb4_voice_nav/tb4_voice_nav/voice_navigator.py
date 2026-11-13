"""
Điều hướng bằng giọng nói: văn bản lệnh (/voice/text) -> khu vực trong zones.yaml -> Nav2 NavigateToPose.

Luồng xử lý:
  khởi động / giám sát (timer 1 Hz, không bao giờ bỏ cuộc):
    1. chờ AMCL active; đặt initial pose nếu AMCL chưa có vị trí (gửi lặp tới khi /amcl_pose xác nhận)
    2. chờ bt_navigator active + action server /navigate_to_pose; lệnh nói trước đó được giữ lại và chạy khi sẵn sàng
    3. lifecycle manager kẹt (lỗi DDS mất phản hồi change_state) quá nav2_stuck_timeout -> tự configure/activate
    4. Nav2 mất active giữa chừng -> báo lỗi, quay lại bước 2
  mỗi câu lệnh (callback /voice/text):
    CommandParser.parse(text)
      GOTO [X, Y ...]  -> lộ trình. Câu "từ X đến Y": nếu robot KHÔNG ở X thì đi qua điểm đích của X
                          trước rồi mới tới Y; robot đang ở X thì đi thẳng tới Y.
                          Lệnh mới thay thế lệnh đang chạy (hủy goal cũ).
      UNKNOWN_ZONE     -> báo "không có khu vực ...", liệt kê khu vực hợp lệ; robot KHÔNG di chuyển,
                          nhiệm vụ đang chạy (nếu có) vẫn tiếp tục
      STOP             -> hủy goal, xóa lộ trình
      WHERE / LIST     -> trả lời vị trí hiện tại / danh sách khu vực
  gửi goal (bất đồng bộ, node vẫn nhận lệnh mới trong lúc robot chạy):
    nếu đang ở dock -> action /undock trước
    pose đích = điểm đích hợp lệ của khu vực, hướng = hướng di chuyển (từ vị trí robot tới đích)
    kết quả: SUCCEEDED -> kiểm tra robot thật sự nằm trong khu vực -> điểm kế tiếp
             ABORTED   -> xóa costmap, thử lại `retries` lần;  quá goal_timeout -> hủy
  Phản hồi bằng chữ lên /voice/feedback; khu vực đang tới lên ~/active_zone (zone_publisher tô sáng).

Dùng:
  ros2 run tb4_voice_nav voice_navigator
  ros2 topic pub --once /voice/text std_msgs/msg/String "data: 'Đi đến khu vực A'"
"""

import math
import os
import signal
import time

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from irobot_create_msgs.action import Undock
from irobot_create_msgs.msg import DockStatus, HazardDetection, HazardDetectionVector
from lifecycle_msgs.msg import Transition
from lifecycle_msgs.srv import ChangeState, GetState
from nav2_msgs.action import NavigateToPose
from nav2_msgs.srv import ClearEntireCostmap
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, qos_profile_sensor_data, QoSProfile
from rclpy.signals import SignalHandlerOptions
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener

from tb4_voice_nav import command_parser as cp
from tb4_voice_nav.zones import ZoneMap

STATUS_TEXT = {GoalStatus.STATUS_SUCCEEDED: 'SUCCEEDED', GoalStatus.STATUS_ABORTED: 'ABORTED',
               GoalStatus.STATUS_CANCELED: 'CANCELED'}
LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
# Thứ tự giống lifecycle manager của nav2_bringup (localization_launch.py / navigation_launch.py, Humble)
LOC_NODES = ['map_server', 'amcl']
NAV_NODES = ['controller_server', 'smoother_server', 'planner_server', 'behavior_server', 'bt_navigator',
             'waypoint_follower', 'velocity_smoother']


def default_zones_file():
    from ament_index_python.packages import get_package_share_directory
    return os.path.join(get_package_share_directory('tb4_voice_nav'), 'config', 'zones.yaml')


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class VoiceNavigator(Node):

    def __init__(self):
        super().__init__('voice_navigator')
        self.declare_parameter('zones_file', '')            # '' = config/zones.yaml của package
        self.declare_parameter('text_topic', '/voice/text')
        self.declare_parameter('feedback_topic', '/voice/feedback')
        self.declare_parameter('robot_frame', 'base_link')
        # auto: chỉ đặt initial pose khi AMCL chưa có vị trí; always; never
        self.declare_parameter('initial_pose', 'auto')
        self.declare_parameter('init_x', 0.0)
        self.declare_parameter('init_y', 0.0)
        self.declare_parameter('init_yaw_deg', 0.0)
        self.declare_parameter('undock', True)
        self.declare_parameter('nav2_stuck_timeout', 45.0)  # s Nav2 chưa active -> coi lifecycle manager kẹt
        self.declare_parameter('lifecycle_fallback', True)  # tự configure / activate Nav2 khi bị kẹt
        self.declare_parameter('goal_timeout', 240.0)       # s tối đa cho mỗi khu vực
        self.declare_parameter('retries', 1)                # thử lại khi Nav2 ABORTED
        self.declare_parameter('face_travel_direction', True)  # hướng đích = hướng di chuyển
        self.declare_parameter('fuzzy_threshold', 0.8)

        self.p = lambda name: self.get_parameter(name).value  # noqa: E731
        path = os.path.expanduser(self.p('zones_file') or default_zones_file())
        self.zones = ZoneMap(path)
        self.parser = cp.CommandParser(self.zones, self.p('fuzzy_threshold'))
        self.frame_id = self.zones.frame_id

        self.feedback_pub = self.create_publisher(String, self.p('feedback_topic'), 10)
        self.active_pub = self.create_publisher(String, '~/active_zone', LATCHED)
        self.initial_pose_pub = self.create_publisher(PoseWithCovarianceStamped, 'initialpose', 10)
        self.create_subscription(String, self.p('text_topic'), self.on_text, 10)
        self.amcl_pose = None
        self.create_subscription(PoseWithCovarianceStamped, 'amcl_pose',
                                 lambda m: setattr(self, 'amcl_pose', m.pose.pose), LATCHED)
        self.is_docked = None
        self.create_subscription(DockStatus, 'dock_status',
                                 lambda m: setattr(self, 'is_docked', m.is_docked), qos_profile_sensor_data)
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.undock_client = ActionClient(self, Undock, 'undock')
        self.clear_clients = [self.create_client(ClearEntireCostmap, s) for s in (
            'global_costmap/clear_entirely_global_costmap', 'local_costmap/clear_entirely_local_costmap')]
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        # Giám sát tránh vật cản: va chạm bumper/stall của Create 3, khoảng cách LiDAR nhỏ nhất khi chạy
        self.bumps, self._bumping, self.min_range = 0, False, math.inf
        self.create_subscription(HazardDetectionVector, 'hazard_detection', self.on_hazard,
                                 qos_profile_sensor_data)
        self.create_subscription(LaserScan, 'scan', self.on_scan, qos_profile_sensor_data)

        self.ready = False
        self.lc = {}               # trạng thái lifecycle của các node Nav2 / localization
        self.pending_cmd = None    # lệnh nhận khi Nav2 chưa sẵn sàng, chạy khi sẵn sàng
        self.initial_pose_sent = False
        self.last_initial_pose = 0.0
        self.enter('amcl')
        self.gen = 0               # tăng mỗi khi lệnh mới / dừng -> bỏ qua callback của goal cũ
        self.route = []            # khu vực còn phải đi
        self.current = None        # khu vực đang tới
        self.goal_handle = None
        self.result_future = None
        self.attempt = 0
        self.goal_t0 = 0.0
        self.leg_bumps = 0
        self.last_fb_log = 0.0
        self.distance = None
        self.stats = {'lệnh': 0, 'đến nơi': 0, 'thất bại': 0, 'khu vực không tồn tại': 0, 'không hiểu': 0}
        self.create_timer(1.0, self.watchdog)
        self.create_timer(1.0, self.supervise)
        self.set_active('')
        self.get_logger().info(f'{len(self.zones)} khu vực từ {path}: {self.zones.summary()}')
        self.get_logger().info('Chờ AMCL và Nav2 ...')

    # ---------------------------------------------------------------- tiện ích
    def say(self, text, level='info'):
        # rclpy cấm một dòng gọi logger đổi mức log giữa các lần gọi -> mỗi mức một dòng riêng
        if level == 'error':
            self.get_logger().error(text)
        elif level == 'warn':
            self.get_logger().warn(text)
        else:
            self.get_logger().info(text)
        self.feedback_pub.publish(String(data=text))

    def set_active(self, zone_id):
        self.active_pub.publish(String(data=zone_id))

    def robot_pose(self):
        try:
            t = self.tf_buffer.lookup_transform(self.frame_id, self.p('robot_frame'), Time())
        except TransformException:
            return None
        return t.transform.translation.x, t.transform.translation.y, yaw_of(t.transform.rotation)

    def current_zone(self):
        pose = self.robot_pose()
        return (self.zones.zone_at(pose[0], pose[1]), pose) if pose else (None, None)

    def on_hazard(self, msg):
        bumping = any(d.type in (HazardDetection.BUMP, HazardDetection.STALL) for d in msg.detections)
        if bumping and not self._bumping:
            self.bumps += 1
            self.get_logger().warn(f'VA CHẠM (bumper/stall) lần {self.bumps}')
        self._bumping = bumping

    def on_scan(self, msg):
        if self.current is None:
            return
        valid = [r for r in msg.ranges if msg.range_min < r < msg.range_max]
        if valid:
            self.min_range = min(self.min_range, min(valid))

    # ---------------------------------------------------------------- khởi động / giám sát Nav2
    # Chạy bằng timer, không bao giờ bỏ cuộc: amcl -> initial pose -> Nav2 active -> sẵn sàng; Nav2 mất active
    # thì quay lại chờ. Lifecycle manager của Nav2 (Humble) có thể kẹt vĩnh viễn khi phản hồi change_state bị
    # DDS làm mất ("failed to send response to .../change_state (timeout)") -> sau nav2_stuck_timeout s node tự
    # configure / activate các node Nav2 theo đúng thứ tự của lifecycle manager.
    def lc_state(self, name):
        """Trạng thái lifecycle gần nhất của node (None nếu node chưa chạy). Hỏi bất đồng bộ, không chặn."""
        e = self.lc.get(name)
        if e is None:
            e = self.lc[name] = {'get': self.create_client(GetState, f'{name}/get_state'),
                                 'set': self.create_client(ChangeState, f'{name}/change_state'),
                                 'state': None, 'state_t': 0.0, 'fut': None, 't': 0.0,
                                 'set_fut': None, 'set_t': 0.0, 'set_done': None}
        now = time.monotonic()
        if e['fut'] is not None:
            if e['fut'].done():
                res = e['fut'].result()
                e['state'] = res.current_state.label if res is not None else None
                e['state_t'], e['fut'] = e['t'], None
            elif now - e['t'] > 3.0:   # phản hồi bị mất -> hỏi lại
                e['fut'] = None
        if not e['get'].service_is_ready():
            e['state'] = None
        elif e['fut'] is None:
            e['fut'], e['t'] = e['get'].call_async(GetState.Request()), now
        return e['state']

    def bring_up(self, nodes):
        """Đưa các node lên active theo thứ tự: configure lần lượt, rồi activate lần lượt (mỗi lần một node)."""
        states = [self.lc_state(n) for n in nodes]
        if any(st is None or st in ('configuring', 'activating', 'cleaningup', 'deactivating')
               for st in states):
            return
        now = time.monotonic()
        for name in nodes:          # lệnh chuyển trước đó: chờ xong VÀ đã đọc trạng thái mới sau khi xong
            e = self.lc[name]
            if e['set_fut'] is None:
                continue
            if not e['set_fut'].done() and now - e['set_t'] < 10.0:
                return
            if e['set_done'] is None:
                e['set_done'] = now
            if e['state_t'] <= e['set_done'] and now - e['set_done'] < 5.0:
                return
            e['set_fut'], e['set_done'] = None, None
        for want, transition in (('unconfigured', Transition.TRANSITION_CONFIGURE),
                                 ('inactive', Transition.TRANSITION_ACTIVATE)):
            for name, st in zip(nodes, states):
                if st != want:
                    continue
                e = self.lc[name]
                self.get_logger().warn(f'Tự {"configure" if want == "unconfigured" else "activate"} {name}')
                e['set_fut'] = e['set'].call_async(ChangeState.Request(transition=Transition(id=transition)))
                e['set_t'] = now
                return

    def enter(self, phase):
        self.phase, self.phase_t, self.stuck_warned = phase, time.monotonic(), False

    def publish_initial_pose(self):
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = self.frame_id  # stamp = 0: AMCL dùng TF mới nhất
        yaw = math.radians(self.p('init_yaw_deg'))
        msg.pose.pose.position.x = float(self.p('init_x'))
        msg.pose.pose.position.y = float(self.p('init_y'))
        msg.pose.pose.orientation.z = math.sin(yaw / 2)
        msg.pose.pose.orientation.w = math.cos(yaw / 2)
        msg.pose.covariance[0] = msg.pose.covariance[7] = 0.25
        msg.pose.covariance[35] = 0.06853891909122467
        self.initial_pose_pub.publish(msg)
        self.last_initial_pose = time.monotonic()

    def supervise(self):
        now = time.monotonic()
        waited = now - self.phase_t
        stuck = waited > self.p('nav2_stuck_timeout') and self.p('lifecycle_fallback')
        log = self.get_logger()

        if self.phase == 'amcl':
            st = self.lc_state('amcl')
            if st is None:                 # node chưa chạy: chưa tính thời gian "kẹt"
                self.phase_t = now
            if st == 'active':
                self.enter('pose')
            else:
                log.info(f'Chờ AMCL ({st or "chưa chạy"}) ...', throttle_duration_sec=10.0)
                if stuck and st is not None:
                    if not self.stuck_warned:
                        self.say('Localization kẹt ở bước khởi động, tự kích hoạt map_server / amcl', 'warn')
                        self.stuck_warned = True
                    self.bring_up(LOC_NODES)

        elif self.phase == 'pose':
            mode = self.p('initial_pose')
            if waited < 2.0:               # chờ /amcl_pose latched nếu AMCL đã có vị trí
                return
            if mode == 'always' and not self.initial_pose_sent:
                self.amcl_pose = None
            if mode == 'never' or self.amcl_pose is not None:
                if self.initial_pose_sent:
                    log.info(f'Đã đặt initial pose ({self.p("init_x"):.2f}, {self.p("init_y"):.2f}, '
                             f'{self.p("init_yaw_deg"):.0f}°)')
                self.enter('nav2')
            elif self.initial_pose_pub.get_subscription_count() > 0 and now - self.last_initial_pose > 3.0:
                # bản tin đầu tiên ngay sau khi khớp kết nối DDS hay bị mất -> gửi lặp tới khi AMCL xác nhận
                self.publish_initial_pose()
                self.initial_pose_sent = True

        elif self.phase == 'nav2':
            # đủ 7 node: thiếu velocity_smoother thì lệnh vận tốc của controller không tới /cmd_vel
            states = {n: self.lc_state(n) for n in NAV_NODES}
            st = states['bt_navigator']
            if any(v is None for v in states.values()):   # Nav2 chưa chạy đủ: chưa tính thời gian "kẹt"
                self.phase_t = now
            if all(v == 'active' for v in states.values()) and self.nav_client.server_is_ready():
                self.enter('ready')
                self.ready = True
                zone, _ = self.current_zone()
                where = f', robot đang ở {zone.title}' if zone else ''
                self.say(f'Sẵn sàng nhận lệnh giọng nói{where}. Ví dụ: "Đi đến khu vực A"')
                if self.pending_cmd is not None:
                    cmd, self.pending_cmd = self.pending_cmd, None
                    self.say(f'Thực hiện lệnh đã nhận trước đó: "{cmd.text}"')
                    self.handle_goto(cmd)
                return
            waiting = [f'{n}: {v or "chưa chạy"}' for n, v in states.items() if v != 'active']
            log.info(f'Chờ Nav2 ({", ".join(waiting) or "action navigate_to_pose"}) ...',
                     throttle_duration_sec=10.0)
            if stuck and st is not None:
                if not self.stuck_warned:
                    self.say(f'Nav2 chưa active sau {waited:.0f} s: lifecycle manager có thể bị kẹt '
                             '(mất phản hồi change_state), tự configure / activate các node Nav2', 'warn')
                    self.stuck_warned = True
                self.bring_up(NAV_NODES)

        elif self.phase == 'ready':
            down = [f'{n}: {v}' for n, v in ((n, self.lc_state(n)) for n in NAV_NODES) if v not in (None, 'active')]
            if down:
                self.ready = False
                self.say(f'Nav2 không còn active ({", ".join(down)}), chờ Nav2 lên lại', 'error')
                self.enter('nav2')

    # ---------------------------------------------------------------- lệnh
    def on_text(self, msg):
        cmd = self.parser.parse(msg.data)
        self.stats['lệnh'] += 1
        self.get_logger().info(f'Nghe: "{msg.data}" -> {cp.describe(cmd)}')
        if cmd.kind == cp.GOTO:
            self.handle_goto(cmd)
        elif cmd.kind == cp.UNKNOWN_ZONE:
            self.stats['khu vực không tồn tại'] += 1
            names = ', '.join(f'"{u}"' for u in cmd.unknown)
            keep = f' Vẫn tiếp tục tới {self.current.title}.' if self.current else ''
            self.say(f'Không có khu vực {names} trên bản đồ, robot không di chuyển.{keep} '
                     f'Các khu vực hiện có: {self.zones.summary()}', 'warn')
        elif cmd.kind == cp.AMBIGUOUS:
            self.say(f'"{cmd.unknown[0]}" có thể là: {", ".join(z.title for z in cmd.candidates)}. '
                     'Hãy nói rõ hơn.', 'warn')
        elif cmd.kind == cp.STOP:
            if self.pending_cmd is not None:
                self.pending_cmd = None
                self.say('Đã hủy lệnh đang chờ')
            elif self.current or self.route:
                self.cancel_all()
                self.say('Đã dừng, hủy lộ trình')
            else:
                self.say('Robot đang đứng yên')
        elif cmd.kind == cp.WHERE:
            zone, pose = self.current_zone()
            if pose is None:
                self.say('Chưa biết vị trí robot (chưa có TF map)')
            else:
                self.say(f'Robot ở ({pose[0]:.2f}, {pose[1]:.2f}), '
                         f'{"trong " + zone.title if zone else "ngoài các khu vực"}')
        elif cmd.kind == cp.LIST:
            self.say(f'Có {len(self.zones)} khu vực: {self.zones.summary()}')
        else:
            self.stats['không hiểu'] += 1
            self.say(f'Không hiểu câu lệnh "{msg.data}". Ví dụ: "Đi đến khu vực A", '
                     '"Di chuyển từ khu vực A đến khu vực B", "Dừng lại"', 'warn')

    def handle_goto(self, cmd):
        if not self.ready:
            self.pending_cmd = cmd
            self.say('Nav2 chưa sẵn sàng; đã ghi nhận lệnh, robot sẽ thực hiện ngay khi Nav2 sẵn sàng', 'warn')
            return
        route = list(cmd.route)
        here, _ = self.current_zone()
        note = ''
        if cmd.has_from and len(route) > 1:
            if here is route[0]:
                note = f' Robot đang ở {here.title} nên đi thẳng tới {route[1].title}.'
                route = route[1:]
            else:
                note = (f' Robot đang ở {here.title if here else "ngoài các khu vực"} '
                        f'-> đi qua {route[0].title} trước.')
        elif len(route) == 1 and here is route[0]:
            note = f' Robot đã ở trong {here.title}, di chuyển tới điểm đích của khu vực.'
        if self.current is not None:
            note += f' Hủy lộ trình cũ (đang tới {self.current.title}).'
        self.say('Lộ trình: ' + ' -> '.join(z.title for z in route) + '.' + note)
        self.cancel_all(then=lambda: self.start_route(route))

    def start_route(self, route):
        self.route = list(route)
        self.next_zone()

    def cancel_all(self, then=None):
        """Hủy goal đang chạy. then(): gọi sau khi Nav2 trả kết quả goal cũ (gửi goal mới ngay trong lúc
        bt_navigator đang dừng goal cũ thì goal mới bị abort)."""
        self.gen += 1
        gen = self.gen
        self.route = []
        self.current = None
        self.set_active('')
        handle, result = self.goal_handle, self.result_future
        self.goal_handle = self.result_future = None
        if handle is not None:
            handle.cancel_goal_async()
        if then is None:
            return
        if result is None or result.done():
            then()
            return
        fired = []

        def go(*_):
            if fired or gen != self.gen:   # đã chạy rồi, hoặc lại có lệnh mới hơn
                return
            fired.append(True)
            self.destroy_timer(timer)
            then()

        timer = self.create_timer(3.0, go)  # dự phòng nếu Nav2 không trả kết quả
        result.add_done_callback(go)

    # ---------------------------------------------------------------- điều hướng
    def next_zone(self):
        if not self.route:
            self.current = None
            self.set_active('')
            return
        self.current = self.route.pop(0)
        self.attempt = 0
        self.set_active(self.current.id)
        self.send_goal()

    def send_goal(self):
        gen, zone = self.gen, self.current
        if self.p('undock') and self.is_docked:
            self.say('Robot đang ở dock -> undock trước')
            self.is_docked = None
            fut = self.undock_client.send_goal_async(Undock.Goal())
            fut.add_done_callback(lambda f: self.on_undock_accepted(f, gen))
            return
        x, y, yaw_deg = zone.goal
        yaw = math.radians(yaw_deg)
        pose = self.robot_pose()
        if self.p('face_travel_direction') and pose and math.hypot(x - pose[0], y - pose[1]) > 0.3:
            yaw = math.atan2(y - pose[1], x - pose[0])
        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = self.frame_id
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x, goal.pose.pose.position.y = float(x), float(y)
        goal.pose.pose.orientation.z, goal.pose.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        if self.attempt == 0:
            self.leg_bumps, self.min_range = self.bumps, math.inf
        self.attempt += 1
        self.goal_t0, self.distance = time.monotonic(), None
        self.get_logger().info(f'Gửi goal {zone.title}: ({x:.2f}, {y:.2f}, {math.degrees(yaw):.0f}°)'
                               + (f', lần thử {self.attempt}' if self.attempt > 1 else ''))
        fut = self.nav_client.send_goal_async(goal, feedback_callback=self.on_feedback)
        fut.add_done_callback(lambda f: self.on_goal_accepted(f, gen, zone))

    def on_undock_accepted(self, fut, gen):
        handle = fut.result()
        if gen != self.gen:
            return
        if handle is None or not handle.accepted:
            self.say('Undock bị từ chối, vẫn thử điều hướng', 'warn')
            self.send_goal()
            return
        handle.get_result_async().add_done_callback(lambda f: self.on_undock_done(f, gen))

    def on_undock_done(self, fut, gen):
        if gen != self.gen:
            return
        ok = fut.result().status == GoalStatus.STATUS_SUCCEEDED
        self.get_logger().info('Undock xong' if ok else 'Undock không thành công, vẫn thử điều hướng')
        self.is_docked = False
        self.send_goal()

    def on_goal_accepted(self, fut, gen, zone):
        handle = fut.result()
        if gen != self.gen:      # lệnh đã bị thay thế trong lúc chờ Nav2 nhận goal
            if handle is not None and handle.accepted:
                handle.cancel_goal_async()
            return
        if handle is None or not handle.accepted:
            self.fail(zone, 'Nav2 từ chối goal')
            return
        self.goal_handle = handle
        self.result_future = handle.get_result_async()
        self.result_future.add_done_callback(lambda f: self.on_result(f, gen, zone))

    def on_feedback(self, msg):
        fb = msg.feedback
        self.distance = fb.distance_remaining
        if self.current and time.monotonic() - self.last_fb_log > 3.0:
            self.last_fb_log = time.monotonic()
            self.get_logger().info(f'  -> {self.current.title}: còn {fb.distance_remaining:.2f} m, '
                                   f'recovery {fb.number_of_recoveries}')

    def on_result(self, fut, gen, zone):
        if gen != self.gen:
            return
        self.goal_handle = self.result_future = None
        status = fut.result().status
        dt = time.monotonic() - self.goal_t0
        if status == GoalStatus.STATUS_SUCCEEDED:
            here, pose = self.current_zone()
            check = ''
            if pose:
                d = math.hypot(pose[0] - zone.goal[0], pose[1] - zone.goal[1])
                ok = here is zone
                check = (f', robot ở ({pose[0]:.2f}, {pose[1]:.2f}) cách đích {d:.2f} m, '
                         f'{"đúng" if ok else "KHÔNG nằm trong"} khu vực {zone.id}; '
                         f'va chạm {self.bumps - self.leg_bumps}, gần vật cản nhất {self.min_range:.2f} m')
            self.stats['đến nơi'] += 1
            self.say(f'Đã đến {zone.title} sau {dt:.0f} s{check}')
            self.next_zone()
            if self.current is None:
                self.say('Hoàn thành lộ trình, chờ lệnh tiếp theo')
        elif status == GoalStatus.STATUS_ABORTED and self.attempt <= self.p('retries'):
            self.say(f'Nav2 không tới được {zone.title}, xóa costmap và thử lại', 'warn')
            for c in self.clear_clients:
                if c.service_is_ready():
                    c.call_async(ClearEntireCostmap.Request())
            self.send_goal()
        else:
            self.fail(zone, STATUS_TEXT.get(status, f'trạng thái {status}'))

    def fail(self, zone, reason):
        self.stats['thất bại'] += 1
        self.say(f'Không tới được {zone.title}: {reason}. Hủy phần còn lại của lộ trình', 'error')
        self.cancel_all()

    def watchdog(self):
        if self.goal_handle is not None and self.current is not None and \
                time.monotonic() - self.goal_t0 > self.p('goal_timeout'):
            zone = self.current
            self.fail(zone, f'quá {self.p("goal_timeout"):.0f} s')

    def summary(self):
        return ' | '.join(f'{k}: {v}' for k, v in self.stats.items()) + f' | va chạm: {self.bumps}'


def _interrupt(*_):
    raise KeyboardInterrupt


def main():
    # Tự xử lý Ctrl+C / SIGTERM để còn context ROS gửi lệnh hủy goal (mặc định rclpy shutdown ngay)
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    signal.signal(signal.SIGINT, _interrupt)
    signal.signal(signal.SIGTERM, _interrupt)
    node = VoiceNavigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if node.goal_handle is not None:
            node.get_logger().warn('Tắt node: hủy goal đang chạy')
            fut = node.goal_handle.cancel_goal_async()
            rclpy.spin_until_future_complete(node, fut, timeout_sec=2.0)
        print('\n[voice_navigator] TỔNG KẾT ' + node.summary())
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
