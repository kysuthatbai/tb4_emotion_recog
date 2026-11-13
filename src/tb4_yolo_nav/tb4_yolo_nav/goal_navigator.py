"""
Điều hướng tự động tuần tự qua các Goal trong goals.yaml bằng nav2_simple_commander.

Luồng xử lý:
  1. Chờ AMCL active (gọi amcl/get_state có timeout, tránh treo như waitUntilNav2Active).
  2. Đặt initial pose nếu AMCL chưa có vị trí (chờ AMCL subscribe /initialpose rồi mới publish).
  3. Chờ bt_navigator active và action server /navigate_to_pose sẵn sàng.
  4. Nếu robot đang ở dock -> gọi action /undock.
  5. Lần lượt goToPose(Goal i): theo dõi feedback, hủy nếu quá goal_timeout, thử lại nếu FAILED.
     Đến nơi thì dừng dwell_time giây (cho node YOLO quan sát) rồi đi Goal kế tiếp.
  6. In bảng tổng kết: kết quả từng Goal, thời gian, sai số vị trí cuối, số lần va chạm (bumper).

Trạng thái được phát lên ~/status (std_msgs/String) để node YOLO gắn nhãn Goal vào kết quả.

Dùng:
  ros2 run tb4_yolo_nav goal_navigator
  ros2 run tb4_yolo_nav goal_navigator --ros-args -p goals_file:=/đường/dẫn/goals.yaml
"""

import math
import os
import signal
import time

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from irobot_create_msgs.action import Undock
from irobot_create_msgs.msg import DockStatus, HazardDetection, HazardDetectionVector
from lifecycle_msgs.srv import GetState
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.qos import DurabilityPolicy, qos_profile_sensor_data, QoSProfile
from rclpy.signals import SignalHandlerOptions
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener
import yaml

DEFAULT_GOALS = os.path.join(os.path.expanduser('~'), 'tb4_ws', 'config', 'goals.yaml')
RESULT_TEXT = {TaskResult.SUCCEEDED: 'SUCCEEDED', TaskResult.CANCELED: 'CANCELED',
               TaskResult.FAILED: 'FAILED'}


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class GoalNavigator(BasicNavigator):

    def __init__(self):
        super().__init__(node_name='goal_navigator')
        self.declare_parameter('goals_file', DEFAULT_GOALS)
        # auto: chỉ đặt initial pose khi AMCL chưa có vị trí; always: luôn đặt; never: không đặt
        self.declare_parameter('initial_pose', 'auto')
        self.declare_parameter('init_x', 0.0)
        self.declare_parameter('init_y', 0.0)
        self.declare_parameter('init_yaw_deg', 0.0)
        self.declare_parameter('undock', True)
        self.declare_parameter('startup_timeout', 120.0)  # s, chờ Nav2/AMCL active
        self.declare_parameter('goal_timeout', 180.0)     # s, tối đa cho mỗi lần thử một Goal
        self.declare_parameter('retries', 1)              # số lần thử lại khi Goal FAILED
        self.declare_parameter('dwell_time', 3.0)         # s, dừng tại Goal cho YOLO quan sát
        self.declare_parameter('stop_on_failure', False)
        self.declare_parameter('loop', False)             # chạy lặp lại danh sách Goal

        self.p = lambda name: self.get_parameter(name).value  # noqa: E731
        self.frame_id, self.goals = self._load_goals(os.path.expanduser(self.p('goals_file')))

        self.status_pub = self.create_publisher(
            String, '~/status', QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.amcl_pose = None  # BasicNavigator đã subscribe amcl_pose (transient local)
        self.create_subscription(PoseWithCovarianceStamped, 'amcl_pose', self._amcl_cb,
                                 QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.is_docked = None
        self.create_subscription(DockStatus, 'dock_status', self._dock_cb, qos_profile_sensor_data)
        self.undock_client = ActionClient(self, Undock, 'undock')
        # Pose robot tại Goal lấy qua TF map->base_link (/amcl_pose chỉ cập nhật sau mỗi 0.25 m)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Giám sát va chạm: sự kiện BUMP/STALL của Create 3 và khoảng cách LiDAR nhỏ nhất
        self.bumps = 0
        self._bumping = False
        self.min_range = math.inf
        self.create_subscription(HazardDetectionVector, 'hazard_detection', self._hazard_cb,
                                 qos_profile_sensor_data)
        self.create_subscription(LaserScan, 'scan', self._scan_cb, qos_profile_sensor_data)

    # ---------- Dữ liệu vào ----------

    def _load_goals(self, path):
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f'Không có file Goal {path}. Ghi Goal bằng: ros2 run tb4_yolo_nav goal_recorder')
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        goals = list(data.get('goals') or [])
        if not goals:
            raise ValueError(f'File {path} không có Goal nào')
        return data.get('frame_id', 'map'), goals

    def _amcl_cb(self, msg):
        self.amcl_pose = msg.pose.pose

    def _dock_cb(self, msg):
        self.is_docked = msg.is_docked

    def _hazard_cb(self, msg):
        bumping = any(d.type in (HazardDetection.BUMP, HazardDetection.STALL)
                      for d in msg.detections)
        if bumping and not self._bumping:
            self.bumps += 1
            self.warn(f'VA CHẠM: bumper/stall lần {self.bumps}')
        self._bumping = bumping

    def _scan_cb(self, msg):
        valid = [r for r in msg.ranges if msg.range_min < r < msg.range_max]
        if valid:
            self.min_range = min(self.min_range, min(valid))

    def publish_status(self, text):
        self.status_pub.publish(String(data=text))

    def spin_for(self, seconds, done=lambda: False):
        """Spin trong `seconds` giây hoặc tới khi done() đúng. Trả về done()."""
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.1)
            if done():
                return True
        return done()

    # ---------- Khởi động ----------

    def wait_active(self, node_name, deadline):
        """Như _waitForNodeToActivate của BasicNavigator nhưng mỗi lần gọi đều có timeout."""
        client = self.create_client(GetState, f'{node_name}/get_state')
        try:
            while time.monotonic() < deadline:
                if not client.wait_for_service(timeout_sec=1.0):
                    self.info(f'Chờ service {node_name}/get_state ...')
                    continue
                future = client.call_async(GetState.Request())
                rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
                if future.done() and future.result() is not None:
                    state = future.result().current_state.label
                    if state == 'active':
                        self.info(f'{node_name}: active')
                        return True
                    self.info(f'{node_name}: {state}, chờ ...')
                self.spin_for(1.0)
            return False
        finally:
            self.destroy_client(client)

    def ensure_initial_pose(self):
        mode = self.p('initial_pose')
        self.spin_for(1.0, lambda: self.amcl_pose is not None)  # nhận /amcl_pose latched nếu có
        if mode == 'never' or (mode == 'auto' and self.amcl_pose is not None):
            if self.amcl_pose is not None:
                p = self.amcl_pose
                self.info(f'AMCL đã có vị trí ({p.position.x:.2f}, {p.position.y:.2f}, '
                          f'{math.degrees(yaw_of(p.orientation)):.1f}°), không đặt lại')
            return True

        if not self.spin_for(10.0, lambda: self.initial_pose_pub.get_subscription_count() > 0):
            self.error('AMCL chưa subscribe /initialpose')
            return False
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = self.frame_id  # stamp = 0: AMCL dùng TF mới nhất
        yaw = math.radians(self.p('init_yaw_deg'))
        msg.pose.pose.position.x = float(self.p('init_x'))
        msg.pose.pose.position.y = float(self.p('init_y'))
        msg.pose.pose.orientation.z = math.sin(yaw / 2)
        msg.pose.pose.orientation.w = math.cos(yaw / 2)
        msg.pose.covariance[0] = msg.pose.covariance[7] = 0.25
        msg.pose.covariance[35] = 0.06853891909122467
        for _ in range(3):
            self.amcl_pose = None
            self.initial_pose_pub.publish(msg)
            if self.spin_for(5.0, lambda: self.amcl_pose is not None):
                self.info(f'Đã đặt initial pose ({msg.pose.pose.position.x:.2f}, '
                          f'{msg.pose.pose.position.y:.2f}, {self.p("init_yaw_deg"):.1f}°)')
                return True
        self.error('AMCL không xác nhận initial pose')
        return False

    def undock_if_needed(self):
        if not self.p('undock') or not self.spin_for(3.0, lambda: self.is_docked is not None):
            return
        if not self.is_docked:
            return
        self.publish_status('undocking')
        self.info('Robot đang ở dock -> undock')
        if not self.undock_client.wait_for_server(timeout_sec=10.0):
            self.warn('Không có action /undock, bỏ qua')
            return
        future = self.undock_client.send_goal_async(Undock.Goal())
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        handle = future.result()
        if handle is None or not handle.accepted:
            self.warn('Undock bị từ chối')
            return
        result = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result, timeout_sec=60.0)
        ok = result.done() and result.result().status == GoalStatus.STATUS_SUCCEEDED
        self.info('Undock xong' if ok else 'Undock không thành công, vẫn tiếp tục')

    def startup(self):
        deadline = time.monotonic() + self.p('startup_timeout')
        if not self.wait_active('amcl', deadline):
            self.error('AMCL chưa active. Đã chạy mô phỏng với localization:=true chưa?')
            return False
        if not self.ensure_initial_pose():
            return False
        if not self.wait_active('bt_navigator', deadline):
            self.error('bt_navigator chưa active. Đã chạy với nav2:=true chưa?')
            return False
        if not self.nav_to_pose_client.wait_for_server(
                timeout_sec=max(deadline - time.monotonic(), 1.0)):
            self.error('Action /navigate_to_pose không sẵn sàng')
            return False
        self.info('Nav2 sẵn sàng')
        self.undock_if_needed()
        return True

    # ---------- Điều hướng ----------

    def make_pose(self, goal):
        pose = PoseStamped()
        pose.header.frame_id = self.frame_id
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = float(goal['x'])
        pose.pose.position.y = float(goal['y'])
        yaw = math.radians(goal.get('yaw_deg', 0.0))
        pose.pose.orientation.z = math.sin(yaw / 2)
        pose.pose.orientation.w = math.cos(yaw / 2)
        return pose

    def navigate(self, goal, label):
        """Gửi một Goal, chờ kết quả. Trả về TaskResult."""
        if not self.goToPose(self.make_pose(goal)):
            return TaskResult.FAILED
        timeout = Duration(seconds=self.p('goal_timeout'))
        last_print = 0.0
        while not self.isTaskComplete():  # isTaskComplete() spin node 0.1 s
            fb = self.getFeedback()
            if fb is None:
                continue
            if Duration.from_msg(fb.navigation_time) > timeout:
                self.warn(f'{label}: quá {self.p("goal_timeout"):.0f} s, hủy')
                self.cancelTask()
                break
            if time.monotonic() - last_print > 2.0:
                last_print = time.monotonic()
                eta = Duration.from_msg(fb.estimated_time_remaining).nanoseconds / 1e9
                self.info(f'{label}: còn {fb.distance_remaining:.2f} m, ETA {eta:.0f} s, '
                          f'recovery {fb.number_of_recoveries}')
        return self.getResult()

    def robot_pose(self):
        """(x, y, yaw) của base_link trong frame map, None nếu chưa có TF."""
        try:
            t = self.tf_buffer.lookup_transform(self.frame_id, 'base_link', Time())
        except TransformException:
            return None
        return t.transform.translation.x, t.transform.translation.y, yaw_of(t.transform.rotation)

    def pose_error(self, goal):
        pose = self.robot_pose()
        if pose is None:
            return None, None
        x, y, yaw = pose
        dist = math.hypot(x - goal['x'], y - goal['y'])
        dyaw = abs(wrap(yaw - math.radians(goal.get('yaw_deg', 0.0))))
        return dist, math.degrees(dyaw)

    def run_mission(self):
        rows = []
        n = len(self.goals)
        for i, goal in enumerate(self.goals, 1):
            name = goal.get('name', f'Goal {i}')
            label = f'[{i}/{n}] {name}'
            self.info(f'{label} -> ({goal["x"]:.2f}, {goal["y"]:.2f}, '
                      f'{goal.get("yaw_deg", 0.0):.1f}°)')
            self.publish_status(f'{name}: navigating')
            t0 = time.monotonic()
            for attempt in range(1, self.p('retries') + 2):
                result = self.navigate(goal, label)
                if result == TaskResult.SUCCEEDED or attempt > self.p('retries'):
                    break
                self.warn(f'{label}: {RESULT_TEXT.get(result, result)}, '
                          f'xóa costmap và thử lại ({attempt}/{self.p("retries")})')
                self.clearAllCostmaps()
            dt = time.monotonic() - t0
            self.spin_for(0.5)  # nhận TF mới nhất
            dist, dyaw = self.pose_error(goal)
            text = RESULT_TEXT.get(result, str(result))
            rows.append((name, text, attempt, dt, dist, dyaw))
            err = f', lệch {dist:.2f} m / {dyaw:.1f}°' if dist is not None else ''
            if result == TaskResult.SUCCEEDED:
                self.info(f'{label}: ĐÃ ĐẾN sau {dt:.1f} s{err}')
                self.publish_status(f'{name}: arrived')
                self.spin_for(self.p('dwell_time'))
            else:
                self.error(f'{label}: {text} sau {dt:.1f} s')
                self.publish_status(f'{name}: {text.lower()}')
                if self.p('stop_on_failure'):
                    break
        return rows

    def print_summary(self, rows):
        lines = ['', 'TỔNG KẾT', f"{'Goal':10s} {'kết quả':10s} {'lần':>3s} {'t (s)':>7s} "
                 f"{'lệch (m)':>9s} {'lệch (°)':>9s}"]
        for name, text, attempt, dt, dist, dyaw in rows:
            d = f'{dist:9.2f}' if dist is not None else f"{'-':>9s}"
            a = f'{dyaw:9.1f}' if dyaw is not None else f"{'-':>9s}"
            lines.append(f'{name:10s} {text:10s} {attempt:3d} {dt:7.1f} {d} {a}')
        ok = sum(r[1] == 'SUCCEEDED' for r in rows)
        lines.append(f'{ok}/{len(self.goals)} Goal thành công | va chạm (bumper/stall): '
                     f'{self.bumps} | khoảng cách LiDAR nhỏ nhất: {self.min_range:.2f} m')
        self.info('\n'.join(lines))


def _interrupt(*_):
    raise KeyboardInterrupt


def main():
    # Tắt signal handler của rclpy: mặc định Ctrl+C shutdown context ngay, khi đó không gửi được
    # lệnh hủy Goal và robot vẫn chạy tiếp. Ở đây Ctrl+C chỉ sinh KeyboardInterrupt.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    signal.signal(signal.SIGTERM, _interrupt)  # launch dừng -> cũng hủy Goal
    node = GoalNavigator()
    try:
        if node.startup():
            while True:
                node.print_summary(node.run_mission())
                if not node.p('loop'):
                    break
            node.publish_status('done')
    except KeyboardInterrupt:
        node.warn('Ctrl+C: hủy Goal đang chạy')
        node.cancelTask()
        node.publish_status('canceled')
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
