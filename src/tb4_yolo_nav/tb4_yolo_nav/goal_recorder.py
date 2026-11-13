"""
Ghi lại tọa độ mục tiêu (x, y, yaw) trong frame map, lưu ra file YAML cho node điều hướng dùng lại.

Hai cách thêm một Goal:
  1. Công cụ "Publish Point" trên RViz (topic /clicked_point), click 2 lần:
       - lần 1: vị trí Goal
       - lần 2: điểm mà robot cần quay mặt về (ví dụ vật thể)  -> yaw = atan2(dy, dx)
       (click lần 2 trùng chỗ lần 1 -> giữ yaw = 0°)
  2. Lấy vị trí hiện tại của robot theo AMCL (/amcl_pose):
       ros2 service call /goal_recorder/save_robot_pose std_srvs/srv/Trigger

Dịch vụ khác:
  /goal_recorder/undo   xóa Goal cuối
  /goal_recorder/clear  xóa hết
  /goal_recorder/list   in lại bảng Goal

Mỗi Goal được kiểm tra trên /map: cảnh báo nếu nằm ở ô chưa biết/vật cản hoặc quá gần tường.

Dùng:
  ros2 run tb4_yolo_nav goal_recorder
  ros2 run tb4_yolo_nav goal_recorder --ros-args -p output_file:=/đường/dẫn/goals.yaml
"""

import math
import os

from geometry_msgs.msg import PointStamped, PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray
import yaml

DEFAULT_OUTPUT = os.path.join(os.path.expanduser('~'), 'tb4_ws', 'config', 'goals.yaml')
OCCUPIED = 65  # ngưỡng chiếm chỗ (%), giống occupied_thresh 0.65 của map


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class GoalRecorder(Node):

    def __init__(self):
        super().__init__('goal_recorder')
        self.declare_parameter('output_file', DEFAULT_OUTPUT)
        self.declare_parameter('frame_id', 'map')
        self.declare_parameter('min_clearance', 0.4)  # m, khoảng cách tối thiểu tới vật cản

        self.output_file = os.path.expanduser(self.get_parameter('output_file').value)
        self.frame_id = self.get_parameter('frame_id').value
        self.min_clearance = self.get_parameter('min_clearance').value

        self.goals = self._load()
        self.pending = None  # (x, y) đã click lần 1, chờ click lần 2 để chọn hướng
        self.amcl_pose = None
        self.map = None

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.marker_pub = self.create_publisher(MarkerArray, 'goal_markers', latched)
        self.create_subscription(PointStamped, 'clicked_point', self._clicked_cb, 10)
        # /amcl_pose và /map đều là transient local (AMCL chỉ phát khi robot di chuyển)
        self.create_subscription(PoseWithCovarianceStamped, 'amcl_pose', self._amcl_cb, latched)
        self.create_subscription(OccupancyGrid, 'map', self._map_cb, latched)
        self.create_service(Trigger, '~/save_robot_pose', self._save_robot_pose_srv)
        self.create_service(Trigger, '~/undo', self._undo_srv)
        self.create_service(Trigger, '~/clear', self._clear_srv)
        self.create_service(Trigger, '~/list', self._list_srv)

        self._publish_markers()
        self.get_logger().info(
            f'File Goal: {self.output_file} (đã có {len(self.goals)} Goal). '
            'Dùng "Publish Point" trên RViz: click vị trí Goal, rồi click điểm cần nhìn về.')
        if self.goals:
            self._print_table()

    # ---------- File ----------

    def _load(self):
        if not os.path.isfile(self.output_file):
            return []
        with open(self.output_file) as f:
            data = yaml.safe_load(f) or {}
        return list(data.get('goals') or [])

    def _save(self):
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        with open(self.output_file, 'w') as f:
            f.write('# Tọa độ mục tiêu trong frame map, ghi bởi tb4_yolo_nav/goal_recorder.\n'
                    '# yaw_deg: hướng robot tại Goal (0° = trục +x của map, ngược chiều kim '
                    'đồng hồ là dương).\n')
            yaml.safe_dump({'frame_id': self.frame_id, 'goals': self.goals}, f,
                           sort_keys=False, allow_unicode=True)

    # ---------- Kiểm tra trên bản đồ ----------

    def _map_cb(self, msg):
        self.map = msg

    def check_on_map(self, x, y):
        """Trả về chuỗi cảnh báo ('' nếu ổn) cho một vị trí trên bản đồ."""
        if self.map is None:
            return 'chưa nhận được /map nên chưa kiểm tra được'
        info = self.map.info
        res = info.resolution
        c = int((x - info.origin.position.x) / res)
        r = int((y - info.origin.position.y) / res)
        if not (0 <= c < info.width and 0 <= r < info.height):
            return 'nằm ngoài bản đồ'
        grid = np.asarray(self.map.data, dtype=np.int16).reshape(info.height, info.width)
        value = grid[r, c]
        if value < 0:
            return 'nằm ở vùng CHƯA BIẾT (xám) trên bản đồ'
        if value >= OCCUPIED:
            return 'nằm trên VẬT CẢN'
        k = int(math.ceil(1.5 / res))
        win = grid[max(r - k, 0):r + k + 1, max(c - k, 0):c + k + 1]
        rr, cc = np.nonzero(win >= OCCUPIED)
        if rr.size:
            dist = np.hypot(rr + max(r - k, 0) - r, cc + max(c - k, 0) - c).min() * res
            if dist < self.min_clearance:
                return f'quá gần tường ({dist:.2f} m < {self.min_clearance:.2f} m)'
        return ''

    # ---------- Thêm Goal ----------

    def _clicked_cb(self, msg):
        if msg.header.frame_id and msg.header.frame_id != self.frame_id:
            self.get_logger().warn(f'Điểm click ở frame "{msg.header.frame_id}", '
                                   f'cần đặt Fixed Frame của RViz là "{self.frame_id}".')
            return
        x, y = msg.point.x, msg.point.y
        if self.pending is None:
            self.pending = (x, y)
            warn = self.check_on_map(x, y)
            self.get_logger().info(
                f'[1/2] Vị trí Goal {len(self.goals) + 1}: ({x:.3f}, {y:.3f}). '
                'Click thêm 1 điểm để chọn hướng.' + (f'  CẢNH BÁO: {warn}' if warn else ''))
            self._publish_markers()
            return

        px, py = self.pending
        self.pending = None
        if math.hypot(x - px, y - py) < 0.15:
            yaw = 0.0
            self.get_logger().info('[2/2] Click trùng vị trí -> giữ yaw = 0°')
        else:
            yaw = math.degrees(math.atan2(y - py, x - px))
        self.add_goal(px, py, yaw, 'clicked_point')

    def _amcl_cb(self, msg):
        self.amcl_pose = msg.pose.pose

    def _save_robot_pose_srv(self, request, response):
        if self.amcl_pose is None:
            response.success = False
            response.message = 'Chưa nhận được /amcl_pose (chưa đặt initial pose?)'
            return response
        p = self.amcl_pose
        goal = self.add_goal(p.position.x, p.position.y,
                             math.degrees(yaw_of(p.orientation)), 'amcl_pose')
        response.success = True
        response.message = f"Đã lưu {goal['name']} từ /amcl_pose"
        return response

    def add_goal(self, x, y, yaw_deg, source):
        goal = {'name': f'Goal {len(self.goals) + 1}',
                'x': round(float(x), 3),
                'y': round(float(y), 3),
                'yaw_deg': round(float(yaw_deg), 1),
                'source': source}
        warn = self.check_on_map(x, y)
        if warn:
            goal['warning'] = warn
        self.goals.append(goal)
        self._save()
        self._publish_markers()
        yaw = math.radians(yaw_deg)
        self.get_logger().info(
            f"Đã lưu {goal['name']}: x={goal['x']:.3f} y={goal['y']:.3f} "
            f"yaw={goal['yaw_deg']:.1f}° (quaternion z={math.sin(yaw / 2):.4f} "
            f"w={math.cos(yaw / 2):.4f})" + (f'  CẢNH BÁO: {warn}' if warn else ''))
        self._print_table()
        return goal

    # ---------- Dịch vụ khác ----------

    def _undo_srv(self, request, response):
        if self.pending is not None:
            self.pending = None
            response.message = 'Đã hủy điểm đang chọn dở'
        elif self.goals:
            response.message = f"Đã xóa {self.goals.pop()['name']}"
            self._save()
        else:
            response.message = 'Không có Goal nào'
        self._publish_markers()
        self.get_logger().info(response.message)
        response.success = True
        return response

    def _clear_srv(self, request, response):
        self.goals, self.pending = [], None
        self._save()
        self._publish_markers()
        response.success = True
        response.message = 'Đã xóa toàn bộ Goal'
        self.get_logger().info(response.message)
        return response

    def _list_srv(self, request, response):
        self._print_table()
        response.success = True
        response.message = '; '.join(
            f"{g['name']}: ({g['x']}, {g['y']}, {g['yaw_deg']}°)" for g in self.goals)
        return response

    def _print_table(self):
        lines = ['', f"{'Goal':8s} {'x (m)':>8s} {'y (m)':>8s} {'yaw (°)':>8s}  nguồn"]
        for g in self.goals:
            lines.append(f"{g['name']:8s} {g['x']:8.3f} {g['y']:8.3f} {g['yaw_deg']:8.1f}  "
                         f"{g['source']}" + (f"  ⚠ {g['warning']}" if g.get('warning') else ''))
        self.get_logger().info('\n'.join(lines))

    # ---------- RViz ----------

    def _publish_markers(self):
        msg = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        msg.markers.append(clear)

        for i, g in enumerate(self.goals):
            yaw = math.radians(g['yaw_deg'])
            arrow = self._marker('goals', i, Marker.ARROW, g['x'], g['y'], 0.05)
            arrow.pose.orientation.z = math.sin(yaw / 2)
            arrow.pose.orientation.w = math.cos(yaw / 2)
            arrow.scale.x, arrow.scale.y, arrow.scale.z = 0.6, 0.1, 0.1
            arrow.color.r, arrow.color.g, arrow.color.b = (1.0, 0.6, 0.0) \
                if g.get('warning') else (0.1, 0.9, 0.2)
            text = self._marker('goal_labels', i, Marker.TEXT_VIEW_FACING,
                                g['x'], g['y'], 0.5)
            text.scale.z = 0.35
            text.color.r = text.color.g = text.color.b = 1.0
            text.text = f'G{i + 1}'
            msg.markers += [arrow, text]

        if self.pending is not None:
            dot = self._marker('pending', 0, Marker.SPHERE, *self.pending, 0.1)
            dot.scale.x = dot.scale.y = dot.scale.z = 0.2
            dot.color.r, dot.color.g, dot.color.b = 1.0, 1.0, 0.0
            msg.markers.append(dot)
        self.marker_pub.publish(msg)

    def _marker(self, ns, mid, mtype, x, y, z):
        m = Marker()
        m.header.frame_id = self.frame_id
        m.ns, m.id, m.type = ns, mid, mtype
        m.pose.position.x, m.pose.position.y, m.pose.position.z = float(x), float(y), z
        m.pose.orientation.w = 1.0
        m.color.a = 1.0
        return m


def main():
    rclpy.init()
    node = GoalRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
