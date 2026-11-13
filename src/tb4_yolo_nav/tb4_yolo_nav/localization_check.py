"""
So sánh vị trí AMCL (/amcl_pose) với vị trí thật trong Gazebo (/sim_ground_truth_pose).

/amcl_pose ở frame map, /sim_ground_truth_pose ở frame world.
Robot được spawn ở gốc world đúng như lúc quét bản đồ nên frame map trùng frame world.
Nếu không, khai báo độ lệch bằng offset_x / offset_y / offset_yaw_deg
(pose của gốc map trong frame world).

Dùng:
  ros2 run tb4_yolo_nav localization_check
Ctrl+C để dừng và in thống kê sai số.
"""

import math

from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, qos_profile_sensor_data, QoSProfile


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class LocalizationCheck(Node):

    def __init__(self):
        super().__init__('localization_check')
        self.declare_parameter('period', 1.0)
        self.declare_parameter('offset_x', 0.0)
        self.declare_parameter('offset_y', 0.0)
        self.declare_parameter('offset_yaw_deg', 0.0)

        self.amcl = None
        self.truth = None
        self.errors = []  # (sai số vị trí m, sai số góc độ)

        # AMCL chỉ phát /amcl_pose khi robot di chuyển, dạng transient local -> phải
        # subscribe transient local để nhận được bản tin cuối khi robot đứng yên
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(PoseWithCovarianceStamped, 'amcl_pose', self._amcl_cb, latched)
        self.create_subscription(Odometry, 'sim_ground_truth_pose', self._truth_cb,
                                 qos_profile_sensor_data)
        self.create_timer(self.get_parameter('period').value, self._report)
        self.get_logger().info('Đang so sánh /amcl_pose với /sim_ground_truth_pose ...')

    def _amcl_cb(self, msg):
        self.amcl = msg.pose

    def _truth_cb(self, msg):
        self.truth = msg.pose.pose

    def truth_in_map(self):
        """Đổi pose thật từ frame world sang frame map."""
        ox = self.get_parameter('offset_x').value
        oy = self.get_parameter('offset_y').value
        oyaw = math.radians(self.get_parameter('offset_yaw_deg').value)
        dx, dy = self.truth.position.x - ox, self.truth.position.y - oy
        x = math.cos(oyaw) * dx + math.sin(oyaw) * dy
        y = -math.sin(oyaw) * dx + math.cos(oyaw) * dy
        return x, y, wrap(yaw_of(self.truth.orientation) - oyaw)

    def _report(self):
        if self.amcl is None or self.truth is None:
            missing = [n for n, v in (('/amcl_pose', self.amcl),
                                      ('/sim_ground_truth_pose', self.truth)) if v is None]
            self.get_logger().info(f'Chờ dữ liệu: {", ".join(missing)} '
                                   '(chưa đặt initial pose?)', throttle_duration_sec=5.0)
            return

        p = self.amcl.pose
        ax, ay, ayaw = p.position.x, p.position.y, yaw_of(p.orientation)
        tx, ty, tyaw = self.truth_in_map()
        dist = math.hypot(ax - tx, ay - ty)
        dyaw = math.degrees(abs(wrap(ayaw - tyaw)))
        cov = self.amcl.covariance
        self.errors.append((dist, dyaw))

        self.get_logger().info(
            f'AMCL ({ax:6.2f},{ay:6.2f},{math.degrees(ayaw):6.1f}°) | '
            f'Thật ({tx:6.2f},{ty:6.2f},{math.degrees(tyaw):6.1f}°) | '
            f'lệch {dist:.3f} m, {dyaw:.1f}° | '
            f'σ AMCL ({math.sqrt(cov[0]):.2f} m, {math.sqrt(cov[7]):.2f} m, '
            f'{math.degrees(math.sqrt(cov[35])):.1f}°)')

    def summary(self):
        if not self.errors:
            return 'Không có mẫu nào.'
        d = [e[0] for e in self.errors]
        a = [e[1] for e in self.errors]
        return (f'{len(d)} mẫu | sai số vị trí: TB {sum(d) / len(d):.3f} m, max {max(d):.3f} m | '
                f'sai số góc: TB {sum(a) / len(a):.1f}°, max {max(a):.1f}°')


def main():
    rclpy.init()
    node = LocalizationCheck()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        print('\n[localization_check] ' + node.summary())
        node.destroy_node()
        rclpy.try_shutdown()
