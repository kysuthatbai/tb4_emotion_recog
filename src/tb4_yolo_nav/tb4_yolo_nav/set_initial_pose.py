"""
Đặt vị trí ban đầu (initial pose) cho AMCL, thay cho thao tác "2D Pose Estimate" trên RViz.

Dùng:
  ros2 run tb4_yolo_nav set_initial_pose                       # mặc định (0, 0, 0°)
  ros2 run tb4_yolo_nav set_initial_pose --ros-args -p x:=1.0 -p y:=-0.5 -p yaw_deg:=90.0
"""

import math
import time

from geometry_msgs.msg import PoseWithCovarianceStamped
import rclpy
from rclpy.node import Node


class InitialPoseSetter(Node):

    def __init__(self):
        super().__init__('set_initial_pose')
        self.declare_parameter('x', 0.0)
        self.declare_parameter('y', 0.0)
        self.declare_parameter('yaw_deg', 0.0)
        self.declare_parameter('frame_id', 'map')
        self.declare_parameter('wait_timeout', 60.0)
        # Cùng giá trị mặc định với công cụ 2D Pose Estimate của RViz
        self.declare_parameter('cov_xy', 0.25)
        self.declare_parameter('cov_yaw', 0.06853891909122467)

        self.pub = self.create_publisher(PoseWithCovarianceStamped, 'initialpose', 10)
        self.amcl_pose = None
        self.create_subscription(PoseWithCovarianceStamped, 'amcl_pose', self._amcl_cb, 10)

    def _amcl_cb(self, msg):
        self.amcl_pose = msg

    def build_msg(self):
        p = lambda name: self.get_parameter(name).value  # noqa: E731
        yaw = math.radians(p('yaw_deg'))
        msg = PoseWithCovarianceStamped()
        # stamp = 0: AMCL dùng TF mới nhất, tránh lỗi extrapolation
        msg.header.frame_id = p('frame_id')
        msg.pose.pose.position.x = float(p('x'))
        msg.pose.pose.position.y = float(p('y'))
        msg.pose.pose.orientation.z = math.sin(yaw / 2)
        msg.pose.pose.orientation.w = math.cos(yaw / 2)
        msg.pose.covariance[0] = msg.pose.covariance[7] = float(p('cov_xy'))
        msg.pose.covariance[35] = float(p('cov_yaw'))
        return msg

    def spin_for(self, seconds, done):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.2)
            if done():
                return True
        return False

    def run(self):
        self.get_logger().info('Đang chờ AMCL subscribe /initialpose ...')
        if not self.spin_for(self.get_parameter('wait_timeout').value,
                             lambda: self.pub.get_subscription_count() > 0):
            self.get_logger().error('Không thấy AMCL. Đã chạy với localization:=true chưa?')
            return False

        msg = self.build_msg()
        self.amcl_pose = None
        self.pub.publish(msg)
        self.get_logger().info(
            f'Đã gửi initial pose: x={msg.pose.pose.position.x:.3f} '
            f'y={msg.pose.pose.position.y:.3f} yaw={self.get_parameter("yaw_deg").value:.1f}°')

        # AMCL phát lại /amcl_pose ngay khi nhận initial pose -> dùng để xác nhận
        if not self.spin_for(10.0, lambda: self.amcl_pose is not None):
            self.get_logger().warn('Chưa thấy /amcl_pose cập nhật (AMCL có thể chưa active). '
                                   'Kiểm tra lại trên RViz.')
            return False
        p = self.amcl_pose.pose.pose
        yaw = math.degrees(2 * math.atan2(p.orientation.z, p.orientation.w))
        self.get_logger().info(
            f'AMCL xác nhận: x={p.position.x:.3f} y={p.position.y:.3f} yaw={yaw:.1f}°')
        return True


def main():
    rclpy.init()
    node = InitialPoseSetter()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
