"""
Chuẩn bị robot cho demo (chạy một lần rồi thoát, sim.launch.py tự gọi).

Robot spawn ở trạng thái đang sạc, dock nằm ngay trước mặt và LiDAR không thấy dock (dock cao ~10 cm,
mặt quét LiDAR ~19 cm). Node này:
  1. chờ /dock_status; nếu đang docked -> gọi action /undock (robot lùi ~0.35 m và quay ~180 độ)
  2. tiến thêm advance_distance m theo /odom để ra giữa sân, cách dock đủ xa cho quỹ đạo tròn
Đây là bước dựng môi trường, không phải hành vi điều khiển bằng cử chỉ.

Dùng:
  ros2 run tb4_gesture_control prepare_robot
  ros2 run tb4_gesture_control prepare_robot --ros-args -p advance_distance:=0.8
"""

import math
import time

from geometry_msgs.msg import Twist
from irobot_create_msgs.action import Undock
from irobot_create_msgs.msg import DockStatus
from nav_msgs.msg import Odometry
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data


class PrepareRobot(Node):

    def __init__(self):
        super().__init__('prepare_robot')
        self.declare_parameter('advance_distance', 0.6)  # m
        self.declare_parameter('speed', 0.2)             # m/s
        self.declare_parameter('wait_timeout', 90.0)     # s chờ sim/robot sẵn sàng
        self.dock = None
        self.odom = None
        self.create_subscription(DockStatus, 'dock_status', lambda m: setattr(self, 'dock', m),
                                 qos_profile_sensor_data)
        self.create_subscription(Odometry, 'odom', lambda m: setattr(self, 'odom', m),
                                 qos_profile_sensor_data)
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.undock_client = ActionClient(self, Undock, 'undock')

    def spin_until(self, cond, timeout):
        end = time.monotonic() + timeout
        while rclpy.ok() and not cond() and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.1)
        return cond()

    def run(self):
        log = self.get_logger()
        timeout = self.get_parameter('wait_timeout').value
        if not self.spin_until(lambda: self.dock is not None and self.odom is not None, timeout):
            log.error('Không nhận được /dock_status hoặc /odom, bỏ qua bước chuẩn bị')
            return False

        if self.dock.is_docked:
            log.info('Robot đang ở dock -> undock')
            if not self.undock_client.wait_for_server(timeout_sec=timeout):
                log.error('Không có action /undock')
                return False
            fut = self.undock_client.send_goal_async(Undock.Goal())
            rclpy.spin_until_future_complete(self, fut)
            handle = fut.result()
            if not handle.accepted:
                log.error('Undock bị từ chối')
                return False
            res = handle.get_result_async()
            rclpy.spin_until_future_complete(self, res, timeout_sec=60.0)
            if not res.done() or res.result().result.is_docked:
                log.error('Undock không thành công')
                return False
        else:
            log.info('Robot không ở dock, bỏ qua undock')

        dist = self.get_parameter('advance_distance').value
        speed = self.get_parameter('speed').value
        self.spin_until(lambda: False, 0.5)       # lấy odom mới nhất sau undock
        start = self.odom.pose.pose.position
        x0, y0 = start.x, start.y
        log.info(f'Tiến {dist:.2f} m ra giữa sân')
        end = time.monotonic() + dist / speed * 2.0 + 3.0
        cmd = Twist()
        while rclpy.ok() and time.monotonic() < end:
            p = self.odom.pose.pose.position
            remain = dist - math.hypot(p.x - x0, p.y - y0)
            if remain <= 0.0:
                break
            cmd.linear.x = speed * min(max(remain / 0.12, 0.3), 1.0)
            self.cmd_pub.publish(cmd)
            rclpy.spin_once(self, timeout_sec=0.05)
        for _ in range(5):
            self.cmd_pub.publish(Twist())
            rclpy.spin_once(self, timeout_sec=0.05)
        p = self.odom.pose.pose.position
        log.info(f'Sẵn sàng tại odom ({p.x:.2f}, {p.y:.2f}). Có thể bắt đầu điều khiển bằng cử chỉ')
        return True


def main():
    rclpy.init()
    node = PrepareRobot()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
