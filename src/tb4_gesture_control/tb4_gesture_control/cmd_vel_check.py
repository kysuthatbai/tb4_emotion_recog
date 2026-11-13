"""
Phần 1: kiểm tra robot nhận lệnh vận tốc qua /cmd_vel.

Gửi lần lượt: xoay trái tại chỗ, xoay phải về chỗ cũ, tiến 1 đoạn ngắn, lùi lại.
Đo dịch chuyển thật trên /odom cho mỗi bước và in ĐẠT / KHÔNG ĐẠT.

Dùng (sau khi sim đã chạy và robot đã ra khỏi dock):
  ros2 run tb4_gesture_control cmd_vel_check
"""

import math
import sys
import time

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

STEPS = [  # (mô tả, v m/s, w rad/s, thời gian s)
    ('xoay trai', 0.0, 0.8, 2.0),
    ('xoay phai', 0.0, -0.8, 2.0),
    ('tien', 0.15, 0.0, 2.0),
    ('lui', -0.15, 0.0, 2.0),
]


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class CmdVelCheck(Node):

    def __init__(self):
        super().__init__('cmd_vel_check')
        self.odom = None
        self.create_subscription(Odometry, 'odom', lambda m: setattr(self, 'odom', m),
                                 qos_profile_sensor_data)
        self.pub = self.create_publisher(Twist, 'cmd_vel', 10)

    def pose(self):
        p = self.odom.pose.pose
        return p.position.x, p.position.y, yaw_of(p.orientation)

    def drive(self, v, w, duration):
        cmd = Twist()
        cmd.linear.x, cmd.angular.z = v, w
        end = time.monotonic() + duration
        while time.monotonic() < end:
            self.pub.publish(cmd)
            rclpy.spin_once(self, timeout_sec=0.05)
        for _ in range(10):
            self.pub.publish(Twist())
            rclpy.spin_once(self, timeout_sec=0.05)


def main():
    rclpy.init()
    node = CmdVelCheck()
    end = time.monotonic() + 10.0
    while node.odom is None and time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.1)
    if node.odom is None:
        print('[cmd_vel_check] Không có /odom: sim đã chạy chưa?')
        return 1
    print(f'[cmd_vel_check] /cmd_vel có {node.count_subscribers("cmd_vel")} subscriber')
    ok_all = True
    for name, v, w, t in STEPS:
        x0, y0, a0 = node.pose()
        node.drive(v, w, t)
        x1, y1, a1 = node.pose()
        dist = math.hypot(x1 - x0, y1 - y0)
        dyaw = math.degrees(math.atan2(math.sin(a1 - a0), math.cos(a1 - a0)))
        expected = abs(v or w) * t
        measured = dist if v else abs(math.radians(dyaw))
        ok = measured > 0.5 * expected
        ok_all &= ok
        print(f'  {name:10s} lệnh v={v:+.2f} w={w:+.2f} trong {t:.0f} s -> '
              f'dịch {dist:.3f} m, quay {dyaw:+6.1f} độ  [{"ĐẠT" if ok else "KHÔNG ĐẠT"}]')
    print('[cmd_vel_check] ' + ('Robot nhận lệnh /cmd_vel bình thường' if ok_all
                                else 'Robot KHÔNG phản hồi đúng lệnh /cmd_vel'))
    node.destroy_node()
    rclpy.try_shutdown()
    return 0 if ok_all else 1


if __name__ == '__main__':
    sys.exit(main())
