"""
Hiển thị các khu vực (zones.yaml) trên RViz.

  /zones/grid     nav_msgs/OccupancyGrid (latched): mỗi khu vực một giá trị 1..98 -> RViz Map với
                  color scheme "costmap" tô mỗi khu vực một màu; ô ngoài khu vực = 0 (trong suốt)
  /zones/markers  visualization_msgs/MarkerArray (latched): đường biên, nhãn "A - Phong hoc", điểm đích
                  (trụ đỏ). Khu vực robot đang tới (từ /voice_navigator/active_zone) được tô vàng, viền dày.
Nhãn trên RViz bỏ dấu vì phông chữ của RViz không có ký tự tiếng Việt (ọ, ệ ...).

Dùng:
  ros2 run tb4_voice_nav zone_publisher
"""

import os
import unicodedata

import numpy as np
from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

from tb4_voice_nav.segmentation import palette
from tb4_voice_nav.zones import ZoneMap

LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)


def default_zones_file():
    from ament_index_python.packages import get_package_share_directory
    return os.path.join(get_package_share_directory('tb4_voice_nav'), 'config', 'zones.yaml')


def ascii_text(text):
    text = text.replace('đ', 'd').replace('Đ', 'D')
    return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')


class ZonePublisher(Node):

    def __init__(self):
        super().__init__('zone_publisher')
        self.declare_parameter('zones_file', '')
        self.declare_parameter('active_topic', '/voice_navigator/active_zone')
        self.declare_parameter('label_height', 0.45)   # m, cỡ chữ nhãn
        path = os.path.expanduser(self.get_parameter('zones_file').value or default_zones_file())
        self.zones = ZoneMap(path)
        self.colors = {z.id: tuple(c / 255.0 for c in reversed(bgr))
                       for z, bgr in zip(self.zones, palette(len(self.zones)))}
        self.active = ''
        self.grid_pub = self.create_publisher(OccupancyGrid, '/zones/grid', LATCHED)
        self.marker_pub = self.create_publisher(MarkerArray, '/zones/markers', LATCHED)
        self.create_subscription(String, self.get_parameter('active_topic').value, self.on_active, LATCHED)
        self.publish_grid()
        self.publish_markers()
        self.get_logger().info(f'Phát {len(self.zones)} khu vực lên /zones/grid và /zones/markers')

    def on_active(self, msg):
        if msg.data != self.active:
            self.active = msg.data
            self.publish_markers()

    def publish_grid(self):
        zm = self.zones
        msg = OccupancyGrid()
        msg.header.frame_id = zm.frame_id
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.info.resolution = zm.resolution
        msg.info.width, msg.info.height = zm.width, zm.height
        msg.info.origin.position.x, msg.info.origin.position.y = zm.origin[0], zm.origin[1]
        msg.info.origin.orientation.w = 1.0
        data = np.zeros((zm.height, zm.width), np.int8)
        if zm.labels is not None:
            n = len(zm.zones)
            for i, z in enumerate(zm.zones):
                # rải giá trị trên thang màu costmap (1..98) theo tỷ lệ vàng để khu vực kề nhau khác màu
                data[zm.labels == z.label] = 1 + int(((i * 0.618033988749895) % 1.0) * 97) if n > 1 else 50
        msg.data = np.flipud(data).ravel().tolist()   # OccupancyGrid: hàng 0 ở dưới (gốc origin)
        self.grid_pub.publish(msg)

    def marker(self, ns, mid, mtype):
        m = Marker()
        m.header.frame_id = self.zones.frame_id
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns, m.id, m.type, m.action = ns, mid, mtype, Marker.ADD
        m.pose.orientation.w = 1.0
        return m

    def publish_markers(self):
        arr = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)
        h = self.get_parameter('label_height').value
        for i, z in enumerate(self.zones):
            active = z.id == self.active
            r, g, b = (1.0, 0.85, 0.0) if active else self.colors[z.id]

            edge = self.marker('zone_edges', i, Marker.LINE_STRIP)
            edge.scale.x = 0.12 if active else 0.05
            edge.color.r, edge.color.g, edge.color.b, edge.color.a = r, g, b, 1.0
            edge.points = [Point(x=float(x), y=float(y), z=0.02) for x, y in z.polygon + z.polygon[:1]]

            goal = self.marker('zone_goals', i, Marker.CYLINDER)
            goal.pose.position.x, goal.pose.position.y = z.goal[0], z.goal[1]
            goal.pose.position.z = 0.15
            goal.scale.x = goal.scale.y = 0.25
            goal.scale.z = 0.3
            goal.color.r, goal.color.g, goal.color.b, goal.color.a = (1.0, 0.85, 0.0, 1.0) if active \
                else (0.85, 0.1, 0.1, 0.9)

            text = self.marker('zone_labels', i, Marker.TEXT_VIEW_FACING)
            text.pose.position.x, text.pose.position.y = z.goal[0], z.goal[1] + 0.45
            text.pose.position.z = 0.8
            text.scale.z = h * (1.3 if active else 1.0)
            text.color.r, text.color.g, text.color.b, text.color.a = (1.0, 0.85, 0.0, 1.0) if active \
                else (0.05, 0.05, 0.05, 1.0)
            name = '' if z.name == f'Khu vực {z.id}' else ' - ' + z.name
            text.text = ascii_text(z.id + name)
            arr.markers += [edge, goal, text]
        self.marker_pub.publish(arr)


def main():
    rclpy.init()
    node = ZonePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
