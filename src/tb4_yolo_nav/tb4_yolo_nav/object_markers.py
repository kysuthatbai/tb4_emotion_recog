"""Phát vị trí thật của các vật thể (theo objects.yaml) thành MarkerArray để xem trên RViz."""

import os

from ament_index_python.packages import get_package_share_directory
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from visualization_msgs.msg import Marker, MarkerArray

from tb4_yolo_nav.objects import load_objects


class ObjectMarkers(Node):

    def __init__(self):
        super().__init__('object_markers')
        default = os.path.join(get_package_share_directory('tb4_yolo_nav'),
                               'config', 'objects.yaml')
        self.declare_parameter('objects_file', default)
        self.declare_parameter('frame_id', 'map')

        _, objects = load_objects(self.get_parameter('objects_file').value)
        frame = self.get_parameter('frame_id').value

        # Transient local: RViz mở sau vẫn nhận được marker
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub = self.create_publisher(MarkerArray, 'object_markers', qos)

        msg = MarkerArray()
        for i, obj in enumerate(objects):
            body = Marker()
            body.header.frame_id = frame
            body.ns = 'objects'
            body.id = i
            body.type = Marker.CYLINDER
            body.pose.position.x = float(obj['x'])
            body.pose.position.y = float(obj['y'])
            body.pose.position.z = 0.25
            body.pose.orientation.w = 1.0
            body.scale.x = body.scale.y = 0.35
            body.scale.z = 0.5
            body.color.r, body.color.g, body.color.b, body.color.a = 1.0, 0.55, 0.0, 0.8

            text = Marker()
            text.header.frame_id = frame
            text.ns = 'object_labels'
            text.id = i
            text.type = Marker.TEXT_VIEW_FACING
            text.pose.position.x = float(obj['x'])
            text.pose.position.y = float(obj['y'])
            text.pose.position.z = 0.8
            text.pose.orientation.w = 1.0
            text.scale.z = 0.3
            text.color.r = text.color.g = text.color.b = text.color.a = 1.0
            text.text = obj.get('label', obj['name'])

            msg.markers += [body, text]

        self.pub.publish(msg)
        self.get_logger().info(f'Đã phát {len(objects)} marker vật thể lên /object_markers')


def main():
    rclpy.init()
    node = ObjectMarkers()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
