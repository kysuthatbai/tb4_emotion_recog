"""
Phần 2 + 3: gesture_recognition_node (webcam + MediaPipe) và robot_behavior_node (máy trạng thái -> /cmd_vel).

Chạy sau khi sim đã lên (sim.launch.py). Tham số ở config/gesture_control.yaml.

Dùng:
  ros2 launch tb4_gesture_control gesture_control.launch.py
  ros2 launch tb4_gesture_control gesture_control.launch.py camera_index:=1
"""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node

ARGUMENTS = [
    DeclareLaunchArgument('params', default_value=PathJoinSubstitution(
        [get_package_share_directory('tb4_gesture_control'), 'config', 'gesture_control.yaml'])),
    DeclareLaunchArgument('camera_index', default_value='0', description='Số của /dev/videoN'),
]


def generate_launch_description():
    params = LaunchConfiguration('params')
    recognition = Node(package='tb4_gesture_control', executable='gesture_recognition_node',
                       name='gesture_recognition_node', output='screen', emulate_tty=True,
                       parameters=[params, {'camera_index': LaunchConfiguration('camera_index')}])
    behavior = Node(package='tb4_gesture_control', executable='robot_behavior_node',
                    name='robot_behavior_node', output='screen', emulate_tty=True,
                    parameters=[params])
    return LaunchDescription(ARGUMENTS + [recognition, behavior])
