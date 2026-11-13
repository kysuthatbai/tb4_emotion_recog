"""
emotion_recognition_node (webcam + DeepFace) và emotion_behavior_node (biểu cảm -> quỹ đạo -> /cmd_vel).

Chạy sau khi sim đã lên (sim.launch.py). Tham số ở config/emotion_control.yaml.

Dùng:
  ros2 launch tb4_emotion_control emotion_control.launch.py
  ros2 launch tb4_emotion_control emotion_control.launch.py camera_index:=1
  ros2 launch tb4_emotion_control emotion_control.launch.py input:=/thư_mục/ảnh   # không cần webcam
"""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node

ARGUMENTS = [
    DeclareLaunchArgument('params', default_value=PathJoinSubstitution(
        [get_package_share_directory('tb4_emotion_control'), 'config', 'emotion_control.yaml'])),
    DeclareLaunchArgument('camera_index', default_value='0', description='Số của /dev/videoN'),
    DeclareLaunchArgument('input', default_value='',
                          description="'' = webcam | file video | thư mục ảnh"),
    DeclareLaunchArgument('show_window', default_value='true', choices=['true', 'false']),
]


def generate_launch_description():
    params = LaunchConfiguration('params')
    recognition = Node(package='tb4_emotion_control', executable='emotion_recognition_node',
                       name='emotion_recognition_node', output='screen', emulate_tty=True,
                       parameters=[params, {'camera_index': LaunchConfiguration('camera_index'),
                                            'input': LaunchConfiguration('input'),
                                            'show_window': LaunchConfiguration('show_window')}])
    behavior = Node(package='tb4_emotion_control', executable='emotion_behavior_node',
                    name='emotion_behavior_node', output='screen', emulate_tty=True,
                    parameters=[params])
    return LaunchDescription(ARGUMENTS + [recognition, behavior])
