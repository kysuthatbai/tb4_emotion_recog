"""
Mô phỏng cho bài điều khiển bằng biểu cảm: TurtleBot 4 trong Gazebo trên sân trống gesture_arena (8 x 8 m).

Dùng lại sim.launch.py của tb4_gesture_control (sửa resource path, biến GPU NVIDIA, spawn robot + dock,
prepare_robot undock rồi tiến ra giữa sân). Mặc định tiến thêm 1.0 m cho đủ chỗ chạy quỹ đạo cong.

Dùng:
  ros2 launch tb4_emotion_control sim.launch.py
  ros2 launch tb4_emotion_control sim.launch.py rviz:=true
"""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution

ARGS = [('world', 'gesture_arena'), ('model', 'standard'), ('rviz', 'false'), ('prepare', 'true'),
        ('advance_distance', '1.0'), ('nvidia', 'true')]


def generate_launch_description():
    sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([PathJoinSubstitution(
            [get_package_share_directory('tb4_gesture_control'), 'launch', 'sim.launch.py'])]),
        launch_arguments=[(name, LaunchConfiguration(name)) for name, _ in ARGS])
    return LaunchDescription([DeclareLaunchArgument(name, default_value=d) for name, d in ARGS] + [sim])
