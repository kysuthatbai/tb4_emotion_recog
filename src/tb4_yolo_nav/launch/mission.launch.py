# Phần 3 + 4: chạy node YOLO và node điều hướng tự động qua các Goal trong goals.yaml.
# Chạy sau khi mô phỏng đã bật với localization:=true nav2:=true (và đã spawn vật thể).
#
# Dùng:
#   ros2 launch tb4_yolo_nav mission.launch.py
#   ros2 launch tb4_yolo_nav mission.launch.py show_window:=true rviz:=true
#   ros2 launch tb4_yolo_nav mission.launch.py goals_file:=/đường/dẫn/goals.yaml yolo:=false

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from tb4_yolo_nav.goal_navigator import DEFAULT_GOALS
from tb4_yolo_nav.yolo_detector import DEFAULT_MODEL


def generate_launch_description():
    pkg = get_package_share_directory('tb4_yolo_nav')
    use_sim_time = LaunchConfiguration('use_sim_time')
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true', choices=['true', 'false']),
        DeclareLaunchArgument('goals_file', default_value=DEFAULT_GOALS),
        DeclareLaunchArgument('model', default_value=DEFAULT_MODEL,
                              description='Trọng số YOLO (.pt)'),
        DeclareLaunchArgument('show_window', default_value='false', choices=['true', 'false'],
                              description='Hiện cửa sổ OpenCV kết quả YOLO'),
        DeclareLaunchArgument('yolo', default_value='true', choices=['true', 'false']),
        DeclareLaunchArgument('rviz', default_value='false', choices=['true', 'false']),

        Node(
            package='tb4_yolo_nav',
            executable='yolo_detector',
            name='yolo_detector',
            parameters=[{'use_sim_time': use_sim_time,
                         'model': LaunchConfiguration('model'),
                         'show_window': LaunchConfiguration('show_window')}],
            output='screen',
            emulate_tty=True,
            condition=IfCondition(LaunchConfiguration('yolo'))),

        Node(
            package='tb4_yolo_nav',
            executable='goal_navigator',
            name='goal_navigator',
            parameters=[{'use_sim_time': use_sim_time,
                         'goals_file': LaunchConfiguration('goals_file')}],
            output='screen',
            emulate_tty=True),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(pkg, 'launch', 'rviz.launch.py')),
            launch_arguments={'use_sim_time': use_sim_time}.items(),
            condition=IfCondition(LaunchConfiguration('rviz'))),
    ])
