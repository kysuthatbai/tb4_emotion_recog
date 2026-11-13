# Mở RViz với cấu hình rviz/yolo_nav.rviz (bản đồ, LiDAR, particle AMCL, costmap,
# đường đi Nav2, ảnh camera RGB, marker Goal và vật thể).
#
# Dùng: ros2 launch tb4_yolo_nav rviz.launch.py

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('tb4_yolo_nav')
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true',
                              choices=['true', 'false']),
        DeclareLaunchArgument('rviz_config',
                              default_value=PathJoinSubstitution([pkg, 'rviz', 'yolo_nav.rviz'])),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', LaunchConfiguration('rviz_config')],
            parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
            output='screen'),
    ])
