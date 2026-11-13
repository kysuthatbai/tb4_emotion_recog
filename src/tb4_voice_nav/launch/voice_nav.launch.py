"""
Điều hướng bằng giọng nói: zone_publisher (RViz) + voice_navigator (Nav2) + speech_recognizer (micro).

Chạy sau khi mô phỏng đã lên (sim.launch.py). Tham số ở config/voice_nav.yaml.

Dùng:
  ros2 launch tb4_voice_nav voice_nav.launch.py
  ros2 launch tb4_voice_nav voice_nav.launch.py engine:=vosk rviz:=false
  ros2 launch tb4_voice_nav voice_nav.launch.py speech:=false      # chỉ gõ lệnh bằng voice_console
  ros2 launch tb4_voice_nav voice_nav.launch.py input:=/thư_mục/wav  # nhận dạng từ file thay micro
"""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node

PKG = get_package_share_directory('tb4_voice_nav')
ARGUMENTS = [
    DeclareLaunchArgument('params', default_value=PathJoinSubstitution([PKG, 'config', 'voice_nav.yaml'])),
    DeclareLaunchArgument('zones_file', default_value=PathJoinSubstitution([PKG, 'config', 'zones.yaml'])),
    DeclareLaunchArgument('speech', default_value='true', choices=['true', 'false'],
                          description='Chạy speech_recognizer'),
    DeclareLaunchArgument('engine', default_value='whisper', choices=['whisper', 'vosk']),
    DeclareLaunchArgument('input', default_value='mic', description='mic | file .wav | thư mục .wav'),
    DeclareLaunchArgument('rviz', default_value='true', choices=['true', 'false']),
    DeclareLaunchArgument('use_sim_time', default_value='true', choices=['true', 'false']),
]


def generate_launch_description():
    params = LaunchConfiguration('params')
    common = {'use_sim_time': LaunchConfiguration('use_sim_time'),
              'zones_file': LaunchConfiguration('zones_file')}
    return LaunchDescription(ARGUMENTS + [
        Node(package='tb4_voice_nav', executable='zone_publisher', name='zone_publisher',
             output='screen', parameters=[params, common]),
        Node(package='tb4_voice_nav', executable='voice_navigator', name='voice_navigator',
             output='screen', emulate_tty=True, parameters=[params, common]),
        Node(package='tb4_voice_nav', executable='speech_recognizer', name='speech_recognizer',
             output='screen', emulate_tty=True, condition=IfCondition(LaunchConfiguration('speech')),
             parameters=[params, common, {'engine': LaunchConfiguration('engine'),
                                          'input': LaunchConfiguration('input')}]),
        Node(package='rviz2', executable='rviz2', name='rviz2', output='log',
             arguments=['-d', PathJoinSubstitution([PKG, 'rviz', 'voice_nav.rviz'])],
             parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
             condition=IfCondition(LaunchConfiguration('rviz'))),
    ])
