"""
Mô phỏng cho bài điều hướng bằng giọng nói: Gazebo world maze + TurtleBot 4 + AMCL (bản đồ maze đầy đủ) + Nav2.

Khởi động tách bước (Gazebo -> sau loc_delay s: localization -> sau nav2_delay s: Nav2) vì bật tất cả
cùng lúc đôi khi làm lifecycle manager của Nav2 bị kẹt. Biến môi trường NVIDIA đặt trong
GroupAction(scoped=False): Gazebo chạy trên GPU Intel thì LiDAR trả toàn range_min.

Dùng:
  ros2 launch tb4_voice_nav sim.launch.py
  ros2 launch tb4_voice_nav sim.launch.py gui:=false        # Gazebo không cửa sổ (máy yếu / test)
  ros2 launch tb4_voice_nav sim.launch.py map:=$HOME/tb4_ws/maps/map.yaml
"""

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription, \
    SetEnvironmentVariable, TimerAction
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node

ARGUMENTS = [
    DeclareLaunchArgument('map', default_value=PathJoinSubstitution(
        [get_package_share_directory('tb4_voice_nav'), 'maps', 'maze.yaml']),
        description='Bản đồ cho AMCL / Nav2 (phải cùng bản đồ đã phân vùng)'),
    DeclareLaunchArgument('world', default_value='maze'),
    DeclareLaunchArgument('model', default_value='standard', choices=['standard', 'lite']),
    DeclareLaunchArgument('gui', default_value='true', choices=['true', 'false'],
                          description='Mở cửa sổ Gazebo (false: chỉ chạy server, render cảm biến headless)'),
    DeclareLaunchArgument('nvidia', default_value='true', choices=['true', 'false'],
                          description='Render bằng GPU NVIDIA (PRIME offload)'),
    DeclareLaunchArgument('loc_delay', default_value='15.0', description='s chờ trước khi bật AMCL'),
    DeclareLaunchArgument('nav2_delay', default_value='25.0', description='s chờ trước khi bật Nav2'),
]


def generate_launch_description():
    share = get_package_share_directory
    pkg_bringup = share('turtlebot4_ignition_bringup')
    pkg_nav = share('turtlebot4_navigation')
    world, model = LaunchConfiguration('world'), LaunchConfiguration('model')

    # Như ignition.launch.py của turtlebot4 nhưng thêm chế độ không GUI
    env = [
        SetEnvironmentVariable('IGN_GAZEBO_RESOURCE_PATH', ':'.join([
            os.path.join(pkg_bringup, 'worlds'),
            os.path.join(share('irobot_create_ignition_bringup'), 'worlds'),
            str(Path(share('turtlebot4_description')).parent.resolve()),
            str(Path(share('irobot_create_description')).parent.resolve())])),
        SetEnvironmentVariable('IGN_GUI_PLUGIN_PATH', ':'.join([
            os.path.join(share('turtlebot4_ignition_gui_plugins'), 'lib'),
            os.path.join(share('irobot_create_ignition_plugins'), 'lib')])),
    ]
    ign_gazebo = PathJoinSubstitution([share('ros_ign_gazebo'), 'launch', 'ign_gazebo.launch.py'])
    gazebo_gui = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([ign_gazebo]), condition=IfCondition(LaunchConfiguration('gui')),
        launch_arguments=[('ign_args', [world, '.sdf -r -v 2 --gui-config ',
                                        PathJoinSubstitution([pkg_bringup, 'gui', model, 'gui.config'])])])
    gazebo_server = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([ign_gazebo]), condition=UnlessCondition(LaunchConfiguration('gui')),
        launch_arguments=[('ign_args', [world, '.sdf -r -s --headless-rendering -v 2'])])
    clock_bridge = Node(package='ros_gz_bridge', executable='parameter_bridge', name='clock_bridge',
                        output='screen', arguments=['/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock'])

    nvidia = GroupAction(scoped=False, condition=IfCondition(LaunchConfiguration('nvidia')), actions=[
        SetEnvironmentVariable('__NV_PRIME_RENDER_OFFLOAD', '1'),
        SetEnvironmentVariable('__GLX_VENDOR_LIBRARY_NAME', 'nvidia'),
        SetEnvironmentVariable('__EGL_VENDOR_LIBRARY_FILENAMES',
                               '/usr/share/glvnd/egl_vendor.d/10_nvidia.json')])

    spawn = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [PathJoinSubstitution([pkg_bringup, 'launch', 'turtlebot4_spawn.launch.py'])]),
        launch_arguments=[('world', world), ('model', model)])
    localization = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([PathJoinSubstitution([pkg_nav, 'launch', 'localization.launch.py'])]),
        launch_arguments=[('use_sim_time', 'true'), ('map', LaunchConfiguration('map'))])
    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([PathJoinSubstitution([pkg_nav, 'launch', 'nav2.launch.py'])]),
        launch_arguments=[('use_sim_time', 'true')])

    ld = LaunchDescription(ARGUMENTS)
    for action in env + [nvidia, gazebo_gui, gazebo_server, clock_bridge]:
        ld.add_action(action)
    ld.add_action(spawn)
    ld.add_action(TimerAction(period=LaunchConfiguration('loc_delay'), actions=[localization]))
    ld.add_action(TimerAction(period=LaunchConfiguration('nav2_delay'), actions=[nav2]))
    return ld
