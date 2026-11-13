"""
Phần 1: khởi chạy TurtleBot 4 trong Gazebo (Ignition Fortress) trên sân trống gesture_arena.

Tương đương turtlebot4_ignition.launch.py nhưng thêm thư mục worlds của package này vào
IGN_GAZEBO_RESOURCE_PATH (launch gốc ghi đè biến này nên không tìm thấy world ngoài), rồi:
  - spawn robot + dock (turtlebot4_spawn.launch.py, truyền đúng tên world cho bridge LiDAR/camera)
  - prepare_robot: undock và đưa robot ra giữa sân (tắt bằng prepare:=false)

Dùng:
  ros2 launch tb4_gesture_control sim.launch.py
  ros2 launch tb4_gesture_control sim.launch.py world:=warehouse rviz:=true
"""

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, GroupAction, IncludeLaunchDescription,
                            SetEnvironmentVariable)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node

ARGUMENTS = [
    DeclareLaunchArgument('world', default_value='gesture_arena',
                          description='Tên world (không có .sdf): gesture_arena, maze, warehouse, depot'),
    DeclareLaunchArgument('model', default_value='standard', choices=['standard', 'lite']),
    DeclareLaunchArgument('rviz', default_value='false', choices=['true', 'false']),
    DeclareLaunchArgument('prepare', default_value='true', choices=['true', 'false'],
                          description='Undock và tiến ra giữa sân sau khi spawn'),
    DeclareLaunchArgument('advance_distance', default_value='0.6',
                          description='Quãng tiến thêm sau undock (m)'),
    DeclareLaunchArgument('nvidia', default_value='true', choices=['true', 'false'],
                          description='Render bằng GPU NVIDIA (PRIME offload); GPU Intel làm hỏng LiDAR'),
]


def generate_launch_description():
    share = lambda pkg: get_package_share_directory(pkg)  # noqa: E731
    pkg_self = share('tb4_gesture_control')
    pkg_tb4_bringup = share('turtlebot4_ignition_bringup')
    pkg_create_bringup = share('irobot_create_ignition_bringup')

    resource_path = SetEnvironmentVariable('IGN_GAZEBO_RESOURCE_PATH', ':'.join([
        os.path.join(pkg_self, 'worlds'),
        os.path.join(pkg_tb4_bringup, 'worlds'),
        os.path.join(pkg_create_bringup, 'worlds'),
        str(Path(share('turtlebot4_description')).parent.resolve()),
        str(Path(share('irobot_create_description')).parent.resolve())]))
    gui_plugin_path = SetEnvironmentVariable('IGN_GUI_PLUGIN_PATH', ':'.join([
        os.path.join(share('turtlebot4_ignition_gui_plugins'), 'lib'),
        os.path.join(share('irobot_create_ignition_plugins'), 'lib')]))
    # scoped=False: mặc định GroupAction khôi phục môi trường khi ra khỏi group -> Gazebo mất biến NVIDIA,
    # chạy trên GPU Intel và LiDAR trả toàn range_min
    nvidia = GroupAction(scoped=False, condition=IfCondition(LaunchConfiguration('nvidia')), actions=[
        SetEnvironmentVariable('__NV_PRIME_RENDER_OFFLOAD', '1'),
        SetEnvironmentVariable('__GLX_VENDOR_LIBRARY_NAME', 'nvidia'),
        SetEnvironmentVariable('__EGL_VENDOR_LIBRARY_FILENAMES',
                               '/usr/share/glvnd/egl_vendor.d/10_nvidia.json')])

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [PathJoinSubstitution([share('ros_ign_gazebo'), 'launch', 'ign_gazebo.launch.py'])]),
        launch_arguments=[('ign_args', [
            LaunchConfiguration('world'), '.sdf', ' -r', ' -v 2', ' --gui-config ',
            PathJoinSubstitution([pkg_tb4_bringup, 'gui', LaunchConfiguration('model'), 'gui.config'])])])

    clock_bridge = Node(package='ros_gz_bridge', executable='parameter_bridge', name='clock_bridge',
                        output='screen',
                        arguments=['/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock'])

    spawn = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [PathJoinSubstitution([pkg_tb4_bringup, 'launch', 'turtlebot4_spawn.launch.py'])]),
        launch_arguments=[('world', LaunchConfiguration('world')),
                          ('model', LaunchConfiguration('model')),
                          ('rviz', LaunchConfiguration('rviz'))])

    prepare = Node(package='tb4_gesture_control', executable='prepare_robot', name='prepare_robot',
                   output='screen', condition=IfCondition(LaunchConfiguration('prepare')),
                   parameters=[{'advance_distance': LaunchConfiguration('advance_distance')}])

    ld = LaunchDescription(ARGUMENTS)
    for action in (resource_path, gui_plugin_path, nvidia, gazebo, clock_bridge, spawn, prepare):
        ld.add_action(action)
    return ld
