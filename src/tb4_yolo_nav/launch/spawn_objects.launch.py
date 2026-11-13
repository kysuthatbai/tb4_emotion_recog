# Thả các vật thể trong config/objects.yaml vào world Gazebo đang chạy
# và phát marker vị trí thật của chúng lên RViz (topic /object_markers).
#
# Dùng (sau khi Gazebo đã chạy):
#   ros2 launch tb4_yolo_nav spawn_objects.launch.py
#   ros2 launch tb4_yolo_nav spawn_objects.launch.py objects:=/đường/dẫn/khác.yaml

import math

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node

from tb4_yolo_nav.objects import load_objects, resolve_model


def launch_setup(context, *args, **kwargs):
    objects_file = LaunchConfiguration('objects').perform(context)
    world, objects = load_objects(objects_file)
    world = LaunchConfiguration('world').perform(context) or world

    actions = []
    for obj in objects:
        path = resolve_model(obj['model'])
        source = 'Fuel (tải qua mạng)' if path.startswith('http') else path
        actions.append(LogInfo(msg=f"[spawn] {obj['name']} ({obj.get('label', '?')}) "
                                   f"tại ({obj['x']}, {obj['y']}) <- {source}"))
        actions.append(Node(
            package='ros_gz_sim',
            executable='create',
            name=f"spawn_{obj['name']}",
            arguments=[
                '-world', world,
                '-name', obj['name'],
                '-file', path,
                '-x', str(obj['x']),
                '-y', str(obj['y']),
                '-z', str(obj.get('z', 0.0)),
                '-Y', str(math.radians(obj.get('yaw_deg', 0.0))),
            ],
            output='screen'))

    actions.append(Node(
        package='tb4_yolo_nav',
        executable='object_markers',
        name='object_markers',
        parameters=[{'objects_file': objects_file,
                     'use_sim_time': LaunchConfiguration('use_sim_time')}],
        output='screen'))
    return actions


def generate_launch_description():
    pkg = get_package_share_directory('tb4_yolo_nav')
    return LaunchDescription([
        DeclareLaunchArgument(
            'objects', default_value=PathJoinSubstitution([pkg, 'config', 'objects.yaml']),
            description='File YAML liệt kê vật thể cần thả vào world'),
        DeclareLaunchArgument(
            'world', default_value='',
            description='Tên world Gazebo (mặc định lấy theo khóa "world" trong file YAML)'),
        DeclareLaunchArgument('use_sim_time', default_value='true',
                              choices=['true', 'false']),
        OpaqueFunction(function=launch_setup),
    ])
