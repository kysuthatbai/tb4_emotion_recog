from glob import glob
import os

from setuptools import setup

package_name = 'tb4_yolo_nav'


def tree(src):
    """Liệt kê (thư mục đích, [file]) cho toàn bộ cây thư mục src, giữ nguyên cấu trúc."""
    out = []
    for root, _, files in os.walk(src):
        if files:
            out.append((os.path.join('share', package_name, root),
                        [os.path.join(root, f) for f in files]))
    return out


setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/rviz', glob('rviz/*.rviz')),
    ] + tree('models'),
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sonvu',
    maintainer_email='ngsonvu2005008@gmail.com',
    description='TurtleBot 4 localization, goal survey, Nav2 navigation and YOLO detection',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'set_initial_pose = tb4_yolo_nav.set_initial_pose:main',
            'localization_check = tb4_yolo_nav.localization_check:main',
            'goal_recorder = tb4_yolo_nav.goal_recorder:main',
            'object_markers = tb4_yolo_nav.object_markers:main',
            'goal_navigator = tb4_yolo_nav.goal_navigator:main',
            'yolo_detector = tb4_yolo_nav.yolo_detector:main',
        ],
    },
)
