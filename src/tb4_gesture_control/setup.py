from glob import glob

from setuptools import setup

package_name = 'tb4_gesture_control'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/worlds', glob('worlds/*.sdf')),
        ('share/' + package_name + '/models', glob('models/*.task')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sonvu',
    maintainer_email='ngsonvu2005008@gmail.com',
    description='TurtleBot 4 hand-gesture control with MediaPipe',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'webcam_view = tb4_gesture_control.webcam_view:main',
            'cmd_vel_check = tb4_gesture_control.cmd_vel_check:main',
            'prepare_robot = tb4_gesture_control.prepare_robot:main',
            'gesture_recognition_node = tb4_gesture_control.gesture_recognition_node:main',
            'robot_behavior_node = tb4_gesture_control.robot_behavior_node:main',
        ],
    },
)
