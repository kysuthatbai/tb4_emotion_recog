from glob import glob

from setuptools import setup

package_name = 'tb4_emotion_control'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sonvu',
    maintainer_email='ngsonvu2005008@gmail.com',
    description='TurtleBot 4 facial-expression control with DeepFace',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'emotion_recognition_node = tb4_emotion_control.emotion_recognition_node:main',
            'emotion_behavior_node = tb4_emotion_control.emotion_behavior_node:main',
        ],
    },
)
