from glob import glob

from setuptools import setup

package_name = 'tb4_voice_nav'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml') + glob('config/*.png')),
        ('share/' + package_name + '/maps', glob('maps/*')),
        ('share/' + package_name + '/rviz', glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sonvu',
    maintainer_email='ngsonvu2005008@gmail.com',
    description='TurtleBot 4: phân vùng bản đồ, gắn nhãn khu vực và điều hướng Nav2 bằng giọng nói',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'segment_map = tb4_voice_nav.segment_map:main',
            'zone_publisher = tb4_voice_nav.zone_publisher:main',
            'speech_recognizer = tb4_voice_nav.speech_recognizer:main',
            'voice_navigator = tb4_voice_nav.voice_navigator:main',
            'voice_console = tb4_voice_nav.voice_console:main',
        ],
    },
)
