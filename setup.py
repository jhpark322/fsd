from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'lane_length_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jhp',
    maintainer_email='jhp@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
        'lane_detection_node = lane_length_pkg.lane_detection_node:main',
        'lane_decision_node = lane_length_pkg.lane_decision_node:main',
        'lane_follow_control_node = lane_length_pkg.lane_follow_control_node:main',
        'lane_guidance_mux_node = lane_length_pkg.lane_guidance_mux_node:main',
        'lane_memory_node = lane_length_pkg.lane_memory_node:main',
        ],
    },
)
