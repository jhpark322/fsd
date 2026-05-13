from setuptools import find_packages, setup

package_name = 'v2v_cooperative_pkg'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='todo',
    maintainer_email='todo@todo.com',
    description='Visual V2V 기반 협력 양보 주행 시스템',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # Python 전용 노드
            'v2v_decision_node      = v2v_cooperative_pkg.v2v_decision_node:main',
            'spatial_memory_node    = v2v_cooperative_pkg.spatial_memory_node:main',
            'visual_v2v_perception_node = v2v_cooperative_pkg.visual_v2v_perception_node:main',
            'negotiation_hmi_node   = v2v_cooperative_pkg.negotiation_hmi_node:main',
            'led_interface_node     = v2v_cooperative_pkg.led_interface_node:main',
            # Python fallback (C++ v2v_cpp_nodes 패키지 사용 권장)
            'reverse_path_planner_node_py = v2v_cooperative_pkg.reverse_path_planner_node:main',
            'chassis_control_node_py      = v2v_cooperative_pkg.chassis_control_node:main',
            'safety_supervisor_node_py    = v2v_cooperative_pkg.safety_supervisor_node:main',
        ],
    },
)
