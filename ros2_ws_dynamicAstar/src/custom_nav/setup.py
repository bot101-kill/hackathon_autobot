from setuptools import find_packages, setup

package_name = 'custom_nav'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kartik',
    maintainer_email='kartik@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
		'state_node = custom_nav.state_node:main',
		'planner_node = custom_nav.planner_node:main',
		'controller_node = custom_nav.controller_node:main',
		'multi_random_box = custom_nav.multi_random_box:main',
		'spawn_boxes = custom_nav.spawn_boxes:main',
		'random_box = custom_nav.random_box:main',
        ],
    },
)
