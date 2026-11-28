from setuptools import find_packages, setup

package_name = 'car_explorer'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/explore_launch.py']),
        ('share/' + package_name + '/config', ['config/car_explorer.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='SFU Racerbot',
    maintainer_email='bryanmaubc@gmail.com',
    description="Steps gap_follow's live settings up to find the car's limits. Publishes no drive topic.",
    license='MIT',
    tests_require=['pytest'],
    entry_points={'console_scripts': ['explorer_node = car_explorer.explorer_node:main']},
)
