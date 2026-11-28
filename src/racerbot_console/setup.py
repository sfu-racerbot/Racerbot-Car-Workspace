from setuptools import find_packages, setup

package_name = 'racerbot_console'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'requirements.txt']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='SFU Racerbot',
    maintainer_email='bryanmaubc@gmail.com',
    description='Terminal menu for running the cars at a test session (watch only for now).',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'console = racerbot_console.app:main',
        ],
    },
)
