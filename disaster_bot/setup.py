from setuptools import find_packages, setup

package_name = 'disaster_bot'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='viator',
    maintainer_email='pedralation@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mapping = disaster_bot.auto_mapping_explorer:main',
            'dibot = disaster_bot.firex_person:main',
            'test = disaster_bot.firex_person_webcam:main',
        ],
    },
)
