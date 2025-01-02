# launch/auto_mapping_launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    package_name = 'disaster_bot'
    nav2_bringup_package = 'nav2_bringup'

    # 설정 파일 경로
    slam_config = os.path.join(
        get_package_share_directory(package_name),
        'config',
        'slam.yaml'
    )
    nav2_config = os.path.join(
        get_package_share_directory(package_name),
        'config',
        'nav2.yaml'
    )

    # SLAM 런치 포함
    slam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory(package_name),
                'launch',
                'slam_launch.py'  # SLAM 패키지의 실제 런치 파일 이름으로 변경
            )
        ),
        launch_arguments={'params_file': slam_config}.items(),
    )

    # Navigation2 런치 포함
    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory(nav2_bringup_package),
                'launch',
                'navigation_launch.py'
            )
        ),
        launch_arguments={'params_file': nav2_config}.items(),
    )

    # AutoMappingExplorer 노드 실행
    mapping_explorer_node = Node(
        package=package_name,
        executable='auto_mapping_explorer',
        name='auto_mapping_explorer',
        output='screen',
        parameters=[slam_config, nav2_config]  # 필요 시 추가 파라미터
    )

    return LaunchDescription([
        slam_launch,
        nav2_launch,
        mapping_explorer_node
    ])
