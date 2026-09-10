from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    default_config = str(
        Path(get_package_share_directory("franka_ros2_bridge"))
        / "config"
        / "franka_bridge.yaml"
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_file",
                default_value=default_config,
                description="Path to the Franka bridge parameter YAML file",
            ),
            Node(
                package="franka_ros2_bridge",
                executable="bridge_node",
                name="franka_bridge",
                output="screen",
                parameters=[LaunchConfiguration("config_file")],
            ),
        ]
    )
