"""Shared helpers between the ROS node and the robot controller."""

from franka_ros2_bridge.core.math_utils import matrix_to_pose, pose_to_matrix
from franka_ros2_bridge.core.safety import validate_joint_positions
from franka_ros2_bridge.core.types import (
    DEFAULT_JOINT_LOWER_LIMITS,
    DEFAULT_JOINT_UPPER_LIMITS,
    JOINT_NAMES,
)

# Backward-compatible aliases used by tests and external imports.
validate_joint_command = validate_joint_positions

__all__ = [
    "DEFAULT_JOINT_LOWER_LIMITS",
    "DEFAULT_JOINT_UPPER_LIMITS",
    "JOINT_NAMES",
    "matrix_to_pose",
    "pose_to_matrix",
    "validate_joint_command",
]
