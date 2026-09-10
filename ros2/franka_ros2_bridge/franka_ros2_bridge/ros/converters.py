"""Convert between ROS messages and shared types."""

from __future__ import annotations

from typing import Any

from franka_ros2_bridge.core.types import JOINT_NAMES, RobotState


def joint_trajectory_fields(message: Any) -> tuple[list[str], list[float]]:
    if len(message.points) != 1:
        raise ValueError("trajectory must contain exactly one point")
    return list(message.joint_names), list(message.points[0].positions)


def pose_stamped_fields(message: Any) -> tuple[str, tuple[float, float, float], tuple[float, float, float, float]]:
    position = (
        float(message.pose.position.x),
        float(message.pose.position.y),
        float(message.pose.position.z),
    )
    quaternion = (
        float(message.pose.orientation.x),
        float(message.pose.orientation.y),
        float(message.pose.orientation.z),
        float(message.pose.orientation.w),
    )
    return str(message.header.frame_id), position, quaternion


def fill_joint_state_msg(message: Any, state: RobotState, *, stamp: Any, frame_id: str) -> Any:
    message.header.stamp = stamp
    message.header.frame_id = frame_id
    message.name = list(JOINT_NAMES)
    message.position = list(state.joints)
    return message


def fill_pose_stamped_msg(message: Any, state: RobotState, *, stamp: Any, frame_id: str) -> Any:
    message.header.stamp = stamp
    message.header.frame_id = frame_id
    message.pose.position.x, message.pose.position.y, message.pose.position.z = state.end_pose.position
    (
        message.pose.orientation.x,
        message.pose.orientation.y,
        message.pose.orientation.z,
        message.pose.orientation.w,
    ) = state.end_pose.quaternion
    return message
