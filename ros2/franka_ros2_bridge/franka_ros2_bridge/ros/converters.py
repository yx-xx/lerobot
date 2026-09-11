"""Convert between ROS messages and shared types."""

from __future__ import annotations

from typing import Any

from franka_ros2_bridge.core.types import JOINT_NAMES, RobotState


def joint_cmd_from_msg(message: Any) -> tuple[list[str], list[float]]:
    names = list(message.name)
    positions = [float(value) for value in message.position]
    if len(names) != len(positions):
        raise ValueError("joint_cmd name and position must have the same length")
    return names, positions


def end_pose_cmd_from_msg(
    message: Any,
) -> tuple[str, tuple[float, float, float], tuple[float, float, float, float]]:
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


def to_joint_state_msg(message: Any, state: RobotState, *, stamp: Any, frame_id: str) -> Any:
    message.header.stamp = stamp
    message.header.frame_id = frame_id
    message.name = list(JOINT_NAMES)
    message.position = list(state.joints)
    return message


def to_end_pose_msg(message: Any, state: RobotState, *, stamp: Any, frame_id: str) -> Any:
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
