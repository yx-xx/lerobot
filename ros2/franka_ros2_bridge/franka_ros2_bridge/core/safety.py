"""Validation and workspace / joint-limit checks."""

from __future__ import annotations

import math
from typing import Sequence

from franka_ros2_bridge.core.math_utils import finite, matrix_to_pose, pose_to_matrix
from franka_ros2_bridge.core.types import JOINT_NAMES, EndPose, EndPoseCommand, JointCommand


def validate_joint_positions(
    joint_names: Sequence[str], positions: Sequence[float]
) -> tuple[float, ...]:
    """Validate and return a seven-joint command in canonical order."""
    if tuple(joint_names) != JOINT_NAMES:
        raise ValueError(f"joint_names must exactly match {JOINT_NAMES}")
    if len(positions) != len(JOINT_NAMES):
        raise ValueError("positions must contain exactly seven values")
    result = tuple(float(value) for value in positions)
    if not finite(result):
        raise ValueError("positions must contain only finite values")
    return result


def ensure_joint_limits(
    positions: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
) -> None:
    if len(positions) != len(JOINT_NAMES) or len(lower) != len(JOINT_NAMES) or len(upper) != len(
        JOINT_NAMES
    ):
        raise ValueError("joint limits must define seven bounds")
    if any(
        value < lo or value > hi
        for value, lo, hi in zip(positions, lower, upper, strict=True)
    ):
        raise ValueError("position is outside the configured joint limits")


def ensure_workspace(
    position: Sequence[float],
    workspace_min: Sequence[float],
    workspace_max: Sequence[float],
) -> None:
    if len(position) != 3 or len(workspace_min) != 3 or len(workspace_max) != 3:
        raise ValueError("workspace bounds must define three axes")
    if any(
        value < lo or value > hi
        for value, lo, hi in zip(position, workspace_min, workspace_max, strict=True)
    ):
        raise ValueError("position is outside the configured workspace")


def build_joint_command(
    joint_names: Sequence[str],
    positions: Sequence[float],
    *,
    lower: Sequence[float],
    upper: Sequence[float],
    received_at: float,
) -> JointCommand:
    validated = validate_joint_positions(joint_names, positions)
    ensure_joint_limits(validated, lower, upper)
    return JointCommand(positions=validated, received_at=received_at)


def build_end_pose_command(
    position: Sequence[float],
    quaternion: Sequence[float],
    *,
    frame_id: str,
    base_frame: str,
    workspace_min: Sequence[float],
    workspace_max: Sequence[float],
    received_at: float,
) -> EndPoseCommand:
    if frame_id and frame_id != base_frame:
        raise ValueError(f"frame_id must be empty or equal to {base_frame!r}")
    pose_to_matrix(position, quaternion)
    xyz = (float(position[0]), float(position[1]), float(position[2]))
    qx, qy, qz, qw = (float(value) for value in quaternion)
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    unit_quaternion = (qx / norm, qy / norm, qz / norm, qw / norm)
    ensure_workspace(xyz, workspace_min, workspace_max)
    return EndPoseCommand(position=xyz, quaternion=unit_quaternion, received_at=received_at)


def validate_robot_state_joints(joints: Sequence[float]) -> tuple[float, ...]:
    result = tuple(float(value) for value in joints)
    if len(result) != len(JOINT_NAMES) or not finite(result):
        raise ValueError("robot returned an invalid joint state")
    return result


def end_pose_from_matrix(matrix: Sequence[float]) -> EndPose:
    position, quaternion = matrix_to_pose(matrix)
    return EndPose(position=position, quaternion=quaternion)
