import math
import time

import pytest

from franka_ros2_bridge.core.command_queue import CommandQueue
from franka_ros2_bridge.core.math_utils import matrix_to_pose, pose_to_matrix
from franka_ros2_bridge.core.safety import build_joint_command, build_pose_command
from franka_ros2_bridge.core.types import (
    DEFAULT_JOINT_LOWER_LIMITS,
    DEFAULT_JOINT_UPPER_LIMITS,
    JOINT_NAMES,
    JointCommand,
)


def test_pose_matrix_round_trip() -> None:
    position = (0.4, -0.2, 0.5)
    quaternion = (0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5))
    matrix = pose_to_matrix(position, quaternion)

    assert matrix[12:15] == pytest.approx(position)
    restored_position, restored_quaternion = matrix_to_pose(matrix)
    assert restored_position == pytest.approx(position)
    assert abs(sum(a * b for a, b in zip(quaternion, restored_quaternion))) == pytest.approx(
        1.0
    )


def test_pose_to_matrix_rejects_unnormalized_quaternion() -> None:
    with pytest.raises(ValueError, match="normalized"):
        pose_to_matrix((0, 0, 0), (0, 0, 0, 2))


def test_validate_joint_command() -> None:
    from franka_ros2_bridge.core import validate_joint_command

    positions = validate_joint_command(JOINT_NAMES, range(7))
    assert positions == tuple(float(value) for value in range(7))


@pytest.mark.parametrize(
    ("names", "positions"),
    [
        (JOINT_NAMES[:-1], range(7)),
        (tuple(reversed(JOINT_NAMES)), range(7)),
        (JOINT_NAMES, range(6)),
        (JOINT_NAMES, [0, 0, 0, 0, 0, 0, float("nan")]),
    ],
)
def test_validate_joint_command_rejects_invalid_input(names, positions) -> None:
    from franka_ros2_bridge.core import validate_joint_command

    with pytest.raises(ValueError):
        validate_joint_command(names, positions)


def test_pose_validation_rejects_bad_values() -> None:
    with pytest.raises(ValueError):
        pose_to_matrix((0, 0, 0), (0, 0, 0, 0))
    with pytest.raises(ValueError):
        matrix_to_pose([0.0] * 15)


def test_build_joint_command_respects_limits() -> None:
    mid = tuple(
        0.5 * (lo + hi)
        for lo, hi in zip(DEFAULT_JOINT_LOWER_LIMITS, DEFAULT_JOINT_UPPER_LIMITS, strict=True)
    )
    command = build_joint_command(
        JOINT_NAMES,
        mid,
        lower=DEFAULT_JOINT_LOWER_LIMITS,
        upper=DEFAULT_JOINT_UPPER_LIMITS,
        received_at=0.0,
    )
    assert command.positions == pytest.approx(mid)
    with pytest.raises(ValueError, match="joint limits"):
        build_joint_command(
            JOINT_NAMES,
            [10.0] * 7,
            lower=DEFAULT_JOINT_LOWER_LIMITS,
            upper=DEFAULT_JOINT_UPPER_LIMITS,
            received_at=0.0,
        )


def test_build_pose_command_checks_frame_and_workspace() -> None:
    command = build_pose_command(
        (0.4, 0.0, 0.3),
        (0.0, 0.0, 0.0, 1.0),
        frame_id="panda_link0",
        base_frame="panda_link0",
        workspace_min=(0.2, -0.6, 0.02),
        workspace_max=(0.8, 0.6, 0.9),
        received_at=1.0,
    )
    assert len(command.matrix) == 16
    with pytest.raises(ValueError, match="frame_id"):
        build_pose_command(
            (0.4, 0.0, 0.3),
            (0.0, 0.0, 0.0, 1.0),
            frame_id="other",
            base_frame="panda_link0",
            workspace_min=(0.2, -0.6, 0.02),
            workspace_max=(0.8, 0.6, 0.9),
            received_at=1.0,
        )


def test_command_queue_keeps_latest_and_detects_stale() -> None:
    queue = CommandQueue(timeout_sec=0.05)
    queue.push_joint(JointCommand(positions=(0.0,) * 7, received_at=time.monotonic() - 1.0))
    queue.push_joint(JointCommand(positions=(0.1,) * 7, received_at=time.monotonic() - 1.0))
    command = queue.take(wait_timeout_sec=0.01)
    assert command is not None
    assert command.mode == "joint"
    assert command.joint is not None
    assert command.joint.positions[0] == pytest.approx(0.1)
    assert queue.is_stale(command)
