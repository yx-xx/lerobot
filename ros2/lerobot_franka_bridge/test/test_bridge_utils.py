import math

import pytest

from lerobot_franka_bridge.bridge_node import (
    JOINT_NAMES,
    matrix_to_pose,
    pose_to_matrix,
    validate_joint_command,
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
    with pytest.raises(ValueError):
        validate_joint_command(names, positions)


def test_pose_validation_rejects_bad_values() -> None:
    with pytest.raises(ValueError):
        pose_to_matrix((0, 0, 0), (0, 0, 0, 0))
    with pytest.raises(ValueError):
        matrix_to_pose([0.0] * 15)
