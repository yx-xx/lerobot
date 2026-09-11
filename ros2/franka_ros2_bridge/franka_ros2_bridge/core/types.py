"""Shared types used by communication and control layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


JOINT_NAMES = tuple(f"panda_joint{i}" for i in range(1, 8))
DEFAULT_JOINT_LOWER_LIMITS = (-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973)
DEFAULT_JOINT_UPPER_LIMITS = (2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973)

ControlMode = Literal["joint", "cartesian"]


@dataclass(frozen=True)
class EndPose:
    """End-effector pose: XYZ metres and XYZW quaternion."""

    position: tuple[float, float, float]
    quaternion: tuple[float, float, float, float]


@dataclass(frozen=True)
class RobotState:
    """Joint angles (rad) and end-effector pose sampled together."""

    joints: tuple[float, ...]
    end_pose: EndPose


@dataclass(frozen=True)
class JointCommand:
    positions: tuple[float, ...]
    received_at: float


@dataclass(frozen=True)
class EndPoseCommand:
    """Cartesian command: XYZ metres and XYZW quaternion."""

    position: tuple[float, float, float]
    quaternion: tuple[float, float, float, float]
    received_at: float


@dataclass(frozen=True)
class MotionCommand:
    mode: ControlMode
    joint: JointCommand | None = None
    end_pose: EndPoseCommand | None = None

    @property
    def received_at(self) -> float:
        if self.mode == "joint":
            assert self.joint is not None
            return self.joint.received_at
        assert self.end_pose is not None
        return self.end_pose.received_at
