"""Hardware control interface (no ROS dependency)."""

from __future__ import annotations

from typing import Protocol

from franka_ros2_bridge.core.types import JointCommand, PoseCommand, RobotState


class RobotController(Protocol):
    """Backend that talks to the physical Franka arm."""

    def connect(self) -> None: ...

    def disconnect(self) -> None: ...

    def read_state(self) -> RobotState: ...

    def move_joint(self, command: JointCommand) -> None: ...

    def move_pose(self, command: PoseCommand) -> None: ...
