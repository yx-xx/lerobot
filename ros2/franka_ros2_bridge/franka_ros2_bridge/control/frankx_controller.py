"""frankx-based Franka controller."""

from __future__ import annotations

import threading
from typing import Any

from franka_ros2_bridge.core.safety import end_pose_from_matrix, validate_robot_state_joints
from franka_ros2_bridge.core.types import JointCommand, PoseCommand, RobotState


class FrankxController:
    """Wrap frankx so the ROS node never imports it directly."""

    def __init__(
        self,
        robot_ip: str,
        *,
        velocity_rel: float = 0.15,
        acceleration_rel: float = 0.1,
    ) -> None:
        if not 0.0 < velocity_rel <= 1.0 or not 0.0 < acceleration_rel <= 1.0:
            raise ValueError("velocity_rel and acceleration_rel must be in (0, 1]")
        self._robot_ip = robot_ip
        self._velocity_rel = velocity_rel
        self._acceleration_rel = acceleration_rel
        self._frankx: Any | None = None
        self._np: Any | None = None
        self._robot: Any | None = None
        self._lock = threading.Lock()

    def connect(self) -> None:
        try:
            import frankx
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("Install frankx and numpy in the ROS environment") from exc

        with self._lock:
            self._frankx = frankx
            self._np = np
            self._robot = frankx.Robot(self._robot_ip)
            self._robot.set_default_behavior()
            self._robot.recover_from_errors()
            self._robot.velocity_rel = self._velocity_rel
            self._robot.acceleration_rel = self._acceleration_rel

    def disconnect(self) -> None:
        with self._lock:
            self._robot = None
            self._frankx = None
            self._np = None

    def read_state(self) -> RobotState:
        with self._lock:
            robot = self._require_robot()
            np = self._np
            state = robot.read_once()
            joints = validate_robot_state_joints(state.q)
            raw_transform = state.O_T_EE
            if hasattr(raw_transform, "matrix"):
                raw_transform = raw_transform.matrix()
            transform = tuple(np.asarray(raw_transform, dtype=float).reshape(-1, order="F"))
        return RobotState(joints=joints, end_pose=end_pose_from_matrix(transform))

    def move_joint(self, command: JointCommand) -> None:
        with self._lock:
            robot = self._require_robot()
            frankx = self._frankx
            robot.velocity_rel = self._velocity_rel
            robot.acceleration_rel = self._acceleration_rel
            motion = frankx.JointMotion(frankx.JointState(list(command.positions)))
            robot.move(motion)

    def move_pose(self, command: PoseCommand) -> None:
        with self._lock:
            robot = self._require_robot()
            frankx = self._frankx
            np = self._np
            robot.velocity_rel = self._velocity_rel
            robot.acceleration_rel = self._acceleration_rel
            matrix = np.asarray(command.matrix, dtype=float).reshape((4, 4), order="F")
            motion = frankx.LinearMotion(frankx.Affine(matrix))
            robot.move(motion)

    def _require_robot(self) -> Any:
        if self._robot is None or self._frankx is None or self._np is None:
            raise RuntimeError("FrankxController is not connected")
        return self._robot
