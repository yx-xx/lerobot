"""1 kHz libfranka Cartesian pose stream, plus frankx gripper."""

from __future__ import annotations

import math
import threading
from typing import Any

from franka_ros2_bridge.core.safety import validate_robot_state_joints
from franka_ros2_bridge.core.types import (
    EndPose,
    EndPoseCommand,
    GripperCommand,
    JointCommand,
    RobotState,
)


class StreamController:
    """Keep libfranka's Cartesian pose loop open and only update the latest target."""

    def __init__(
        self,
        robot_ip: str,
        *,
        max_linear_velocity: float = 0.35,
        max_angular_velocity: float = 1.2,
        gripper_speed: float = 0.04,
    ) -> None:
        if not robot_ip.strip():
            raise ValueError("robot_ip must not be empty")
        if not 0.0 < max_linear_velocity <= 2.0:
            raise ValueError("max_linear_velocity must be in (0, 2]")
        if not 0.0 < max_angular_velocity <= 3.0:
            raise ValueError("max_angular_velocity must be in (0, 3]")
        if not 0.0 < gripper_speed <= 1.0:
            raise ValueError("gripper_speed must be in (0, 1]")
        self._robot_ip = robot_ip
        self._max_linear_velocity = max_linear_velocity
        self._max_angular_velocity = max_angular_velocity
        self._gripper_speed = gripper_speed
        self._stream: Any | None = None
        self._gripper: Any | None = None
        self._gripper_lock = threading.Lock()
        self._last_state: RobotState | None = None

    def connect(self) -> None:
        try:
            from franka_ros2_bridge.cartesian_stream import CartesianStreamer
        except ImportError as exc:
            raise RuntimeError(
                "cartesian_stream native module is missing. On the Franka computer install "
                "libfranka and pybind11, then rebuild without FRANKA_SKIP_STREAM_EXT=1."
            ) from exc
        stream = CartesianStreamer(
            self._robot_ip,
            self._max_linear_velocity,
            self._max_angular_velocity,
        )
        stream.start()
        try:
            import frankx
        except ImportError as exc:
            stream.stop()
            raise RuntimeError("Install frankx to use the gripper") from exc
        gripper = frankx.Gripper(self._robot_ip)
        gripper.gripper_speed = self._gripper_speed
        self._stream = stream
        self._gripper = gripper

    def disconnect(self) -> None:
        stream = self._stream
        self._stream = None
        with self._gripper_lock:
            self._gripper = None
        if stream is not None:
            stream.stop()
        self._last_state = None

    def read_state(self) -> RobotState:
        stream = self._require_stream()
        joints, position, quaternion = stream.get_state()
        width = self._read_gripper_width()
        sampled = RobotState(
            joints=validate_robot_state_joints(joints),
            end_pose=EndPose(
                position=(float(position[0]), float(position[1]), float(position[2])),
                quaternion=(
                    float(quaternion[0]),
                    float(quaternion[1]),
                    float(quaternion[2]),
                    float(quaternion[3]),
                ),
            ),
            gripper_width=width,
        )
        self._last_state = sampled
        return sampled

    def set_end_pose_target(self, command: EndPoseCommand) -> None:
        stream = self._require_stream()
        x, y, z = command.position
        qx, qy, qz, qw = command.quaternion
        stream.set_target(float(x), float(y), float(z), float(qx), float(qy), float(qz), float(qw))

    def move_end_pose(self, command: EndPoseCommand) -> None:
        self.set_end_pose_target(command)

    def start_end_pose(self, command: EndPoseCommand) -> None:
        self.set_end_pose_target(command)

    def move_joint(self, command: JointCommand) -> None:
        raise RuntimeError("joint streaming is not enabled; cartesian stream is running")

    def start_joint(self, command: JointCommand) -> None:
        raise RuntimeError("joint streaming is not enabled; cartesian stream is running")

    def stop_arm(self) -> None:
        return None

    def arm_is_moving(self) -> bool:
        stream = self._stream
        return stream is not None and bool(stream.running())

    def move_gripper(self, command: GripperCommand) -> None:
        with self._gripper_lock:
            gripper = self._gripper
            if gripper is None:
                raise RuntimeError("StreamController is not connected")
            gripper.gripper_speed = self._gripper_speed
        gripper.move(float(command.width))

    def _require_stream(self) -> Any:
        if self._stream is None:
            raise RuntimeError("StreamController is not connected")
        return self._stream

    def _read_gripper_width(self) -> float:
        with self._gripper_lock:
            gripper = self._gripper
            if gripper is None:
                raise RuntimeError("StreamController is not connected")
            try:
                width = float(gripper.width())
            except Exception:
                width = float("nan")
        if math.isfinite(width):
            return width if width <= 1.0 else width / 1000.0
        if self._last_state is not None:
            return self._last_state.gripper_width
        raise RuntimeError("robot returned an invalid gripper width")
