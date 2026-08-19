"""Thread-safe ROS 2 bridge around the optional ``frankx`` package."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Sequence


JOINT_NAMES = tuple(f"panda_joint{i}" for i in range(1, 8))
DEFAULT_JOINT_LOWER_LIMITS = (-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973)
DEFAULT_JOINT_UPPER_LIMITS = (2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973)

try:
    import rclpy
    from geometry_msgs.msg import PoseStamped
    from rclpy.node import Node
    from sensor_msgs.msg import JointState as JointStateMsg
    from trajectory_msgs.msg import JointTrajectory

    _ROS_IMPORT_ERROR: Exception | None = None
except ImportError as exc:  # Keep utility functions importable without ROS 2.
    rclpy = None  # type: ignore[assignment]
    PoseStamped = JointStateMsg = JointTrajectory = Any  # type: ignore[misc,assignment]
    Node = object  # type: ignore[assignment,misc]
    _ROS_IMPORT_ERROR = exc


def _finite(values: Sequence[float]) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def validate_joint_command(
    joint_names: Sequence[str], positions: Sequence[float]
) -> tuple[float, ...]:
    """Validate and return a seven-joint command in canonical order."""
    if tuple(joint_names) != JOINT_NAMES:
        raise ValueError(f"joint_names must exactly match {JOINT_NAMES}")
    if len(positions) != len(JOINT_NAMES):
        raise ValueError("positions must contain exactly seven values")
    result = tuple(float(value) for value in positions)
    if not _finite(result):
        raise ValueError("positions must contain only finite values")
    return result


def pose_to_matrix(
    position: Sequence[float], quaternion: Sequence[float]
) -> tuple[float, ...]:
    """Convert XYZ and XYZW quaternion to a column-major homogeneous matrix."""
    if len(position) != 3 or len(quaternion) != 4:
        raise ValueError("position and quaternion must have lengths 3 and 4")
    p = tuple(float(value) for value in position)
    q = tuple(float(value) for value in quaternion)
    if not _finite(p + q):
        raise ValueError("pose values must be finite")
    qx, qy, qz, qw = q
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm < 1e-12:
        raise ValueError("quaternion norm must be non-zero")
    if not math.isclose(norm, 1.0, rel_tol=1e-5, abs_tol=1e-5):
        raise ValueError("quaternion must be normalized")
    qx, qy, qz, qw = (value / norm for value in q)

    r00 = 1.0 - 2.0 * (qy * qy + qz * qz)
    r01 = 2.0 * (qx * qy - qz * qw)
    r02 = 2.0 * (qx * qz + qy * qw)
    r10 = 2.0 * (qx * qy + qz * qw)
    r11 = 1.0 - 2.0 * (qx * qx + qz * qz)
    r12 = 2.0 * (qy * qz - qx * qw)
    r20 = 2.0 * (qx * qz - qy * qw)
    r21 = 2.0 * (qy * qz + qx * qw)
    r22 = 1.0 - 2.0 * (qx * qx + qy * qy)
    x, y, z = p
    return (
        r00,
        r10,
        r20,
        0.0,
        r01,
        r11,
        r21,
        0.0,
        r02,
        r12,
        r22,
        0.0,
        x,
        y,
        z,
        1.0,
    )


def matrix_to_pose(
    matrix: Sequence[float],
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Convert a column-major homogeneous matrix to XYZ and XYZW quaternion."""
    if len(matrix) != 16:
        raise ValueError("matrix must contain exactly 16 values")
    m = tuple(float(value) for value in matrix)
    if not _finite(m):
        raise ValueError("matrix must contain only finite values")

    r00, r10, r20 = m[0], m[1], m[2]
    r01, r11, r21 = m[4], m[5], m[6]
    r02, r12, r22 = m[8], m[9], m[10]
    trace = r00 + r11 + r22
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (r21 - r12) / scale
        qy = (r02 - r20) / scale
        qz = (r10 - r01) / scale
    elif r00 > r11 and r00 > r22:
        scale = math.sqrt(1.0 + r00 - r11 - r22) * 2.0
        qw = (r21 - r12) / scale
        qx = 0.25 * scale
        qy = (r01 + r10) / scale
        qz = (r02 + r20) / scale
    elif r11 > r22:
        scale = math.sqrt(1.0 + r11 - r00 - r22) * 2.0
        qw = (r02 - r20) / scale
        qx = (r01 + r10) / scale
        qy = 0.25 * scale
        qz = (r12 + r21) / scale
    else:
        scale = math.sqrt(1.0 + r22 - r00 - r11) * 2.0
        qw = (r10 - r01) / scale
        qx = (r02 + r20) / scale
        qy = (r12 + r21) / scale
        qz = 0.25 * scale
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    quaternion = (qx / norm, qy / norm, qz / norm, qw / norm)
    return (m[12], m[13], m[14]), quaternion


@dataclass(frozen=True)
class _Command:
    mode: str
    value: tuple[float, ...]
    received_at: float


class FrankaBridgeNode(Node):
    """Publish Franka state and serialize validated motion commands."""

    def __init__(self) -> None:
        if _ROS_IMPORT_ERROR is not None:
            raise RuntimeError("ROS 2 Python packages are required") from _ROS_IMPORT_ERROR
        super().__init__("franka_bridge")

        self.declare_parameter("robot_ip", "172.16.0.2")
        self.declare_parameter("joint_state_topic", "/franka/joint_states")
        self.declare_parameter("ee_pose_topic", "/franka/ee_pose")
        self.declare_parameter("joint_command_topic", "/franka/joint_trajectory")
        self.declare_parameter("ee_command_topic", "/franka/ee_pose_command")
        self.declare_parameter("base_frame", "panda_link0")
        self.declare_parameter("publish_rate_hz", 50.0)
        self.declare_parameter("velocity_rel", 0.15)
        self.declare_parameter("acceleration_rel", 0.1)
        self.declare_parameter("command_timeout_sec", 0.5)
        self.declare_parameter("joint_lower_limits", list(DEFAULT_JOINT_LOWER_LIMITS))
        self.declare_parameter("joint_upper_limits", list(DEFAULT_JOINT_UPPER_LIMITS))
        self.declare_parameter("workspace_min", [0.20, -0.60, 0.02])
        self.declare_parameter("workspace_max", [0.80, 0.60, 0.90])

        self.base_frame = str(self.get_parameter("base_frame").value)
        self.velocity_rel = float(self.get_parameter("velocity_rel").value)
        self.acceleration_rel = float(self.get_parameter("acceleration_rel").value)
        self.command_timeout = float(self.get_parameter("command_timeout_sec").value)
        self.joint_lower_limits = tuple(
            float(v) for v in self.get_parameter("joint_lower_limits").value
        )
        self.joint_upper_limits = tuple(
            float(v) for v in self.get_parameter("joint_upper_limits").value
        )
        self.workspace_min = tuple(
            float(v) for v in self.get_parameter("workspace_min").value
        )
        self.workspace_max = tuple(
            float(v) for v in self.get_parameter("workspace_max").value
        )
        rate = float(self.get_parameter("publish_rate_hz").value)
        if rate <= 0.0 or self.command_timeout <= 0.0:
            raise ValueError("publish_rate_hz and command_timeout_sec must be positive")
        if not 0.0 < self.velocity_rel <= 1.0 or not 0.0 < self.acceleration_rel <= 1.0:
            raise ValueError("velocity_rel and acceleration_rel must be in (0, 1]")
        if (
            len(self.joint_lower_limits) != len(JOINT_NAMES)
            or len(self.joint_upper_limits) != len(JOINT_NAMES)
            or any(
                lower >= upper
                for lower, upper in zip(
                    self.joint_lower_limits, self.joint_upper_limits, strict=True
                )
            )
        ):
            raise ValueError("joint limits must define seven increasing bounds")
        if (
            len(self.workspace_min) != 3
            or len(self.workspace_max) != 3
            or any(
                lo >= hi
                for lo, hi in zip(self.workspace_min, self.workspace_max, strict=True)
            )
        ):
            raise ValueError("workspace_min/max must define three increasing bounds")

        # frankx remains an optional dependency until the node is constructed.
        try:
            import frankx
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("Install frankx and numpy in the ROS environment") from exc
        self._frankx = frankx
        self._np = np
        self._robot = frankx.Robot(str(self.get_parameter("robot_ip").value))
        self._robot.set_default_behavior()
        self._robot.recover_from_errors()

        self._hardware_lock = threading.Lock()
        self._command_lock = threading.Lock()
        self._latest_command: _Command | None = None
        self._command_event = threading.Event()
        self._stop_event = threading.Event()

        self._joint_pub = self.create_publisher(
            JointStateMsg, str(self.get_parameter("joint_state_topic").value), 10
        )
        self._pose_pub = self.create_publisher(
            PoseStamped, str(self.get_parameter("ee_pose_topic").value), 10
        )
        self.create_subscription(
            JointTrajectory,
            str(self.get_parameter("joint_command_topic").value),
            self._on_joint_command,
            10,
        )
        self.create_subscription(
            PoseStamped,
            str(self.get_parameter("ee_command_topic").value),
            self._on_pose_command,
            10,
        )
        self.create_timer(1.0 / rate, self._publish_state)
        self._worker = threading.Thread(
            target=self._command_worker, name="franka-command-worker", daemon=True
        )
        self._worker.start()

    def _replace_command(self, command: _Command) -> None:
        with self._command_lock:
            self._latest_command = command
            self._command_event.set()

    def _on_joint_command(self, message: JointTrajectory) -> None:
        try:
            if len(message.points) != 1:
                raise ValueError("trajectory must contain exactly one point")
            positions = validate_joint_command(
                message.joint_names, message.points[0].positions
            )
            if any(
                value < lower or value > upper
                for value, lower, upper in zip(
                    positions,
                    self.joint_lower_limits,
                    self.joint_upper_limits,
                    strict=True,
                )
            ):
                raise ValueError("position is outside the configured joint limits")
        except (TypeError, ValueError) as exc:
            self.get_logger().warning(f"Rejected joint command: {exc}")
            return
        self._replace_command(_Command("joint", positions, time.monotonic()))

    def _on_pose_command(self, message: PoseStamped) -> None:
        position = (
            message.pose.position.x,
            message.pose.position.y,
            message.pose.position.z,
        )
        quaternion = (
            message.pose.orientation.x,
            message.pose.orientation.y,
            message.pose.orientation.z,
            message.pose.orientation.w,
        )
        try:
            if message.header.frame_id and message.header.frame_id != self.base_frame:
                raise ValueError(
                    f"frame_id must be empty or equal to {self.base_frame!r}"
                )
            matrix = pose_to_matrix(position, quaternion)
            if any(
                value < lo or value > hi
                for value, lo, hi in zip(
                    position, self.workspace_min, self.workspace_max, strict=True
                )
            ):
                raise ValueError("position is outside the configured workspace")
        except (TypeError, ValueError) as exc:
            self.get_logger().warning(f"Rejected Cartesian command: {exc}")
            return
        self._replace_command(_Command("cartesian", matrix, time.monotonic()))

    def _command_worker(self) -> None:
        while not self._stop_event.is_set():
            self._command_event.wait(timeout=0.1)
            with self._command_lock:
                command = self._latest_command
                self._latest_command = None
                if command is None:
                    self._command_event.clear()
                    continue
            if time.monotonic() - command.received_at > self.command_timeout:
                self.get_logger().warning(f"Dropped stale {command.mode} command")
                continue
            try:
                with self._hardware_lock:
                    self._robot.velocity_rel = self.velocity_rel
                    self._robot.acceleration_rel = self.acceleration_rel
                    if command.mode == "joint":
                        motion = self._frankx.JointMotion(
                            self._frankx.JointState(list(command.value))
                        )
                    else:
                        matrix = self._np.asarray(command.value, dtype=float).reshape(
                            (4, 4), order="F"
                        )
                        affine = self._frankx.Affine(matrix)
                        motion = self._frankx.LinearMotion(affine)
                    self._robot.move(motion)
            except Exception as exc:  # Hardware/API failures must not kill the worker.
                self.get_logger().error(f"Motion failed: {exc}")

    def _publish_state(self) -> None:
        try:
            with self._hardware_lock:
                state = self._robot.read_once()
                joints = tuple(float(value) for value in state.q)
                if len(joints) != len(JOINT_NAMES) or not _finite(joints):
                    raise ValueError("robot returned an invalid joint state")
                raw_transform = state.O_T_EE
                if hasattr(raw_transform, "matrix"):
                    raw_transform = raw_transform.matrix()
                transform_array = self._np.asarray(raw_transform, dtype=float)
                transform = tuple(transform_array.reshape(-1, order="F"))
            position, quaternion = matrix_to_pose(transform)
        except Exception as exc:
            self.get_logger().error(f"State read failed: {exc}")
            return

        stamp = self.get_clock().now().to_msg()
        joint_message = JointStateMsg()
        joint_message.header.stamp = stamp
        joint_message.header.frame_id = self.base_frame
        joint_message.name = list(JOINT_NAMES)
        joint_message.position = list(joints)
        self._joint_pub.publish(joint_message)

        pose_message = PoseStamped()
        pose_message.header.stamp = stamp
        pose_message.header.frame_id = self.base_frame
        pose_message.pose.position.x, pose_message.pose.position.y, pose_message.pose.position.z = (
            position
        )
        (
            pose_message.pose.orientation.x,
            pose_message.pose.orientation.y,
            pose_message.pose.orientation.z,
            pose_message.pose.orientation.w,
        ) = quaternion
        self._pose_pub.publish(pose_message)

    def destroy_node(self) -> bool:
        self._stop_event.set()
        self._command_event.set()
        self._worker.join(timeout=2.0)
        return super().destroy_node()


def main(args: Sequence[str] | None = None) -> None:
    if rclpy is None:
        raise RuntimeError("ROS 2 Python packages are required") from _ROS_IMPORT_ERROR
    rclpy.init(args=args)
    node: FrankaBridgeNode | None = None
    try:
        node = FrankaBridgeNode()
        rclpy.spin(node)
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
