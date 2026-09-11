"""ROS 2 node that wires communication to the robot controller."""

from __future__ import annotations

import threading
import time
from typing import Any, Sequence

from franka_ros2_bridge.core.command_queue import CommandQueue
from franka_ros2_bridge.core.safety import build_end_pose_command, build_joint_command
from franka_ros2_bridge.core.types import (
    DEFAULT_JOINT_LOWER_LIMITS,
    DEFAULT_JOINT_UPPER_LIMITS,
    JOINT_NAMES,
)
from franka_ros2_bridge.control.frankx_controller import FrankxController
from franka_ros2_bridge.ros.converters import (
    end_pose_cmd_from_msg,
    joint_cmd_from_msg,
    to_end_pose_msg,
    to_joint_state_msg,
)

# Re-export pure helpers so existing imports from bridge_node keep working.
from franka_ros2_bridge.core.math_utils import matrix_to_pose, pose_to_matrix
from franka_ros2_bridge.core.safety import validate_joint_positions as validate_joint_command

try:
    import rclpy
    from geometry_msgs.msg import PoseStamped
    from rclpy.node import Node
    from sensor_msgs.msg import JointState as JointStateMsg

    _ROS_IMPORT_ERROR: Exception | None = None
except ImportError as exc:
    rclpy = None  # type: ignore[assignment]
    PoseStamped = JointStateMsg = Any  # type: ignore[misc,assignment]
    Node = object  # type: ignore[assignment,misc]
    _ROS_IMPORT_ERROR = exc


class FrankaBridgeNode(Node):
    """Thin ROS front-end: subscribe / publish / queue / call the controller."""

    def __init__(self) -> None:
        if _ROS_IMPORT_ERROR is not None:
            raise RuntimeError("ROS 2 Python packages are required") from _ROS_IMPORT_ERROR
        super().__init__("franka_bridge")

        self.declare_parameter("robot_ip", "172.16.0.2")
        self.declare_parameter("joint_state_topic", "/franka/joint_state")
        self.declare_parameter("end_pose_topic", "/franka/end_pose")
        self.declare_parameter("joint_cmd_topic", "/franka/joint_cmd")
        self.declare_parameter("end_pose_cmd_topic", "/franka/end_pose_cmd")
        self.declare_parameter("base_frame", "panda_link0")
        self.declare_parameter("publish_rate_hz", 50.0)
        self.declare_parameter("velocity_rel", 0.15)
        self.declare_parameter("acceleration_rel", 0.1)
        self.declare_parameter("jerk_rel", 0.1)
        self.declare_parameter("command_timeout_sec", 0.5)
        self.declare_parameter("joint_lower_limits", list(DEFAULT_JOINT_LOWER_LIMITS))
        self.declare_parameter("joint_upper_limits", list(DEFAULT_JOINT_UPPER_LIMITS))
        self.declare_parameter("workspace_min", [0.20, -0.60, 0.02])
        self.declare_parameter("workspace_max", [0.80, 0.60, 0.90])

        self.base_frame = str(self.get_parameter("base_frame").value)
        self.command_timeout = float(self.get_parameter("command_timeout_sec").value)
        self.joint_lower_limits = tuple(
            float(v) for v in self.get_parameter("joint_lower_limits").value
        )
        self.joint_upper_limits = tuple(
            float(v) for v in self.get_parameter("joint_upper_limits").value
        )
        self.workspace_min = tuple(float(v) for v in self.get_parameter("workspace_min").value)
        self.workspace_max = tuple(float(v) for v in self.get_parameter("workspace_max").value)
        rate = float(self.get_parameter("publish_rate_hz").value)
        velocity_rel = float(self.get_parameter("velocity_rel").value)
        acceleration_rel = float(self.get_parameter("acceleration_rel").value)
        jerk_rel = float(self.get_parameter("jerk_rel").value)

        if rate <= 0.0 or self.command_timeout <= 0.0:
            raise ValueError("publish_rate_hz and command_timeout_sec must be positive")
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

        self._controller = FrankxController(
            str(self.get_parameter("robot_ip").value),
            velocity_rel=velocity_rel,
            acceleration_rel=acceleration_rel,
            jerk_rel=jerk_rel,
        )
        self._controller.connect()
        self._commands = CommandQueue(self.command_timeout)
        self._stop_event = threading.Event()

        self._joint_state_pub = self.create_publisher(
            JointStateMsg, str(self.get_parameter("joint_state_topic").value), 10
        )
        self._end_pose_pub = self.create_publisher(
            PoseStamped, str(self.get_parameter("end_pose_topic").value), 10
        )
        self.create_subscription(
            JointStateMsg,
            str(self.get_parameter("joint_cmd_topic").value),
            self._on_joint_cmd,
            10,
        )
        self.create_subscription(
            PoseStamped,
            str(self.get_parameter("end_pose_cmd_topic").value),
            self._on_end_pose_cmd,
            10,
        )
        self.create_timer(1.0 / rate, self._publish_state)
        self._worker = threading.Thread(
            target=self._command_worker, name="franka-command-worker", daemon=True
        )
        self._worker.start()

    def _on_joint_cmd(self, message: JointStateMsg) -> None:
        try:
            names, positions = joint_cmd_from_msg(message)
            command = build_joint_command(
                names,
                positions,
                lower=self.joint_lower_limits,
                upper=self.joint_upper_limits,
                received_at=time.monotonic(),
            )
        except (TypeError, ValueError) as exc:
            self.get_logger().warning(f"Rejected joint_cmd: {exc}")
            return
        self._commands.push_joint(command)

    def _on_end_pose_cmd(self, message: PoseStamped) -> None:
        try:
            frame_id, position, quaternion = end_pose_cmd_from_msg(message)
            command = build_end_pose_command(
                position,
                quaternion,
                frame_id=frame_id,
                base_frame=self.base_frame,
                workspace_min=self.workspace_min,
                workspace_max=self.workspace_max,
                received_at=time.monotonic(),
            )
        except (TypeError, ValueError) as exc:
            self.get_logger().warning(f"Rejected end_pose_cmd: {exc}")
            return
        self._commands.push_end_pose(command)

    def _command_worker(self) -> None:
        while not self._stop_event.is_set():
            command = self._commands.take(wait_timeout_sec=0.1)
            if command is None:
                continue
            if self._commands.is_stale(command):
                self.get_logger().warning(f"Dropped stale {command.mode} command")
                continue
            try:
                if command.mode == "joint":
                    assert command.joint is not None
                    self._controller.move_joint(command.joint)
                else:
                    assert command.end_pose is not None
                    self._controller.move_end_pose(command.end_pose)
            except Exception as exc:
                self.get_logger().error(f"Motion failed: {exc}")

    def _publish_state(self) -> None:
        try:
            state = self._controller.read_state()
        except Exception as exc:
            self.get_logger().error(f"State read failed: {exc}")
            return

        stamp = self.get_clock().now().to_msg()
        joint_message = to_joint_state_msg(
            JointStateMsg(), state, stamp=stamp, frame_id=self.base_frame
        )
        end_pose_message = to_end_pose_msg(
            PoseStamped(), state, stamp=stamp, frame_id=self.base_frame
        )
        self._joint_state_pub.publish(joint_message)
        self._end_pose_pub.publish(end_pose_message)

    def destroy_node(self) -> bool:
        self._stop_event.set()
        self._commands.wake()
        try:
            self._controller.disconnect()
        except Exception:
            self.get_logger().exception("Failed to disconnect Franka controller")
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
