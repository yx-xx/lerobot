"""ROS 2 node that wires communication to the robot controller."""

from __future__ import annotations

import math
import threading
import time
from typing import Any, Sequence

from franka_ros2_bridge.core.command_queue import CommandQueue
from franka_ros2_bridge.core.safety import (
    build_end_pose_command,
    build_gripper_command,
    build_joint_command,
)
from franka_ros2_bridge.core.types import (
    DEFAULT_GRIPPER_MAX,
    DEFAULT_GRIPPER_MIN,
    DEFAULT_JOINT_LOWER_LIMITS,
    DEFAULT_JOINT_UPPER_LIMITS,
    JOINT_NAMES,
)
from franka_ros2_bridge.control.frankx_controller import FrankxController
from franka_ros2_bridge.control.stream_controller import StreamController
from franka_ros2_bridge.ros.converters import (
    end_pose_cmd_from_msg,
    joint_cmd_from_msg,
    gripper_cmd_from_msg,
    to_end_pose_msg,
    to_gripper_state_msg,
    to_joint_state_msg,
)

# Re-export pure helpers so existing imports from bridge_node keep working.
from franka_ros2_bridge.core.math_utils import matrix_to_pose, pose_to_matrix
from franka_ros2_bridge.core.safety import validate_joint_positions as validate_joint_command

try:
    import rclpy
    from geometry_msgs.msg import PoseStamped
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
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
        self.declare_parameter("gripper_state_topic", "/franka/gripper_state")
        self.declare_parameter("gripper_cmd_topic", "/franka/gripper_cmd")
        self.declare_parameter("gripper_min", DEFAULT_GRIPPER_MIN)
        self.declare_parameter("gripper_max", DEFAULT_GRIPPER_MAX)
        self.declare_parameter("gripper_speed", 0.04)
        self.declare_parameter("cartesian_mode", "stream")
        self.declare_parameter("max_linear_velocity", 0.40)
        self.declare_parameter("max_linear_acceleration", 2.0)
        self.declare_parameter("max_linear_jerk", 12.0)
        self.declare_parameter("max_angular_velocity", 0.80)
        self.declare_parameter("max_angular_acceleration", 4.0)
        self.declare_parameter("max_angular_jerk", 25.0)
        self.declare_parameter("tracking_frequency_hz", 5.0)
        self.declare_parameter("linear_deadband", 0.001)
        self.declare_parameter("angular_deadband", 0.010)
        self.declare_parameter("lock_elbow", True)
        self.declare_parameter("control_cpu", 4)
        self.declare_parameter("initial_sync_scale", 0.60)
        self.declare_parameter("startup_ramp_sec", 0.30)
        self.declare_parameter("gripper_command_deadband", 0.001)
        self.declare_parameter("gripper_poll_rate_hz", 2.0)

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
        self.gripper_min = float(self.get_parameter("gripper_min").value)
        self.gripper_max = float(self.get_parameter("gripper_max").value)
        rate = float(self.get_parameter("publish_rate_hz").value)
        velocity_rel = float(self.get_parameter("velocity_rel").value)
        acceleration_rel = float(self.get_parameter("acceleration_rel").value)
        jerk_rel = float(self.get_parameter("jerk_rel").value)
        gripper_speed = float(self.get_parameter("gripper_speed").value)
        cartesian_mode = str(self.get_parameter("cartesian_mode").value).strip().lower()
        max_linear_velocity = float(self.get_parameter("max_linear_velocity").value)
        max_linear_acceleration = float(self.get_parameter("max_linear_acceleration").value)
        max_linear_jerk = float(self.get_parameter("max_linear_jerk").value)
        max_angular_velocity = float(self.get_parameter("max_angular_velocity").value)
        max_angular_acceleration = float(self.get_parameter("max_angular_acceleration").value)
        max_angular_jerk = float(self.get_parameter("max_angular_jerk").value)
        tracking_frequency_hz = float(self.get_parameter("tracking_frequency_hz").value)
        linear_deadband = float(self.get_parameter("linear_deadband").value)
        angular_deadband = float(self.get_parameter("angular_deadband").value)
        lock_elbow = bool(self.get_parameter("lock_elbow").value)
        control_cpu = int(self.get_parameter("control_cpu").value)
        initial_sync_scale = float(self.get_parameter("initial_sync_scale").value)
        startup_ramp_sec = float(self.get_parameter("startup_ramp_sec").value)
        gripper_poll_rate_hz = float(self.get_parameter("gripper_poll_rate_hz").value)
        self.gripper_command_deadband = float(
            self.get_parameter("gripper_command_deadband").value
        )

        if rate <= 0.0 or self.command_timeout <= 0.0:
            raise ValueError("publish_rate_hz and command_timeout_sec must be positive")
        if not math.isfinite(self.gripper_command_deadband) or self.gripper_command_deadband < 0.0:
            raise ValueError("gripper_command_deadband must be finite and non-negative")
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
        if (
            not math.isfinite(self.gripper_min)
            or not math.isfinite(self.gripper_max)
            or self.gripper_min < 0.0
            or self.gripper_max < self.gripper_min
        ):
            raise ValueError("gripper_min/max must be finite and increasing")

        robot_ip = str(self.get_parameter("robot_ip").value)
        if cartesian_mode == "stream":
            self._streaming = True
            self._controller = StreamController(
                robot_ip,
                max_linear_velocity=max_linear_velocity,
                max_linear_acceleration=max_linear_acceleration,
                max_linear_jerk=max_linear_jerk,
                max_angular_velocity=max_angular_velocity,
                max_angular_acceleration=max_angular_acceleration,
                max_angular_jerk=max_angular_jerk,
                tracking_frequency_hz=tracking_frequency_hz,
                linear_deadband=linear_deadband,
                angular_deadband=angular_deadband,
                lock_elbow=lock_elbow,
                control_cpu=control_cpu,
                initial_sync_scale=initial_sync_scale,
                startup_ramp_sec=startup_ramp_sec,
                gripper_speed=gripper_speed,
                gripper_poll_rate_hz=gripper_poll_rate_hz,
            )
        elif cartesian_mode == "ptp":
            self._streaming = False
            self._controller = FrankxController(
                robot_ip,
                velocity_rel=velocity_rel,
                acceleration_rel=acceleration_rel,
                jerk_rel=jerk_rel,
                gripper_speed=gripper_speed,
            )
        else:
            raise ValueError("cartesian_mode must be 'stream' or 'ptp'")
        self._controller.connect()
        self._controller.read_state()
        self._state_error_logged = False
        self._controller_fault = threading.Event()
        self._last_gripper_command: float | None = None
        self._last_delayed_callbacks = 0
        self._last_target_messages = 0
        self._ros_target_messages = 0
        self._last_ros_target_messages = 0
        self._last_ros_target_time = 0.0
        self._ros_max_gap_ms = 0.0
        self._dispatch_delay_ms = 0.0
        self._last_startup_state = 0
        self._last_diagnostics_time = time.monotonic()
        if self._streaming:
            self.get_logger().info(
                "Resolved-rate Cartesian stream is running at 1 kHz; "
                "absolute latest-target tracking with Ruckig 0.15.3; waiting for a target"
            )
        self._commands = CommandQueue(self.command_timeout)
        self._stop_event = threading.Event()

        self._joint_state_pub = self.create_publisher(
            JointStateMsg, str(self.get_parameter("joint_state_topic").value), 10
        )
        self._end_pose_pub = self.create_publisher(
            PoseStamped, str(self.get_parameter("end_pose_topic").value), 10
        )
        self._gripper_state_pub = self.create_publisher(
            JointStateMsg, str(self.get_parameter("gripper_state_topic").value), 10
        )
        command_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.create_subscription(
            JointStateMsg,
            str(self.get_parameter("joint_cmd_topic").value),
            self._on_joint_cmd,
            command_qos,
        )
        self.create_subscription(
            PoseStamped,
            str(self.get_parameter("end_pose_cmd_topic").value),
            self._on_end_pose_cmd,
            command_qos,
        )
        self.create_subscription(
            JointStateMsg,
            str(self.get_parameter("gripper_cmd_topic").value),
            self._on_gripper_cmd,
            command_qos,
        )
        self.create_timer(1.0 / rate, self._publish_state)
        if self._streaming:
            self.create_timer(5.0, self._log_stream_diagnostics)
            self.create_timer(0.1, self._watch_startup)
        self._worker = threading.Thread(
            target=self._command_worker, name="franka-command-worker", daemon=True
        )
        self._gripper_worker = threading.Thread(
            target=self._gripper_worker_loop, name="franka-gripper-worker", daemon=True
        )
        self._worker.start()
        self._gripper_worker.start()

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
        if self._last_ros_target_time:
            self._ros_max_gap_ms = max(
                self._ros_max_gap_ms, 1000.0 * (command.received_at - self._last_ros_target_time)
            )
        self._last_ros_target_time = command.received_at
        self._ros_target_messages += 1
        self._commands.push_end_pose(command)

    def _on_gripper_cmd(self, message: JointStateMsg) -> None:
        try:
            width = gripper_cmd_from_msg(message)
            command = build_gripper_command(
                width,
                min_width=self.gripper_min,
                max_width=self.gripper_max,
                received_at=time.monotonic(),
            )
        except (TypeError, ValueError) as exc:
            self.get_logger().warning(f"Rejected gripper_cmd: {exc}")
            return
        self._commands.push_gripper(command)

    def _command_worker(self) -> None:
        while not self._stop_event.is_set():
            command = self._commands.take(wait_timeout_sec=0.02)
            if command is None:
                continue
            if self._commands.is_stale(command):
                self.get_logger().warning(f"Dropped stale {command.mode} command")
                continue
            if command.mode == "joint":
                if self._streaming:
                    self.get_logger().warning("Ignoring joint_cmd while cartesian stream is running")
                    continue
                try:
                    assert command.joint is not None
                    self._controller.start_joint(command.joint)
                except Exception as exc:
                    self.get_logger().error(f"Motion failed: {exc}")
                continue
            assert command.end_pose is not None
            if self._controller_fault.is_set():
                continue
            try:
                self._dispatch_delay_ms = 1000.0 * (time.monotonic() - command.end_pose.received_at)
                self._controller.set_end_pose_target(command.end_pose)
            except Exception as exc:
                if self._streaming:
                    if not self._controller_fault.is_set():
                        self.get_logger().error(
                            "Cartesian stream failed; target updates are disabled until the "
                            f"bridge is restarted: {exc}"
                        )
                    self._controller_fault.set()
                else:
                    self.get_logger().error(f"Failed to update cartesian target: {exc}")

    def _gripper_worker_loop(self) -> None:
        while not self._stop_event.is_set():
            command = self._commands.take_gripper(wait_timeout_sec=0.1)
            if command is None:
                continue
            if self._commands.is_gripper_stale(command):
                self.get_logger().warning("Dropped stale gripper command")
                continue
            if (
                self._last_gripper_command is not None
                and abs(command.width - self._last_gripper_command)
                < self.gripper_command_deadband
            ):
                continue
            try:
                self._controller.move_gripper(command)
                self._last_gripper_command = command.width
            except Exception as exc:
                self.get_logger().error(f"Gripper motion failed: {exc}")

    def _publish_state(self) -> None:
        try:
            state = self._controller.read_state()
            self._state_error_logged = False
        except Exception as exc:
            # Never stamp an old sample as fresh: clients use receipt time for
            # their stale-state watchdog and must notice a stopped control loop.
            if not self._state_error_logged:
                self.get_logger().error(f"State read failed; state publication stopped: {exc}")
                self._state_error_logged = True
            if self._streaming:
                self._controller_fault.set()
            return

        stamp = self.get_clock().now().to_msg()
        joint_message = to_joint_state_msg(
            JointStateMsg(), state, stamp=stamp, frame_id=self.base_frame
        )
        end_pose_message = to_end_pose_msg(
            PoseStamped(), state, stamp=stamp, frame_id=self.base_frame
        )
        gripper_message = to_gripper_state_msg(
            JointStateMsg(), state, stamp=stamp, frame_id=self.base_frame
        )
        self._joint_state_pub.publish(joint_message)
        self._end_pose_pub.publish(end_pose_message)
        if not self._streaming or self._controller.gripper_state_is_fresh():
            self._gripper_state_pub.publish(gripper_message)

    def _log_stream_diagnostics(self) -> None:
        if self._controller_fault.is_set():
            return
        try:
            diagnostics = self._controller.get_diagnostics()
        except Exception:
            return
        success_rate = float(diagnostics["control_command_success_rate"])
        delayed_callbacks = int(diagnostics["delayed_callbacks"])
        delayed_delta = delayed_callbacks - self._last_delayed_callbacks
        target_messages = int(diagnostics["target_messages"])
        now = time.monotonic()
        diagnostics_period = max(now - self._last_diagnostics_time, 1e-6)
        target_rate = (target_messages - self._last_target_messages) / diagnostics_period
        ros_rate = (self._ros_target_messages - self._last_ros_target_messages) / diagnostics_period
        ros_gap_ms = self._ros_max_gap_ms
        if self._last_ros_target_time:
            ros_gap_ms = max(ros_gap_ms, 1000.0 * (now - self._last_ros_target_time))
        target_age_ms = float(diagnostics["target_age_ms"])
        startup_states = {0: "waiting", 1: "ramping", 2: "live"}
        startup = startup_states.get(int(diagnostics["startup_state"]), "unknown")
        message = (
            f"FCI success_rate={success_rate:.3f} "
            f"max_period_ms={float(diagnostics['max_period_ms']):.3f} "
            f"delayed_callbacks={delayed_callbacks}(+{delayed_delta}) "
            f"control_cpu={int(diagnostics['control_cpu'])} "
            f"cpu_migrations={int(diagnostics['control_cpu_migrations'])} "
            f"elbow_locked={bool(diagnostics['elbow_locked'])} "
            f"startup={startup} "
            f"min_sigma={float(diagnostics['minimum_singular_value']):.3f} "
            f"joint_scale={float(diagnostics['joint_velocity_scale']):.3f} "
            f"planner_scale={float(diagnostics['planner_velocity_scale']):.3f}/"
            f"{float(diagnostics['planner_acceleration_scale']):.3f}/"
            f"{float(diagnostics['planner_jerk_scale']):.3f} "
            f"trajectory_sync_fallbacks={int(diagnostics['trajectory_sync_fallbacks'])} "
            f"position_error_mm={1000.0 * float(diagnostics['position_error']):.1f} "
            f"orientation_error_deg={math.degrees(float(diagnostics['orientation_error'])):.1f} "
            f"target_hz={target_rate:.1f} target_age_ms={target_age_ms:.1f} "
            f"ros_hz={ros_rate:.1f} ros_max_gap_ms={ros_gap_ms:.1f} "
            f"dispatch_ms={self._dispatch_delay_ms:.1f} "
            f"gripper_age_ms={float(diagnostics['gripper_age_ms']):.1f} "
            f"gripper_poll_errors={int(diagnostics['gripper_poll_errors'])}"
        )
        target_is_stale = target_messages > 0 and target_age_ms > 200.0
        if (
            success_rate < 0.99
            or target_is_stale
            or not self._controller.gripper_state_is_fresh()
        ):
            self.get_logger().warning(message)
        else:
            self.get_logger().info(message)
        self._last_delayed_callbacks = delayed_callbacks
        self._last_target_messages = target_messages
        self._last_ros_target_messages = self._ros_target_messages
        self._ros_max_gap_ms = 0.0
        self._last_diagnostics_time = now

    def _watch_startup(self) -> None:
        if self._controller_fault.is_set():
            return
        try:
            state = int(self._controller.get_diagnostics()["startup_state"])
        except Exception:
            return
        if state == self._last_startup_state:
            return
        self._last_startup_state = state
        if state == 1:
            self.get_logger().info(
                "Absolute tracking started; startup limits are ramping while following the latest target"
            )
        elif state == 2:
            self.get_logger().info(
                "Startup ramp complete; absolute tracking continues at full configured limits"
            )

    def destroy_node(self) -> bool:
        self._stop_event.set()
        self._commands.wake()
        try:
            self._controller.disconnect()
        except Exception:
            self.get_logger().exception("Failed to disconnect Franka controller")
        self._worker.join(timeout=2.0)
        self._gripper_worker.join(timeout=2.0)
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
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
