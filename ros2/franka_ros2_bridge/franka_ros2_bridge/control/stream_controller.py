"""1 kHz Cartesian tracking and independent, GIL-releasing gripper I/O."""

from __future__ import annotations

import math
import threading
import time
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
    """Track the latest Cartesian target through a joint-limited 1 kHz loop."""

    def __init__(
        self,
        robot_ip: str,
        *,
        max_linear_velocity: float = 0.40,
        max_linear_acceleration: float = 2.0,
        max_linear_jerk: float = 12.0,
        max_angular_velocity: float = 0.80,
        max_angular_acceleration: float = 4.0,
        max_angular_jerk: float = 25.0,
        tracking_frequency_hz: float = 5.0,
        linear_deadband: float = 0.001,
        angular_deadband: float = 0.010,
        lock_elbow: bool = True,
        control_cpu: int = 4,
        initial_sync_scale: float = 0.60,
        startup_ramp_sec: float = 0.30,
        gripper_speed: float = 0.04,
        gripper_poll_rate_hz: float = 2.0,
    ) -> None:
        if not robot_ip.strip():
            raise ValueError("robot_ip must not be empty")
        if not 0.0 < max_linear_velocity <= 2.0:
            raise ValueError("max_linear_velocity must be in (0, 2]")
        if not 0.0 < max_linear_acceleration <= 20.0:
            raise ValueError("max_linear_acceleration must be in (0, 20]")
        if not 0.0 < max_linear_jerk <= 10000.0:
            raise ValueError("max_linear_jerk must be in (0, 10000]")
        if not 0.0 < max_angular_velocity <= 3.0:
            raise ValueError("max_angular_velocity must be in (0, 3]")
        if not 0.0 < max_angular_acceleration <= 30.0:
            raise ValueError("max_angular_acceleration must be in (0, 30]")
        if not 0.0 < max_angular_jerk <= 15000.0:
            raise ValueError("max_angular_jerk must be in (0, 15000]")
        if not 0.1 <= tracking_frequency_hz <= 10.0:
            raise ValueError("tracking_frequency_hz must be in [0.1, 10]")
        if not 0.0 <= linear_deadband <= 0.05:
            raise ValueError("linear_deadband must be in [0, 0.05]")
        if not 0.0 <= angular_deadband <= 0.5:
            raise ValueError("angular_deadband must be in [0, 0.5]")
        if not isinstance(lock_elbow, bool):
            raise TypeError("lock_elbow must be a bool")
        if not isinstance(control_cpu, int) or isinstance(control_cpu, bool) or control_cpu < -1:
            raise ValueError("control_cpu must be -1 or a non-negative integer")
        if not 0.0 < initial_sync_scale <= 1.0:
            raise ValueError("initial_sync_scale must be in (0, 1]")
        if not 0.0 < startup_ramp_sec <= 5.0:
            raise ValueError("startup_ramp_sec must be in (0, 5]")
        if not 0.0 < gripper_speed <= 1.0:
            raise ValueError("gripper_speed must be in (0, 1]")
        if not 0.1 <= gripper_poll_rate_hz <= 50.0:
            raise ValueError("gripper_poll_rate_hz must be in [0.1, 50]")
        self._robot_ip = robot_ip
        self._max_linear_velocity = max_linear_velocity
        self._max_linear_acceleration = max_linear_acceleration
        self._max_linear_jerk = max_linear_jerk
        self._max_angular_velocity = max_angular_velocity
        self._max_angular_acceleration = max_angular_acceleration
        self._max_angular_jerk = max_angular_jerk
        self._tracking_frequency_hz = tracking_frequency_hz
        self._linear_deadband = linear_deadband
        self._angular_deadband = angular_deadband
        self._lock_elbow = lock_elbow
        self._control_cpu = control_cpu
        self._initial_sync_scale = initial_sync_scale
        self._startup_ramp_sec = startup_ramp_sec
        self._gripper_speed = gripper_speed
        self._stream: Any | None = None
        self._gripper: Any | None = None
        self._gripper_lock = threading.Lock()
        self._gripper_poll_stop = threading.Event()
        self._gripper_poll_thread: threading.Thread | None = None
        self._last_state: RobotState | None = None
        self._last_gripper_width = math.nan
        self._last_gripper_poll = 0.0
        self._gripper_poll_period = 1.0 / gripper_poll_rate_hz
        self._gripper_poll_errors = 0

    def connect(self) -> None:
        try:
            from franka_ros2_bridge.cartesian_stream import CartesianStreamer, GripperIO
        except ImportError as exc:
            raise RuntimeError(
                "cartesian_stream native module is missing. On the Franka computer install "
                "libfranka and pybind11, then rebuild without FRANKA_SKIP_STREAM_EXT=1."
            ) from exc
        stream = CartesianStreamer(
            self._robot_ip,
            self._max_linear_velocity,
            self._max_linear_acceleration,
            self._max_linear_jerk,
            self._max_angular_velocity,
            self._max_angular_acceleration,
            self._max_angular_jerk,
            self._tracking_frequency_hz,
            self._linear_deadband,
            self._angular_deadband,
            self._lock_elbow,
            self._control_cpu,
            self._initial_sync_scale,
            self._startup_ramp_sec,
        )
        stream.start()
        try:
            gripper = GripperIO(self._robot_ip)
            width = float(gripper.width())
            if not math.isfinite(width) or width < 0.0:
                raise RuntimeError("robot returned an invalid initial gripper width")
        except Exception:
            stream.stop()
            raise
        self._stream = stream
        self._gripper = gripper
        self._last_gripper_width = width
        self._last_gripper_poll = time.monotonic()
        self._gripper_poll_errors = 0
        self._gripper_poll_stop = threading.Event()
        self._gripper_poll_thread = threading.Thread(
            target=self._poll_gripper, args=(gripper, self._gripper_poll_stop),
            name="franka-gripper-state", daemon=True,
        )
        self._gripper_poll_thread.start()

    def disconnect(self) -> None:
        stream = self._stream
        self._stream = None
        self._gripper_poll_stop.set()
        with self._gripper_lock:
            self._gripper = None
        if stream is not None:
            stream.stop()
        if self._gripper_poll_thread is not None:
            self._gripper_poll_thread.join(timeout=2.0)
            self._gripper_poll_thread = None
        self._last_state = None
        self._last_gripper_width = math.nan
        self._last_gripper_poll = 0.0

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

    def get_diagnostics(self) -> dict[str, float | int]:
        stream = self._require_stream()
        diagnostics = dict(stream.get_diagnostics())
        diagnostics["gripper_age_ms"] = (
            1000.0 * (time.monotonic() - self._last_gripper_poll)
            if self._last_gripper_poll else -1.0
        )
        diagnostics["gripper_poll_errors"] = self._gripper_poll_errors
        return diagnostics

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
        if not gripper.move(float(command.width), self._gripper_speed):
            raise RuntimeError("gripper move did not complete")

    def _require_stream(self) -> Any:
        stream = self._stream
        if stream is None:
            raise RuntimeError("StreamController is not connected")
        stream.check_error()
        if not stream.running():
            raise RuntimeError("libfranka cartesian stream is not running")
        return stream

    def _read_gripper_width(self) -> float:
        # Called by the ROS state timer: never wait for gripper/network I/O.
        if math.isfinite(self._last_gripper_width):
            return self._last_gripper_width
        raise RuntimeError("initial gripper state is unavailable")

    def gripper_state_is_fresh(self) -> bool:
        return (
            self._last_gripper_poll > 0.0
            and time.monotonic() - self._last_gripper_poll <= max(1.0, 3.0 * self._gripper_poll_period)
        )

    def _poll_gripper(self, gripper: Any, stop: threading.Event) -> None:
        while not stop.wait(self._gripper_poll_period):
            try:
                width = float(gripper.width())
                if not math.isfinite(width) or width < 0.0:
                    raise RuntimeError("invalid gripper width")
            except Exception:
                if not stop.is_set():
                    self._gripper_poll_errors += 1
                continue
            if not stop.is_set():
                self._last_gripper_width = width
                self._last_gripper_poll = time.monotonic()
