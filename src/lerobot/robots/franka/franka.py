#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import math
import threading
import time
from functools import cached_property
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_franka import FrankaConfig

logger = logging.getLogger(__name__)

try:
    import rclpy
    from geometry_msgs.msg import PoseStamped
    from rclpy.duration import Duration
    from rclpy.executors import SingleThreadedExecutor
    from sensor_msgs.msg import JointState
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    _ROS_IMPORT_ERROR: ImportError | None = None
except ImportError as error:
    rclpy = None
    PoseStamped = None
    Duration = None
    SingleThreadedExecutor = None
    JointState = None
    JointTrajectory = None
    JointTrajectoryPoint = None
    _ROS_IMPORT_ERROR = error


FRANKA_JOINT_NAMES = tuple(f"panda_joint{i}" for i in range(1, 8))
LEROBOT_JOINT_KEYS = tuple(f"j{i}.pos" for i in range(1, 8))
EE_KEYS = ("ee.x", "ee.y", "ee.z", "ee.qx", "ee.qy", "ee.qz", "ee.qw")


class FrankaRobot(Robot):
    """ROS 2 client for a Franka Panda arm."""

    config_class = FrankaConfig
    name = "franka"

    def __init__(self, config: FrankaConfig):
        super().__init__(config)
        self.config = config

        self._connected = False
        self._lock = threading.Lock()
        self._joint_positions: dict[str, float] | None = None
        self._ee_pose: tuple[float, ...] | None = None
        self._joint_state_time: float | None = None
        self._ee_pose_time: float | None = None
        self._context = None
        self._node = None
        self._executor = None
        self._executor_thread: threading.Thread | None = None
        self._joint_publisher = None
        self._ee_publisher = None
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _motors_ft(self) -> dict[str, type]:
        return dict.fromkeys(LEROBOT_JOINT_KEYS, float)

    @property
    def _ee_ft(self) -> dict[str, type]:
        return dict.fromkeys(EE_KEYS, float)

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
            for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._ee_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft if self.config.control_mode == "joint" else self._ee_ft

    @property
    def is_connected(self) -> bool:
        return self._connected and all(cam.is_connected for cam in self.cameras.values())

    def connect(self, calibrate: bool = True) -> None:
        if self._connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")
        if rclpy is None:
            raise RuntimeError(
                "ROS 2 Python packages are unavailable. Install/source rclpy, sensor_msgs, "
                "trajectory_msgs, and geometry_msgs before connecting."
            ) from _ROS_IMPORT_ERROR

        try:
            self._context = rclpy.context.Context()
            rclpy.init(context=self._context)
            self._node = rclpy.create_node(self.config.node_name, context=self._context)
            self._node.create_subscription(
                JointState,
                self.config.joint_state_topic,
                self._joint_state_callback,
                self.config.qos_depth,
            )
            self._node.create_subscription(
                PoseStamped,
                self.config.end_pose_topic,
                self._ee_pose_callback,
                self.config.qos_depth,
            )
            self._joint_publisher = self._node.create_publisher(
                JointTrajectory, self.config.joint_cmd_topic, self.config.qos_depth
            )
            self._ee_publisher = self._node.create_publisher(
                PoseStamped, self.config.end_pose_cmd_topic, self.config.qos_depth
            )
            self._executor = SingleThreadedExecutor(context=self._context)
            self._executor.add_node(self._node)
            self._executor_thread = threading.Thread(
                target=self._executor.spin,
                name=f"{self.config.node_name}-executor",
                daemon=True,
            )
            self._executor_thread.start()
            self._wait_for_initial_state()

            for camera in self.cameras.values():
                camera.connect()
            self._connected = True
        except Exception:
            self._cleanup()
            raise

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        with self._lock:
            now = time.monotonic()
            if (
                self._joint_positions is None
                or self._ee_pose is None
                or self._joint_state_time is None
                or self._ee_pose_time is None
            ):
                raise RuntimeError("Franka state is incomplete.")
            if now - self._joint_state_time > self.config.state_timeout_s:
                raise RuntimeError("Franka joint state is stale.")
            if now - self._ee_pose_time > self.config.state_timeout_s:
                raise RuntimeError("Franka end-effector pose is stale.")
            observation = {
                **{
                    key: self._joint_positions[name]
                    for key, name in zip(LEROBOT_JOINT_KEYS, FRANKA_JOINT_NAMES, strict=True)
                },
                **dict(zip(EE_KEYS, self._ee_pose, strict=True)),
            }

        for camera_key, camera in self.cameras.items():
            observation[camera_key] = camera.async_read()
        return observation

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        if self.config.control_mode == "joint":
            return self.send_joint_action(action)
        return self.send_ee_pose(action)

    def disconnect(self) -> None:
        if not self._connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        self._cleanup()

    def send_joint_action(self, action: dict[str, float]) -> dict[str, float]:
        self._require_connected()
        values = self._validate_action(action, LEROBOT_JOINT_KEYS)
        with self._lock:
            if self._joint_positions is None or self._joint_state_time is None:
                raise RuntimeError("Franka joint state is unavailable.")
            if time.monotonic() - self._joint_state_time > self.config.state_timeout_s:
                raise RuntimeError("Franka joint state is stale.")
            present = {
                f"j{i}": self._joint_positions[name]
                for i, name in enumerate(FRANKA_JOINT_NAMES, start=1)
            }

        goals = {key.removesuffix(".pos"): value for key, value in values.items()}
        if self.config.max_relative_target is not None:
            goals = ensure_safe_goal_position(
                {name: (goals[name], present[name]) for name in goals},
                self.config.max_relative_target,
            )

        message = JointTrajectory()
        message.joint_names = list(FRANKA_JOINT_NAMES)
        point = JointTrajectoryPoint()
        point.positions = [goals[f"j{i}"] for i in range(1, 8)]
        point.time_from_start = self._duration_message(self.config.command_duration_s)
        message.points = [point]
        self._joint_publisher.publish(message)
        return {f"j{i}.pos": point.positions[i - 1] for i in range(1, 8)}

    def send_ee_pose(self, action: dict[str, float]) -> dict[str, float]:
        self._require_connected()
        values = self._validate_action(action, EE_KEYS)
        target_position = tuple(values[key] for key in EE_KEYS[:3])
        target_quaternion = tuple(values[key] for key in EE_KEYS[3:])
        norm = math.sqrt(sum(value * value for value in target_quaternion))
        if not math.isclose(norm, 1.0, rel_tol=1e-5, abs_tol=1e-5):
            raise ValueError("End-effector quaternion must be normalized.")
        target_quaternion = tuple(value / norm for value in target_quaternion)

        with self._lock:
            if self._ee_pose is None or self._ee_pose_time is None:
                raise RuntimeError("Franka end-effector pose is unavailable.")
            if time.monotonic() - self._ee_pose_time > self.config.state_timeout_s:
                raise RuntimeError("Franka end-effector pose is stale.")
            current_position = self._ee_pose[:3]
            current_quaternion = self._normalize_quaternion(self._ee_pose[3:])

        delta = tuple(target - current for target, current in zip(target_position, current_position, strict=True))
        distance = math.sqrt(sum(value * value for value in delta))
        if distance > self.config.max_relative_translation:
            scale = self.config.max_relative_translation / distance
            target_position = tuple(
                current + value * scale for current, value in zip(current_position, delta, strict=True)
            )

        angle = self._quaternion_angle(current_quaternion, target_quaternion)
        if angle > self.config.max_relative_rotation:
            target_quaternion = self._slerp(
                current_quaternion, target_quaternion, self.config.max_relative_rotation / angle
            )

        message = PoseStamped()
        message.header.frame_id = self.config.base_frame
        message.header.stamp = self._node.get_clock().now().to_msg()
        message.pose.position.x, message.pose.position.y, message.pose.position.z = target_position
        (
            message.pose.orientation.x,
            message.pose.orientation.y,
            message.pose.orientation.z,
            message.pose.orientation.w,
        ) = target_quaternion
        self._ee_publisher.publish(message)
        sent = (*target_position, *target_quaternion)
        return dict(zip(EE_KEYS, sent, strict=True))

    def _joint_state_callback(self, message: Any) -> None:
        if len(message.name) != len(message.position):
            return
        positions = dict(zip(message.name, message.position, strict=True))
        if not all(name in positions and math.isfinite(positions[name]) for name in FRANKA_JOINT_NAMES):
            return
        with self._lock:
            self._joint_positions = {name: float(positions[name]) for name in FRANKA_JOINT_NAMES}
            self._joint_state_time = time.monotonic()

    def _ee_pose_callback(self, message: Any) -> None:
        if message.header.frame_id and message.header.frame_id != self.config.base_frame:
            logger.warning(
                "Ignoring Franka pose in frame %s; expected %s.",
                message.header.frame_id,
                self.config.base_frame,
            )
            return
        pose = message.pose
        values = (
            pose.position.x,
            pose.position.y,
            pose.position.z,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        )
        if not all(math.isfinite(value) for value in values):
            return
        try:
            quaternion = self._normalize_quaternion(values[3:])
        except ValueError:
            return
        with self._lock:
            self._ee_pose = (*map(float, values[:3]), *quaternion)
            self._ee_pose_time = time.monotonic()

    def _wait_for_initial_state(self) -> None:
        deadline = time.monotonic() + self.config.connect_timeout_s
        while time.monotonic() < deadline:
            with self._lock:
                if self._joint_positions is not None and self._ee_pose is not None:
                    return
            time.sleep(0.01)
        raise RuntimeError(
            "Timed out waiting for complete Franka joint state and end-effector pose."
        )

    def _cleanup(self) -> None:
        self._connected = False
        for camera in self.cameras.values():
            try:
                if camera.is_connected:
                    camera.disconnect()
            except Exception:
                logger.exception("Failed to disconnect Franka camera.")
        if self._executor is not None:
            try:
                self._executor.shutdown()
            except Exception:
                logger.exception("Failed to stop the Franka ROS executor.")
        if self._executor_thread is not None and self._executor_thread is not threading.current_thread():
            self._executor_thread.join(timeout=self.config.connect_timeout_s)
        if self._node is not None:
            try:
                self._node.destroy_node()
            except Exception:
                logger.exception("Failed to destroy the Franka ROS node.")
        if self._context is not None:
            try:
                self._context.try_shutdown()
            except Exception:
                logger.exception("Failed to shut down the Franka ROS context.")
        self._executor_thread = None
        self._executor = None
        self._node = None
        self._context = None
        self._joint_publisher = None
        self._ee_publisher = None
        with self._lock:
            self._joint_positions = None
            self._ee_pose = None
            self._joint_state_time = None
            self._ee_pose_time = None

    def _require_connected(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

    @staticmethod
    def _validate_action(action: dict[str, float], expected_keys: tuple[str, ...]) -> dict[str, float]:
        if set(action) != set(expected_keys):
            raise ValueError(f"Action keys must be exactly: {', '.join(expected_keys)}.")
        if any(
            isinstance(action[key], bool)
            or not isinstance(action[key], (int, float))
            or not math.isfinite(action[key])
            for key in expected_keys
        ):
            raise ValueError("Action values must be finite numbers.")
        return {key: float(action[key]) for key in expected_keys}

    def _duration_message(self, duration_s: float) -> Any:
        duration = Duration(seconds=duration_s)
        return duration.to_msg()

    @staticmethod
    def _normalize_quaternion(quaternion: tuple[float, ...]) -> tuple[float, ...]:
        norm = math.sqrt(sum(value * value for value in quaternion))
        if norm <= 1e-12:
            raise ValueError("Quaternion norm must be non-zero.")
        return tuple(value / norm for value in quaternion)

    @staticmethod
    def _quaternion_angle(first: tuple[float, ...], second: tuple[float, ...]) -> float:
        dot = abs(sum(a * b for a, b in zip(first, second, strict=True)))
        return 2.0 * math.acos(min(1.0, max(-1.0, dot)))

    @staticmethod
    def _slerp(
        first: tuple[float, ...], second: tuple[float, ...], fraction: float
    ) -> tuple[float, ...]:
        dot = sum(a * b for a, b in zip(first, second, strict=True))
        if dot < 0:
            second = tuple(-value for value in second)
            dot = -dot
        dot = min(1.0, max(-1.0, dot))
        if dot > 0.9995:
            interpolated = tuple(
                a + fraction * (b - a) for a, b in zip(first, second, strict=True)
            )
            return FrankaRobot._normalize_quaternion(interpolated)
        theta = math.acos(dot)
        sin_theta = math.sin(theta)
        first_scale = math.sin((1.0 - fraction) * theta) / sin_theta
        second_scale = math.sin(fraction * theta) / sin_theta
        return tuple(
            first_scale * a + second_scale * b for a, b in zip(first, second, strict=True)
        )
