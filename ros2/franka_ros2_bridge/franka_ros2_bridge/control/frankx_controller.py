"""frankx-based Franka controller, matching the verified test_robot scripts."""

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

_JOINT_GETTERS = ("current_joint_positions", "currentJointPositions")


def joints_from_frankx_robot(robot: Any) -> tuple[float, ...]:
    """Read 7 joint angles. Prefer current_* APIs; some frankx builds only have read_once().q."""
    for name in _JOINT_GETTERS:
        getter = getattr(robot, name, None)
        if callable(getter):
            return tuple(float(value) for value in getter())

    get_state = getattr(robot, "get_state", None)
    if callable(get_state):
        joints = getattr(get_state(), "q", None)
        if joints is not None:
            return tuple(float(value) for value in joints)

    # 你现场的 frankx 没有 current_joint_positions，关节只能从这里拿。
    read_once = getattr(robot, "read_once", None)
    if callable(read_once):
        joints = getattr(read_once(), "q", None)
        if joints is not None:
            return tuple(float(value) for value in joints)

    raise RuntimeError(
        "this frankx build exposes neither current_joint_positions, "
        "currentJointPositions, get_state, nor read_once"
    )


class FrankxController:
    """Wrap frankx so the ROS node never imports it directly."""

    def __init__(
        self,
        robot_ip: str,
        *,
        velocity_rel: float = 0.15,
        acceleration_rel: float = 0.1,
        jerk_rel: float = 0.1,
        gripper_speed: float = 0.04,
    ) -> None:
        if (
            not 0.0 < velocity_rel <= 1.0
            or not 0.0 < acceleration_rel <= 1.0
            or not 0.0 < jerk_rel <= 1.0
            or not 0.0 < gripper_speed <= 1.0
        ):
            raise ValueError(
                "velocity_rel, acceleration_rel, jerk_rel, and gripper_speed must be in (0, 1]"
            )
        self._robot_ip = robot_ip
        self._velocity_rel = velocity_rel
        self._acceleration_rel = acceleration_rel
        self._jerk_rel = jerk_rel
        self._gripper_speed = gripper_speed
        self._frankx: Any | None = None
        self._Affine: Any | None = None
        self._robot: Any | None = None
        self._gripper: Any | None = None
        self._lock = threading.Lock()
        self._gripper_lock = threading.Lock()
        self._stop = threading.Event()
        self._last_state: RobotState | None = None
        self._arm_motion_thread: threading.Thread | None = None

    def connect(self) -> None:
        try:
            import frankx
        except ImportError as exc:
            raise RuntimeError("Install frankx in the ROS environment") from exc
        try:
            from pyaffx import Affine
        except ImportError:
            Affine = getattr(frankx, "Affine", None)
            if Affine is None:
                raise RuntimeError("Install pyaffx (or a frankx build that exports Affine)") from None

        with self._lock:
            self._stop.clear()
            self._frankx = frankx
            self._Affine = Affine
            robot = frankx.Robot(self._robot_ip)
            # frankx calls PyErr_CheckSignals() from the realtime thread unless this is off.
            # That segfaults CPython 3.10; verified on hardware in test_robot/03_simple_control.py.
            robot.stop_at_python_signal = False
            robot.set_default_behavior()
            robot.recover_from_errors()
            self._apply_dynamics(robot)
            self._robot = robot
            self._last_state = None
            self._arm_motion_thread = None
            gripper = frankx.Gripper(self._robot_ip)
            gripper.gripper_speed = self._gripper_speed
            self._gripper = gripper

    def disconnect(self) -> None:
        self._stop.set()
        with self._lock:
            robot = self._robot
            self._robot = None
            self._frankx = None
            self._Affine = None
            self._last_state = None
            self._arm_motion_thread = None
        with self._gripper_lock:
            self._gripper = None
        if robot is not None:
            try:
                robot.stop()
            except Exception:
                pass

    def read_state(self) -> RobotState:
        with self._lock:
            robot = self._require_robot()
            try:
                joints = validate_robot_state_joints(self._read_joints(robot))
                pose = self._read_pose(robot)
                translation = tuple(float(value) for value in pose.translation())
                quaternion = tuple(float(value) for value in pose.quaternion())
            except Exception:
                if self._last_state is None:
                    raise
                joints = self._last_state.joints
                translation = self._last_state.end_pose.position
                quaternion = self._last_state.end_pose.quaternion
        width = self._read_gripper_width()
        if len(translation) != 3 or len(quaternion) != 4:
            raise RuntimeError("robot returned an invalid end pose")
        if not math.isfinite(width):
            raise RuntimeError("robot returned an invalid gripper width")
        sampled = RobotState(
            joints=joints,
            end_pose=EndPose(position=translation, quaternion=quaternion),
            gripper_width=width,
        )
        with self._lock:
            self._last_state = sampled
        return sampled

    def start_joint(self, command: JointCommand) -> None:
        self.stop_arm()
        with self._lock:
            robot = self._require_robot()
            frankx = self._frankx
            self._apply_dynamics(robot)
            thread = robot.move_async(frankx.JointMotion(list(command.positions)))
            self._arm_motion_thread = thread

    def start_end_pose(self, command: EndPoseCommand) -> None:
        self.stop_arm()
        with self._lock:
            robot = self._require_robot()
            frankx = self._frankx
            self._apply_dynamics(robot)
            thread = robot.move_async(frankx.LinearMotion(self._affine_from_end_pose(command)))
            self._arm_motion_thread = thread

    def move_joint(self, command: JointCommand) -> None:
        self.start_joint(command)
        self._wait_for_current_motion()

    def move_end_pose(self, command: EndPoseCommand) -> None:
        self.start_end_pose(command)
        self._wait_for_current_motion()

    def stop_arm(self) -> None:
        with self._lock:
            robot = self._robot
            thread = self._arm_motion_thread
        if robot is not None:
            try:
                robot.stop()
            except Exception:
                pass
        if thread is not None:
            thread.join(timeout=2.0)
            self._clear_arm_motion(thread)

    def arm_is_moving(self) -> bool:
        with self._lock:
            thread = self._arm_motion_thread
        return thread is not None and thread.is_alive()

    def move_gripper(self, command: GripperCommand) -> None:
        with self._gripper_lock:
            gripper = self._gripper
            if gripper is None:
                raise RuntimeError("FrankxController is not connected")
            gripper.gripper_speed = self._gripper_speed
        # 不要在阻塞的 move() 期间握着锁，否则状态发布会被夹爪运动卡住。
        gripper.move(float(command.width))

    def _apply_dynamics(self, robot: Any) -> None:
        robot.velocity_rel = self._velocity_rel
        robot.acceleration_rel = self._acceleration_rel
        robot.jerk_rel = self._jerk_rel

    def _affine_from_end_pose(self, command: EndPoseCommand) -> Any:
        # Affine takes (x, y, z, qw, qx, qy, qz). ROS / current_pose() use XYZW.
        x, y, z = command.position
        qx, qy, qz, qw = command.quaternion
        return self._Affine(x, y, z, qw, qx, qy, qz)

    def _wait_for_current_motion(self) -> None:
        with self._lock:
            robot = self._robot
            thread = self._arm_motion_thread
        if thread is None:
            return
        while thread.is_alive() and not self._stop.is_set():
            thread.join(timeout=0.5)
        if self._stop.is_set() and thread.is_alive() and robot is not None:
            try:
                robot.stop()
            except Exception:
                pass
            thread.join(timeout=2.0)
        self._clear_arm_motion(thread)

    def _require_robot(self) -> Any:
        if self._robot is None or self._frankx is None or self._Affine is None:
            raise RuntimeError("FrankxController is not connected")
        return self._robot

    def _read_joints(self, robot: Any) -> tuple[float, ...]:
        return joints_from_frankx_robot(robot)

    def _read_pose(self, robot: Any) -> Any:
        return robot.current_pose()

    def _clear_arm_motion(self, thread: Any) -> None:
        with self._lock:
            if self._arm_motion_thread is thread:
                self._arm_motion_thread = None

    def _read_gripper_width(self) -> float:
        with self._gripper_lock:
            gripper = self._gripper
            if gripper is None:
                raise RuntimeError("FrankxController is not connected")
            try:
                width = float(gripper.width())
            except Exception:
                width = float("nan")
        if math.isfinite(width):
            return width
        with self._lock:
            if self._last_state is not None:
                return self._last_state.gripper_width
        raise RuntimeError("robot returned an invalid gripper width")
