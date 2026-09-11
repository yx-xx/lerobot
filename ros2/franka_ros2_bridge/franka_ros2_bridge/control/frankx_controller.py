"""frankx-based Franka controller, matching the verified test_robot scripts."""

from __future__ import annotations

import threading
from typing import Any

from franka_ros2_bridge.core.safety import validate_robot_state_joints
from franka_ros2_bridge.core.types import EndPose, EndPoseCommand, JointCommand, RobotState


class FrankxController:
    """Wrap frankx so the ROS node never imports it directly."""

    def __init__(
        self,
        robot_ip: str,
        *,
        velocity_rel: float = 0.15,
        acceleration_rel: float = 0.1,
        jerk_rel: float = 0.1,
    ) -> None:
        if (
            not 0.0 < velocity_rel <= 1.0
            or not 0.0 < acceleration_rel <= 1.0
            or not 0.0 < jerk_rel <= 1.0
        ):
            raise ValueError("velocity_rel, acceleration_rel, and jerk_rel must be in (0, 1]")
        self._robot_ip = robot_ip
        self._velocity_rel = velocity_rel
        self._acceleration_rel = acceleration_rel
        self._jerk_rel = jerk_rel
        self._frankx: Any | None = None
        self._Affine: Any | None = None
        self._robot: Any | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()

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

    def disconnect(self) -> None:
        self._stop.set()
        with self._lock:
            robot = self._robot
            self._robot = None
            self._frankx = None
            self._Affine = None
        if robot is not None:
            try:
                robot.stop()
            except Exception:
                pass

    def read_state(self) -> RobotState:
        with self._lock:
            robot = self._require_robot()
            state = robot.read_once()
            joints = validate_robot_state_joints(state.q)
            pose = robot.current_pose()
            translation = tuple(float(value) for value in pose.translation())
            quaternion = tuple(float(value) for value in pose.quaternion())
        if len(translation) != 3 or len(quaternion) != 4:
            raise RuntimeError("robot returned an invalid end pose")
        return RobotState(
            joints=joints,
            end_pose=EndPose(position=translation, quaternion=quaternion),
        )

    def move_joint(self, command: JointCommand) -> None:
        with self._lock:
            robot = self._require_robot()
            frankx = self._frankx
            self._apply_dynamics(robot)
            motion = frankx.JointMotion(list(command.positions))
            thread = robot.move_async(motion)
        self._wait_for_motion(robot, thread)

    def move_end_pose(self, command: EndPoseCommand) -> None:
        with self._lock:
            robot = self._require_robot()
            frankx = self._frankx
            self._apply_dynamics(robot)
            motion = frankx.LinearMotion(self._affine_from_end_pose(command))
            thread = robot.move_async(motion)
        self._wait_for_motion(robot, thread)

    def _apply_dynamics(self, robot: Any) -> None:
        robot.velocity_rel = self._velocity_rel
        robot.acceleration_rel = self._acceleration_rel
        robot.jerk_rel = self._jerk_rel

    def _affine_from_end_pose(self, command: EndPoseCommand) -> Any:
        # Affine takes (x, y, z, qw, qx, qy, qz). ROS / current_pose() use XYZW.
        x, y, z = command.position
        qx, qy, qz, qw = command.quaternion
        return self._Affine(x, y, z, qw, qx, qy, qz)

    def _wait_for_motion(self, robot: Any, thread: Any) -> None:
        while thread.is_alive() and not self._stop.is_set():
            thread.join(timeout=0.5)
        if self._stop.is_set() and thread.is_alive():
            try:
                robot.stop()
            except Exception:
                pass
            thread.join(timeout=2.0)

    def _require_robot(self) -> Any:
        if self._robot is None or self._frankx is None or self._Affine is None:
            raise RuntimeError("FrankxController is not connected")
        return self._robot
