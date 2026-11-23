"""Thread-safe latest-command mailbox with timeout."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import replace

from franka_ros2_bridge.core.types import EndPoseCommand, GripperCommand, JointCommand, MotionCommand


class CommandQueue:
    """Keep only the newest motion command; drop it if it becomes stale."""

    def __init__(self, timeout_sec: float, *, gripper_deadband: float = 0.001) -> None:
        if timeout_sec <= 0.0:
            raise ValueError("timeout_sec must be positive")
        if not math.isfinite(gripper_deadband) or gripper_deadband < 0.0:
            raise ValueError("gripper_deadband must be finite and non-negative")
        self._timeout_sec = timeout_sec
        self._gripper_deadband = gripper_deadband
        self._gripper_direction_width: float | None = None
        self._gripper_closing: bool | None = None
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._gripper_event = threading.Event()
        self._latest: MotionCommand | None = None
        self._latest_gripper: GripperCommand | None = None

    def push_joint(self, command: JointCommand) -> None:
        self._push(MotionCommand(mode="joint", joint=command))

    def push_end_pose(self, command: EndPoseCommand) -> None:
        self._push(MotionCommand(mode="cartesian", end_pose=command))

    def push_gripper(self, command: GripperCommand) -> None:
        with self._lock:
            previous_width = self._gripper_direction_width
            changed = previous_width is None or abs(command.width - previous_width) > self._gripper_deadband
            if command.closing is not None:
                self._gripper_closing = command.closing
            elif previous_width is not None and changed:
                self._gripper_closing = command.width < previous_width
            if changed:
                # Keep the anchor for sub-deadband samples so slow input motion
                # accumulates instead of being filtered out indefinitely.
                self._gripper_direction_width = command.width
            # Detect reversals before overwriting a pending sample: the worker
            # may be blocked in stop() and miss the narrowest requested opening.
            self._latest_gripper = replace(command, closing=self._gripper_closing)
            self._gripper_event.set()

    def _push(self, command: MotionCommand) -> None:
        with self._lock:
            self._latest = command
            self._event.set()

    def take(self, wait_timeout_sec: float = 0.1) -> MotionCommand | None:
        """Take the newest command, or None if the mailbox is empty."""
        self._event.wait(timeout=wait_timeout_sec)
        with self._lock:
            command = self._latest
            self._latest = None
            if command is None:
                self._event.clear()
            return command

    def take_gripper(self, wait_timeout_sec: float = 0.1) -> GripperCommand | None:
        self._gripper_event.wait(timeout=wait_timeout_sec)
        with self._lock:
            command = self._latest_gripper
            self._latest_gripper = None
            if command is None:
                self._gripper_event.clear()
            return command

    def is_stale(self, command: MotionCommand) -> bool:
        return time.monotonic() - command.received_at > self._timeout_sec

    def is_gripper_stale(self, command: GripperCommand) -> bool:
        return time.monotonic() - command.received_at > self._timeout_sec

    def wake(self) -> None:
        self._event.set()
        self._gripper_event.set()
