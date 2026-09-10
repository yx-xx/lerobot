"""Thread-safe latest-command mailbox with timeout."""

from __future__ import annotations

import threading
import time

from franka_ros2_bridge.core.types import JointCommand, MotionCommand, PoseCommand


class CommandQueue:
    """Keep only the newest motion command; drop it if it becomes stale."""

    def __init__(self, timeout_sec: float) -> None:
        if timeout_sec <= 0.0:
            raise ValueError("timeout_sec must be positive")
        self._timeout_sec = timeout_sec
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._latest: MotionCommand | None = None

    def push_joint(self, command: JointCommand) -> None:
        self._push(MotionCommand(mode="joint", joint=command))

    def push_pose(self, command: PoseCommand) -> None:
        self._push(MotionCommand(mode="cartesian", pose=command))

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

    def is_stale(self, command: MotionCommand) -> bool:
        return time.monotonic() - command.received_at > self._timeout_sec

    def wake(self) -> None:
        self._event.set()
