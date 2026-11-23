"""Grasp/release and command interruption without hardware or ROS."""

import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest

from franka_ros2_bridge.bridge_node import FrankaBridgeNode
from franka_ros2_bridge.control.stream_controller import StreamController
from franka_ros2_bridge.core.command_queue import CommandQueue
from franka_ros2_bridge.core.types import EndPoseCommand, GripperCommand


def test_closing_grasps_and_opening_releases_even_below_object_width():
    controller = StreamController("172.16.0.2", gripper_force=15.0)
    gripper = MagicMock()
    gripper.stop.return_value = gripper.grasp.return_value = gripper.move.return_value = True
    controller._gripper = gripper
    controller._last_gripper_width = 0.04  # Object keeps the actual opening at 40 mm.

    controller.move_gripper(GripperCommand(0.005, time.monotonic()))
    # The operator is opening, although this target is still below the object width.
    controller.move_gripper(GripperCommand(0.02, time.monotonic()))
    controller.move_gripper(GripperCommand(0.08, time.monotonic()))

    assert gripper.mock_calls == [
        call.grasp(0.005, 0.10, 15.0, 0.005, 0.08),
        call.stop(), call.move(0.02, 0.10),
        call.move(0.08, 0.10),
    ]


def test_failed_grasp_retry_still_uses_grasp():
    controller = StreamController("172.16.0.2")
    controller._gripper = MagicMock()
    controller._gripper.stop.return_value = True
    controller._gripper.grasp.side_effect = [False, True]
    controller._last_gripper_width = 0.08
    command = GripperCommand(0.005, time.monotonic())
    with pytest.raises(RuntimeError, match="grasp did not complete"):
        controller.move_gripper(command)
    controller.move_gripper(command)
    assert controller._gripper.grasp.call_count == 2
    controller._gripper.move.assert_not_called()


def test_failed_stop_does_not_start_another_gripper_motion():
    controller = StreamController("172.16.0.2")
    controller._gripper = MagicMock()
    controller._gripper.stop.return_value = False
    controller._gripper_needs_stop = True
    controller._last_gripper_width = 0.04
    with pytest.raises(RuntimeError, match="stop did not complete"):
        controller.move_gripper(GripperCommand(0.08, time.monotonic()))
    controller._gripper.move.assert_not_called()
    controller._gripper.grasp.assert_not_called()


def test_outdated_native_gripper_fails_before_connecting_robot(monkeypatch):
    native = SimpleNamespace(CartesianStreamer=MagicMock(), GripperIO=object)
    monkeypatch.setitem(sys.modules, "franka_ros2_bridge.cartesian_stream", native)
    with pytest.raises(RuntimeError, match="outdated.*rebuild"):
        StreamController("172.16.0.2").connect()
    native.CartesianStreamer.assert_not_called()


def test_outdated_position_controller_fails_before_connecting_robot(monkeypatch):
    native = SimpleNamespace(
        CartesianStreamer=MagicMock(), GripperIO=SimpleNamespace(grasp=MagicMock(), stop=MagicMock())
    )
    monkeypatch.setitem(sys.modules, "franka_ros2_bridge.cartesian_stream", native)
    with pytest.raises(RuntimeError, match="outdated.*rebuild.*position boundary guard"):
        StreamController("172.16.0.2").connect()
    native.CartesianStreamer.assert_not_called()


@pytest.mark.parametrize("force", [0.0, -1.0, 71.0, float("nan"), float("inf")])
def test_invalid_grasp_force_is_rejected(force):
    with pytest.raises(ValueError, match="gripper_force"):
        StreamController("172.16.0.2", gripper_force=force)


def _worker_node(controller, deadband=0.001, *, queue=None, retarget_interval=0.1):
    logger = MagicMock()
    node = SimpleNamespace(
        _controller=controller,
        _commands=queue if queue is not None else CommandQueue(2.0),
        _stop_event=threading.Event(),
        _last_gripper_command=0.04,
        gripper_command_deadband=deadband,
        gripper_retarget_interval_sec=retarget_interval,
        gripper_retarget_deadband=0.002,
        get_logger=lambda: logger,
    )
    worker = threading.Thread(target=FrankaBridgeNode._stream_gripper_worker_loop, args=(node,))
    worker.start()
    return node, worker, logger


def _shutdown(node, worker):
    node._stop_event.set()
    node._commands.wake()
    worker.join(timeout=2.0)
    assert not worker.is_alive()


@pytest.mark.parametrize("deadband", [0.0, 0.001])
def test_open_interrupts_blocked_close_and_only_latest_target_runs(deadband):
    started, stop_called, finish_close, opened = (threading.Event() for _ in range(4))
    widths = []
    active_calls = []

    def move(command):
        assert not active_calls, "Two blocking gripper commands overlap"
        active_calls.append(command.width)
        widths.append(command.width)
        try:
            if command.width < 0.01:
                started.set()
                assert finish_close.wait(timeout=1.0)
                raise RuntimeError("grasp interrupted by stop")
            opened.set()
        finally:
            active_calls.pop()

    controller = SimpleNamespace(move_gripper=move, stop_gripper=stop_called.set)
    node, worker, logger = _worker_node(controller, deadband)
    try:
        node._commands.push_gripper(GripperCommand(0.005, time.monotonic()))
        assert started.wait(timeout=1.0)
        # Duplicate 30 Hz targets must not start another grasp.
        node._commands.push_gripper(GripperCommand(0.005, time.monotonic()))
        node._commands.push_gripper(GripperCommand(0.02, time.monotonic()))
        assert stop_called.wait(timeout=1.0)
        node._commands.push_gripper(GripperCommand(0.08, time.monotonic()))
        finish_close.set()
        assert opened.wait(timeout=1.0)
        assert widths == [0.005, 0.08]
        logger.error.assert_not_called()  # An intentional stop is not a hardware fault.
    finally:
        finish_close.set()
        _shutdown(node, worker)


@pytest.mark.parametrize("widths", [(0.03, 0.01), (0.05, 0.08)])
def test_large_same_direction_target_does_not_wait_for_blocking_motion(widths):
    started, stopped, followed = (threading.Event() for _ in range(3))
    executed = []

    def move(command):
        executed.append(command.width)
        if len(executed) == 1:
            started.set()
            assert stopped.wait(timeout=1.0)
            raise RuntimeError("intentionally interrupted")
        followed.set()

    controller = SimpleNamespace(move_gripper=move, stop_gripper=stopped.set)
    node, worker, logger = _worker_node(controller, retarget_interval=0.05)
    try:
        node._commands.push_gripper(GripperCommand(widths[0], time.monotonic()))
        assert started.wait(timeout=1.0)
        node._commands.push_gripper(GripperCommand(widths[1], time.monotonic()))
        assert followed.wait(timeout=0.5), "New target waited for the old blocking motion"
        assert executed == list(widths)
        logger.error.assert_not_called()
    finally:
        stopped.set()
        _shutdown(node, worker)


def test_small_same_direction_change_does_not_repeatedly_stop_motion():
    started, finish, followed = (threading.Event() for _ in range(3))
    executed = []

    def move(command):
        executed.append(command.width)
        if len(executed) == 1:
            started.set()
            assert finish.wait(timeout=1.0)
        else:
            followed.set()

    controller = SimpleNamespace(move_gripper=move, stop_gripper=MagicMock())
    node, worker, _ = _worker_node(controller, retarget_interval=0.04)
    try:
        node._commands.push_gripper(GripperCommand(0.03, time.monotonic()))
        assert started.wait(timeout=1.0)
        node._commands.push_gripper(GripperCommand(0.0285, time.monotonic()))
        assert not followed.wait(timeout=0.12)
        controller.stop_gripper.assert_not_called()
        finish.set()
        assert followed.wait(timeout=1.0)
    finally:
        finish.set()
        _shutdown(node, worker)


def test_idle_gripper_starts_without_an_extra_stop_round_trip():
    controller = StreamController("172.16.0.2")
    controller._gripper = MagicMock()
    controller._gripper.grasp.return_value = True
    controller._last_gripper_width = 0.04
    controller.move_gripper(GripperCommand(0.01, time.monotonic()))
    controller._gripper.stop.assert_not_called()
    controller._gripper.grasp.assert_called_once()


def test_interrupted_grasp_is_not_stopped_a_second_time_before_release():
    controller = StreamController("172.16.0.2")
    gripper = MagicMock()
    gripper.stop.return_value = gripper.grasp.return_value = gripper.move.return_value = True
    controller._gripper = gripper
    controller._last_gripper_width = 0.04
    controller.move_gripper(GripperCommand(0.01, time.monotonic()))
    controller.stop_gripper()
    controller.move_gripper(GripperCommand(0.08, time.monotonic()))
    gripper.stop.assert_called_once()


def test_release_retries_stop_if_first_stop_precedes_sdk_motion():
    entered, start_sdk, sdk_running, stopped, opened = (threading.Event() for _ in range(5))
    stop_calls = []

    def move(command):
        if command.width < 0.01:
            entered.set()
            assert start_sdk.wait(timeout=1.0)
            sdk_running.set()
            assert stopped.wait(timeout=1.0)
        else:
            opened.set()

    def stop():
        stop_calls.append(True)
        if sdk_running.is_set():
            stopped.set()
        else:
            start_sdk.set()

    node, worker, _ = _worker_node(SimpleNamespace(move_gripper=move, stop_gripper=stop))
    try:
        node._commands.push_gripper(GripperCommand(0.005, time.monotonic()))
        assert entered.wait(timeout=1.0)
        node._commands.push_gripper(GripperCommand(0.08, time.monotonic()))
        assert opened.wait(timeout=1.0)
        assert len(stop_calls) >= 2
    finally:
        start_sdk.set()
        stopped.set()
        _shutdown(node, worker)


def test_shutdown_stops_a_blocked_grasp():
    entered, stopped = threading.Event(), threading.Event()

    def move(command):
        entered.set()
        assert stopped.wait(timeout=1.0)

    node, worker, _ = _worker_node(SimpleNamespace(move_gripper=move, stop_gripper=stopped.set))
    node._commands.push_gripper(GripperCommand(0.005, time.monotonic()))
    try:
        assert entered.wait(timeout=1.0)
    finally:
        _shutdown(node, worker)
    assert stopped.is_set()


def test_blocked_grasp_allows_arm_updates_and_can_be_released():
    entered, stopped, opened = (threading.Event() for _ in range(3))

    def grasp(*args):
        entered.set()
        assert stopped.wait(timeout=1.0)
        return False  # Native grasp returns after cancellation.

    def stop():
        if entered.is_set():
            stopped.set()
        return True

    controller = StreamController("172.16.0.2")
    controller._gripper = SimpleNamespace(
        grasp=grasp, stop=stop, move=lambda *args: opened.set() or True,
    )
    controller._last_gripper_width = 0.04
    controller._stream = SimpleNamespace(
        check_error=lambda: None, running=lambda: True, set_target=MagicMock(),
        get_state=lambda: ((0.0,) * 7, (0.4, 0.0, 0.3), (0.0, 0.0, 0.0, 1.0)),
    )
    node, worker, logger = _worker_node(controller)
    try:
        node._commands.push_gripper(GripperCommand(0.005, time.monotonic()))
        assert entered.wait(timeout=1.0)
        assert controller.read_state().gripper_width == 0.04
        controller.set_end_pose_target(EndPoseCommand((0.45, 0.0, 0.3), (0.0, 0.0, 0.0, 1.0), time.monotonic()))
        controller._stream.set_target.assert_called_once_with(0.45, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0)
        assert not stopped.is_set()
        node._commands.push_gripper(GripperCommand(0.08, time.monotonic()))
        assert opened.wait(timeout=1.0)
        logger.error.assert_not_called()
    finally:
        stopped.set()
        _shutdown(node, worker)


class _ObservedQueue(CommandQueue):
    """Acknowledge a command once the worker starts its next iteration."""

    def __init__(self):
        super().__init__(2.0)
        self._taken = None
        self._processed = set()
        self._processed_condition = threading.Condition()

    def take_gripper(self, wait_timeout_sec=0.1):
        if self._taken is not None:
            with self._processed_condition:
                self._processed.add(self._taken.received_at)
                self._processed_condition.notify_all()
        self._taken = super().take_gripper(wait_timeout_sec)
        return self._taken

    def send(self, width):
        command = GripperCommand(width, time.monotonic())
        self.push_gripper(command)
        with self._processed_condition:
            assert self._processed_condition.wait_for(
                lambda: command.received_at in self._processed, timeout=1.0
            ), "Gripper worker did not process the target"


@pytest.mark.parametrize("widths", [(0.03, 0.02, 0.01), (0.05, 0.06, 0.08)])
def test_same_direction_targets_do_not_interrupt_and_only_latest_runs(widths):
    started, finish_first, followed = (threading.Event() for _ in range(3))
    executed = []
    queue = _ObservedQueue()

    def move(command):
        executed.append(command.width)
        if len(executed) == 1:
            started.set()
            assert finish_first.wait(timeout=1.0)
        else:
            followed.set()

    controller = SimpleNamespace(move_gripper=move, stop_gripper=MagicMock())
    node, worker, logger = _worker_node(controller, queue=queue)
    try:
        queue.send(widths[0])
        assert started.wait(timeout=1.0)
        queue.send(widths[1])
        queue.send(widths[2])
        controller.stop_gripper.assert_not_called()
        assert executed == [widths[0]]
        finish_first.set()
        assert followed.wait(timeout=1.0)
        assert executed == [widths[0], widths[2]]
        logger.error.assert_not_called()
    finally:
        finish_first.set()
        _shutdown(node, worker)


def test_reversal_of_pending_close_releases_below_active_grasp_target():
    started, stopped, released = (threading.Event() for _ in range(3))
    queue = _ObservedQueue()
    controller = StreamController("172.16.0.2")
    controller._last_gripper_width = 0.04

    def grasp(*args):
        started.set()
        assert stopped.wait(timeout=1.0)
        return False

    def stop():
        if started.is_set():
            stopped.set()
        return True

    gripper = SimpleNamespace(
        grasp=grasp, stop=stop, move=MagicMock(side_effect=lambda *args: released.set() or True)
    )
    controller._gripper = gripper
    node, worker, logger = _worker_node(controller, queue=queue)
    try:
        queue.send(0.03)
        assert started.wait(timeout=1.0)
        queue.send(0.01)  # Keep closing; do not restart the active grasp.
        queue.send(0.0105)  # Sub-deadband noise must not look like a reversal.
        assert not stopped.is_set()
        queue.send(0.02)  # Open relative to the pending target, below the active target.
        assert released.wait(timeout=1.0)
        gripper.move.assert_called_once_with(0.02, controller._gripper_speed)
        logger.error.assert_not_called()
    finally:
        stopped.set()
        _shutdown(node, worker)
