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
        call.stop(), call.grasp(0.005, 0.04, 15.0, 0.005, 0.08),
        call.stop(), call.move(0.02, 0.04),
        call.stop(), call.move(0.08, 0.04),
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


@pytest.mark.parametrize("force", [0.0, -1.0, 71.0, float("nan"), float("inf")])
def test_invalid_grasp_force_is_rejected(force):
    with pytest.raises(ValueError, match="gripper_force"):
        StreamController("172.16.0.2", gripper_force=force)


def _worker_node(controller, deadband=0.001):
    logger = MagicMock()
    node = SimpleNamespace(
        _controller=controller,
        _commands=CommandQueue(2.0),
        _stop_event=threading.Event(),
        _last_gripper_command=None,
        gripper_command_deadband=deadband,
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
