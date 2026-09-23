import math
import time
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from franka_ros2_bridge.core.command_queue import CommandQueue
from franka_ros2_bridge.core.math_utils import matrix_to_pose, pose_to_matrix
from franka_ros2_bridge.core.safety import (
    build_end_pose_command,
    build_gripper_command,
    build_joint_command,
)
from franka_ros2_bridge.core.types import (
    DEFAULT_JOINT_LOWER_LIMITS,
    DEFAULT_JOINT_UPPER_LIMITS,
    GRIPPER_JOINT_NAME,
    JOINT_NAMES,
    EndPose,
    EndPoseCommand,
    JointCommand,
    RobotState,
)
from franka_ros2_bridge.control.frankx_controller import joints_from_frankx_robot
from franka_ros2_bridge.control.stream_controller import StreamController
from franka_ros2_bridge.ros.converters import (
    end_pose_cmd_from_msg,
    gripper_cmd_from_msg,
    joint_cmd_from_msg,
    to_end_pose_msg,
    to_gripper_state_msg,
    to_joint_state_msg,
)


def test_pose_matrix_round_trip() -> None:
    position = (0.4, -0.2, 0.5)
    quaternion = (0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5))
    matrix = pose_to_matrix(position, quaternion)

    assert matrix[12:15] == pytest.approx(position)
    restored_position, restored_quaternion = matrix_to_pose(matrix)
    assert restored_position == pytest.approx(position)
    assert abs(sum(a * b for a, b in zip(quaternion, restored_quaternion))) == pytest.approx(
        1.0
    )


def test_pose_to_matrix_rejects_unnormalized_quaternion() -> None:
    with pytest.raises(ValueError, match="normalized"):
        pose_to_matrix((0, 0, 0), (0, 0, 0, 2))


def test_validate_joint_command() -> None:
    from franka_ros2_bridge.core import validate_joint_command

    positions = validate_joint_command(JOINT_NAMES, range(7))
    assert positions == tuple(float(value) for value in range(7))


@pytest.mark.parametrize(
    ("names", "positions"),
    [
        (JOINT_NAMES[:-1], range(7)),
        (tuple(reversed(JOINT_NAMES)), range(7)),
        (JOINT_NAMES, range(6)),
        (JOINT_NAMES, [0, 0, 0, 0, 0, 0, float("nan")]),
    ],
)
def test_validate_joint_command_rejects_invalid_input(names, positions) -> None:
    from franka_ros2_bridge.core import validate_joint_command

    with pytest.raises(ValueError):
        validate_joint_command(names, positions)


def test_pose_validation_rejects_bad_values() -> None:
    with pytest.raises(ValueError):
        pose_to_matrix((0, 0, 0), (0, 0, 0, 0))
    with pytest.raises(ValueError):
        matrix_to_pose([0.0] * 15)


def test_build_joint_command_respects_limits() -> None:
    mid = tuple(
        0.5 * (lo + hi)
        for lo, hi in zip(DEFAULT_JOINT_LOWER_LIMITS, DEFAULT_JOINT_UPPER_LIMITS, strict=True)
    )
    command = build_joint_command(
        JOINT_NAMES,
        mid,
        lower=DEFAULT_JOINT_LOWER_LIMITS,
        upper=DEFAULT_JOINT_UPPER_LIMITS,
        received_at=0.0,
    )
    assert command.positions == pytest.approx(mid)
    with pytest.raises(ValueError, match="joint limits"):
        build_joint_command(
            JOINT_NAMES,
            [10.0] * 7,
            lower=DEFAULT_JOINT_LOWER_LIMITS,
            upper=DEFAULT_JOINT_UPPER_LIMITS,
            received_at=0.0,
        )


def test_build_end_pose_command_checks_frame_and_workspace() -> None:
    command = build_end_pose_command(
        (0.4, 0.0, 0.3),
        (0.0, 0.0, 0.0, 1.0),
        frame_id="panda_link0",
        base_frame="panda_link0",
        workspace_min=(0.2, -0.6, 0.02),
        workspace_max=(0.8, 0.6, 0.9),
        received_at=1.0,
    )
    assert command.position == pytest.approx((0.4, 0.0, 0.3))
    assert command.quaternion == pytest.approx((0.0, 0.0, 0.0, 1.0))
    with pytest.raises(ValueError, match="frame_id"):
        build_end_pose_command(
            (0.4, 0.0, 0.3),
            (0.0, 0.0, 0.0, 1.0),
            frame_id="other",
            base_frame="panda_link0",
            workspace_min=(0.2, -0.6, 0.02),
            workspace_max=(0.8, 0.6, 0.9),
            received_at=1.0,
        )


def test_command_queue_take_zero_timeout_is_nonblocking() -> None:
    queue = CommandQueue(timeout_sec=1.0)
    assert queue.take(wait_timeout_sec=0.0) is None
    queue.push_end_pose(
        EndPoseCommand(
            position=(0.4, 0.0, 0.3),
            quaternion=(0.0, 0.0, 0.0, 1.0),
            received_at=time.monotonic(),
        )
    )
    command = queue.take(wait_timeout_sec=0.0)
    assert command is not None
    assert command.mode == "cartesian"


def test_command_queue_keeps_latest_and_detects_stale() -> None:
    queue = CommandQueue(timeout_sec=0.05)
    queue.push_joint(JointCommand(positions=(0.0,) * 7, received_at=time.monotonic() - 1.0))
    queue.push_joint(JointCommand(positions=(0.1,) * 7, received_at=time.monotonic() - 1.0))
    command = queue.take(wait_timeout_sec=0.01)
    assert command is not None
    assert command.mode == "joint"
    assert command.joint is not None
    assert command.joint.positions[0] == pytest.approx(0.1)
    assert queue.is_stale(command)


def test_joint_cmd_converters_round_trip() -> None:
    state = RobotState(
        joints=tuple(float(index) for index in range(7)),
        end_pose=EndPose(position=(0.4, 0.0, 0.3), quaternion=(0.0, 0.0, 0.0, 1.0)),
        gripper_width=0.04,
    )
    joint_msg = SimpleNamespace(header=SimpleNamespace(stamp=None, frame_id=""), name=[], position=[])
    to_joint_state_msg(joint_msg, state, stamp="stamp", frame_id="panda_link0")
    names, positions = joint_cmd_from_msg(joint_msg)
    assert names == list(JOINT_NAMES)
    assert positions == list(state.joints)

    pose_msg = SimpleNamespace(
        header=SimpleNamespace(stamp=None, frame_id=""),
        pose=SimpleNamespace(
            position=SimpleNamespace(x=0.0, y=0.0, z=0.0),
            orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=0.0),
        ),
    )
    to_end_pose_msg(pose_msg, state, stamp="stamp", frame_id="panda_link0")
    frame_id, position, quaternion = end_pose_cmd_from_msg(pose_msg)
    assert frame_id == "panda_link0"
    assert position == pytest.approx(state.end_pose.position)
    assert quaternion == pytest.approx(state.end_pose.quaternion)

    gripper_msg = SimpleNamespace(header=SimpleNamespace(stamp=None, frame_id=""), name=[], position=[])
    to_gripper_state_msg(gripper_msg, state, stamp="stamp", frame_id="panda_link0")
    assert gripper_cmd_from_msg(gripper_msg) == pytest.approx(0.04)
    assert gripper_msg.name == [GRIPPER_JOINT_NAME]


def test_stream_controller_rejects_bad_limits() -> None:
    with pytest.raises(ValueError):
        StreamController("172.16.0.2", max_linear_velocity=0.0)
    with pytest.raises(ValueError):
        StreamController("172.16.0.2", max_linear_acceleration=0.0)
    with pytest.raises(ValueError):
        StreamController("172.16.0.2", max_linear_jerk=0.0)
    with pytest.raises(ValueError):
        StreamController("172.16.0.2", max_angular_velocity=-1.0)
    with pytest.raises(ValueError):
        StreamController("172.16.0.2", max_angular_acceleration=0.0)
    with pytest.raises(ValueError):
        StreamController("172.16.0.2", max_angular_jerk=0.0)
    with pytest.raises(ValueError):
        StreamController("172.16.0.2", tracking_frequency_hz=0.0)
    with pytest.raises(ValueError):
        StreamController("172.16.0.2", linear_deadband=-0.001)
    with pytest.raises(ValueError):
        StreamController("172.16.0.2", angular_deadband=-0.001)
    with pytest.raises(TypeError):
        StreamController("172.16.0.2", lock_elbow=1)
    with pytest.raises(ValueError):
        StreamController("172.16.0.2", control_cpu=-2)
    with pytest.raises(ValueError):
        StreamController("172.16.0.2", initial_sync_scale=0.0)
    with pytest.raises(ValueError):
        StreamController("172.16.0.2", initial_sync_scale=1.1)
    with pytest.raises(ValueError):
        StreamController("172.16.0.2", gripper_poll_rate_hz=0.0)


@pytest.mark.parametrize("duration", [0.0, -0.1, 5.1, math.nan, math.inf])
def test_startup_ramp_rejects_invalid_duration(duration) -> None:
    with pytest.raises(ValueError, match="startup_ramp_sec"):
        StreamController("172.16.0.2", startup_ramp_sec=duration)


def test_stream_controller_forwards_absolute_target_without_rebasing() -> None:
    controller = StreamController("172.16.0.2")
    native_stream = SimpleNamespace(check_error=MagicMock(), running=lambda: True, set_target=MagicMock())
    controller._stream = native_stream
    for position in ((0.43, -0.1, 0.4), (0.44, -0.1, 0.4)):
        command = EndPoseCommand(position=position, quaternion=(0.0, 0.0, 0.0, 1.0), received_at=0.0)
        controller.set_end_pose_target(command)
        native_stream.set_target.assert_called_with(*position, 0.0, 0.0, 0.0, 1.0)


def test_stream_controller_rejects_a_stopped_native_stream() -> None:
    controller = StreamController("172.16.0.2")
    native_stream = SimpleNamespace(check_error=MagicMock(), running=lambda: False)
    controller._stream = native_stream

    with pytest.raises(RuntimeError, match="not running"):
        controller._require_stream()

    native_stream.check_error.assert_called_once_with()


def test_stream_controller_propagates_native_control_error() -> None:
    controller = StreamController("172.16.0.2")
    controller._stream = SimpleNamespace(
        check_error=MagicMock(side_effect=RuntimeError("control failed")),
        running=lambda: False,
    )

    with pytest.raises(RuntimeError, match="control failed"):
        controller._require_stream()


def test_stream_controller_returns_native_diagnostics() -> None:
    controller = StreamController("172.16.0.2")
    diagnostics = {"control_command_success_rate": 1.0, "minimum_singular_value": 0.2}
    native_stream = SimpleNamespace(
        check_error=MagicMock(), running=lambda: True, get_diagnostics=lambda: diagnostics
    )
    controller._stream = native_stream

    result = controller.get_diagnostics()
    assert result["control_command_success_rate"] == diagnostics["control_command_success_rate"]
    assert result["minimum_singular_value"] == diagnostics["minimum_singular_value"]
    assert result["gripper_age_ms"] == -1.0
    assert result["gripper_poll_errors"] == 0
    native_stream.check_error.assert_called_once_with()


def test_blocked_gripper_poll_does_not_block_arm_state_or_targets() -> None:
    controller = StreamController("172.16.0.2")
    entered, release, stop = threading.Event(), threading.Event(), threading.Event()
    def delayed_width():
        entered.set()
        release.wait(timeout=2.0)
        return .03
    gripper = SimpleNamespace(width=MagicMock(side_effect=delayed_width))
    controller._gripper = gripper
    controller._gripper_poll_period = .001
    controller._last_gripper_width = .04
    controller._last_gripper_poll = time.monotonic() - 10.0
    controller._stream = SimpleNamespace(
        check_error=MagicMock(), running=lambda: True, set_target=MagicMock(),
        get_state=lambda: ((0.0,) * 7, (.4, 0., .3), (0., 0., 0., 1.)),
    )
    worker = threading.Thread(target=controller._poll_gripper, args=(gripper, stop), daemon=True)
    worker.start()
    try:
        assert entered.wait(timeout=1.0)
        assert controller.read_state().gripper_width == pytest.approx(.04)
        assert not controller.gripper_state_is_fresh()
        command = EndPoseCommand(position=(.45, 0., .3), quaternion=(0., 0., 0., 1.), received_at=0.)
        controller.set_end_pose_target(command)
        controller._stream.set_target.assert_called_once_with(.45, 0., .3, 0., 0., 0., 1.)
        gripper.width.assert_called_once_with()
    finally:
        stop.set()
        release.set()
        worker.join(timeout=1.0)


def test_stale_gripper_still_publishes_fresh_arm_state() -> None:
    from franka_ros2_bridge.bridge_node import FrankaBridgeNode
    state = RobotState(
        joints=(0.,) * 7, end_pose=EndPose((.4, 0., .3), (0., 0., 0., 1.)), gripper_width=.04,
    )
    node = SimpleNamespace(
        _controller=SimpleNamespace(read_state=lambda: state, gripper_state_is_fresh=lambda: False),
        _streaming=True, _state_error_logged=False, base_frame="panda_link0",
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(to_msg=lambda: None)),
        _joint_state_pub=MagicMock(), _end_pose_pub=MagicMock(), _gripper_state_pub=MagicMock(),
    )
    # Converters use actual ROS messages when ROS is installed.
    import franka_ros2_bridge.bridge_node as module
    if module.rclpy is None:
        pytest.skip("ROS message constructors are unavailable")
    from builtin_interfaces.msg import Time
    node.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(to_msg=lambda: Time()))
    FrankaBridgeNode._publish_state(node)
    node._joint_state_pub.publish.assert_called_once()
    node._end_pose_pub.publish.assert_called_once()
    node._gripper_state_pub.publish.assert_not_called()


def test_joints_from_frankx_prefer_current_joint_positions() -> None:
    robot = SimpleNamespace(
        current_joint_positions=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        read_once=lambda: SimpleNamespace(q=[9.0] * 7),
    )
    assert joints_from_frankx_robot(robot) == (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)


def test_joints_from_frankx_falls_back_to_read_once_q() -> None:
    robot = SimpleNamespace(read_once=lambda: SimpleNamespace(q=list(range(7))))
    assert joints_from_frankx_robot(robot) == tuple(float(value) for value in range(7))


def test_build_gripper_command_clips_to_limits() -> None:
    command = build_gripper_command(0.20, min_width=0.0, max_width=0.08, received_at=1.0)
    assert command.width == pytest.approx(0.08)
    with pytest.raises(ValueError, match="finite"):
        build_gripper_command(float("nan"), min_width=0.0, max_width=0.08, received_at=1.0)
