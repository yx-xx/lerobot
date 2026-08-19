#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from lerobot.robots.franka import FrankaConfig, FrankaRobot
from lerobot.robots.franka import franka as franka_module
from lerobot.robots.franka.franka import EE_KEYS, FRANKA_JOINT_NAMES, LEROBOT_JOINT_KEYS


class _JointState:
    def __init__(self):
        self.name = list(FRANKA_JOINT_NAMES)
        self.position = [float(index) for index in range(7)]


class _PoseStamped:
    def __init__(self):
        self.header = SimpleNamespace(frame_id="", stamp=None)
        self.pose = SimpleNamespace(
            position=SimpleNamespace(x=0.4, y=0.0, z=0.3),
            orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
        )


class _JointTrajectory:
    def __init__(self):
        self.joint_names = []
        self.points = []


class _JointTrajectoryPoint:
    def __init__(self):
        self.positions = []
        self.time_from_start = None


class _Context:
    def __init__(self):
        self.stop = threading.Event()
        self.shutdown_called = False

    def try_shutdown(self):
        self.shutdown_called = True
        self.stop.set()


class _Publisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


class _Node:
    def __init__(self):
        self.subscriptions = []
        self.publishers = {}
        self.destroyed = False

    def create_subscription(self, message_type, topic, callback, qos_depth):
        self.subscriptions.append((message_type, topic, callback, qos_depth))

    def create_publisher(self, message_type, topic, qos_depth):
        publisher = _Publisher()
        self.publishers[topic] = (message_type, qos_depth, publisher)
        return publisher

    def get_clock(self):
        return SimpleNamespace(now=lambda: SimpleNamespace(to_msg=lambda: "stamp"))

    def destroy_node(self):
        self.destroyed = True


class _Executor:
    def __init__(self, context):
        self.context = context
        self.node = None
        self.shutdown_called = False

    def add_node(self, node):
        self.node = node

    def spin(self):
        for message_type, _, callback, _ in self.node.subscriptions:
            callback(message_type())
        self.context.stop.wait()

    def shutdown(self):
        self.shutdown_called = True
        self.context.stop.set()


class _Duration:
    def __init__(self, seconds):
        self.seconds = seconds

    def to_msg(self):
        return self.seconds


@pytest.fixture
def ros_mocks():
    nodes = []

    def create_node(*_args, **_kwargs):
        node = _Node()
        nodes.append(node)
        return node

    fake_rclpy = SimpleNamespace(
        context=SimpleNamespace(Context=_Context),
        duration=SimpleNamespace(Duration=_Duration),
        init=MagicMock(),
        create_node=create_node,
    )
    with (
        patch.object(franka_module, "rclpy", fake_rclpy),
        patch.object(franka_module, "JointState", _JointState),
        patch.object(franka_module, "PoseStamped", _PoseStamped),
        patch.object(franka_module, "JointTrajectory", _JointTrajectory),
        patch.object(franka_module, "JointTrajectoryPoint", _JointTrajectoryPoint),
        patch.object(franka_module, "Duration", _Duration),
        patch.object(franka_module, "SingleThreadedExecutor", _Executor),
    ):
        yield fake_rclpy, nodes


def _make_robot(tmp_path, control_mode="joint", **kwargs):
    config = FrankaConfig(
        calibration_dir=tmp_path,
        control_mode=control_mode,
        connect_timeout_s=0.5,
        state_timeout_s=1.0,
        **kwargs,
    )
    return FrankaRobot(config)


def test_config_validation():
    with pytest.raises(ValueError, match="control_mode"):
        FrankaConfig(control_mode="invalid")
    with pytest.raises(ValueError, match="qos_depth"):
        FrankaConfig(qos_depth=0)
    with pytest.raises(ValueError, match="j1 through j7"):
        FrankaConfig(max_relative_target={"j1": 0.1})
    with pytest.raises(ValueError, match="normalized|positive"):
        FrankaConfig(max_relative_rotation=0.0)


def test_connect_observe_and_disconnect(tmp_path, ros_mocks):
    fake_rclpy, nodes = ros_mocks
    robot = _make_robot(tmp_path)
    robot.connect()

    assert robot.is_connected
    assert set(robot.observation_features) == {*LEROBOT_JOINT_KEYS, *EE_KEYS}
    observation = robot.get_observation()
    assert [observation[key] for key in LEROBOT_JOINT_KEYS] == [float(index) for index in range(7)]
    assert [observation[key] for key in EE_KEYS] == [0.4, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0]

    context = robot._context
    executor = robot._executor
    robot.disconnect()
    assert not robot.is_connected
    assert executor.shutdown_called
    assert context.shutdown_called
    assert nodes[0].destroyed
    fake_rclpy.init.assert_called_once_with(context=context)


def test_joint_action_is_clamped_and_published(tmp_path, ros_mocks):
    _, nodes = ros_mocks
    robot = _make_robot(tmp_path, max_relative_target=0.05, command_duration_s=0.2)
    robot.connect()
    action = {key: float(index + 1) for index, key in enumerate(LEROBOT_JOINT_KEYS)}

    sent = robot.send_action(action)

    assert sent == {key: float(index) + 0.05 for index, key in enumerate(LEROBOT_JOINT_KEYS)}
    publisher = nodes[0].publishers[robot.config.joint_command_topic][2]
    message = publisher.messages[-1]
    assert message.joint_names == list(FRANKA_JOINT_NAMES)
    assert message.points[0].positions == list(sent.values())
    assert message.points[0].time_from_start == 0.2
    robot.disconnect()


def test_cartesian_action_is_limited_and_published(tmp_path, ros_mocks):
    _, nodes = ros_mocks
    robot = _make_robot(
        tmp_path,
        control_mode="cartesian",
        max_relative_translation=0.1,
        max_relative_rotation=0.2,
    )
    robot.connect()
    assert set(robot.action_features) == set(EE_KEYS)

    action = {
        "ee.x": 0.7,
        "ee.y": 0.0,
        "ee.z": 0.3,
        "ee.qx": 0.0,
        "ee.qy": 0.0,
        "ee.qz": 1.0,
        "ee.qw": 0.0,
    }
    sent = robot.send_action(action)

    assert sent["ee.x"] == pytest.approx(0.5)
    assert FrankaRobot._quaternion_angle(
        (0.0, 0.0, 0.0, 1.0),
        tuple(sent[key] for key in EE_KEYS[3:]),
    ) == pytest.approx(0.2)
    publisher = nodes[0].publishers[robot.config.ee_command_topic][2]
    message = publisher.messages[-1]
    assert message.header.frame_id == robot.config.base_frame
    assert message.header.stamp == "stamp"
    robot.disconnect()


def test_action_validation(tmp_path, ros_mocks):
    robot = _make_robot(tmp_path, control_mode="cartesian")
    robot.connect()
    with pytest.raises(ValueError, match="exactly"):
        robot.send_action({"ee.x": 0.0})
    invalid = dict.fromkeys(EE_KEYS, 0.0)
    invalid["ee.qw"] = 2.0
    with pytest.raises(ValueError, match="normalized"):
        robot.send_action(invalid)
    invalid["ee.qw"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        robot.send_action(invalid)
    robot.disconnect()


def test_connect_without_ros_reports_runtime_error(tmp_path):
    robot = _make_robot(tmp_path)
    with (
        patch.object(franka_module, "rclpy", None),
        patch.object(franka_module, "_ROS_IMPORT_ERROR", ImportError("missing ROS")),
        pytest.raises(RuntimeError, match="ROS 2 Python packages are unavailable"),
    ):
        robot.connect()
