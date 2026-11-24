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
from lerobot.robots.franka.franka import (
    END_POSE_KEYS,
    GRIPPER_JOINT_NAME,
    GRIPPER_KEY,
    JOINT_KEYS,
    JOINT_NAMES,
)


class _JointState:
    def __init__(self):
        self.header = SimpleNamespace(frame_id="", stamp=None)
        self.name = list(JOINT_NAMES)
        self.position = [float(index) for index in range(7)]


class _GripperState:
    def __init__(self):
        self.header = SimpleNamespace(frame_id="", stamp=None)
        self.name = [GRIPPER_JOINT_NAME]
        self.position = [0.04]


class _PoseStamped:
    def __init__(self):
        self.header = SimpleNamespace(frame_id="", stamp=None)
        self.pose = SimpleNamespace(
            position=SimpleNamespace(x=0.4, y=0.0, z=0.3),
            orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
        )


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
        for message_type, topic, callback, _ in self.node.subscriptions:
            if "gripper_state" in topic:
                callback(_GripperState())
            else:
                callback(message_type())
        self.context.stop.wait()

    def shutdown(self):
        self.shutdown_called = True
        self.context.stop.set()


@pytest.fixture
def ros_mocks():
    nodes = []

    def create_node(*_args, **_kwargs):
        node = _Node()
        nodes.append(node)
        return node

    fake_rclpy = SimpleNamespace(
        context=SimpleNamespace(Context=_Context),
        init=MagicMock(),
        create_node=create_node,
    )
    with (
        patch.object(franka_module, "rclpy", fake_rclpy),
        patch.object(franka_module, "JointState", _JointState),
        patch.object(franka_module, "PoseStamped", _PoseStamped),
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
    assert set(robot.observation_features) == {*JOINT_KEYS, *END_POSE_KEYS, GRIPPER_KEY}
    observation = robot.get_observation()
    assert [observation[key] for key in JOINT_KEYS] == [float(index) for index in range(7)]
    assert [observation[key] for key in END_POSE_KEYS] == [0.4, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0]
    assert observation[GRIPPER_KEY] == 0.04

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
    robot = _make_robot(tmp_path, max_relative_target=0.05)
    robot.connect()
    action = {key: float(index + 1) for index, key in enumerate(JOINT_KEYS)}
    action[GRIPPER_KEY] = 0.06

    sent = robot.send_action(action)

    assert {key: sent[key] for key in JOINT_KEYS} == {
        key: float(index) + 0.05 for index, key in enumerate(JOINT_KEYS)
    }
    assert sent[GRIPPER_KEY] == 0.06
    publisher = nodes[0].publishers[robot.config.joint_cmd_topic][2]
    message = publisher.messages[-1]
    assert message.name == list(JOINT_NAMES)
    assert message.position == [sent[key] for key in JOINT_KEYS]
    gripper_message = nodes[0].publishers[robot.config.gripper_cmd_topic][2].messages[-1]
    assert gripper_message.name == [GRIPPER_JOINT_NAME]
    assert gripper_message.position == [0.06]
    assert message.header.frame_id == robot.config.base_frame
    assert message.header.stamp == "stamp"
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
    assert set(robot.action_features) == {*END_POSE_KEYS, GRIPPER_KEY}

    action = {
        "end_pose.x": 0.7,
        "end_pose.y": 0.0,
        "end_pose.z": 0.3,
        "end_pose.qx": 0.0,
        "end_pose.qy": 0.0,
        "end_pose.qz": 1.0,
        "end_pose.qw": 0.0,
        GRIPPER_KEY: 0.09,
    }
    sent = robot.send_action(action)

    assert sent["end_pose.x"] == pytest.approx(0.5)
    assert FrankaRobot._quaternion_angle(
        (0.0, 0.0, 0.0, 1.0),
        tuple(sent[key] for key in END_POSE_KEYS[3:]),
    ) == pytest.approx(0.2)
    assert sent[GRIPPER_KEY] == 0.08
    publisher = nodes[0].publishers[robot.config.end_pose_cmd_topic][2]
    message = publisher.messages[-1]
    assert message.header.frame_id == robot.config.base_frame
    assert message.header.stamp == "stamp"
    robot.disconnect()


def test_action_validation(tmp_path, ros_mocks):
    robot = _make_robot(tmp_path, control_mode="cartesian")
    robot.connect()
    with pytest.raises(ValueError, match="exactly"):
        robot.send_action({"end_pose.x": 0.0})
    invalid = dict.fromkeys((*END_POSE_KEYS, GRIPPER_KEY), 0.0)
    invalid["end_pose.qw"] = 2.0
    with pytest.raises(ValueError, match="normalized"):
        robot.send_action(invalid)
    invalid["end_pose.qw"] = float("nan")
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


@pytest.mark.parametrize(
    ("timestamp", "state_name"),
    [
        ("_joint_state_time", "joint state"),
        ("_end_pose_time", "end-effector pose"),
        ("_gripper_state_time", "gripper state"),
    ],
)
def test_stale_observation_reports_all_state_ages(tmp_path, ros_mocks, monkeypatch, timestamp, state_name):
    monkeypatch.setenv("ROS_DOMAIN_ID", "23")
    robot = _make_robot(tmp_path)
    robot.connect()
    try:
        with robot._lock:
            robot._joint_state_time = robot._end_pose_time = robot._gripper_state_time = 99.9
            setattr(robot, timestamp, 96.5)
        with patch.object(franka_module.time, "monotonic", return_value=100.0):
            with pytest.raises(RuntimeError, match=f"Franka {state_name} is stale") as caught:
                robot.get_observation()

        message = str(caught.value)
        assert "timeout=1.00s" in message
        for topic, attr in [
            (robot.config.joint_state_topic, "_joint_state_time"),
            (robot.config.end_pose_topic, "_end_pose_time"),
            (robot.config.gripper_state_topic, "_gripper_state_time"),
        ]:
            age = "3.50s" if attr == timestamp else "0.10s"
            assert f"{topic}={age}" in message
        assert "ROS_DOMAIN_ID=23" in message
        assert "ROS executor=running" in message
        assert "control-PC franka_ros2_bridge log" in message
    finally:
        robot.disconnect()


@pytest.mark.parametrize("control_mode", ["joint", "cartesian"])
def test_stale_action_state_prevents_arm_and_gripper_publication(tmp_path, ros_mocks, control_mode):
    _, nodes = ros_mocks
    robot = _make_robot(tmp_path, control_mode=control_mode)
    robot.connect()
    try:
        observation = robot.get_observation()
        action = {key: observation[key] for key in robot.action_features}
        with robot._lock:
            robot._joint_state_time = robot._end_pose_time = 96.5
        with patch.object(franka_module.time, "monotonic", return_value=100.0):
            with pytest.raises(RuntimeError, match="is stale.*timeout=1.00s"):
                robot.send_action(action)
        assert all(not publisher.messages for _, _, publisher in nodes[0].publishers.values())
    finally:
        robot.disconnect()


def test_invalid_joint_messages_do_not_refresh_watchdog(tmp_path, ros_mocks):
    robot = _make_robot(tmp_path)
    robot.connect()
    try:
        with robot._lock:
            robot._joint_state_time = 96.5
        invalid = _JointState()
        invalid.position[0] = float("nan")
        with patch.object(franka_module.time, "monotonic", return_value=100.0):
            robot._joint_state_callback(invalid)
            with pytest.raises(RuntimeError, match="joint state is stale"):
                robot.get_observation()
        assert robot._joint_state_time == 96.5
    finally:
        robot.disconnect()


def test_executor_startup_failure_is_reported_and_connection_can_retry(tmp_path, ros_mocks):
    _, nodes = ros_mocks
    robot = _make_robot(tmp_path)
    error = RuntimeError("DDS receive failure")
    with patch.object(_Executor, "spin", side_effect=error):
        with pytest.raises(RuntimeError, match="ROS executor failed.*DDS receive failure") as caught:
            robot.connect()
    assert caught.value.__cause__ is error
    assert not robot.is_connected
    assert robot._executor_thread is None
    assert nodes[0].destroyed

    robot.connect()
    try:
        assert robot.get_observation()[GRIPPER_KEY] == 0.04
    finally:
        robot.disconnect()


def test_executor_unexpected_exit_during_connect_is_reported(tmp_path, ros_mocks):
    robot = _make_robot(tmp_path)
    with patch.object(_Executor, "spin", return_value=None):
        with pytest.raises(RuntimeError, match="ROS executor stopped"):
            robot.connect()
    assert not robot.is_connected
    assert robot._executor_thread is None


@pytest.mark.parametrize("control_mode", ["joint", "cartesian"])
def test_executor_failure_stops_observation_and_actions_with_fresh_cached_state(tmp_path, ros_mocks, control_mode):
    _, nodes = ros_mocks
    error = RuntimeError("DDS receive failure")
    original_spin = _Executor.spin

    def fail_after_initial_state(executor):
        original_spin(executor)
        raise error

    robot = _make_robot(tmp_path, control_mode=control_mode)
    with patch.object(_Executor, "spin", fail_after_initial_state):
        robot.connect()
        try:
            observation = robot.get_observation()
            action = {key: observation[key] for key in robot.action_features}
            robot._context.stop.set()
            robot._executor_thread.join(timeout=1.0)
            assert not robot._executor_thread.is_alive()
            with pytest.raises(RuntimeError, match="ROS executor failed.*DDS receive failure") as caught:
                robot.get_observation()
            assert caught.value.__cause__ is error
            with pytest.raises(RuntimeError, match="ROS executor failed"):
                robot.send_action(action)
            assert all(not publisher.messages for _, _, publisher in nodes[0].publishers.values())
        finally:
            robot.disconnect()


def test_executor_shutdown_exception_is_not_reported_as_failure(tmp_path, ros_mocks):
    original_spin = _Executor.spin

    def exit_on_shutdown(executor):
        original_spin(executor)
        raise RuntimeError("context shut down")

    robot = _make_robot(tmp_path)
    with patch.object(_Executor, "spin", exit_on_shutdown):
        robot.connect()
        robot.disconnect()
    assert robot._executor_error is None
