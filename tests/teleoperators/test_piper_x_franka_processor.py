#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import math

import numpy as np

from lerobot.processor import TransitionKey
from lerobot.processor.converters import create_transition
from lerobot.teleoperators.piper_x.piper_x_processor import (
    MapPiperXEndposeToFrankaAction,
    make_piper_x_to_franka_teleop_processor,
)
from lerobot.utils.rotation import Rotation

EXPECTED_ACTION_KEYS = {
    "end_pose.x",
    "end_pose.y",
    "end_pose.z",
    "end_pose.qx",
    "end_pose.qy",
    "end_pose.qz",
    "end_pose.qw",
    "gripper.pos",
}


def _piper_action(x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0, pendant=12.0):
    return {
        "endpose.x": x,
        "endpose.y": y,
        "endpose.z": z,
        "endpose.roll": roll,
        "endpose.pitch": pitch,
        "endpose.yaw": yaw,
        "teaching_pendant.pos": pendant,
    }


def _franka_obs(x=0.9, y=0.9, z=0.9):
    return {
        "j1.pos": 0.0,
        "end_pose.x": x,
        "end_pose.y": y,
        "end_pose.z": z,
        "end_pose.qx": 0.0,
        "end_pose.qy": 0.0,
        "end_pose.qz": 0.0,
        "end_pose.qw": 1.0,
    }


def _apply(processor, action, observation=None):
    if observation is None:
        observation = _franka_obs()
    transition = create_transition(action=action, observation=observation)
    return processor(transition)[TransitionKey.ACTION]


def _assert_quat_close(actual, expected, atol=1e-6):
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    if np.dot(actual, expected) < 0:
        actual = -actual
    np.testing.assert_allclose(actual, expected, atol=atol)


def test_absolute_pose_maps_pendant_to_gripper():
    processor = MapPiperXEndposeToFrankaAction()
    mapped = _apply(processor, _piper_action(x=400.0, y=-50.0, z=300.0, pendant=33.0), _franka_obs())

    assert set(mapped) == EXPECTED_ACTION_KEYS
    assert "teaching_pendant.pos" not in mapped
    assert mapped["end_pose.x"] == 0.4
    assert mapped["end_pose.y"] == -0.05
    assert mapped["end_pose.z"] == 0.3
    assert mapped["gripper.pos"] == 0.033
    _assert_quat_close(
        [mapped["end_pose.qx"], mapped["end_pose.qy"], mapped["end_pose.qz"], mapped["end_pose.qw"]],
        [0.0, 0.0, 0.0, 1.0],
    )


def test_scale_and_offset_apply_to_absolute_position():
    processor = MapPiperXEndposeToFrankaAction(
        position_scale=0.5,
        position_offset=(0.10, -0.02, 0.05),
    )
    mapped = _apply(processor, _piper_action(x=200.0, y=-100.0, z=40.0))

    assert mapped["end_pose.x"] == 0.10 + 0.200 * 0.5
    assert mapped["end_pose.y"] == -0.02 + (-0.100) * 0.5
    assert mapped["end_pose.z"] == 0.05 + 0.040 * 0.5


def test_absolute_yaw_maps_to_franka_quaternion():
    processor = MapPiperXEndposeToFrankaAction()
    mapped = _apply(processor, _piper_action(x=400.0, yaw=90.0))

    expected = Rotation.from_rotvec(np.array([0.0, 0.0, math.pi / 2.0])).as_quat()
    _assert_quat_close(
        [mapped["end_pose.qx"], mapped["end_pose.qy"], mapped["end_pose.qz"], mapped["end_pose.qw"]],
        expected,
    )
    assert mapped["end_pose.x"] == 0.4


def test_position_axes_remap_absolute_translation():
    processor = MapPiperXEndposeToFrankaAction(position_axes=("-y", "x", "z"))
    mapped = _apply(processor, _piper_action(x=100.0, y=50.0, z=80.0))

    assert mapped["end_pose.x"] == -0.050
    assert mapped["end_pose.y"] == 0.100
    assert mapped["end_pose.z"] == 0.080


def test_mapping_does_not_depend_on_current_franka_pose():
    processor = MapPiperXEndposeToFrankaAction()
    first = _apply(processor, _piper_action(x=350.0), _franka_obs(x=0.1, y=0.2, z=0.3))
    second = _apply(processor, _piper_action(x=350.0), _franka_obs(x=0.8, y=-0.4, z=0.6))
    assert first == second
    assert first["end_pose.x"] == 0.35


def test_gripper_scale_offset_and_clip():
    processor = MapPiperXEndposeToFrankaAction(
        gripper_scale=0.5,
        gripper_offset=0.01,
        gripper_min=0.0,
        gripper_max=0.04,
    )
    mapped = _apply(processor, _piper_action(pendant=80.0))
    assert mapped["gripper.pos"] == 0.04
    mapped = _apply(processor, _piper_action(pendant=20.0))
    assert mapped["gripper.pos"] == 0.02


def test_pipeline_factory_matches_step_output():
    pipeline = make_piper_x_to_franka_teleop_processor()
    mapped = pipeline((_piper_action(x=50.0), _franka_obs()))
    assert mapped["end_pose.x"] == 0.05
    assert set(mapped) == EXPECTED_ACTION_KEYS
