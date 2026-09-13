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
    map_calibrated_range,
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

PIPER_BOX = ((0.0, 400.0), (-100.0, 100.0), (0.0, 200.0))
FRANKA_BOX = ((0.20, 0.60), (-0.10, 0.10), (0.10, 0.30))
PENDANT = (0.0, 80.0)
GRIPPER = (0.00, 0.08)


def _processor(**kwargs):
    defaults = {
        "piper_xyz_mm": PIPER_BOX,
        "franka_xyz_m": FRANKA_BOX,
        "pendant_mm": PENDANT,
        "gripper_m": GRIPPER,
    }
    defaults.update(kwargs)
    return MapPiperXEndposeToFrankaAction(**defaults)


def _piper_action(x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0, pendant=0.0):
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


def test_map_calibrated_range_endpoints_and_midpoint():
    assert map_calibrated_range(0.0, (0.0, 400.0), (0.20, 0.60)) == 0.20
    assert map_calibrated_range(400.0, (0.0, 400.0), (0.20, 0.60)) == 0.60
    assert map_calibrated_range(200.0, (0.0, 400.0), (0.20, 0.60)) == 0.40


def test_cuboid_maps_piper_box_onto_franka_box():
    processor = _processor()
    mapped = _apply(processor, _piper_action(x=200.0, y=0.0, z=100.0, pendant=40.0))

    assert set(mapped) == EXPECTED_ACTION_KEYS
    assert mapped["end_pose.x"] == 0.40
    assert mapped["end_pose.y"] == 0.00
    assert mapped["end_pose.z"] == 0.20
    assert mapped["gripper.pos"] == 0.04


def test_values_outside_source_box_are_clipped():
    processor = _processor()
    mapped = _apply(processor, _piper_action(x=-50.0, y=500.0, z=1000.0, pendant=999.0))
    assert mapped["end_pose.x"] == 0.20
    assert mapped["end_pose.y"] == 0.10
    assert mapped["end_pose.z"] == 0.30
    assert mapped["gripper.pos"] == 0.08


def test_reversed_destination_inverts_axis():
    processor = _processor(franka_xyz_m=((0.60, 0.20), (-0.10, 0.10), (0.10, 0.30)))
    mapped = _apply(processor, _piper_action(x=0.0, y=0.0, z=0.0))
    assert mapped["end_pose.x"] == 0.60
    mapped = _apply(processor, _piper_action(x=400.0, y=0.0, z=0.0))
    assert mapped["end_pose.x"] == 0.20


def test_absolute_yaw_maps_to_franka_quaternion():
    processor = _processor()
    mapped = _apply(processor, _piper_action(x=0.0, yaw=90.0))
    expected = Rotation.from_rotvec(np.array([0.0, 0.0, math.pi / 2.0])).as_quat()
    _assert_quat_close(
        [mapped["end_pose.qx"], mapped["end_pose.qy"], mapped["end_pose.qz"], mapped["end_pose.qw"]],
        expected,
    )


def test_mapping_does_not_depend_on_current_franka_pose():
    processor = _processor()
    first = _apply(processor, _piper_action(x=200.0), _franka_obs(x=0.1, y=0.2, z=0.3))
    second = _apply(processor, _piper_action(x=200.0), _franka_obs(x=0.8, y=-0.4, z=0.6))
    assert first == second
    assert first["end_pose.x"] == 0.40


def test_pipeline_factory_matches_step_output():
    pipeline = make_piper_x_to_franka_teleop_processor(
        piper_xyz_mm=PIPER_BOX,
        franka_xyz_m=FRANKA_BOX,
        pendant_mm=PENDANT,
        gripper_m=GRIPPER,
    )
    mapped = pipeline((_piper_action(x=200.0, pendant=40.0), _franka_obs()))
    assert mapped["end_pose.x"] == 0.40
    assert mapped["gripper.pos"] == 0.04
    assert set(mapped) == EXPECTED_ACTION_KEYS
