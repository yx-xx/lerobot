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
import pytest

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
CALIBRATED_DOWNWARD_QUAT_XYZW = (
    0.9992919885928564,
    -0.03241738912274881,
    0.00980888008203098,
    0.016383537106962606,
)


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


def test_identity_refs_keep_piper_yaw_as_franka_quaternion():
    processor = _processor()
    mapped = _apply(processor, _piper_action(x=0.0, yaw=90.0))
    expected = Rotation.from_rotvec(np.array([0.0, 0.0, math.pi / 2.0])).as_quat()
    _assert_quat_close(
        [mapped["end_pose.qx"], mapped["end_pose.qy"], mapped["end_pose.qz"], mapped["end_pose.qw"]],
        expected,
    )


def test_corresponding_pose_maps_piper_ref_to_franka_ref():
    franka_ref = Rotation.from_rotvec(np.array([0.0, math.pi / 2.0, 0.0])).as_quat()
    processor = _processor(
        piper_ref_rpy_deg=(10.0, -20.0, 30.0),
        franka_ref_quat_xyzw=tuple(franka_ref),
    )
    mapped = _apply(processor, _piper_action(roll=10.0, pitch=-20.0, yaw=30.0))
    _assert_quat_close(
        [mapped["end_pose.qx"], mapped["end_pose.qy"], mapped["end_pose.qz"], mapped["end_pose.qw"]],
        franka_ref,
    )


def test_piper_base_frame_yaw_is_franka_base_frame_yaw():
    franka_ref = Rotation.from_rotvec(np.array([math.pi / 2.0, 0.0, 0.0]))
    processor = _processor(franka_ref_quat_xyzw=tuple(franka_ref.as_quat()))
    mapped = _apply(processor, _piper_action(yaw=90.0))
    expected = (Rotation.from_rotvec(np.array([0.0, 0.0, math.pi / 2.0])) * franka_ref).as_quat()
    _assert_quat_close(
        [mapped["end_pose.qx"], mapped["end_pose.qy"], mapped["end_pose.qz"], mapped["end_pose.qw"]],
        expected,
    )


@pytest.mark.parametrize(
    ("roll", "pitch", "tilt_axis", "direction"),
    [(170.0, 0.0, 0, -1.0), (190.0, 0.0, 0, 1.0), (180.0, 10.0, 1, 1.0), (180.0, -10.0, 1, -1.0)],
    ids=["negative_x", "positive_x", "positive_y", "negative_y"],
)
def test_downward_tools_keep_same_tilt_direction_despite_reference_yaw(roll, pitch, tilt_axis, direction):
    # Both tool Z axes point down, but their tool X/Y axes differ by 90 degrees.
    processor = _processor(
        piper_ref_rpy_deg=(180.0, 0.0, -90.0),
        franka_ref_quat_xyzw=(1.0, 0.0, 0.0, 0.0),
    )
    mapped = _apply(processor, _piper_action(roll=roll, pitch=pitch, yaw=-90.0))
    rotation = Rotation.from_quat(
        np.array([mapped["end_pose.qx"], mapped["end_pose.qy"], mapped["end_pose.qz"], mapped["end_pose.qw"]])
    )
    expected_z = np.array([0.0, 0.0, -math.cos(math.radians(10.0))])
    expected_z[tilt_axis] = direction * math.sin(math.radians(10.0))
    np.testing.assert_allclose(rotation.apply(np.array([0.0, 0.0, 1.0])), expected_z, atol=1e-6)


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


@pytest.mark.parametrize("rpy", [(0.0, 0.0, 0.0), (180.0, 90.0, -180.0), (-90.0, -90.0, 90.0)])
def test_fixed_calibrated_downward_orientation_ignores_master_rotation(rpy):
    processor = _processor(fixed_franka_quat_xyzw=CALIBRATED_DOWNWARD_QUAT_XYZW)
    mapped = _apply(processor, _piper_action(x=200.0, pendant=40.0, roll=rpy[0], pitch=rpy[1], yaw=rpy[2]))
    quat = [mapped[f"end_pose.{key}"] for key in ("qx", "qy", "qz", "qw")]
    _assert_quat_close(quat, CALIBRATED_DOWNWARD_QUAT_XYZW, atol=1e-12)
    assert mapped["end_pose.x"] == 0.40
    assert mapped["gripper.pos"] == 0.04


def test_fixed_orientation_does_not_require_master_orientation_measurements():
    processor = _processor(fixed_franka_quat_xyzw=(2.0, 0.0, 0.0, 0.0))
    mapped = _apply(processor, {"endpose.x": 0.0, "endpose.y": 0.0, "endpose.z": 0.0})
    assert mapped["end_pose.qx"] == 1.0
    assert mapped["end_pose.qw"] == 0.0


def test_position_deadband_holds_boundary_noise_but_accumulates_real_motion():
    processor = _processor(position_deadband_m=0.0015)
    initial = _apply(processor, _piper_action(x=400.0, y=0.0, z=100.0))
    for x in [399.8, 399.0, 400.2, 399.5]:
        assert _apply(processor, _piper_action(x=x, y=0.0, z=100.0)) == initial
    # Changes accumulate relative to the held target, not the previous sample.
    moved = _apply(processor, _piper_action(x=398.0, y=0.0, z=100.0))
    assert moved["end_pose.x"] == pytest.approx(0.598)
    assert moved["end_pose.y"] == initial["end_pose.y"]
    assert moved["end_pose.z"] == initial["end_pose.z"]


def test_position_deadband_on_one_axis_is_not_released_by_another_axis():
    processor = _processor(position_deadband_m=0.0015)
    _apply(processor, _piper_action(x=400.0, y=0.0, z=100.0))
    moved = _apply(processor, _piper_action(x=399.0, y=10.0, z=100.0))
    assert moved["end_pose.x"] == 0.60
    assert moved["end_pose.y"] == pytest.approx(0.01)


def test_pipeline_factory_supports_fixed_orientation_and_position_deadband():
    pipeline = make_piper_x_to_franka_teleop_processor(
        piper_xyz_mm=PIPER_BOX, franka_xyz_m=FRANKA_BOX, pendant_mm=PENDANT, gripper_m=GRIPPER,
        fixed_franka_quat_xyzw=CALIBRATED_DOWNWARD_QUAT_XYZW, position_deadband_m=0.0015,
    )
    mapped = pipeline((_piper_action(x=200.0, pitch=90.0), _franka_obs()))
    _assert_quat_close(
        [mapped[f"end_pose.{key}"] for key in ("qx", "qy", "qz", "qw")],
        CALIBRATED_DOWNWARD_QUAT_XYZW,
        atol=1e-12,
    )


def test_piper_franka_setup_opens_past_object_on_first_release_input():
    from examples.piper_x_to_franka.common import make_mapped_action_processor

    pipeline = make_mapped_action_processor()
    observation = {"gripper.pos": 0.04}  # A 40 mm object blocks a 4.5 mm closing target.
    closed = pipeline((_piper_action(pendant=51.0), observation))
    released = pipeline((_piper_action(pendant=60.0), observation))
    assert closed["gripper.pos"] == pytest.approx(0.0045)
    assert released["gripper.pos"] == pytest.approx(0.045)


def test_release_clearance_is_latched_without_creeping_open_and_resets_on_close():
    processor = _processor(gripper_release_clearance_m=0.005)
    _apply(processor, _piper_action(pendant=5.0), {"gripper.pos": 0.04})
    assert _apply(processor, _piper_action(pendant=7.0), {"gripper.pos": 0.04})["gripper.pos"] == pytest.approx(0.045)
    # Moving feedback must not keep adding another 5 mm, nor can input noise re-grasp.
    for pendant, actual in ((7.4, 0.045), (7.0, 0.045), (8.5, 0.045), (20.0, 0.045)):
        assert _apply(processor, _piper_action(pendant=pendant), {"gripper.pos": actual})["gripper.pos"] == pytest.approx(0.045)
    # The original absolute mapping resumes above the floor and on deliberate closure.
    assert _apply(processor, _piper_action(pendant=55.0), {"gripper.pos": 0.045})["gripper.pos"] == pytest.approx(0.055)
    assert _apply(processor, _piper_action(pendant=50.0), {"gripper.pos": 0.055})["gripper.pos"] == pytest.approx(0.05)
    assert _apply(processor, _piper_action(pendant=5.0), {"gripper.pos": 0.04})["gripper.pos"] == pytest.approx(0.005)


def test_release_hysteresis_filters_noise_and_accumulates_slow_opening():
    processor = _processor(gripper_release_clearance_m=0.005)
    for pendant in (5.0, 5.4, 5.0, 5.4, 5.8):
        width = _apply(processor, _piper_action(pendant=pendant), {"gripper.pos": 0.04})["gripper.pos"]
        assert width < 0.006
    released = _apply(processor, _piper_action(pendant=6.2), {"gripper.pos": 0.04})
    assert released["gripper.pos"] == pytest.approx(0.045)


def test_release_never_exceeds_calibrated_maximum_and_reset_clears_latch():
    processor = _processor(gripper_release_clearance_m=0.005)
    _apply(processor, _piper_action(pendant=5.0), {"gripper.pos": 0.079})
    assert _apply(processor, _piper_action(pendant=7.0), {"gripper.pos": 0.079})["gripper.pos"] == pytest.approx(0.08)
    processor.reset()
    assert _apply(processor, _piper_action(pendant=7.0), {"gripper.pos": 0.04})["gripper.pos"] == pytest.approx(0.007)


@pytest.mark.parametrize("actual", [None, float("nan"), float("inf"), -0.01, 0.09, True])
def test_release_without_valid_width_feedback_keeps_absolute_mapping(actual):
    processor = _processor(gripper_release_clearance_m=0.005)
    observation = {"gripper.pos": actual}
    _apply(processor, _piper_action(pendant=5.0), observation)
    assert _apply(processor, _piper_action(pendant=7.0), observation)["gripper.pos"] == pytest.approx(0.007)


def test_release_clearance_is_opt_in_for_other_robot_setups():
    processor = _processor()
    _apply(processor, _piper_action(pendant=5.0), {"gripper.pos": 0.04})
    assert _apply(processor, _piper_action(pendant=7.0), {"gripper.pos": 0.04})["gripper.pos"] == pytest.approx(0.007)


@pytest.mark.parametrize("setting", ["gripper_release_clearance_m", "gripper_direction_deadband_m"])
@pytest.mark.parametrize("value", [-0.001, float("nan"), float("inf"), True])
def test_invalid_gripper_release_parameters_are_rejected(setting, value):
    with pytest.raises(ValueError, match=setting):
        _processor(**{setting: value})
