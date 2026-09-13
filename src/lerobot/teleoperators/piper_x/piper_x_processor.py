#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from dataclasses import dataclass, field

import numpy as np

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    ProcessorStepRegistry,
    RobotAction,
    RobotActionProcessorStep,
    RobotObservation,
    RobotProcessorPipeline,
)
from lerobot.processor.converters import (
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.utils.rotation import Rotation

PIPER_ENDPOSE_KEYS = (
    "endpose.x",
    "endpose.y",
    "endpose.z",
    "endpose.roll",
    "endpose.pitch",
    "endpose.yaw",
)
FRANKA_END_POSE_KEYS = (
    "end_pose.x",
    "end_pose.y",
    "end_pose.z",
    "end_pose.qx",
    "end_pose.qy",
    "end_pose.qz",
    "end_pose.qw",
)
TEACHING_PENDANT_KEY = "teaching_pendant.pos"
GRIPPER_KEY = "gripper.pos"
_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def _axis_alignment_matrix(axes: tuple[str, str, str]) -> np.ndarray:
    """Build R_align such that ``p_franka = R_align @ p_piper``."""
    matrix = np.zeros((3, 3), dtype=float)
    names: list[str] = []
    for row, spec in enumerate(axes):
        token = spec.strip().lower()
        if not token:
            raise ValueError("position_axes entries must not be empty")
        sign = -1.0 if token.startswith("-") else 1.0
        name = token[1:] if token[0] in "+-" else token
        if name not in _AXIS_INDEX:
            raise ValueError(f"position_axes must use x/y/z with optional signs, got {spec!r}")
        matrix[row, _AXIS_INDEX[name]] = sign
        names.append(name)
    if set(names) != set(_AXIS_INDEX):
        raise ValueError("position_axes must be a signed permutation of x, y, z")
    return matrix


def _rpy_zyx_to_rotation(roll: float, pitch: float, yaw: float) -> Rotation:
    """Piper / pyAgxArm convention: R = Rz(yaw) Ry(pitch) Rx(roll)."""
    return (
        Rotation.from_rotvec(np.array([0.0, 0.0, yaw]))
        * Rotation.from_rotvec(np.array([0.0, pitch, 0.0]))
        * Rotation.from_rotvec(np.array([roll, 0.0, 0.0]))
    )


def _require_finite(values: dict[str, float], keys: tuple[str, ...], label: str) -> None:
    missing = [key for key in keys if key not in values]
    if missing:
        raise ValueError(f"{label} is missing: {', '.join(missing)}")
    if any(
        isinstance(values[key], bool)
        or not isinstance(values[key], (int, float))
        or not np.isfinite(values[key])
        for key in keys
    ):
        raise ValueError(f"{label} values must be finite numbers")


def _piper_pose_from_action(action: RobotAction) -> tuple[np.ndarray, Rotation]:
    _require_finite(action, PIPER_ENDPOSE_KEYS, "Piper-X action")
    position_m = (
        np.array(
            [action["endpose.x"], action["endpose.y"], action["endpose.z"]],
            dtype=float,
        )
        / 1000.0
    )
    rotation = _rpy_zyx_to_rotation(
        roll=np.deg2rad(float(action["endpose.roll"])),
        pitch=np.deg2rad(float(action["endpose.pitch"])),
        yaw=np.deg2rad(float(action["endpose.yaw"])),
    )
    return position_m, rotation


@ProcessorStepRegistry.register("map_piper_x_endpose_to_franka_action")
@dataclass
class MapPiperXEndposeToFrankaAction(RobotActionProcessorStep):
    """Map Piper-X end-effector space onto Franka end-effector space.

    Each Piper pose is converted and sent as an absolute Franka cartesian
    target: millimetres to metres, ZYX RPY to an XYZW quaternion, then

        p_franka = R_align @ (position_scale * p_piper) + position_offset
        R_franka = R_align @ R_piper

    There is no first-frame latch and no incremental command.
    ``teaching_pendant.pos`` (millimetres) maps to ``gripper.pos`` (metres).
    """

    position_scale: float = 1.0
    position_axes: tuple[str, str, str] = ("x", "y", "z")
    position_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    gripper_scale: float = 1.0
    gripper_offset: float = 0.0
    gripper_min: float = 0.0
    gripper_max: float = 0.08
    _axis_matrix: np.ndarray = field(init=False, repr=False)
    _offset: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not np.isfinite(self.position_scale) or self.position_scale <= 0:
            raise ValueError("position_scale must be a finite positive number")
        if len(self.position_offset) != 3 or any(
            not isinstance(value, (int, float)) or isinstance(value, bool) or not np.isfinite(value)
            for value in self.position_offset
        ):
            raise ValueError("position_offset must be three finite numbers in metres")
        if not np.isfinite(self.gripper_scale):
            raise ValueError("gripper_scale must be finite")
        if not np.isfinite(self.gripper_offset):
            raise ValueError("gripper_offset must be finite")
        if (
            not np.isfinite(self.gripper_min)
            or not np.isfinite(self.gripper_max)
            or self.gripper_min < 0
            or self.gripper_max < self.gripper_min
        ):
            raise ValueError("gripper_min/max must be finite and increasing")
        self._axis_matrix = _axis_alignment_matrix(self.position_axes)
        self._offset = np.array(self.position_offset, dtype=float)

    def action(self, action: RobotAction) -> RobotAction:
        piper_pos, piper_rot = _piper_pose_from_action(action)
        target_pos = self._axis_matrix @ (piper_pos * self.position_scale) + self._offset
        target_rot = Rotation.from_matrix(self._axis_matrix @ piper_rot.as_matrix())
        quat = target_rot.as_quat()
        return {
            "end_pose.x": float(target_pos[0]),
            "end_pose.y": float(target_pos[1]),
            "end_pose.z": float(target_pos[2]),
            "end_pose.qx": float(quat[0]),
            "end_pose.qy": float(quat[1]),
            "end_pose.qz": float(quat[2]),
            "end_pose.qw": float(quat[3]),
            GRIPPER_KEY: self._gripper_from_pendant(action),
        }

    def _gripper_from_pendant(self, action: RobotAction) -> float:
        if TEACHING_PENDANT_KEY not in action:
            pendant_mm = 0.0
        else:
            pendant = action[TEACHING_PENDANT_KEY]
            if (
                isinstance(pendant, bool)
                or not isinstance(pendant, (int, float))
                or not np.isfinite(pendant)
            ):
                raise ValueError("teaching_pendant.pos must be a finite number")
            pendant_mm = float(pendant)
        width = pendant_mm / 1000.0 * self.gripper_scale + self.gripper_offset
        return float(min(max(width, self.gripper_min), self.gripper_max))

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        action_features = features[PipelineFeatureType.ACTION]
        for key in (*PIPER_ENDPOSE_KEYS, TEACHING_PENDANT_KEY):
            action_features.pop(key, None)
        for key in (*FRANKA_END_POSE_KEYS, GRIPPER_KEY):
            action_features[key] = PolicyFeature(type=FeatureType.ACTION, shape=(1,))
        return features


def make_piper_x_to_franka_teleop_processor(
    position_scale: float = 1.0,
    position_axes: tuple[str, str, str] = ("x", "y", "z"),
    position_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    gripper_scale: float = 1.0,
    gripper_offset: float = 0.0,
    gripper_min: float = 0.0,
    gripper_max: float = 0.08,
) -> RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction]:
    return RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[
            MapPiperXEndposeToFrankaAction(
                position_scale=position_scale,
                position_axes=position_axes,
                position_offset=position_offset,
                gripper_scale=gripper_scale,
                gripper_offset=gripper_offset,
                gripper_min=gripper_min,
                gripper_max=gripper_max,
            )
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
