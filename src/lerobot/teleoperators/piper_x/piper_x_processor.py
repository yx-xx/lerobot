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

AxisRange = tuple[float, float]
Box3 = tuple[AxisRange, AxisRange, AxisRange]


def _finite_pair(bounds: AxisRange, name: str) -> AxisRange:
    if len(bounds) != 2:
        raise ValueError(f"{name} must be (min, max)")
    start, end = bounds
    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, (int, float))
        or not isinstance(end, (int, float))
        or not np.isfinite(start)
        or not np.isfinite(end)
    ):
        raise ValueError(f"{name} bounds must be finite numbers")
    if start == end:
        raise ValueError(f"{name} bounds must not be identical")
    return (float(start), float(end))


def _validate_box(box: Box3, name: str) -> Box3:
    if len(box) != 3:
        raise ValueError(f"{name} must contain x, y, z ranges")
    return (
        _finite_pair(box[0], f"{name}.x"),
        _finite_pair(box[1], f"{name}.y"),
        _finite_pair(box[2], f"{name}.z"),
    )


def map_calibrated_range(value: float, source: AxisRange, destination: AxisRange) -> float:
    """Map ``source[0] -> destination[0]`` and ``source[1] -> destination[1]``.

    The input is clipped to the source interval. Reversing a destination pair
    inverts that axis.
    """
    source_start, source_end = source
    dest_start, dest_end = destination
    low, high = (source_start, source_end) if source_start <= source_end else (source_end, source_start)
    clipped = min(max(float(value), low), high)
    fraction = (clipped - source_start) / (source_end - source_start)
    return dest_start + fraction * (dest_end - dest_start)


def _rpy_zyx_to_rotation(roll: float, pitch: float, yaw: float) -> Rotation:
    """Piper / pyAgxArm convention: R = Rz(yaw) Ry(pitch) Rx(roll)."""
    return (
        Rotation.from_rotvec(np.array([0.0, 0.0, yaw]))
        * Rotation.from_rotvec(np.array([0.0, pitch, 0.0]))
        * Rotation.from_rotvec(np.array([roll, 0.0, 0.0]))
    )


def _finite_rpy_deg(values: tuple[float, float, float], name: str) -> tuple[float, float, float]:
    if len(values) != 3:
        raise ValueError(f"{name} must be (roll, pitch, yaw) in degrees")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value)
        for value in values
    ):
        raise ValueError(f"{name} values must be finite numbers")
    return (float(values[0]), float(values[1]), float(values[2]))


def _finite_quat_xyzw(
    values: tuple[float, float, float, float], name: str
) -> tuple[float, float, float, float]:
    if len(values) != 4:
        raise ValueError(f"{name} must be (qx, qy, qz, qw)")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value)
        for value in values
    ):
        raise ValueError(f"{name} values must be finite numbers")
    quat = np.asarray(values, dtype=float)
    norm = float(np.linalg.norm(quat))
    if norm < 1e-8:
        raise ValueError(f"{name} must be a non-zero quaternion")
    quat = quat / norm
    return (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))


def align_rotation_from_corresponding_poses(
    piper_rpy_deg: tuple[float, float, float],
    franka_quat_xyzw: tuple[float, float, float, float],
) -> Rotation:
    """Build R_franka = R_align @ R_piper from one matching pair of poses."""
    roll, pitch, yaw = piper_rpy_deg
    piper_ref = _rpy_zyx_to_rotation(
        roll=np.deg2rad(roll),
        pitch=np.deg2rad(pitch),
        yaw=np.deg2rad(yaw),
    )
    franka_ref = Rotation.from_quat(np.asarray(franka_quat_xyzw, dtype=float))
    return franka_ref * piper_ref.inv()


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


@ProcessorStepRegistry.register("map_piper_x_endpose_to_franka_action")
@dataclass
class MapPiperXEndposeToFrankaAction(RobotActionProcessorStep):
    """Map Piper-X end-effector space onto Franka end-effector space.

    Position is an axis-aligned cuboid map: each Piper XYZ millimetre range is
    linearly mapped onto the matching Franka XYZ metre range. Teaching pendant
    millimetres are mapped onto Franka gripper metres the same way.

    Orientation uses one corresponding pose pair. When Piper is at
    ``piper_ref_rpy_deg``, Franka should be at ``franka_ref_quat_xyzw``. Later
    poses keep that fixed frame offset:
    ``R_franka = R_franka_ref @ R_piper_ref.inv() @ R_piper``.
    """

    piper_xyz_mm: Box3
    franka_xyz_m: Box3
    pendant_mm: AxisRange
    gripper_m: AxisRange
    piper_ref_rpy_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    franka_ref_quat_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    _align: Rotation = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.piper_xyz_mm = _validate_box(self.piper_xyz_mm, "piper_xyz_mm")
        self.franka_xyz_m = _validate_box(self.franka_xyz_m, "franka_xyz_m")
        self.pendant_mm = _finite_pair(self.pendant_mm, "pendant_mm")
        self.gripper_m = _finite_pair(self.gripper_m, "gripper_m")
        self.piper_ref_rpy_deg = _finite_rpy_deg(self.piper_ref_rpy_deg, "piper_ref_rpy_deg")
        self.franka_ref_quat_xyzw = _finite_quat_xyzw(
            self.franka_ref_quat_xyzw, "franka_ref_quat_xyzw"
        )
        self._align = align_rotation_from_corresponding_poses(
            self.piper_ref_rpy_deg, self.franka_ref_quat_xyzw
        )

    def action(self, action: RobotAction) -> RobotAction:
        _require_finite(action, PIPER_ENDPOSE_KEYS, "Piper-X action")
        target_pos = [
            map_calibrated_range(
                float(action[f"endpose.{axis}"]),
                self.piper_xyz_mm[index],
                self.franka_xyz_m[index],
            )
            for index, axis in enumerate("xyz")
        ]
        rotation = self._align * _rpy_zyx_to_rotation(
            roll=np.deg2rad(float(action["endpose.roll"])),
            pitch=np.deg2rad(float(action["endpose.pitch"])),
            yaw=np.deg2rad(float(action["endpose.yaw"])),
        )
        quat = rotation.as_quat()
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
            pendant_mm = self.pendant_mm[0]
        else:
            pendant = action[TEACHING_PENDANT_KEY]
            if (
                isinstance(pendant, bool)
                or not isinstance(pendant, (int, float))
                or not np.isfinite(pendant)
            ):
                raise ValueError("teaching_pendant.pos must be a finite number")
            pendant_mm = float(pendant)
        return float(map_calibrated_range(pendant_mm, self.pendant_mm, self.gripper_m))

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
    piper_xyz_mm: Box3,
    franka_xyz_m: Box3,
    pendant_mm: AxisRange,
    gripper_m: AxisRange,
    piper_ref_rpy_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
    franka_ref_quat_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
) -> RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction]:
    return RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[
            MapPiperXEndposeToFrankaAction(
                piper_xyz_mm=piper_xyz_mm,
                franka_xyz_m=franka_xyz_m,
                pendant_mm=pendant_mm,
                gripper_m=gripper_m,
                piper_ref_rpy_deg=piper_ref_rpy_deg,
                franka_ref_quat_xyzw=franka_ref_quat_xyzw,
            )
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
