#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import logging
from dataclasses import dataclass, field

import numpy as np

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    ProcessorStepRegistry,
    RobotAction,
    RobotActionProcessorStep,
    RobotObservation,
    RobotProcessorPipeline,
    TransitionKey,
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

logger = logging.getLogger(__name__)

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
    """Build the fixed tool offset in R_franka = R_piper @ R_tool_offset.

    Piper and Franka base axes follow the same XYZ correspondence as position.
    The reference pair accounts for different tool axes, not a base rotation.
    """
    roll, pitch, yaw = piper_rpy_deg
    piper_ref = _rpy_zyx_to_rotation(
        roll=np.deg2rad(roll),
        pitch=np.deg2rad(pitch),
        yaw=np.deg2rad(yaw),
    )
    franka_ref = Rotation.from_quat(np.asarray(franka_quat_xyzw, dtype=float))
    return piper_ref.inv() * franka_ref


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
    poses apply the Piper base-frame rotation increment to the Franka reference:
    ``R_franka = R_piper @ R_piper_ref.inv() @ R_franka_ref``.
    Base XYZ axes correspond as in the position map. The right-hand constant
    accounts for different tool axes without rotating the motion directions.

    ``fixed_franka_quat_xyzw`` overrides orientation mapping for position-only
    collection. An optional per-axis position deadband holds the last accepted
    target until input changes accumulate past it, including at clipped edges.

    With ``gripper_release_clearance_m > 0``, an opening reversal sets a minimum
    target above the observed Franka width. An object can block closure far above
    the mapped target; opening must clear that actual width to release it. The
    minimum is latched for the opening stroke, not recomputed on every frame.
    """

    piper_xyz_mm: Box3
    franka_xyz_m: Box3
    pendant_mm: AxisRange
    gripper_m: AxisRange
    piper_ref_rpy_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    franka_ref_quat_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    fixed_franka_quat_xyzw: tuple[float, float, float, float] | None = None
    position_deadband_m: float = 0.0
    gripper_release_clearance_m: float = 0.0
    gripper_direction_deadband_m: float = 0.001
    _tool_offset: Rotation = field(init=False, repr=False)
    _held_position: list[float] | None = field(init=False, default=None, repr=False)
    _gripper_direction_width: float | None = field(init=False, default=None, repr=False)
    _gripper_opening: bool | None = field(init=False, default=None, repr=False)
    _gripper_release_floor: float | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        self.piper_xyz_mm = _validate_box(self.piper_xyz_mm, "piper_xyz_mm")
        self.franka_xyz_m = _validate_box(self.franka_xyz_m, "franka_xyz_m")
        self.pendant_mm = _finite_pair(self.pendant_mm, "pendant_mm")
        self.gripper_m = _finite_pair(self.gripper_m, "gripper_m")
        self.piper_ref_rpy_deg = _finite_rpy_deg(self.piper_ref_rpy_deg, "piper_ref_rpy_deg")
        self.franka_ref_quat_xyzw = _finite_quat_xyzw(
            self.franka_ref_quat_xyzw, "franka_ref_quat_xyzw"
        )
        if self.fixed_franka_quat_xyzw is not None:
            self.fixed_franka_quat_xyzw = _finite_quat_xyzw(
                self.fixed_franka_quat_xyzw, "fixed_franka_quat_xyzw"
            )
        if (
            isinstance(self.position_deadband_m, bool)
            or not isinstance(self.position_deadband_m, (int, float))
            or not np.isfinite(self.position_deadband_m)
            or self.position_deadband_m < 0.0
        ):
            raise ValueError("position_deadband_m must be finite and non-negative")
        for name in ("gripper_release_clearance_m", "gripper_direction_deadband_m"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
                or value < 0.0
            ):
                raise ValueError(f"{name} must be finite and non-negative")
        self._tool_offset = align_rotation_from_corresponding_poses(
            self.piper_ref_rpy_deg, self.franka_ref_quat_xyzw
        )

    def action(self, action: RobotAction) -> RobotAction:
        required_keys = PIPER_ENDPOSE_KEYS if self.fixed_franka_quat_xyzw is None else PIPER_ENDPOSE_KEYS[:3]
        _require_finite(action, required_keys, "Piper-X action")
        target_pos = [
            map_calibrated_range(
                float(action[f"endpose.{axis}"]),
                self.piper_xyz_mm[index],
                self.franka_xyz_m[index],
            )
            for index, axis in enumerate("xyz")
        ]
        if self._held_position is not None and self.position_deadband_m > 0.0:
            target_pos = [
                held if abs(target - held) < self.position_deadband_m else target
                for target, held in zip(target_pos, self._held_position, strict=True)
            ]
        self._held_position = target_pos
        if self.fixed_franka_quat_xyzw is not None:
            quat = self.fixed_franka_quat_xyzw
        else:
            piper_rotation = _rpy_zyx_to_rotation(
                roll=np.deg2rad(float(action["endpose.roll"])),
                pitch=np.deg2rad(float(action["endpose.pitch"])),
                yaw=np.deg2rad(float(action["endpose.yaw"])),
            )
            rotation = piper_rotation * self._tool_offset
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
        width = float(map_calibrated_range(pendant_mm, self.pendant_mm, self.gripper_m))
        if self.gripper_release_clearance_m <= 0.0:
            return width

        if self._gripper_direction_width is None:
            self._gripper_direction_width = width
        elif abs(width - self._gripper_direction_width) > self.gripper_direction_deadband_m:
            opening = width > self._gripper_direction_width
            self._gripper_direction_width = width
            if not opening:
                self._gripper_release_floor = None
            elif self._gripper_opening is not True:
                observation = {}
                if self._current_transition is not None:
                    observation = self.transition.get(TransitionKey.OBSERVATION) or {}
                actual = observation.get(GRIPPER_KEY)
                low, high = sorted(self.gripper_m)
                if (
                    not isinstance(actual, bool)
                    and isinstance(actual, (int, float, np.floating))
                    and np.isfinite(actual)
                    and low <= actual <= high
                ):
                    self._gripper_release_floor = min(high, float(actual) + self.gripper_release_clearance_m)
                    if self._gripper_release_floor > width:
                        logger.info(
                            "Gripper release: mapped=%.1f mm, actual=%.1f mm, target=%.1f mm",
                            1000.0 * width, 1000.0 * actual, 1000.0 * self._gripper_release_floor,
                        )
            self._gripper_opening = opening

        if self._gripper_release_floor is not None:
            width = max(width, self._gripper_release_floor)
        return width

    def reset(self) -> None:
        self._held_position = None
        self._gripper_direction_width = None
        self._gripper_opening = None
        self._gripper_release_floor = None
        self._current_transition = None

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
    fixed_franka_quat_xyzw: tuple[float, float, float, float] | None = None,
    position_deadband_m: float = 0.0,
    gripper_release_clearance_m: float = 0.0,
    gripper_direction_deadband_m: float = 0.001,
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
                fixed_franka_quat_xyzw=fixed_franka_quat_xyzw,
                position_deadband_m=position_deadband_m,
                gripper_release_clearance_m=gripper_release_clearance_m,
                gripper_direction_deadband_m=gripper_direction_deadband_m,
            )
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
