#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from dataclasses import dataclass, field
from typing import Literal

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("franka")
@dataclass
class FrankaConfig(RobotConfig):
    control_mode: Literal["joint", "cartesian"] = "joint"
    node_name: str = "lerobot_franka_client"
    joint_state_topic: str = "/franka/joint_states"
    ee_pose_topic: str = "/franka/ee_pose"
    joint_command_topic: str = "/franka/joint_trajectory"
    ee_command_topic: str = "/franka/ee_pose_command"
    connect_timeout_s: float = 5.0
    state_timeout_s: float = 1.0
    command_duration_s: float = 0.1
    qos_depth: int = 10
    base_frame: str = "panda_link0"
    max_relative_target: float | dict[str, float] | None = 0.05
    max_relative_translation: float = 0.02
    max_relative_rotation: float = math.radians(10.0)
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.control_mode not in ("joint", "cartesian"):
            raise ValueError("control_mode must be either 'joint' or 'cartesian'.")

        non_empty_strings = {
            "node_name": self.node_name,
            "joint_state_topic": self.joint_state_topic,
            "ee_pose_topic": self.ee_pose_topic,
            "joint_command_topic": self.joint_command_topic,
            "ee_command_topic": self.ee_command_topic,
            "base_frame": self.base_frame,
        }
        for name, value in non_empty_strings.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string.")

        positive_values = {
            "connect_timeout_s": self.connect_timeout_s,
            "state_timeout_s": self.state_timeout_s,
            "command_duration_s": self.command_duration_s,
            "max_relative_translation": self.max_relative_translation,
            "max_relative_rotation": self.max_relative_rotation,
        }
        for name, value in positive_values.items():
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive number.")
        if self.max_relative_rotation > math.pi:
            raise ValueError("max_relative_rotation must not exceed pi radians.")
        if not isinstance(self.qos_depth, int) or isinstance(self.qos_depth, bool) or self.qos_depth <= 0:
            raise ValueError("qos_depth must be a positive integer.")

        if self.max_relative_target is not None:
            limits = (
                self.max_relative_target.values()
                if isinstance(self.max_relative_target, dict)
                else [self.max_relative_target]
            )
            if isinstance(self.max_relative_target, dict) and set(self.max_relative_target) != {
                f"j{i}" for i in range(1, 8)
            }:
                raise ValueError("max_relative_target keys must be j1 through j7.")
            if any(
                not isinstance(limit, (int, float))
                or not math.isfinite(limit)
                or limit <= 0
                for limit in limits
            ):
                raise ValueError("max_relative_target limits must be finite positive numbers.")
            if not isinstance(self.max_relative_target, dict):
                self.max_relative_target = float(self.max_relative_target)
