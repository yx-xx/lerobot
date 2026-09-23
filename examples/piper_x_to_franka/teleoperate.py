# !/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Teleoperate a remote Franka arm from a local Piper-X teaching arm.

The Franka control computer must already be running ``ros2/franka_ros2_bridge``.
This script talks to that bridge over ROS 2 and never imports libfranka.

On this machine, before starting Python:

    source /opt/ros/humble/setup.bash
    export ROS_DOMAIN_ID=23
    pip install -e ".[piper]"
    sudo ip link set can0 up type can bitrate 1000000

Then:

    python examples/piper_x_to_franka/teleoperate.py

Position is a cuboid-to-cuboid map. Orientation mapping remains calibrated, but
can be temporarily disabled so Franka keeps the orientation measured at startup.
"""

import math
import os

from lerobot.processor import make_default_processors
from lerobot.robots.franka import FrankaConfig, FrankaRobot
from lerobot.teleoperate import teleop_loop
from lerobot.teleoperators import make_teleoperator_from_config
from lerobot.teleoperators.piper_x import (
    PiperXTeleoperatorConfig,
    make_piper_x_to_franka_teleop_processor,
)
from lerobot.utils.visualization_utils import _init_rerun

FPS = 30

# 启动姿态锁定
LOCK_FRANKA_STARTUP_ORIENTATION = True

# 标定：Piper 末端工作空间，单位毫米，相对 Piper 基座。
PIPER_X_MM = (150.0, 450.0)
PIPER_Y_MM = (-290.0, 290.0)
PIPER_Z_MM = (120.0, 500.0)

# 标定：Franka 末端工作空间，单位米，相对 panda_link0。
FRANKA_X_M = (0.40, 0.60)
FRANKA_Y_M = (-0.35, 0.35)
FRANKA_Z_M = (0.17, 0.60)

# 标定：示教器开口（毫米）→ Franka 夹爪开口（米）。
PENDANT_MM = (51.0, 98.0)
GRIPPER_M = (0.0045, 0.0846)

# 标定：对应姿态。
PIPER_REF_RPY_DEG = (-177.32, -3.01, -86.34)
FRANKA_REF_QUAT_XYZW = (0.9997537578, -0.0156632982, 0.0147608992, 0.0054037621)


def main() -> None:
    teleop_config = PiperXTeleoperatorConfig(
        id="piper_x",
        can_name=os.environ.get("PIPER_CAN_NAME", "can0"),
    )
    robot_config = FrankaConfig(
        id="panda",
        control_mode="cartesian",
        state_timeout_s=3.0,
        max_relative_translation=0.15,
        max_relative_rotation=math.radians(30.0),
    )
    teleop = make_teleoperator_from_config(teleop_config)
    robot = FrankaRobot(robot_config)
    mapped_action_processor = make_piper_x_to_franka_teleop_processor(
        piper_xyz_mm=(PIPER_X_MM, PIPER_Y_MM, PIPER_Z_MM),
        franka_xyz_m=(FRANKA_X_M, FRANKA_Y_M, FRANKA_Z_M),
        pendant_mm=PENDANT_MM,
        gripper_m=GRIPPER_M,
        piper_ref_rpy_deg=PIPER_REF_RPY_DEG,
        franka_ref_quat_xyzw=FRANKA_REF_QUAT_XYZW,
    )
    quaternion_keys = ("end_pose.qx", "end_pose.qy", "end_pose.qz", "end_pose.qw")
    fixed_quaternion: tuple[float, float, float, float] | None = None

    def teleop_action_processor(data):
        if fixed_quaternion is None:
            raise RuntimeError("Franka startup orientation has not been read")
        action = mapped_action_processor(data)
        if LOCK_FRANKA_STARTUP_ORIENTATION:
            for key, value in zip(quaternion_keys, fixed_quaternion):
                action[key] = value
        return action

    _, robot_action_processor, robot_observation_processor = make_default_processors()

    try:
        print("Connecting Franka ROS 2 client. Confirm joint, end_pose, and gripper topics.")
        robot.connect()
        initial_observation = robot.get_observation()
        initial_position = tuple(float(initial_observation[f"end_pose.{axis}"]) for axis in "xyz")
        fixed_quaternion = tuple(float(initial_observation[key]) for key in quaternion_keys)
        print(f"Franka startup XYZ (m): {initial_position}")
        print(f"Franka startup quaternion XYZW: {fixed_quaternion}")
        if LOCK_FRANKA_STARTUP_ORIENTATION:
            print("Franka startup orientation is locked for this run.")
        print(f"Connecting Piper-X on {teleop_config.can_name}. Keep dragging the arm until connected.")
        teleop.connect()
        print("Connected. Cuboid map Piper -> Franka. Ctrl+C to stop.")

        _init_rerun(session_name="piper_x_to_franka_teleop")
        teleop_loop(
            teleop=teleop,
            robot=robot,
            fps=FPS,
            display_data=True,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
        )
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        if teleop.is_connected:
            teleop.disconnect()
        if robot.is_connected:
            robot.disconnect()


if __name__ == "__main__":
    main()
