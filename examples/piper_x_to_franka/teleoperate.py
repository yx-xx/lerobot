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

Position is a cuboid-to-cuboid map. For position-only collection, orientation
is fixed at the calibrated downward Franka pose, independently of the startup pose.
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

# 每轴 1.5 mm 的目标死区；抑制静止/映射边缘噪声，移动累计超过死区后继续跟随。
POSITION_DEADBAND_M = 0.0015

# 标定：Piper 末端工作空间，单位毫米，相对 Piper 基座。
PIPER_X_MM = (150.0, 450.0)
PIPER_Y_MM = (-290.0, 290.0)
PIPER_Z_MM = (120.0, 500.0)

# 标定：Franka 末端工作空间，单位米，相对 panda_link0。
FRANKA_X_M = (0.35, 0.80)
FRANKA_Y_M = (-0.35, 0.35)
FRANKA_Z_M = (0.17, 0.60)

# 标定：示教器开口（毫米）→ Franka 夹爪开口（米）。
PENDANT_MM = (51.0, 98.0)
# 与 Franka 客户端和桥接器的 80 mm 上限一致，避免全开后的闭合输入先被截断。
GRIPPER_M = (0.0045, 0.08)

# 标定：对应姿态。用同目录 calibrate_orientation.py 一次性采集后，一起替换这两个常量。
# Piper 使用 RPY 角度；Franka 四元数顺序为 (qx, qy, qz, qw)。
PIPER_REF_RPY_DEG = (-177.8787291384727, -3.2697089814813505, -84.67317794279184)
FRANKA_REF_QUAT_XYZW = (
    0.9992919885928564,
    -0.03241738912274881,
    0.00980888008203098,
    0.016383537106962606,
)

# 标定时的 Franka 姿态就是实际夹爪朝下的姿态；固定模式直接复用整组四元数。
# 保留标定时的完整朝向，包括夹爪绕竖直方向的转角。设为 None 才恢复主臂姿态跟随。
FRANKA_FIXED_QUAT_XYZW = FRANKA_REF_QUAT_XYZW


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
        qos_depth=1,
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
        fixed_franka_quat_xyzw=FRANKA_FIXED_QUAT_XYZW,
        position_deadband_m=POSITION_DEADBAND_M,
    )
    quaternion_keys = ("end_pose.qx", "end_pose.qy", "end_pose.qz", "end_pose.qw")

    _, robot_action_processor, robot_observation_processor = make_default_processors()

    try:
        print("Connecting Franka ROS 2 client. Confirm joint, end_pose, and gripper topics.")
        robot.connect()
        initial_observation = robot.get_observation()
        initial_position = tuple(float(initial_observation[f"end_pose.{axis}"]) for axis in "xyz")
        startup_quaternion = tuple(float(initial_observation[key]) for key in quaternion_keys)
        print(f"Franka startup XYZ (m): {initial_position}")
        print(f"Franka startup quaternion XYZW: {startup_quaternion}")
        if FRANKA_FIXED_QUAT_XYZW is not None:
            print(f"Position-only mode: calibrated downward quaternion XYZW = {FRANKA_FIXED_QUAT_XYZW}")
        print(f"Connecting Piper-X on {teleop_config.can_name}. Keep dragging the arm until connected.")
        teleop.connect()
        print("Connected. Cuboid map Piper -> Franka. Ctrl+C to stop.")

        display_data = os.environ.get("TELEOP_DISPLAY_DATA", "0") == "1"
        if display_data:
            _init_rerun(session_name="piper_x_to_franka_teleop")
        teleop_loop(
            teleop=teleop,
            robot=robot,
            fps=FPS,
            display_data=display_data,
            teleop_action_processor=mapped_action_processor,
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
