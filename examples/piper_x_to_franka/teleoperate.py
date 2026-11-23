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
    sudo ip link set can0 up type can bitrate 1000000
    pip install -e ".[piper,intelrealsense]"
    # 相机路径和序列号已有默认配置；遥操作时自动启动 LeRobot 的 Rerun 界面。
    # 更换设备时再用下面两行覆盖默认配置。
    # export FRANKA_WRIST_REALSENSE_SERIAL=260422274496
    # export FRANKA_LEFT_CAMERA=/dev/v4l/by-path/pci-0000:00:14.0-usb-0:11:1.4-video-index0

Then:

    python examples/piper_x_to_franka/teleoperate.py

Position is a cuboid-to-cuboid map. For position-only collection, orientation
is fixed at the calibrated downward Franka pose, independently of the startup pose.
"""

import math
import os
from pathlib import Path

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig

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
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# 末端是 RealSense D405，左侧是 Orbbec Gemini 335。
# RealSense 使用序列号识别，Orbbec 使用固定 USB 接口的 /dev/v4l/by-path 路径识别。
# 如果设备更换，可以用环境变量覆盖下面的默认值，不用改控制逻辑。
DEFAULT_WRIST_REALSENSE_SERIAL = "260422274496"
# Orbbec 的 video-index0 在热插拔后可能指向深度/辅助节点；
# 这个 by-path 节点对应当前 USB 接口上的彩色图像流，避免 /dev/video 编号变化。
DEFAULT_LEFT_CAMERA = (
    "/dev/v4l/by-path/"
    "pci-0000:00:14.0-usb-0:11:1.4-video-index0"
)


def _camera_source(environment_name: str, default: str) -> int | Path:
    source = os.environ.get(environment_name, default).strip()
    return int(source) if source.isdigit() else Path(source)


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
        cameras={
            "wrist_camera": RealSenseCameraConfig(
                serial_number_or_name=os.environ.get(
                    "FRANKA_WRIST_REALSENSE_SERIAL", DEFAULT_WRIST_REALSENSE_SERIAL
                ),
                width=CAMERA_WIDTH,
                height=CAMERA_HEIGHT,
                fps=CAMERA_FPS,
                use_depth=False,
            ),
            "left_camera": OpenCVCameraConfig(
                index_or_path=_camera_source("FRANKA_LEFT_CAMERA", DEFAULT_LEFT_CAMERA),
                width=CAMERA_WIDTH,
                height=CAMERA_HEIGHT,
                fps=CAMERA_FPS,
            ),
        },
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
        print(f"Wrist camera: RealSense {robot_config.cameras['wrist_camera'].serial_number_or_name}")
        print(f"Left camera: {robot_config.cameras['left_camera'].index_or_path}")
        if FRANKA_FIXED_QUAT_XYZW is not None:
            print(f"Position-only mode: calibrated downward quaternion XYZW = {FRANKA_FIXED_QUAT_XYZW}")
        print(f"Connecting Piper-X on {teleop_config.can_name}. Keep dragging the arm until connected.")
        teleop.connect()
        print("Connected. Cuboid map Piper -> Franka. Ctrl+C to stop.")

        # 每次遥操作都开启 LeRobot 自带的 Rerun 界面，并显示机器人状态和两路相机。
        display_data = True
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
