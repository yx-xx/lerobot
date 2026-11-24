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

from lerobot.processor import make_default_processors
from lerobot.robots.franka import FrankaRobot
from lerobot.teleoperate import teleop_loop
from lerobot.teleoperators import make_teleoperator_from_config
from lerobot.utils.utils import init_logging
from lerobot.utils.visualization_utils import _init_rerun

try:
    from .common import (
        FRANKA_FIXED_QUAT_XYZW,
        GRIPPER_RELEASE_CLEARANCE_M,
        FPS,
        make_mapped_action_processor,
        make_piper_franka_configs,
    )
except ImportError:
    from common import (
        FRANKA_FIXED_QUAT_XYZW,
        GRIPPER_RELEASE_CLEARANCE_M,
        FPS,
        make_mapped_action_processor,
        make_piper_franka_configs,
    )


def main() -> None:
    init_logging()
    teleop_config, robot_config = make_piper_franka_configs()
    teleop = make_teleoperator_from_config(teleop_config)
    robot = FrankaRobot(robot_config)
    mapped_action_processor = make_mapped_action_processor()
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
        print(f"Gripper release: clear measured opening by {1000.0 * GRIPPER_RELEASE_CLEARANCE_M:.1f} mm.")

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
