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

Piper-X end-effector pose is mapped directly into Franka cartesian space
(mm to m, ZYX RPY to XYZW). The teaching pendant maps to ``gripper.pos``
(mm to m). There is no incremental latch. If the axes feel mirrored, change
``POSITION_AXES``. If the two workspaces do not overlap, set
``POSITION_OFFSET`` in metres.
"""

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

FPS = 15
POSITION_SCALE = 1.0
POSITION_AXES = ("x", "y", "z")
POSITION_OFFSET = (0.0, 0.0, 0.0)


def main() -> None:
    teleop_config = PiperXTeleoperatorConfig(
        id="piper_x",
        can_name=os.environ.get("PIPER_CAN_NAME", "can0"),
    )
    robot_config = FrankaConfig(
        id="panda",
        control_mode="cartesian",
    )
    teleop = make_teleoperator_from_config(teleop_config)
    robot = FrankaRobot(robot_config)
    teleop_action_processor = make_piper_x_to_franka_teleop_processor(
        position_scale=POSITION_SCALE,
        position_axes=POSITION_AXES,
        position_offset=POSITION_OFFSET,
    )
    _, robot_action_processor, robot_observation_processor = make_default_processors()

    print("Connecting Franka ROS 2 client. Confirm joint, end_pose, and gripper topics.")
    robot.connect()
    print(f"Connecting Piper-X on {teleop_config.can_name}. Keep dragging the arm until connected.")
    teleop.connect()
    print("Connected. Piper endpose maps directly to Franka end_pose. Ctrl+C to stop.")

    _init_rerun(session_name="piper_x_to_franka_teleop")
    try:
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
        teleop.disconnect()
        robot.disconnect()


if __name__ == "__main__":
    main()
