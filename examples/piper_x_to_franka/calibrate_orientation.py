#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""一次性采集 Piper-X / Franka 对应姿态，打印供遥操作使用的固定常量。

先按 teleoperate.py 的说明准备 ROS 2、CAN 和 Franka 桥接器，停掉遥操作程序。
Franka 预先摆到参考姿态（固定模式下希望保持的朝下姿态），再运行：

    python examples/piper_x_to_franka/calibrate_orientation.py

连接时持续拖动 Piper，连接后把 Piper 摆到与 Franka 对应的参考姿态。
两臂保持静止，按回车采集。脚本只读取 Franka 状态，不发送运动或夹爪命令。
"""

import os

from lerobot.robots.franka import FrankaConfig, FrankaRobot
from lerobot.teleoperators.piper_x import PiperXTeleoperator, PiperXTeleoperatorConfig


def main() -> None:
    robot = FrankaRobot(FrankaConfig(id="panda", control_mode="cartesian", state_timeout_s=3.0))
    teleop = PiperXTeleoperator(
        PiperXTeleoperatorConfig(id="piper_x", can_name=os.environ.get("PIPER_CAN_NAME", "can0"))
    )
    try:
        print("连接 Franka ROS 2 状态。请保持 Franka 在预先摆好的参考姿态。")
        robot.connect()
        print(f"连接 Piper-X（{teleop.config.can_name}），连接期间请持续拖臂。")
        teleop.connect()
        input("把 Piper 摆到与 Franka 对应的参考姿态，两臂保持静止，按回车采集：")

        piper = teleop.get_action()
        franka = robot.get_observation()
        piper_rpy = tuple(float(piper[f"endpose.{axis}"]) for axis in ("roll", "pitch", "yaw"))
        # ROS / LeRobot 按字段名取 XYZW；只有 Affine 构造函数需要 WXYZ。
        franka_quat = tuple(float(franka[f"end_pose.{axis}"]) for axis in ("qx", "qy", "qz", "qw"))

        print("\n把下面两行一起替换到 examples/piper_x_to_franka/teleoperate.py：")
        print(f"PIPER_REF_RPY_DEG = {piper_rpy!r}")
        print(f"FRANKA_REF_QUAT_XYZW = {franka_quat!r}")
        print("固定模式会直接保持 FRANKA_REF_QUAT_XYZW；请将 Franka 摆到希望保持的朝下姿态再采集。")
        print("将 FRANKA_FIXED_QUAT_XYZW 设为 None 后才启用姿态遥操作。")
    except KeyboardInterrupt:
        print("\n已取消采集。")
    finally:
        if teleop.is_connected:
            teleop.disconnect()
        if robot.is_connected:
            robot.disconnect()


if __name__ == "__main__":
    main()
