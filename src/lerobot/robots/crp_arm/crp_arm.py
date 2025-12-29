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

import logging
import time
from functools import cached_property
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import (
    FeetechMotorsBus,
    OperatingMode,
)

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_crp_arm import CRPArmConfig


# import the CrpRobotPy
from CrpRobotPy import CrpRobotPy, RobotMode


logger = logging.getLogger(__name__)


class CRPArm(Robot):

    config_class = CRPArmConfig
    name = "crp_arm"

    def __init__(self, config: CRPArmConfig):
        super().__init__(config)
        self.config = config
        
        self.crp_arm_robot = CrpRobotPy()

        self.crp_joints = {    
        "j1": float,
        "j2": float, 
        "j3": float,
        "j4": float,
        "j5": float,
        "j6": float,
        }

        self.cameras = make_cameras_from_configs(config.cameras)

    # @property
    # def _motors_ft(self) -> dict[str, type]:
    #     return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{joint}.pos": joint_type for joint, joint_type in self.crp_joints.items()}
    
    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    
    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft


    @property
    def is_connected(self) -> bool:
        return self.crp_arm_robot.is_connected() and all(cam.is_connected for cam in self.cameras.values())

    def connect(self, calibrate: bool = True) -> None:
        """
        We assume that at connection time, arm is in a rest position,
        and torque can be safely disabled to run calibration.
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")
        # self.bus.connect() #连接电机
        # if not self.is_calibrated and calibrate:
        #     logger.info(
        #         "Mismatch between calibration values in the motor and the calibration file or no calibration file found"
        #     )
        #     self.calibrate()

        self.crp_arm_robot.connect(self.config.port)
        self.crp_arm_robot.servo_power_on()
        self.crp_arm_robot.switch_work_mode(RobotMode.Manual)

        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        logger.info(f"{self} connected.")

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        # self.bus.disconnect(self.config.disable_torque_on_disconnect)  #断连电机

        self.crp_arm_robot.servo_power_off()
        self.crp_arm_robot.disconnect()

        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")




    # 校准
    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass



    def configure(self) -> None:
        pass



    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Read arm position
        start = time.perf_counter()

        # obs_dict = self.bus.sync_read("Present_Position")
        # obs_dict = {f"{motor}.pos": val for motor, val in obs_dict.items()}

        crp_joints_dict = self.crp_arm_robot.read_joints()

        obs_dict = {f"{motor}.pos": val for motor, val in crp_joints_dict.items()}

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict


    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Command arm to move to a target joint configuration.

        The relative action magnitude may be clipped depending on the configuration parameter
        `max_relative_target`. In this case, the action sent differs from original action.
        Thus, this function always returns the action actually sent.

        Raises:
            RobotDeviceNotConnectedError: if robot is not connected.

        Returns:
            the action sent to the motors, potentially clipped.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}

        self.crp_arm_robot.movej(goal_pos)
 
        return {f"{motor}.pos": val for motor, val in goal_pos.items()}


    def send_endpose(self, endpose: list[float]):
        if len(endpose) != 6:
            print("crp_arm: [ERROR] endpose必须包含6个值")
            return

        try:
            _ = self.crp_arm_robot.movel_user(endpose)
        except Exception as e:
            print("crp_arm: [ERROR] movel_user机械臂运动失败", e)




    def send_GPs(self, start_index: int, GPs: list[float]):
        self.crp_arm_robot.set_GPs(start_index, GPs)
        return

    def set_GI(self, index: int, value: int) -> bool:
        self.crp_arm_robot.set_GI(index, value)
        return

    def get_GI(self, index: int) -> int:
        return self.crp_arm_robot.get_GI(index)




    def set_speed_ratio(self, ratio: int):
        self.crp_arm_robot.set_speed_ratio(ratio)
        return
    
    def get_speed_ratio(self) -> int:
        return self.crp_arm_robot.get_speed_ratio()
    
    def get_current_endpose(self) -> list[float]:
        """
        获取当前末端位置姿态
        返回: [x, y, z, roll, pitch, yaw]
        """
        x, y, z, roll, pitch, yaw = self.crp_arm_robot.read_end_pose_user()
        return [x, y, z, roll, pitch, yaw]
    