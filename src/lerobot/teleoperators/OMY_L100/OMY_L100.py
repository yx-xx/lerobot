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

from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError


from ..teleoperator import Teleoperator
from .config_OMY_L100 import OMYL100Config
import threading
from typing import Optional
import math
from scipy.spatial.transform import Rotation as R

# rclpy is an optional runtime dependency used when this teleoperator is connected
try:
    import rclpy
    from geometry_msgs.msg import Pose
except Exception:  # pragma: no cover - if ROS2 not available, node will not connect
    rclpy = None
    Pose = None

logger = logging.getLogger(__name__)



class OMYL100(Teleoperator):

    config_class = OMYL100Config
    name = "OMY_L100"

    def __init__(self, config: OMYL100Config):
        super().__init__(config)
        self.config = config
        self.OMY_joints = {    
        "j1": float,
        "j2": float, 
        "j3": float,
        "j4": float,
        "j5": float,
        "j6": float,
        }
        # ROS related members
        self._ros_node = None
        self._ros_thread = None
        self._ros_running = False
        self._last_pose = None
        self._pose_lock = threading.Lock()


    @property
    def action_features(self) -> dict[str, type]:
        return {f"{joint}.pos": joint_type for joint, joint_type in self.OMY_joints.items()}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}
 
    @property
    def is_connected(self) -> bool:
        return getattr(self, "_ros_node", None) is not None and getattr(self, "_ros_thread", None) is not None and getattr(self, "_ros_thread").is_alive()


    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        # initialize ROS if available
        if rclpy is None or Pose is None:
            raise RuntimeError("ROS2 (rclpy) or geometry_msgs is not available in this environment")

        try:
            # rclpy.init() may raise if already initialized; ignore in that case
            rclpy.init()
        except Exception:
            # ignore initialization errors (likely already initialized)
            pass

        # create node and subscription
        self._ros_node = rclpy.create_node(f"{self.name.lower()}_teleoperator_node")

        def _pose_cb(msg: Pose):
            # store the latest pose thread-safely
            with self._pose_lock:
                self._last_pose = msg

        # subscribe to end effector pose
        self._ros_node.create_subscription(Pose, "/end_effector_pose", _pose_cb, 10)

        # run a spin loop in a background thread so callbacks are processed
        self._ros_running = True

        def _spin_loop():
            try:
                while self._ros_running:
                    rclpy.spin_once(self._ros_node, timeout_sec=0.1)
            except Exception:
                logger.exception("Exception in ROS spin loop")

        self._ros_thread = threading.Thread(target=_spin_loop, name=f"{self.name}_ros_spin", daemon=True)
        self._ros_thread.start()

        # Run any configuration steps
        self.configure()
        logger.info(f"{self} connected.")


    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass


    def configure(self) -> None:
        pass


    def get_action(self) -> dict[str, float]:
        # Return the latest end-effector pose as position (x,y,z) and Euler angles (roll, pitch, yaw)
        with self._pose_lock:
            pose = self._last_pose

        if pose is None:
            # no data yet: return zeros
            logger.debug(f"{self} no pose received yet, returning zeros")
            return {
                # positions in millimeters, orientations in degrees
                "ee.x": 0.0,
                "ee.y": 0.0,
                "ee.z": 0.0,
                "ee.roll": 0.0,
                "ee.pitch": 0.0,
                "ee.yaw": 0.0,
            }

        # extract position
        # pose positions are expected in meters; convert to millimeters for output
        x = float(pose.position.x) * 1000.0
        y = float(pose.position.y) * 1000.0
        z = float(pose.position.z) * 1000.0

        # convert quaternion to Euler angles (ZYX order) in degrees in base frame
        qx = pose.orientation.x
        qy = pose.orientation.y
        qz = pose.orientation.z
        qw = pose.orientation.w

        # scipy Rotation expects quaternion as [x, y, z, w]
        rot = R.from_quat([qx, qy, qz, qw])
        # as_euler('xyz') returns [x, y, z] angles
        x_y_z = rot.as_euler('xyz', degrees=True)
        roll_deg, pitch_deg, yaw_deg = float(x_y_z[0]), float(x_y_z[1]), float(x_y_z[2])

        action = {
            "ee.x": x,
            "ee.y": y,
            "ee.z": z,
            "ee.roll": float(roll_deg),
            "ee.pitch": float(pitch_deg),
            "ee.yaw": float(yaw_deg),
        }

        # logger.debug(
        #     f"{self} read action (pose): pos=({x:.3f}mm,{y:.3f}mm,{z:.3f}mm) rpy_deg=({roll_deg:.3f},{pitch_deg:.3f},{yaw_deg:.3f})"
        # )
        return action


    def send_feedback(self, feedback: dict[str, float]) -> None:
        # TODO(rcadene, aliberts): Implement force feedback
        raise NotImplementedError


    def disconnect(self) -> None:
        if not self.is_connected:
            DeviceNotConnectedError(f"{self} is not connected.")

        # stop ROS spin thread and destroy node
        try:
            self._ros_running = False
            if self._ros_thread is not None:
                self._ros_thread.join(timeout=1.0)
            if self._ros_node is not None:
                try:
                    self._ros_node.destroy_node()
                except Exception:
                    # ignore destroy errors
                    pass
        finally:
            self._ros_thread = None
            self._ros_node = None

        logger.info(f"{self} disconnected.")
