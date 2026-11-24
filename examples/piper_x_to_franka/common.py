#!/usr/bin/env python

"""Shared Piper-X to Franka hardware configuration.

Keep the teleoperation, recording, and evaluation entry points on the same
camera, workspace, gripper, and orientation calibration values.
"""

import math
import os
from pathlib import Path

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.robots.franka import FrankaConfig
from lerobot.teleoperators.piper_x import (
    PiperXTeleoperatorConfig,
    make_piper_x_to_franka_teleop_processor,
)

FPS = 30
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# End-effector RealSense D405; left-side Orbbec Gemini 335.
DEFAULT_WRIST_REALSENSE_SERIAL = "260422274496"
DEFAULT_LEFT_CAMERA = (
    "/dev/v4l/by-path/"
    "pci-0000:00:14.0-usb-0:11:1.4-video-index0"
)

# Suppress small mapped position changes caused by sensor noise.
POSITION_DEADBAND_M = 0.0015

# Piper-X end-effector workspace, in millimetres relative to the Piper base.
PIPER_X_MM = (150.0, 450.0)
PIPER_Y_MM = (-290.0, 290.0)
PIPER_Z_MM = (120.0, 500.0)

# Franka end-effector workspace, in metres relative to panda_link0.
FRANKA_X_M = (0.35, 0.80)
FRANKA_Y_M = (-0.35, 0.35)
FRANKA_Z_M = (0.17, 0.60)

# Teaching pendant opening in millimetres -> Franka gripper opening in metres.
PENDANT_MM = (51.0, 98.0)
GRIPPER_M = (0.0045, 0.08)
# On an opening reversal, clear the measured object width immediately.
# Hold the 5 mm clearance for this opening stroke; 1 mm input hysteresis rejects noise.
GRIPPER_RELEASE_CLEARANCE_M = 0.005
GRIPPER_DIRECTION_DEADBAND_M = 0.001

# Piper reference RPY and Franka reference quaternion from orientation calibration.
PIPER_REF_RPY_DEG = (-177.8787291384727, -3.2697089814813505, -84.67317794279184)
FRANKA_REF_QUAT_XYZW = (
    0.9992919885928564,
    -0.03241738912274881,
    0.00980888008203098,
    0.016383537106962606,
)

# Position-only mode keeps the calibrated downward Franka orientation.
FRANKA_FIXED_QUAT_XYZW = FRANKA_REF_QUAT_XYZW


def _camera_source(environment_name: str, default: str) -> int | Path:
    source = os.environ.get(environment_name, default).strip()
    return int(source) if source.isdigit() else Path(source)


def make_piper_franka_configs() -> tuple[PiperXTeleoperatorConfig, FrankaConfig]:
    """Build the Piper-X teleoperator and Franka robot configs."""
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
    return teleop_config, robot_config


def make_mapped_action_processor():
    """Create the Piper-X -> Franka action mapping used by all workflows."""
    return make_piper_x_to_franka_teleop_processor(
        piper_xyz_mm=(PIPER_X_MM, PIPER_Y_MM, PIPER_Z_MM),
        franka_xyz_m=(FRANKA_X_M, FRANKA_Y_M, FRANKA_Z_M),
        pendant_mm=PENDANT_MM,
        gripper_m=GRIPPER_M,
        piper_ref_rpy_deg=PIPER_REF_RPY_DEG,
        franka_ref_quat_xyzw=FRANKA_REF_QUAT_XYZW,
        fixed_franka_quat_xyzw=FRANKA_FIXED_QUAT_XYZW,
        position_deadband_m=POSITION_DEADBAND_M,
        gripper_release_clearance_m=GRIPPER_RELEASE_CLEARANCE_M,
        gripper_direction_deadband_m=GRIPPER_DIRECTION_DEADBAND_M,
    )
