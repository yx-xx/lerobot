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

"""Run Piper-X as a LeRobot teleoperator and print get_action()."""

import os
import time

from lerobot.teleoperators import make_teleoperator_from_config
from lerobot.teleoperators.piper_x import PiperXTeleoperatorConfig
from lerobot.utils.robot_utils import busy_wait

FPS = 30

config = PiperXTeleoperatorConfig(
    id="piper_x",
    can_name=os.environ.get("PIPER_CAN_NAME", "can0"),
)
teleop = make_teleoperator_from_config(config)

print(f"Connecting on {config.can_name}. Drag the arm until connected.")
teleop.connect()
print("Teleop loop. Drag the arm and teaching pendant; Ctrl+C to stop.")
print("joints (deg) | endpose + teaching pendant")

try:
    while True:
        t0 = time.perf_counter()
        action = teleop.get_action()
        joints = teleop.get_joint_angles()
        print(
            " ".join(f"{joints[key]:7.2f}" for key in teleop.joint_features),
            "|",
            " ".join(f"{action[key]:8.1f}" for key in teleop.action_features),
        )
        busy_wait(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
except KeyboardInterrupt:
    print("Stopped.")
finally:
    teleop.disconnect()
