#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from dataclasses import dataclass

from ..config import TeleoperatorConfig

PIPER_X_FIRMWARE_VERSIONS = ("default", "v183", "v188", "v189")


@TeleoperatorConfig.register_subclass("piper_x")
@dataclass
class PiperXTeleoperatorConfig(TeleoperatorConfig):
    """Configuration for an AgileX Piper-X used as a teaching input arm.

    The adapter talks to the vendored ``third_party/pyAgxArm`` SDK. Firmware
    ``S-V1.9-0`` maps to ``firmware_version="v189"`` (S-V1.8-9 and later).
    """

    can_name: str = "can0"
    can_interface: str = "socketcan"
    can_bitrate: int = 1_000_000
    firmware_version: str = "v189"
    # Kept as the public CAN-precheck switch used by existing scripts.
    # Mapped to pyAgxArm ``enable_check_can``.
    judge_flag: bool = True
    configure_master_mode: bool = True
    # Real hardware: the leader (0x15x) stream may start ~25s after the
    # master config is applied; the feedback (0x2Ax) stream is usually live
    # immediately. The firmware only broadcasts while the arm moves.
    startup_timeout_s: float = 30.0
    data_timeout_s: float = 1.0

    def __post_init__(self) -> None:
        if not self.can_name:
            raise ValueError("can_name must not be empty")
        if not self.can_interface:
            raise ValueError("can_interface must not be empty")
        if self.can_bitrate <= 0:
            raise ValueError("can_bitrate must be positive")
        if self.firmware_version not in PIPER_X_FIRMWARE_VERSIONS:
            raise ValueError(
                f"firmware_version must be one of {PIPER_X_FIRMWARE_VERSIONS}, "
                f"got {self.firmware_version!r}"
            )
        if self.startup_timeout_s <= 0:
            raise ValueError("startup_timeout_s must be positive")
        if self.data_timeout_s <= 0:
            raise ValueError("data_timeout_s must be positive")
