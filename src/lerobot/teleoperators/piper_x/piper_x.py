#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import logging
import math
import sys
import time
from pathlib import Path
from typing import Any

from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..teleoperator import Teleoperator
from .configuration_piper_x import PiperXTeleoperatorConfig

logger = logging.getLogger(__name__)

ENDPOSE_FEATURES = (
    "endpose.x",
    "endpose.y",
    "endpose.z",
    "endpose.roll",
    "endpose.pitch",
    "endpose.yaw",
)
JOINT_FEATURES = (
    "joint1.pos",
    "joint2.pos",
    "joint3.pos",
    "joint4.pos",
    "joint5.pos",
    "joint6.pos",
)
TEACHING_PENDANT_FEATURES = ("teaching_pendant.pos",)
_JOINT_SOURCES = ("leader", "feedback")
# Leader teaching-pendant control codes: 0x04..0x07 are angle mode.
_TEACHING_PENDANT_ANGLE_STATUS = {0x04, 0x05, 0x06, 0x07}


def _candidate_pyagxarm_roots() -> list[Path]:
    here = Path(__file__).resolve()
    candidates = [
        here.parents[4] / "third_party" / "pyAgxArm",
        Path.cwd() / "third_party" / "pyAgxArm",
    ]
    for parent in here.parents:
        candidates.append(parent / "third_party" / "pyAgxArm")
    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique.append(candidate)
    return unique


def _ensure_pyagxarm_on_path() -> Path | None:
    """Make the vendored AgileX SDK importable without a separate pip install."""
    try:
        import pyAgxArm  # noqa: F401

        return Path(pyAgxArm.__file__).resolve().parent.parent
    except ImportError:
        pass

    for sdk_root in _candidate_pyagxarm_roots():
        if sdk_root.is_dir():
            if str(sdk_root) not in sys.path:
                sys.path.insert(0, str(sdk_root))
            return sdk_root
    return None


def _make_piper_arm(config: PiperXTeleoperatorConfig):
    sdk_root = _ensure_pyagxarm_on_path()
    try:
        from pyAgxArm import AgxArmFactory, ArmModel, create_agx_arm_config
    except ImportError as exc:
        missing = getattr(exc, "name", "") or str(exc)
        if missing == "can" or "No module named 'can'" in str(exc):
            raise ImportError(
                "Piper-X support requires python-can. Install it with `pip install python-can`."
            ) from exc
        hint = (
            f" Keep `{sdk_root}` in the repo, or run "
            "`pip install -e third_party/pyAgxArm`."
            if sdk_root is not None
            else " Clone pyAgxArm into `third_party/pyAgxArm`."
        )
        raise ImportError(
            "Piper-X support requires the vendored AgileX SDK in "
            f"`third_party/pyAgxArm`.{hint}"
        ) from exc

    arm_config = create_agx_arm_config(
        robot=ArmModel.PIPER_X,
        firmeware_version=config.firmware_version,
        channel=config.can_name,
        interface=config.can_interface,
        bitrate=config.can_bitrate,
        enable_check_can=bool(config.judge_flag),
    )
    return AgxArmFactory.create_arm(arm_config)


def _message_values(message: Any, expected_len: int) -> list[float] | None:
    values = getattr(message, "msg", None)
    if not isinstance(values, (list, tuple)) or len(values) < expected_len:
        return None
    parsed = [float(value) for value in values[:expected_len]]
    if not all(math.isfinite(value) for value in parsed):
        return None
    return parsed


def _pose_mm_deg_from_m_rad(pose_m_rad: list[float]) -> dict[str, float]:
    x, y, z, roll, pitch, yaw = pose_m_rad
    return {
        "endpose.x": x * 1000.0,
        "endpose.y": y * 1000.0,
        "endpose.z": z * 1000.0,
        "endpose.roll": math.degrees(roll),
        "endpose.pitch": math.degrees(pitch),
        "endpose.yaw": math.degrees(yaw),
    }


def _joints_deg_from_rad(joints_rad: list[float]) -> dict[str, float]:
    return {
        name: math.degrees(value)
        for name, value in zip(JOINT_FEATURES, joints_rad, strict=True)
    }


def _parse_teaching_pendant_message(message: Any) -> tuple[float, dict[str, float]] | None:
    payload = getattr(message, "msg", None)
    if payload is None:
        return None
    timestamp = float(getattr(message, "timestamp", 0.0) or 0.0)
    if timestamp <= 0:
        return None
    try:
        value = float(payload.value)
    except (TypeError, ValueError, AttributeError):
        return None
    if not math.isfinite(value):
        return None

    mode = getattr(payload, "mode", None)
    if mode is None and hasattr(payload, "status_code"):
        try:
            status_code = int(payload.status_code)
        except (TypeError, ValueError):
            status_code = 0
        # 0x159 leader frames are always decoded as metres by pyAgxArm.
        # Angle-mode teaching pendants are actually millidegrees; *1000 recovers deg.
        if status_code in _TEACHING_PENDANT_ANGLE_STATUS:
            value = value * 1000.0
        else:
            value = value * 1000.0
    elif str(mode or "width").lower() != "angle":
        value = value * 1000.0

    return timestamp, {"teaching_pendant.pos": value}


class PiperXTeleoperator(Teleoperator):
    """AgileX Piper-X teaching arm backed by ``third_party/pyAgxArm``.

    ``get_action`` emits the absolute end-effector pose in millimetres and
    XYZ Euler degrees, relative to the Piper-X ``base_link`` frame.
    ``get_joint_angles`` emits the six joint angles in degrees.

    Hardware notes (verified on firmware S-V1.9-0):
    - The arm broadcasts joint state ONLY while it moves; a few seconds after
      it stops, the stream goes quiet. Read methods therefore serve the last
      known state while the stream is idle, and ``connect`` expects the arm to
      be moved (dragged) at least once within ``startup_timeout_s``.
    - Joint state may arrive either as the leader/control stream or as the
      feedback stream; the adapter locks onto whichever stream ticks and
      follows it if the arm switches.
    """

    config_class = PiperXTeleoperatorConfig
    name = "piper_x"

    def __init__(self, config: PiperXTeleoperatorConfig):
        super().__init__(config)
        self.config = config
        self._arm = None
        self._teaching_pendant = None
        self._joint_source: str | None = None
        self._last_sdk_timestamps: dict[str, float] = {}
        self._last_local_update = 0.0
        self._last_joints_rad: list[float] | None = None
        self._last_pose_m_rad: list[float] | None = None
        self._last_teaching_pendant: dict[str, float] | None = None

    @property
    def action_features(self) -> dict[str, type]:
        return dict.fromkeys((*ENDPOSE_FEATURES, *TEACHING_PENDANT_FEATURES), float)

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def joint_features(self) -> dict[str, type]:
        return dict.fromkeys(JOINT_FEATURES, float)

    @property
    def teaching_pendant_features(self) -> dict[str, type]:
        return dict.fromkeys(TEACHING_PENDANT_FEATURES, float)

    @property
    def is_connected(self) -> bool:
        return self._arm is not None and bool(self._arm.is_connected())

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self._arm = _make_piper_arm(self.config)
        try:
            self._init_teaching_pendant()
            self._arm.connect()
            self.configure()
            self._wait_for_first_frame()
        except Exception:
            try:
                if self._arm is not None:
                    self._arm.disconnect()
            finally:
                self._reset_runtime_state()
            raise

        logger.info("%s connected on %s", self, self.config.can_name)

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        # Piper-X calibration and follower-frame alignment are handled outside
        # this adapter. The SDK supplies an already calibrated kinematic model.
        return None

    def configure(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self.config.configure_master_mode:
            self._arm.set_leader_mode()

    def _init_teaching_pendant(self) -> None:
        try:
            self._teaching_pendant = self._arm.init_effector(self._arm.OPTIONS.EFFECTOR.AGX_GRIPPER)
        except Exception:
            logger.warning("%s failed to initialize teaching pendant", self, exc_info=True)
            self._teaching_pendant = None

    def _refresh_teaching_pendant(self) -> None:
        if self._teaching_pendant is None:
            return
        # Leader/teaching-pendant motion is on 0x159 control frames.
        readers = (
            getattr(self._teaching_pendant, "get_gripper_ctrl_states", None),
            getattr(self._teaching_pendant, "get_gripper_status", None),
        )
        for reader in readers:
            if reader is None:
                continue
            try:
                parsed = _parse_teaching_pendant_message(reader())
            except Exception:
                logger.debug("%s failed to read teaching pendant", self, exc_info=True)
                continue
            if parsed is None:
                continue
            _timestamp, pendant = parsed
            self._last_teaching_pendant = pendant
            return

    def _read_joint_message(self, source: str):
        if source == "leader":
            return self._arm.get_leader_joint_angles()
        return self._arm.get_joint_angles()

    def _iter_joint_samples(self):
        for source in _JOINT_SOURCES:
            try:
                message = self._read_joint_message(source)
            except Exception:
                logger.debug("%s failed to read %s joints", self, source, exc_info=True)
                continue
            if message is None:
                continue
            timestamp = float(getattr(message, "timestamp", 0.0) or 0.0)
            joints = _message_values(message, expected_len=6)
            if timestamp <= 0 or joints is None:
                continue
            yield source, timestamp, joints

    def _iter_flange_samples(self):
        try:
            message = self._arm.get_flange_pose()
        except Exception:
            logger.debug("%s failed to read flange pose", self, exc_info=True)
            return
        if message is None:
            return
        timestamp = float(getattr(message, "timestamp", 0.0) or 0.0)
        pose = _message_values(message, expected_len=6)
        if timestamp <= 0 or pose is None:
            return
        yield "flange", timestamp, pose

    def _iter_state_samples(self):
        yield from self._iter_joint_samples()
        yield from self._iter_flange_samples()

    def _cache_flange_sample(self, timestamp: float, pose_m_rad: list[float]) -> None:
        self._joint_source = "flange"
        self._last_sdk_timestamps["flange"] = timestamp
        self._last_local_update = time.perf_counter()
        self._last_pose_m_rad = list(pose_m_rad)
        joints = self._arm.get_joint_angles()
        joint_values = _message_values(joints, expected_len=6) if joints is not None else None
        if joint_values is not None:
            self._last_joints_rad = joint_values
        elif self._last_joints_rad is None:
            self._last_joints_rad = [0.0] * 6
        self._refresh_teaching_pendant()

    def _cache_sample(self, source: str, timestamp: float, joints_rad: list[float]) -> None:
        self._joint_source = source
        self._last_sdk_timestamps[source] = timestamp
        self._last_local_update = time.perf_counter()
        self._last_joints_rad = list(joints_rad)
        try:
            pose = self._arm.fk(list(joints_rad))
        except Exception:
            pose = None
        parsed_pose = None
        if isinstance(pose, (list, tuple)) and len(pose) >= 6:
            parsed_pose = [float(value) for value in pose[:6]]
            if not all(math.isfinite(value) for value in parsed_pose):
                parsed_pose = None
        if parsed_pose is not None:
            self._last_pose_m_rad = parsed_pose
        else:
            flange = self._arm.get_flange_pose()
            flange_values = _message_values(flange, expected_len=6) if flange is not None else None
            if flange_values is not None:
                self._last_pose_m_rad = flange_values
        self._refresh_teaching_pendant()

    def _wait_for_first_frame(self) -> None:
        baselines = {source: 0.0 for source in (*_JOINT_SOURCES, "flange")}
        for source, timestamp, _payload in self._iter_state_samples():
            baselines[source] = timestamp

        self._last_local_update = 0.0
        deadline = time.perf_counter() + self.config.startup_timeout_s
        next_config_retry = time.perf_counter() + 5.0
        while time.perf_counter() < deadline:
            for source, timestamp, payload in self._iter_state_samples():
                baseline = baselines.get(source, 0.0)
                if timestamp <= 0:
                    continue
                if baseline == 0.0 or timestamp != baseline:
                    if source == "flange":
                        self._cache_flange_sample(timestamp, payload)
                    else:
                        self._cache_sample(source, timestamp, payload)
                    return
            if self.config.configure_master_mode and time.perf_counter() >= next_config_retry:
                self._arm.set_leader_mode()
                next_config_retry = time.perf_counter() + 5.0
            time.sleep(0.01)

        fps = None
        is_ok = None
        try:
            fps = self._arm.get_fps()
            is_ok = self._arm.is_ok()
        except Exception:
            pass
        raise TimeoutError(
            f"No Piper-X state frame received on {self.config.can_name} within "
            f"{self.config.startup_timeout_s:.1f}s (is_ok={is_ok}, fps={fps}). "
            "This firmware only broadcasts while the arm moves — keep dragging during "
            "the entire connect window. Try `python examples/piper_x/diagnose.py`, "
            "or `--firmware-version default`, or `--judge-flag 0` for PCIe CAN."
        )

    def _refresh_freshness(self) -> None:
        samples = list(self._iter_state_samples())
        preferred = self._joint_source or "feedback"
        source_order = tuple(dict.fromkeys((preferred, *_JOINT_SOURCES, "flange")))
        for source in source_order:
            for sample_source, timestamp, payload in samples:
                if sample_source != source:
                    continue
                if timestamp == self._last_sdk_timestamps.get(sample_source, 0.0):
                    continue
                if sample_source == "flange":
                    self._cache_flange_sample(timestamp, payload)
                else:
                    self._cache_sample(sample_source, timestamp, payload)
                return

    def _require_fresh_or_cached_state(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self._refresh_freshness()
        self._refresh_teaching_pendant()
        if self._last_joints_rad is None or self._last_local_update == 0.0:
            raise RuntimeError(
                f"No Piper-X joint state received yet on {self.config.can_name}. This firmware "
                "only broadcasts while the arm moves — slowly drag the arm by hand once."
            )
        if time.perf_counter() - self._last_local_update > self.config.data_timeout_s:
            # Verified on real hardware: the stream goes quiet a few seconds
            # after the arm stops moving. A quiet stream means the operator is
            # holding the arm still, not a failure — keep serving the last state.
            logger.debug("%s joint-state stream idle (arm held still), serving last state", self)

    def get_joint_angles(self) -> dict[str, float]:
        """Return the current 6 joint angles in degrees."""
        self._require_fresh_or_cached_state()
        return _joints_deg_from_rad(self._last_joints_rad)

    def get_endpose(self) -> dict[str, float]:
        """Return the current flange pose as ``endpose.*`` in millimetres / degrees."""
        self._require_fresh_or_cached_state()
        if self._last_pose_m_rad is None:
            raise RuntimeError("piper_x could not compute a finite endpose from joint angles")
        return _pose_mm_deg_from_m_rad(self._last_pose_m_rad)

    def get_teaching_pendant(self) -> dict[str, float]:
        """Return teaching-pendant opening.

        ``teaching_pendant.pos`` is millimetres in width mode, degrees in angle mode.
        """
        self._require_fresh_or_cached_state()
        if self._last_teaching_pendant is None:
            return {"teaching_pendant.pos": 0.0}
        return dict(self._last_teaching_pendant)

    def get_state(self) -> dict[str, float]:
        """Return joints, endpose, and teaching pendant in one snapshot."""
        return {**self.get_joint_angles(), **self.get_endpose(), **self.get_teaching_pendant()}

    def get_action(self) -> dict[str, float]:
        return {**self.get_endpose(), **self.get_teaching_pendant()}

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        # The Piper-X teaching pendant is input-only in this integration.
        return None

    def _reset_runtime_state(self) -> None:
        self._arm = None
        self._teaching_pendant = None
        self._joint_source = None
        self._last_sdk_timestamps = {}
        self._last_local_update = 0.0
        self._last_joints_rad = None
        self._last_pose_m_rad = None
        self._last_teaching_pendant = None

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        arm = self._arm
        try:
            arm.disconnect()
        finally:
            self._reset_runtime_state()

        logger.info("%s disconnected", self)
