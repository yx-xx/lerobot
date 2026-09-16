from types import SimpleNamespace
from unittest.mock import MagicMock

from lerobot.teleoperators.piper_x.piper_x import PiperXTeleoperator


def test_refresh_continues_with_flange_source() -> None:
    flange_pose = [0.3, 0.0, 0.4, 0.0, 0.0, 0.0]
    teleop = SimpleNamespace(
        _joint_source="flange",
        _last_sdk_timestamps={"flange": 1.0},
        _iter_state_samples=lambda: iter([("flange", 2.0, flange_pose)]),
        _cache_flange_sample=MagicMock(),
        _cache_sample=MagicMock(),
    )

    PiperXTeleoperator._refresh_freshness(teleop)

    teleop._cache_flange_sample.assert_called_once_with(2.0, flange_pose)
    teleop._cache_sample.assert_not_called()


def test_refresh_can_switch_from_flange_to_leader_source() -> None:
    leader_joints = [0.0] * 6
    teleop = SimpleNamespace(
        _joint_source="flange",
        _last_sdk_timestamps={"flange": 1.0},
        _iter_state_samples=lambda: iter([("leader", 2.0, leader_joints)]),
        _cache_flange_sample=MagicMock(),
        _cache_sample=MagicMock(),
    )

    PiperXTeleoperator._refresh_freshness(teleop)

    teleop._cache_sample.assert_called_once_with("leader", 2.0, leader_joints)
    teleop._cache_flange_sample.assert_not_called()
