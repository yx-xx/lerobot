"""Compile and exercise the production C++ planner without a robot or ROS."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.fixture(scope="session")
def tracking_binary(tmp_path_factory):
    if os.environ.get("FRANKA_SKIP_STREAM_EXT") == "1":
        pytest.skip("native checks explicitly disabled by FRANKA_SKIP_STREAM_EXT=1")
    package = Path(__file__).resolve().parents[1]
    franka_root = Path(os.environ.get("FRANKA_ROOT", "/home/rt/franka/libfranka"))
    ruckig = package / "third_party" / "ruckig"
    compiler = shutil.which("c++")
    assert compiler is not None, "C++ compiler is required for native controller regression tests"
    binary = tmp_path_factory.mktemp("cartesian_tracking") / "tracking_test"
    result = subprocess.run(
        [compiler, "-O2", "-std=c++17", str(package / "test" / "cartesian_tracking_test.cpp"),
         *map(str, sorted((ruckig / "src" / "ruckig").glob("*.cpp"))),
         f"-I{package / 'src'}", f"-I{ruckig / 'include'}", f"-I{franka_root / 'include'}",
         f"-I{os.environ.get('EIGEN3_INCLUDE_DIR', '/usr/include/eigen3')}",
         f"-L{franka_root / 'lib'}", f"-Wl,-rpath,{franka_root / 'lib'}", "-lfranka", "-o", str(binary)],
        capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return binary


@pytest.mark.parametrize("scenario", [
    "fixed", "startup", "reversal", "saturation", "moving", "drops", "rotation", "noise", "circle",
    "changing_limits", "feedback_delay", "stress",
    "reported_posture", "zero_deadband", "joint_moving",
])
def test_native_absolute_tracking(tracking_binary, scenario):
    result = subprocess.run([str(tracking_binary), scenario], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
    print(result.stdout, end="")
