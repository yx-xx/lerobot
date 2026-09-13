"""把 Franka 复位到官方准备姿态（关节空间）。

用法：
    python3 06_reset_home.py [robot_ip]

默认 robot_ip 为 172.16.0.2。目标关节角为：

    0, -π/4, 0, -3π/4, 0, π/2, π/4

本脚本会发 JointMotion，请先清空工作区、确认急停可用，并停掉其它 FCI 客户端。
速度限制与 03_simple_control.py 相同（rel = 0.05）。Ctrl+C 会 robot.stop()。
"""

from __future__ import annotations

import math
import sys

import frankx

DEFAULT_IP = "172.16.0.2"
HOME_Q = (
    0.0,
    -math.pi / 4.0,
    0.0,
    -3.0 * math.pi / 4.0,
    0.0,
    math.pi / 2.0,
    math.pi / 4.0,
)
JOINT_LOWER = (-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973)
JOINT_UPPER = (2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973)


def connect(ip: str) -> frankx.Robot:
    print(f"[连接] 机器人 {ip} ...")
    robot = frankx.Robot(ip)
    # 见 03_simple_control.py：实时线程里 PyErr_CheckSignals() 会段错误。
    robot.stop_at_python_signal = False
    robot.set_default_behavior()
    robot.recover_from_errors()
    robot.velocity_rel = 0.05
    robot.acceleration_rel = 0.05
    robot.jerk_rel = 0.05
    state = robot.read_once()
    print("已连接，动态参数受限（rel = 0.05）")
    print("诊断：robot_mode =", getattr(state, "robot_mode", "N/A"), " errors =", getattr(state, "errors", None))
    return robot


def move_and_wait(robot: frankx.Robot, motion) -> None:
    thread = robot.move_async(motion)
    try:
        while thread.is_alive():
            thread.join(timeout=0.5)
    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，正在停止运动 ...")
        robot.stop()
        thread.join(timeout=2.0)
        raise SystemExit("已手动停止运动。")


def format_q(values) -> str:
    return "[" + " ".join(f"{v:8.4f}" for v in values) + "]"


def reset_home(robot: frankx.Robot) -> None:
    for index, (q, lo, hi) in enumerate(zip(HOME_Q, JOINT_LOWER, JOINT_UPPER, strict=True), start=1):
        if not lo <= q <= hi:
            raise SystemExit(f"复位目标关节 {index} = {q:.4f} 超出限位 [{lo}, {hi}]")

    state = robot.read_once()
    print("当前关节 q =", format_q(state.q))
    print("复位目标 q =", format_q(HOME_Q))
    print("执行 JointMotion ...")
    move_and_wait(robot, frankx.JointMotion(list(HOME_Q)))

    after = list(robot.read_once().q)
    print("到位后   q =", format_q(after))
    err = max(abs(a - b) for a, b in zip(after, HOME_Q, strict=True))
    print(f"与目标最大关节误差 = {err:.4f} rad")
    if err > 0.05:
        raise SystemExit("复位后关节误差过大，请检查碰撞、限位或 Desk 状态。")
    print("复位完成。")


def main() -> None:
    ip = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IP
    robot = connect(ip)
    reset_home(robot)


if __name__ == "__main__":
    main()
