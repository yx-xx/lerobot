"""验证步骤一：连接 Franka 机器人并读取一次状态。

用法：
    python3 01_connect.py [robot_ip]

默认 robot_ip 为 172.16.0.2。连接成功后设置默认行为、尝试从错误恢复，
再读取一次状态来确认链路有效。
"""

import sys

import frankx

DEFAULT_IP = "172.16.0.2"


def main() -> None:
    ip = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IP

    print(f"[1/3] 连接机器人 {ip} ...")
    robot = frankx.Robot(ip)

    print("[2/3] 设置默认行为并尝试从错误恢复 ...")
    robot.set_default_behavior()
    robot.recover_from_errors()

    print("[3/3] 读取一次状态以确认连接有效 ...")
    state = robot.read_once()

    print("连接成功，机器人模式 robot_mode =", state.robot_mode)
    print("关节位置 q       =", [f"{v:.4f}" for v in state.q])
    print("末端平移 x,y,z   =", [f"{v:.4f}" for v in state.O_T_EE.translation()])
    print("末端四元数       =", [f"{v:.4f}" for v in state.O_T_EE.quaternion()])
    print("OK")


if __name__ == "__main__":
    main()