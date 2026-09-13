"""白灯引导下拖动 Franka，记录末端 XYZ 与夹爪开口范围。

用法：
    python3 05_record_workspace.py [robot_ip]

默认 robot_ip 为 172.16.0.2。本脚本只读状态，不发任何运动或夹爪命令。

操作：
1. 停掉 ROS 桥和其它 FCI 客户端（同一时刻只能有一个控制端）。
2. 运行本脚本。
3. 在 Desk 或手臂按钮上打开引导模式，灯变白后再拖动。
4. 把末端拖满希望使用的工作空间；夹爪开到最大、再合到最小。
5. Ctrl+C 结束。把打印出的常量贴进
   examples/piper_x_to_franka/teleoperate.py。
"""

from __future__ import annotations

import sys
import time

import frankx

DEFAULT_IP = "172.16.0.2"
RATE_HZ = 10.0


def _finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


def read_pose(robot: frankx.Robot) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    # 不要用 read_once().O_T_EE：frankx 绑定里它可能是 16 维平铺矩阵。
    pose = robot.current_pose()
    xyz = tuple(float(v) for v in pose.translation())
    quat = tuple(float(v) for v in pose.quaternion())
    if len(xyz) != 3 or not all(_finite(v) for v in xyz):
        raise SystemExit("current_pose() 返回了非有限平移")
    if len(quat) != 4 or not all(_finite(v) for v in quat):
        raise SystemExit("current_pose() 返回了非有限四元数")
    return xyz, quat


def read_gripper_m(gripper: frankx.Gripper) -> float:
    raw = float(gripper.width())
    if not _finite(raw):
        raise SystemExit("夹爪 width() 返回了非有限值")
    return raw / 1000.0 if raw > 1.0 else raw


def update_range(bounds: list[float] | None, value: float) -> list[float]:
    if bounds is None:
        return [value, value]
    bounds[0] = min(bounds[0], value)
    bounds[1] = max(bounds[1], value)
    return bounds


def format_pair(bounds: list[float] | None) -> str:
    if bounds is None:
        return "(未采样)"
    return f"({bounds[0]:.4f}, {bounds[1]:.4f})"


def print_live(
    xyz: tuple[float, float, float],
    quat: tuple[float, float, float, float],
    gripper_m: float | None,
    x_range: list[float] | None,
    y_range: list[float] | None,
    z_range: list[float] | None,
    g_range: list[float] | None,
    robot_mode: object,
) -> None:
    x, y, z = xyz
    gripper_text = "n/a" if gripper_m is None else f"{gripper_m:.4f}"
    print(
        f"mode={robot_mode}  "
        f"x={x:7.4f} {format_pair(x_range)}  "
        f"y={y:7.4f} {format_pair(y_range)}  "
        f"z={z:7.4f} {format_pair(z_range)}  "
        f"g={gripper_text} {format_pair(g_range)}  "
        f"q=[{quat[0]:6.3f} {quat[1]:6.3f} {quat[2]:6.3f} {quat[3]:6.3f}]"
    )


def print_result(
    x_range: list[float] | None,
    y_range: list[float] | None,
    z_range: list[float] | None,
    g_range: list[float] | None,
    last_quat: tuple[float, float, float, float] | None,
) -> None:
    print("\n已停止。把下面常量贴进 examples/piper_x_to_franka/teleoperate.py：\n")
    print(f"FRANKA_X_M = {format_pair(x_range)}")
    print(f"FRANKA_Y_M = {format_pair(y_range)}")
    print(f"FRANKA_Z_M = {format_pair(z_range)}")
    print(f"GRIPPER_M = {format_pair(g_range)}")
    if last_quat is not None:
        print(
            "FRANKA_REF_QUAT_XYZW = "
            f"({last_quat[0]:.6f}, {last_quat[1]:.6f}, {last_quat[2]:.6f}, {last_quat[3]:.6f})"
        )
        print("（四元数是松开 Ctrl+C 那一瞬的末端朝向，用来和 Piper 的对应姿态配对）")
    if any(bounds is None or bounds[0] == bounds[1] for bounds in (x_range, y_range, z_range, g_range)):
        print("\n有轴几乎没有变化。请确认已白灯引导，并沿该轴拖满 / 把夹爪开合到位后再采一次。")


def connect_robot(ip: str) -> frankx.Robot:
    print(f"[连接] 机器人 {ip} ...")
    robot = frankx.Robot(ip)
    robot.set_default_behavior()
    robot.recover_from_errors()
    state = robot.read_once()
    print("已连接（只读，不会发运动）。robot_mode =", getattr(state, "robot_mode", "N/A"))
    return robot


def connect_gripper(ip: str) -> frankx.Gripper | None:
    print(f"[连接] 夹爪 {ip} ...")
    try:
        gripper = frankx.Gripper(ip)
    except Exception as exc:
        print(f"夹爪连接失败，只记录末端范围：{exc}")
        return None
    if getattr(gripper, "has_error", False):
        print("夹爪处于错误状态，只记录末端范围。请先在 Desk 里恢复后再采夹爪。")
        return None
    width_m = read_gripper_m(gripper)
    print(f"已连接夹爪，当前开口 = {width_m:.4f} m（只读，不会发夹爪命令）")
    return gripper


def main() -> None:
    ip = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IP
    period = 1.0 / RATE_HZ

    robot = connect_robot(ip)
    gripper = connect_gripper(ip)

    print(
        f"\n请打开白灯引导，拖动末端走满工作空间，并把夹爪开到最大、合到最小。"
        f"\n实时打印 {RATE_HZ:.0f} Hz，Ctrl+C 结束并输出范围。\n"
    )

    x_range: list[float] | None = None
    y_range: list[float] | None = None
    z_range: list[float] | None = None
    g_range: list[float] | None = None
    last_quat: tuple[float, float, float, float] | None = None

    try:
        while True:
            t0 = time.monotonic()
            xyz, last_quat = read_pose(robot)
            x_range = update_range(x_range, xyz[0])
            y_range = update_range(y_range, xyz[1])
            z_range = update_range(z_range, xyz[2])

            gripper_m = None
            if gripper is not None:
                gripper_m = read_gripper_m(gripper)
                g_range = update_range(g_range, gripper_m)

            state = robot.read_once()
            print_live(
                xyz,
                last_quat,
                gripper_m,
                x_range,
                y_range,
                z_range,
                g_range,
                getattr(state, "robot_mode", "N/A"),
            )
            time.sleep(max(0.0, period - (time.monotonic() - t0)))
    except KeyboardInterrupt:
        print_result(x_range, y_range, z_range, g_range, last_quat)


if __name__ == "__main__":
    main()
