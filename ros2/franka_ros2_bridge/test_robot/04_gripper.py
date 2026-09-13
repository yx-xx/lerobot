"""验证步骤四：夹爪开合控制。

用法：
    python3 04_gripper.py [robot_ip] [delta_m]

- robot_ip  默认 172.16.0.2
- delta_m   相对当前开口的变化量（米），默认 0.01

先读当前开口，再沿张开方向小幅运动（到限位则改为闭合），读一次状态，
最后回到初始开口。速度较慢。夹爪里不要夹东西。

本脚本直连 frankx.Gripper，不经过 ROS 2 桥。确认夹爪硬件可用后再开桥。
"""

from __future__ import annotations

import sys
import time

import frankx

DEFAULT_IP = "172.16.0.2"
DEFAULT_DELTA_M = 0.01
GRIPPER_MIN_M = 0.002
GRIPPER_MAX_M = 0.08
GRIPPER_SPEED_MPS = 0.02


def connect_gripper(ip: str) -> frankx.Gripper:
    print(f"[连接] 夹爪 {ip} ...")
    gripper = frankx.Gripper(ip)
    gripper.gripper_speed = GRIPPER_SPEED_MPS
    if getattr(gripper, "has_error", False):
        raise SystemExit("夹爪处于错误状态，请先在 Desk 里恢复后再试。")
    print(f"已连接，gripper_speed = {GRIPPER_SPEED_MPS:.3f} m/s")
    return gripper


def read_width_m(gripper: frankx.Gripper) -> tuple[float, bool]:
    """返回 (开口米, width() 是否以毫米返回)。"""
    raw = float(gripper.width())
    if not _finite(raw):
        raise SystemExit("夹爪 width() 返回了非有限值")
    if raw > 1.0:
        return raw / 1000.0, True
    return raw, False


def _finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


def clip_width(width_m: float) -> float:
    return min(max(width_m, GRIPPER_MIN_M), GRIPPER_MAX_M)


def move_and_wait(gripper: frankx.Gripper, width_m: float, width_is_mm: bool) -> None:
    command = width_m * 1000.0 if width_is_mm else width_m
    move_async = getattr(gripper, "moveAsync", None)
    if move_async is None:
        ok = gripper.move(command)
        if ok is False:
            raise SystemExit("夹爪 move() 返回 False")
        return

    future = move_async(command)
    try:
        while not future.done():
            time.sleep(0.1)
        ok = future.result()
    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，等待当前夹爪命令结束 ...")
        try:
            future.result(timeout=2.0)
        except Exception:
            pass
        raise SystemExit("已手动停止。")
    if ok is False:
        raise SystemExit("夹爪 moveAsync() 返回 False")


def format_width(width_m: float) -> str:
    return f"{width_m:.4f} m ({width_m * 1000.0:.1f} mm)"


def verify_gripper(gripper: frankx.Gripper, delta_m: float) -> None:
    if delta_m <= 0.0:
        raise SystemExit("delta_m 必须为正")

    start_m, width_is_mm = read_width_m(gripper)
    unit = "mm" if width_is_mm else "m"
    print(f"当前开口 = {format_width(start_m)}（width() 按 {unit} 解释）")

    opened = clip_width(start_m + delta_m)
    closed = clip_width(start_m - delta_m)
    if abs(opened - start_m) >= abs(closed - start_m):
        target_m = opened
        direction = "张开"
    else:
        target_m = closed
        direction = "闭合"
    if abs(target_m - start_m) < 1e-4:
        raise SystemExit("当前开口已靠近限位，换一个更大的 delta_m 或先手动打开夹爪。")

    print(f"目标：{direction}到 {format_width(target_m)}")
    print("执行运动 ...")
    move_and_wait(gripper, target_m, width_is_mm)
    after_m, _ = read_width_m(gripper)
    print(f"运动后开口 = {format_width(after_m)}")

    print("返回初始开口 ...")
    move_and_wait(gripper, start_m, width_is_mm)
    back_m, _ = read_width_m(gripper)
    print(f"返回后开口 = {format_width(back_m)}")
    if abs(after_m - start_m) < 1e-3:
        raise SystemExit("夹爪开口几乎没有变化，控制可能失败。")
    print("夹爪控制验证通过。")


def main() -> None:
    ip = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IP
    delta_m = float(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_DELTA_M

    gripper = connect_gripper(ip)
    verify_gripper(gripper, delta_m)
    print("\n全部完成。验证通过。")


if __name__ == "__main__":
    main()
