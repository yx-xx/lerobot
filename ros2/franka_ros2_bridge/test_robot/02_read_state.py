"""验证步骤二：持续读取并打印机器人状态，按 Ctrl+C 退出。

用法：
    python3 02_read_state.py [robot_ip]

默认 robot_ip 为 172.16.0.2。验证状态流是否稳定可读。
"""

import sys
import time

import frankx

DEFAULT_IP = "172.16.0.2"
RATE_HZ = 5.0


def main() -> None:
    ip = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IP
    period = 1.0 / RATE_HZ

    robot = frankx.Robot(ip)
    robot.set_default_behavior()
    robot.recover_from_errors()

    print(f"开始读取状态（{RATE_HZ} Hz），按 Ctrl+C 退出 ...")
    try:
        while True:
            t0 = time.monotonic()
            state = robot.read_once()
            q = state.q
            trans = state.O_T_EE.translation()
            print(
                f"q=[{' '.join(f'{v:7.4f}' for v in q)}] "
                f"x,y,z=[{' '.join(f'{v:6.4f}' for v in trans)}]"
            )
            # 校正一次读取耗时，尽量按固定频率打印
            time.sleep(max(0.0, period - (time.monotonic() - t0)))
    except KeyboardInterrupt:
        print("\n已停止。")


if __name__ == "__main__":
    main()