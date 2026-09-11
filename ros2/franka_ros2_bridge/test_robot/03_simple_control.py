"""验证步骤三：运动控制接口验证（关节控制 + 末端位姿控制）。

用法：
    python3 03_simple_control.py [robot_ip] [joint_index] [delta_rad] [cartesian_axis] [cartesian_m]

- robot_ip        默认 172.16.0.2
- joint_index     关节序号（1..7），默认 6
- delta_rad       单关节运动的角位移（弧度），默认 0.2
- cartesian_axis  末端平移的轴（x/y/z），默认 z
- cartesian_m     末端平移的距离（米），默认 0.02

本脚本分两个部分对运动控制接口做验证：

1. 关节空间控制：使用 JointMotion 让单个关节做小幅运动，再回到初始关节位置。
2. 末端位姿控制：使用 LinearMotion/Affine 让末端沿指定轴做小幅平移，再回到初始位姿。

所有运动均使用较低的速度/加速度/加加速度，保证安全。
"""

import sys

import frankx
from pyaffx import Affine

DEFAULT_IP = "172.16.0.2"
JOINT_LOWER = (-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973)
JOINT_UPPER = (2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973)
AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def connect(ip: str) -> frankx.Robot:
    print(f"[连接] 机器人 {ip} ...")
    robot = frankx.Robot(ip)
    # 关键修复：关闭在实时控制线程里调用 PyErr_CheckSignals() 的机制。
    # frankx 为了让 Ctrl+C 打断运动，会在 libfranka 的实时线程（非主线程）中
    # 每个控制周期调用 PyErr_CheckSignals()。该 CPython C-API 在不持有 GIL 的
    # 非主线程里调用会段错误（CPython 3.10 必现）。此处已在真机上复现验证。
    # 代价：阻塞的 move() 期间 Ctrl+C 无法自动打断。
    # 解决办法：所有运动改用 move_and_wait()（后台线程 + 主动 robot.stop()）。
    robot.stop_at_python_signal = False
    robot.set_default_behavior()
    robot.recover_from_errors()

    # 使用较小的动态参数，保证运动缓慢安全
    robot.velocity_rel = 0.05
    robot.acceleration_rel = 0.05
    robot.jerk_rel = 0.05
    print("已连接，动态参数受限（rel = 0.05）")

    # 诊断：打印当前机器人模式与错误状态，便于排查 move 崩溃
    state = robot.read_once()
    robot_mode = getattr(state, "robot_mode", "N/A")
    errors = getattr(state, "errors", None)
    print("诊断：robot_mode =", robot_mode, " errors =", errors)
    return robot


def move_and_wait(robot: frankx.Robot, motion) -> None:
    """在后台线程中执行阻塞的 move()，主线程等待。

    由于停用了 stop_at_python_signal，实时线程里不再检查 Python 信号，
    因此把 move 放到后台线程、主线程用 join 等待，保留 Ctrl+C 中断能力。
    收到中断时主动调用 robot.stop() 停止运动。
    """
    thread = robot.move_async(motion)  # 后台 daemon 线程执行 move
    try:
        while thread.is_alive():
            thread.join(timeout=0.5)
    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，正在停止运动 ...")
        robot.stop()
        thread.join(timeout=2.0)
        raise SystemExit("已手动停止运动。")


def verify_joint_control(robot: frankx.Robot, joint_index: int, delta: float) -> None:
    print("\n===== 第 1 部分：关节空间控制（JointMotion）=====")
    idx = joint_index - 1
    if not 0 <= idx < 7:
        raise SystemExit("joint_index 必须在 1..7 之间")

    state = robot.read_once()
    q_start = list(state.q)
    print("初始关节位置 q =", [f"{v:.4f}" for v in q_start])

    q_target = list(q_start)
    q_target[idx] += delta
    if not (JOINT_LOWER[idx] <= q_target[idx] <= JOINT_UPPER[idx]):
        raise SystemExit(f"目标关节 {joint_index} 超出限位: {q_target[idx]:.4f}")

    print(f"目标：关节 {joint_index} 移动 {delta:+.4f} rad -> {q_target[idx]:.4f}")
    print("执行运动 ...")
    move_and_wait(robot, frankx.JointMotion(q_target))

    state = robot.read_once()
    print("运动后关节位置 q =", [f"{v:.4f}" for v in state.q])

    print("返回初始位置 ...")
    move_and_wait(robot, frankx.JointMotion(q_start))
    state = robot.read_once()
    print("返回后关节位置 q =", [f"{v:.4f}" for v in state.q])
    print("关节控制验证通过。")


def make_pose(x: float, y: float, z: float, q) -> Affine:
    """用平移 + 四元数构造 Affine。

    注意 affx 的构造函数参数顺序是 (x, y, z, q_w, q_x, q_y, q_z)，四元数标量在第一位；
    而 Affine.quaternion() 返回的是 [q_x, q_y, q_z, q_w]。二者顺序不一致，
    直接按 quaternion() 的返回顺序传给构造函数会把四元数搅乱（旋转矩阵不正交），
    libfranka 会报 “Has to be column major!”。
    """
    qw, qx, qy, qz = q[3], q[0], q[1], q[2]
    return Affine(x, y, z, qw, qx, qy, qz)


def verify_cartesian_control(robot: frankx.Robot, axis: str, dist_m: float) -> None:
    print("\n===== 第 2 部分：末端位姿控制（LinearMotion / Affine）=====")
    if axis not in AXIS_INDEX:
        raise SystemExit(f"cartesian_axis 只能是 {list(AXIS_INDEX)} 之一")
    ax = AXIS_INDEX[axis]

    # 注意：不能用 read_once().O_T_EE，它在 frankx 绑定里是 16 维平铺矩阵(list)，
    # 没有 .translation()/.quaternion()。要拿 pyaffx Affine 用 robot.current_pose()。
    pose = robot.current_pose()
    t = list(pose.translation())  # [x, y, z]
    q = list(pose.quaternion())   # [qx, qy, qz, qw]
    print("初始化后末端位姿 x,y,z =", [f"{v:.4f}" for v in t],
          " quat[qw,qx,qy,qz] =", [f"{v:.4f}" for v in [q[3], q[0], q[1], q[2]]])

    start_pose = make_pose(t[0], t[1], t[2], q)
    t_target = list(t)
    t_target[ax] += dist_m
    target_pose = make_pose(t_target[0], t_target[1], t_target[2], q)

    # 诊断：校验重构出的 start_pose 与当前位姿是否完全一致（位置+姿态）
    sq = list(start_pose.quaternion())
    diff = max(abs(a - b) for a, b in zip([t[0], t[1], t[2]] + q, [start_pose.translation()[0], start_pose.translation()[1], start_pose.translation()[2]] + sq))
    print("诊断：start_pose 与当前位姿最大元素差 =", f"{diff:.2e}")
    print("诊断：start_pose quat[qw,qx,qy,qz] =",
          [f"{v:.4f}" for v in [sq[3], sq[0], sq[1], sq[2]]])

    print(f"目标：末端沿 {axis} 轴平移 {dist_m:+.4f} m -> x,y,z = "
          f"[{t_target[0]:.4f}, {t_target[1]:.4f}, {t_target[2]:.4f}]")
    print("执行运动 ...")
    move_and_wait(robot, frankx.LinearMotion(target_pose))

    print("运动后末端位移 x,y,z =", [f"{v:.4f}" for v in robot.current_pose().translation()])

    print("返回初始位姿 ...")
    move_and_wait(robot, frankx.LinearMotion(start_pose))
    print("返回后末端位移 x,y,z =", [f"{v:.4f}" for v in robot.current_pose().translation()])
    print("末端位姿控制验证通过。")


def main() -> None:
    ip = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IP
    joint_index = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    delta = float(sys.argv[3]) if len(sys.argv) > 3 else 0.2
    axis = sys.argv[4] if len(sys.argv) > 4 else "z"
    dist_m = float(sys.argv[5]) if len(sys.argv) > 5 else 0.02

    robot = connect(ip)
    verify_joint_control(robot, joint_index, delta)
    verify_cartesian_control(robot, axis, dist_m)

    print("\n全部完成。验证通过。")


if __name__ == "__main__":
    main()