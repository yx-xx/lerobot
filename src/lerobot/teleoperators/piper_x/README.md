# Piper-X

AgileX Piper-X 作为 LeRobot 遥操作主臂。驱动来自 `third_party/pyAgxArm`。

## 准备

```bash
pip install -e ".[piper]"
sudo ip link set can0 up type can bitrate 1000000
```

若提示 `Device or resource busy`，说明 `can0` 已在运行。`ip -details link show can0` 应显示 `state UP`。

固件只在机械臂运动时广播。连接阶段请持续拖臂。

## 读关节角、末端位姿和示教器

```bash
python examples/piper_x/read.py
```

## 作为遥操作器

通过工厂创建 `PiperXTeleoperator`，循环调用 `get_action()` 并打印，不需要从臂。

```bash
python examples/piper_x/teleoperate.py
```

## 遥操作远程 Franka

控制机先跑 `ros2/franka_ros2_bridge`。本机 source ROS 2 Humble、设置相同的 `ROS_DOMAIN_ID`，再：

```bash
python examples/piper_x_to_franka/teleoperate.py
```

末端位置是两套长方体的线性映射；姿态把 Piper 相对参考姿态的基座系旋转增量作用到 Franka 参考姿态。
两臂基座的 XYZ 轴按位置映射视为同向，末端工具轴的差异由一对对应姿态补偿：
`R_franka = R_piper @ R_piper_ref.inv() @ R_franka_ref`。
补偿放在右侧，避免把工具轴的偏差变成左右、前后倾斜方向的偏差。
参数都在 `examples/piper_x_to_franka/teleoperate.py` 里填。

### 一次性姿态标定

当前已填入实测对应姿态，`LOCK_FRANKA_STARTUP_ORIENTATION = False`，启用姿态遥操作。
设为 `True` 时，Franka 保持启动时的朝向。重新标定时，停掉遥操作程序，保留 Franka 桥接器，
将 Franka 预先摆到参考姿态，运行：

```bash
python examples/piper_x_to_franka/calibrate_orientation.py
```

连接时持续拖动 Piper；连接后把 Piper 摆到希望与 Franka 对应的参考姿态，两臂静止时按回车。
脚本只读取 Franka 状态，打印 `PIPER_REF_RPY_DEG` 和 `FRANKA_REF_QUAT_XYZW`。
将两行一起替换到 `teleoperate.py`，再将 `LOCK_FRANKA_STARTUP_ORIENTATION` 改为 `False`。
机械安装和末端工具不变时，这组常量可以一直复用。

Franka 四元数统一使用 `(qx, qy, qz, qw)`：ROS 按 `orientation.x/y/z/w` 读取，
`Affine.quaternion()` 也输出 XYZW；只有 `Affine(x, y, z, qw, qx, qy, qz)` 构造时需要 WXYZ。
桥接层已处理构造顺序，标定常量不要再换序。
