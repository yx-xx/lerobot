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

末端位置是两套长方体的线性映射；姿态用一对对应姿态消掉两边法兰零位和轴定义的差别。都在 `examples/piper_x_to_franka/teleoperate.py` 里填。
