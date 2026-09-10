# lerobot_franka_bridge

面向 Ubuntu 22.04 / ROS 2 Humble 的简洁 `ament_python` Franka 桥接包。节点通过
`frankx` 发布机器人状态，并在单一工作线程内串行执行最新的有效运动命令。

## 接口

- 发布 `/franka/joint_state` (`sensor_msgs/JointState`)
- 发布 `/franka/end_pose` (`geometry_msgs/PoseStamped`)
- 订阅 `/franka/joint_cmd` (`trajectory_msgs/JointTrajectory`)
- 订阅 `/franka/end_pose_cmd` (`geometry_msgs/PoseStamped`)

关节命令必须包含且仅包含一个轨迹点，关节名必须严格按
`panda_joint1` 到 `panda_joint7` 排列，且目标位于配置的关节限位内。笛卡尔命令的
`frame_id` 必须为空或等于 `base_frame`，并包含有限、非零四元数，且位置位于配置的
workspace 内。回调只替换“最新命令”；工作线程串行执行，并丢弃超过
`command_timeout_sec` 的命令。读取与运动共用硬件互斥锁。

## 环境与构建

先安装 ROS 2 Humble，并在实际运行节点的同一 Python 环境中安装可导入的
`frankx`。本包不会自动安装 `frankx`。

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=23  # 同一 ROS 网络中的进程必须一致

mkdir -p ~/franka_ws/src
ln -s /home/pc108/project/lerobot/ros2/lerobot_franka_bridge \
  ~/franka_ws/src/lerobot_franka_bridge
cd ~/franka_ws
colcon build --symlink-install --packages-select lerobot_franka_bridge
source install/setup.bash
```

按现场网络、速度限制和安全空间编辑 `config/franka_bridge.yaml`，或复制后通过
launch 参数传入：

```bash
ros2 launch lerobot_franka_bridge franka_bridge.launch.py \
  config_file:=/absolute/path/to/franka_bridge.yaml
```

## 安全启动

1. 确认机器人急停、碰撞保护和 Franka Desk 安全配置可用。
2. 首次测试移除工具与负载，清空工作区，保持人员在机器人运动范围外。
3. 确认控制机与机器人网络连通，`robot_ip` 正确且没有其他控制客户端占用机器人。
4. 从较小的 `velocity_rel`、`acceleration_rel` 和收紧的 workspace 开始。
5. 先只启动节点并检查状态话题，再发送单个小幅命令；不要把本桥接节点当作安全控制器。

所有参数均位于 YAML 中，包括 IP、四个 topic、`base_frame`、发布频率、相对速度/
加速度、命令超时、7 轴关节限位和 XYZ workspace 边界。矩阵使用 libfranka 风格的
4×4 列优先排列。

## 测试

纯数学和关节校验函数不依赖 ROS 2，可直接测试：

```bash
cd /home/pc108/project/lerobot/ros2/lerobot_franka_bridge
python3 -m pytest -q test/test_bridge_utils.py
python3 -m py_compile lerobot_franka_bridge/*.py launch/*.py setup.py
```
