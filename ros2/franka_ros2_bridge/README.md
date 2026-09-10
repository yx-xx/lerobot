# franka_ros2_bridge

面向 Ubuntu 22.04 / ROS 2 Humble 的通用 `ament_python` Franka 桥接包。
任意 ROS 2 客户端（不限于 LeRobot）都可订阅/发布本包定义的话题。

## 模块结构

```text
franka_ros2_bridge/
  bridge_node.py   # ROS 节点接线（薄）
  ros/             # 消息收发与转换
  core/          # 公共类型、安全校验、命令队列
  control/         # 机器人控制后端（当前为 frankx）
```

| 目录 | 职责 | 后续扩展时改哪里 |
|------|------|------------------|
| `ros/` | Topic 收发、ROS 消息 ↔ 内部类型 | 换消息格式 / Topic |
| `core/` | 状态/命令结构、限位检查、最新命令队列 | 改安全策略 / 排队规则 |
| `control/` | `read_state` / `move_joint` / `move_pose` | 加夹爪、阻抗、其它后端 |

## 接口

- 发布 `/franka/joint_state` (`sensor_msgs/JointState`)
- 发布 `/franka/end_pose` (`geometry_msgs/PoseStamped`)
- 订阅 `/franka/joint_cmd` (`trajectory_msgs/JointTrajectory`)
- 订阅 `/franka/end_pose_cmd` (`geometry_msgs/PoseStamped`)

关节命令必须包含且仅包含一个轨迹点，关节名必须严格按
`panda_joint1` 到 `panda_joint7` 排列，且目标位于配置的关节限位内。笛卡尔命令的
`frame_id` 必须为空或等于 `base_frame`，并包含有限、非零四元数，且位置位于配置的
workspace 内。回调只替换“最新命令”；工作线程串行执行，并丢弃超过
`command_timeout_sec` 的命令。读取与运动共用控制层硬件锁。

## 环境与构建

先安装 ROS 2 Humble，并在实际运行节点的同一 Python 环境中安装可导入的
`frankx`。本包不会自动安装 `frankx`。

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=23  # 同一 ROS 网络中的进程必须一致

mkdir -p ~/franka_ws/src
# 将本目录放入工作空间，例如：
# ln -s /path/to/franka_ros2_bridge ~/franka_ws/src/franka_ros2_bridge
cd ~/franka_ws
colcon build --symlink-install --packages-select franka_ros2_bridge
source install/setup.bash
```

按现场网络、速度限制和安全空间编辑 `config/franka_bridge.yaml`，或复制后通过
launch 参数传入：

```bash
ros2 launch franka_ros2_bridge franka_bridge.launch.py \
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

纯函数与队列逻辑不依赖 ROS 2 / frankx，可直接测试：

```bash
cd /path/to/franka_ros2_bridge
PYTHONPATH=. python3 -m pytest -q test/test_bridge_utils.py
python3 -m py_compile franka_ros2_bridge/*.py franka_ros2_bridge/*/*.py launch/*.py setup.py
```
