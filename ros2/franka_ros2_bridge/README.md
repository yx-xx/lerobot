# franka_ros2_bridge

面向 Ubuntu 22.04 / ROS 2 Humble 的 Franka 桥接包。在直连机器人的控制机上运行，
通过 4 个 ROS 2 话题对外提供关节/末端状态，并接收运动命令。

## 每次启动

每个新终端都要先执行下面两行，再启动节点。不要写进 `~/.bashrc`，以免污染其它终端。

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=23
```

工作空间 overlay（包已经 `colcon build` 过之后）：

```bash
source ~/franka_ws/install/setup.bash
```

启动桥接节点（默认读取包内 `config/franka_bridge.yaml`）：

```bash
ros2 launch franka_ros2_bridge franka_bridge.launch.py
```

使用自定义配置：

```bash
ros2 launch franka_ros2_bridge franka_bridge.launch.py \
  config_file:=/absolute/path/to/franka_bridge.yaml
```

确认状态话题在发：

```bash
ros2 topic hz /franka/joint_state
ros2 topic echo --once /franka/end_pose
```

同一 ROS 网络里的其它机器（例如跑 LeRobot 的电脑）也必须 `source /opt/ros/humble/setup.bash`，
并且 `ROS_DOMAIN_ID` 与这里相同。

## 首次构建

先安装 ROS 2 Humble，并在运行节点的同一 Python 环境中安装可导入的 `frankx` 与 `pyaffx`。
本包不会自动安装它们。

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=23

mkdir -p ~/franka_ws/src
# 将本目录放入工作空间，例如：
# ln -s /path/to/franka_ros2_bridge ~/franka_ws/src/franka_ros2_bridge
cd ~/franka_ws
colcon build --symlink-install --packages-select franka_ros2_bridge
source install/setup.bash
```

按现场网络、速度限制和安全空间编辑 `config/franka_bridge.yaml`。

## 接口

- 发布 `/franka/joint_state` (`sensor_msgs/JointState`)
- 发布 `/franka/end_pose` (`geometry_msgs/PoseStamped`)
- 订阅 `/franka/joint_cmd` (`sensor_msgs/JointState`)
- 订阅 `/franka/end_pose_cmd` (`geometry_msgs/PoseStamped`)

关节命令与状态使用同一消息类型：`name` 必须严格为 `panda_joint1` 到 `panda_joint7`，
`position` 为目标关节角（弧度），且位于配置的关节限位内。末端命令的 `frame_id` 必须为空
或等于 `base_frame`，四元数为 XYZW、有限且归一化，位置位于配置的 workspace 内。
回调只替换“最新命令”；工作线程串行执行，并丢弃超过 `command_timeout_sec` 的命令。
运动用 `move_async` 在后台执行，状态发布在运动期间仍可继续。

所有参数均位于 YAML 中，包括 IP、四个 topic、`base_frame`、发布频率、相对速度/
加速度/加加速度、命令超时、7 轴关节限位和 XYZ workspace 边界。
末端位姿单位为米 + XYZW 四元数。

## 安全启动

1. 确认机器人急停、碰撞保护和 Franka Desk 安全配置可用。
2. 首次测试移除工具与负载，清空工作区，保持人员在机器人运动范围外。
3. 确认控制机与机器人网络连通，`robot_ip` 正确且没有其他控制客户端占用机器人。
4. 从较小的 `velocity_rel`、`acceleration_rel`、`jerk_rel` 和收紧的 workspace 开始。
5. 先只启动节点并检查状态话题，再发送单个小幅命令；不要把本桥接节点当作安全控制器。

## 模块结构

控制层按 `test_robot/` 中已在真机验证的 frankx 脚本实现：`stop_at_python_signal=False`、
`move_async` 等待、`JointMotion([q1..q7])`、`LinearMotion(Affine(x, y, z, qw, qx, qy, qz))`，
末端位姿用 `robot.current_pose()` 读取。

```text
franka_ros2_bridge/
  bridge_node.py   # ROS 节点接线（薄）
  ros/             # 消息收发与转换
  core/            # 公共类型、安全校验、命令队列
  control/         # 机器人控制后端（当前为 frankx）
```

| 目录 | 职责 | 后续扩展时改哪里 |
|------|------|------------------|
| `ros/` | Topic 收发、ROS 消息 ↔ 内部类型 | 换消息格式 / Topic |
| `core/` | 状态/命令结构、限位检查、最新命令队列 | 改安全策略 / 排队规则 |
| `control/` | `read_state` / `move_joint` / `move_end_pose` | 加夹爪、阻抗、其它后端 |

## 测试

纯函数与队列逻辑不依赖 ROS 2 / frankx，可直接测试：

```bash
cd /path/to/franka_ros2_bridge
PYTHONPATH=. python3 -m pytest -q test/test_bridge_utils.py
python3 -m py_compile franka_ros2_bridge/*.py franka_ros2_bridge/*/*.py launch/*.py setup.py
```
