# franka_ros2_bridge

面向 Ubuntu 22.04 / ROS 2 Humble 的 Franka 桥接包。在直连机器人的控制机上运行，
通过 ROS 2 话题对外提供关节/末端状态，并接收运动命令。

默认笛卡尔模式是 **libfranka 1 kHz 流式位姿环**：控制线程不退出，ROS 只更新最新目标。
夹爪仍用 frankx。点到点 `LinearMotion` 仅在 `cartesian_mode: ptp` 时启用。

## 每次启动

每个新终端都要先执行下面两行，再启动节点。不要写进 `~/.bashrc`。

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=23
source ~/franka_ws/install/setup.bash
ros2 launch franka_ros2_bridge franka_bridge.launch.py
```

确认状态话题在发：

```bash
ros2 topic hz /franka/joint_state
ros2 topic echo --once /franka/end_pose
```

同一 ROS 网络里的其它机器也必须 source Humble，并且 `ROS_DOMAIN_ID` 相同。

## 首次构建

控制机需要：ROS 2 Humble、可链接的 **libfranka**、**pybind11**、以及可导入的 **frankx**（夹爪）。

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=23
python3 -m pip install pybind11

mkdir -p ~/franka_ws/src
# ln -s /path/to/franka_ros2_bridge ~/franka_ws/src/franka_ros2_bridge
cd ~/franka_ws
colcon build --symlink-install --packages-select franka_ros2_bridge
source install/setup.bash
```

没有 libfranka 的机器不要编原生扩展：

```bash
FRANKA_SKIP_STREAM_EXT=1 colcon build --symlink-install --packages-select franka_ros2_bridge
```

启动后日志应出现 `Cartesian pose stream is running at 1 kHz`。
若提示 `cartesian_stream native module is missing`，说明扩展没编上，检查 pybind11、`-lfranka` 和 colcon 编译输出。

按现场网络和安全空间编辑 `config/franka_bridge.yaml`。
流式限速是 `max_linear_velocity`（m/s）和 `max_angular_velocity`（rad/s）。

## 接口

- 发布 `/franka/joint_state` (`sensor_msgs/JointState`)
- 发布 `/franka/end_pose` (`geometry_msgs/PoseStamped`)
- 发布 `/franka/gripper_state` (`sensor_msgs/JointState`，`name=panda_finger`，开口单位米)
- 订阅 `/franka/joint_cmd` (`sensor_msgs/JointState`)，流式模式下忽略
- 订阅 `/franka/end_pose_cmd` (`geometry_msgs/PoseStamped`)
- 订阅 `/franka/gripper_cmd` (`sensor_msgs/JointState`，`name=panda_finger`，开口单位米)

末端命令的 `frame_id` 必须为空或等于 `base_frame`，四元数为 XYZW，位置在 workspace 内。
流式模式只替换最新笛卡尔目标；1 kHz 回调里按速度上限插值，避免位姿跳变。
状态从同一控制回调读取，运动期间仍然新鲜。

## 安全启动

1. 确认急停、碰撞保护和 Franka Desk 安全配置可用。
2. 首次测试清空工作区，人员在运动范围外。
3. 确认 `robot_ip` 正确，没有其它 FCI 客户端占用机器人。
4. 从较小的 `max_linear_velocity` / `max_angular_velocity` 和收紧的 workspace 开始。
5. 不要把本桥接节点当作安全控制器。
