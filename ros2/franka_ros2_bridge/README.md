# franka_ros2_bridge

面向 Ubuntu 22.04 / ROS 2 Humble 的 Franka 桥接包。在直连机器人的控制机上运行，
通过 ROS 2 话题对外提供关节/末端状态，并接收运动命令。

默认笛卡尔模式是 **libfranka 1 kHz resolved-rate 流式控制环**：ROS 只更新最新的绝对末端
目标，控制线程用在线轨迹生成器计算包含制动的连续速度，再用 Jacobian 转换成关节速度，
并在发送前限制关节速度、加速度和 jerk。这样不再依赖 Franka 笛卡尔 motion generator 的内部 IK。
流式模式的夹爪通过释放 Python GIL 的 libfranka 接口独立轮询；frankx 和 `LinearMotion`
仅在 `cartesian_mode: ptp` 时使用。

流式模式中，示教器目标开口减小时使用 `grasp()` 带力抓取，`gripper_force` 默认 20 N；
目标开口增大时按需 `stop()` 解除抓取保持，再 `move()` 张开。开合意图按前后两次目标开口判断，
因此物体挡住夹爪、实际开口大于目标值时，仍能正确识别释放操作。
抓取宽度容差允许在目标开口以上接触物体，适用于遥操作时物体宽度未知的情况。
夹爪开合速度 `gripper_speed` 默认 `0.10 m/s`。同方向连续目标只保留最新开口；
与正在执行的目标相差至少 `gripper_retarget_deadband=0.002 m` 时，运行满
`gripper_retarget_interval_sec=0.10 s` 即请求打断并更新，不再无限等待旧阻塞调用完成。
小变化等当前动作完成后接续，避免每帧目标都触发停止和重启。反向开合立即请求中断，
待旧调用退出后执行最新目标；已经打断的动作不再额外重复 `stop()`。
此处 100 ms 是同向更新的调度间隔，不是包含 SDK 停止确认、网络和机械动作的延迟保证。
开合方向在 ROS 回调将目标写入队列时记录，再与最新目标一起交给工作线程。
例如工作线程执行过 30 mm，中间收到合到 5 mm 再张到 20 mm，
即使 5 mm 被合并掉，20 mm 仍保留张开意图，不会被误判为继续抓紧。
重复目标和小幅噪声由死区过滤，慢速输入的变化会相对上次有效开口累计。
`gripper_poll_rate_hz` 默认 20 Hz，降低显示和数据采集中的状态滞后；它不限制运动指令速度。
`gripper_dispatch_ms` 是最新命令从 ROS 接收到开始执行的等待时间，
`gripper_motion_ms` 是最近完成的 SDK 调用耗时（包含等待或被打断），二者均不是实测机械起动延迟。
`gripper_pending_ms` 是当前待执行请求在工作线程中累计等待的时间；
`gripper_request_wait_ms` 是最近启动的请求累计等候的时间。同方向更新和重复目标不会将此计时清零。
`gripper_stop_ms` 是最近一次中断旧动作时等待 `stop()` 返回的耗时。
若仍然释放迟缓，可对比这三项，区分排队、停止确认与 SDK 动作耗时。
启动时应看到 `Gripper queue preserves open/close intent before coalescing targets`，
用于确认控制机已加载这次修复；只重启主臂脚本不会更新控制机上的队列。

夹爪修复同时修改了 Python 和 C++ 原生接口。更新时将完整包同步到 Franka 控制机，
按下文步骤重新 `colcon build`，再重启桥接器。若启动报 `GripperIO is outdated`，说明仍在
加载旧扩展。`CartesianStreamer is outdated` 则表示原生扩展尚不包含位置边界制动。
仅重启主臂遥操作脚本不足以应用此次更新。

## 固定朝下的位置遥操作

`examples/piper_x_to_franka/teleoperate.py` 默认使用
`FRANKA_FIXED_QUAT_XYZW = FRANKA_REF_QUAT_XYZW`，保持标定时实际摆好的 Franka 朝下姿态，
四元数顺序为 `(qx, qy, qz, qw)`，包括标定时夹爪绕竖直方向的转角。
更新标定常量会同步更新固定姿态目标。启动时的任意姿态和主臂的转动均不会改变该目标。
控制器仍按旋转动态限制平滑到达目标姿态。
将此常量设为 `None` 才恢复原有姿态标定和跟随；届时也应将桥接配置中的
`position_boundary_guard` 设为 `false`，单独验证完整姿态的工作空间。
位置映射保留 X `[0.35, 0.70]`、Y `[-0.35, 0.35]`、Z `[0.17, 0.60]` m，
每轴增加 1.5 mm 的目标死区；小幅变化相对保持的目标累计，不会让慢速移动一直被滤掉。
夹爪映射上限对齐实际接口的 80 mm；旧的 84.6 mm 映射会被客户端截断，
使示教器从全开开始合拢时出现一段不改变机器人目标的空行程。
默认关闭逐帧可视化，避免影响 30 Hz 指令循环；需要时设置 `TELEOP_DISPLAY_DATA=1`。
录制程序复用 `make_piper_x_to_franka_teleop_processor` 时也应传入
`fixed_franka_quat_xyzw=FRANKA_REF_QUAT_XYZW` 和 `position_deadband_m=0.0015`，
以便记录与遥操作一致的固定姿态动作。

矩形工作空间并不意味着所有组合角落都可在固定姿态下到达。当前固定姿态配置启用
`position_boundary_guard: true`。原生控制器在关节接近边界，
或 Jacobian 最小奇异值下降到 0.12 以下且仍朝目标运动时，提前制动并保持一个固定 XYZ 目标，
不再持续追逐外侧目标。日志 `position_limited=1` 表示位置边界保护生效；
主臂映射目标沿被阻挡方向退回保持点内侧至少 5 mm 后解除。该保护可能提前限制组合角落，
不是全工作空间的可达性证明，也不是路径规划器。部署后先检查目标/实测位姿与边界保持效果，
再开始正式采集。

## 控制机每次启动

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=23

source ~/franka_ws/install/setup.bash
ros2 launch franka_ros2_bridge franka_bridge.launch.py
```

```bash
ros2 topic hz /franka/joint_state
ros2 topic echo --once /franka/end_pose
```

## 控制机首次 build

先确认工作区中的包是完整的当前源码。该包必须包含
`src/cartesian_stream.cpp`、`src/cartesian_tracking.h`、`src/cartesian_ik.h`、`src/joint_limits.h` 和 `third_party/ruckig`；控制机若使用软链接，请让它指向实际的仓库目录，
不要指向旧版本或不完整的拷贝：

```bash
ls -l ~/franka_ws/src/franka_ros2_bridge
test -f ~/franka_ws/src/franka_ros2_bridge/src/cartesian_stream.cpp
```

第二条命令没有输出即表示文件存在。若不存在，先更新或重新同步
`franka_ros2_bridge` 源码，再继续构建。

原生扩展需要 libfranka、Eigen3 和 pybind11。libfranka 默认位于
`~/franka/libfranka`；自定义位置时设置 `FRANKA_ROOT`。Eigen3 默认使用
`/usr/include/eigen3`；自定义位置时设置 `EIGEN3_INCLUDE_DIR`。
在线轨迹使用包内固定版本的 Ruckig Community 0.15.3（MIT），随扩展一起编译；
不使用 frankx 中的旧版 0.2.x，也不需要额外 pip 安装或运行时联网。

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=23
python3 -m pip install pybind11
sudo apt install libeigen3-dev

cd ~/franka_ws
colcon build --symlink-install --packages-select franka_ros2_bridge
source install/setup.bash
```

代码更新后，在控制机重新执行：

```bash
source /opt/ros/humble/setup.bash
cd ~/franka_ws
colcon build --symlink-install --packages-select franka_ros2_bridge
source install/setup.bash
```

没有 libfranka 的机器不要编原生扩展：

```bash
FRANKA_SKIP_STREAM_EXT=1 colcon build --symlink-install --packages-select franka_ros2_bridge
```

启动后日志应出现 `Resolved-rate Cartesian stream is running at 1 kHz`。
若提示 `cartesian_stream native module is missing`，说明扩展没编上，检查 pybind11、
Eigen3、`-lfranka` 和 colcon 编译输出。

按现场网络和安全空间编辑 `config/franka_bridge.yaml`。
流式轨迹同时限制速度、加速度和 jerk：

- 平移：`max_linear_velocity`、`max_linear_acceleration`、`max_linear_jerk`
- 旋转：`max_angular_velocity`、`max_angular_acceleration`、`max_angular_jerk`
- 最短轨迹规划时间：`1/(2π × tracking_frequency_hz)`（`5.0` 对应约 32 ms）
- 静止输入死区：`linear_deadband`、`angular_deadband`
- 冗余姿态保持：`lock_elbow`（在 Jacobian 零空间中保持启动时的第三关节姿态）
- 启动限制渐变：`initial_sync_scale: 0.60`，在 `startup_ramp_sec: 0.30` 秒内平滑恢复完整限制
- 实时线程 CPU：`control_cpu`（本机默认 CPU 4，`-1` 表示允许迁移）
- 夹爪状态轮询：`gripper_poll_rate_hz`（默认 20 Hz，后台更新缓存，不阻塞 ROS 回调）

命令话题使用 `KEEP_LAST(1)`，每次只追踪最新目标，避免遥操作链路短暂拥塞后回放旧指令。

每个 FCI 回调从机器人实际接受的 `q_d`、`dq_d`、`ddq_d` 计算位姿、速度和加速度，
再针对最新绝对目标规划下一个 1 ms 的运动和制动。三者使用同一时间基准，避免把阻抗伺服的
实测位置延迟混入规划状态。实测位姿继续发布到 ROS，并用于跟踪误差诊断。
规划前按 Jacobian 逆映射分配关节动态能力，`planner_scale` 记录速度/加速度/jerk 的分配比例。
平移与旋转独立规划，四元数取最短旋转；按目标方向、当前速度和制动需求分配三轴能力，
每组分配向量的范数不超过 1。非主运动轴保留修正余量，避免运动学曲率引起侧向漂移。
逆运动学优先保持末端 XYZ，旋转在平移任务的零空间内求解；最小奇异值低于 0.10 时，
阻尼限制较弱的旋转方向，避免姿态目标以额外平移为代价。阻尼作用于本周期速度修正，
不会每毫秒重复削减已经接受的运动速度。第三关节保持项使用独立的正交零空间投影，
避免阻尼逆解使保持肘部的动作影响末端。正常构型不加入固定 Jacobian 阻尼。

大角度姿态目标若继续把关节推向限位，控制器会提前制动并保持当时的可行姿态，
同时继续跟踪 XYZ；目标转回可用方向后自动恢复旋转。日志 `orientation_limited=1` 表示
这项约束正在生效。矩形 XYZ 工作空间并不保证其中每一点都能实现任意姿态。
此次 IK 修改包含 C++，需在控制机重新构建原生扩展。
加速度与 FCI 的离散速度差保持一致，逆映射时扣除一次 `Jdot*dq` 曲率项。
关节命令仍从实际接受的状态连续限速；丢失多个周期后也只推进一个 1 ms 限速步，
libfranka 自带的关节 rate limiter 保持开启。规划失败会锁存控制故障，需恢复机器人并重启 bridge。

默认 YAML 是响应型遥操作参数：平移 `0.40 / 2.0 / 12.0`，旋转
`0.80 / 4.0 / 25.0`（速度 / 加速度 / jerk）。若现场负载、工具或工作区变化，先降低
加速度和 jerk 验证，再逐步恢复；不要靠降低 `tracking_frequency_hz` 处理静止噪声，静止噪声由
`linear_deadband` 和 `angular_deadband` 负责。

运行时每 5 秒输出一次 `FCI success_rate`、`max_period_ms`、本周期新增的
`delayed_callbacks`、实际 `control_cpu`、CPU 迁移次数、Jacobian 最小奇异值、关节降速比例和
末端误差。`cpu_migrations` 正常应保持为 `0`；`joint_scale` 长期小于 `1` 表示关节速度或关节
限位正在约束运动。遥操作运行时 `target_hz` 应接近主臂的 30 Hz，`target_age_ms` 通常应小于
50 ms；目标年龄超过 200 ms 会告警。`ros_hz` 和 `ros_max_gap_ms` 显示 ROS 回调频率及
本窗口最大接收间隔，`dispatch_ms` 显示最近一次目标从回调到 C++ 的转发延迟。
`gripper_age_ms` 和 `gripper_poll_errors` 单独反映夹爪状态；夹爪读取迟延不会阻塞机械臂状态
发布或目标接收，过期夹爪状态也不会继续冒充新观测发布。

控制线程异常会停止状态发布并输出实际的 libfranka 错误，避免把缓存状态伪装成新状态。

若更新并重新 build 后仍出现 `cartesian_motion_generator_*`，先确认启动日志包含
`Resolved-rate Cartesian stream`；该错误名称通常说明仍在运行旧的笛卡尔 motion generator
二进制。任何流式控制故障都应先在 Desk 恢复机器人，再重启 bridge。故障会被锁存，不会继续
接受目标或重复刷同一异常。日志中的
`control_command_success_rate` 应接近 `1.0`；低于 `0.99` 时不能靠提高轨迹参数解决，必须先处理
控制机实时性和有线网络。控制前至少确认电源策略为 `performance`、`irqbalance` 不会移动机器人
网卡 IRQ，并让 bridge 与该 IRQ 使用不同的物理 CPU。例如：

```bash
powerprofilesctl set performance
sudo systemctl stop irqbalance

# 先从 /proc/interrupts 确认机器人网卡 IRQ 和 CPU，再设置亲和性。
cat /proc/interrupts | grep -E 'enp|eth'
cat /proc/irq/<IRQ>/smp_affinity_list

# 本机当前 enp3s0 是 IRQ 129，固定到 CPU 5，与 YAML 中 control_cpu=4 分离。
echo 5 | sudo tee /proc/irq/129/smp_affinity_list

ros2 launch franka_ros2_bridge franka_bridge.launch.py
```

遥操作脚本默认关闭 Rerun 和逐帧数据显示，避免干扰 30 Hz 指令循环。需要调试显示时显式设置
`TELEOP_DISPLAY_DATA=1`。该脚本的主臂端 ROS 队列深度为 1；桥接端命令订阅使用
`BEST_EFFORT + KEEP_LAST(1)`。
若主臂运行在另一台电脑，需要同步 LeRobot 的 `src/lerobot/robots/franka/franka.py`、
`src/lerobot/teleoperate.py` 和 `examples/piper_x_to_franka/teleoperate.py`；仅更新控制机不能
改变远端发布者。Piper-X 到 Franka 仍使用绝对工作空间映射，不计算接管偏移。

第一条 ROS 目标与机器人初始位姿可以不同：例如机器人当前 `x=0.40 m`，收到 `x=0.43 m`，
就从当前状态平滑运动到 `0.43 m`。随后收到 `0.44 m` 时立即以它作为最新目标，不添加启动偏移，
也不等待到达 `0.43 m`。位置和姿态输入死区只抑制各自的小幅噪声，第一帧不会被死区忽略。

启动后的 `startup_ramp_sec=0.30` 秒内，动态限制从 `initial_sync_scale=0.60` 平滑升到完整值，
期间一直响应最新目标。日志 `startup=waiting/ramping/live` 只表示等待首帧/限制渐变/完整限制，
不表示需要等机器人到位；可以直接操作主臂。主臂端保持原有绝对工作空间标定即可。
如果停止发布，控制器继续平滑到达并保持最后一个目标；`command_timeout_sec` 是队列过期时间，
不是运动中的断流急停时间。

可在无机器人连接时运行回归测试：

```bash
source /opt/ros/humble/setup.bash
cd ~/franka_ws
colcon test --packages-select franka_ros2_bridge
colcon test-result --verbose
```

测试覆盖首帧偏差、启动期间更新目标、30 Hz 移动、换向、停手、噪声、四元数符号切换和
下游限速，检查收敛及速度/加速度/jerk。还使用实测关节姿态的 Panda 七关节运动学模型
验证 5 cm 目标响应、连续主臂轨迹、零死区重复目标，以及夹爪阻塞时的状态和目标转发。
它们是离线测试，实机延迟、负载和 FCI 实时性需现场验证。

## 接口

- 发布 `/franka/joint_state` (`sensor_msgs/JointState`)
- 发布 `/franka/end_pose` (`geometry_msgs/PoseStamped`)
- 发布 `/franka/gripper_state` (`sensor_msgs/JointState`，`name=panda_finger`，开口单位米)
- 订阅 `/franka/joint_cmd` (`sensor_msgs/JointState`)，流式模式下忽略
- 订阅 `/franka/end_pose_cmd` (`geometry_msgs/PoseStamped`)
- 订阅 `/franka/gripper_cmd` (`sensor_msgs/JointState`，`name=panda_finger`，开口单位米)

末端命令的 `frame_id` 必须为空或等于 `base_frame`，四元数为 XYZW，位置在 workspace 内。
流式模式只替换最新绝对笛卡尔目标；1 kHz 回调通过在线轨迹、阻尼 Jacobian 逆解和关节动态
限制产生连续关节速度。
状态从同一控制回调读取，运动期间仍然新鲜。

## 安全启动

1. 确认急停、碰撞保护和 Franka Desk 安全配置可用。
2. 首次测试清空工作区，人员在运动范围外。
3. 确认 `robot_ip` 正确，没有其它 FCI 客户端占用机器人。
4. 从较小的 `max_linear_velocity` / `max_angular_velocity` 和收紧的 workspace 开始。
5. 不要把本桥接节点当作安全控制器。
