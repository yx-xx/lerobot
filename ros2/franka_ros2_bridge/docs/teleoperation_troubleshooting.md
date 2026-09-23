# Franka 遥操作问题复盘与学习指南

本文记录 Piper-X 主臂遥操作 Franka Panda 过程中实际遇到的问题、判断依据、修复方式，
以及理解这些问题所需要的基础知识。它不是 Franka 官方安全手册，也不能替代 Desk、急停、
碰撞保护和现场风险评估。

更新日期：2026-09-23。

当前模式：绝对位姿映射，启动时允许主从有小偏差，第一帧开始就连续跟踪最新目标。
不添加接管偏移，不再锁存第一点等待到位。

## 1. 系统结构

当前控制链路包含三个不同频率的循环：

```text
Piper-X 主臂
    │ 读取末端位姿，约 30 Hz
    ▼
LeRobot teleoperate.py
    │ 标定映射、工作空间映射、单帧目标裁剪
    ▼
ROS 2 /franka/end_pose_cmd
    │ BEST_EFFORT + KEEP_LAST(1)
    ▼
franka_ros2_bridge Python 节点
    │ 校验 frame、四元数和工作空间，只保留最新目标
    ▼
cartesian_stream.cpp
    │ 1 kHz 在线轨迹（含制动）+ 阻尼 Jacobian 逆解 + 关节动态限速 + FCI
    ▼
Franka Panda

Franka 状态 ──50 Hz──> ROS 2 ──> LeRobot
夹爪状态   ───2 Hz───> ROS 2 ──> LeRobot
```

理解问题时必须区分这三个时间尺度：

| 环节 | 典型频率 | 周期 | 作用 |
|---|---:|---:|---|
| 主臂采样与目标发布 | 30 Hz | 33.3 ms | 决定操作者输入刷新率 |
| ROS 状态发布 | 50 Hz | 20 ms | 提供关节、末端和夹爪观测 |
| libfranka FCI | 1000 Hz | 1 ms | 真正产生每个机器人控制命令 |

30 Hz 目标不能直接作为 1 kHz 位姿命令发送。桥接器必须在两帧主臂数据之间继续生成连续、
可停止的 1 kHz 轨迹。

## 2. 问题一：主臂静止，从臂仍然波动

### 表现

- 主臂已经保持不动，Franka 仍在目标附近来回移动。
- 目标位姿变化很小，但从臂没有稳定停住。
- 旧控制器在固定目标仿真中会反复越过目标。

### 原因

这是“控制稳定性”和“输入噪声”叠加的问题。

第一，旧控制方式根据位置误差自行积分速度、加速度和 jerk。如果没有正确计算制动距离，
机器人接近目标时仍保留较大速度，越过目标后再反向加速，形成持续振荡。

第二，真实主臂即使静止，编码器、运动学计算和浮点转换仍会产生微小位置和姿态变化。
如果每个微小变化都被当成新目标，轨迹器就永远不会真正进入保持状态。

### 修复

- 使用固定版本 Ruckig 0.15.3，根据当前位置、速度、加速度规划运动及制动，再生成关节速度。
- 规划前把关节动态限制通过 Jacobian 逆映射纳入笛卡尔约束，避免下游限速后制动距离失配。
- 使用 `linear_deadband` 和 `angular_deadband` 分别过滤位置、姿态输入；第一帧始终保留。
- 目标速度和加速度为零，停止更新后平滑到达最后目标。死区只过滤输入，不直接切断运动速度。
- 规划位姿由 `q_d` 正运动学得到，与 `dq_d`、`ddq_d` 使用同一时间基准；实测位姿用于 ROS
  状态和误差诊断。不要将有阻抗伺服延迟的实测位置与当前指令速度混合，否则会刹车过晚。

2026-09-23 的离线复现还发现，后来的“P 位置误差转速度 + jerk 限制”方案也会在固定目标
附近持续振荡。限速器只能限制导数，不能代替包含制动的轨迹规划；当前已经移除这一组合。

当前死区：

```yaml
linear_deadband: 0.001   # 1 mm
angular_deadband: 0.010  # 约 0.57 deg
```

### 需要学习的概念

- 闭环稳定性、阻尼和超调
- 位置、速度、加速度和 jerk 的关系
- 测量噪声、死区和滞环
- commanded state 与 measured state 的区别

## 3. 问题二：关节加速度不连续触发安全反射

### 表现

```text
cartesian_motion_generator_joint_acceleration_discontinuity
motion aborted by reflex
```

机器人立即停止，随后状态读取也报错。

### 原因

Franka 的笛卡尔位姿命令最终仍需通过逆运动学转成关节运动。笛卡尔位置看起来连续，并不自动
保证每个关节的速度和加速度连续。以下情况都会放大风险：

- 新位姿命令发生跳变。
- 控制器没有连续限制速度、加速度和 jerk。
- 姿态变化在某些构型下引起很大的关节运动。
- 1 kHz 数据包延迟，机器人连续几周期没有收到新命令。
- 机器人接近奇异位形或关节限位。

`motion aborted by reflex` 是 Franka 安全系统主动终止运动，不应通过提高碰撞阈值来掩盖。

### 修复

- 不再使用 Franka Cartesian pose motion generator 的内部 IK。
- 在 1 kHz 回调中用 `q_d` 处 Jacobian 的阻尼伪逆将末端速度转换成 7 关节速度。
- 接近奇异位形时增大阻尼；接近关节限位或关节速度上限时整体降速。
- 以机器人实际接受的 `dq_d`、`ddq_d` 为起点限制关节速度、加速度和 jerk。
- 保持 libfranka 关节 rate limiter 开启，作为通信抖动下的最后一道独立保护。
- 在 Jacobian 零空间中保持启动时的第三关节姿态，避免冗余构型漂移。
- 记录 `control_command_success_rate`、周期延迟、`min_sigma` 和 `joint_scale`。

### 重要关系

```text
笛卡尔命令连续
    不等于
关节命令一定连续

轨迹连续 + 构型合理 + FCI 周期稳定
    才能显著降低关节加速度反射风险
```

## 4. 历史问题：Ruckig 返回 -110

### 表现

```text
Ruckig failed to generate a safe Cartesian trajectory: -110
```

`-110` 在当时使用的 Ruckig 版本中表示：

```text
ErrorExecutionTimeCalculation
```

### 根因

项目使用的是 frankx 内附的旧版 Ruckig `0.2.x`。连续动态重规划时，轨迹状态可能恰好位于
速度或加速度约束边界，例如：

```text
velocity = 0.100000000000...
acceleration = 0.00000000000000008
max_velocity = 0.10
```

从物理意义看这个状态有效，但旧版算法的浮点边界判断可能无法构造下一条轨迹，于是返回
`-110`。这不是目标超出工作空间，也不是机器人已经碰撞。

### 当前处理

桥接器内固定包含 Ruckig Community 0.15.3 源码及 MIT 许可，不再链接 frankx 内的 0.2.x。
每个周期使用机器人实际接受的运动状态重新规划；输出速度按离散加速度构造，避免连续轨迹的
速度样本与 FCI 离散 `ddq_d` 定义不一致，造成实际 jerk 只有规划值的一半、刹车过晚。

若三轴共同到达时间的数值计算失败（`-111`），仅该步改用独立轴规划；目标及各轴动态限制相同。
`trajectory_sync_fallbacks` 记录发生次数。其它规划错误仍会终止本次 FCI 控制并锁存故障，
由 libfranka/机器人执行停止处理，不忽略错误继续追目标。

测试使用生产代码中的轨迹计算器和 libfranka 限速器，覆盖位置、姿态、换向、下游动态限制等场景。
离线通过不代表实机网络、动力学和接触工况已验证。

## 5. 问题四：机器人动作很慢、不跟手

### 表现

- 主臂已经移动较远，从臂仍在慢慢追赶。
- 日志显示命令速度长期贴着 `0.10 m/s` 上限。
- 快速操作时主从位置误差越来越大。

### 原因

原始参数主要面向保守调试，不适合低延迟遥操作：

```yaml
max_linear_velocity: 0.10
max_linear_acceleration: 0.30
max_linear_jerk: 1.0
tracking_frequency_hz: 2.0
```

各参数影响不同：

| 参数 | 太小时的表现 | 太大时的风险 |
|---|---|---|
| 最大速度 | 长距离追赶慢 | 高速运动、碰撞能量增加 |
| 最大加速度 | 起步和反向迟钝 | 更容易激发关节动态限制 |
| 最大 jerk | 加速变化缓慢、手感软 | 冲击增大，对周期抖动更敏感 |
| tracking frequency | 小位移也要较长时间 | 更容易追踪输入噪声 |

另一个问题是 LeRobot 的相对目标裁剪。如果每帧只允许目标位于当前从臂前方 3 cm，轨迹器就会
不断规划“3 cm 后停车”，而不是持续追赶主臂。这个安全窗口不能替代底层速度限制。

绝对映射下，第一点与 Franka 当前位姿不同是正常情况。比如机器人在 `x=0.40 m`，第一帧
是 `0.43 m`，就从当前状态平滑接近 `0.43 m`；下一帧是 `0.44 m` 时立即改追最新目标。
不能将第一帧减掉再加到机器人初始位置上，否则就变成相对接管，产生固定位置偏移。

以前“锁住第一帧，到位停稳才启用遥操作”的流程会让从臂追旧目标。当前已删除到位门槛，
启动过程只是动态限制随时间渐变，主臂从一开始就可以移动。

### 当前参数

```yaml
max_linear_velocity: 0.40
max_linear_acceleration: 2.0
max_linear_jerk: 12.0
max_angular_velocity: 0.80
max_angular_acceleration: 4.0
max_angular_jerk: 25.0
tracking_frequency_hz: 5.0
initial_sync_scale: 0.60
startup_ramp_sec: 0.30
```

`initial_sync_scale` 保留旧名字，现仅表示启动限制比例。0.30 秒内平滑恢复完整限制，
全程追踪最新绝对目标。`tracking_frequency_hz` 现在设置最短规划时间 `1/(2πf)`，5 Hz 约为
32 ms，不是原来的 P 增益，也不能保证机器人实际达到该带宽。

平移、旋转独立规划；根据目标变化方向、位置误差以及当前速度和加速度分配三轴能力，
每组分配向量范数不超过 1。非主运动轴保留 10% 修正余量后再归一化，避免曲率和离散采样
引起侧向漂移。再按关节能力缩减动态限制，并经过最终关节限速器。
单轴运动不再固定除以 `√3`，静止轴也不再预留完整的运动能力。
提高 YAML 上限前应先看 `planner_scale` 和 `joint_scale`。

### 2026-09-23：参数已经提高，实际仍然很慢

这次日志与早期低速度参数的问题不同：

```text
min_sigma=0.207~0.223
joint_scale=1.000
planner_scale≈0.75/0.28/1.00
target_hz=6~21
target_age_ms 最大约 2008
success_rate=0.970~0.990，max_period_ms=5
```

这说明同时存在规划能力被压低、目标链路间歇停顿、FCI 周期抖动。`joint_scale=1` 只能说明
最终的关节速度/限位缩放没有介入，不能说明前面的轨迹没有被减速。
最后的 `User Stop pressed!` 是用户停止按钮触发，不是加速度反射。

代码中确认并修复了以下问题：

1. **闲置方向占用关节预算。** 原来同时预留 XYZ 平移、三轴旋转的全额能力，在此次姿态
   把加速度比例压到约 0.30。现在按实际运动方向分配；同一姿态 Z 向运动初始比例约 0.60，
   随机器人构型变化继续调整，关节限速保持有效。
2. **固定阻尼在正常构型反复衰减速度。** 每个 1 ms 回调都重新逆解实际接受的速度，
   即使很小的固定阻尼也会不断削弱它。现在 `min_sigma >= 0.10` 不加阻尼，接近奇异位形
   时平滑增加。该阈值对应当前米/弧度 Jacobian 的尺度。
3. **离散加速度与曲率转换不一致。** `ddq_d` 是离散速度差。规划输入按同一时间基准
   恢复笛卡尔加速度；输出转回当前 Jacobian 时扣除一次 `Jdot*dq`，避免重复注入曲率。
   非主运动轴保留足够修正能力，防止微小偏差拖慢整个三轴同步轨迹。
4. **夹爪读取阻塞 ROS 和目标工作线程。** 旧 `frankx.Gripper.width()` 在阻塞读取网络时
   没有释放 Python GIL。只把它移到另一个 Python 线程仍会阻塞其它 Python 线程。
   当前使用释放 GIL 的原生 libfranka 接口在后台轮询，ROS 回调只读缓存。夹爪状态过期时
   暂停夹爪发布，机械臂状态和目标转发继续运行。
5. **主臂端可能积压旧消息。** LeRobot 末端、夹爪发布者改为 `BEST_EFFORT + KEEP_LAST(1)`；
   关闭显示时也不再每帧打印耗时。远端主臂电脑需要同步相应源码。

离线回归使用日志里的七个关节角、Panda 几何模型和实际关节限速器。5 cm Z 向目标约
0.48 秒进入 1 mm 误差范围，超调小于 1 mm；零死区、重复相同目标也能收敛。
另有约 30 Hz 的三维移动和姿态变化测试。测试验证运动学与算法，不能代替实机负载、
阻抗伺服和 FCI 网络测试。本轮保留 YAML 中的速度、加速度和 jerk 数值。

### 调参原则

必须先让 `control_command_success_rate` 稳定，再调响应参数。网络和调度不稳定时继续增加加速度
或 jerk，只会让安全反射更容易发生。

推荐顺序：

1. 确认 FCI 通信稳定、CPU 不迁移。
2. 调整死区，使静止时不抖动。
3. 调整 `tracking_frequency_hz`，改善短距离响应。
4. 调整加速度和 jerk，改善起停手感。
5. 最后调整最大速度，满足大范围跟随需求。

每次只改变一个参数，并记录日志和主观手感。

## 6. 问题五：间歇性卡住，稍后又恢复

### 表现

机器人有时很跟手，但会短暂停住或反应明显变慢，之后恢复。一次实测日志为：

```text
FCI success_rate=0.860
max_period_ms=4.000
delayed_callbacks 持续增加
ruckig_fallbacks=0
```

### 判断

这一次卡顿不是 Ruckig 导致的，因为：

- `ruckig_fallbacks=0`。
- Ruckig 没有退出控制。
- FCI 成功率同时从 `0.99` 降到 `0.86`。
- 目标为 1 ms 的控制周期出现了 4 ms 周期。

这说明控制机没有稳定地按 1 kHz 收发 FCI 数据包。机器人在缺少新包时会使用内部保护行为，
操作者感受到的就是短暂卡顿。

控制器不能假设每个发出的命令都被机器人接收。当前实现每个周期都以机器人实际接受的
`dq_d` 和 `ddq_d` 为关节限速起点，并且每个可发送的数据包只推进一个 1 ms 限速步。这样会在
通信不稳时暂时降低实际运动速度，而不会为了追赶墙钟时间制造关节命令跳变。

### 已发现的控制机状态

```text
实时内核: 5.15 PREEMPT_RT
电源模式（最初）: balanced
Intel EPP（最初）: powersave
irqbalance: active
机器人网卡: enp3s0, 172.16.0.3
机器人地址: 172.16.0.2
机器人网卡 IRQ: 129
IRQ 当前 CPU: 5
普通 ping: 0% 丢包，平均 0.184 ms，最大 1.126 ms
```

普通 ping 零丢包并不能证明 1 kHz FCI 合格。FCI 每 1 ms 都需要及时处理数据，偶发的 2 至 4 ms
调度延迟已经足以降低成功率。

### 可能的延迟来源

- CPU 节能状态唤醒。
- 实时线程在 CPU 之间迁移，缓存和调度状态发生变化。
- 控制线程与机器人网卡 IRQ 落在同一 CPU。
- `irqbalance` 在运行中移动网卡 IRQ。
- 网卡驱动、软中断或 PCIe 电源管理产生尾延迟。
- 夹爪状态和机械臂 FCI 共用网卡；阻塞的状态读取还可能占用 Python 执行锁。
- 其它设备中断、Wi-Fi、磁盘或图形任务抢占 CPU。

### 当前措施

- 电源模式设置为 `performance`。
- 1 kHz libfranka 控制线程固定到 CPU 4。
- 机器人网卡 IRQ 129 当前位于 CPU 5；停止 `irqbalance` 后才能可靠地固定在该 CPU。
- 夹爪状态在释放 GIL 的独立线程以 2 Hz 更新缓存，机械臂 ROS 回调不等待夹爪网络 I/O。
- 记录实际控制 CPU 和 CPU 迁移次数。

`irqbalance` 需要 sudo 权限停止，每次重新启动系统后都需要确认。

## 7. 问题六：主臂/ROS 命令链路也可能卡顿

FCI 正常不代表主臂命令一定正常。主臂在另一台机器时，命令还会经过：

```text
Piper CAN -> LeRobot Python -> DDS -> Wi-Fi/局域网 -> bridge Python -> C++ buffer
```

任一环节暂停，Franka 都只能继续追踪最后一个收到的目标。

为区分上游命令卡顿与 FCI 卡顿，日志增加了：

| 指标 | 正常值 | 异常含义 |
|---|---:|---|
| `ros_hz` | 约 30 Hz | ROS 订阅回调收到的有效目标频率降低 |
| `target_hz` | 约 30 Hz | 转发到 C++ 的目标频率降低 |
| `ros_max_gap_ms` | 通常约 33 ms | 本窗口内 ROS 最大接收间隔，含当前持续断流 |
| `dispatch_ms` | 通常数毫秒以内 | 最近一次目标在 Python 队列中的等待时间 |
| `target_age_ms` | 通常小于 50 ms | 大于 200 ms 表示最新目标已经过期 |

ROS 命令发布和订阅都使用 `BEST_EFFORT + KEEP_LAST(1)`，原因是遥操作只关心最新目标。网络短暂拥塞后，
系统不应该依次回放已经过期的历史目标。

## 8. 两个容易误判的次生错误

### State read failed

运动控制已经被 libfranka 中止后，共用同一个机器人连接的状态读取也会失败。因此：

```text
运动控制异常
    └── State read failed
```

通常是同一个故障的前因和后果，不是两个独立问题。应首先阅读最早出现的运动错误。

### Ctrl+C 时 rcl_shutdown already called

这是 ROS 2 Python 上下文被重复 shutdown，和机器人控制质量无关。当前代码在调用 shutdown 前
检查 `rclpy.ok()`，已经修复。

## 9. 诊断日志说明

bridge 每 5 秒输出一行：

```text
FCI success_rate=1.000
max_period_ms=1.000
delayed_callbacks=0(+0)
control_cpu=4
cpu_migrations=0
min_sigma=0.180
joint_scale=1.000
position_error_mm=0.5
orientation_error_deg=0.2
target_hz=30.0
target_age_ms=12.0
```

| 字段 | 含义 | 期望值 |
|---|---|---|
| `success_rate` | 最近 FCI 命令成功比例 | `0.99` 至 `1.00`，越接近 1 越好 |
| `max_period_ms` | 启动以来观察到的最大回调周期 | 理想约 1 ms，偶发 2 ms 需关注 |
| `delayed_callbacks` | 超过 1.5 ms 的累计次数 | 每个 5 秒窗口新增接近 0 |
| `control_cpu` | 控制回调实际运行 CPU | 当前应为 4 |
| `cpu_migrations` | 控制线程迁移次数 | 应为 0 |
| `elbow_locked` | 是否启用零空间第三关节姿态保持 | 当前应为 `True` |
| `startup` | 动态限制启动状态 | `waiting` / `ramping` / `live`；`ramping` 已能跟随 |
| `min_sigma` | Jacobian 最小奇异值 | 等待目标时为 -1；运动时越接近 0 越接近奇异位形 |
| `joint_scale` | 关节速度/限位保护的整体缩放 | 正常为 `1.0` |
| `planner_scale` | 规划前按关节能力分配的速度/加速度/jerk 比例 | 小于 1 表示该构型限制相应动态能力 |
| `trajectory_sync_fallbacks` | 三轴同步数值失败后使用独立轴规划的累计次数 | 正常为 0；持续增长时记录工况分析 |
| `position_error_mm` | 当前末端位置跟踪误差 | 静止到位后约小于 1 mm |
| `orientation_error_deg` | 当前末端姿态跟踪误差 | 静止到位后约小于 0.57° |
| `ros_hz` | ROS 订阅回调收到有效目标的频率 | 遥操作时约 30 Hz |
| `target_hz` | Python 工作线程转发给 C++ 的目标频率 | 遥操作时约 30 Hz |
| `ros_max_gap_ms` | 本窗口内 ROS 最大接收间隔，含当前持续断流 | 通常约 33 ms |
| `dispatch_ms` | 最近一次目标从 ROS 回调到 C++ 的排队时间 | 通常数毫秒以内 |
| `gripper_age_ms` | 最近有效夹爪观测的年龄 | 2 Hz 轮询时通常小于 500 ms |
| `gripper_poll_errors` | 夹爪后台读取失败的累计次数 | 正常为 0 |
| `target_age_ms` | 最新目标距离现在的时间 | 通常小于 50 ms |

注意：`max_period_ms` 是启动以来最大值，不会自动下降；判断当前是否还在抖动，应看
`delayed_callbacks` 括号中的新增数量。

`startup` 的三个状态为：尚未收到目标时 `waiting`，启动限制渐变时 `ramping`，渐变结束后
`live`。这不是接管许可状态；`ramping` 期间已经响应最新目标，无需保持主臂静止或等待到位。

位置、姿态误差按最新收到的原始绝对目标计算，因此可能保留输入死区范围内的小误差。
停止发目标后会继续到达并保持最后目标，当前没有运动中的断流急停；目标年龄告警只用于诊断。

## 10. 快速诊断决策树

```text
机器人发生卡顿
    │
    ├─ target_hz 明显低于 30，或 target_age_ms > 200
    │      └─ 对照 ros_hz、ros_max_gap_ms、dispatch_ms；检查主臂、DDS 或 Python 阻塞
    │
    ├─ target 正常，但 success_rate < 0.99 或 delayed_callbacks 快速增加
    │      └─ 检查电源模式、实时调度、CPU 亲和性、IRQ、网卡和夹爪流量
    │
    ├─ FCI 和 target 都正常，但 joint_scale 长期低于 1
    │      └─ 检查关节速度饱和、关节限位以及目标是否接近不可达区域
    │
    ├─ min_sigma 接近 0，同时动作明显变慢
    │      └─ 当前构型接近奇异位形；先改变构型，不要继续提高速度参数
    │
    ├─ 所有诊断正常，但始终缓慢
    │      └─ 分析速度、加速度、jerk、tracking frequency 和目标裁剪
    │
    └─ 机器人静止时抖动
           └─ 检查输入噪声、死区、目标是否持续漂移以及控制器是否越过目标
```

## 11. 每次硬件测试前的检查

确认人员离开运动范围、急停可用、Desk 状态正常，然后执行：

```bash
powerprofilesctl set performance
sudo systemctl stop irqbalance
echo 5 | sudo tee /proc/irq/129/smp_affinity_list

powerprofilesctl get
systemctl is-active irqbalance
cat /proc/irq/129/smp_affinity_list
```

期望输出：

```text
performance
inactive
5
```

启动 bridge：

```bash
source /opt/ros/humble/setup.bash
source ~/franka_ws/install/setup.bash
ros2 launch franka_ros2_bridge franka_bridge.launch.py
```

主臂电脑需同步 LeRobot 的 `src/lerobot/robots/franka/franka.py`、`src/lerobot/teleoperate.py`
和 `examples/piper_x_to_franka/teleoperate.py`。默认关闭 Rerun 和逐帧终端输出；需要显示时再设置：

```bash
TELEOP_DISPLAY_DATA=1 python examples/piper_x_to_franka/teleoperate.py
```

## 12. 当前代码中的对应位置

| 内容 | 文件 |
|---|---|
| 1 kHz Jacobian 逆解、关节限速、CPU 亲和性和诊断 | `src/cartesian_stream.cpp` |
| 最新绝对目标、启动渐变、在线轨迹和四元数计算 | `src/cartesian_tracking.h` |
| 关节动态预算、限速值和奇异位形阻尼 | `src/joint_limits.h` |
| 固定版本轨迹库和许可 | `third_party/ruckig` |
| 无硬件回归仿真 | `test/cartesian_tracking_test.cpp`、`test/test_cartesian_tracking.py` |
| 动态限制、死区、控制 CPU 和工作空间 | `config/franka_bridge.yaml` |
| ROS QoS、命令队列、状态发布和诊断日志 | `franka_ros2_bridge/bridge_node.py` |
| C++ stream 与夹爪接口 | `franka_ros2_bridge/control/stream_controller.py` |
| Piper 到 Franka 的标定和遥操作入口 | `examples/piper_x_to_franka/teleoperate.py` |

## 13. 建议学习顺序

### 第一阶段：轨迹基础

- 位置、速度、加速度、jerk。
- 梯形速度轨迹与 S 曲线轨迹。
- 为什么到达位置目标时还必须满足目标速度和目标加速度。
- 制动距离和在线轨迹生成（Online Trajectory Generation）。

### 第二阶段：机器人接口

- 笛卡尔空间和关节空间的区别。
- 逆运动学、奇异位形和冗余自由度。
- Franka joint velocity motion generator。
- `O_T_EE`、`q_d`、`dq_d` 和 `ddq_d`。
- 阻尼伪逆、Jacobian 最小奇异值与零空间姿态控制。

### 第三阶段：实时系统

- PREEMPT_RT 与普通 Linux 调度的区别。
- `SCHED_FIFO`、实时优先级和内存锁定。
- CPU affinity、超线程和缓存。
- 硬中断、软中断、IRQ affinity 和 `irqbalance`。
- 平均延迟与最坏情况延迟的区别。

### 第四阶段：ROS 2 通信

- DDS QoS：reliable、best effort、history 和 depth。
- 为什么遥操作通常使用 `KEEP_LAST(1)`。
- 发布频率、端到端延迟、消息年龄和时钟。
- 为什么高频日志和可视化也会影响控制循环。

### 第五阶段：数值计算

- 浮点误差和约束边界。
- 为什么数学上相等的值在计算机中不一定相等。
- 数值裕量、饱和和可行域。
- 四元数归一化、双覆盖和最短旋转。

## 14. 继续学习时可以重点讨论的问题

1. 速度、加速度和 jerk 限速器如何从当前关节状态平滑追踪新的关节速度？
2. 为什么笛卡尔轨迹连续仍可能触发关节加速度不连续？
3. `control_command_success_rate` 是怎样计算的，丢一个 1 ms 周期会发生什么？
4. CPU affinity 和 IRQ affinity 为什么要放在不同物理核？
5. `BEST_EFFORT + KEEP_LAST(1)` 为什么适合遥操作，而不适合所有机器人命令？
6. 如何定量测量主臂到从臂的端到端延迟，而不是只凭主观手感？
7. 如何在不增加静止抖动的前提下继续降低跟随误差？
