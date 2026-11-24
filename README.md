# LeRobot_use

这是一个使用LeRobot框架的机器人开发项目

## 首次安装

```bash
conda create -y -n lerobotx python=3.10
conda activate lerobotx

conda install ffmpeg -c conda-forge
sudo apt update
sudo apt install -y can-utils

cd ~/project/lerobot
pip install -e ".[feetech,piper]"
pip install scipy==1.15.3
```



## 启动 Piper-X 遥操作 Franka

```bash
conda activate lerobotx
cd /home/pc108/project/lerobot

source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=23

sudo ip link set can0 down
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up

python examples/piper_x_to_franka/teleoperate.py
```


## Franka 状态超时排查

`Franka joint state is stale` 表示客户端超过 `state_timeout_s` 没有收到可用的关节状态；
当前 Piper-X 配置的阈值为 3 秒。启动时能读到位姿，只能说明连接时收到过状态。
控制循环随后出错、ROS 消息传输中断，或收到的关节消息不符合要求，都可能触发这个检查。

客户端的超时报错会列出关节、末端和夹爪三个话题最近有效数据的年龄、ROS domain 和
接收线程状态。如果本机 ROS 接收线程异常，则直接报告 `Franka ROS executor failed`
及原始异常，避免等到缓存超时后才给出笼统报错。

先查看 **Franka 控制电脑** 运行 `franka_ros2_bridge` 的终端，保留故障前后的完整日志，
特别是以下两类消息及其后面的原始错误：

```text
State read failed; state publication stopped: ...
Cartesian stream failed; target updates are disabled until the bridge is restarted: ...
```

桥接器读取状态失败时会停止发布，不能通过重复发布旧状态或调大超时来恢复控制。
如果日志明确提示目标更新已禁用，需要先根据原始错误处理控制器故障，再重启桥接器。

在本机另开终端，使用与遥操作相同的 ROS 环境，检查消息是否持续到达：

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=23

ros2 topic info /franka/joint_state --verbose
timeout 8s ros2 topic hz /franka/joint_state
timeout 8s ros2 topic hz /franka/end_pose
timeout 8s ros2 topic hz /franka/gripper_state
```

也在控制电脑运行相同的频率检查：若控制电脑有持续数据而本机没有，优先检查两机的
ROS domain、网络和 QoS；若两端都没有数据，优先检查桥接器日志。发现发布者只说明
ROS 图中有该端点，不能证明它仍在持续发布。如果只有关节状态超时，还需检查消息中的
`panda_joint1` 至 `panda_joint7` 是否齐全、位置数组是否匹配且数值有限。


## 夹爪松开响应

Piper-X 的开口按 `common.py` 中的 `PENDANT_MM` 映射成 Franka 开口。
夹着物体时，Franka 实际开口可能大于映射的闭合目标。例如 40 mm 的物体挡住了
4.5 mm 的闭合目标，示教器松开到映射值 19 mm 时，目标仍小于物体宽度。

当前遥操作和采集都会在检测到明确的张开方向后，将目标至少设为
**当前 Franka 实际开口 + 5 mm**，并限制在 80 mm 上限内。
这个下限在本次张开过程中保持，避免后续小目标重新闭合，也不会每帧再累加 5 mm。
主动闭合时恢复原始绝对开口映射。方向判断的 1 mm 死区作用于映射后的输入开口，
小幅噪声不会切换开合方向，缓慢张开的输入会累计。

参数位于 `examples/piper_x_to_franka/common.py`：

```python
GRIPPER_RELEASE_CLEARANCE_M = 0.005
GRIPPER_DIRECTION_DEADBAND_M = 0.001
```

更新后重启遥操作或采集脚本即可加载。触发修正时终端会打印
`Gripper release: mapped=... mm, actual=... mm, target=... mm`，
分别表示原始映射开口、实际反馈开口和本次修正目标。采集保存的是修正后的 Franka 目标。

## 项目结构

```
├── benchmarks
│   └── video
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── crp.txt
├── docker
│   ├── Dockerfile.internal
│   └── Dockerfile.user
├── docs
│   ├── README.md
│   └── source
├── docs-requirements.txt
├── examples
│   ├── backward_compatibility
│   ├── dataset
│   ├── lekiwi
│   ├── phone_to_so100
│   ├── port_datasets
│   ├── so100_to_so100_EE
│   └── training
├── LICENSE
├── license.key
├── Makefile
├── MANIFEST.in
├── media
│   ├── gym
│   ├── hope_jr
│   ├── lekiwi
│   ├── lerobot-logo-light.png
│   ├── lerobot-logo-thumbnail.png
│   ├── so100
│   ├── so101
│   └── wandb.png
├── outputs
├── pyproject.toml
├── README_lerobot.md
├── README.md
├── requirements.in
├── requirements-macos.txt
├── requirements-ubuntu.txt
├── src
│   ├── lerobot
│   └── lerobot.egg-info
└── tests
    ├── artifacts
    ├── async_inference
    ├── cameras
    ├── configs
    ├── conftest.py
    ├── datasets
    ├── envs
    ├── fixtures
    ├── __init__.py
    ├── mocks
    ├── motors
    ├── optim
    ├── plugins
    ├── policies
    ├── processor
    ├── rl
    ├── robots
    ├── teleoperators
    ├── test_available.py
    ├── test_control_robot.py
    ├── transport
    ├── utils
    └── utils.py
```

## Piper-X 遥操作 Franka：数据采集与模型训练

当前这套硬件使用 Piper-X 作为示教器，通过末端位置映射控制 Franka。数据采集使用专用脚本，以确保每一帧保存的是映射后的 Franka `end_pose.*` 动作，而不是 Piper-X 原始动作。

### 1. 采集前准备

先启动 Franka ROS 2 bridge，再在本机执行：

```bash
conda activate lerobotx
cd /home/pc108/project/lerobot
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=23

sudo ip link set can0 down
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up
```

第一次建议只采集少量短 episode 做链路检查：

```bash
/home/pc108/miniforge3/envs/lerobotx/bin/python \
  examples/piper_x_to_franka/record_dataset.py \
  --repo-id=local/piper-franka-smoke \
  --root=data/piper-franka-smoke \
  --task="Pick up the cube and place it in the bin" \
  --episodes=3 \
  --episode-time-s=15 \
  --reset-time-s=8
```

确认相机、动作和保存都正常后，再采集正式数据，例如：

```bash
/home/pc108/miniforge3/envs/lerobotx/bin/python \
  examples/piper_x_to_franka/record_dataset.py \
  --repo-id=local/piper-franka-pick \
  --root=data/piper-franka-pick \
  --task="Pick up the cube and place it in the bin" \
  --episodes=50 \
  --episode-time-s=30 \
  --reset-time-s=10
```

采集过程中可以使用右方向键提前结束当前 episode，左方向键重录当前 episode，Esc 停止后续采集。正式数据建议保持任务描述、相机位置和物体摆放区域一致，先采集 30–50 个成功 episode，再根据泛化需求增加物体位置、光照和初始姿态的变化。重新接着已有数据采集时，保留相同参数并增加 `--resume`。

### 2. 检查数据集

采集完成后，先确认数据集元数据中的动作维度为 8，包含 Franka 的 7 维末端位姿和夹爪开口；观测应包含 15 维状态以及 `wrist_camera`、`left_camera` 两路图像。也可以用 LeRobot 的可视化工具查看 episode：

```bash
/home/pc108/miniforge3/envs/lerobotx/bin/python \
  -m lerobot.scripts.visualize_dataset \
  --repo-id=local/piper-franka-pick \
  --root=data/piper-franka-pick \
  --episode-index=0
```

### 3. 训练 ACT

RTX 3090 可以先用下面的配置做一次短跑验证：

```bash
/home/pc108/miniforge3/envs/lerobotx/bin/lerobot-train \
  --dataset.repo_id=local/piper-franka-pick \
  --dataset.root=data/piper-franka-pick \
  --policy.type=act \
  --policy.device=cuda \
  --policy.use_amp=true \
  --output_dir=outputs/train/piper_franka_act_smoke \
  --job_name=piper_franka_act_smoke \
  --batch_size=8 \
  --steps=1000 \
  --eval_freq=0 \
  --save_freq=500 \
  --num_workers=4
```

短跑能正常读取图像、产生 loss 并保存 checkpoint 后，再运行正式训练：

```bash
/home/pc108/miniforge3/envs/lerobotx/bin/lerobot-train \
  --dataset.repo_id=local/piper-franka-pick \
  --dataset.root=data/piper-franka-pick \
  --policy.type=act \
  --policy.device=cuda \
  --policy.use_amp=true \
  --output_dir=outputs/train/piper_franka_act \
  --job_name=piper_franka_act \
  --batch_size=8 \
  --steps=100000 \
  --eval_freq=0 \
  --save_freq=10000 \
  --num_workers=4
```

训练完成后，模型位于 `outputs/train/piper_franka_act/checkpoints/last/pretrained_model`。后续在 Franka 上做策略回放时，必须继续使用同一组相机名称、图像分辨率、ROS 2 topic 和 `control_mode=cartesian` 配置。
