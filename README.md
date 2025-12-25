# LeRobot_use

这是一个使用LeRobot框架的机器人开发项目

## 构建说明

```bash
conda create -y -n lerobotx python=3.10 && conda activate lerobotx

conda install ffmpeg -c conda-forge

cd ~/lerobot
pip install -e ".[feetech]"

pip install scipy==1.15.3
```

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
├── tests
│   ├── artifacts
│   ├── async_inference
│   ├── cameras
│   ├── configs
│   ├── conftest.py
│   ├── datasets
│   ├── envs
│   ├── fixtures
│   ├── __init__.py
│   ├── mocks
│   ├── motors
│   ├── optim
│   ├── plugins
│   ├── policies
│   ├── processor
│   ├── rl
│   ├── robots
│   ├── teleoperators
│   ├── test_available.py
│   ├── test_control_robot.py
│   ├── transport
│   ├── utils
│   └── utils.py
└── third_party
    └── CrpRobotPy
```

