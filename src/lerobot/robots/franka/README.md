# Franka ROS 2 client

`FrankaRobot` runs on the LeRobot computer. It does not connect to `libfranka`
directly; the hardware computer must run the package in
`ros2/lerobot_franka_bridge`.

Both Ubuntu 22.04 computers must source ROS 2 Humble and use the same DDS domain:

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=23
```

Check the bridge before starting LeRobot:

```bash
ros2 topic hz /franka/joint_state
ros2 topic echo --once /franka/end_pose
```

Joint control:

```bash
lerobot-record \
  --robot.type=franka \
  --robot.id=panda \
  --robot.control_mode=joint
```

Cartesian control uses metres and an XYZW quaternion:

```bash
lerobot-record \
  --robot.type=franka \
  --robot.id=panda \
  --robot.control_mode=cartesian
```

Observations always contain `j1.pos` through `j7.pos` and
`ee.x/ee.y/ee.z/ee.qx/ee.qy/ee.qz/ee.qw`. `send_action()` follows
`control_mode`; application code may call `send_joint_action()` or
`send_ee_pose()` explicitly. State and command topic names, timeouts, command
duration, base frame, QoS depth, and per-step safety limits are configurable.

The bridge uses latest-target point-to-point motion, not a hard real-time
streaming controller. Keep Franka Desk collision protection and emergency stop
available, begin with small target changes, and keep people outside the robot
workspace.
