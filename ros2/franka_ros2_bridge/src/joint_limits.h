#pragma once

#include "cartesian_tracking.h"

namespace franka_bridge {

// These stay below libfranka's limits so the bridge, rather than the robot
// reflex, controls the joint-space dynamics produced by resolved-rate IK.
constexpr std::array<double, 7> kJointVelocityLimits{
    1.40, 1.40, 1.40, 1.40, 1.80, 1.80, 1.80};
constexpr std::array<double, 7> kJointAccelerationLimits{
    6.0, 3.0, 4.0, 5.0, 6.0, 8.0, 8.0};
constexpr std::array<double, 7> kJointJerkLimits{
    300.0, 150.0, 200.0, 250.0, 300.0, 400.0, 400.0};

inline double jacobian_damping(double minimum_singular_value) {
  const double proximity = std::clamp((.10 - minimum_singular_value) / .10, 0.0, 1.0);
  // A permanent damping term attenuates the entire accepted velocity again
  // every millisecond. In a healthy posture that acts as artificial drag.
  return .12 * proximity * proximity;
}

// Bound the simultaneous axis allocation in each joint. Unused directions do
// not reserve their full configured velocity/acceleration/jerk.
template <typename Inverse>
DynamicScale allocate_joint_dynamics(const Inverse& inverse, const Twist& allocation,
                                     MotionLimits linear_limits, MotionLimits angular_limits) {
  DynamicScale scale;
  for (std::size_t joint = 0; joint < 7; ++joint) {
    double linear = 0.0, angular = 0.0;
    for (std::size_t axis = 0; axis < 3; ++axis) {
      linear += std::abs(inverse(joint, axis)) * allocation[axis];
      angular += std::abs(inverse(joint, axis + 3)) * allocation[axis + 3];
    }
    const double velocity = linear * linear_limits.velocity + angular * angular_limits.velocity;
    const double acceleration = linear * linear_limits.acceleration + angular * angular_limits.acceleration;
    const double jerk = linear * linear_limits.jerk + angular * angular_limits.jerk;
    if (velocity > 0.0) scale.velocity = std::min(scale.velocity, .8 * kJointVelocityLimits[joint] / velocity);
    if (acceleration > 0.0) scale.acceleration = std::min(scale.acceleration, .7 * kJointAccelerationLimits[joint] / acceleration);
    if (jerk > 0.0) scale.jerk = std::min(scale.jerk, .5 * kJointJerkLimits[joint] / jerk);
  }
  return scale;
}

}  // namespace franka_bridge
