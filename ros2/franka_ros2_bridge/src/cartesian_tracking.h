#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>

#include <franka/rate_limiting.h>
#include <ruckig/ruckig.hpp>

namespace franka_bridge {

constexpr double kControlPeriod = 0.001;
constexpr double kPi = 3.14159265358979323846;
using Twist = std::array<double, 6>;

struct Pose {
  double p[3]{};
  double q[4]{0.0, 0.0, 0.0, 1.0};
};

inline void normalize_quat(double* q) {
  const double norm = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  if (norm < 1e-12) {
    q[0] = 0.0;
    q[1] = 0.0;
    q[2] = 0.0;
    q[3] = 1.0;
    return;
  }
  q[0] /= norm;
  q[1] /= norm;
  q[2] /= norm;
  q[3] /= norm;
}

inline double norm3(const double* values) {
  return std::sqrt(values[0] * values[0] + values[1] * values[1] + values[2] * values[2]);
}

inline void quaternion_multiply(const double* first, const double* second, double* result) {
  const double x1 = first[0];
  const double y1 = first[1];
  const double z1 = first[2];
  const double w1 = first[3];
  const double x2 = second[0];
  const double y2 = second[1];
  const double z2 = second[2];
  const double w2 = second[3];
  result[0] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2;
  result[1] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2;
  result[2] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2;
  result[3] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2;
}

inline void quaternion_error_rotvec(const double* current, const double* target, double* rotvec) {
  const double conjugate[4] = {-current[0], -current[1], -current[2], current[3]};
  double error[4]{};
  quaternion_multiply(target, conjugate, error);
  normalize_quat(error);
  if (error[3] < 0.0) {
    for (double& value : error) {
      value = -value;
    }
  }
  const double sin_half_angle = norm3(error);
  if (sin_half_angle < 1e-12) {
    rotvec[0] = rotvec[1] = rotvec[2] = 0.0;
    return;
  }
  const double angle = 2.0 * std::atan2(sin_half_angle, std::max(0.0, error[3]));
  const double scale = angle / sin_half_angle;
  rotvec[0] = error[0] * scale;
  rotvec[1] = error[1] * scale;
  rotvec[2] = error[2] * scale;
}

inline double quaternion_distance(const double* first, const double* second) {
  const double dot = std::abs(first[0] * second[0] + first[1] * second[1] +
                              first[2] * second[2] + first[3] * second[3]);
  return 2.0 * std::acos(std::clamp(dot, 0.0, 1.0));
}

struct MotionLimits {
  double velocity;
  double acceleration;
  double jerk;
};

struct DynamicScale {
  double velocity{1.0};
  double acceleration{1.0};
  double jerk{1.0};
};

// Pure control math: no robot, Python, ROS, locks, or heap allocation in step().
// Every step starts from the robot's accepted motion state, so joint saturation
// or lost FCI packets cannot make a virtual trajectory run ahead of the arm.
class AbsolutePoseTracker {
 public:
  AbsolutePoseTracker(MotionLimits linear, MotionLimits angular, double frequency,
                      double linear_deadband, double angular_deadband,
                      double startup_scale, double startup_ramp_sec)
      : linear_(linear), angular_(angular), linear_deadband_(linear_deadband),
        angular_deadband_(angular_deadband), startup_scale_(startup_scale),
        startup_ramp_sec_(startup_ramp_sec) {
    translation_input_.minimum_duration = 1.0 / (2.0 * kPi * frequency);
    rotation_input_.minimum_duration = translation_input_.minimum_duration;
  }

  void reset(const Pose& current) {
    target_ = current;
    raw_target_ = current;
    received_ = false;
    elapsed_ = 0.0;
    translation_.reset();
    rotation_.reset();
    synchronization_fallbacks_ = 0;
    target_change_ = {};
  }

  void set_target(const Pose& pose) {
    // Noise rejection is per channel; a rotating master must not reintroduce
    // translation noise. The first absolute target is never rebased or skipped.
    raw_target_ = pose;
    const double delta[3] = {pose.p[0] - target_.p[0], pose.p[1] - target_.p[1],
                             pose.p[2] - target_.p[2]};
    if (!received_ || (norm3(delta) > 1e-12 && norm3(delta) >= linear_deadband_)) {
      std::copy(delta, delta + 3, target_change_.begin());
      std::copy(pose.p, pose.p + 3, target_.p);
    }
    double rotation_delta[3]{};
    quaternion_error_rotvec(target_.q, pose.q, rotation_delta);
    if (!received_ || (norm3(rotation_delta) > 1e-12 && norm3(rotation_delta) >= angular_deadband_)) {
      std::copy(rotation_delta, rotation_delta + 3, target_change_.begin() + 3);
      std::copy(pose.q, pose.q + 4, target_.q);
    }
    received_ = true;
  }

  const Pose& target() const { return target_; }
  const Pose& raw_target() const { return raw_target_; }
  unsigned synchronization_fallbacks() const { return synchronization_fallbacks_; }
  int startup_state() const {
    return !received_ ? 0 : (elapsed_ < startup_ramp_sec_ ? 1 : 2);
  }
  double motion_scale() const {
    const double progress = std::clamp(elapsed_ / startup_ramp_sec_, 0.0, 1.0);
    return startup_scale_ + (1.0 - startup_scale_) * progress * progress * (3.0 - 2.0 * progress);
  }

  Twist axis_allocation(const Pose& accepted_pose, const Twist& velocity,
                        const Twist& acceleration) const {
    double rotation_error[3]{};
    quaternion_error_rotvec(accepted_pose.q, target_.q, rotation_error);
    Twist weights{};
    constexpr double horizon = 0.15;
    for (std::size_t i = 0; i < 6; ++i) {
      const double error = i < 3 ? target_.p[i] - accepted_pose.p[i] : rotation_error[i - 3];
      // Reserve capacity for braking axes that are moving even if their new
      // position error is zero. Latch target changes so allocation does not
      // collapse as the arm approaches a stationary goal.
      weights[i] = std::max(std::abs(error), std::abs(target_change_[i])) + horizon * std::abs(velocity[i]) +
                   .5 * horizon * horizon * std::abs(acceleration[i]);
    }
    for (std::size_t offset : {0U, 3U}) {
      const double deadband = offset == 0 ? linear_deadband_ : angular_deadband_;
      const double demand = std::max(norm3(weights.data() + offset), std::max(2.0 * deadband, 1e-6));
      for (std::size_t i = offset; i < offset + 3; ++i) {
        // Changing Jacobians and discrete sampling need correction capacity
        // even on nominally idle axes. A 1% floor caused drift and slow tails.
        weights[i] = std::max(.10, weights[i] / demand);
      }
      const double total = std::max(1.0, norm3(weights.data() + offset));
      for (std::size_t i = offset; i < offset + 3; ++i) weights[i] /= total;
    }
    return weights;
  }

  Twist step(const Pose& accepted_pose, const Twist& accepted_velocity,
             const Twist& accepted_acceleration, DynamicScale dynamics = {}) {
    if (!received_) {
      return {};
    }
    // Advance only one command period, even after a delayed callback. New
    // targets are accepted throughout this time ramp, with no arrival gate.
    const double scale = motion_scale();
    double rotation_error[3]{};
    quaternion_error_rotvec(accepted_pose.q, target_.q, rotation_error);
    const double translation_error[3] = {target_.p[0] - accepted_pose.p[0],
                                         target_.p[1] - accepted_pose.p[1],
                                         target_.p[2] - accepted_pose.p[2]};
    for (std::size_t offset : {0U, 3U}) {
      const double error = norm3(offset == 0 ? translation_error : rotation_error);
      if (error < 1e-6 && norm3(accepted_velocity.data() + offset) < 1e-6 &&
          norm3(accepted_acceleration.data() + offset) < 1e-5) {
        std::fill(target_change_.begin() + offset, target_change_.begin() + offset + 3, 0.0);
      }
    }
    for (std::size_t i = 0; i < 3; ++i) {
      translation_input_.current_position[i] = 0.0;
      translation_input_.target_position[i] = target_.p[i] - accepted_pose.p[i];
      rotation_input_.current_position[i] = 0.0;
      // A fresh base-frame tangent at each 1 ms step avoids Euler wrapping.
      rotation_input_.target_position[i] = rotation_error[i];
      // Matrix products leave ~1e-16 residues on stationary axes. Those are
      // numerical zero, but can make a synchronized zero-distance plan ill-conditioned.
      translation_input_.current_velocity[i] = numerical_zero(accepted_velocity[i], 1e-10);
      translation_input_.current_acceleration[i] = numerical_zero(accepted_acceleration[i], 1e-8);
      rotation_input_.current_velocity[i] = numerical_zero(accepted_velocity[i + 3], 1e-10);
      rotation_input_.current_acceleration[i] = numerical_zero(accepted_acceleration[i + 3], 1e-8);
    }
    const auto allocation = axis_allocation(accepted_pose, accepted_velocity, accepted_acceleration);
    set_limits(translation_input_, linear_, scale, dynamics, allocation.data());
    set_limits(rotation_input_, angular_, scale, dynamics, allocation.data() + 3);
    update(translation_, translation_input_, translation_output_, "translation");
    update(rotation_, rotation_input_, rotation_output_, "rotation");

    Twist requested{};
    for (std::size_t i = 0; i < 3; ++i) {
      // FCI ddq_d is a discrete velocity difference. Using new_velocity here
      // while reinitializing from that difference would halve the planned jerk
      // and delay braking. Command the planned end-of-step acceleration instead.
      requested[i] = accepted_velocity[i] + kControlPeriod * translation_output_.new_acceleration[i];
      requested[i + 3] = accepted_velocity[i + 3] + kControlPeriod * rotation_output_.new_acceleration[i];
    }
    elapsed_ = std::min(startup_ramp_sec_, elapsed_ + kControlPeriod);
    return franka::limitRate(scale * linear_.velocity, scale * linear_.acceleration,
                             scale * linear_.jerk, scale * angular_.velocity,
                             scale * angular_.acceleration, scale * angular_.jerk,
                             requested, accepted_velocity, accepted_acceleration);
  }

 private:
  static double numerical_zero(double value, double epsilon) {
    return std::abs(value) < epsilon ? 0.0 : value;
  }

  static void set_limits(ruckig::InputParameter<3>& input, MotionLimits limits, double scale,
                         DynamicScale dynamics, const double* allocation) {
    // Allocation has norm <= 1, so vector limits hold without dividing an
    // isolated X/Y/Z motion by sqrt(3) or reserving full speed for idle axes.
    for (std::size_t i = 0; i < 3; ++i) {
      input.max_velocity[i] = scale * allocation[i] * limits.velocity * dynamics.velocity;
      input.max_acceleration[i] = scale * allocation[i] * limits.acceleration * dynamics.acceleration;
      input.max_jerk[i] = scale * allocation[i] * limits.jerk * dynamics.jerk;
    }
  }

  void update(ruckig::Ruckig<3>& generator, const ruckig::InputParameter<3>& input,
              ruckig::OutputParameter<3>& output, const char* channel) {
    auto result = generator.update(input, output);
    if (result == ruckig::Result::ErrorSynchronizationCalculation) {
      // A shared arrival time may be numerically infeasible at a boundary.
      // Independent axes keep the same targets and derivative limits.
      auto independent = input;
      independent.synchronization = ruckig::Synchronization::None;
      generator.reset();
      result = generator.update(independent, output);
      ++synchronization_fallbacks_;
    }
    if (result < 0) {
      throw std::runtime_error(std::string("Cartesian ") + channel +
                               " trajectory failed (Ruckig 0.15.3): " +
                               std::to_string(static_cast<int>(result)) + " input=" + input.to_string());
    }
  }

  MotionLimits linear_;
  MotionLimits angular_;
  double linear_deadband_;
  double angular_deadband_;
  double startup_scale_;
  double startup_ramp_sec_;
  double elapsed_{0.0};
  bool received_{false};
  unsigned synchronization_fallbacks_{0};
  Pose target_{};
  Pose raw_target_{};
  Twist target_change_{};
  ruckig::Ruckig<3> translation_{kControlPeriod};
  ruckig::Ruckig<3> rotation_{kControlPeriod};
  ruckig::InputParameter<3> translation_input_;
  ruckig::InputParameter<3> rotation_input_;
  ruckig::OutputParameter<3> translation_output_;
  ruckig::OutputParameter<3> rotation_output_;
};

}  // namespace franka_bridge
