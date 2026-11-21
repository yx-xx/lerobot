#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <stdexcept>

#include "joint_limits.h"

namespace franka_bridge {

using CartesianJacobian = Eigen::Matrix<double, 6, 7>;
using CartesianInverse = Eigen::Matrix<double, 7, 6>;
using JointVector = Eigen::Matrix<double, 7, 1>;

// Fixed-size symmetric solve, including rank-deficient Cartesian tasks.
inline Eigen::Matrix3d task_gram_inverse(const Eigen::Matrix3d& gram, double damping) {
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> decomposition(gram);
  if (decomposition.info() != Eigen::Success) {
    throw std::runtime_error("failed to solve Cartesian task Jacobian");
  }
  Eigen::Vector3d gains;
  for (Eigen::Index i = 0; i < 3; ++i) {
    const double value = std::max(0.0, decomposition.eigenvalues()[i]);
    gains[i] = value > 1e-12 ? 1.0 / (value + damping * damping) : 0.0;
  }
  return decomposition.eigenvectors() * gains.asDiagonal() * decomposition.eigenvectors().transpose();
}

// Translation is the primary task. Damping a weak orientation direction must
// reduce that rotation rather than trade it for an unrequested XYZ movement.
inline CartesianInverse position_priority_inverse(const CartesianJacobian& jacobian,
                                                   double damping) {
  using TaskJacobian = Eigen::Matrix<double, 3, 7>;
  using TaskInverse = Eigen::Matrix<double, 7, 3>;
  using Matrix7 = Eigen::Matrix<double, 7, 7>;
  const TaskJacobian position = jacobian.topRows<3>();
  const TaskJacobian orientation = jacobian.bottomRows<3>();
  const TaskInverse position_inverse =
      position.transpose() * task_gram_inverse(position * position.transpose(), 0.0);
  const Matrix7 position_nullspace = Matrix7::Identity() - position_inverse * position;
  const TaskJacobian remaining_orientation = orientation * position_nullspace;
  const TaskInverse orientation_inverse = position_nullspace * remaining_orientation.transpose() *
      task_gram_inverse(remaining_orientation * remaining_orientation.transpose(), damping);
  CartesianInverse inverse;
  inverse.leftCols<3>() = position_inverse - orientation_inverse * orientation * position_inverse;
  inverse.rightCols<3>() = orientation_inverse;
  return inverse;
}

inline JointVector project_posture_velocity(const CartesianJacobian& jacobian,
                                            const JointVector& posture) {
  // I - J_damped^+ J is not a nullspace projector: near singularities it
  // injects task motion. The last column of this orthogonal Q is in ker(J),
  // regardless of the damping used by the primary Cartesian tasks.
  const Eigen::HouseholderQR<Eigen::Matrix<double, 7, 6>> decomposition(jacobian.transpose());
  const JointVector direction = decomposition.householderQ() * JointVector::Unit(6);
  return direction * direction.dot(posture);
}

class OrientationTargetLimiter {
 public:
  explicit OrientationTargetLimiter(MotionLimits angular = {.8, 4., 25.})
      : braking_horizon_(angular.velocity / angular.acceleration + angular.acceleration / angular.jerk) {}

  void reset() { limited_ = false; }
  bool limited() const { return limited_; }

  Pose constrain(const Pose& current, const Pose& target, const CartesianInverse& inverse,
                 const std::array<double, 7>& position,
                 const std::array<double, 7>& velocity) {
    double error[3]{};
    quaternion_error_rotvec(limited_ ? hold_.q : current.q, target.q, error);
    const JointVector direction = inverse.rightCols<3>() * Eigen::Map<const Eigen::Vector3d>(error);
    if (limited_) {
      bool still_outward = false;
      for (std::size_t i = 0; i < 7; ++i) {
        still_outward = still_outward ||
            (lower_[i] && position[i] < release_position_[i] && direction[i] < -1e-6) ||
            (upper_[i] && position[i] > release_position_[i] && direction[i] > 1e-6);
      }
      limited_ = still_outward;
    }
    if (!limited_) {
      for (std::size_t i = 0; i < 7; ++i) {
        // Brake before the downstream joint slowdown alters the Cartesian
        // direction. Reserve the configured angular braking horizon plus
        // 0.03 rad for discrete sampling and the change in Jacobian.
        const double braking = .03 + braking_horizon_ * std::abs(velocity[i]);
        const double lower = kJointLowerLimits[i] + kJointLimitBuffer + kJointLimitSlowdownRange + braking;
        const double upper = kJointUpperLimits[i] - kJointLimitBuffer - kJointLimitSlowdownRange - braking;
        lower_[i] = position[i] <= lower && direction[i] < -1e-6;
        upper_[i] = position[i] >= upper && direction[i] > 1e-6;
        release_position_[i] = position[i] + (lower_[i] ? .08 : -.08);
        limited_ = limited_ || lower_[i] || upper_[i];
      }
      if (limited_) hold_ = current;
    }
    Pose constrained = target;
    if (limited_) {
      // Hold one fixed feasible orientation while braking. Updating the hold
      // pose every millisecond would make the braking destination move away.
      std::copy(hold_.q, hold_.q + 4, constrained.q);
    }
    return constrained;
  }

 private:
  double braking_horizon_;
  bool limited_{false};
  Pose hold_{};
  std::array<bool, 7> lower_{}, upper_{};
  std::array<double, 7> release_position_{};
};

inline JointVector cartesian_joint_velocity(const CartesianJacobian& jacobian,
                                            const CartesianInverse& inverse,
                                            const JointVector& accepted_velocity,
                                            const Eigen::Matrix<double, 6, 1>& requested_twist,
                                            const JointVector& posture_velocity) {
  // Damping applies to the requested correction, not the whole accepted
  // velocity again every millisecond. Repeated attenuation creates artificial
  // acceleration that the Cartesian planner then tries to compensate.
  const JointVector task_velocity = accepted_velocity +
      inverse * (requested_twist - jacobian * accepted_velocity);
  return task_velocity + project_posture_velocity(jacobian, posture_velocity - accepted_velocity);
}

}  // namespace franka_bridge
