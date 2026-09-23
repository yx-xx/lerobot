#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <pthread.h>
#include <sched.h>
#include <franka/exception.h>
#include <franka/gripper.h>
#include <franka/model.h>
#include <franka/rate_limiting.h>
#include <franka/robot.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cartesian_tracking.h"
#include "joint_limits.h"

namespace py = pybind11;

namespace {

static_assert(std::atomic<double>::is_always_lock_free,
              "The 1 kHz controller requires lock-free double atomics");
static_assert(std::atomic<std::uint64_t>::is_always_lock_free,
              "The 1 kHz controller requires lock-free sequence counters");

using namespace franka_bridge;
using Jacobian = Eigen::Matrix<double, 6, 7>;
using PseudoInverse = Eigen::Matrix<double, 7, 6>;
using Vector6 = Eigen::Matrix<double, 6, 1>;
using Vector7 = Eigen::Matrix<double, 7, 1>;

struct RobotSnapshot {
  std::array<double, 7> joints{};
  std::array<double, 7> desired_joint_velocities{};
  std::array<double, 7> desired_joint_accelerations{};
  std::array<double, 6> commanded_cartesian_velocity{};
  std::array<double, 6> commanded_cartesian_acceleration{};
  Pose pose{};
  double control_command_success_rate{};
};

constexpr std::array<double, 7> kJointLowerLimits{
    -2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973};
constexpr std::array<double, 7> kJointUpperLimits{
    2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973};
constexpr double kJointLimitBuffer = 0.05;
constexpr double kJointLimitSlowdownRange = 0.15;

class AtomicPoseBuffer {
 public:
  void store(const Pose& pose) {
    sequence_.fetch_add(1, std::memory_order_acq_rel);
    for (std::size_t i = 0; i < 3; ++i) {
      position_[i].store(pose.p[i], std::memory_order_relaxed);
    }
    for (std::size_t i = 0; i < 4; ++i) {
      quaternion_[i].store(pose.q[i], std::memory_order_relaxed);
    }
    sequence_.fetch_add(1, std::memory_order_release);
  }

  bool try_load(Pose* pose) const {
    const std::uint64_t before = sequence_.load(std::memory_order_acquire);
    if ((before & 1U) != 0U) {
      return false;
    }
    for (std::size_t i = 0; i < 3; ++i) {
      pose->p[i] = position_[i].load(std::memory_order_relaxed);
    }
    for (std::size_t i = 0; i < 4; ++i) {
      pose->q[i] = quaternion_[i].load(std::memory_order_relaxed);
    }
    const std::uint64_t after = sequence_.load(std::memory_order_acquire);
    return before == after && (after & 1U) == 0U;
  }

 private:
  std::atomic<std::uint64_t> sequence_{0};
  std::array<std::atomic<double>, 3> position_{};
  std::array<std::atomic<double>, 4> quaternion_{};
};

class AtomicRobotStateBuffer {
 public:
  void store(const franka::RobotState& state, const Pose& pose) {
    sequence_.fetch_add(1, std::memory_order_acq_rel);
    for (std::size_t i = 0; i < joints_.size(); ++i) {
      joints_[i].store(state.q[i], std::memory_order_relaxed);
      desired_joint_velocities_[i].store(state.dq_d[i], std::memory_order_relaxed);
      desired_joint_accelerations_[i].store(state.ddq_d[i], std::memory_order_relaxed);
    }
    for (std::size_t i = 0; i < commanded_cartesian_velocity_.size(); ++i) {
      commanded_cartesian_velocity_[i].store(state.O_dP_EE_c[i], std::memory_order_relaxed);
      commanded_cartesian_acceleration_[i].store(state.O_ddP_EE_c[i],
                                                 std::memory_order_relaxed);
    }
    for (std::size_t i = 0; i < 3; ++i) {
      position_[i].store(pose.p[i], std::memory_order_relaxed);
    }
    for (std::size_t i = 0; i < 4; ++i) {
      quaternion_[i].store(pose.q[i], std::memory_order_relaxed);
    }
    control_command_success_rate_.store(state.control_command_success_rate,
                                        std::memory_order_relaxed);
    sequence_.fetch_add(1, std::memory_order_release);
  }

  bool try_load(RobotSnapshot* snapshot) const {
    const std::uint64_t before = sequence_.load(std::memory_order_acquire);
    if ((before & 1U) != 0U) {
      return false;
    }
    for (std::size_t i = 0; i < joints_.size(); ++i) {
      snapshot->joints[i] = joints_[i].load(std::memory_order_relaxed);
      snapshot->desired_joint_velocities[i] =
          desired_joint_velocities_[i].load(std::memory_order_relaxed);
      snapshot->desired_joint_accelerations[i] =
          desired_joint_accelerations_[i].load(std::memory_order_relaxed);
    }
    for (std::size_t i = 0; i < commanded_cartesian_velocity_.size(); ++i) {
      snapshot->commanded_cartesian_velocity[i] =
          commanded_cartesian_velocity_[i].load(std::memory_order_relaxed);
      snapshot->commanded_cartesian_acceleration[i] =
          commanded_cartesian_acceleration_[i].load(std::memory_order_relaxed);
    }
    for (std::size_t i = 0; i < 3; ++i) {
      snapshot->pose.p[i] = position_[i].load(std::memory_order_relaxed);
    }
    for (std::size_t i = 0; i < 4; ++i) {
      snapshot->pose.q[i] = quaternion_[i].load(std::memory_order_relaxed);
    }
    snapshot->control_command_success_rate =
        control_command_success_rate_.load(std::memory_order_relaxed);
    const std::uint64_t after = sequence_.load(std::memory_order_acquire);
    return before == after && (after & 1U) == 0U;
  }

 private:
  std::atomic<std::uint64_t> sequence_{0};
  std::array<std::atomic<double>, 7> joints_{};
  std::array<std::atomic<double>, 7> desired_joint_velocities_{};
  std::array<std::atomic<double>, 7> desired_joint_accelerations_{};
  std::array<std::atomic<double>, 6> commanded_cartesian_velocity_{};
  std::array<std::atomic<double>, 6> commanded_cartesian_acceleration_{};
  std::array<std::atomic<double>, 3> position_{};
  std::array<std::atomic<double>, 4> quaternion_{};
  std::atomic<double> control_command_success_rate_{0.0};
};

void matrix_to_pose(const std::array<double, 16>& m, Pose* pose) {
  const double r00 = m[0];
  const double r10 = m[1];
  const double r20 = m[2];
  const double r01 = m[4];
  const double r11 = m[5];
  const double r21 = m[6];
  const double r02 = m[8];
  const double r12 = m[9];
  const double r22 = m[10];
  const double trace = r00 + r11 + r22;
  double qx = 0.0;
  double qy = 0.0;
  double qz = 0.0;
  double qw = 1.0;
  if (trace > 0.0) {
    const double scale = std::sqrt(trace + 1.0) * 2.0;
    qw = 0.25 * scale;
    qx = (r21 - r12) / scale;
    qy = (r02 - r20) / scale;
    qz = (r10 - r01) / scale;
  } else if (r00 > r11 && r00 > r22) {
    const double scale = std::sqrt(1.0 + r00 - r11 - r22) * 2.0;
    qw = (r21 - r12) / scale;
    qx = 0.25 * scale;
    qy = (r01 + r10) / scale;
    qz = (r02 + r20) / scale;
  } else if (r11 > r22) {
    const double scale = std::sqrt(1.0 + r11 - r00 - r22) * 2.0;
    qw = (r02 - r20) / scale;
    qx = (r01 + r10) / scale;
    qy = 0.25 * scale;
    qz = (r12 + r21) / scale;
  } else {
    const double scale = std::sqrt(1.0 + r22 - r00 - r11) * 2.0;
    qw = (r10 - r01) / scale;
    qx = (r02 + r20) / scale;
    qy = (r12 + r21) / scale;
    qz = 0.25 * scale;
  }
  pose->p[0] = m[12];
  pose->p[1] = m[13];
  pose->p[2] = m[14];
  pose->q[0] = qx;
  pose->q[1] = qy;
  pose->q[2] = qz;
  pose->q[3] = qw;
  normalize_quat(pose->q);
}

void set_default_collision(franka::Robot* robot) {
  robot->setCollisionBehavior(
      {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}}, {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}},
      {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}}, {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}},
      {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}}, {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}},
      {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}}, {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}});
}

// Blocking gripper I/O lives behind GIL-releasing bindings. The ROS arm-state
// callback only reads a Python cache populated by a separate polling thread.
class GripperIO {
 public:
  explicit GripperIO(const std::string& ip) : gripper_(ip) {}
  double width() { return gripper_.readOnce().width; }
  bool move(double width, double speed) { return gripper_.move(width, speed); }

 private:
  franka::Gripper gripper_;
};

class CartesianStreamer {
 public:
  CartesianStreamer(std::string ip, double max_linear_velocity, double max_linear_acceleration,
                    double max_linear_jerk, double max_angular_velocity,
                    double max_angular_acceleration, double max_angular_jerk,
                    double tracking_frequency_hz, double linear_deadband,
                    double angular_deadband, bool lock_elbow, int control_cpu,
                    double initial_sync_scale, double startup_ramp_sec)
      : ip_(std::move(ip)), max_linear_velocity_(max_linear_velocity),
        max_linear_acceleration_(max_linear_acceleration), max_linear_jerk_(max_linear_jerk),
        max_angular_velocity_(max_angular_velocity), max_angular_acceleration_(max_angular_acceleration),
        max_angular_jerk_(max_angular_jerk),
        linear_deadband_(linear_deadband), angular_deadband_(angular_deadband),
        lock_elbow_(lock_elbow), control_cpu_(control_cpu),
        initial_sync_scale_(initial_sync_scale),
        tracker_({max_linear_velocity, max_linear_acceleration, max_linear_jerk},
                 {max_angular_velocity, max_angular_acceleration, max_angular_jerk},
                 tracking_frequency_hz, linear_deadband, angular_deadband,
                 initial_sync_scale, startup_ramp_sec) {
    if (ip_.empty()) {
      throw std::invalid_argument("robot ip must not be empty");
    }
    if (!(max_linear_velocity_ > 0.0) || !(max_linear_acceleration_ > 0.0) ||
        !(max_linear_jerk_ > 0.0) || !(max_angular_velocity_ > 0.0) ||
        !(max_angular_acceleration_ > 0.0) || !(max_angular_jerk_ > 0.0)) {
      throw std::invalid_argument("velocity, acceleration, and jerk limits must be positive");
    }
    if (!(tracking_frequency_hz >= 0.1 && tracking_frequency_hz <= 10.0)) {
      throw std::invalid_argument("tracking frequency must be in [0.1, 10] Hz");
    }
    if (!(linear_deadband_ >= 0.0 && linear_deadband_ <= 0.05) ||
        !(angular_deadband_ >= 0.0 && angular_deadband_ <= 0.5)) {
      throw std::invalid_argument("deadbands must be non-negative and within safe bounds");
    }
    if (control_cpu_ < -1 || control_cpu_ >= CPU_SETSIZE) {
      throw std::invalid_argument("control CPU must be -1 or a valid CPU index");
    }
    if (!(initial_sync_scale_ > 0.0 && initial_sync_scale_ <= 1.0)) {
      throw std::invalid_argument("initial sync scale must be in (0, 1]");
    }
    if (!(startup_ramp_sec > 0.0 && startup_ramp_sec <= 5.0)) {
      throw std::invalid_argument("startup ramp duration must be in (0, 5] seconds");
    }
  }

  ~CartesianStreamer() { stop(); }

  void start() {
    if (running()) {
      return;
    }
    if (thread_.joinable()) {
      thread_.join();
    }
    clear_error();
    robot_ = std::make_unique<franka::Robot>(ip_);
    robot_->automaticErrorRecovery();
    set_default_collision(robot_.get());
    model_ = std::make_unique<franka::Model>(robot_->loadModel());
    const franka::RobotState state = robot_->readOnce();
    matrix_to_pose(state.O_T_EE, &commanded_);
    active_target_ = commanded_;
    target_.store(commanded_);
    tracker_.reset(commanded_);
    previous_jacobian_valid_ = false;
    state_.store(state, commanded_);
    posture_reference_ = state.q_d;
    commanded_cartesian_velocity_ = {};
    target_received_.store(false, std::memory_order_release);
    target_message_count_.store(0, std::memory_order_relaxed);
    last_target_message_ns_.store(0, std::memory_order_relaxed);
    last_period_ms_.store(0.0, std::memory_order_relaxed);
    max_period_ms_.store(0.0, std::memory_order_relaxed);
    delayed_callback_count_.store(0, std::memory_order_relaxed);
    control_cpu_current_.store(-1, std::memory_order_relaxed);
    control_cpu_migration_count_.store(0, std::memory_order_relaxed);
    minimum_singular_value_.store(-1.0, std::memory_order_relaxed);
    joint_velocity_scale_.store(1.0, std::memory_order_relaxed);
    planner_velocity_scale_.store(1.0, std::memory_order_relaxed);
    planner_acceleration_scale_.store(1.0, std::memory_order_relaxed);
    planner_jerk_scale_.store(1.0, std::memory_order_relaxed);
    trajectory_sync_fallbacks_.store(0, std::memory_order_relaxed);
    position_error_.store(0.0, std::memory_order_relaxed);
    orientation_error_.store(0.0, std::memory_order_relaxed);
    singular_value_cycle_ = 0;
    startup_state_.store(0, std::memory_order_relaxed);
    first_command_sent_ = false;
    stop_requested_.store(false, std::memory_order_release);
    control_started_.store(false, std::memory_order_release);
    running_.store(false, std::memory_order_release);
    thread_ = std::thread([this] { loop(); });

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (std::chrono::steady_clock::now() < deadline) {
      if (control_started_.load(std::memory_order_acquire)) {
        return;
      }
      if (has_error_.load(std::memory_order_acquire)) {
        thread_.join();
        check_error();
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    stop_requested_.store(true, std::memory_order_release);
    if (thread_.joinable()) {
      thread_.join();
    }
    running_.store(false, std::memory_order_release);
    throw std::runtime_error("libfranka control callback did not start within 5 seconds");
  }

  void stop() {
    stop_requested_.store(true, std::memory_order_release);
    if (thread_.joinable()) {
      thread_.join();
    }
    running_.store(false, std::memory_order_release);
    model_.reset();
    robot_.reset();
  }

  void set_target(double x, double y, double z, double qx, double qy, double qz, double qw) {
    Pose pose;
    pose.p[0] = x;
    pose.p[1] = y;
    pose.p[2] = z;
    pose.q[0] = qx;
    pose.q[1] = qy;
    pose.q[2] = qz;
    pose.q[3] = qw;
    for (double value : pose.p) {
      if (!std::isfinite(value)) {
        throw std::invalid_argument("target position must be finite");
      }
    }
    double quaternion_norm = 0.0;
    for (double value : pose.q) {
      if (!std::isfinite(value)) {
        throw std::invalid_argument("target quaternion must be finite");
      }
      quaternion_norm += value * value;
    }
    if (!std::isfinite(quaternion_norm) || quaternion_norm < 1e-12) {
      throw std::invalid_argument("target quaternion must have a finite nonzero norm");
    }
    normalize_quat(pose.q);
    check_error();
    if (!running()) {
      throw std::runtime_error("cartesian stream is not running");
    }
    const auto now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now().time_since_epoch())
                            .count();
    last_target_message_ns_.store(static_cast<std::uint64_t>(now_ns),
                                  std::memory_order_relaxed);
    target_message_count_.fetch_add(1, std::memory_order_relaxed);
    target_.store(pose);
    target_received_.store(true, std::memory_order_release);
  }

  py::tuple get_state() {
    check_error();
    if (!running()) {
      throw std::runtime_error("cartesian stream is not running");
    }
    RobotSnapshot snapshot;
    for (int attempt = 0; attempt < 1000; ++attempt) {
      if (state_.try_load(&snapshot)) {
        return py::make_tuple(
            std::vector<double>(snapshot.joints.begin(), snapshot.joints.end()),
            std::vector<double>{snapshot.pose.p[0], snapshot.pose.p[1], snapshot.pose.p[2]},
            std::vector<double>{snapshot.pose.q[0], snapshot.pose.q[1], snapshot.pose.q[2],
                                snapshot.pose.q[3]});
      }
      std::this_thread::yield();
    }
    throw std::runtime_error("could not read a consistent cartesian stream state");
  }

  py::dict get_diagnostics() {
    check_error();
    RobotSnapshot snapshot;
    for (int attempt = 0; attempt < 1000; ++attempt) {
      if (state_.try_load(&snapshot)) {
        py::dict diagnostics;
        diagnostics["control_command_success_rate"] = snapshot.control_command_success_rate;
        diagnostics["last_period_ms"] = last_period_ms_.load(std::memory_order_relaxed);
        diagnostics["max_period_ms"] = max_period_ms_.load(std::memory_order_relaxed);
        diagnostics["delayed_callbacks"] =
            delayed_callback_count_.load(std::memory_order_relaxed);
        diagnostics["control_cpu"] = control_cpu_current_.load(std::memory_order_relaxed);
        diagnostics["control_cpu_migrations"] =
            control_cpu_migration_count_.load(std::memory_order_relaxed);
        diagnostics["elbow_locked"] = lock_elbow_;
        diagnostics["startup_state"] =
            startup_state_.load(std::memory_order_relaxed);
        diagnostics["minimum_singular_value"] =
            minimum_singular_value_.load(std::memory_order_relaxed);
        diagnostics["joint_velocity_scale"] =
            joint_velocity_scale_.load(std::memory_order_relaxed);
        diagnostics["planner_velocity_scale"] = planner_velocity_scale_.load(std::memory_order_relaxed);
        diagnostics["planner_acceleration_scale"] = planner_acceleration_scale_.load(std::memory_order_relaxed);
        diagnostics["planner_jerk_scale"] = planner_jerk_scale_.load(std::memory_order_relaxed);
        diagnostics["trajectory_sync_fallbacks"] = trajectory_sync_fallbacks_.load(std::memory_order_relaxed);
        diagnostics["position_error"] = position_error_.load(std::memory_order_relaxed);
        diagnostics["orientation_error"] = orientation_error_.load(std::memory_order_relaxed);
        const std::uint64_t last_target_ns =
            last_target_message_ns_.load(std::memory_order_relaxed);
        const auto now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count();
        diagnostics["target_messages"] = target_message_count_.load(std::memory_order_relaxed);
        diagnostics["target_age_ms"] =
            last_target_ns == 0
                ? -1.0
                : static_cast<double>(static_cast<std::uint64_t>(now_ns) - last_target_ns) / 1e6;
        return diagnostics;
      }
      std::this_thread::yield();
    }
    throw std::runtime_error("could not read consistent Cartesian stream diagnostics");
  }

  bool running() const {
    return running_.load(std::memory_order_acquire) && !has_error_.load(std::memory_order_acquire);
  }

  void check_error() {
    if (!has_error_.load(std::memory_order_acquire)) {
      return;
    }
    std::lock_guard<std::mutex> lock(error_mu_);
    if (error_ != nullptr) {
      std::rethrow_exception(error_);
    }
    throw std::runtime_error("cartesian stream stopped with an unknown error");
  }

 private:
  void clear_error() {
    std::lock_guard<std::mutex> lock(error_mu_);
    error_ = nullptr;
    has_error_.store(false, std::memory_order_release);
  }

  void loop() {
    try {
      if (control_cpu_ >= 0) {
        cpu_set_t cpu_set;
        CPU_ZERO(&cpu_set);
        CPU_SET(control_cpu_, &cpu_set);
        const int affinity_result =
            pthread_setaffinity_np(pthread_self(), sizeof(cpu_set), &cpu_set);
        if (affinity_result != 0) {
          throw std::runtime_error("failed to pin the Cartesian control thread to CPU " +
                                   std::to_string(control_cpu_) + ": " +
                                   std::strerror(affinity_result));
        }
      }
      robot_->control(
          [this](const franka::RobotState& state, franka::Duration period) {
            return on_control(state, period);
          },
          // The bridge applies conservative joint-space limits after resolved-rate
          // IK. Keep libfranka's limiter enabled as the final independent guard.
          franka::ControllerMode::kJointImpedance, true, franka::kMaxCutoffFrequency);
    } catch (...) {
      const std::exception_ptr original_error = std::current_exception();
      std::exception_ptr reported_error = original_error;
      try {
        std::rethrow_exception(original_error);
      } catch (const std::exception& exception) {
        RobotSnapshot snapshot;
        std::ostringstream details;
        details << exception.what();
        if (state_.try_load(&snapshot)) {
          details << "\nlast_control_state: success_rate="
                  << snapshot.control_command_success_rate << " q=[";
          for (std::size_t i = 0; i < snapshot.joints.size(); ++i) {
            details << (i == 0 ? "" : ",") << snapshot.joints[i];
          }
          details << "] dq_d=[";
          for (std::size_t i = 0; i < snapshot.desired_joint_velocities.size(); ++i) {
            details << (i == 0 ? "" : ",") << snapshot.desired_joint_velocities[i];
          }
          details << "] ddq_d=[";
          for (std::size_t i = 0; i < snapshot.desired_joint_accelerations.size(); ++i) {
            details << (i == 0 ? "" : ",") << snapshot.desired_joint_accelerations[i];
          }
          details << "] O_dP_EE_c=[";
          for (std::size_t i = 0; i < snapshot.commanded_cartesian_velocity.size(); ++i) {
            details << (i == 0 ? "" : ",") << snapshot.commanded_cartesian_velocity[i];
          }
          details << "] O_ddP_EE_c=[";
          for (std::size_t i = 0; i < snapshot.commanded_cartesian_acceleration.size(); ++i) {
            details << (i == 0 ? "" : ",") << snapshot.commanded_cartesian_acceleration[i];
          }
          details << "] commanded_xyz=[" << commanded_.p[0] << "," << commanded_.p[1] << ","
                  << commanded_.p[2] << "] target_xyz=[" << active_target_.p[0] << ","
                  << active_target_.p[1] << "," << active_target_.p[2] << "] commanded_xyzw=["
                  << commanded_.q[0] << "," << commanded_.q[1] << "," << commanded_.q[2] << ","
                  << commanded_.q[3] << "] target_xyzw=[" << active_target_.q[0] << ","
                  << active_target_.q[1] << "," << active_target_.q[2] << ","
                  << active_target_.q[3] << "] last_period_ms="
                  << last_period_ms_.load(std::memory_order_relaxed) << " max_period_ms="
                  << max_period_ms_.load(std::memory_order_relaxed) << " delayed_callbacks="
                  << delayed_callback_count_.load(std::memory_order_relaxed)
                  << " minimum_singular_value="
                  << minimum_singular_value_.load(std::memory_order_relaxed)
                  << " joint_velocity_scale="
                  << joint_velocity_scale_.load(std::memory_order_relaxed)
                  << " position_error=" << position_error_.load(std::memory_order_relaxed)
                  << " orientation_error=" << orientation_error_.load(std::memory_order_relaxed)
                  << " elbow_locked=" << lock_elbow_
                  << " startup_state="
                  << startup_state_.load(std::memory_order_relaxed)
                  << " target_received="
                  << target_received_.load(std::memory_order_relaxed)
                  << " target_messages="
                  << target_message_count_.load(std::memory_order_relaxed);
          const std::uint64_t last_target_ns =
              last_target_message_ns_.load(std::memory_order_relaxed);
          if (last_target_ns == 0) {
            details << " target_age_ms=-1";
          } else {
            const auto now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    std::chrono::steady_clock::now().time_since_epoch())
                                    .count();
            details << " target_age_ms="
                    << static_cast<double>(static_cast<std::uint64_t>(now_ns) - last_target_ns) /
                           1e6;
          }
        }
        reported_error = std::make_exception_ptr(std::runtime_error(details.str()));
      } catch (...) {
      }
      {
        std::lock_guard<std::mutex> lock(error_mu_);
        error_ = reported_error;
      }
      has_error_.store(true, std::memory_order_release);
    }
    running_.store(false, std::memory_order_release);
  }

  PseudoInverse inverse_jacobian(const Jacobian& jacobian) {
    using Matrix6 = Eigen::Matrix<double, 6, 6>;
    const Matrix6 gram = jacobian * jacobian.transpose();
    double minimum_singular_value =
        minimum_singular_value_.load(std::memory_order_relaxed);
    if (singular_value_cycle_ == 0) {
      const Eigen::JacobiSVD<Jacobian> singular_value_solver(jacobian);
      minimum_singular_value = singular_value_solver.singularValues().minCoeff();
      minimum_singular_value_.store(minimum_singular_value, std::memory_order_relaxed);
    }
    singular_value_cycle_ = (singular_value_cycle_ + 1) % 10;

    const double damping = jacobian_damping(minimum_singular_value);
    const Matrix6 regularized = gram + damping * damping * Matrix6::Identity();
    const Eigen::LDLT<Matrix6> decomposition(regularized);
    if (decomposition.info() != Eigen::Success) {
      throw std::runtime_error("failed to solve damped Cartesian Jacobian");
    }
    return jacobian.transpose() * decomposition.solve(Matrix6::Identity());
  }

  std::array<double, 7> resolved_joint_velocity(const franka::RobotState& state,
                                                const std::array<double, 6>& twist,
                                                const Jacobian& jacobian,
                                                const PseudoInverse& pseudo_inverse) {
    using Matrix7 = Eigen::Matrix<double, 7, 7>;
    const Eigen::Map<const Vector6> desired_twist(twist.data());
    Vector7 desired_joint_velocity = pseudo_inverse * desired_twist;

    if (lock_elbow_) {
      Vector7 posture_velocity = Vector7::Zero();
      posture_velocity(2) = -0.8 * (state.q[2] - posture_reference_[2]);
      desired_joint_velocity +=
          (Matrix7::Identity() - pseudo_inverse * jacobian) * posture_velocity;
    }

    double velocity_scale = 1.0;
    for (std::size_t i = 0; i < 7; ++i) {
      const double speed = std::abs(desired_joint_velocity(static_cast<Eigen::Index>(i)));
      if (speed > kJointVelocityLimits[i]) {
        velocity_scale = std::min(velocity_scale, kJointVelocityLimits[i] / speed);
      }

      const double velocity = desired_joint_velocity(static_cast<Eigen::Index>(i));
      const double lower_boundary = kJointLowerLimits[i] + kJointLimitBuffer;
      const double upper_boundary = kJointUpperLimits[i] - kJointLimitBuffer;
      if (velocity > 0.0 && state.q[i] > upper_boundary - kJointLimitSlowdownRange) {
        velocity_scale = std::min(
            velocity_scale,
            std::clamp((upper_boundary - state.q[i]) / kJointLimitSlowdownRange, 0.0, 1.0));
      } else if (velocity < 0.0 && state.q[i] < lower_boundary + kJointLimitSlowdownRange) {
        velocity_scale = std::min(
            velocity_scale,
            std::clamp((state.q[i] - lower_boundary) / kJointLimitSlowdownRange, 0.0, 1.0));
      }
    }
    joint_velocity_scale_.store(velocity_scale, std::memory_order_relaxed);

    std::array<double, 7> desired{};
    for (std::size_t i = 0; i < desired.size(); ++i) {
      desired[i] = velocity_scale * desired_joint_velocity(static_cast<Eigen::Index>(i));
    }
    return franka::limitRate(kJointVelocityLimits, kJointAccelerationLimits, kJointJerkLimits,
                             desired, state.dq_d, state.ddq_d);
  }

  franka::JointVelocities on_control(const franka::RobotState& state, franka::Duration period) {
    const int current_cpu = sched_getcpu();
    const int previous_cpu = control_cpu_current_.exchange(current_cpu, std::memory_order_relaxed);
    if (previous_cpu >= 0 && current_cpu >= 0 && previous_cpu != current_cpu) {
      control_cpu_migration_count_.fetch_add(1, std::memory_order_relaxed);
    }
    const double period_ms = 1000.0 * period.toSec();
    last_period_ms_.store(period_ms, std::memory_order_relaxed);
    double observed_max = max_period_ms_.load(std::memory_order_relaxed);
    while (period_ms > observed_max &&
           !max_period_ms_.compare_exchange_weak(observed_max, period_ms,
                                                 std::memory_order_relaxed)) {
    }
    if (period_ms > 1.5) {
      delayed_callback_count_.fetch_add(1, std::memory_order_relaxed);
    }
    Pose measured;
    matrix_to_pose(state.O_T_EE, &measured);
    commanded_ = measured;
    state_.store(state, measured);

    const std::array<double, 7> stopped_velocity{};
    if (!target_received_.load(std::memory_order_acquire)) {
      active_target_ = commanded_;
      commanded_cartesian_velocity_ = {};
      position_error_.store(0.0, std::memory_order_relaxed);
      orientation_error_.store(0.0, std::memory_order_relaxed);

      if (first_command_sent_) {
        if (!control_started_.exchange(true, std::memory_order_acq_rel)) {
          running_.store(true, std::memory_order_release);
        }
      } else {
        first_command_sent_ = true;
      }

      if (stop_requested_.load(std::memory_order_acquire)) {
        return franka::MotionFinished(franka::JointVelocities(stopped_velocity));
      }
      return franka::JointVelocities(stopped_velocity);
    }

    if (!control_started_.exchange(true, std::memory_order_acq_rel)) {
      running_.store(true, std::memory_order_release);
    }
    Pose latest_target;
    if (target_.try_load(&latest_target)) {
      tracker_.set_target(latest_target);
      active_target_ = tracker_.raw_target();
    }

    if (stop_requested_.load(std::memory_order_acquire)) {
      const auto braking = franka::limitRate(kJointVelocityLimits, kJointAccelerationLimits,
                                             kJointJerkLimits, stopped_velocity,
                                             state.dq_d, state.ddq_d);
      bool stopped = true;
      for (std::size_t i = 0; i < braking.size(); ++i) {
        stopped = stopped && std::abs(braking[i]) < 1e-5 && std::abs(state.ddq_d[i]) < 1e-3;
      }
      return stopped ? franka::MotionFinished(franka::JointVelocities(braking))
                     : franka::JointVelocities(braking);
    }

    const double translation_error[3] = {active_target_.p[0] - commanded_.p[0],
                                         active_target_.p[1] - commanded_.p[1],
                                         active_target_.p[2] - commanded_.p[2]};
    position_error_.store(norm3(translation_error), std::memory_order_relaxed);
    orientation_error_.store(quaternion_distance(commanded_.q, active_target_.q),
                             std::memory_order_relaxed);

    // q_d, dq_d and ddq_d describe commands actually accepted by the robot.
    // Use their Jacobian (including its change) to seed the next Cartesian step.
    const auto jacobian_array = model_->zeroJacobian(
        franka::Frame::kEndEffector, state.q_d, state.F_T_EE, state.EE_T_K);
    const Eigen::Map<const Jacobian> jacobian(jacobian_array.data());
    const Eigen::Map<const Vector7> accepted_dq(state.dq_d.data());
    const Eigen::Map<const Vector7> accepted_ddq(state.ddq_d.data());
    const Vector6 velocity = jacobian * accepted_dq;
    Vector6 curvature = Vector6::Zero();
    Jacobian jacobian_derivative = Jacobian::Zero();
    if (previous_jacobian_valid_ && period.toSec() > 0.0) {
      jacobian_derivative = (jacobian - previous_jacobian_) / period.toSec();
      curvature = jacobian_derivative * accepted_dq;
    }
    // ddq_d is a discrete difference. At a regular 1 ms period this equals
    // (J*dq - previous_J*previous_dq)/dt, rather than counting the change of J
    // twice on the acceleration increment. After packet loss, retain the
    // robot's latest ddq_d and estimate Jdot over the observed interval.
    const Vector6 acceleration =
        (jacobian - kControlPeriod * jacobian_derivative) * accepted_ddq + curvature;
    previous_jacobian_ = jacobian;
    previous_jacobian_valid_ = true;
    Twist accepted_velocity{}, accepted_acceleration{};
    Eigen::Map<Vector6>(accepted_velocity.data()) = velocity;
    Eigen::Map<Vector6>(accepted_acceleration.data()) = acceleration;
    // Keep position, velocity and acceleration on the same accepted command
    // timeline. The measured pose includes the impedance servo's delay; using
    // it with dq_d/ddq_d makes the planner brake late and hunt around the goal.
    Pose accepted_pose;
    matrix_to_pose(model_->pose(franka::Frame::kEndEffector, state.q_d,
                                state.F_T_EE, state.EE_T_K), &accepted_pose);
    const PseudoInverse inverse = inverse_jacobian(jacobian);
    const Twist allocation = tracker_.axis_allocation(accepted_pose, accepted_velocity,
                                                      accepted_acceleration);
    const DynamicScale dynamics = allocate_joint_dynamics(
        inverse, allocation,
        {max_linear_velocity_, max_linear_acceleration_, max_linear_jerk_},
        {max_angular_velocity_, max_angular_acceleration_, max_angular_jerk_});
    planner_velocity_scale_.store(dynamics.velocity, std::memory_order_relaxed);
    planner_acceleration_scale_.store(dynamics.acceleration, std::memory_order_relaxed);
    planner_jerk_scale_.store(dynamics.jerk, std::memory_order_relaxed);
    commanded_cartesian_velocity_ = tracker_.step(accepted_pose, accepted_velocity,
                                                  accepted_acceleration, dynamics);
    trajectory_sync_fallbacks_.store(tracker_.synchronization_fallbacks(), std::memory_order_relaxed);
    Twist ik_velocity = commanded_cartesian_velocity_;
    // The next Cartesian velocity includes Jdot*dq from motion of the frame.
    // Inverting the current J must subtract that contribution once; otherwise
    // feeding it back as additional joint acceleration injects it every 1 ms.
    for (std::size_t i = 0; i < 6; ++i) ik_velocity[i] -= kControlPeriod * curvature[i];
    const auto joint_velocity = resolved_joint_velocity(state, ik_velocity,
                                                        jacobian, inverse);
    startup_state_.store(tracker_.startup_state(), std::memory_order_relaxed);
    return franka::JointVelocities(joint_velocity);
  }

  std::string ip_;
  double max_linear_velocity_;
  double max_linear_acceleration_;
  double max_linear_jerk_;
  double max_angular_velocity_;
  double max_angular_acceleration_;
  double max_angular_jerk_;
  double linear_deadband_;
  double angular_deadband_;
  bool lock_elbow_;
  int control_cpu_;
  double initial_sync_scale_;
  AbsolutePoseTracker tracker_;
  Jacobian previous_jacobian_ = Jacobian::Zero();
  bool previous_jacobian_valid_{false};
  std::unique_ptr<franka::Robot> robot_;
  std::unique_ptr<franka::Model> model_;
  std::thread thread_;
  Pose commanded_{};
  Pose active_target_{};
  std::array<double, 7> posture_reference_{};
  std::array<double, 6> commanded_cartesian_velocity_{};
  AtomicPoseBuffer target_;
  AtomicRobotStateBuffer state_;
  std::mutex error_mu_;
  std::exception_ptr error_{};
  std::atomic<bool> running_{false};
  std::atomic<bool> control_started_{false};
  std::atomic<bool> stop_requested_{false};
  std::atomic<bool> has_error_{false};
  std::atomic<bool> target_received_{false};
  std::atomic<double> last_period_ms_{0.0};
  std::atomic<double> max_period_ms_{0.0};
  std::atomic<std::uint64_t> delayed_callback_count_{0};
  std::atomic<int> control_cpu_current_{-1};
  std::atomic<std::uint64_t> control_cpu_migration_count_{0};
  std::atomic<std::uint64_t> target_message_count_{0};
  std::atomic<std::uint64_t> last_target_message_ns_{0};
  std::atomic<int> startup_state_{0};
  std::atomic<double> minimum_singular_value_{-1.0};
  std::atomic<double> joint_velocity_scale_{1.0};
  std::atomic<double> planner_velocity_scale_{1.0};
  std::atomic<double> planner_acceleration_scale_{1.0};
  std::atomic<double> planner_jerk_scale_{1.0};
  std::atomic<unsigned> trajectory_sync_fallbacks_{0};
  std::atomic<double> position_error_{0.0};
  std::atomic<double> orientation_error_{0.0};
  bool first_command_sent_{false};
  std::uint32_t singular_value_cycle_{0};
};

}  // namespace

PYBIND11_MODULE(cartesian_stream, m) {
  py::class_<GripperIO>(m, "GripperIO")
      .def(py::init<const std::string&>(), py::call_guard<py::gil_scoped_release>())
      .def("width", &GripperIO::width, py::call_guard<py::gil_scoped_release>())
      .def("move", &GripperIO::move, py::call_guard<py::gil_scoped_release>());

  py::class_<CartesianStreamer>(m, "CartesianStreamer")
      .def(py::init<std::string, double, double, double, double, double, double, double, double,
                    double, bool, int, double, double>(),
           py::arg("ip"), py::arg("max_linear_velocity") = 0.40,
           py::arg("max_linear_acceleration") = 2.0, py::arg("max_linear_jerk") = 12.0,
           py::arg("max_angular_velocity") = 0.80,
           py::arg("max_angular_acceleration") = 4.0, py::arg("max_angular_jerk") = 25.0,
           py::arg("tracking_frequency_hz") = 5.0, py::arg("linear_deadband") = 0.001,
           py::arg("angular_deadband") = 0.010, py::arg("lock_elbow") = true,
           py::arg("control_cpu") = 4, py::arg("initial_sync_scale") = 0.60,
           py::arg("startup_ramp_sec") = 0.30)
      .def("start", &CartesianStreamer::start)
      .def("stop", &CartesianStreamer::stop)
      .def("set_target", &CartesianStreamer::set_target)
      .def("get_state", &CartesianStreamer::get_state)
      .def("get_diagnostics", &CartesianStreamer::get_diagnostics)
      .def("running", &CartesianStreamer::running)
      .def("check_error", &CartesianStreamer::check_error);
}
