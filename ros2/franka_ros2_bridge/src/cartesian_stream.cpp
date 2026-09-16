#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <franka/exception.h>
#include <franka/robot.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace {

static_assert(std::atomic<double>::is_always_lock_free,
              "The 1 kHz controller requires lock-free double atomics");
static_assert(std::atomic<std::uint64_t>::is_always_lock_free,
              "The 1 kHz controller requires lock-free sequence counters");

struct Pose {
  double p[3]{};
  double q[4]{0.0, 0.0, 0.0, 1.0};
};

struct MotionState {
  double linear_velocity[3]{};
  double linear_acceleration[3]{};
  double angular_velocity[3]{};
  double angular_acceleration[3]{};
};

struct RobotSnapshot {
  std::array<double, 7> joints{};
  Pose pose{};
};

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
    }
    for (std::size_t i = 0; i < 3; ++i) {
      position_[i].store(pose.p[i], std::memory_order_relaxed);
    }
    for (std::size_t i = 0; i < 4; ++i) {
      quaternion_[i].store(pose.q[i], std::memory_order_relaxed);
    }
    sequence_.fetch_add(1, std::memory_order_release);
  }

  bool try_load(RobotSnapshot* snapshot) const {
    const std::uint64_t before = sequence_.load(std::memory_order_acquire);
    if ((before & 1U) != 0U) {
      return false;
    }
    for (std::size_t i = 0; i < joints_.size(); ++i) {
      snapshot->joints[i] = joints_[i].load(std::memory_order_relaxed);
    }
    for (std::size_t i = 0; i < 3; ++i) {
      snapshot->pose.p[i] = position_[i].load(std::memory_order_relaxed);
    }
    for (std::size_t i = 0; i < 4; ++i) {
      snapshot->pose.q[i] = quaternion_[i].load(std::memory_order_relaxed);
    }
    const std::uint64_t after = sequence_.load(std::memory_order_acquire);
    return before == after && (after & 1U) == 0U;
  }

 private:
  std::atomic<std::uint64_t> sequence_{0};
  std::array<std::atomic<double>, 7> joints_{};
  std::array<std::atomic<double>, 3> position_{};
  std::array<std::atomic<double>, 4> quaternion_{};
};

void normalize_quat(double* q) {
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

double norm3(const double* values) {
  return std::sqrt(values[0] * values[0] + values[1] * values[1] + values[2] * values[2]);
}

double dot3(const double* first, const double* second) {
  return first[0] * second[0] + first[1] * second[1] + first[2] * second[2];
}

void clamp_norm3(double* values, double limit) {
  const double norm = norm3(values);
  if (norm <= limit || norm < 1e-12) {
    return;
  }
  const double scale = limit / norm;
  for (std::size_t i = 0; i < 3; ++i) {
    values[i] *= scale;
  }
}

void constrain_acceleration_near_velocity_limit(const double* velocity, double max_velocity,
                                                double max_jerk, double* acceleration) {
  const double speed = norm3(velocity);
  if (speed < 1e-12) {
    return;
  }
  const double direction[3] = {velocity[0] / speed, velocity[1] / speed, velocity[2] / speed};
  const double parallel_acceleration = dot3(acceleration, direction);
  if (parallel_acceleration <= 0.0) {
    return;
  }
  // Keep a small numerical guard below the hard velocity bound so the jerk
  // ramp reaches zero acceleration before floating-point/discrete-time clipping.
  const double guarded_velocity = 0.99 * max_velocity;
  const double remaining_velocity = std::max(0.0, guarded_velocity - speed);
  const double allowed_parallel_acceleration = std::sqrt(2.0 * max_jerk * remaining_velocity);
  if (parallel_acceleration <= allowed_parallel_acceleration) {
    return;
  }
  const double correction = parallel_acceleration - allowed_parallel_acceleration;
  for (std::size_t i = 0; i < 3; ++i) {
    acceleration[i] -= correction * direction[i];
  }
}

void quaternion_multiply(const double* first, const double* second, double* result) {
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

void quaternion_error_rotvec(const double* current, const double* target, double* rotvec) {
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

void update_derivatives(const double* error, double dt, double max_velocity, double max_acceleration,
                        double max_jerk, double* velocity, double* acceleration) {
  // Critically damped position feedback supplies a convergent acceleration
  // target; jerk limiting then makes changes to that acceleration continuous.
  constexpr double natural_frequency = 4.0;
  constexpr double position_gain = natural_frequency * natural_frequency;
  constexpr double velocity_gain = 2.0 * natural_frequency;
  double desired_acceleration[3]{};
  for (std::size_t i = 0; i < 3; ++i) {
    desired_acceleration[i] = position_gain * error[i] - velocity_gain * velocity[i];
  }
  clamp_norm3(desired_acceleration, max_acceleration);

  double jerk[3]{};
  double previous_acceleration[3]{};
  for (std::size_t i = 0; i < 3; ++i) {
    previous_acceleration[i] = acceleration[i];
    jerk[i] = (desired_acceleration[i] - acceleration[i]) / dt;
  }
  clamp_norm3(jerk, max_jerk);
  for (std::size_t i = 0; i < 3; ++i) {
    acceleration[i] += jerk[i] * dt;
  }
  clamp_norm3(acceleration, max_acceleration);
  constrain_acceleration_near_velocity_limit(velocity, max_velocity, max_jerk, acceleration);
  double acceleration_delta[3]{};
  for (std::size_t i = 0; i < 3; ++i) {
    acceleration_delta[i] = acceleration[i] - previous_acceleration[i];
  }
  clamp_norm3(acceleration_delta, max_jerk * dt);
  for (std::size_t i = 0; i < 3; ++i) {
    acceleration[i] = previous_acceleration[i] + acceleration_delta[i];
  }
  for (std::size_t i = 0; i < 3; ++i) {
    velocity[i] += acceleration[i] * dt;
  }
  clamp_norm3(velocity, max_velocity);
}

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

std::array<double, 16> pose_to_matrix(const Pose& pose) {
  double q[4] = {pose.q[0], pose.q[1], pose.q[2], pose.q[3]};
  normalize_quat(q);
  const double qx = q[0];
  const double qy = q[1];
  const double qz = q[2];
  const double qw = q[3];
  std::array<double, 16> m{};
  m[0] = 1.0 - 2.0 * (qy * qy + qz * qz);
  m[1] = 2.0 * (qx * qy + qz * qw);
  m[2] = 2.0 * (qx * qz - qy * qw);
  m[3] = 0.0;
  m[4] = 2.0 * (qx * qy - qz * qw);
  m[5] = 1.0 - 2.0 * (qx * qx + qz * qz);
  m[6] = 2.0 * (qy * qz + qx * qw);
  m[7] = 0.0;
  m[8] = 2.0 * (qx * qz + qy * qw);
  m[9] = 2.0 * (qy * qz - qx * qw);
  m[10] = 1.0 - 2.0 * (qx * qx + qy * qy);
  m[11] = 0.0;
  m[12] = pose.p[0];
  m[13] = pose.p[1];
  m[14] = pose.p[2];
  m[15] = 1.0;
  return m;
}

Pose rate_limit(const Pose& current, const Pose& target, double dt, double max_linear_velocity,
                double max_linear_acceleration, double max_linear_jerk, double max_angular_velocity,
                double max_angular_acceleration, double max_angular_jerk, MotionState* motion) {
  Pose out = current;
  const double translation_error[3] = {target.p[0] - current.p[0], target.p[1] - current.p[1],
                                       target.p[2] - current.p[2]};
  update_derivatives(translation_error, dt, max_linear_velocity, max_linear_acceleration,
                     max_linear_jerk, motion->linear_velocity, motion->linear_acceleration);
  double translation_step[3]{};
  for (std::size_t i = 0; i < 3; ++i) {
    translation_step[i] = motion->linear_velocity[i] * dt;
    out.p[i] += translation_step[i];
  }
  const double translation_distance = norm3(translation_error);
  if (dot3(translation_step, translation_error) > 0.0 && norm3(translation_step) >= translation_distance &&
      norm3(motion->linear_velocity) <= max_linear_acceleration * dt + max_linear_jerk * dt * dt &&
      norm3(motion->linear_acceleration) <= max_linear_jerk * dt) {
    for (std::size_t i = 0; i < 3; ++i) {
      out.p[i] = target.p[i];
      motion->linear_velocity[i] = 0.0;
      motion->linear_acceleration[i] = 0.0;
    }
  }

  double rotation_error[3]{};
  quaternion_error_rotvec(current.q, target.q, rotation_error);
  update_derivatives(rotation_error, dt, max_angular_velocity, max_angular_acceleration,
                     max_angular_jerk, motion->angular_velocity, motion->angular_acceleration);
  double rotation_step[3]{};
  for (std::size_t i = 0; i < 3; ++i) {
    rotation_step[i] = motion->angular_velocity[i] * dt;
  }
  const double step_angle = norm3(rotation_step);
  const double rotation_distance = norm3(rotation_error);
  if (dot3(rotation_step, rotation_error) > 0.0 && step_angle >= rotation_distance &&
      norm3(motion->angular_velocity) <= max_angular_acceleration * dt + max_angular_jerk * dt * dt &&
      norm3(motion->angular_acceleration) <= max_angular_jerk * dt) {
    for (std::size_t i = 0; i < 4; ++i) {
      out.q[i] = target.q[i];
    }
    for (std::size_t i = 0; i < 3; ++i) {
      motion->angular_velocity[i] = 0.0;
      motion->angular_acceleration[i] = 0.0;
    }
  } else if (step_angle > 1e-12) {
    const double half_angle = 0.5 * step_angle;
    const double scale = std::sin(half_angle) / step_angle;
    const double delta[4] = {rotation_step[0] * scale, rotation_step[1] * scale,
                             rotation_step[2] * scale, std::cos(half_angle)};
    quaternion_multiply(delta, current.q, out.q);
    normalize_quat(out.q);
  }
  return out;
}

void set_default_collision(franka::Robot* robot) {
  robot->setCollisionBehavior(
      {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}}, {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}},
      {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}}, {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}},
      {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}}, {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}},
      {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}}, {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}});
}

class CartesianStreamer {
 public:
  CartesianStreamer(std::string ip, double max_linear_velocity, double max_linear_acceleration,
                    double max_linear_jerk, double max_angular_velocity,
                    double max_angular_acceleration, double max_angular_jerk)
      : ip_(std::move(ip)), max_linear_velocity_(max_linear_velocity),
        max_linear_acceleration_(max_linear_acceleration), max_linear_jerk_(max_linear_jerk),
        max_angular_velocity_(max_angular_velocity), max_angular_acceleration_(max_angular_acceleration),
        max_angular_jerk_(max_angular_jerk) {
    if (ip_.empty()) {
      throw std::invalid_argument("robot ip must not be empty");
    }
    if (!(max_linear_velocity_ > 0.0) || !(max_linear_acceleration_ > 0.0) ||
        !(max_linear_jerk_ > 0.0) || !(max_angular_velocity_ > 0.0) ||
        !(max_angular_acceleration_ > 0.0) || !(max_angular_jerk_ > 0.0)) {
      throw std::invalid_argument("velocity, acceleration, and jerk limits must be positive");
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
    const franka::RobotState state = robot_->readOnce();
    matrix_to_pose(state.O_T_EE, &commanded_);
    active_target_ = commanded_;
    target_.store(commanded_);
    state_.store(state, commanded_);
    motion_ = MotionState{};
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
    normalize_quat(pose.q);
    check_error();
    if (!running()) {
      throw std::runtime_error("cartesian stream is not running");
    }
    target_.store(pose);
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
      robot_->control([this](const franka::RobotState& state, franka::Duration period) {
        return on_control(state, period);
      });
    } catch (...) {
      {
        std::lock_guard<std::mutex> lock(error_mu_);
        error_ = std::current_exception();
      }
      has_error_.store(true, std::memory_order_release);
    }
    running_.store(false, std::memory_order_release);
  }

  franka::CartesianPose on_control(const franka::RobotState& state, franka::Duration period) {
    if (!control_started_.exchange(true, std::memory_order_acq_rel)) {
      running_.store(true, std::memory_order_release);
    }
    Pose measured;
    matrix_to_pose(state.O_T_EE, &measured);
    state_.store(state, measured);
    Pose latest_target;
    if (target_.try_load(&latest_target)) {
      active_target_ = latest_target;
    }

    if (stop_requested_.load(std::memory_order_acquire)) {
      return franka::MotionFinished(franka::CartesianPose(pose_to_matrix(commanded_)));
    }

    double dt = period.toSec();
    if (dt <= 0.0 || dt > 0.01) {
      dt = 0.001;
    }
    commanded_ = rate_limit(commanded_, active_target_, dt, max_linear_velocity_,
                            max_linear_acceleration_, max_linear_jerk_, max_angular_velocity_,
                            max_angular_acceleration_, max_angular_jerk_, &motion_);
    return franka::CartesianPose(pose_to_matrix(commanded_));
  }

  std::string ip_;
  double max_linear_velocity_;
  double max_linear_acceleration_;
  double max_linear_jerk_;
  double max_angular_velocity_;
  double max_angular_acceleration_;
  double max_angular_jerk_;
  std::unique_ptr<franka::Robot> robot_;
  std::thread thread_;
  Pose commanded_{};
  Pose active_target_{};
  MotionState motion_{};
  AtomicPoseBuffer target_;
  AtomicRobotStateBuffer state_;
  std::mutex error_mu_;
  std::exception_ptr error_{};
  std::atomic<bool> running_{false};
  std::atomic<bool> control_started_{false};
  std::atomic<bool> stop_requested_{false};
  std::atomic<bool> has_error_{false};
};

}  // namespace

PYBIND11_MODULE(cartesian_stream, m) {
  py::class_<CartesianStreamer>(m, "CartesianStreamer")
      .def(py::init<std::string, double, double, double, double, double, double>(), py::arg("ip"),
           py::arg("max_linear_velocity") = 0.35, py::arg("max_linear_acceleration") = 1.0,
           py::arg("max_linear_jerk") = 5.0, py::arg("max_angular_velocity") = 1.2,
           py::arg("max_angular_acceleration") = 2.0, py::arg("max_angular_jerk") = 10.0)
      .def("start", &CartesianStreamer::start)
      .def("stop", &CartesianStreamer::stop)
      .def("set_target", &CartesianStreamer::set_target)
      .def("get_state", &CartesianStreamer::get_state)
      .def("running", &CartesianStreamer::running)
      .def("check_error", &CartesianStreamer::check_error);
}
