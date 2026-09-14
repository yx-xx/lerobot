#include <array>
#include <atomic>
#include <cmath>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <franka/exception.h>
#include <franka/robot.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace {

struct Pose {
  double p[3]{};
  double q[4]{0.0, 0.0, 0.0, 1.0};
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

Pose rate_limit(const Pose& current, const Pose& target, double dt, double vmax, double wmax) {
  Pose out = current;
  const double dp[3] = {target.p[0] - current.p[0], target.p[1] - current.p[1],
                        target.p[2] - current.p[2]};
  const double dist = std::sqrt(dp[0] * dp[0] + dp[1] * dp[1] + dp[2] * dp[2]);
  const double max_step = vmax * dt;
  if (dist > 1e-12) {
    const double scale = std::min(dist, max_step) / dist;
    out.p[0] = current.p[0] + dp[0] * scale;
    out.p[1] = current.p[1] + dp[1] * scale;
    out.p[2] = current.p[2] + dp[2] * scale;
  } else {
    out.p[0] = target.p[0];
    out.p[1] = target.p[1];
    out.p[2] = target.p[2];
  }

  double q_from[4] = {current.q[0], current.q[1], current.q[2], current.q[3]};
  double q_to[4] = {target.q[0], target.q[1], target.q[2], target.q[3]};
  normalize_quat(q_from);
  normalize_quat(q_to);
  double dot = q_from[0] * q_to[0] + q_from[1] * q_to[1] + q_from[2] * q_to[2] + q_from[3] * q_to[3];
  if (dot < 0.0) {
    q_to[0] = -q_to[0];
    q_to[1] = -q_to[1];
    q_to[2] = -q_to[2];
    q_to[3] = -q_to[3];
    dot = -dot;
  }
  dot = std::min(1.0, std::max(-1.0, dot));
  const double angle = std::acos(dot);
  const double max_angle = wmax * dt;
  if (angle < 1e-8 || max_angle >= angle) {
    out.q[0] = q_to[0];
    out.q[1] = q_to[1];
    out.q[2] = q_to[2];
    out.q[3] = q_to[3];
  } else {
    const double t = max_angle / angle;
    const double theta = angle * t;
    const double sin_angle = std::sin(angle);
    const double s0 = std::sin((1.0 - t) * angle) / sin_angle;
    const double s1 = std::sin(theta) / sin_angle;
    out.q[0] = s0 * q_from[0] + s1 * q_to[0];
    out.q[1] = s0 * q_from[1] + s1 * q_to[1];
    out.q[2] = s0 * q_from[2] + s1 * q_to[2];
    out.q[3] = s0 * q_from[3] + s1 * q_to[3];
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

}  // namespace

class CartesianStreamer {
 public:
  CartesianStreamer(std::string ip, double max_linear_velocity, double max_angular_velocity)
      : ip_(std::move(ip)), vmax_(max_linear_velocity), wmax_(max_angular_velocity) {
    if (ip_.empty()) {
      throw std::invalid_argument("robot ip must not be empty");
    }
    if (!(vmax_ > 0.0) || !(wmax_ > 0.0)) {
      throw std::invalid_argument("velocity limits must be positive");
    }
  }

  ~CartesianStreamer() { stop(); }

  void start() {
    if (running_.load()) {
      return;
    }
    robot_ = std::make_unique<franka::Robot>(ip_);
    robot_->automaticErrorRecovery();
    set_default_collision(robot_.get());
    const franka::RobotState state = robot_->readOnce();
    {
      std::lock_guard<std::mutex> lock(mu_);
      matrix_to_pose(state.O_T_EE, &commanded_);
      desired_ = commanded_;
      measured_ = commanded_;
      q_.assign(state.q.begin(), state.q.end());
      has_state_.store(true);
    }
    running_.store(true);
    thread_ = std::thread([this] { loop(); });
  }

  void stop() {
    running_.store(false);
    if (thread_.joinable()) {
      thread_.join();
    }
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
    std::lock_guard<std::mutex> lock(mu_);
    desired_ = pose;
  }

  py::tuple get_state() {
    std::lock_guard<std::mutex> lock(mu_);
    if (!has_state_.load()) {
      throw std::runtime_error("cartesian stream has no robot state yet");
    }
    return py::make_tuple(
        std::vector<double>(q_.begin(), q_.end()),
        std::vector<double>{measured_.p[0], measured_.p[1], measured_.p[2]},
        std::vector<double>{measured_.q[0], measured_.q[1], measured_.q[2], measured_.q[3]});
  }

  bool running() const { return running_.load(); }

 private:
  void loop() {
    try {
      robot_->control([this](const franka::RobotState& state, franka::Duration period) {
        return on_control(state, period);
      });
    } catch (...) {
      running_.store(false);
      std::lock_guard<std::mutex> lock(mu_);
      error_ = std::current_exception();
    }
  }

  franka::CartesianPose on_control(const franka::RobotState& state, franka::Duration period) {
    Pose desired;
    Pose commanded;
    {
      std::lock_guard<std::mutex> lock(mu_);
      q_.assign(state.q.begin(), state.q.end());
      matrix_to_pose(state.O_T_EE, &measured_);
      has_state_.store(true);
      desired = desired_;
      commanded = commanded_;
    }

    if (!running_.load()) {
      return franka::MotionFinished(franka::CartesianPose(state.O_T_EE));
    }

    double dt = period.toSec();
    if (dt <= 0.0 || dt > 0.01) {
      dt = 0.001;
    }
    commanded = rate_limit(commanded, desired, dt, vmax_, wmax_);
    {
      std::lock_guard<std::mutex> lock(mu_);
      commanded_ = commanded;
    }
    return franka::CartesianPose(pose_to_matrix(commanded));
  }

  std::string ip_;
  double vmax_;
  double wmax_;
  std::unique_ptr<franka::Robot> robot_;
  std::thread thread_;
  std::mutex mu_;
  Pose desired_{};
  Pose commanded_{};
  Pose measured_{};
  std::vector<double> q_ = std::vector<double>(7, 0.0);
  std::atomic<bool> running_{false};
  std::atomic<bool> has_state_{false};
  std::exception_ptr error_{};
};

PYBIND11_MODULE(cartesian_stream, m) {
  py::class_<CartesianStreamer>(m, "CartesianStreamer")
      .def(py::init<std::string, double, double>(), py::arg("ip"),
           py::arg("max_linear_velocity") = 0.35, py::arg("max_angular_velocity") = 1.2)
      .def("start", &CartesianStreamer::start)
      .def("stop", &CartesianStreamer::stop)
      .def("set_target", &CartesianStreamer::set_target)
      .def("get_state", &CartesianStreamer::get_state)
      .def("running", &CartesianStreamer::running);
}
