#include "cartesian_tracking.h"
#include "joint_limits.h"
#include <Eigen/Dense>

#include <cassert>
#include <cstdlib>
#include <deque>
#include <iostream>
#include <string>

using namespace franka_bridge;

namespace {
void require(bool condition, const char* description) {
  if (!condition) {
    throw std::runtime_error(description);
  }
}

Pose pose_at(double x, double y = -0.1, double z = 0.4, double angle = 0.0) {
  Pose pose{{x, y, z}, {0.0, 0.0, std::sin(angle / 2.0), std::cos(angle / 2.0)}};
  return pose;
}

double distance(const Pose& a, const Pose& b) {
  double difference[3] = {a.p[0] - b.p[0], a.p[1] - b.p[1], a.p[2] - b.p[2]};
  return norm3(difference);
}

struct Simulation {
  AbsolutePoseTracker tracker{{0.4, 2.0, 12.0}, {0.8, 4.0, 25.0}, 5.0, .001, .01, .6, .3};
  Pose pose = pose_at(.4);
  Pose measured_pose = pose;
  Twist velocity{}, acceleration{};
  double maximum_velocity = 0.0, maximum_acceleration = 0.0, maximum_jerk = 0.0;
  std::deque<Pose> feedback_history;

  Simulation() { tracker.reset(pose); }

  void step(bool downstream_limit = false, int feedback_delay = 0, DynamicScale limits = {}) {
    if (downstream_limit) limits = DynamicScale{.035 / .4, .4 / 2., 3. / 12.};
    feedback_history.push_back(pose);
    while (feedback_history.size() > static_cast<std::size_t>(feedback_delay + 1)) {
      feedback_history.pop_front();
    }
    measured_pose = feedback_history.front();
    // The real loop computes accepted FK from q_d, alongside dq_d and ddq_d.
    // Measured pose is delayed by the impedance servo and used for diagnostics.
    Twist next = tracker.step(pose, velocity, acceleration, limits);
    Twist next_acceleration{}, jerk{};
    if (downstream_limit) {
      // A restrictive downstream controller (e.g. joint speed/jerk saturation).
      // Feed back what it actually accepted rather than the planner output.
      next = franka::limitRate(.035, .4, 3.0, .8, 4.0, 25.0, next, velocity, acceleration);
    }
    for (std::size_t i = 0; i < 6; ++i) {
      require(std::isfinite(next[i]), "nonfinite command");
      next_acceleration[i] = (next[i] - velocity[i]) / kControlPeriod;
      jerk[i] = (next_acceleration[i] - acceleration[i]) / kControlPeriod;
    }
    maximum_velocity = std::max(maximum_velocity, norm3(next.data()));
    maximum_acceleration = std::max(maximum_acceleration, norm3(next_acceleration.data()));
    maximum_jerk = std::max(maximum_jerk, norm3(jerk.data()));
    require(norm3(next.data()) <= .400001, "translation velocity exceeded");
    require(norm3(next_acceleration.data()) <= 2.000001, "translation acceleration exceeded");
    if (norm3(jerk.data()) > 12.000001) {
      std::cerr << "jerk=" << norm3(jerk.data()) << " v=" << norm3(velocity.data())
                << " a=" << norm3(acceleration.data()) << " next_a=" << norm3(next_acceleration.data()) << "\n";
    }
    require(norm3(jerk.data()) <= 12.000001, "translation jerk exceeded");
    require(norm3(next.data() + 3) <= .800001, "rotation velocity exceeded");
    require(norm3(next_acceleration.data() + 3) <= 4.000001, "rotation acceleration exceeded");
    require(norm3(jerk.data() + 3) <= 25.000001, "rotation jerk exceeded");
    for (std::size_t i = 0; i < 3; ++i) {
      pose.p[i] += .5 * (velocity[i] + next[i]) * kControlPeriod;
    }
    double rotation[3]{};
    for (std::size_t i = 0; i < 3; ++i) {
      rotation[i] = .5 * (velocity[i + 3] + next[i + 3]) * kControlPeriod;
    }
    const double angle = norm3(rotation);
    if (angle > 1e-15) {
      double delta[4]{}, quaternion[4]{};
      for (std::size_t i = 0; i < 3; ++i) {
        delta[i] = rotation[i] * std::sin(angle / 2.0) / angle;
      }
      delta[3] = std::cos(angle / 2.0);
      quaternion_multiply(delta, pose.q, quaternion);
      std::copy(quaternion, quaternion + 4, pose.q);
      normalize_quat(pose.q);
    }
    velocity = next;
    acceleration = next_acceleration;
  }

  void settled(const Pose& target) const {
    require(distance(pose, tracker.target()) < .0001, "position failed to converge within 0.1 mm");
    require(quaternion_distance(pose.q, tracker.target().q) < .001, "orientation failed to converge");
    require(distance(pose, target) < .0011, "raw target position exceeds configured deadband");
    require(distance(measured_pose, target) < .0011, "delayed measured position did not settle");
    require(quaternion_distance(pose.q, target.q) < .011, "raw target rotation exceeds configured deadband");
    require(norm3(velocity.data()) < .0001, "translation did not stop");
    require(norm3(velocity.data() + 3) < .001, "rotation did not stop");
  }
};

void test_fixed() {
  for (double offset : {.0005, .005, .02, .05, .10}) {
    Simulation sim;
    const Pose target = pose_at(.4 + offset, -.1 + offset * .3, .4 - offset * .2, offset * 3.0);
    sim.tracker.set_target(target);
    require(distance(sim.tracker.target(), target) < 1e-12, "first target was rebased or suppressed");
    double overshoot = 0, settling = -1;
    for (int i = 0; i < 5000; ++i) {
      sim.step();
      overshoot = std::max(overshoot, sim.pose.p[0] - target.p[0]);
      if (settling < 0 && distance(sim.pose, target) < .001) settling = i * .001;
    }
    sim.settled(target);
    require(overshoot < .0003, "fixed target overshoot exceeds 0.3 mm");
    std::cout << "offset=" << offset << " settling_s=" << settling << " overshoot_m=" << overshoot
              << " v/a/j=" << sim.maximum_velocity << "/" << sim.maximum_acceleration
              << "/" << sim.maximum_jerk << "\n";
  }
}

void test_live_startup() {
  Simulation sim;
  sim.tracker.set_target(pose_at(.5));
  for (int i = 0; i < 30; ++i) sim.step();
  const Pose latest = pose_at(.2, .1, .3, -.2);
  sim.tracker.set_target(latest);
  require(distance(sim.tracker.target(), latest) < 1e-12, "startup latched the first target");
  for (int i = 0; i < 271; ++i) sim.step();
  require(sim.tracker.startup_state() == 2, "startup waits for arrival instead of elapsed ramp");
  require(distance(sim.pose, latest) > .01, "test must finish ramp while still moving");
  for (int i = 0; i < 5000; ++i) sim.step();
  sim.settled(latest);
}

void test_reversal(bool downstream_limit) {
  Simulation sim;
  Pose goal = pose_at(.45, -.08, .4, .3);
  for (int i = 0; i < 8000; ++i) {
    if (i == 80) goal = pose_at(.37, -.13, .4, -.2);
    if (i == 800) goal = pose_at(.42, -.10, .4, .1);
    if (i % 33 == 0) sim.tracker.set_target(goal);
    sim.step(downstream_limit);
  }
  sim.settled(goal);
}

void test_moving(bool drop_samples) {
  Simulation sim;
  Pose goal = sim.pose;
  double peak_error = 0;
  for (int i = 0; i < 6000; ++i) {
    double time = i * .001;
    if (i % 33 == 0 && (!drop_samples || i % 99 == 0 || i > 2000)) {
      goal = pose_at(.4 + .08 * std::min(time, 2.0), -.1, .4, .2 * std::min(time, 2.0));
      sim.tracker.set_target(goal);
    }
    sim.step();
    if (i > 600 && i < 1800) peak_error = std::max(peak_error, distance(sim.pose, goal));
  }
  sim.settled(goal);
  require(peak_error < .022, "30 Hz ramp tracking error exceeds 22 mm at 80 mm/s");
  std::cout << "peak_ramp_error_m=" << peak_error << "\n";
}

void test_rotation() {
  Simulation sim;
  // Mixed axis orientation near pi, followed by the equivalent negative quaternion.
  Pose goal = sim.pose;
  const double angle = 3.12;
  for (int i = 0; i < 3; ++i) goal.q[i] = std::sin(angle / 2) / std::sqrt(3.0);
  goal.q[3] = std::cos(angle / 2);
  for (int i = 0; i < 14000; ++i) {
    if (i % 33 == 0) {
      Pose sample = goal;
      if (i % 66 == 0) for (double& value : sample.q) value = -value;
      sim.tracker.set_target(sample);
    }
    sim.step();
  }
  sim.settled(goal);
}

void test_noise() {
  Simulation sim;
  Pose goal = sim.pose;
  sim.tracker.set_target(goal);
  for (int i = 0; i < 5000; ++i) {
    if (i % 33 == 0) {
      Pose sample = pose_at(.4 + .0003 * std::sin(i), -.1, .4, .003 * std::cos(i));
      sim.tracker.set_target(sample);
    }
    sim.step();
  }
  sim.settled(goal);
  require(sim.maximum_velocity < 1e-12, "sub-deadband stationary noise caused motion");
  Pose rotating = pose_at(.4003, -.1, .4, .2);
  sim.tracker.set_target(rotating);
  require(std::abs(sim.tracker.target().p[0] - .4) < 1e-12, "rotation released position noise");
  sim.tracker.reset(goal);
  require(sim.tracker.startup_state() == 0, "restart did not reset startup");
  sim.tracker.set_target(rotating);
  require(distance(sim.tracker.target(), rotating) < 1e-12, "restart rebased absolute target");
}

void test_circle() {
  Simulation sim;
  Pose goal = sim.pose;
  for (int i = 0; i < 7000; ++i) {
    if (i % 33 == 0) {
      const double phase = 2.0 * kPi * .5 * std::min(i * .001, 4.0);
      goal = pose_at(.4 + .02 * std::sin(phase), -.1 + .02 * (1.0 - std::cos(phase)),
                     .4 + .01 * std::sin(phase), .1 * std::sin(phase));
      sim.tracker.set_target(goal);
    }
    sim.step();
  }
  sim.settled(goal);
}

void test_changing_limits() {
  Simulation sim;
  Pose goal = pose_at(.46, -.08, .42, .15);
  sim.tracker.set_target(goal);
  for (int i = 0; i < 9000; ++i) {
    if (i == 300) {
      goal = pose_at(.38, -.12, .39, -.1);
      sim.tracker.set_target(goal);
    }
    // Enter and leave a restrictive region while still moving.
    // Planner bounds may shrink below current speed, but hardware limits stay
    // fixed while it brakes into the smaller envelope (no instantaneous clamp).
    sim.step(false, 0, i >= 200 && i < 2000 ? DynamicScale{.1, .3, .4} : DynamicScale{});
  }
  sim.settled(goal);
}

void test_feedback_delay() {
  for (int delay : {10, 30}) {
    Simulation sim;
    const Pose goal = pose_at(.43, -.09, .41, .15);
    sim.tracker.set_target(goal);
    for (int i = 0; i < 6000; ++i) sim.step(false, delay);
    sim.settled(goal);
  }
}

void test_stress() {
  Simulation sim;
  Pose goal = sim.pose;
  for (int i = 0; i < 50000; ++i) {
    if (i % 33 == 0) {
      const double time = std::min(i * .001, 45.0);
      goal = pose_at(.4 + .04 * std::sin(1.7 * time), -.1 + .05 * std::sin(2.3 * time),
                     .4 + .03 * std::sin(.9 * time), .4 * std::sin(2.1 * time));
      sim.tracker.set_target(goal);
    }
    sim.step();
  }
  sim.settled(goal);
  std::cout << "sync_fallbacks=" << sim.tracker.synchronization_fallbacks() << "\n";
}

// Panda modified-DH model, with the standard 0.1034 m hand offset. This is an
// independent kinematic plant; it uses the posture from the reported slow run.
struct PandaPlant {
  using J = Eigen::Matrix<double, 6, 7>;
  std::array<double, 7> q{-.00265755, -.446267, -.0483538, -2.59455, .0281065, 2.1841, .783124};
  std::array<double, 7> dq{}, ddq{};
  Pose pose{};
  J jacobian;
  void forward() {
    constexpr double a[7] = {0, 0, 0, .0825, -.0825, 0, .088};
    constexpr double d[7] = {.333, 0, .316, 0, .384, 0, 0};
    const double alpha[7] = {0, -kPi/2, kPi/2, kPi/2, -kPi/2, kPi/2, kPi/2};
    Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
    Eigen::Vector3d origins[7], axes[7];
    for (std::size_t i = 0; i < 7; ++i) {
      transform.rotate(Eigen::AngleAxisd(alpha[i], Eigen::Vector3d::UnitX()));
      transform.translate(Eigen::Vector3d(a[i], 0, 0));
      origins[i] = transform.translation();
      axes[i] = transform.linear().col(2);
      transform.rotate(Eigen::AngleAxisd(q[i], Eigen::Vector3d::UnitZ()));
      transform.translate(Eigen::Vector3d(0, 0, d[i]));
    }
    transform.translate(Eigen::Vector3d(0, 0, .2104));
    transform.rotate(Eigen::AngleAxisd(-kPi/4, Eigen::Vector3d::UnitZ()));
    const Eigen::Quaterniond quaternion(transform.linear());
    Eigen::Map<Eigen::Vector3d>(pose.p) = transform.translation();
    Eigen::Map<Eigen::Vector4d>(pose.q) = quaternion.coeffs();
    for (std::size_t i = 0; i < 7; ++i) {
      jacobian.col(i).head<3>() = axes[i].cross(transform.translation() - origins[i]);
      jacobian.col(i).tail<3>() = axes[i];
    }
  }
};

struct PandaSimulation {
  PandaPlant plant;
  AbsolutePoseTracker tracker;
  PandaPlant::J last_jacobian;
  double elbow, maximum_joint_speed = 0.0;

  explicit PandaSimulation(double deadband = .001)
      : tracker{{.4,2.,12.}, {.8,4.,25.}, 5., deadband, deadband == 0.0 ? 0.0 : .01, .6, .3} {
    plant.forward();
    tracker.reset(plant.pose);
    last_jacobian = plant.jacobian;
    elbow = plant.q[2];
  }

  void step() {
    plant.forward();
    const auto& jacobian = plant.jacobian;
    const double sigma = Eigen::JacobiSVD<PandaPlant::J>(jacobian).singularValues().minCoeff();
    const double damping = jacobian_damping(sigma);
    using Matrix6 = Eigen::Matrix<double,6,6>;
    const Matrix6 regularized = jacobian * jacobian.transpose() + damping * damping * Matrix6::Identity();
    const Eigen::LDLT<Matrix6> decomposition(regularized);
    require(decomposition.info() == Eigen::Success, "Jacobian solve failed");
    const auto inverse = (jacobian.transpose() * decomposition.solve(Matrix6::Identity())).eval();
    const Eigen::Map<const Eigen::Matrix<double,7,1>> dq(plant.dq.data()), ddq(plant.ddq.data());
    const Eigen::Matrix<double,6,1> velocity = jacobian * dq;
    const Eigen::Matrix<double,6,1> curvature = (jacobian - last_jacobian) * dq / .001;
    const Eigen::Matrix<double,6,1> acceleration = last_jacobian * ddq + curvature;
    Twist v{}, acc{};
    Eigen::Map<Eigen::Matrix<double,6,1>>(v.data()) = velocity;
    Eigen::Map<Eigen::Matrix<double,6,1>>(acc.data()) = acceleration;
    const auto weights = tracker.axis_allocation(plant.pose, v, acc);
    const auto limits = allocate_joint_dynamics(inverse, weights, {.4,2.,12.}, {.8,4.,25.});
    auto twist = tracker.step(plant.pose, v, acc, limits);
    for (std::size_t i = 0; i < 6; ++i) twist[i] -= .001 * curvature[i];
    Eigen::Matrix<double,7,1> posture = Eigen::Matrix<double,7,1>::Zero();
    posture(2) = -.8 * (plant.q[2] - elbow);
    const Eigen::Matrix<double,7,1> desired = inverse * Eigen::Map<const Eigen::Matrix<double,6,1>>(twist.data()) +
        (Eigen::Matrix<double,7,7>::Identity() - inverse * jacobian) * posture;
    std::array<double,7> requested{};
    Eigen::Map<Eigen::Matrix<double,7,1>>(requested.data()) = desired;
    const auto next = franka::limitRate(kJointVelocityLimits, kJointAccelerationLimits, kJointJerkLimits,
                                        requested, plant.dq, plant.ddq);
    for (std::size_t i = 0; i < 7; ++i) {
      const double next_acceleration = (next[i] - plant.dq[i]) / .001;
      require(std::abs(next[i]) <= kJointVelocityLimits[i] + 1e-8, "joint velocity exceeded");
      require(std::abs(next_acceleration) <= kJointAccelerationLimits[i] + 1e-8, "joint acceleration exceeded");
      require(std::abs(next_acceleration - plant.ddq[i]) / .001 <= kJointJerkLimits[i] + 1e-7, "joint jerk exceeded");
      maximum_joint_speed = std::max(maximum_joint_speed, std::abs(next[i]));
      plant.q[i] += .0005 * (next[i] + plant.dq[i]);
      plant.dq[i] = next[i];
      plant.ddq[i] = next_acceleration;
    }
    last_jacobian = jacobian;
    plant.forward();
  }
};

void test_reported_posture(bool zero_deadband = false) {
  PandaSimulation sim(zero_deadband ? 0.0 : .001);
  const Pose observed = pose_at(.400048, -.0145474, .305967);
  require(distance(sim.plant.pose, observed) < .00001, "Panda fixture disagrees with measured FK");
  const auto initial_inverse = (sim.plant.jacobian.transpose() *
      (sim.plant.jacobian * sim.plant.jacobian.transpose() + .0001 * Eigen::Matrix<double,6,6>::Identity()).inverse()).eval();
  Twist old_allocation;
  old_allocation.fill(1.0 / std::sqrt(3.0));
  const auto old_limits = allocate_joint_dynamics(initial_inverse, old_allocation, {.4,2.,12.}, {.8,4.,25.});
  require(old_limits.acceleration > .29 && old_limits.acceleration < .31,
          "fixture must reproduce the logged ~30 percent acceleration cap");
  Pose goal = sim.plant.pose;
  goal.p[2] -= .05;
  sim.tracker.set_target(goal);
  const auto allocation = sim.tracker.axis_allocation(sim.plant.pose, {}, {});
  require(norm3(allocation.data()) <= 1.0 + 1e-12, "axis budget exceeds vector limits");
  const auto new_limits = allocate_joint_dynamics(initial_inverse, allocation, {.4,2.,12.}, {.8,4.,25.});
  require(new_limits.acceleration > .55,
          "single-axis acceleration allocation did not improve sufficiently");
  double arrival = -1.0, lateral_error = 0.0, overshoot = 0.0;
  for (int cycle = 0; cycle < 3000; ++cycle) {
    // The realtime buffer is sampled at 1 kHz, even between ROS messages.
    Pose sample = goal;
    if (cycle % 2 == 0) for (double& value : sample.q) value = -value;
    sim.tracker.set_target(sample);
    sim.step();
    lateral_error = std::max(lateral_error, std::hypot(sim.plant.pose.p[0] - goal.p[0], sim.plant.pose.p[1] - goal.p[1]));
    overshoot = std::max(overshoot, goal.p[2] - sim.plant.pose.p[2]);
    if (arrival < 0 && distance(sim.plant.pose, goal) < .001) arrival = (cycle + 1) * .001;
  }
  require(distance(sim.plant.pose, goal) < .0001, "seven-joint plant failed to reach absolute goal");
  require(quaternion_distance(sim.plant.pose.q, goal.q) < .001, "seven-joint orientation drifted");
  require(arrival > 0 && arrival < .75, "5 cm Z motion is still too slow in reported posture");
  require(lateral_error < .0003, "stationary axes drifted during Z motion");
  require(overshoot < .001, "seven-joint Z overshoot exceeds 1 mm");
  std::cout << "reported_posture: old_accel_scale=" << old_limits.acceleration
            << " new_accel_scale=" << new_limits.acceleration << " arrival_1mm_s=" << arrival
            << " lateral_error_m=" << lateral_error << " overshoot_m=" << overshoot
            << " max_joint_speed=" << sim.maximum_joint_speed << "\n";
}

void test_joint_moving() {
  PandaSimulation sim;
  const Pose start = sim.plant.pose;
  Pose goal = start;
  double peak_error = 0.0, peak_rotation_error = 0.0;
  for (int cycle = 0; cycle < 10000; ++cycle) {
    if (cycle % 33 == 0) {
      const double phase = 2.0 * kPi * .4 * std::min(cycle * .001, 6.0);
      goal.p[0] = start.p[0] + .015 * std::sin(phase);
      goal.p[1] = start.p[1] + .015 * (1.0 - std::cos(phase));
      goal.p[2] = start.p[2] + .025 * std::sin(phase);
      const double angle = .12 * std::sin(phase);
      const double rotation[4] = {0.0, std::sin(angle/2), 0.0, std::cos(angle/2)};
      quaternion_multiply(rotation, start.q, goal.q);
    }
    sim.tracker.set_target(goal);
    sim.step();
    if (cycle > 600 && cycle < 6000) {
      peak_error = std::max(peak_error, distance(sim.plant.pose, goal));
      peak_rotation_error = std::max(peak_rotation_error, quaternion_distance(sim.plant.pose.q, goal.q));
    }
  }
  require(peak_error < .022, "30 Hz joint-space tracking lag exceeds 22 mm at ~80 mm/s");
  require(peak_rotation_error < .07, "30 Hz joint-space rotation lag exceeds 4 degrees");
  require(distance(sim.plant.pose, goal) < .0011, "seven-joint moving goal did not settle");
  require(quaternion_distance(sim.plant.pose.q, goal.q) < .011, "seven-joint rotation did not settle");
  const Eigen::Matrix<double,6,1> final_twist = sim.plant.jacobian *
      Eigen::Map<const Eigen::Matrix<double,7,1>>(sim.plant.dq.data());
  // The redundant joint posture may still return slowly in the nullspace.
  require(final_twist.head<3>().norm() < .0001, "seven-joint translation did not stop");
  require(final_twist.tail<3>().norm() < .001, "seven-joint rotation did not stop");
  std::cout << "joint_moving: peak_error_m=" << peak_error
            << " peak_rotation_error_rad=" << peak_rotation_error << "\n";
}
}  // namespace

int main(int argc, char** argv) {
  try {
    require(argc == 2, "expected scenario argument");
    const std::string scenario = argv[1];
    if (scenario == "fixed") test_fixed();
    else if (scenario == "startup") test_live_startup();
    else if (scenario == "reversal") test_reversal(false);
    else if (scenario == "saturation") test_reversal(true);
    else if (scenario == "moving") test_moving(false);
    else if (scenario == "drops") test_moving(true);
    else if (scenario == "rotation") test_rotation();
    else if (scenario == "noise") test_noise();
    else if (scenario == "circle") test_circle();
    else if (scenario == "changing_limits") test_changing_limits();
    else if (scenario == "feedback_delay") test_feedback_delay();
    else if (scenario == "stress") test_stress();
    else if (scenario == "reported_posture") test_reported_posture();
    else if (scenario == "zero_deadband") test_reported_posture(true);
    else if (scenario == "joint_moving") test_joint_moving();
    else throw std::runtime_error("unknown scenario");
    std::cout << scenario << " PASS\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
}
