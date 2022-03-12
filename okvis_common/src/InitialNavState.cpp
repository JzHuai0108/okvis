#include "swift_vio/InitialNavState.hpp"
#include <glog/logging.h>

namespace swift_vio {
InitialNavState::InitialNavState()
    : initializeToCustomPose(false),
      stateTime(),
      p_WS(0, 0, 0),
      q_WS(1, 0, 0, 0),
      v_WS(0, 0, 0),
      std_p_WS(1e-4, 1e-4, 1e-4),
      std_q_WS(30 * M_PI / 180, 30 * M_PI / 180, 1e-4),
      std_v_WS(0.1, 0.1, 0.1) {}

// v_WS, and std_v_WS are to be recalculated later according to updated p_WS and
// q_ws
InitialNavState::InitialNavState(const InitialNavState& rhs)
    : initializeToCustomPose(rhs.initializeToCustomPose),
      stateTime(rhs.stateTime),
      p_WS(rhs.p_WS),
      q_WS(rhs.q_WS),
      v_WS(rhs.v_WS),
      std_p_WS(rhs.std_p_WS),
      std_q_WS(rhs.std_q_WS),
      std_v_WS(rhs.std_v_WS) {}

void InitialNavState::updatePose(const okvis::kinematics::Transformation& T_WS,
                                 const okvis::Time state_time) {
  stateTime = state_time;
  p_WS = T_WS.r();
  q_WS = T_WS.q();
}

void InitialNavState::toInformation(
    Eigen::Matrix<double, 6, 6> *information) const {
  information->setZero();
  Eigen::Vector3d positionVariance = std_p_WS.cwiseAbs2();
  Eigen::Vector3d orientationVariance = std_q_WS.cwiseAbs2();
  information->diagonal().head<3>() = positionVariance.cwiseInverse();
  information->diagonal().tail<3>() = orientationVariance.cwiseInverse();
}

void InitialNavState::toCovariance(
    Eigen::Matrix<double, 6, 6> *covariance) const {
  covariance->setZero();
  covariance->diagonal().head<3>() = std_p_WS.cwiseAbs2();
  covariance->diagonal().tail<3>() = std_q_WS.cwiseAbs2();
}

InitialNavState& InitialNavState::operator=(const InitialNavState& other) {
  if (&other == this) return *this;
  initializeToCustomPose = other.initializeToCustomPose;
  stateTime = other.stateTime;
  p_WS = other.p_WS;
  q_WS = other.q_WS;
  v_WS = other.v_WS;
  std_p_WS = other.std_p_WS;
  std_q_WS = other.std_q_WS;
  std_v_WS = other.std_v_WS;
  return *this;
}

std::string InitialNavState::toString() const {
  std::stringstream ss;
  if (initializeToCustomPose) {
    ss << "initializeToCustomPose p_WS " << p_WS.transpose() << " "
       << q_WS.coeffs().transpose() << " at time " << stateTime;
  } else {
    ss << "Not initializeToCustomPose.";
  }
  ss << " v_WS " << v_WS.transpose() << ", std " << std_v_WS.transpose()
     << "\n";
  ss << " p_WS std " << std_p_WS.transpose() << " q_WS std "
     << std_q_WS.transpose();
  return ss.str();
}

void alignZ(const Eigen::Vector3d& a_S, Eigen::Quaterniond* q_WS) {
  *q_WS = Eigen::Quaterniond::FromTwoVectors(a_S, Eigen::Vector3d(0, 0, 1));
}

// Initialise pose from IMU measurements. For convenience as static.
bool initPoseFromImu(
    const okvis::ImuMeasurementDeque &imuMeasurements,
    okvis::kinematics::Transformation &T_WS) {
  // set translation to zero, unit rotation
  T_WS.setIdentity();
  if (imuMeasurements.size() == 0)
    return false;

  // acceleration vector
  Eigen::Vector3d acc_B = Eigen::Vector3d::Zero();
  for (okvis::ImuMeasurementDeque::const_iterator it = imuMeasurements.begin();
      it < imuMeasurements.end(); ++it) {
    acc_B += it->measurement.accelerometers;
  }
  acc_B /= double(imuMeasurements.size());
  Eigen::Vector3d e_acc = acc_B.normalized();

  // align with ez_W:
  Eigen::Vector3d ez_W(0.0, 0.0, 1.0);
  Eigen::Matrix<double, 6, 1> poseIncrement;
  poseIncrement.head<3>() = Eigen::Vector3d::Zero();
  poseIncrement.tail<3>() = ez_W.cross(e_acc).normalized();
  double angle = std::acos(ez_W.transpose() * e_acc);
  poseIncrement.tail<3>() *= angle;
  T_WS.oplus(-poseIncrement);
  return true;
}

bool initPoseFromImu(const okvis::ImuMeasurementDeque &imuMeasurements,
                     okvis::Time time, okvis::kinematics::Transformation &T_WS,
                     okvis::Duration radius) {
  T_WS.setIdentity();

  // acceleration vector
  Eigen::Vector3d acc_B = Eigen::Vector3d::Zero();
  size_t usedImuData = 0u;
  okvis::Time start = time - radius;
  okvis::Time finish = time + radius;

  for (okvis::ImuMeasurementDeque::const_iterator it = imuMeasurements.begin();
      it < imuMeasurements.end(); ++it) {
    if (it->timeStamp >= start && it->timeStamp <= finish) {
      acc_B += it->measurement.accelerometers;
      ++usedImuData;
    }
  }
  if (usedImuData == 0u)
    return false;

  acc_B /= double(usedImuData);
  Eigen::Vector3d e_acc = acc_B.normalized();

  // align with ez_W:
  Eigen::Vector3d ez_W(0.0, 0.0, 1.0);
  Eigen::Matrix<double, 6, 1> poseIncrement;
  poseIncrement.head<3>() = Eigen::Vector3d::Zero();
  poseIncrement.tail<3>() = ez_W.cross(e_acc).normalized();
  double angle = std::acos(ez_W.transpose() * e_acc);
  poseIncrement.tail<3>() *= angle;
  T_WS.oplus(-poseIncrement);
  return true;
}

void initBiasesFromStaticImu(const okvis::ImuMeasurementDeque &imuMeasurements,
                             const Eigen::Vector3d &gravityB,
                             okvis::ImuMeasurement *biases) {
  Eigen::Vector3d gyroSum = Eigen::Vector3d::Zero();
  Eigen::Vector3d accelSum = Eigen::Vector3d::Zero();
  for (const auto &imudata : imuMeasurements) {
    gyroSum += imudata.measurement.gyroscopes;
    accelSum += imudata.measurement.accelerometers;
  }
  if (imuMeasurements.size() == 0u) {
    LOG(WARNING) << "No IMU data found to initialize biases!";
    return;
  }
  double invnum = 1.0 / imuMeasurements.size();
  biases->measurement.gyroscopes = gyroSum * invnum;
  biases->measurement.accelerometers = accelSum * invnum + gravityB;
}
}  // namespace swift_vio
