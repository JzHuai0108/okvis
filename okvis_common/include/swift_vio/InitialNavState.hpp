#ifndef INITIAL_NAV_STATE_HPP
#define INITIAL_NAV_STATE_HPP

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <okvis/ImuMeasurements.hpp>
#include <okvis/kinematics/Transformation.hpp>
#include <okvis/Time.hpp>

namespace swift_vio {
struct InitialNavState {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  bool initializeToCustomPose;
  okvis::Time stateTime;  // epoch for the initialization values
  Eigen::Vector3d p_WS;
  Eigen::Quaterniond q_WS;
  Eigen::Vector3d v_WS;
  Eigen::Vector3d std_p_WS;
  Eigen::Vector3d
      std_q_WS;  // std of $\delta \theta$ which is expressed in the world frame
  Eigen::Vector3d std_v_WS;

  InitialNavState();

  InitialNavState(const InitialNavState& rhs);

  void updatePose(const okvis::kinematics::Transformation& T_WS,
                  const okvis::Time state_time);

  okvis::kinematics::Transformation pose() const {
    return okvis::kinematics::Transformation(p_WS, q_WS);
  }

  void toInformation(Eigen::Matrix<double, 6, 6>* information) const;

  void toCovariance(Eigen::Matrix<double, 6, 6>* covariance) const;

  InitialNavState& operator=(const InitialNavState& other);

  std::string toString() const;
};

/**
 * @brief alignZ
 * @param a_S
 * @param[out] q_WS q_WS * a_S = [0, 0, s]
 */
void alignZ(const Eigen::Vector3d &a_S, Eigen::Quaterniond *q_WS);

/**
 * @brief Initialise pose from IMU measurements. For convenience as static.
 * @param[in]  imuMeasurements The IMU measurements to be used for this.
 * @param[out] T_WS initialised pose.
 * @return True if successful.
 */
bool initPoseFromImu(const okvis::ImuMeasurementDeque &imuMeasurements,
                     okvis::kinematics::Transformation &T_WS);

/**
 * @brief Initialize orientation at time by using acceleration in [time - radius, time + radius],
 * translation is set to zero.
 * @param imuMeasurements
 * @param time
 * @param T_WS
 * @param radius
 * @return
 */
bool initPoseFromImu(const okvis::ImuMeasurementDeque &imuMeasurements,
                     okvis::Time time, okvis::kinematics::Transformation &T_WS,
                     okvis::Duration radius = okvis::Duration(0.05));

void initBiasesFromStaticImu(const okvis::ImuMeasurementDeque &imuMeasurements,
                             const Eigen::Vector3d &gravityB,
                             okvis::ImuMeasurement *biases);
}  // namespace swift_vio
#endif  // INITIAL_NAV_STATE_HPP
