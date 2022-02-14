#ifndef INCLUDE_SWIFT_VIO_IMU_ODOMETRY_H_
#define INCLUDE_SWIFT_VIO_IMU_ODOMETRY_H_
#include <vector>

#include <okvis/Measurements.hpp>
#include <okvis/Parameters.hpp>
#include <okvis/Time.hpp>
#include <okvis/Variables.hpp>
#include <okvis/assert_macros.hpp>

#include "swift_vio/imu/odeHybrid.hpp"

namespace swift_vio {
okvis::kinematics::Transformation
propagationConstVelocity(const okvis::kinematics::Transformation &T_WS,
                         const Eigen::Vector3d &v_WS,
                         const Eigen::Vector3d &omega_S, double dt);

class ImuOdometry {
  /// \brief The type of the covariance.
  typedef Eigen::Matrix<double, 15, 15> covariance_t;

  /// \brief The type of the information (same matrix dimension as covariance).
  typedef covariance_t information_t;

  /// \brief The type of hte overall Jacobian.
  typedef Eigen::Matrix<double, 15, 15> jacobian_t;

 public:
  /**
   * @brief Propagates pose, speeds and biases with given IMU measurements.
   * Extends okvis::ceres::ImuError::propagation to handle a generic IMU error
   * model and given linearization point for position and velocity.
   * @remark This can be used externally to perform propagation
   * @warning covariance and jacobian should be provided at the same time.
   * @param[in] imuMeasurements All the IMU measurements.
   * @param[in] imuParams The parameters to be used.
   * @param[in,out] T_WS Start pose.
   * @param[in,out] speedAndBiases Start speed and biases.
   * @param[in] t_start Start time.
   * @param[in] t_end End time.
   * @param[in,out] covariance Covariance for the propagated state.
   * @param[in,out] jacobian Jacobian w.r.t. the start state.
   * @param[in] postionVelocityLin is the linearization points of position
   * p_WS and velocity v_WS at t_start.
   * @return Number of integration steps.
   * assume W frame has z axis pointing up
   * Euler approximation is used to incrementally compute the integrals, and
   * the length of integral interval only adversely affect the covariance and
   * jacobian a little.
   */
  static int propagation(
      const okvis::ImuMeasurementDeque& imuMeasurements,
      const okvis::ImuParameters& imuParams,
      okvis::kinematics::Transformation& T_WS, Eigen::Vector3d& v_WS,
      const ImuErrorModel<double>& iem, const okvis::Time& t_start,
      const okvis::Time& t_end,
      Eigen::MatrixXd* covariance = nullptr,
      Eigen::MatrixXd* jacobian = nullptr,
      const Eigen::Matrix<double, 6, 1>* positionVelocityLin = nullptr);

  /**
   * @brief propagationRightInvariantError
   * @param imuMeasurements
   * @param imuParams
   * @param T_WS
   * @param v_WS
   * @param iem
   * @param t_start
   * @param t_end
   * @param covariance error vector order \delta[\phi, v, p, ba, bg]
   * @param jacobian
   * @return
   */
  static int propagationRightInvariantError(
      const okvis::ImuMeasurementDeque& imuMeasurements,
      const okvis::ImuParameters& imuParams,
      okvis::kinematics::Transformation& T_WS, Eigen::Vector3d& v_WS,
      const ImuErrorModel<double>& iem, const okvis::Time& t_start,
      const okvis::Time& t_end,
      Eigen::Matrix<double, 15, 15>* covariance = nullptr,
      Eigen::Matrix<double, 15, 15>* jacobian = nullptr);

  // t_start is greater than t_end
  static int propagationBackward(
      const okvis::ImuMeasurementDeque& imuMeasurements,
      const okvis::ImuParameters& imuParams,
      okvis::kinematics::Transformation& T_WS, Eigen::Vector3d& v_WS,
      const ImuErrorModel<double>& iem, const okvis::Time& t_start,
      const okvis::Time& t_end);

  static int propagation_RungeKutta(
      const okvis::ImuMeasurementDeque& imuMeasurements,
      const okvis::ImuParameters& imuParams,
      okvis::kinematics::Transformation& T_WS,
      okvis::SpeedAndBiases& speedAndBias,
      const ImuErrorModel<double>& iem, const okvis::Time& t_start,
      const okvis::Time& t_end,
      Eigen::MatrixXd* P_ptr = nullptr,
      Eigen::MatrixXd* F_tot_ptr = nullptr);

  /**
   * @brief propagationBackward_RungeKutta propagate pose, speed and biases.
   * @warning This method assumes that z direction of the world frame is along negative gravity.
   * @param imuMeasurements [t0, t1, ..., t_{n-1}]
   * @param imuParams
   * @param T_WS pose at t_start
   * @param speedAndBias linear velocity and bias at t_start
   * @param iem
   * @param t_start
   * @param t_end t_start >= t_end
   * @return number of used IMU measurements
   */
  static int propagationBackward_RungeKutta(
      const okvis::ImuMeasurementDeque& imuMeasurements,
      const okvis::ImuParameters& imuParams,
      okvis::kinematics::Transformation& T_WS,
      okvis::SpeedAndBiases& speedAndBias,
      const ImuErrorModel<double>& iem, const okvis::Time& t_start,
      const okvis::Time& t_end);

  /**
   * @brief interpolateInertialData linearly interpolate inertial readings
   *     at queryTime given imuMeas
   * @param imuMeas has size greater than 0
   * @param iem The intermediate members of iem may be changed.
   * @param queryTime
   * @param queryValue
   * @return false if interpolation is impossible, e.g., in the case of
   *     extrapolation or empty imuMeas
   */
  static bool interpolateInertialData(const okvis::ImuMeasurementDeque& imuMeas,
                                      const ImuErrorModel<double>& iem,
                                      const okvis::Time& queryTime,
                                      okvis::ImuMeasurement& queryValue);
}; // class ImuOdometry

/**
 * @brief poseAndVelocityAtObservation for feature i, estimate
 *     $p_B^G(t_{f_i})$, $R_B^G(t_{f_i})$, $v_B^G(t_{f_i})$, and
 *     $\omega_{GB}^B(t_{f_i})$ with imu measurements
 * @param imuMeas cover stateEpoch to the extent of featureTime
 * @param imuAugmentedParams imu params exccept for gyro and accel biases
 * @param imuParameters
 * @param stateEpoch
 * @param featureTime
 * @param T_WB[in/out] in: pose at stateEpoch;
 *     out: pose at stateEpoch + featureTime
 * @param sb[in/out] in: speed and biases at stateEpoch;
 *     out: speed and biases at stateEpoch + featureTime
 * @param interpolatedInertialData[out] inertial measurements at stateEpoch +
 *     featureTime after correction for biases etc.
 */
void poseAndVelocityAtObservation(
    const okvis::ImuMeasurementDeque& imuMeas,
    const Eigen::Matrix<double, Eigen::Dynamic, 1>& imuAugmentedParams,
    const okvis::ImuParameters& imuParameters, const okvis::Time& stateEpoch,
    const okvis::Duration& featureTime, okvis::kinematics::Transformation* T_WB,
    okvis::SpeedAndBiases* sb, okvis::ImuMeasurement* interpolatedInertialData,
    bool use_RK4);

/**
 * @brief poseAndLinearVelocityAtObservation Similarly to
 *     poseAndVelocityAtObservation except that the inertial data is not
 *     interpolated and the RK4 propagation is not used.
 * @param imuMeas
 * @param imuAugmentedParams
 * @param imuParameters
 * @param stateEpoch
 * @param featureTime
 * @param T_WB
 * @param sb
 */
void poseAndLinearVelocityAtObservation(
    const okvis::ImuMeasurementDeque& imuMeas,
    const Eigen::Matrix<double, Eigen::Dynamic, 1>& imuAugmentedParams,
    const okvis::ImuParameters& imuParameters, const okvis::Time& stateEpoch,
    const okvis::Duration& featureTime, okvis::kinematics::Transformation* T_WB,
    okvis::SpeedAndBiases* sb);

}  // namespace swift_vio
#endif // INCLUDE_SWIFT_VIO_IMU_ODOMETRY_H_
