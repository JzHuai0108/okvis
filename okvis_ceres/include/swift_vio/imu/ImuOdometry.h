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

  template <typename ImuModelT>
  static int propagation_RungeKutta(
      const okvis::ImuMeasurementDeque &imuMeasurements,
      const okvis::ImuParameters &imuParams,
      okvis::kinematics::Transformation &T_WS,
      okvis::SpeedAndBiases &speedAndBias, const ImuModelT &imuModel,
      const okvis::Time &t_start, const okvis::Time &t_end,
      Eigen::Matrix<double, ImuModelT::kAugmentedMinDim + kNavStateBiasMinDim,
                    ImuModelT::kAugmentedMinDim + kNavStateBiasMinDim> *P_ptr =
          nullptr,
      Eigen::Matrix<double, ImuModelT::kAugmentedMinDim + kNavStateBiasMinDim,
                    ImuModelT::kAugmentedMinDim + kNavStateBiasMinDim>
          *F_tot_ptr = nullptr);

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
  template<typename ImuModelT>
  static int propagationBackward_RungeKutta(
      const okvis::ImuMeasurementDeque& imuMeasurements,
      const okvis::ImuParameters& imuParams,
      okvis::kinematics::Transformation& T_WS,
      okvis::SpeedAndBiases& speedAndBias,
      const ImuModelT &iem, const okvis::Time& t_start,
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
  template <typename ImuModelT>
  static bool interpolateInertialData(const okvis::ImuMeasurementDeque& imuMeas,
                                      const ImuModelT& iem,
                                      const okvis::Time& queryTime,
                                      okvis::ImuMeasurement& queryValue);

}; // class ImuOdometry

OKVIS_DEFINE_EXCEPTION(Exception, std::runtime_error)

// propagate pose, speedAndBias, and possibly covariance
// note the RungeKutta method assumes that the z direction of the world frame is
// negative gravity direction
template<typename ImuModelT>
int ImuOdometry::propagation_RungeKutta(
    const okvis::ImuMeasurementDeque& imuMeasurements, const okvis::ImuParameters& imuParams,
    okvis::kinematics::Transformation& T_WS, okvis::SpeedAndBiases& speedAndBias,
    const ImuModelT &iem, const okvis::Time& startTime,
    const okvis::Time& finishTime,
    Eigen::Matrix<double, ImuModelT::kAugmentedMinDim + kNavStateBiasMinDim,
                  ImuModelT::kAugmentedMinDim + kNavStateBiasMinDim> *P_ptr,
    Eigen::Matrix<double, ImuModelT::kAugmentedMinDim + kNavStateBiasMinDim,
                  ImuModelT::kAugmentedMinDim + kNavStateBiasMinDim> *F_tot_ptr) {
  if (imuMeasurements.begin()->timeStamp > startTime) {
    std::cout << "imuMeas begin time and startTime "
              << imuMeasurements.begin()->timeStamp << " " << startTime
              << std::endl;
    OKVIS_ASSERT_TRUE(Exception,
                      imuMeasurements.begin()->timeStamp <= startTime,
                      "IMU data do not extend to the previous state epoch");
  }
  OKVIS_ASSERT_TRUE(Exception,
                    imuMeasurements.rbegin()->timeStamp >= finishTime,
                    "IMU data do not extend to the current estimated epoch");

  Eigen::Vector3d p_WS_W = T_WS.r();
  Eigen::Quaterniond q_WS = T_WS.q();

  bool hasStarted = false;
  int numUsedImuMeasurements = 0;
  auto iterLast = imuMeasurements.end();
  for (auto iter = imuMeasurements.begin(); iter != imuMeasurements.end();
       ++iter) {
    if (iter->timeStamp <= startTime) {
      iterLast = iter;
      continue;
    }

    if (hasStarted == false) {
      hasStarted = true;
      if (iter->timeStamp >= finishTime)  // in case the interval of start and
                                          // finish time is very small
      {
        ode::integrateOneStep_RungeKutta(
            iterLast->measurement.gyroscopes,
            iterLast->measurement.accelerometers, iter->measurement.gyroscopes,
            iter->measurement.accelerometers, imuParams.g, imuParams.sigma_g_c,
            imuParams.sigma_a_c, imuParams.sigma_gw_c, imuParams.sigma_aw_c,
            (finishTime - startTime).toSec(), p_WS_W, q_WS, speedAndBias,
            iem, P_ptr, F_tot_ptr);
        ++numUsedImuMeasurements;
        break;
      }

      ode::integrateOneStep_RungeKutta(
          iterLast->measurement.gyroscopes,
          iterLast->measurement.accelerometers, iter->measurement.gyroscopes,
          iter->measurement.accelerometers, imuParams.g, imuParams.sigma_g_c,
          imuParams.sigma_a_c, imuParams.sigma_gw_c, imuParams.sigma_aw_c,
          (iter->timeStamp - startTime).toSec(), p_WS_W, q_WS, speedAndBias,
          iem, P_ptr, F_tot_ptr);

    } else {
      if (iter->timeStamp >= finishTime) {
        ode::integrateOneStep_RungeKutta(
            iterLast->measurement.gyroscopes,
            iterLast->measurement.accelerometers, iter->measurement.gyroscopes,
            iter->measurement.accelerometers, imuParams.g, imuParams.sigma_g_c,
            imuParams.sigma_a_c, imuParams.sigma_gw_c, imuParams.sigma_aw_c,
            (finishTime - iterLast->timeStamp).toSec(), p_WS_W, q_WS,
            speedAndBias, iem, P_ptr, F_tot_ptr);
        ++numUsedImuMeasurements;
        break;
      }

      ode::integrateOneStep_RungeKutta(
          iterLast->measurement.gyroscopes,
          iterLast->measurement.accelerometers, iter->measurement.gyroscopes,
          iter->measurement.accelerometers, imuParams.g, imuParams.sigma_g_c,
          imuParams.sigma_a_c, imuParams.sigma_gw_c, imuParams.sigma_aw_c,
          (iter->timeStamp - iterLast->timeStamp).toSec(), p_WS_W, q_WS,
          speedAndBias, iem, P_ptr, F_tot_ptr);
    }
    iterLast = iter;
    ++numUsedImuMeasurements;
  }
  T_WS = okvis::kinematics::Transformation(p_WS_W, q_WS);
  return numUsedImuMeasurements;
}

template <typename ImuModelT>
int ImuOdometry::propagationBackward_RungeKutta(
    const okvis::ImuMeasurementDeque& imuMeasurements, const okvis::ImuParameters& imuParams,
    okvis::kinematics::Transformation& T_WS, okvis::SpeedAndBiases& speedAndBias,
    const ImuModelT &iem, const okvis::Time& startTime,
    const okvis::Time& finishTime) {
  OKVIS_ASSERT_TRUE(
      Exception, imuMeasurements.begin()->timeStamp <= finishTime,
      "Backward: IMU data do not extend to the current estimated epoch");
  OKVIS_ASSERT_TRUE(
      Exception, imuMeasurements.rbegin()->timeStamp >= startTime,
      "Backward: IMU data do not extend to the previous state epoch");

  Eigen::Vector3d p_WS_W = T_WS.r();
  Eigen::Quaterniond q_WS = T_WS.q();

  bool hasStarted = false;
  int numUsedImuMeasurements = 0;
  auto iterLast = imuMeasurements.rend();
  for (auto iter = imuMeasurements.rbegin(); iter != imuMeasurements.rend();
       ++iter) {
    if (iter->timeStamp >= startTime) {
      iterLast = iter;
      continue;
    }

    if (hasStarted == false) {
      hasStarted = true;
      if (iter->timeStamp <= finishTime)  // in case the interval of start and
                                          // finish time is very small
      {
        ode::integrateOneStepBackward_RungeKutta(
            iter->measurement.gyroscopes, iter->measurement.accelerometers,
            iterLast->measurement.gyroscopes,
            iterLast->measurement.accelerometers, imuParams.g,
            imuParams.sigma_g_c, imuParams.sigma_a_c, imuParams.sigma_gw_c,
            imuParams.sigma_aw_c, (startTime - finishTime).toSec(), p_WS_W,
            q_WS, speedAndBias, iem);
        ++numUsedImuMeasurements;
        break;
      }

      ode::integrateOneStepBackward_RungeKutta(
          iter->measurement.gyroscopes, iter->measurement.accelerometers,
          iterLast->measurement.gyroscopes,
          iterLast->measurement.accelerometers, imuParams.g,
          imuParams.sigma_g_c, imuParams.sigma_a_c, imuParams.sigma_gw_c,
          imuParams.sigma_aw_c, (startTime - iter->timeStamp).toSec(), p_WS_W,
          q_WS, speedAndBias, iem);

    } else {
      if (iter->timeStamp <= finishTime) {
        ode::integrateOneStepBackward_RungeKutta(
            iter->measurement.gyroscopes, iter->measurement.accelerometers,
            iterLast->measurement.gyroscopes,
            iterLast->measurement.accelerometers, imuParams.g,
            imuParams.sigma_g_c, imuParams.sigma_a_c, imuParams.sigma_gw_c,
            imuParams.sigma_aw_c, (iterLast->timeStamp - finishTime).toSec(),
            p_WS_W, q_WS, speedAndBias, iem);
        ++numUsedImuMeasurements;
        break;
      }

      ode::integrateOneStepBackward_RungeKutta(
          iter->measurement.gyroscopes, iter->measurement.accelerometers,
          iterLast->measurement.gyroscopes,
          iterLast->measurement.accelerometers, imuParams.g,
          imuParams.sigma_g_c, imuParams.sigma_a_c, imuParams.sigma_gw_c,
          imuParams.sigma_aw_c, (iterLast->timeStamp - iter->timeStamp).toSec(),
          p_WS_W, q_WS, speedAndBias, iem);
    }
    iterLast = iter;
    ++numUsedImuMeasurements;
  }
  T_WS = okvis::kinematics::Transformation(p_WS_W, q_WS);
  return numUsedImuMeasurements;
}

template <typename ImuModelT>
bool ImuOdometry::interpolateInertialData(
    const okvis::ImuMeasurementDeque& imuMeas, const ImuModelT& iem,
    const okvis::Time& queryTime, okvis::ImuMeasurement& queryValue) {
  OKVIS_ASSERT_GT(Exception, imuMeas.size(), 0u, "not enough imu meas!");
  auto iterLeft = imuMeas.begin(), iterRight = imuMeas.end();
  OKVIS_ASSERT_TRUE_DBG(Exception, iterLeft->timeStamp <= queryTime,
                        "Imu measurements has wrong timestamps");
  if (imuMeas.back().timeStamp < queryTime) {
    LOG(WARNING) << "Using the gyro value at " << imuMeas.back().timeStamp
                 << " instead of the requested at " << queryTime;
    queryValue = imuMeas.back();
    return false;
  }
  for (auto iter = imuMeas.begin(); iter != imuMeas.end(); ++iter) {
    if (iter->timeStamp < queryTime) {
      iterLeft = iter;
    } else if (iter->timeStamp == queryTime) {
      queryValue = *iter;
      return true;
    } else {
      iterRight = iter;
      break;
    }
  }
  double ratio = (queryTime - iterLeft->timeStamp).toSec() /
                 (iterRight->timeStamp - iterLeft->timeStamp).toSec();
  queryValue.timeStamp = queryTime;
  Eigen::Vector3d omega_S0 =
      (iterRight->measurement.gyroscopes - iterLeft->measurement.gyroscopes) *
          ratio +
      iterLeft->measurement.gyroscopes;
  Eigen::Vector3d acc_S0 = (iterRight->measurement.accelerometers -
                            iterLeft->measurement.accelerometers) *
                               ratio +
                           iterLeft->measurement.accelerometers;
  iem.correct(omega_S0, acc_S0, &queryValue.measurement.gyroscopes,
               &queryValue.measurement.accelerometers);
  return true;
}

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
