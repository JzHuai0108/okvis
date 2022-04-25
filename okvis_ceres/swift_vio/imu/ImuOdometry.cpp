#include <glog/logging.h>

#include <swift_vio/imu/ImuOdometry.h>
#include <swift_vio/imu/ImuModels.hpp>
#include <swift_vio/ParallaxAnglePoint.hpp>

#include <okvis/Parameters.hpp>
#include <okvis/assert_macros.hpp>
#include <okvis/kinematics/Transformation.hpp>

namespace swift_vio {

okvis::kinematics::Transformation
propagationConstVelocity(const okvis::kinematics::Transformation &T_WS,
                         const Eigen::Vector3d &v_WS,
                         const Eigen::Vector3d &omega_S, double dt) {
  okvis::kinematics::Transformation T_WSp = T_WS;
  T_WSp.setTranslation(T_WS.r() + v_WS * dt);
  Eigen::Vector3d aa = omega_S * dt;
  T_WSp.setRotation(T_WS.q() * Eigen::AngleAxisd(aa.norm(), aa.normalized()));
  return T_WSp;
}

// Propagates pose, speeds and biases with given IMU measurements with trapezoid
// rules assume linear change of acceleration and angular rate between epochs
int ImuOdometry::propagation(
    const okvis::ImuMeasurementDeque& imuMeasurements,
    const okvis::ImuParameters& imuParams,
    okvis::kinematics::Transformation& T_WS, Eigen::Vector3d& v_WS,
    const Imu_BG_BA_TG_TS_TA& iem, const okvis::Time& t_start,
    const okvis::Time& t_end,
    Eigen::MatrixXd* covariance,
    Eigen::MatrixXd* jacobian,
    const Eigen::Matrix<double, 6, 1>* positionVelocityLin) {
  okvis::Time time = t_start;

  // sanity check:
  if (imuMeasurements.front().timeStamp > time) {
    LOG(WARNING)
        << "IMU front meas timestamp is greater than start integration time "
        << imuMeasurements.front().timeStamp << " " << time;
  }

  bool use_first_estimate = (positionVelocityLin != nullptr);
  if (imuMeasurements.back().timeStamp < t_end) {
    LOG(WARNING) << "Imu last reading has an epoch "
                 << imuMeasurements.back().timeStamp
                 << " less than propagation right target " << t_end;
    return -1;
  }

  // initial condition at sensor frame SO
  const Eigen::Vector3d r_0 = T_WS.r();
  const Eigen::Quaterniond q_WS_0 = T_WS.q();
  const Eigen::Matrix3d C_WS_0 = T_WS.C();
  const Eigen::Vector3d g_W = imuParams.g * imuParams.gravityDirection();

  // record the linearization point for position p_WS, and v_WS at two
  // subsequent steps
  Eigen::Matrix<double, 6, 1> linPoint;
  if (use_first_estimate) {
    linPoint = *positionVelocityLin;
  } else {
    linPoint << r_0, v_WS;
  }
  Eigen::Matrix<double, 6, 1> linPoint_1;

  // increments (initialise with identity)
  // denote t_start by $t_0$
  Eigen::Quaterniond Delta_q(1, 0, 0, 0);  // quaternion of DCM from Si to S0
  //$\int_{t_0}^{t_i} R_S^{S_0} dt$
  Eigen::Matrix3d C_integral =
      Eigen::Matrix3d::Zero();  // integrated DCM up to Si expressed in S0 frame
  // $\int_{t_0}^{t_i} \int_{t_0}^{s} R_S^{S_0} dt ds$
  Eigen::Matrix3d C_doubleintegral =
      Eigen::Matrix3d::Zero();  // double integrated DCM up to Si expressed in
                                // S0 frame
  // $\int_{t_0}^{t_i} R_S^{S_0} a^S dt$
  Eigen::Vector3d acc_integral =
      Eigen::Vector3d::Zero();  // integrated acceleration up to Si expressed in
                                // S0 frame
  // $\int_{t_0}^{t_i} \int_{t_0}^{s} R_S^{S_0} a^S dt ds$
  Eigen::Vector3d acc_doubleintegral =
      Eigen::Vector3d::Zero();  // double integrated acceleration up to Si
                                // expressed in S0 frame
  // sub-Jacobians
  // $R_{S_0}^W \frac{d^{S_0}\alpha_{l+1}}{d b_g_{l}} = \frac{d^W\alpha_{l+1}}{d
  // b_g_{l}} $ for ease of implementation, we compute all the Jacobians with
  // positive increment, thus there can be a sign difference between, for
  // instance, dalpha_db_g and \frac{d^{S_0}\alpha_{l+1}}{d b_g_{(l)}}. This
  // difference is adjusted when putting all Jacobians together
  Eigen::Matrix3d dalpha_db_g = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d dalpha_db_a = Eigen::Matrix3d::Zero();
  Eigen::Matrix<double, 3, 9> dalpha_dT_g = Eigen::Matrix<double, 3, 9>::Zero();
  Eigen::Matrix<double, 3, 9> dalpha_dT_s = Eigen::Matrix<double, 3, 9>::Zero();
  Eigen::Matrix<double, 3, 9> dalpha_dT_a = Eigen::Matrix<double, 3, 9>::Zero();

  Eigen::Matrix3d dv_db_g = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d dv_db_a = Eigen::Matrix3d::Zero();
  Eigen::Matrix<double, 3, 9> dv_dT_g = Eigen::Matrix<double, 3, 9>::Zero();
  Eigen::Matrix<double, 3, 9> dv_dT_s = Eigen::Matrix<double, 3, 9>::Zero();
  Eigen::Matrix<double, 3, 9> dv_dT_a = Eigen::Matrix<double, 3, 9>::Zero();

  Eigen::Matrix3d dp_db_g = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d dp_db_a = Eigen::Matrix3d::Zero();
  Eigen::Matrix<double, 3, 9> dp_dT_g = Eigen::Matrix<double, 3, 9>::Zero();
  Eigen::Matrix<double, 3, 9> dp_dT_s = Eigen::Matrix<double, 3, 9>::Zero();
  Eigen::Matrix<double, 3, 9> dp_dT_a = Eigen::Matrix<double, 3, 9>::Zero();

  // intermediate variables, must assign values to them before using.
  Eigen::Matrix<double, 3, 9> dalpha_dT_g_1;
  Eigen::Matrix<double, 3, 9> dalpha_dT_s_1;
  Eigen::Matrix<double, 3, 9> dalpha_dT_a_1;
  Eigen::Matrix3d dv_db_g_1;
  Eigen::Matrix<double, 3, 9> dv_dT_g_1;
  Eigen::Matrix<double, 3, 9> dv_dT_s_1;
  Eigen::Matrix<double, 3, 9> dv_dT_a_1;

  Eigen::MatrixXd P_delta;
  Eigen::MatrixXd T;
  int covRows = 0;
  if (covariance || jacobian) {
    covRows = covariance ?  covariance->rows() : jacobian->rows();

    P_delta = *covariance;
    scaleBlockRows(C_WS_0.transpose(), 3, &P_delta);
    scaleBlockCols(C_WS_0, 3, &P_delta);
  }
  int gravityErrorStartIndex = covRows - 2;
  NormalVectorElement normalGravity(imuParams.gravityDirection());

  double Delta_t = 0;  // integrated time up to Si since S0 frame
  bool hasStarted = false;
  int i = 0;
  Eigen::Matrix3d invTgsa = iem.invTgsa();

  for (okvis::ImuMeasurementDeque::const_iterator it = imuMeasurements.begin();
       it != imuMeasurements.end(); ++it) {
    Eigen::Vector3d omega_S_0 = it->measurement.gyroscopes;
    Eigen::Vector3d acc_S_0 = it->measurement.accelerometers;
    // access out of range element won't happen becsuse the loop breaks once
    // nexttime==t_end
    Eigen::Vector3d omega_S_1 = (it + 1)->measurement.gyroscopes;
    Eigen::Vector3d acc_S_1 = (it + 1)->measurement.accelerometers;

    // time delta
    okvis::Time nexttime;
    if ((it + 1) == imuMeasurements.end()) {
      nexttime = t_end;
    } else
      nexttime = (it + 1)->timeStamp;
    double dt = (nexttime - time).toSec();

    if (t_end < nexttime) {
      double interval = (nexttime - it->timeStamp).toSec();
      nexttime = t_end;
      dt = (nexttime - time).toSec();
      const double r = dt / interval;
      omega_S_1 = ((1.0 - r) * omega_S_0 + r * omega_S_1).eval();
      acc_S_1 = ((1.0 - r) * acc_S_0 + r * acc_S_1).eval();
    }

    if (dt <= 0.0) {
      continue;
    }
    Delta_t += dt;

    if (!hasStarted) {
      hasStarted = true;
      const double r = dt / (nexttime - it->timeStamp).toSec();
      omega_S_0 = (r * omega_S_0 + (1.0 - r) * omega_S_1).eval();
      acc_S_0 = (r * acc_S_0 + (1.0 - r) * acc_S_1).eval();
    }

    // ensure integrity
    double sigma_g_c = imuParams.sigma_g_c;
    double sigma_a_c = imuParams.sigma_a_c;

    if (fabs(omega_S_0[0]) > imuParams.g_max ||
        fabs(omega_S_0[1]) > imuParams.g_max ||
        fabs(omega_S_0[2]) > imuParams.g_max ||
        fabs(omega_S_1[0]) > imuParams.g_max ||
        fabs(omega_S_1[1]) > imuParams.g_max ||
        fabs(omega_S_1[2]) > imuParams.g_max) {
      sigma_g_c *= 100;
      LOG(WARNING) << "gyr saturation";
    }

    if (fabs(acc_S_0[0]) > imuParams.a_max ||
        fabs(acc_S_0[1]) > imuParams.a_max ||
        fabs(acc_S_0[2]) > imuParams.a_max ||
        fabs(acc_S_1[0]) > imuParams.a_max ||
        fabs(acc_S_1[1]) > imuParams.a_max ||
        fabs(acc_S_1[2]) > imuParams.a_max) {
      sigma_a_c *= 100;
      LOG(WARNING) << "acc saturation";
    }

    Eigen::Quaterniond dq;  // q_{S_{i+1}}^{S_i}

    Eigen::Vector3d omega_est;
    Eigen::Vector3d acc_est;
    iem.correct(omega_S_0, acc_S_0, &omega_est, &acc_est);

    Eigen::Vector3d omega_est_1;
    Eigen::Vector3d acc_est_1;
    iem.correct(omega_S_1, acc_S_1, &omega_est_1, &acc_est_1);

    const Eigen::Vector3d omega_S_true = 0.5 * (omega_est + omega_est_1);
    const double theta_half = omega_S_true.norm() * 0.5 * dt;
    const double sinc_theta_half = okvis::kinematics::sinc(theta_half);
    const double cos_theta_half = cos(theta_half);
    dq.vec() = sinc_theta_half * omega_S_true * 0.5 * dt;
    dq.w() = cos_theta_half;
    Eigen::Quaterniond Delta_q_1 = Delta_q * dq;
    // rotation matrix integral:
    const Eigen::Matrix3d C = Delta_q.toRotationMatrix();  // DCM from Si to S0
    const Eigen::Matrix3d C_1 =
        Delta_q_1.toRotationMatrix();  // DCM from S_{i+1} to S0

    const Eigen::Matrix3d C_integral_1 = C_integral + 0.5 * (C + C_1) * dt;
    const Eigen::Vector3d acc_integral_1 =
        acc_integral + 0.5 * (C * acc_est + C_1 * acc_est_1) * dt;
    // rotation matrix double integral:
    C_doubleintegral += 0.5 * (C_integral + C_integral_1) *
                        dt;  // == C_integral*dt + 0.25*(C + C_1)*dt*dt;
    acc_doubleintegral +=
        0.5 * (acc_integral + acc_integral_1) *
        dt;  //==acc_integral*dt + 0.25*(C + C_1)*acc_S_true*dt*dt;

    if (jacobian) {
      dalpha_dT_g_1 =
          dalpha_dT_g +
          0.5 * dt *
              (C * iem.invTg() * dmatrix3_dvector9_multiply(omega_est) +
               C_1 * iem.invTg() * dmatrix3_dvector9_multiply(omega_est_1));
      dalpha_dT_s_1 =
          dalpha_dT_s +
          0.5 * dt *
              (C * iem.invTg() * dmatrix3_dvector9_multiply(acc_est) +
               C_1 * iem.invTg() * dmatrix3_dvector9_multiply(acc_est_1));
      dalpha_dT_a_1 =
          dalpha_dT_a +
          0.5 * dt *
              (C * invTgsa * dmatrix3_dvector9_multiply(acc_est) +
               C_1 * invTgsa * dmatrix3_dvector9_multiply(acc_est_1));

      dv_db_g_1 =
          dv_db_g +
          0.5 * dt *
              (okvis::kinematics::crossMx(C * acc_est) * C_integral +
               okvis::kinematics::crossMx(C_1 * acc_est_1) * C_integral_1) *
              iem.invTg();

      dv_dT_g_1 =
          dv_dT_g +
          0.5 * dt *
              (okvis::kinematics::crossMx(C * acc_est) * dalpha_dT_g +
               okvis::kinematics::crossMx(C_1 * acc_est_1) * dalpha_dT_g_1);

      dv_dT_s_1 =
          dv_dT_s +
          0.5 * dt *
              (okvis::kinematics::crossMx(C * acc_est) * dalpha_dT_s +
               okvis::kinematics::crossMx(C_1 * acc_est_1) * dalpha_dT_s_1);

      dv_dT_a_1 =
          dv_dT_a +
          0.5 * dt *
              (C * iem.invTa() * dmatrix3_dvector9_multiply(acc_est) +
               C_1 * iem.invTa() * dmatrix3_dvector9_multiply(acc_est_1)) +
          0.5 * dt *
              (okvis::kinematics::crossMx(C * acc_est) * dalpha_dT_a +
               okvis::kinematics::crossMx(C_1 * acc_est_1) * dalpha_dT_a_1);
      dp_db_g += 0.5 * dt * (dv_db_g + dv_db_g_1);
      dp_dT_g += 0.5 * dt * (dv_dT_g + dv_dT_g_1);
      dp_dT_s += 0.5 * dt * (dv_dT_s + dv_dT_s_1);
      dp_dT_a += 0.5 * dt * (dv_dT_a + dv_dT_a_1);
    }

    // covariance propagation of \delta[p^W, \alpha, v^W, b_g, b_a, \vec[T_g,
    // T_s, T_a], \delta g^W.
    if (covariance) {
      linPoint_1.head<3>() =
          r_0 + v_WS * Delta_t +
          C_WS_0 * (acc_doubleintegral) + 0.5 * g_W * Delta_t * Delta_t;
      linPoint_1.tail<3>() = v_WS + C_WS_0 * (acc_integral_1) + g_W * Delta_t;

      Eigen::MatrixXd
          F_delta = Eigen::MatrixXd::Identity(covRows, covRows);

      F_delta.block<3, 3>(3, 9) = -0.5 * dt * (C_1 + C) * iem.invTg();
      F_delta.block<3, 3>(3, 12) = 0.5 * dt * (C_1 + C) * invTgsa;

      F_delta.block<3, 3>(6, 9) = 0.25 * dt * dt *
                                  okvis::kinematics::crossMx(C_1 * acc_est_1) *
                                  (C + C_1) * iem.invTg();
      F_delta.block<3, 3>(6, 12) =
          -0.5 * dt * (C + C_1) * iem.invTa() -
          0.25 * pow(dt, 2) * okvis::kinematics::crossMx(C_1 * acc_est_1) *
              (C + C_1) * invTgsa;

      if (use_first_estimate) {
        F_delta.block<3, 3>(6, 3) = okvis::kinematics::crossMx(
            -C_WS_0.transpose() *
            (linPoint_1.tail<3>() - linPoint.tail<3>() - g_W * dt));  // vq
        F_delta.block<3, 3>(0, 3) = okvis::kinematics::crossMx(
            -C_WS_0.transpose() *
            (linPoint_1.head<3>() - linPoint.head<3>() - linPoint.tail<3>() * dt -
             0.5 * g_W * dt * dt));  // pq
      } else {
        F_delta.block<3, 3>(6, 3) = -okvis::kinematics::crossMx(
            0.5 * (C * acc_est + C_1 * acc_est_1) * dt);  // vq
        F_delta.block<3, 3>(0, 3) = 0.5 * dt * F_delta.block<3, 3>(6, 3);  // pq
      }

      F_delta.block<3, 3>(0, 6) = Eigen::Matrix3d::Identity() * dt;
      F_delta.block<3, 3>(0, 9) = 0.5 * dt * F_delta.block<3, 3>(6, 9);
      F_delta.block<3, 3>(0, 12) = 0.5 * dt * F_delta.block<3, 3>(6, 12);

      if (covRows >= (int)(Imu_BG_BA_TG_TS_TA::getMinimalDim() + kNavErrorStateDim)) {
        F_delta.block<3, 9>(3, 15) =
            -0.5 * dt *
            (C * iem.invTg() * dmatrix3_dvector9_multiply(omega_est) +
             C_1 * iem.invTg() * dmatrix3_dvector9_multiply(omega_est_1));
        F_delta.block<3, 9>(3, 24) =
            -0.5 * dt *
            (C * iem.invTg() * dmatrix3_dvector9_multiply(acc_est) +
             C_1 * iem.invTg() * dmatrix3_dvector9_multiply(acc_est_1));
        F_delta.block<3, 9>(3, 33) =
            0.5 * dt *
            (C * invTgsa * dmatrix3_dvector9_multiply(acc_est) +
             C_1 * invTgsa * dmatrix3_dvector9_multiply(acc_est_1));

        F_delta.block<3, 9>(6, 15) =
            -0.5 * dt * okvis::kinematics::crossMx(C_1 * acc_est_1) *
            F_delta.block<3, 9>(3, 15);
        F_delta.block<3, 9>(6, 24) =
            -0.5 * dt * okvis::kinematics::crossMx(C_1 * acc_est_1) *
            F_delta.block<3, 9>(3, 24);
        F_delta.block<3, 9>(6, 33) =
            -0.5 * dt * okvis::kinematics::crossMx(C_1 * acc_est_1) *
                F_delta.block<3, 9>(3, 33) -
            0.5 * dt *
                (C * iem.invTa() * dmatrix3_dvector9_multiply(acc_est) +
                 C_1 * iem.invTa() * dmatrix3_dvector9_multiply(acc_est_1));

        F_delta.block<3, 9>(0, 15) = 0.5 * dt * F_delta.block<3, 9>(6, 15);
        F_delta.block<3, 9>(0, 24) = 0.5 * dt * F_delta.block<3, 9>(6, 24);
        F_delta.block<3, 9>(0, 33) = 0.5 * dt * F_delta.block<3, 9>(6, 33);
      }
      if (imuParams.isGravityDirectionVariable()) {
        Eigen::Matrix<double, 3, 2> dgS0_dunitgW =
            imuParams.g * C_WS_0.transpose() * normalGravity.getM();
        F_delta.block<3, 2>(0, gravityErrorStartIndex) =
            0.5 * dt * dt * dgS0_dunitgW;
        F_delta.block<3, 2>(6, gravityErrorStartIndex) = dt * dgS0_dunitgW;
      }
      P_delta = F_delta * P_delta * F_delta.transpose();
      // add noise. note the scaling effect of T_g and T_a
      Eigen::Matrix<double, 15, 15> GQG = Eigen::Matrix<double, 15, 15>::Zero();
      Eigen::Matrix<double, 15, 15> GQG_1 =
          Eigen::Matrix<double, 15, 15>::Zero();
      Eigen::Matrix3d CinvTg = C * iem.invTg();
      Eigen::Matrix3d CinvTg_1 = C_1 * iem.invTg();
      Eigen::Matrix3d CinvTa = C * iem.invTa();
      Eigen::Matrix3d CinvTa_1 = C_1 * iem.invTa();
      Eigen::Matrix3d CinvTgsa = C * invTgsa;
      Eigen::Matrix3d CinvTgsa_1 = C_1 * invTgsa;
      GQG.block<3, 3>(3, 3) = CinvTg * imuParams.sigma_g_c *
                                  imuParams.sigma_g_c * CinvTg.transpose() +
                              CinvTgsa * imuParams.sigma_a_c *
                                  imuParams.sigma_a_c * CinvTgsa.transpose();
      GQG.block<3, 3>(3, 6) = CinvTgsa * imuParams.sigma_a_c *
                              imuParams.sigma_a_c * CinvTa.transpose();
      GQG.block<3, 3>(6, 3) = GQG.block<3, 3>(3, 6).transpose();
      GQG.block<3, 3>(6, 6) = CinvTa * imuParams.sigma_a_c *
                              imuParams.sigma_a_c * CinvTa.transpose();
      GQG.block<3, 3>(9, 9) = Eigen::Matrix3d::Identity() *
                              imuParams.sigma_gw_c * imuParams.sigma_gw_c;
      GQG.block<3, 3>(12, 12) = Eigen::Matrix3d::Identity() *
                                imuParams.sigma_aw_c * imuParams.sigma_aw_c;

      GQG_1.block<3, 3>(3, 3) = CinvTg_1 * imuParams.sigma_g_c *
                                    imuParams.sigma_g_c * CinvTg_1.transpose() +
                                CinvTgsa_1 * imuParams.sigma_a_c *
                                    imuParams.sigma_a_c *
                                    CinvTgsa_1.transpose();
      GQG_1.block<3, 3>(3, 6) = CinvTgsa_1 * imuParams.sigma_a_c *
                                imuParams.sigma_a_c * CinvTa_1.transpose();
      GQG_1.block<3, 3>(6, 3) = GQG_1.block<3, 3>(3, 6).transpose();
      GQG_1.block<3, 3>(6, 6) = CinvTa_1 * imuParams.sigma_a_c *
                                imuParams.sigma_a_c * CinvTa_1.transpose();
      GQG_1.block<3, 3>(9, 9) = Eigen::Matrix3d::Identity() *
                                imuParams.sigma_gw_c * imuParams.sigma_gw_c;
      GQG_1.block<3, 3>(12, 12) = Eigen::Matrix3d::Identity() *
                                  imuParams.sigma_aw_c * imuParams.sigma_aw_c;

      P_delta.topLeftCorner<15, 15>() +=
          0.5 * dt *
          (F_delta.topLeftCorner<15, 15>() * GQG *
               F_delta.topLeftCorner<15, 15>().transpose() +
           GQG_1);

      linPoint = linPoint_1;
    }

    // memory shift
    Delta_q = Delta_q_1;
    C_integral = C_integral_1;
    acc_integral = acc_integral_1;

    if (jacobian) {
      dalpha_dT_g = dalpha_dT_g_1;
      dalpha_dT_s = dalpha_dT_s_1;
      dalpha_dT_a = dalpha_dT_a_1;

      dv_db_g = dv_db_g_1;
      dv_dT_g = dv_dT_g_1;
      dv_dT_s = dv_dT_s_1;
      dv_dT_a = dv_dT_a_1;
    }

    time = nexttime;
    ++i;
    if (nexttime == t_end) break;
  }

  // actual propagation output:
  T_WS.set(r_0 + v_WS * Delta_t +
               C_WS_0 * (acc_doubleintegral) + 0.5 * g_W * Delta_t * Delta_t,
           q_WS_0 * Delta_q);
  v_WS += C_WS_0 * (acc_integral) + g_W * Delta_t;

  // assign Jacobian, if requested
  if (jacobian) {
    Eigen::MatrixXd& F = *jacobian;
    F.setIdentity();  // holds for all states, including d/dalpha, d/db_g,
                      // d/db_a

    F.block<3, 3>(0, 6) = Eigen::Matrix3d::Identity() * Delta_t;
    F.block<3, 3>(0, 9) = C_WS_0 * dp_db_g;
    dp_db_a = C_doubleintegral * iem.invTa() + dp_db_g * iem.Ts() * iem.invTa();
    F.block<3, 3>(0, 12) = -C_WS_0 * dp_db_a;

    dalpha_db_g = C_integral * iem.invTg();
    F.block<3, 3>(3, 9) = -C_WS_0 * dalpha_db_g;
    dalpha_db_a = C_integral * invTgsa;
    F.block<3, 3>(3, 12) = C_WS_0 * dalpha_db_a;

    if (use_first_estimate) {
      linPoint = *positionVelocityLin;
      F.block<3, 3>(0, 3) = okvis::kinematics::crossMx(
          -(linPoint_1.head<3>() - linPoint.head<3>() - linPoint.tail<3>() * Delta_t -
            0.5 * g_W * Delta_t * Delta_t));  // pq
      F.block<3, 3>(6, 3) = okvis::kinematics::crossMx(
          -(linPoint_1.tail<3>() - linPoint.tail<3>() - g_W * Delta_t));  // vq
    } else {
      F.block<3, 3>(0, 3) =
          -okvis::kinematics::crossMx(C_WS_0 * acc_doubleintegral);
      F.block<3, 3>(6, 3) = -okvis::kinematics::crossMx(C_WS_0 * acc_integral);
    }
    F.block<3, 3>(6, 9) = C_WS_0 * dv_db_g;
    dv_db_a = C_integral * iem.invTa() + dv_db_g * iem.Ts() * iem.invTa();
    F.block<3, 3>(6, 12) = -C_WS_0 * dv_db_a;

    if (covRows >=
        (int)(Imu_BG_BA_TG_TS_TA::getMinimalDim() + kNavErrorStateDim)) {
      F.block<3, 9>(0, 15) = C_WS_0 * dp_dT_g;
      F.block<3, 9>(0, 24) = C_WS_0 * dp_dT_s;
      F.block<3, 9>(0, 33) = -C_WS_0 * dp_dT_a;
      F.block<3, 9>(3, 15) = -C_WS_0 * dalpha_dT_g;
      F.block<3, 9>(3, 24) = -C_WS_0 * dalpha_dT_s;
      F.block<3, 9>(3, 33) = C_WS_0 * dalpha_dT_a;
      F.block<3, 9>(6, 15) = C_WS_0 * dv_dT_g;
      F.block<3, 9>(6, 24) = C_WS_0 * dv_dT_s;
      F.block<3, 9>(6, 33) = -C_WS_0 * dv_dT_a;
    }
    if (imuParams.isGravityDirectionVariable()) {
      F.block<3, 2>(0, gravityErrorStartIndex) =
          0.5 * Delta_t * Delta_t * imuParams.g * normalGravity.getM();
      F.block<3, 2>(6, gravityErrorStartIndex) =
          Delta_t * imuParams.g * normalGravity.getM();
    }
  }
  // overall covariance, if requested
  if (covariance) {
    // transform from local increments to actual states
    *covariance = P_delta;
    scaleBlockRows(C_WS_0, 3, covariance);
    scaleBlockCols(C_WS_0.transpose(), 3, covariance);
  }
  return i;
}

int ImuOdometry::propagation(
    const okvis::ImuMeasurementDeque& imuMeasurements,
    const okvis::ImuParameters& imuParams,
    okvis::kinematics::Transformation& T_WS, Eigen::Vector3d& v_WS,
    Imu_BG_BA_MG_TS_MA &iem, const okvis::Time& t_start,
    const okvis::Time& t_end,
    Eigen::MatrixXd* covariance,
    Eigen::MatrixXd* jacobian,
    const Eigen::Matrix<double, 6, 1>* positionVelocityLin) {
  okvis::Time time = t_start;

  // sanity check:
  if (imuMeasurements.front().timeStamp > time) {
    LOG(WARNING)
        << "IMU front meas timestamp is greater than start integration time "
        << imuMeasurements.front().timeStamp << " " << time;
  }

  if (imuMeasurements.back().timeStamp < t_end) {
    LOG(WARNING) << "Imu last reading has an epoch "
                 << imuMeasurements.back().timeStamp
                 << " less than propagation right target " << t_end;
    return -1;
  }

  // initial condition at sensor frame SO
  const Eigen::Matrix3d C_WS_0 = T_WS.C();
  const Eigen::Vector3d g_W = imuParams.g * imuParams.gravityDirection();

  Eigen::Matrix<double, 6, 1> posVelLinPointAtStart;
  if (positionVelocityLin != nullptr) {
    posVelLinPointAtStart = *positionVelocityLin;
  } else {
    posVelLinPointAtStart << T_WS.r(), v_WS;
  }

  iem.resetPreintegration();
  iem.setPosVelLinPoint(posVelLinPointAtStart);

  Eigen::MatrixXd P_delta;
  bool computeCovariance = covariance || jacobian;
  if (computeCovariance) {
    P_delta = *covariance;
    scaleBlockRows(C_WS_0.transpose(), 3, &P_delta);
    scaleBlockCols(C_WS_0, 3, &P_delta);
  }

  NormalVectorElement normalGravity(imuParams.gravityDirection());

  double Delta_t = 0;  // integrated time up to Si since S0 frame
  bool hasStarted = false;
  int i = 0;

  iem.setInitialStateAndCov(T_WS, v_WS, g_W, P_delta);

  for (okvis::ImuMeasurementDeque::const_iterator it = imuMeasurements.begin();
       it != imuMeasurements.end(); ++it) {
    Eigen::Vector3d omega_S_0 = it->measurement.gyroscopes;
    Eigen::Vector3d acc_S_0 = it->measurement.accelerometers;
    // access out of range element won't happen becsuse the loop breaks once
    // nexttime==t_end
    Eigen::Vector3d omega_S_1 = (it + 1)->measurement.gyroscopes;
    Eigen::Vector3d acc_S_1 = (it + 1)->measurement.accelerometers;

    // time delta
    okvis::Time nexttime;
    if ((it + 1) == imuMeasurements.end()) {
      nexttime = t_end;
    } else
      nexttime = (it + 1)->timeStamp;
    double dt = (nexttime - time).toSec();

    if (t_end < nexttime) {
      double interval = (nexttime - it->timeStamp).toSec();
      nexttime = t_end;
      dt = (nexttime - time).toSec();
      const double r = dt / interval;
      omega_S_1 = ((1.0 - r) * omega_S_0 + r * omega_S_1).eval();
      acc_S_1 = ((1.0 - r) * acc_S_0 + r * acc_S_1).eval();
    }

    if (dt <= 0.0) {
      continue;
    }
    Delta_t += dt;

    if (!hasStarted) {
      hasStarted = true;
      const double r = dt / (nexttime - it->timeStamp).toSec();
      omega_S_0 = (r * omega_S_0 + (1.0 - r) * omega_S_1).eval();
      acc_S_0 = (r * acc_S_0 + (1.0 - r) * acc_S_1).eval();
    }

    // ensure integrity
    double sigma_g_c = imuParams.sigma_g_c;
    double sigma_a_c = imuParams.sigma_a_c;

    if (fabs(omega_S_0[0]) > imuParams.g_max ||
        fabs(omega_S_0[1]) > imuParams.g_max ||
        fabs(omega_S_0[2]) > imuParams.g_max ||
        fabs(omega_S_1[0]) > imuParams.g_max ||
        fabs(omega_S_1[1]) > imuParams.g_max ||
        fabs(omega_S_1[2]) > imuParams.g_max) {
      sigma_g_c *= 100;
      VLOG(0) << "gyr saturation";
    }

    if (fabs(acc_S_0[0]) > imuParams.a_max ||
        fabs(acc_S_0[1]) > imuParams.a_max ||
        fabs(acc_S_0[2]) > imuParams.a_max ||
        fabs(acc_S_1[0]) > imuParams.a_max ||
        fabs(acc_S_1[1]) > imuParams.a_max ||
        fabs(acc_S_1[2]) > imuParams.a_max) {
      sigma_a_c *= 100;
      VLOG(0) << "acc saturation";
    }

    iem.computeDeltaState(dt, omega_S_0, acc_S_0, omega_S_1, acc_S_1);

    if (computeCovariance) {
      iem.computeFdelta(Delta_t, dt, normalGravity, imuParams.g,
                        imuParams.isGravityDirectionVariable());

      iem.updatePdelta(dt, sigma_g_c, sigma_a_c, imuParams.sigma_gw_c,
                       imuParams.sigma_aw_c);
    }

    iem.shiftVariables();

    time = nexttime;
    ++i;
    if (nexttime == t_end) break;
  }

  // actual propagation output:
  iem.getFinalState(Delta_t, &T_WS, &v_WS);
  if (computeCovariance) {
    iem.getFinalJacobian(jacobian, Delta_t, posVelLinPointAtStart,
                         normalGravity, imuParams.g,
                         imuParams.isGravityDirectionVariable());

    iem.getFinalCovariance(covariance, C_WS_0);
  }

  return i;
}

/**
 * @brief GQGt
 * @warning \Delta t is not multiplied.
 * @param p_WB
 * @param v_WB
 * @param sigma_g_c
 * @param sigma_a_c
 * @return GQG' output variable order [\phi, v, p]
 */
Eigen::Matrix<double, 9, 9> computeGQGt(const Eigen::Vector3d& p_WB,
                                 const Eigen::Vector3d& v_WB, double sigma_g_c,
                                 double sigma_a_c) {
  double qgc = sigma_g_c * sigma_g_c;
  double qac = sigma_a_c * sigma_a_c;
  Eigen::Matrix3d px = okvis::kinematics::crossMx(p_WB);
  Eigen::Matrix3d vx = okvis::kinematics::crossMx(v_WB);
  Eigen::Matrix<double, 9, 9> result;
  result << qgc * Eigen::Matrix3d::Identity(), -qgc * vx, -qgc * px,
      qgc * vx,  qac * Eigen::Matrix3d::Identity() - vx * vx * qgc, -vx * px * qgc,
      px * qgc, -px * vx * qgc, -px * px * qgc;
  return result;
}

int ImuOdometry::propagationRightInvariantError(
    const okvis::ImuMeasurementDeque& imuMeasurements,
    const okvis::ImuParameters& imuParams,
    okvis::kinematics::Transformation& T_WS, Eigen::Vector3d& v_WS,
    const Imu_BG_BA& iem, const okvis::Time& t_start,
    const okvis::Time& t_end, Eigen::Matrix<double, 15, 15>* covariance,
    Eigen::Matrix<double, 15, 15>* jacobian) {
  okvis::Time time = t_start;

  if (imuMeasurements.front().timeStamp > time) {
    LOG(WARNING)
        << "IMU front meas timestamp is greater than start integration time "
        << imuMeasurements.front().timeStamp << " " << time;
  }

  if (imuMeasurements.back().timeStamp < t_end) {
    LOG(WARNING) << "Imu last reading has an epoch "
                 << imuMeasurements.back().timeStamp
                 << " less than propagation right target " << t_end;
    return -1;
  }

  // initial condition
  const Eigen::Vector3d r_0 = T_WS.r();
  const Eigen::Quaterniond q_WS_0 = T_WS.q();
  const Eigen::Matrix3d C_WS_0 = T_WS.C();
  const Eigen::Vector3d g_W = Eigen::Vector3d(0, 0, -imuParams.g);
  const Eigen::Matrix3d gx = okvis::kinematics::crossMx(g_W);

  // linearization point for position p_WS, and v_WS at two subsequent steps.
  Eigen::Matrix<double, 6, 1> linPoint;
  linPoint << r_0, v_WS;
  Eigen::Matrix<double, 6, 1> linPoint_1;

  // increments (initialise with identity), denote t_start by $t_0$
  Eigen::Quaterniond Delta_q(1, 0, 0, 0);  // quaternion of DCM from Si to S0
  // integrated DCM up to Si expressed in S0 frame:
  // $\int_{t_0}^{t_i} R_S^{S_0} dt$
  Eigen::Matrix3d C_integral = Eigen::Matrix3d::Zero();
  // double integrated DCM up to Si expressed in S0 frame:
  // $\int_{t_0}^{t_i} \int_{t_0}^{s} R_S^{S_0} dt ds$
  Eigen::Matrix3d C_doubleintegral = Eigen::Matrix3d::Zero();
  // integrated acceleration up to Si expressed in S0 frame:
  // $\int_{t_0}^{t_i} R_S^{S_0} a^S dt$
  Eigen::Vector3d acc_integral = Eigen::Vector3d::Zero();
  // double integrated acceleration up to Si expressed in S0 frame:
  // $\int_{t_0}^{t_i} \int_{t_0}^{s} R_S^{S_0} a^S dt ds$
  Eigen::Vector3d acc_doubleintegral = Eigen::Vector3d::Zero();

  int covRows = 0;
  if (covariance || jacobian) {
    covRows = covariance ? covariance->rows() : jacobian->rows();
    jacobian->setIdentity();
  }
  double Delta_t = 0;  // integrated time up to Si since S0 frame
  bool hasStarted = false;
  int i = 0;

  for (okvis::ImuMeasurementDeque::const_iterator it = imuMeasurements.begin();
       it != imuMeasurements.end(); ++it) {
    Eigen::Vector3d omega_S_0 = it->measurement.gyroscopes;
    Eigen::Vector3d acc_S_0 = it->measurement.accelerometers;
    // access out of range element won't happen becsuse the loop breaks once
    // nexttime==t_end
    Eigen::Vector3d omega_S_1 = (it + 1)->measurement.gyroscopes;
    Eigen::Vector3d acc_S_1 = (it + 1)->measurement.accelerometers;

    // time delta
    okvis::Time nexttime;
    if ((it + 1) == imuMeasurements.end()) {
      nexttime = t_end;
    } else
      nexttime = (it + 1)->timeStamp;
    double dt = (nexttime - time).toSec();

    if (t_end < nexttime) {
      double interval = (nexttime - it->timeStamp).toSec();
      nexttime = t_end;
      dt = (nexttime - time).toSec();
      const double r = dt / interval;
      omega_S_1 = ((1.0 - r) * omega_S_0 + r * omega_S_1).eval();
      acc_S_1 = ((1.0 - r) * acc_S_0 + r * acc_S_1).eval();
    }

    if (dt <= 0.0) {
      continue;
    }
    Delta_t += dt;

    if (!hasStarted) {
      hasStarted = true;
      const double r = dt / (nexttime - it->timeStamp).toSec();
      omega_S_0 = (r * omega_S_0 + (1.0 - r) * omega_S_1).eval();
      acc_S_0 = (r * acc_S_0 + (1.0 - r) * acc_S_1).eval();
    }

    // ensure integrity
    double sigma_g_c = imuParams.sigma_g_c;
    double sigma_a_c = imuParams.sigma_a_c;

    if ((omega_S_0.lpNorm<Eigen::Infinity>() > imuParams.g_max) ||
        (omega_S_1.lpNorm<Eigen::Infinity>() > imuParams.g_max)) {
      sigma_g_c *= 100;
      LOG(WARNING) << "gyr saturation";
    }

    if ((acc_S_0.lpNorm<Eigen::Infinity>() > imuParams.a_max) ||
        (acc_S_1.lpNorm<Eigen::Infinity>() > imuParams.a_max)) {
      sigma_a_c *= 100;
      LOG(WARNING) << "acc saturation";
    }

    Eigen::Quaterniond dq;  // q_{S_{i+1}}^{S_i}
    Eigen::Vector3d omega_est;
    Eigen::Vector3d acc_est;
    iem.correct(omega_S_0, acc_S_0, &omega_est, &acc_est);

    Eigen::Vector3d omega_est_1;
    Eigen::Vector3d acc_est_1;
    iem.correct(omega_S_1, acc_S_1, &omega_est_1, &acc_est_1);

    const Eigen::Vector3d omega_S_true = 0.5 * (omega_est + omega_est_1);
    const double theta_half = omega_S_true.norm() * 0.5 * dt;
    const double sinc_theta_half = okvis::kinematics::sinc(theta_half);
    const double cos_theta_half = cos(theta_half);
    dq.vec() = sinc_theta_half * omega_S_true * 0.5 * dt;
    dq.w() = cos_theta_half;
    Eigen::Quaterniond Delta_q_1 = Delta_q * dq;

    const Eigen::Matrix3d C = Delta_q.toRotationMatrix();  // DCM from Si to S0
    const Eigen::Matrix3d C_1 = Delta_q_1.toRotationMatrix();  // DCM from S_{i+1} to S0
    const Eigen::Matrix3d C_integral_1 = C_integral + 0.5 * (C + C_1) * dt;
    const Eigen::Vector3d acc_integral_1 =
        acc_integral + 0.5 * (C * acc_est + C_1 * acc_est_1) * dt;

    C_doubleintegral += 0.5 * (C_integral + C_integral_1) * dt;
    acc_doubleintegral += 0.5 * (acc_integral + acc_integral_1) * dt;
    if (covariance) {
      // https://github.com/RossHartley/invariant-ekf/blob/master/src/InEKF.cpp#L154-L187
      linPoint_1.head<3>() = r_0 + v_WS * Delta_t + C_WS_0 * acc_doubleintegral +
                       0.5 * g_W * Delta_t * Delta_t;
      linPoint_1.tail<3>() = v_WS + C_WS_0 * acc_integral_1 + g_W * Delta_t;

      Eigen::MatrixXd F_delta = Eigen::MatrixXd::Identity(covRows, covRows);
      double dt2 = dt * dt;
      F_delta.block<3, 3>(3, 0) = gx * dt;
      F_delta.block<3, 3>(6, 0) = gx * 0.5 * dt2;
      F_delta.block<3, 3>(6, 3) = Eigen::Matrix3d::Identity() * dt;

      // use midpoint approach to calculate
      // F_delta = exp(F\Delta t) \approx (F_delta(t_i) + F_delta(t_{i+1})) / 2
      Eigen::Matrix3d R_WS = C_WS_0 * C;
      Eigen::Matrix3d R_WS_1 = C_WS_0 * C_1;
      F_delta.block<3, 3>(3, 9) = -0.5 * dt * (R_WS + R_WS_1);
      F_delta.block<3, 3>(6, 9) = -0.25 * dt2 * (R_WS + R_WS_1);

      // bg
      F_delta.block<3, 3>(0, 12) = F_delta.block<3, 3>(3, 9);
      F_delta.block<3, 3>(3, 12) =
          -0.25 * dt2 * gx * (R_WS + R_WS_1) -
          dt * 0.5 *
              (okvis::kinematics::crossMx(linPoint.tail<3>()) * R_WS +
               okvis::kinematics::crossMx(linPoint_1.tail<3>()) * R_WS_1);
      F_delta.block<3, 3>(6, 12) =
          -dt * 0.5 *
              (okvis::kinematics::crossMx(linPoint.head<3>()) * R_WS +
               okvis::kinematics::crossMx(linPoint_1.head<3>()) * R_WS_1) -
          dt2 * 0.25 *
              (okvis::kinematics::crossMx(linPoint.tail<3>()) * R_WS +
               okvis::kinematics::crossMx(linPoint_1.tail<3>()) * R_WS_1) -
          dt2 / 12 * gx * (R_WS + R_WS_1);
      *covariance = F_delta * *covariance * F_delta.transpose();
      // add noise
      Eigen::Matrix<double, 15, 15> GQGt =
          Eigen::Matrix<double, 15, 15>::Zero();
      Eigen::Matrix<double, 15, 15> GQGt_1 =
          Eigen::Matrix<double, 15, 15>::Zero();
      GQGt.topLeftCorner<9, 9>() = computeGQGt(
          linPoint.head<3>(), linPoint.tail<3>(), imuParams.sigma_g_c, imuParams.sigma_a_c);
      GQGt.block<3, 3>(9, 9) = Eigen::Matrix3d::Identity() *
                               imuParams.sigma_aw_c * imuParams.sigma_aw_c;
      GQGt.block<3, 3>(12, 12) = Eigen::Matrix3d::Identity() *
                                 imuParams.sigma_gw_c * imuParams.sigma_gw_c;

      GQGt_1.topLeftCorner<9, 9>() =
          computeGQGt(linPoint_1.head<3>(), linPoint_1.tail<3>(), imuParams.sigma_g_c,
                      imuParams.sigma_a_c);
      GQGt_1.block<3, 3>(9, 9) = Eigen::Matrix3d::Identity() *
                                 imuParams.sigma_aw_c * imuParams.sigma_aw_c;
      GQGt_1.block<3, 3>(12, 12) = Eigen::Matrix3d::Identity() *
                                   imuParams.sigma_gw_c * imuParams.sigma_gw_c;
      *covariance += 0.5 * dt * (F_delta * GQGt * F_delta.transpose() + GQGt_1);

      jacobian->block<3, 3>(3, 0) = gx * Delta_t;
      jacobian->block<3, 3>(6, 0) = gx * Delta_t * Delta_t * 0.5;
      jacobian->block<3, 3>(6, 3) = Eigen::Matrix3d::Identity() * Delta_t;
      jacobian->topRightCorner(9, covRows - 9) =
          F_delta.topLeftCorner<9, 9>() *
              jacobian->topRightCorner(9, covRows - 9) +
          F_delta.topRightCorner(9, covRows - 9);

      linPoint = linPoint_1;
    }

    // memory shift
    Delta_q = Delta_q_1;
    C_integral = C_integral_1;
    acc_integral = acc_integral_1;

    time = nexttime;
    ++i;
    if (nexttime == t_end) break;
  }

  // actual propagation output:
  T_WS.set(r_0 + v_WS * Delta_t + C_WS_0 * acc_doubleintegral +
               0.5 * g_W * Delta_t * Delta_t,
           q_WS_0 * Delta_q);
  v_WS += C_WS_0 * acc_integral + g_W * Delta_t;
  return i;
}

template<typename ImuModelT>
int ImuOdometry::propagationBackward(
    const okvis::ImuMeasurementDeque& imuMeasurements,
    const okvis::ImuParameters& imuParams,
    okvis::kinematics::Transformation& T_WS, Eigen::Vector3d& v_WS,
    const ImuModelT& iem, const okvis::Time& t_start,
    const okvis::Time& t_end) {
  okvis::Time time = t_start;

  // sanity check:
  if (imuMeasurements.front().timeStamp > t_end) {
    LOG(WARNING) << "IMU front meas timestamp is greater than end integration "
                    "time in backward mode"
                 << imuMeasurements.front().timeStamp << " " << t_end;
  }
  if (imuMeasurements.back().timeStamp < time)
    return -1;  // nothing to do...

  // initial condition at sensor frame SO
  const Eigen::Vector3d r_0 = T_WS.r();
  const Eigen::Quaterniond q_WS_0 = T_WS.q();
  const Eigen::Matrix3d C_WS_0 = T_WS.C();

  // increments (initialise with identity)
  // denote t_start by $t_0$
  Eigen::Quaterniond Delta_q(1, 0, 0, 0);  // quaternion of DCM from Si to S0
  //$\int_{t_0}^{t_i} R_S^{S_0} dt$
  Eigen::Matrix3d C_integral =
      Eigen::Matrix3d::Zero();  // integrated DCM up to Si expressed in S0 frame
  // $\int_{t_0}^{t_i} \int_{t_0}^{t} R_S^{S_0} dt dt$
  Eigen::Matrix3d C_doubleintegral =
      Eigen::Matrix3d::Zero();  // double integrated DCM up to Si expressed in
                                // S0 frame
  // $\int_{t_0}^{t_i} R_S^{S_0} a^S dt$
  Eigen::Vector3d acc_integral =
      Eigen::Vector3d::Zero();  // integrated acceleration up to Si expressed in
                                // S0 frame
  // $\int_{t_0}^{t_i} \int_{t_0}^{t} R_S^{S_0} a^S dt dt$
  Eigen::Vector3d acc_doubleintegral =
      Eigen::Vector3d::Zero();  // double integrated acceleration up to Si
                                // expressed in S0 frame

  double Delta_t = 0;  // integrated time up to Si since S0 frame
  bool hasStarted = false;
  int i = 0;

  for (okvis::ImuMeasurementDeque::const_reverse_iterator it =
           imuMeasurements.rbegin();
       it != imuMeasurements.rend(); ++it) {
    Eigen::Vector3d omega_S_0 = it->measurement.gyroscopes;
    Eigen::Vector3d acc_S_0 = it->measurement.accelerometers;
    // access out of range element won't happen becsuse the loop breaks once
    // nexttime==t_end
    Eigen::Vector3d omega_S_1 = (it + 1)->measurement.gyroscopes;
    Eigen::Vector3d acc_S_1 = (it + 1)->measurement.accelerometers;

    // time delta
    okvis::Time nexttime;
    if ((it + 1) == imuMeasurements.rend()) {
      nexttime = t_end;
    } else
      nexttime = (it + 1)->timeStamp;
    double dt = (nexttime - time).toSec();

    if (t_end > nexttime) {
      double interval = (nexttime - it->timeStamp).toSec();
      nexttime = t_end;
      dt = (nexttime - time).toSec();
      const double r = dt / interval;
      omega_S_1 = ((1.0 - r) * omega_S_0 + r * omega_S_1).eval();
      acc_S_1 = ((1.0 - r) * acc_S_0 + r * acc_S_1).eval();
    }

    if (dt >= 0.0) {
      continue;
    }
    Delta_t += dt;

    if (!hasStarted) {
      hasStarted = true;
      const double r = dt / (nexttime - it->timeStamp).toSec();
      omega_S_0 = (r * omega_S_0 + (1.0 - r) * omega_S_1).eval();
      acc_S_0 = (r * acc_S_0 + (1.0 - r) * acc_S_1).eval();
    }

    // ensure integrity
    double sigma_g_c = imuParams.sigma_g_c;
    double sigma_a_c = imuParams.sigma_a_c;

    if (fabs(omega_S_0[0]) > imuParams.g_max ||
        fabs(omega_S_0[1]) > imuParams.g_max ||
        fabs(omega_S_0[2]) > imuParams.g_max ||
        fabs(omega_S_1[0]) > imuParams.g_max ||
        fabs(omega_S_1[1]) > imuParams.g_max ||
        fabs(omega_S_1[2]) > imuParams.g_max) {
      sigma_g_c *= 100;
      VLOG(0) << "gyr saturation";
    }

    if (fabs(acc_S_0[0]) > imuParams.a_max ||
        fabs(acc_S_0[1]) > imuParams.a_max ||
        fabs(acc_S_0[2]) > imuParams.a_max ||
        fabs(acc_S_1[0]) > imuParams.a_max ||
        fabs(acc_S_1[1]) > imuParams.a_max ||
        fabs(acc_S_1[2]) > imuParams.a_max) {
      sigma_a_c *= 100;
      LOG(WARNING) << "acc saturation";
    }

    // actual propagation
    Eigen::Quaterniond dq;  // q_{S_{i+1}}^{S_i}

    Eigen::Vector3d omega_est;
    Eigen::Vector3d acc_est;
    iem.correct(omega_S_0, acc_S_0, &omega_est, &acc_est);

    Eigen::Vector3d omega_est_1;
    Eigen::Vector3d acc_est_1;
    iem.correct(omega_S_1, acc_S_1, &omega_est_1, &acc_est_1);

    const Eigen::Vector3d omega_S_true = 0.5 * (omega_est + omega_est_1);
    const double theta_half = omega_S_true.norm() * 0.5 * dt;
    const double sinc_theta_half = okvis::kinematics::sinc(theta_half);
    const double cos_theta_half = cos(theta_half);
    dq.vec() = sinc_theta_half * omega_S_true * 0.5 * dt;
    dq.w() = cos_theta_half;
    Eigen::Quaterniond Delta_q_1 = Delta_q * dq;
    // rotation matrix integral:
    const Eigen::Matrix3d C = Delta_q.toRotationMatrix();  // DCM from Si to S0
    const Eigen::Matrix3d C_1 =
        Delta_q_1.toRotationMatrix();  // DCM from S_{i+1} to S0
    const Eigen::Vector3d acc_S_true = 0.5 * (acc_est + acc_est_1);
    const Eigen::Matrix3d C_integral_1 = C_integral + 0.5 * (C + C_1) * dt;
    const Eigen::Vector3d acc_integral_1 =
        acc_integral + 0.5 * (C + C_1) * acc_S_true * dt;
    // rotation matrix double integral:
    C_doubleintegral += 0.5 * (C_integral + C_integral_1) *
                        dt;  // == C_integral*dt + 0.25*(C + C_1)*dt*dt;
    acc_doubleintegral +=
        0.5 * (acc_integral + acc_integral_1) *
        dt;  //==acc_integral*dt + 0.25*(C + C_1)*acc_S_true*dt*dt;

    // memory shift
    Delta_q = Delta_q_1;
    C_integral = C_integral_1;
    acc_integral = acc_integral_1;

    time = nexttime;
    ++i;
    if (nexttime == t_end) break;
  }

  // actual propagation output:
  const Eigen::Vector3d g_W =
      imuParams.g * Eigen::Vector3d(0, 0, 6371009).normalized();
  T_WS.set(r_0 + v_WS * Delta_t +
               C_WS_0 * (acc_doubleintegral)-0.5 * g_W * Delta_t * Delta_t,
           q_WS_0 * Delta_q);
  v_WS += C_WS_0 * (acc_integral)-g_W * Delta_t;
  return i;
}

void poseAndVelocityAtObservation(
    const okvis::ImuMeasurementDeque& imuMeas,
    const Imu_BG_BA_TG_TS_TA &iem,
    const okvis::ImuParameters& imuParameters, const okvis::Time& stateEpoch,
    const okvis::Duration& featureTime, okvis::kinematics::Transformation* T_WB,
    Eigen::Vector3d *v_WB,
    okvis::ImuMeasurement* interpolatedInertialData,
    bool use_RK4) {
  const double wedge = 5e-8;
  double relFeatureTime = featureTime.toSec();
  if (use_RK4) {
    if (relFeatureTime >= wedge) {
      ImuOdometry::propagation_RungeKutta(imuMeas, imuParameters, *T_WB, *v_WB,
                                          iem, stateEpoch,
                                          stateEpoch + featureTime);
    } else if (relFeatureTime <= -wedge) {
      ImuOdometry::propagationBackward_RungeKutta(imuMeas, imuParameters, *T_WB,
                                                  *v_WB, iem, stateEpoch,
                                                  stateEpoch + featureTime);
    }
    ImuOdometry::interpolateInertialData(imuMeas, iem, stateEpoch + featureTime,
                                         *interpolatedInertialData);
  } else {
    if (relFeatureTime >= wedge) {
      ImuOdometry::propagation(imuMeas, imuParameters, *T_WB, *v_WB, iem,
                               stateEpoch, stateEpoch + featureTime);
    } else if (relFeatureTime <= -wedge) {
      ImuOdometry::propagationBackward(imuMeas, imuParameters, *T_WB, *v_WB,
                                       iem, stateEpoch,
                                       stateEpoch + featureTime);
    }
    ImuOdometry::interpolateInertialData(imuMeas, iem, stateEpoch + featureTime,
                                         *interpolatedInertialData);
  }
}

void poseAndLinearVelocityAtObservation(
    const okvis::ImuMeasurementDeque& imuMeas,
    const Imu_BG_BA_TG_TS_TA &iem,
    const okvis::ImuParameters& imuParameters, const okvis::Time& stateEpoch,
    const okvis::Duration& featureTime, okvis::kinematics::Transformation* T_WB,
    Eigen::Vector3d *v_WB) {
  if (featureTime >= okvis::Duration()) {
    ImuOdometry::propagation(
        imuMeas, imuParameters, *T_WB, *v_WB, iem, stateEpoch,
        stateEpoch + featureTime);
  } else {
    ImuOdometry::propagationBackward(
        imuMeas, imuParameters, *T_WB, *v_WB, iem, stateEpoch,
        stateEpoch + featureTime);
  }
}

void poseAndVelocityAtObservation(
    const okvis::ImuMeasurementDeque& imuMeas,
    Imu_BG_BA_MG_TS_MA &iem,
    const okvis::ImuParameters& imuParameters, const okvis::Time& stateEpoch,
    const okvis::Duration& featureTime, okvis::kinematics::Transformation* T_WB,
    Eigen::Vector3d *v_WB,
    okvis::ImuMeasurement* interpolatedInertialData,
    bool use_RK4) {
  const double wedge = 5e-8;
  double relFeatureTime = featureTime.toSec();
  if (use_RK4) {
    if (relFeatureTime >= wedge) {
      ImuOdometry::propagation_RungeKutta(imuMeas, imuParameters, *T_WB, *v_WB,
                                          iem, stateEpoch,
                                          stateEpoch + featureTime);
    } else if (relFeatureTime <= -wedge) {
      ImuOdometry::propagationBackward_RungeKutta(imuMeas, imuParameters, *T_WB,
                                                  *v_WB, iem, stateEpoch,
                                                  stateEpoch + featureTime);
    }
    ImuOdometry::interpolateInertialData(imuMeas, iem, stateEpoch + featureTime,
                                         *interpolatedInertialData);
  } else {
    if (relFeatureTime >= wedge) {
      ImuOdometry::propagation(imuMeas, imuParameters, *T_WB, *v_WB, iem,
                               stateEpoch, stateEpoch + featureTime);
    } else if (relFeatureTime <= -wedge) {
      ImuOdometry::propagationBackward(imuMeas, imuParameters, *T_WB, *v_WB,
                                       iem, stateEpoch,
                                       stateEpoch + featureTime);
    }
    ImuOdometry::interpolateInertialData(imuMeas, iem, stateEpoch + featureTime,
                                         *interpolatedInertialData);
  }
}

void poseAndLinearVelocityAtObservation(
    const okvis::ImuMeasurementDeque& imuMeas,
    Imu_BG_BA_MG_TS_MA &iem,
    const okvis::ImuParameters& imuParameters, const okvis::Time& stateEpoch,
    const okvis::Duration& featureTime, okvis::kinematics::Transformation* T_WB,
    Eigen::Vector3d *v_WB) {
  if (featureTime >= okvis::Duration()) {
    ImuOdometry::propagation(
        imuMeas, imuParameters, *T_WB, *v_WB, iem, stateEpoch,
        stateEpoch + featureTime);
  } else {
    ImuOdometry::propagationBackward(
        imuMeas, imuParameters, *T_WB, *v_WB, iem, stateEpoch,
        stateEpoch + featureTime);
  }
}
}  // namespace swift_vio
