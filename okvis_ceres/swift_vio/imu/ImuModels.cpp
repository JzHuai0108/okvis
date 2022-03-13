#include <swift_vio/imu/ImuModels.hpp>
#include <swift_vio/imu/ImuErrorModel.h>
#include <Eigen/Core>

#include <okvis/kinematics/Transformation.hpp>
namespace swift_vio {
void Imu_BG_BA::resetPreintegration() {
  // increments (initialise with identity)
  Delta_q_ = Eigen::Quaterniond(1, 0, 0, 0);
  C_integral_ = Eigen::Matrix3d::Zero();
  C_doubleintegral_ = Eigen::Matrix3d::Zero();
  acc_integral_ = Eigen::Vector3d::Zero();
  acc_doubleintegral_ = Eigen::Vector3d::Zero();

  // cross matrix accumulatrion
  cross_ = Eigen::Matrix3d::Zero();

  // sub-Jacobians
  dalpha_db_g_ = Eigen::Matrix3d::Zero();
  dv_db_g_ = Eigen::Matrix3d::Zero();
  dp_db_g_ = Eigen::Matrix3d::Zero();

  // the Jacobian of the increment (w/o biases)
  P_delta_ = Eigen::Matrix<double, 15, 15>::Zero();

  // Eigen::Matrix<double, 15, 15> F_tot;
  // F_tot.setIdentity();
}

void Imu_BG_BA::propagate(double dt, const Eigen::Vector3d &omega_S_0,
                          const Eigen::Vector3d &acc_S_0,
                          const Eigen::Vector3d &omega_S_1,
                          const Eigen::Vector3d &acc_S_1, double sigma_g_c,
                          double sigma_a_c, double sigma_gw_c,
                          double sigma_aw_c) {
  Eigen::Vector3d omega_est_0;
  Eigen::Vector3d acc_est_0;
  correct(omega_S_0, acc_S_0, &omega_est_0, &acc_est_0);

  Eigen::Vector3d omega_est_1;
  Eigen::Vector3d acc_est_1;
  correct(omega_S_1, acc_S_1, &omega_est_1, &acc_est_1);

  const Eigen::Vector3d omega_S_true = 0.5 * (omega_est_0 + omega_est_1);
  const double theta_half = omega_S_true.norm() * 0.5 * dt;
  const double sinc_theta_half = okvis::kinematics::sinc(theta_half);
  const double cos_theta_half = cos(theta_half);
  Eigen::Quaterniond dq;
  dq.vec() = sinc_theta_half * omega_S_true * 0.5 * dt;
  dq.w() = cos_theta_half;

  const Eigen::Quaterniond Delta_q_1 = Delta_q_ * dq;
  // rotation matrix integral:
  const Eigen::Matrix3d C = Delta_q_.toRotationMatrix();
  const Eigen::Matrix3d C_1 = Delta_q_1.toRotationMatrix();
  const Eigen::Vector3d acc_S_true = 0.5 * (acc_est_0 + acc_est_1);
  const Eigen::Matrix3d C_integral_1 = C_integral_ + 0.5 * (C + C_1) * dt;
  const Eigen::Vector3d acc_integral_1 =
      acc_integral_ + 0.5 * (C + C_1) * acc_S_true * dt;
  // rotation matrix double integral:
  C_doubleintegral_ += C_integral_ * dt + 0.25 * (C + C_1) * dt * dt;
  acc_doubleintegral_ +=
      acc_integral_ * dt + 0.25 * (C + C_1) * acc_S_true * dt * dt;

  // Jacobian parts
  dalpha_db_g_ +=
      C_1 * okvis::kinematics::rightJacobian(omega_S_true * dt) * dt;
  const Eigen::Matrix3d cross_1 =
      dq.inverse().toRotationMatrix() * cross_ +
      okvis::kinematics::rightJacobian(omega_S_true * dt) * dt;
  const Eigen::Matrix3d acc_S_x = okvis::kinematics::crossMx(acc_S_true);
  const Eigen::Matrix3d dv_db_g_1 =
      dv_db_g_ + 0.5 * dt * (C * acc_S_x * cross_ + C_1 * acc_S_x * cross_1);
  dp_db_g_ += dt * dv_db_g_ +
              0.25 * dt * dt * (C * acc_S_x * cross_ + C_1 * acc_S_x * cross_1);

  // covariance propagation
  Eigen::Matrix<double, 15, 15> F_delta =
      Eigen::Matrix<double, 15, 15>::Identity();
  // transform
#if 0
  F_delta.block<3, 3>(0, 3) = -okvis::kinematics::crossMx(
      acc_integral_ * dt + 0.25 * (C + C_1) * acc_S_true * dt * dt);
  F_delta.block<3, 3>(0, 6) = Eigen::Matrix3d::Identity() * dt;
  F_delta.block<3, 3>(0, 9) = dt * dv_db_g_
      + 0.25 * dt * dt * (C * acc_S_x * cross_ + C_1 * acc_S_x * cross_1);
  F_delta.block<3, 3>(0, 12) = -C_integral_ * dt
      + 0.25 * (C + C_1) * dt * dt;
  F_delta.block<3, 3>(3, 9) = -dt * C_1;
  F_delta.block<3, 3>(6, 3) = -okvis::kinematics::crossMx(
      0.5 * (C + C_1) * acc_S_true * dt);
  F_delta.block<3, 3>(6, 9) = 0.5 * dt
      * (C * acc_S_x * cross_ + C_1 * acc_S_x * cross_1);

#else
  Eigen::Matrix3d deltaCross = cross_1 - cross_;
  F_delta.block<3, 3>(0, 3) = -okvis::kinematics::crossMx(
      /*acc_integral_ * dt +*/ 0.25 * (C + C_1) * acc_S_true * dt * dt);
  F_delta.block<3, 3>(0, 6) = Eigen::Matrix3d::Identity() * dt;
  F_delta.block<3, 3>(0, 9) = /*dt * dv_db_g_ + 0.25 * dt * dt * (C * acc_S_x *
                                 cross_ + C_1 * acc_S_x * cross_1)*/
      0.25 * dt * dt * (C_1 * acc_S_x * deltaCross);
  F_delta.block<3, 3>(0, 12) =
      /*-C_integral_ * dt +*/ 0.25 * (C + C_1) * dt * dt;
  F_delta.block<3, 3>(3, 9) = -dt * C_1;
  F_delta.block<3, 3>(6, 3) =
      -okvis::kinematics::crossMx(0.5 * (C + C_1) * acc_S_true * dt);
  F_delta.block<3, 3>(
      6, 9) = /*0.5 * dt* (C * acc_S_x * cross_ + C_1 * acc_S_x * cross_1)*/
      0.5 * dt * (C_1 * acc_S_x * deltaCross);
#endif

  F_delta.block<3, 3>(6, 12) = -0.5 * (C + C_1) * dt;
  P_delta_ = F_delta * P_delta_ * F_delta.transpose();
  // add noise. Note that transformations with rotation matrices can be ignored,
  // since the noise is isotropic.
  // F_tot = F_delta*F_tot;
  const double sigma2_dalpha = dt * sigma_g_c * sigma_g_c;
  P_delta_(3, 3) += sigma2_dalpha;
  P_delta_(4, 4) += sigma2_dalpha;
  P_delta_(5, 5) += sigma2_dalpha;
  const double sigma2_v = dt * sigma_a_c * sigma_a_c;
  P_delta_(6, 6) += sigma2_v;
  P_delta_(7, 7) += sigma2_v;
  P_delta_(8, 8) += sigma2_v;
  const double sigma2_p = 0.5 * dt * dt * sigma2_v;
  P_delta_(0, 0) += sigma2_p;
  P_delta_(1, 1) += sigma2_p;
  P_delta_(2, 2) += sigma2_p;
  const double sigma2_b_g = dt * sigma_gw_c * sigma_gw_c;
  P_delta_(9, 9) += sigma2_b_g;
  P_delta_(10, 10) += sigma2_b_g;
  P_delta_(11, 11) += sigma2_b_g;
  const double sigma2_b_a = dt * sigma_aw_c * sigma_aw_c;
  P_delta_(12, 12) += sigma2_b_a;
  P_delta_(13, 13) += sigma2_b_a;
  P_delta_(14, 14) += sigma2_b_a;

  Delta_q_ = Delta_q_1;
  C_integral_ = C_integral_1;
  acc_integral_ = acc_integral_1;
  cross_ = cross_1;
  dv_db_g_ = dv_db_g_1;
}

void Imu_BG_BA_TG_TS_TA::propagate(double dt,
                                   const Eigen::Vector3d &omega_S_0,
                                   const Eigen::Vector3d &acc_S_0,
                                   const Eigen::Vector3d &omega_S_1,
                                   const Eigen::Vector3d &acc_S_1,
                                   double sigma_g_c, double sigma_a_c,
                                   double sigma_gw_c, double sigma_aw_c) {
  // intermediate variables, must assign values to them before using.
  Eigen::Matrix<double, 3, 9> dalpha_dT_g_1;
  Eigen::Matrix<double, 3, 9> dalpha_dT_s_1;
  Eigen::Matrix<double, 3, 9> dalpha_dT_a_1;
  Eigen::Matrix3d dv_db_g_1;
  Eigen::Matrix<double, 3, 9> dv_dT_g_1;
  Eigen::Matrix<double, 3, 9> dv_dT_s_1;
  Eigen::Matrix<double, 3, 9> dv_dT_a_1;

  Eigen::Vector3d omega_est_0;
  Eigen::Vector3d acc_est_0;
  correct(omega_S_0, acc_S_0, &omega_est_0, &acc_est_0);

  Eigen::Vector3d omega_est_1;
  Eigen::Vector3d acc_est_1;
  correct(omega_S_1, acc_S_1, &omega_est_1, &acc_est_1);
  const Eigen::Vector3d omega_S_true = 0.5 * (omega_est_0 + omega_est_1);
  const double theta_half = omega_S_true.norm() * 0.5 * dt;
  const double sinc_theta_half = okvis::kinematics::sinc(theta_half);
  const double cos_theta_half = cos(theta_half);
  Eigen::Quaterniond dq;
  dq.vec() = sinc_theta_half * omega_S_true * 0.5 * dt;
  dq.w() = cos_theta_half;
  Eigen::Quaterniond Delta_q_1 = Delta_q_ * dq;
  // rotation matrix integral:
  const Eigen::Matrix3d C = Delta_q_.toRotationMatrix(); // DCM from Si to S0
  const Eigen::Matrix3d C_1 =
      Delta_q_1.toRotationMatrix(); // DCM from S_{i+1} to S0

  const Eigen::Matrix3d C_integral_1 = C_integral_ + 0.5 * (C + C_1) * dt;
  const Eigen::Vector3d acc_integral_1 =
      acc_integral_ + 0.5 * (C * acc_est_0 + C_1 * acc_est_1) * dt;
  // rotation matrix double integral:
  C_doubleintegral_ += 0.5 * (C_integral_ + C_integral_1) *
                       dt; // == C_integral*dt + 0.25*(C + C_1)*dt*dt;
  acc_doubleintegral_ +=
      0.5 * (acc_integral_ + acc_integral_1) *
      dt; //==acc_integral*dt + 0.25*(C + C_1)*acc_S_true*dt*dt;

  dalpha_dT_g_1 = dalpha_dT_g_ +
                  0.5 * dt *
                      (C * invTg_ * dmatrix3_dvector9_multiply(omega_est_0) +
                       C_1 * invTg_ * dmatrix3_dvector9_multiply(omega_est_1));
  dalpha_dT_s_1 =
      dalpha_dT_s_ + 0.5 * dt *
                         (C * invTg_ * dmatrix3_dvector9_multiply(acc_est_0) +
                          C_1 * invTg_ * dmatrix3_dvector9_multiply(acc_est_1));
  dalpha_dT_a_1 = dalpha_dT_a_ +
                  0.5 * dt *
                      (C * invTgsa_ * dmatrix3_dvector9_multiply(acc_est_0) +
                       C_1 * invTgsa_ * dmatrix3_dvector9_multiply(acc_est_1));

  dv_db_g_1 = dv_db_g_ +
              0.5 * dt *
                  (okvis::kinematics::crossMx(C * acc_est_0) * C_integral_ +
                   okvis::kinematics::crossMx(C_1 * acc_est_1) * C_integral_1) *
                  invTg_;

  dv_dT_g_1 = dv_dT_g_ +
              0.5 * dt *
                  (okvis::kinematics::crossMx(C * acc_est_0) * dalpha_dT_g_ +
                   okvis::kinematics::crossMx(C_1 * acc_est_1) * dalpha_dT_g_1);

  dv_dT_s_1 = dv_dT_s_ +
              0.5 * dt *
                  (okvis::kinematics::crossMx(C * acc_est_0) * dalpha_dT_s_ +
                   okvis::kinematics::crossMx(C_1 * acc_est_1) * dalpha_dT_s_1);

  dv_dT_a_1 = dv_dT_a_ +
              0.5 * dt *
                  (C * invTa_ * dmatrix3_dvector9_multiply(acc_est_0) +
                   C_1 * invTa_ * dmatrix3_dvector9_multiply(acc_est_1)) +
              0.5 * dt *
                  (okvis::kinematics::crossMx(C * acc_est_0) * dalpha_dT_a_ +
                   okvis::kinematics::crossMx(C_1 * acc_est_1) * dalpha_dT_a_1);
  dp_db_g_ += 0.5 * dt * (dv_db_g_ + dv_db_g_1);
  dp_dT_g_ += 0.5 * dt * (dv_dT_g_ + dv_dT_g_1);
  dp_dT_s_ += 0.5 * dt * (dv_dT_s_ + dv_dT_s_1);
  dp_dT_a_ += 0.5 * dt * (dv_dT_a_ + dv_dT_a_1);

  // covariance propagation of \f$\delta p^{S0}, \alpha, \delta v^{S0}, b_g, b_a
  // \f$. We discard the Jacobian relative to the extra IMU parameters because they
  // do not contribute to P_delta since it starts from a zero matrix.
  Eigen::Matrix<double, 15, 15> F_delta = Eigen::Matrix<double, 15, 15>::Identity();
  F_delta.block<3, 3>(3, 9) = -0.5 * dt * (C_1 + C) * invTg_;
  F_delta.block<3, 3>(3, 12) = 0.5 * dt * (C_1 + C) * invTgsa_;

  F_delta.block<3, 3>(6, 9) = 0.25 * dt * dt *
                              okvis::kinematics::crossMx(C_1 * acc_est_1) *
                              (C + C_1) * invTg_;
  F_delta.block<3, 3>(6, 12) = -0.5 * dt * (C + C_1) * invTa_ -
                               0.25 * pow(dt, 2) *
                                   okvis::kinematics::crossMx(C_1 * acc_est_1) *
                                   (C + C_1) * invTgsa_;

  F_delta.block<3, 3>(6, 3) = -okvis::kinematics::crossMx(
      0.5 * (C * acc_est_0 + C_1 * acc_est_1) * dt);                // vq
  F_delta.block<3, 3>(0, 3) = 0.5 * dt * F_delta.block<3, 3>(6, 3); // pq

  F_delta.block<3, 3>(0, 6) = Eigen::Matrix3d::Identity() * dt;
  F_delta.block<3, 3>(0, 9) = 0.5 * dt * F_delta.block<3, 3>(6, 9);
  F_delta.block<3, 3>(0, 12) = 0.5 * dt * F_delta.block<3, 3>(6, 12);

  P_delta_ = F_delta * P_delta_ * F_delta.transpose();
  // add noise. note the scaling effect of T_g and T_a
  Eigen::Matrix<double, 15, 15> GQG = Eigen::Matrix<double, 15, 15>::Zero();
  Eigen::Matrix<double, 15, 15> GQG_1 = Eigen::Matrix<double, 15, 15>::Zero();
  Eigen::Matrix3d CinvTg = C * invTg_;
  Eigen::Matrix3d CinvTg_1 = C_1 * invTg_;
  Eigen::Matrix3d CinvTa = C * invTa_;
  Eigen::Matrix3d CinvTa_1 = C_1 * invTa_;
  Eigen::Matrix3d CinvTgsa = C * invTgsa_;
  Eigen::Matrix3d CinvTgsa_1 = C_1 * invTgsa_;
  GQG.block<3, 3>(3, 3) =
      CinvTg * sigma_g_c * sigma_g_c * CinvTg.transpose() +
      CinvTgsa * sigma_a_c * sigma_a_c * CinvTgsa.transpose();
  GQG.block<3, 3>(3, 6) = CinvTgsa * sigma_a_c * sigma_a_c * CinvTa.transpose();
  GQG.block<3, 3>(6, 3) = GQG.block<3, 3>(3, 6).transpose();
  GQG.block<3, 3>(6, 6) = CinvTa * sigma_a_c * sigma_a_c * CinvTa.transpose();
  GQG.block<3, 3>(9, 9) = Eigen::Matrix3d::Identity() * sigma_gw_c * sigma_gw_c;
  GQG.block<3, 3>(12, 12) =
      Eigen::Matrix3d::Identity() * sigma_aw_c * sigma_aw_c;

  GQG_1.block<3, 3>(3, 3) =
      CinvTg_1 * sigma_g_c * sigma_g_c * CinvTg_1.transpose() +
      CinvTgsa_1 * sigma_a_c * sigma_a_c * CinvTgsa_1.transpose();
  GQG_1.block<3, 3>(3, 6) =
      CinvTgsa_1 * sigma_a_c * sigma_a_c * CinvTa_1.transpose();
  GQG_1.block<3, 3>(6, 3) = GQG_1.block<3, 3>(3, 6).transpose();
  GQG_1.block<3, 3>(6, 6) =
      CinvTa_1 * sigma_a_c * sigma_a_c * CinvTa_1.transpose();
  GQG_1.block<3, 3>(9, 9) =
      Eigen::Matrix3d::Identity() * sigma_gw_c * sigma_gw_c;
  GQG_1.block<3, 3>(12, 12) =
      Eigen::Matrix3d::Identity() * sigma_aw_c * sigma_aw_c;

  P_delta_ += 0.5 * dt * (F_delta * GQG * F_delta.transpose() + GQG_1);

  // memory shift
  Delta_q_ = Delta_q_1;
  C_integral_ = C_integral_1;
  acc_integral_ = acc_integral_1;

  dalpha_dT_g_ = dalpha_dT_g_1;
  dalpha_dT_s_ = dalpha_dT_s_1;
  dalpha_dT_a_ = dalpha_dT_a_1;

  dv_db_g_ = dv_db_g_1;
  dv_dT_g_ = dv_dT_g_1;
  dv_dT_s_ = dv_dT_s_1;
  dv_dT_a_ = dv_dT_a_1;
}

void Imu_BG_BA_TG_TS_TA::resetPreintegration() {
  // increments (initialise with identity)
  Delta_q_.setIdentity();
  C_integral_.setZero();
  C_doubleintegral_.setZero();
  acc_integral_.setZero();
  acc_doubleintegral_.setZero();

//    dalpha_db_g_.setZero();
//    dalpha_db_a_.setZero();
  dalpha_dT_g_.setZero();
  dalpha_dT_s_.setZero();
  dalpha_dT_a_.setZero();

  dv_db_g_.setZero();
//    dv_db_a_.setZero();
  dv_dT_g_.setZero();
  dv_dT_s_.setZero();
  dv_dT_a_.setZero();

  dp_db_g_.setZero();
//    dp_db_a_.setZero();
  dp_dT_g_.setZero();
  dp_dT_s_.setZero();
  dp_dT_a_.setZero();

  P_delta_.setZero();
}

void Imu_BG_BA_MG_TS_MA::propagate(double dt,
                                   const Eigen::Vector3d &omega_S_0,
                                   const Eigen::Vector3d &acc_S_0,
                                   const Eigen::Vector3d &omega_S_1,
                                   const Eigen::Vector3d &acc_S_1,
                                   double sigma_g_c, double sigma_a_c,
                                   double sigma_gw_c, double sigma_aw_c) {
  // intermediate variables, must assign values to them before using.
  Eigen::Matrix<double, 3, 9> dalpha_dM_g_1;
  Eigen::Matrix<double, 3, 9> dalpha_dT_s_1;
  Eigen::Matrix<double, 3, 6> dalpha_dM_a_1;
  Eigen::Matrix3d dv_db_g_1;
  Eigen::Matrix<double, 3, 9> dv_dM_g_1;
  Eigen::Matrix<double, 3, 9> dv_dT_s_1;
  Eigen::Matrix<double, 3, 6> dv_dM_a_1;

  Eigen::Vector3d acc_est_0 = Ma_ * (acc_S_0 - ba_);
  Eigen::Vector3d omega_nobias_0 = omega_S_0 - bg_ - Ts_ * acc_est_0;
  Eigen::Vector3d omega_est_0 = Mg_ * omega_nobias_0;

  Eigen::Vector3d acc_est_1 = Ma_ * (acc_S_1 - ba_);
  Eigen::Vector3d omega_nobias_1 = omega_S_1 - bg_ - Ts_ * acc_est_1;
  Eigen::Vector3d omega_est_1 = Mg_ * omega_nobias_1;

  const Eigen::Vector3d omega_S_true = 0.5 * (omega_est_0 + omega_est_1);
  const double theta_half = omega_S_true.norm() * 0.5 * dt;
  const double sinc_theta_half = okvis::kinematics::sinc(theta_half);
  const double cos_theta_half = cos(theta_half);
  Eigen::Quaterniond dq;
  dq.vec() = sinc_theta_half * omega_S_true * 0.5 * dt;
  dq.w() = cos_theta_half;
  Eigen::Quaterniond Delta_q_1 = Delta_q_ * dq;
  // rotation matrix integral:
  const Eigen::Matrix3d C = Delta_q_.toRotationMatrix(); // DCM from Si to S0
  const Eigen::Matrix3d C_1 =
      Delta_q_1.toRotationMatrix(); // DCM from S_{i+1} to S0

  const Eigen::Matrix3d C_integral_1 = C_integral_ + 0.5 * (C + C_1) * dt;
  const Eigen::Vector3d acc_integral_1 =
      acc_integral_ + 0.5 * (C * acc_est_0 + C_1 * acc_est_1) * dt;
  // rotation matrix double integral:
  C_doubleintegral_ += 0.5 * (C_integral_ + C_integral_1) *
                       dt; // == C_integral*dt + 0.25*(C + C_1)*dt*dt;
  acc_doubleintegral_ +=
      0.5 * (acc_integral_ + acc_integral_1) *
      dt; //==acc_integral*dt + 0.25*(C + C_1)*acc_S_true*dt*dt;

  dalpha_dM_g_1 = dalpha_dM_g_ +
                  0.5 * dt *
                      (C * dmatrix3_dvector9_multiply(omega_nobias_0) +
                       C_1 * dmatrix3_dvector9_multiply(omega_nobias_1));
  dalpha_dT_s_1 =
      dalpha_dT_s_ + 0.5 * dt *
                         (C * Mg_ * dmatrix3_dvector9_multiply(acc_S_0 - ba_) +
                          C_1 * Mg_ * dmatrix3_dvector9_multiply(acc_S_1 - ba_));
  dalpha_dM_a_1 = dalpha_dM_a_ +
                  0.5 * dt *
                      (C * Mg_ * Ts_ * dltm3_dvector6_multiply(acc_S_0 - ba_) +
                       C_1 * Mg_ * Ts_ * dltm3_dvector6_multiply(acc_S_1 - ba_));

  dv_db_g_1 = dv_db_g_ +
              0.5 * dt *
                  (okvis::kinematics::crossMx(C * acc_est_0) * C_integral_ +
                   okvis::kinematics::crossMx(C_1 * acc_est_1) * C_integral_1) *
                  Mg_;

  dv_dM_g_1 = dv_dM_g_ +
              0.5 * dt *
                  (okvis::kinematics::crossMx(C * acc_est_0) * dalpha_dM_g_ +
                   okvis::kinematics::crossMx(C_1 * acc_est_1) * dalpha_dM_g_1);

  dv_dT_s_1 = dv_dT_s_ +
              0.5 * dt *
                  (okvis::kinematics::crossMx(C * acc_est_0) * dalpha_dT_s_ +
                   okvis::kinematics::crossMx(C_1 * acc_est_1) * dalpha_dT_s_1);

  dv_dM_a_1 = dv_dM_a_ +
              0.5 * dt *
                  (C * dltm3_dvector6_multiply(acc_S_0 - ba_) +
                   C_1 * dltm3_dvector6_multiply(acc_S_1 - ba_)) +
              0.5 * dt *
                  (okvis::kinematics::crossMx(C * acc_est_0) * dalpha_dM_a_ +
                   okvis::kinematics::crossMx(C_1 * acc_est_1) * dalpha_dM_a_1);
  dp_db_g_ += 0.5 * dt * (dv_db_g_ + dv_db_g_1);
  dp_dM_g_ += 0.5 * dt * (dv_dM_g_ + dv_dM_g_1);
  dp_dT_s_ += 0.5 * dt * (dv_dT_s_ + dv_dT_s_1);
  dp_dM_a_ += 0.5 * dt * (dv_dM_a_ + dv_dM_a_1);

  // covariance propagation of \f$\delta p^{S0}, \alpha, \delta v^{S0}, b_g, b_a
  // \f$. We discard the Jacobian relative to the extra IMU parameters because they
  // do not contribute to P_delta since it starts from a zero matrix.
  F_delta_.setIdentity();
  F_delta_.block<3, 3>(3, 9) = -0.5 * dt * (C_1 + C) * Mg_;
  F_delta_.block<3, 3>(3, 12) = 0.5 * dt * (C_1 + C) * MgTs_;

  F_delta_.block<3, 3>(6, 9) = 0.25 * dt * dt *
                              okvis::kinematics::crossMx(C_1 * acc_est_1) *
                              (C + C_1) * Mg_;
  F_delta_.block<3, 3>(6, 12) = -0.5 * dt * (C + C_1) * Ma_ -
                               0.25 * pow(dt, 2) *
                                   okvis::kinematics::crossMx(C_1 * acc_est_1) *
                                   (C + C_1) * MgTs_;

  F_delta_.block<3, 3>(6, 3) = -okvis::kinematics::crossMx(
      0.5 * (C * acc_est_0 + C_1 * acc_est_1) * dt);                // vq
  F_delta_.block<3, 3>(0, 3) = 0.5 * dt * F_delta_.block<3, 3>(6, 3); // pq

  F_delta_.block<3, 3>(0, 6) = Eigen::Matrix3d::Identity() * dt;
  F_delta_.block<3, 3>(0, 9) = 0.5 * dt * F_delta_.block<3, 3>(6, 9);
  F_delta_.block<3, 3>(0, 12) = 0.5 * dt * F_delta_.block<3, 3>(6, 12);

  P_delta_ = F_delta_ * P_delta_ * F_delta_.transpose();
  // add noise. note the scaling effect of T_g and T_a
  Eigen::Matrix<double, 15, 15> GQG = Eigen::Matrix<double, 15, 15>::Zero();
  Eigen::Matrix<double, 15, 15> GQG_1 = Eigen::Matrix<double, 15, 15>::Zero();
  Eigen::Matrix3d CMg = C * Mg_;
  Eigen::Matrix3d CMg_1 = C_1 * Mg_;
  Eigen::Matrix3d CMa = C * Ma_;
  Eigen::Matrix3d CMa_1 = C_1 * Ma_;
  Eigen::Matrix3d CMgTs = C * MgTs_;
  Eigen::Matrix3d CMgTs_1 = C_1 * MgTs_;
  GQG.block<3, 3>(3, 3) =
      CMg * sigma_g_c * sigma_g_c * CMg.transpose() +
      CMgTs * sigma_a_c * sigma_a_c * CMgTs.transpose();
  GQG.block<3, 3>(3, 6) = CMgTs * sigma_a_c * sigma_a_c * CMa.transpose();
  GQG.block<3, 3>(6, 3) = GQG.block<3, 3>(3, 6).transpose();
  GQG.block<3, 3>(6, 6) = CMa * sigma_a_c * sigma_a_c * CMa.transpose();
  GQG.block<3, 3>(9, 9) = Eigen::Matrix3d::Identity() * sigma_gw_c * sigma_gw_c;
  GQG.block<3, 3>(12, 12) =
      Eigen::Matrix3d::Identity() * sigma_aw_c * sigma_aw_c;

  GQG_1.block<3, 3>(3, 3) =
      CMg_1 * sigma_g_c * sigma_g_c * CMg_1.transpose() +
      CMgTs_1 * sigma_a_c * sigma_a_c * CMgTs_1.transpose();
  GQG_1.block<3, 3>(3, 6) =
      CMgTs_1 * sigma_a_c * sigma_a_c * CMa_1.transpose();
  GQG_1.block<3, 3>(6, 3) = GQG_1.block<3, 3>(3, 6).transpose();
  GQG_1.block<3, 3>(6, 6) =
      CMa_1 * sigma_a_c * sigma_a_c * CMa_1.transpose();
  GQG_1.block<3, 3>(9, 9) =
      Eigen::Matrix3d::Identity() * sigma_gw_c * sigma_gw_c;
  GQG_1.block<3, 3>(12, 12) =
      Eigen::Matrix3d::Identity() * sigma_aw_c * sigma_aw_c;

  P_delta_ += 0.5 * dt * (F_delta_ * GQG * F_delta_.transpose() + GQG_1);

  // memory shift
  Delta_q_ = Delta_q_1;
  C_integral_ = C_integral_1;
  acc_integral_ = acc_integral_1;

  dalpha_dM_g_ = dalpha_dM_g_1;
  dalpha_dT_s_ = dalpha_dT_s_1;
  dalpha_dM_a_ = dalpha_dM_a_1;

  dv_db_g_ = dv_db_g_1;
  dv_dM_g_ = dv_dM_g_1;
  dv_dT_s_ = dv_dT_s_1;
  dv_dM_a_ = dv_dM_a_1;
}

void Imu_BG_BA_MG_TS_MA::resetPreintegration() {
  // increments (initialise with identity)
  Delta_q_.setIdentity();
  C_integral_.setZero();
  C_doubleintegral_.setZero();
  acc_integral_.setZero();
  acc_doubleintegral_.setZero();

//    dalpha_db_g_.setZero();
//    dalpha_db_a_.setZero();
  dalpha_dM_g_.setZero();
  dalpha_dT_s_.setZero();
  dalpha_dM_a_.setZero();

  dv_db_g_.setZero();
//    dv_db_a_.setZero();
  dv_dM_g_.setZero();
  dv_dT_s_.setZero();
  dv_dM_a_.setZero();

  dp_db_g_.setZero();
//    dp_db_a_.setZero();
  dp_dM_g_.setZero();
  dp_dT_s_.setZero();
  dp_dM_a_.setZero();

  F_delta_.setIdentity();
  P_delta_.setZero();
}

void ScaledMisalignedImu::propagate(double /*dt*/,
                                    const Eigen::Vector3d &omega_S_0,
                                    const Eigen::Vector3d &acc_S_0,
                                    const Eigen::Vector3d &omega_S_1,
                                    const Eigen::Vector3d &acc_S_1,
                                    double /*sigma_g_c*/, double /*sigma_a_c*/,
                                    double /*sigma_gw_c*/, double /*sigma_aw_c*/) {
  Eigen::Vector3d omega_est_0;
  Eigen::Vector3d acc_est_0;
  correct(omega_S_0, acc_S_0, &omega_est_0, &acc_est_0);

  Eigen::Vector3d omega_est_1;
  Eigen::Vector3d acc_est_1;
  correct(omega_S_1, acc_S_1, &omega_est_1, &acc_est_1);
  throw std::runtime_error("propagate not implemented for ScaledMisalignImu!");
}

void ScaledMisalignedImu::resetPreintegration() {
  throw std::runtime_error("resetPreintegration not implemented for ScaledMisalignImu!");
}

} // namespace swift_vio
