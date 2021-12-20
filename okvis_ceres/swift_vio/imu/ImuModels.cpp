#include <swift_vio/imu/ImuModels.hpp>

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

void Imu_BG_BA::propagate(double dt, const Eigen::Vector3d &omega_est_0,
                          const Eigen::Vector3d &acc_est_0,
                          const Eigen::Vector3d &omega_est_1,
                          const Eigen::Vector3d &acc_est_1, double sigma_g_c,
                          double sigma_a_c, double sigma_gw_c,
                          double sigma_aw_c) {
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
                                   const Eigen::Vector3d &omega_est_0,
                                   const Eigen::Vector3d &acc_est_0,
                                   const Eigen::Vector3d &omega_est_1,
                                   const Eigen::Vector3d &acc_est_1,
                                   double sigma_g_c, double sigma_a_c,
                                   double sigma_gw_c, double sigma_aw_c) {}

void Imu_BG_BA_TG_TS_TA::resetPreintegration() {}

void ScaledMisalignedImu::propagate(double dt,
                                    const Eigen::Vector3d &omega_est_0,
                                    const Eigen::Vector3d &acc_est_0,
                                    const Eigen::Vector3d &omega_est_1,
                                    const Eigen::Vector3d &acc_est_1,
                                    double sigma_g_c, double sigma_a_c,
                                    double sigma_gw_c, double sigma_aw_c) {}

void ScaledMisalignedImu::resetPreintegration() {}

} // namespace swift_vio
