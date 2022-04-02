
#include <Eigen/Core>
#include <swift_vio/imu/SimpleImuPropagationJacobian.hpp>
#include <swift_vio/ParallaxAnglePoint.hpp>

namespace swift_vio {
void SimpleImuPropagationJacobian::initialize(
    const okvis::Time startEpoch, const okvis::Time endEpoch,
    const okvis::kinematics::Transformation &endT_WB,
    const Eigen::Matrix<double, 3, 1> &endV_W,
    const Eigen::Matrix<double, 3, 1> &endOmega_B) {
  startEpoch_ = startEpoch;
  endEpoch_ = endEpoch;
  endp_WB_B_ = endT_WB.r();
  endq_WB_ = endT_WB.q();
  endV_WB_W_ = endV_W;
  endOmega_WB_B_ = endOmega_B;
}

SimpleImuPropagationJacobian::SimpleImuPropagationJacobian(
    const okvis::Time startEpoch, const okvis::Time endEpoch,
    const okvis::kinematics::Transformation &endT_WB,
    const Eigen::Matrix<double, 3, 1> &endV_W,
    const Eigen::Matrix<double, 3, 1> &endOmega_B)
    : startEpoch_(startEpoch), endEpoch_(endEpoch), endp_WB_B_(endT_WB.r()),
      endq_WB_(endT_WB.q()), endV_WB_W_(endV_W), endOmega_WB_B_(endOmega_B) {}

Eigen::Matrix3d SimpleImuPropagationJacobian::dp_dv_WB(double deltaTime) {
  Eigen::Matrix3d j = Eigen::Matrix3d::Identity();
  j(0, 0) = deltaTime;
  j(1, 1) = deltaTime;
  j(2, 2) = deltaTime;
  return j;
}

Eigen::Matrix<double, 3, 2> SimpleImuPropagationJacobian::dp_dunitgW(
    double deltaTime, const Eigen::Vector3d &gravityDirection,
    double gravityMagnitude) {
  return 0.5 * deltaTime * deltaTime * gravityMagnitude *
         NormalVectorElement(gravityDirection).getM();
}

Eigen::Matrix<double, 3, 2> SimpleImuPropagationJacobian::dtheta_dunitgW(
    double /*deltaTime*/, const Eigen::Vector3d & /*gravityDirection*/,
    double /*gravityMagnitude*/) {
  return Eigen::Matrix<double, 3, 2>::Zero();
}

void SimpleImuPropagationJacobian::Phi_pq(const Eigen::Vector3d &start_p_WB,
                                          const Eigen::Vector3d &start_v_WB,
                                          const Eigen::Vector3d &gW,
                                          Eigen::Matrix3d *phi) const {
  double delta = (endEpoch_ - startEpoch_).toSec();
  swift_vio::Phi_pq(start_p_WB, endp_WB_B_, start_v_WB, gW, delta, phi);
}

} // namespace swift_vio
