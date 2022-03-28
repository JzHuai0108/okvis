#include <swift_vio/ExtrinsicModels.hpp>
#include <swift_vio/TransformMultiplyJacobian.hpp>

namespace swift_vio {
const std::string Extrinsic_p_CB::kName = "P_CB";
const std::string Extrinsic_p_BC_q_BC::kName = "P_BC_Q_BC";
const std::string Extrinsic_p_C0C_q_C0C::kName = "P_C0C_Q_C0C";

void Extrinsic_p_C0C_q_C0C::dT_BC_dExtrinsic(
    const okvis::kinematics::Transformation &T_BCi,
    const okvis::kinematics::Transformation *T_BC0,
    Eigen::Matrix<double, 6, kNumParams> *j_C0Ci) {
  swift_vio::TransformMultiplyJacobian tmj(*T_BC0, T_BC0->inverse() * T_BCi);
  j_C0Ci->topLeftCorner<3, 3>() = tmj.dp_dp_BC();
  j_C0Ci->topRightCorner<3, 3>() = tmj.dp_dtheta_BC();
  j_C0Ci->bottomLeftCorner<3, 3>() = tmj.dtheta_dp_BC();
  j_C0Ci->bottomRightCorner<3, 3>() = tmj.dtheta_dtheta_BC();
}

void Extrinsic_p_C0C_q_C0C::dT_BC_dT_BC0(
    const okvis::kinematics::Transformation &T_BCi,
    const okvis::kinematics::Transformation *T_BC0,
    Eigen::Matrix<double, 6, 6> *j_BC0) {
  swift_vio::TransformMultiplyJacobian tmj(*T_BC0, T_BC0->inverse() * T_BCi);
  j_BC0->topLeftCorner<3, 3>() = tmj.dp_dp_AB();
  j_BC0->topRightCorner<3, 3>() = tmj.dp_dtheta_AB();
  j_BC0->bottomLeftCorner<3, 3>() = tmj.dtheta_dp_AB();
  j_BC0->bottomRightCorner<3, 3>() = tmj.dtheta_dtheta_AB();
}
} // namespace swift_vio
