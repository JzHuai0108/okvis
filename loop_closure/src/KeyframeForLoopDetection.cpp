#include <loop_closure/KeyframeForLoopDetection.hpp>
#include <okvis/kinematics/MatrixPseudoInverse.hpp>

namespace swift_vio {
NeighborConstraintInDatabase::NeighborConstraintInDatabase() {}

NeighborConstraintInDatabase::NeighborConstraintInDatabase(
    uint64_t id, okvis::Time stamp,
    const okvis::kinematics::Transformation& T_BBr,
    PoseConstraintType type) :
  id_(id), stamp_(stamp), T_BBr_(T_BBr), type_(type) {
  squareRootInfo_.setIdentity();
}

NeighborConstraintInDatabase::~NeighborConstraintInDatabase() {}

NeighborConstraintMessage::NeighborConstraintMessage() {}

NeighborConstraintMessage::NeighborConstraintMessage(
    uint64_t id, okvis::Time stamp,
    const okvis::kinematics::Transformation& T_BnBr,
    const okvis::kinematics::Transformation& T_WB,
    PoseConstraintType type) :
  core_(id, stamp, T_BnBr, type), T_WB_(T_WB) {
  cov_T_WB_.setIdentity();
  cov_T_WBr_T_WB_.setZero();
}

NeighborConstraintMessage::~NeighborConstraintMessage() {}

void NeighborConstraintMessage::computeRelativePoseCovariance(
    const okvis::kinematics::Transformation& T_WBr,
    const Eigen::Matrix<double, 6, 6>& cov_T_WBr,
    Eigen::Matrix<double, 6, 6>* cov_T_BnBr) {
  // use T_WBr, T_WBn, cov_T_WBr_WB_, cov_T_WBn, cov_T_WBr to compute cov_T_BnBr
  // var(Y - X) = var(Y) + var(X) - cov(Y, X) - cov(X, Y)
  InverseTransformMultiplyJacobian itmj(T_WB_, T_WBr);
  Eigen::Matrix<double, 6, 6> Jzr, Jzn;
  itmj.dT_dT_WA(&Jzn);
  itmj.dT_dT_WB(&Jzr);
  Eigen::Matrix<double, 6, 6> crossTerm =
      Jzr * cov_T_WBr_T_WB_ * Jzn.transpose();
  *cov_T_BnBr = Jzr * cov_T_WBr * Jzr.transpose() - crossTerm.transpose() - crossTerm + Jzn * cov_T_WB_ * Jzn.transpose();
}

KeyframeInDatabase::KeyframeInDatabase() {}

KeyframeInDatabase::KeyframeInDatabase(
    uint64_t vioId, okvis::Time stamp,
    const okvis::kinematics::Transformation& vio_T_WB,
    const Eigen::Matrix<double, 6, 6>& cov_T_WB)
    : id_(vioId), stamp_(stamp),
      vio_T_WB_(vio_T_WB), cov_vio_T_WB_(cov_T_WB) {

}

void KeyframeInDatabase::setSquareRootInfoFromCovariance(
    size_t j, const Eigen::Matrix<double, 6, 6>& covRawError) {
  Eigen::Matrix<double, 6, 6> pseudoL;
  okvis::MatrixPseudoInverse::pseudoInverseSymmSqrt(
      covRawError, pseudoL,
      std::numeric_limits<double>::epsilon());
  constraintList_.at(j)->squareRootInfo_.noalias() = pseudoL.transpose();
}

}  // namespace swift_vio
