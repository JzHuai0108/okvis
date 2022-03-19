
/**
 * @file RelativePoseError.cpp
 * @brief Source file for the RelativePoseError class.
 * @author
 */

#include <swift_vio/ceres/RelativePoseError.hpp>
#include <okvis/ceres/PoseLocalParameterization.hpp>

namespace swift_vio {

// Construct with measurement and information matrix.
RelativePoseError::RelativePoseError(
    const Eigen::Matrix<double, 6, 6> & information) {
  setInformation(information);
}

// Construct with measurement and variance.
RelativePoseError::RelativePoseError(double translationVariance,
                                     double rotationVariance) {

  information_t information;
  information.setZero();
  information.topLeftCorner<3, 3>() = Eigen::Matrix3d::Identity() * 1.0
      / translationVariance;
  information.bottomRightCorner<3, 3>() = Eigen::Matrix3d::Identity() * 1.0
      / rotationVariance;

  setInformation(information);
}

// Set the information.
void RelativePoseError::setInformation(const information_t & information) {
  information_ = information;
  covariance_ = information.inverse();
  // perform the Cholesky decomposition on order to obtain the correct error weighting
  Eigen::LLT<information_t> lltOfInformation(information_);
  squareRootInformation_ = lltOfInformation.matrixL().transpose();
}

// This evaluates the error term and additionally computes the Jacobians.
bool RelativePoseError::Evaluate(double const* const * parameters,
                                 double* residuals, double** jacobians) const {
  return EvaluateWithMinimalJacobians(parameters, residuals, jacobians, NULL);
}

// This evaluates the error term and additionally computes
// the Jacobians in the minimal internal representation.
bool RelativePoseError::EvaluateWithMinimalJacobians(
    double const* const * parameters, double* residuals, double** jacobians,
    double** jacobiansMinimal) const {

  // compute error
  okvis::kinematics::Transformation T_WS_0(
      Eigen::Vector3d(parameters[0][0], parameters[0][1], parameters[0][2]),
      Eigen::Quaterniond(parameters[0][6], parameters[0][3], parameters[0][4],
                         parameters[0][5]));
  okvis::kinematics::Transformation T_WS_1(
      Eigen::Vector3d(parameters[1][0], parameters[1][1], parameters[1][2]),
      Eigen::Quaterniond(parameters[1][6], parameters[1][3], parameters[1][4],
                         parameters[1][5]));
  // delta pose
  okvis::kinematics::Transformation dp = T_WS_1 * T_WS_0.inverse();
  // get the error
  Eigen::Matrix<double, 6, 1> error;
  const Eigen::Vector3d dtheta = 2 * dp.q().coeffs().head<3>();
  error.head<3>() = T_WS_1.r() - T_WS_0.r();
  error.tail<3>() = dtheta;

  // weigh it
  Eigen::Map<Eigen::Matrix<double, 6, 1> > weighted_error(residuals);
  weighted_error = squareRootInformation_ * error;

  // compute Jacobian...
  if (jacobians != NULL) {
    if (jacobians[0] != NULL) {
      Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor> > J0(
          jacobians[0]);
      Eigen::Matrix<double, 6, 6, Eigen::RowMajor> J0_minimal;
      J0_minimal.setIdentity();
      J0_minimal *= -1.0;
      J0_minimal.block<3, 3>(3, 3) = -okvis::kinematics::plus(dp.q())
          .block<3, 3>(0, 0);
      J0_minimal = (squareRootInformation_ * J0_minimal).eval();

      J0.leftCols<6>() = J0_minimal;
      J0.col(6).setZero();

      if (jacobiansMinimal != NULL) {
        if (jacobiansMinimal[0] != NULL) {
          Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor> > J0_minimal_mapped(
              jacobiansMinimal[0]);
          J0_minimal_mapped = J0_minimal;
        }
      }
    }
    if (jacobians[1] != NULL) {
      Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor> > J1(
          jacobians[1]);
      Eigen::Matrix<double, 6, 6, Eigen::RowMajor> J1_minimal;
      J1_minimal.setIdentity();
      J1_minimal.block<3, 3>(3, 3) = okvis::kinematics::oplus(dp.q())
          .block<3, 3>(0, 0);
      J1_minimal = (squareRootInformation_ * J1_minimal).eval();

      J1.leftCols<6>() = J1_minimal;
      J1.col(6).setZero();

      if (jacobiansMinimal != NULL) {
        if (jacobiansMinimal[1] != NULL) {
          Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor> > J1_minimal_mapped(
              jacobiansMinimal[1]);
          J1_minimal_mapped = J1_minimal;
        }
      }
    }
  }

  return true;
}

}  // namespace swift_vio

