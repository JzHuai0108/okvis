
/**
 * @file PoseError.cpp
 * @brief Source file for the PoseError class with PoseLocalParameterizationSimplified.
 * @author 
 */

#include <swift_vio/ceres/PoseError.hpp>
#include <swift_vio/ExtrinsicModels.hpp>
#include <okvis/kinematics/MatrixPseudoInverse.hpp>

namespace swift_vio {

// Construct with measurement and information matrix.
PoseError::PoseError(const okvis::kinematics::Transformation & measurement,
                     const Eigen::Matrix<double, 6, 6> & information) {
  setMeasurement(measurement);
  setInformation(information);
}

// Construct with measurement and variance.
PoseError::PoseError(const okvis::kinematics::Transformation & measurement,
                     double translationVariance, double rotationVariance) {
  setMeasurement(measurement);

  information_t information;
  information.setZero();
  information.topLeftCorner<3, 3>() = Eigen::Matrix3d::Identity() * 1.0
      / translationVariance;
  information.bottomRightCorner<3, 3>() = Eigen::Matrix3d::Identity() * 1.0
      / rotationVariance;

  setInformation(information);
}

// Set the information.
void PoseError::setInformation(const information_t & information) {
  information_ = information;
  okvis::MatrixPseudoInverse::pseudoInverseSymm(information, covariance_);
  // perform the Cholesky decomposition on order to obtain the correct error weighting
  information_t L;
  okvis::computeMatrixSqrt(information_, L);
  squareRootInformation_ = L.transpose();
}

// This evaluates the error term and additionally computes the Jacobians.
bool PoseError::Evaluate(double const* const * parameters, double* residuals,
                         double** jacobians) const {
  return EvaluateWithMinimalJacobians(parameters, residuals, jacobians, NULL);
}

// This evaluates the error term and additionally computes
// the Jacobians in the minimal internal representation.
bool PoseError::EvaluateWithMinimalJacobians(double const* const * parameters,
                                             double* residuals,
                                             double** jacobians,
                                             double** jacobiansMinimal) const {

  // compute error
  okvis::kinematics::Transformation T_WS(
      Eigen::Vector3d(parameters[0][0], parameters[0][1], parameters[0][2]),
      Eigen::Quaterniond(parameters[0][6], parameters[0][3], parameters[0][4],
                         parameters[0][5]));
  // delta pose
  okvis::kinematics::Transformation dp = measurement_ * T_WS.inverse();
  // get the error
  Eigen::Matrix<double, 6, 1> error;
  const Eigen::Vector3d dtheta = 2 * dp.q().coeffs().head<3>();
  error.head<3>() = measurement_.r() - T_WS.r();
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
          .topLeftCorner<3, 3>();
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
  }

  return true;
}

}  // namespace swift_vio
