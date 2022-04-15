
/**
 * @file EuclideanParamError.hpp
 * @brief Source file for the EuclideanParamError class.
 * @author Jianzhu Huai
 */

#include <okvis/kinematics/MatrixPseudoInverse.hpp>

namespace okvis {
/// \brief ceres Namespace for ceres-related functionality implemented in okvis.
namespace ceres {
// Construct with measurement and variance.
EuclideanParamError::EuclideanParamError(const Eigen::Matrix<double, -1, 1>& measurement,
                                     const Eigen::Matrix<double, -1, 1>& variance) {
  setMeasurement(measurement);
  Eigen::MatrixXd information(measurement.size(), measurement.size());
  information.setIdentity();
  information.diagonal().cwiseQuotient(variance);
  setInformation(information);
}

EuclideanParamError::EuclideanParamError(const Eigen::Matrix<double, -1, 1>& measurement,
                                         double varforall) {
  setMeasurement(measurement);
  Eigen::MatrixXd information(measurement.size(), measurement.size());
  information.setIdentity();
  information.diagonal().setConstant(1.0 / varforall);
  setInformation(information);
}

// Set the information.
void EuclideanParamError::setInformation(const Eigen::MatrixXd &information) {
  information_ = information;
  MatrixPseudoInverse::pseudoInverseSymm(information, covariance_);
  // perform the Cholesky decomposition on order to obtain the correct error weighting
  Eigen::MatrixXd L;
  computeMatrixSqrt(information_, L);
  squareRootInformation_ = L.transpose();
}

// This evaluates the error term and additionally computes the Jacobians.
bool EuclideanParamError::Evaluate(double const* const * parameters,
                                 double* residuals, double** jacobians) const {
  return EvaluateWithMinimalJacobians(parameters, residuals, jacobians, NULL);
}

// This evaluates the error term and additionally computes
// the Jacobians in the minimal internal representation.
bool EuclideanParamError::EvaluateWithMinimalJacobians(
    double const* const * parameters, double* residuals, double** jacobians,
    double** jacobiansMinimal) const {

  // compute error
  size_t resDim = residualDim();
  Eigen::Map<const Eigen::Matrix<double, -1, 1>> estimate(parameters[0], resDim);
  Eigen::Matrix<double, -1, 1> error = measurement_ - estimate;

  // weigh it
  Eigen::Map<Eigen::Matrix<double, -1, 1> > weighted_error(residuals, resDim);
  weighted_error = squareRootInformation_ * error;
  // compute Jacobian - this is rather trivial in this case...
  if (jacobians != NULL) {
    if (jacobians[0] != NULL) {
      Eigen::Map<Eigen::Matrix<double, -1, -1, Eigen::RowMajor> > J0(
          jacobians[0], resDim, resDim);
      J0 = -squareRootInformation_ * Eigen::Matrix<double, -1, -1>::Identity(resDim, resDim);
    }
  }
  if (jacobiansMinimal != NULL) {
    if (jacobiansMinimal[0] != NULL) {
      Eigen::Map<Eigen::Matrix<double, -1, -1, Eigen::RowMajor> > J0min(
          jacobiansMinimal[0], resDim, resDim);
      J0min = -squareRootInformation_ * Eigen::Matrix<double, -1, -1>::Identity(resDim, resDim);
    }
  }
  return true;
}
}  // namespace ceres
}  // namespace okvis
