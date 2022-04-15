
/**
 * @file ScalarError.hpp
 * @brief Source file for the ScalarError class.
 * @author Jianzhu Huai
 */


namespace okvis {
/// \brief ceres Namespace for ceres-related functionality implemented in okvis.
namespace ceres {
// Construct with measurement and variance.
ScalarError::ScalarError(const double& measurement,
                                     const double& variance) {
  setMeasurement(measurement);
  setInformation(1/variance);
}

// Set the information.
void ScalarError::setInformation(const information_t & information) {
  information_ = information;
  covariance_ = 1/information;
  squareRootInformation_ = std::sqrt(information_);
}

// This evaluates the error term and additionally computes the Jacobians.
bool ScalarError::Evaluate(double const* const * parameters,
                                 double* residuals, double** jacobians) const {
  return EvaluateWithMinimalJacobians(parameters, residuals, jacobians, NULL);
}

// This evaluates the error term and additionally computes
// the Jacobians in the minimal internal representation.
bool ScalarError::EvaluateWithMinimalJacobians(
    double const* const * parameters, double* residuals, double** jacobians,
    double** jacobiansMinimal) const {
  double error = measurement_ - parameters[0][0];
  residuals[0] = squareRootInformation_ * error;

  // compute Jacobian - this is rather trivial in this case...
  if (jacobians != NULL) {
    if (jacobians[0] != NULL) {
      jacobians[0][0] = -squareRootInformation_;
    }
  }
  if (jacobiansMinimal != NULL) {
    if (jacobiansMinimal[0] != NULL) {
      jacobiansMinimal[0][0] = -squareRootInformation_;
    }
  }
  return true;
}

}  // namespace ceres
}  // namespace okvis
