
/**
 * @file EuclideanParamError.hpp
 * @brief Header file for the EuclideanParamError class.
 * @author Jianzhu Huai
 */

#ifndef INCLUDE_OKVIS_CERES_EUCLIDEANPARAMERROR_HPP_
#define INCLUDE_OKVIS_CERES_EUCLIDEANPARAMERROR_HPP_

#include <vector>
#include <Eigen/Core>
#include "ceres/ceres.h"
#include <okvis/assert_macros.hpp>
#include <okvis/FrameTypedefs.hpp>
#include <okvis/ceres/ErrorInterface.hpp>

namespace okvis {
/// \brief ceres Namespace for ceres-related functionality implemented in okvis.
namespace ceres {
class EuclideanParamError : public ::ceres::DynamicCostFunction,
    public ErrorInterface {
 public:

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  OKVIS_DEFINE_EXCEPTION(Exception,std::runtime_error)

  /// \brief The base class type.
  typedef ::ceres::DynamicCostFunction base_t;

  /// \brief Default constructor.
  EuclideanParamError();

  /// \brief Construct with measurement and variance.
  /// @param[in] measurement The measurement.
  /// @param[in] variance The variance of each dim of the measurement, i.e. information_ has 1/variance in its diagonal.
  EuclideanParamError(const Eigen::Matrix<double, -1, 1>& measurement,
                      const Eigen::Matrix<double, -1, 1>& variance);

  EuclideanParamError(const Eigen::Matrix<double, -1, 1>& measurement,
                      double varforall);

  void setParameterBlockAndResidualSizes() {
    AddParameterBlock(measurement_.size());
    SetNumResiduals(measurement_.size());
  }

  /// \brief Trivial destructor.
  virtual ~EuclideanParamError() {
  }

  // setters
  /// \brief Set the measurement.
  /// @param[in] measurement The measurement.
  void setMeasurement(const Eigen::Matrix<double, -1, 1> & measurement) {
    measurement_ = measurement;
  }

  /// \brief Set the information.
  /// @param[in] information The information (weight) matrix.
  void setInformation(const Eigen::MatrixXd & information);

  // getters
  /// \brief Get the measurement.
  /// \return The measurement vector.
  const Eigen::Matrix<double, -1, 1>& measurement() const {
    return measurement_;
  }

  /// \brief Get the information matrix.
  /// \return The information (weight) matrix.
  const Eigen::MatrixXd &information() const {
    return information_;
  }

  /// \brief Get the covariance matrix.
  /// \return The inverse information (covariance) matrix.
  const Eigen::MatrixXd &covariance() const {
    return covariance_;
  }

  // error term and Jacobian implementation
  /**
   * @brief This evaluates the error term and additionally computes the Jacobians.
   * @param parameters Pointer to the parameters (see ceres)
   * @param residuals Pointer to the residual vector (see ceres)
   * @param jacobians Pointer to the Jacobians (see ceres)
   * @return success of th evaluation.
   */
  virtual bool Evaluate(double const* const * parameters, double* residuals,
                        double** jacobians) const;

  /**
   * @brief This evaluates the error term and additionally computes
   *        the Jacobians in the minimal internal representation.
   * @param parameters Pointer to the parameters (see ceres)
   * @param residuals Pointer to the residual vector (see ceres)
   * @param jacobians Pointer to the Jacobians (see ceres)
   * @param jacobiansMinimal Pointer to the minimal Jacobians (equivalent to jacobians).
   * @return Success of the evaluation.
   */
  virtual bool EvaluateWithMinimalJacobians(double const* const * parameters,
                                            double* residuals,
                                            double** jacobians,
                                            double** jacobiansMinimal) const;

  // sizes
  /// \brief Residual dimension.
  size_t residualDim() const {
    return measurement_.size();
  }

  /// \brief Number of parameter blocks.
  size_t parameterBlocks() const final {
    return parameter_block_sizes().size();
  }

  /// \brief Dimension of an individual parameter block.
  /// @param[in] parameterBlockId ID of the parameter block of interest.
  /// \return The dimension.
  size_t parameterBlockDim(size_t parameterBlockId) const final {
    return base_t::parameter_block_sizes().at(parameterBlockId);
  }

  /// @brief Residual block type as string
  virtual std::string typeInfo() const {
    return "EuclideanParamError";
  }

 protected:

  // the measurement
  Eigen::Matrix<double, -1, 1> measurement_; ///< The measurement.

  // weighting related
  Eigen::MatrixXd information_; ///< The information matrix.
  Eigen::MatrixXd squareRootInformation_; ///< The square root information matrix.
  Eigen::MatrixXd covariance_; ///< The covariance matrix.

};
}  // namespace ceres
}  // namespace okvis

#endif /* INCLUDE_OKVIS_CERES_EUCLIDEANPARAMERROR_HPP_ */
