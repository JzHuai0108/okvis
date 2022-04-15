
/**
 * @file EuclideanParamErrorSized.hpp
 * @brief Header file for the EuclideanParamErrorSized class.
 * @author Jianzhu Huai
 */

#ifndef INCLUDE_OKVIS_CERES_EUCLIDEANPARAMERRORSIZED_HPP_
#define INCLUDE_OKVIS_CERES_EUCLIDEANPARAMERRORSIZED_HPP_

#include <vector>
#include <Eigen/Core>
#include "ceres/ceres.h"
#include <okvis/assert_macros.hpp>
#include <okvis/FrameTypedefs.hpp>
#include <okvis/ceres/ErrorInterface.hpp>

namespace okvis {
/// \brief ceres Namespace for ceres-related functionality implemented in okvis.
namespace ceres {
template<int kParamDim>
class EuclideanParamErrorSized : public ::ceres::SizedCostFunction<
    kParamDim /* number of residuals */,
    kParamDim /* size of first parameter */>,
    public ErrorInterface {
 public:

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  OKVIS_DEFINE_EXCEPTION(Exception,std::runtime_error)

  /// \brief The base class type.
  typedef ::ceres::SizedCostFunction<kParamDim, kParamDim> base_t;

  /// \brief Number of residuals (kParamDim)
  static const int kNumResiduals = kParamDim;

  /// \brief The information matrix type.
  typedef Eigen::Matrix<double, kParamDim, kParamDim> information_t;

  /// \brief The covariance matrix type (same as information).
  typedef Eigen::Matrix<double, kParamDim, kParamDim> covariance_t;

  /// \brief Default constructor.
  EuclideanParamErrorSized();

  /// \brief Construct with measurement and information matrix
  /// @param[in] measurement The measurement.
  /// @param[in] information The information (weight) matrix.
  EuclideanParamErrorSized(const Eigen::Matrix<double, kParamDim, 1> & measurement,
                      const information_t & information);

  /// \brief Construct with measurement and variance.
  /// @param[in] measurement The measurement.
  /// @param[in] variance The variance of each dim of the measurement, i.e. information_ has 1/variance in its diagonal.
  EuclideanParamErrorSized(const Eigen::Matrix<double, kParamDim, 1>& measurement,
                      const Eigen::Matrix<double, kParamDim, 1>& variance);

  EuclideanParamErrorSized(const Eigen::Matrix<double, kParamDim, 1>& measurement,
                           double varforall);

  /// \brief Trivial destructor.
  virtual ~EuclideanParamErrorSized() {
  }

  // setters
  /// \brief Set the measurement.
  /// @param[in] measurement The measurement.
  void setMeasurement(const Eigen::Matrix<double, kParamDim, 1> & measurement) {
    measurement_ = measurement;
  }

  /// \brief Set the information.
  /// @param[in] information The information (weight) matrix.
  void setInformation(const information_t & information);

  // getters
  /// \brief Get the measurement.
  /// \return The measurement vector.
  const Eigen::Matrix<double, kParamDim, 1>& measurement() const {
    return measurement_;
  }

  /// \brief Get the information matrix.
  /// \return The information (weight) matrix.
  const information_t& information() const {
    return information_;
  }

  /// \brief Get the covariance matrix.
  /// \return The inverse information (covariance) matrix.
  const covariance_t& covariance() const {
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
    return kNumResiduals;
  }

  /// \brief Number of parameter blocks.
  size_t parameterBlocks() const {
    return base_t::parameter_block_sizes().size();
  }

  /// \brief Dimension of an individual parameter block.
  /// @param[in] parameterBlockId ID of the parameter block of interest.
  /// \return The dimension.
  size_t parameterBlockDim(size_t parameterBlockId) const {
    return base_t::parameter_block_sizes().at(parameterBlockId);
  }

  /// @brief Residual block type as string
  virtual std::string typeInfo() const {
    return "EuclideanParamErrorSized";
  }

 protected:

  // the measurement
  Eigen::Matrix<double, kParamDim, 1> measurement_; ///< The measurement.

  // weighting related
  information_t information_; ///< The information matrix.
  information_t squareRootInformation_; ///< The square root information matrix.
  covariance_t covariance_; ///< The covariance matrix.

};

typedef EuclideanParamErrorSized<3> SpeedParameterError;
typedef EuclideanParamErrorSized<6> BiasParameterError;

}  // namespace ceres
}  // namespace okvis

#include "implementation/EuclideanParamErrorSized.hpp"
#endif /* INCLUDE_OKVIS_CERES_EUCLIDEANPARAMERRORSIZED_HPP_ */
