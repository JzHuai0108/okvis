
/**
 * @file RelativePoseError.hpp
 * @brief Header file for the RelativePoseError class with PoseLocalParameterizationSimplified.
 * @author
 */

#ifndef INCLUDE_SWIFT_VIO_RELATIVEPOSEERROR_HPP_
#define INCLUDE_SWIFT_VIO_RELATIVEPOSEERROR_HPP_

#include <vector>
#include "ceres/ceres.h"
#include <okvis/kinematics/Transformation.hpp>
#include <okvis/assert_macros.hpp>
#include <okvis/ceres/ErrorInterface.hpp>

namespace swift_vio {
/// \brief Relative error between two poses.
class RelativePoseError : public ::ceres::SizedCostFunction<
    6 /* number of residuals */,
    7, /* size of first parameter */
    7 /* size of second parameter */>, public okvis::ceres::ErrorInterface {
 public:

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  OKVIS_DEFINE_EXCEPTION(Exception,std::runtime_error)

  /// \brief The base class type.
  typedef ::ceres::SizedCostFunction<6, 7, 7> base_t;

  /// \brief Number of residuals (6).
  static const int kNumResiduals = 6;

  /// \brief The information matrix type (6x6).
  typedef Eigen::Matrix<double, 6, 6> information_t;

  /// \brief The covariance matrix type (same as information).
  typedef Eigen::Matrix<double, 6, 6> covariance_t;

  /// \brief Default constructor.
  RelativePoseError();

  /// \brief Construct with measurement and information matrix
  /// @param[in] information The information (weight) matrix.
  RelativePoseError(const Eigen::Matrix<double, 6, 6> & information);

  /// \brief Construct with measurement and variance.
  /// @param[in] translationVariance The (relative) translation variance.
  /// @param[in] rotationVariance The (relative) rotation variance.
  RelativePoseError(double translationVariance, double rotationVariance);

  /// \brief Trivial destructor.
  virtual ~RelativePoseError() {
  }

  // setters
  /// \brief Set the information.
  /// @param[in] information The information (weight) matrix.
  void setInformation(const information_t & information);

  // getters
  /// \brief Get the information matrix.
  /// \return The information (weight) matrix.
  const information_t& information() const {
    return information_;
  }

  /// \brief Get the covariance matrix.
  /// \return The inverse information (covariance) matrix.
  const information_t& covariance() const {
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
  bool EvaluateWithMinimalJacobians(double const* const * parameters,
                                    double* residuals, double** jacobians,
                                    double** jacobiansMinimal) const;

  // sizes
  /// \brief Residual dimension.
  size_t residualDim() const {
    return kNumResiduals;
  }

  /// \brief Number of parameter blocks.
  size_t parameterBlocks() const {
    return parameter_block_sizes().size();
  }

  /// \brief Dimension of an individual parameter block.
  size_t parameterBlockDim(size_t parameterBlockId) const {
    return base_t::parameter_block_sizes().at(parameterBlockId);
  }

  /// @brief Residual block type as string
  virtual std::string typeInfo() const {
    return "RelativePoseError";
  }

 protected:

  // weighting related
  information_t information_; ///< The 6x6 information matrix.
  information_t squareRootInformation_; ///< The 6x6 square root information matrix.
  covariance_t covariance_; ///< The 6x6 covariance matrix.

};

}  // namespace swift_vio


#endif /* INCLUDE_SWIFT_VIO_RELATIVEPOSEERROR_HPP_ */
