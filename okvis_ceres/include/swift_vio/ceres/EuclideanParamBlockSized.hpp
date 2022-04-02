
/**
 * @file EuclideanParamBlockSized.hpp
 * @brief Header file for the EuclideanParamBlockSized class.
 * @author Jianzhu Huai
 */

#ifndef INCLUDE_OKVIS_CERES_EUCLIDEANPARAMBLOCKSIZED_HPP_
#define INCLUDE_OKVIS_CERES_EUCLIDEANPARAMBLOCKSIZED_HPP_

#include <Eigen/Core>
#include <okvis/Time.hpp>
#include <okvis/ceres/ParameterBlockSized.hpp>

namespace okvis {
/// \brief ceres Namespace for ceres-related functionality implemented in okvis.
namespace ceres {

/// \brief Wraps the parameter block for camera intrinsics estimate
template <int Dim>
class EuclideanParamBlockSized
    : public okvis::ceres::ParameterBlockSized<Dim, Dim, Eigen::Matrix<double, Dim, 1>> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  /// \brief The base class type.
  typedef okvis::ceres::ParameterBlockSized<Dim, Dim, Eigen::Matrix<double, Dim, 1>> base_t;

  /// \brief The estimate type (9D vector).
  typedef Eigen::Matrix<double, Dim, 1> estimate_t;

  /// \brief Default constructor (assumes not fixed).
  EuclideanParamBlockSized() : base_t::ParameterBlockSized() {
    okvis::ceres::ParameterBlock::setFixed(false);
  }

  /// \brief Constructor with estimate and time.
  /// @param[in] intrinsicParams The fx,fy,cx,cy estimate.
  /// @param[in] id The (unique) ID of this block.
  /// @param[in] timestamp The timestamp of this state.
  EuclideanParamBlockSized(const Eigen::Matrix<double, Dim, 1>& intrinsicParams,
                           uint64_t id) {
    setEstimate(intrinsicParams);
    okvis::ceres::ParameterBlock::setId(id);
    okvis::ceres::ParameterBlock::setFixed(false);
  }

  /// \brief Trivial destructor.
  virtual ~EuclideanParamBlockSized() {}

  // setters
  /// @brief Set estimate of this parameter block.
  /// @param[in] intrinsicParams The estimate to set this to.
  virtual void setEstimate(
      const Eigen::Matrix<double, Dim, 1>& intrinsicParams) {
    for (int i = 0; i < base_t::Dimension; ++i)
      base_t::parameters_[i] = intrinsicParams[i];
  }

  // getters
  /// @brief Get estimate.
  /// \return The estimate.
  virtual Eigen::Matrix<double, Dim, 1> estimate() const {
    Eigen::Matrix<double, Dim, 1> intrinsicParams;
    for (int i = 0; i < base_t::Dimension; ++i)
      intrinsicParams[i] = base_t::parameters_[i];
    return intrinsicParams;
  }

  // Delta_Chi=x0_plus_Delta[-]x0
  /// \brief Computes the minimal difference between a variable x and a perturbed variable x_plus_delta
  /// @param[in] x0 Variable.
  /// @param[in] x0_plus_Delta Perturbed variable.
  /// @param[out] Delta_Chi Minimal difference.
  /// \return True on success.
  void minus(const double* x0, const double* x0_plus_Delta,
                     double* Delta_Chi) const final {
    Eigen::Map<const Eigen::Matrix<double, Dim, 1> > x0_(x0);
    Eigen::Map<Eigen::Matrix<double, Dim, 1> > Delta_Chi_(Delta_Chi);
    Eigen::Map<const Eigen::Matrix<double, Dim, 1> > x0_plus_Delta_(
        x0_plus_Delta);
    Delta_Chi_ = x0_plus_Delta_ - x0_;
  }

  /// \brief Computes the Jacobian from minimal space to naively overparameterised space as used by ceres.
//  /// @param[in] x0 Variable.
  /// @param[out] jacobian the Jacobian (dimension minDim x dim).
  /// \return True on success.
  void liftJacobian(const double * /*unused: x*/,
                    double *jacobian) const final {
    Eigen::Map<Eigen::Matrix<double, Dim, Dim, Eigen::RowMajor>> identity(
        jacobian);
    identity.setIdentity();
  }

  /// @brief Return parameter block type as string
  virtual std::string typeInfo() const { return "EuclideanParamBlockSized"; }
};
typedef EuclideanParamBlockSized<9> ShapeMatrixParamBlock;

typedef EuclideanParamBlockSized<6> BiasParameterBlock; // bg ba.
}  // namespace ceres
}  // namespace okvis

#endif /* INCLUDE_OKVIS_CERES_EUCLIDEANPARAMBLOCKSIZED_HPP_ */
