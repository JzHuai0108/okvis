
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
  EuclideanParamBlockSized() : base_t::ParameterBlockSized(), linPointFixed_(false) {
    okvis::ceres::ParameterBlock::setFixed(false);
  }

  /// \brief Constructor with estimate and time.
  /// @param[in] intrinsicParams The fx,fy,cx,cy estimate.
  /// @param[in] id The (unique) ID of this block.
  /// @param[in] timestamp The timestamp of this state.
  EuclideanParamBlockSized(const Eigen::Matrix<double, Dim, 1>& intrinsicParams,
                           uint64_t id, const okvis::Time& timestamp) : linPointFixed_(false) {
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
    if (!linPointFixed_) {
      linPoint_ = intrinsicParams;
    }
  }

  void setLinPoint(const Eigen::Matrix<double, Dim, 1>& intrinsicParams) {
    linPoint_ = intrinsicParams;
    linPointFixed_ = true;
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

  Eigen::Matrix<double, Dim, 1> linPoint() const {
    return linPoint_;
  }

  /// @brief Return parameter block type as string
  virtual std::string typeInfo() const { return "EuclideanParamBlockSized"; }

 private:
  okvis::Time timestamp_;  ///< Time of this state.
  Eigen::Matrix<double, Dim, 1> linPoint_;
  bool linPointFixed_;
};
typedef EuclideanParamBlockSized<9> ShapeMatrixParamBlock;
}  // namespace ceres
}  // namespace okvis

#endif /* INCLUDE_OKVIS_CERES_EUCLIDEANPARAMBLOCKSIZED_HPP_ */
