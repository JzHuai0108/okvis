
/**
 * @file EuclideanParamBlockSizedLin.hpp
 * @brief Header file for the EuclideanParamBlockSizedLin class.
 * Compared to EuclideanParamBlockSizedLin, this class maintains the linearization point for the parameter block.
 * @author Jianzhu Huai
 */

#ifndef INCLUDE_OKVIS_CERES_EUCLIDEANPARAMBLOCKSIZEDLIN_HPP_
#define INCLUDE_OKVIS_CERES_EUCLIDEANPARAMBLOCKSIZEDLIN_HPP_

#include <Eigen/Core>
#include <okvis/Time.hpp>
#include <okvis/ceres/ParameterBlockSized.hpp>

namespace okvis {
/// \brief ceres Namespace for ceres-related functionality implemented in okvis.
namespace ceres {

template <int Dim>
class EuclideanParamBlockSizedLin
    : public okvis::ceres::ParameterBlockSized<Dim, Dim, Eigen::Matrix<double, Dim, 1>> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  /// \brief The base class type.
  typedef okvis::ceres::ParameterBlockSized<Dim, Dim, Eigen::Matrix<double, Dim, 1>> base_t;

  /// \brief The estimate type (9D vector).
  typedef Eigen::Matrix<double, Dim, 1> estimate_t;

  /// \brief Default constructor (assumes not fixed).
  EuclideanParamBlockSizedLin() : base_t::ParameterBlockSized(), linPointFixed_(false) {
    okvis::ceres::ParameterBlock::setFixed(false);
  }

  /// \brief Constructor with estimate and time.
  /// @param[in] intrinsicParams The fx,fy,cx,cy estimate.
  /// @param[in] id The (unique) ID of this block.
  /// @param[in] timestamp The timestamp of this state.
  EuclideanParamBlockSizedLin(const Eigen::Matrix<double, Dim, 1>& intrinsicParams,
                           uint64_t id) : linPointFixed_(false) {
    setEstimate(intrinsicParams);
    okvis::ceres::ParameterBlock::setId(id);
    okvis::ceres::ParameterBlock::setFixed(false);
  }

  /// \brief Trivial destructor.
  virtual ~EuclideanParamBlockSizedLin() {}

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

  void fixLinPoint(const Eigen::Matrix<double, Dim, 1>& intrinsicParams) {
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
  virtual std::string typeInfo() const { return "EuclideanParamBlockSizedLin"; }

 private:
  okvis::Time timestamp_;  ///< Time of this state.
  Eigen::Matrix<double, Dim, 1> linPoint_;
  bool linPointFixed_;
};
typedef EuclideanParamBlockSizedLin<3> SpeedParameterBlock;
}  // namespace ceres
}  // namespace okvis

#endif /* INCLUDE_OKVIS_CERES_EUCLIDEANPARAMBLOCKSIZEDLIN_HPP_ */
