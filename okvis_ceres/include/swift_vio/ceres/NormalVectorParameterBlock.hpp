
#ifndef INCLUDE_SWIFTVIO_CERES_NORMALVECTORPARAMETERBLOCK_HPP_
#define INCLUDE_SWIFTVIO_CERES_NORMALVECTORPARAMETERBLOCK_HPP_

#include <Eigen/Core>
#include <okvis/ceres/ParameterBlockSized.hpp>

namespace okvis {
namespace ceres {

/// \brief Wraps the parameter block for a pose estimate
class NormalVectorParameterBlock
    : public ParameterBlockSized<3, 2, Eigen::Vector3d> {
public:
  /// \brief The estimate type.
  typedef Eigen::Vector3d estimate_t;

  /// \brief The base class type.
  typedef ParameterBlockSized<3, 2, Eigen::Vector3d> base_t;

  NormalVectorParameterBlock();

  NormalVectorParameterBlock(const Eigen::Vector3d vecIn, uint64_t id);

  virtual ~NormalVectorParameterBlock();

  // setters
  /// @brief Set estimate of this parameter block.
  /// @param[in] vecIn The estimate to set this to.
  void setEstimate(const Eigen::Vector3d &vecIn) final;

  // getters
  /// @brief Get estimate.
  /// \return The estimate.
  Eigen::Vector3d estimate() const final;

  // minimal internal parameterization
  // x0_plus_Delta=Delta_Chi[+]x0
  /// \brief Generalization of the addition operation,
  ///        x_plus_delta = Plus(x, delta)
  ///        with the condition that Plus(x, 0) = x.
  /// @param[in] x0 Variable.
  /// @param[in] Delta_Chi Perturbation.
  /// @param[out] x0_plus_Delta Perturbed x.
  virtual void plus(const double *x0, const double *Delta_Chi,
                    double *x0_plus_Delta) const;

  /// \brief The jacobian of Plus(x, delta) w.r.t delta at delta = 0.
  /// @param[in] x0 Variable.
  /// @param[out] jacobian The Jacobian.
  virtual void plusJacobian(const double *x0, double *jacobian) const;

  // Delta_Chi=x0_plus_Delta[-]x0
  /// \brief Computes the minimal difference between a variable x and a
  /// perturbed variable x_plus_delta
  /// @param[in] x0 Variable.
  /// @param[in] x0_plus_Delta Perturbed variable.
  /// @param[out] Delta_Chi Minimal difference.
  /// \return True on success.
  virtual void minus(const double *x0, const double *x0_plus_Delta,
                     double *Delta_Chi) const;

  /// \brief Computes the Jacobian from minimal space to naively
  /// overparameterised space as used by ceres.
  /// @param[in] x0 Variable.
  /// @param[out] jacobian the Jacobian (dimension minDim x dim).
  /// \return True on success.
  virtual void liftJacobian(const double *x0, double *jacobian) const;

  /// @brief Return parameter block type as string
  std::string typeInfo() const final;
};

} // namespace ceres
} // namespace okvis

#endif  // INCLUDE_SWIFTVIO_CERES_NORMALVECTORPARAMETERBLOCK_HPP_
