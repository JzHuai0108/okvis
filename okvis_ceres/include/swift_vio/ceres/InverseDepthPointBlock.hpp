
/**
 * @file InverseDepthPointBlock.hpp
 * @brief Header file for the InverseDepthPointBlock class.
 * @author
 */

#ifndef INCLUDE_OKVIS_CERES_INVERSEDEPTHPOINTBLOCK_HPP_
#define INCLUDE_OKVIS_CERES_INVERSEDEPTHPOINTBLOCK_HPP_

#include <okvis/ceres/ParameterBlockSized.hpp>
#include <swift_vio/PointLandmarkModels.hpp>
#include <Eigen/Core>

/// \brief okvis Main namespace of this package.
namespace okvis {
/// \brief ceres Namespace for ceres-related functionality implemented in okvis.
namespace ceres {

/// \brief Wraps the parameter block for a speed / IMU biases estimate
class InverseDepthPointBlock : public ParameterBlockSized<4, 3,
    Eigen::Vector4d>
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /// \brief The estimate type (4D vector).
  typedef Eigen::Vector4d estimate_t;

  /// \brief The base class type.
  typedef ParameterBlockSized<4, 3, estimate_t> base_t;

  /// \brief Default constructor (assumes not fixed).
  InverseDepthPointBlock();

  /// \brief Constructor with estimate and time.
  /// @param[in] point The homogeneous point estimate.
  /// @param[in] id The (unique) ID of this block.
  /// @param[in] initialized Whether or not the 3d position is considered initialised.
  InverseDepthPointBlock(const Eigen::Vector4d& point, uint64_t id,
                         bool initialized = true);

  /// \brief Constructor with estimate and time.
  /// @param[in] point The homogeneous point estimate.
  /// @param[in] id The (unique) ID of this block.
  /// @param[in] initialized Whether or not the 3d position is considered initialised.
  InverseDepthPointBlock(const Eigen::Vector3d& point, uint64_t id,
                         bool initialized = true);

  /// \brief Trivial destructor.
  virtual ~InverseDepthPointBlock();

  /// @name Setters
  /// @{

  /// @brief Set estimate of this parameter block.
  /// @param[in] point The estimate to set this to.
  virtual void setEstimate(const Eigen::Vector4d& point);

  /// \brief Set initialisaiton status.
  /// @param[in] initialized Whether or not the 3d position is considered initialised.
  void setInitialized(bool initialized)
  {
    initialized_ = initialized;
  }

  /// @}

  /// @name Getters
  /// @{

  /// @brief Get estimate
  /// \return The estimate.
  virtual Eigen::Vector4d estimate() const;

  /// \brief Get initialisaiton status.
  /// \return Whether or not the 3d position is considered initialised.
  bool initialized() const
  {
    return initialized_;
  }

  /// @}

  // minimal internal parameterization
  // x0_plus_Delta=Delta_Chi[+]x0
  /// \brief Generalization of the addition operation,
  ///        x_plus_delta = Plus(x, delta)
  ///        with the condition that Plus(x, 0) = x.
  /// @param[in] x0 Variable.
  /// @param[in] Delta_Chi Perturbation.
  /// @param[out] x0_plus_Delta Perturbed x.
  virtual void plus(const double* x0, const double* Delta_Chi,
                    double* x0_plus_Delta) const
  {
    swift_vio::InverseDepthParameterization::plus(x0, Delta_Chi, x0_plus_Delta);
  }

  /// \brief The jacobian of Plus(x, delta) w.r.t delta at delta = 0.
  /// @param[in] x0 Variable.
  /// @param[out] jacobian The Jacobian.
  virtual void plusJacobian(const double* x0, double* jacobian) const
  {
    swift_vio::InverseDepthParameterization::plusJacobian(x0, jacobian);
  }

  // Delta_Chi=x0_plus_Delta[-]x0
  /// \brief Computes the minimal difference between a variable x and a perturbed variable x_plus_delta
  /// @param[in] x0 Variable.
  /// @param[in] x0_plus_Delta Perturbed variable.
  /// @param[out] Delta_Chi Minimal difference.
  /// \return True on success.
  virtual void minus(const double* x0, const double* x0_plus_Delta,
                     double* Delta_Chi) const
  {
    swift_vio::InverseDepthParameterization::minus(x0, x0_plus_Delta, Delta_Chi);
  }

  /// \brief Computes the Jacobian from minimal space to naively overparameterised space as used by ceres.
  /// @param[in] x0 Variable.
  /// @param[out] jacobian the Jacobian (dimension minDim x dim).
  /// \return True on success.
  virtual void liftJacobian(const double* x0, double* jacobian) const
  {
    swift_vio::InverseDepthParameterization::liftJacobian(x0, jacobian);
  }

  /// @brief Return parameter block type as string
  virtual std::string typeInfo() const
  {
    return "InverseDepthPointBlock";
  }

 private:
  bool initialized_;  ///< Whether or not the 3d position is considered initialised.

};

}  // namespace ceres
}  // namespace okvis

#endif /* INCLUDE_OKVIS_CERES_INVERSEDEPTHPOINTBLOCK_HPP_ */
