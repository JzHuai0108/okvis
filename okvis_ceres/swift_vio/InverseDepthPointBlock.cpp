
/**
 * @file InverseDepthPointBlock.cpp
 * @brief Source file for the InverseDepthPointBlock class.
 * @author
 */

#include <swift_vio/ceres/InverseDepthPointBlock.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {
/// \brief ceres Namespace for ceres-related functionality implemented in okvis.
namespace ceres {

// Default constructor (assumes not fixed).
InverseDepthPointBlock::InverseDepthPointBlock()
    : base_t::ParameterBlockSized(),
      initialized_(false) {
  setFixed(false);
}
// Trivial destructor.
InverseDepthPointBlock::~InverseDepthPointBlock() {
}

// Constructor with estimate and time.
InverseDepthPointBlock::InverseDepthPointBlock(
    const Eigen::Vector4d& point, uint64_t id, bool initialized) {
  setEstimate(point);
  setId(id);
  setInitialized(initialized);
  setFixed(false);
}

// Constructor with estimate and time.
InverseDepthPointBlock::InverseDepthPointBlock(
    const Eigen::Vector3d& point, uint64_t id, bool initialized) {
  double invz = 1.0 / point[2];
  setEstimate(Eigen::Vector4d(point[0] * invz, point[1] * invz, 1.0, invz));
  setId(id);
  setInitialized(initialized);
  setFixed(false);
}

// setters
// Set estimate of this parameter block.
void InverseDepthPointBlock::setEstimate(const Eigen::Vector4d& point) {
  for (int i = 0; i < base_t::Dimension; ++i)
    parameters_[i] = point[i];
}

// getters
// Get estimate.
Eigen::Vector4d InverseDepthPointBlock::estimate() const {
  return Eigen::Vector4d(parameters_[0], parameters_[1], parameters_[2],
                         parameters_[3]);
}

}  // namespace ceres
}  // namespace okvis
