
#include <swift_vio/ceres/NormalVectorParameterBlock.hpp>
#include <swift_vio/ParallaxAnglePoint.hpp>

namespace okvis {
namespace ceres {

NormalVectorParameterBlock::NormalVectorParameterBlock()
    : base_t::ParameterBlockSized() {
  setFixed(false);
}

NormalVectorParameterBlock::NormalVectorParameterBlock(
    const Eigen::Vector3d vecIn, uint64_t id) {
  setEstimate(vecIn);
  setId(id);
  setFixed(false);
}

NormalVectorParameterBlock::~NormalVectorParameterBlock() {}

void NormalVectorParameterBlock::setEstimate(const Eigen::Vector3d &vecIn) {
  memcpy(parameters_, vecIn.data(), sizeof(double) * 3);
}

Eigen::Vector3d NormalVectorParameterBlock::estimate() const {
  return Eigen::Map<const Eigen::Vector3d>(parameters_);
}

void NormalVectorParameterBlock::plus(const double *x0, const double *Delta_Chi,
                                      double *x0_plus_Delta) const {
  swift_vio::NormalVectorParameterization::plus(x0, Delta_Chi, x0_plus_Delta);
}

void NormalVectorParameterBlock::plusJacobian(const double *x0,
                                              double *jacobian) const {
  swift_vio::NormalVectorParameterization::plusJacobian(x0, jacobian);
}

void NormalVectorParameterBlock::minus(const double *x0,
                                       const double *x0_plus_Delta,
                                       double *Delta_Chi) const {
  swift_vio::NormalVectorParameterization::minus(x0, x0_plus_Delta, Delta_Chi);
}

void NormalVectorParameterBlock::liftJacobian(const double *x0,
                                              double *jacobian) const {
  swift_vio::NormalVectorParameterization::liftJacobian(x0, jacobian);
}

std::string NormalVectorParameterBlock::typeInfo() const {
  return "NormalVectorParameterBlock";
}

} // namespace ceres
} // namespace okvis
