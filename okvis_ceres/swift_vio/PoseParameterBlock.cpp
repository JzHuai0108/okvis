/**
 * @file PoseParameterBlock.cpp
 * @brief Source file for the PoseParameterBlock class.
 */

#include <swift_vio/ceres/PoseParameterBlock.hpp>

namespace swift_vio {

// Default constructor (assumes not fixed).
PoseParameterBlock::PoseParameterBlock()
    : base_t::ParameterBlockSized(), linPointFixed_(false) {
  setFixed(false);
}

// Trivial destructor.
PoseParameterBlock::~PoseParameterBlock() {
}

// Constructor with estimate and time.
PoseParameterBlock::PoseParameterBlock(
    const okvis::kinematics::Transformation& T_WS, uint64_t id,
    const okvis::Time& timestamp) : linPointFixed_(false) {
  setEstimate(T_WS);
  setId(id);
  setFixed(false);
}

// setters
// Set estimate of this parameter block.
void PoseParameterBlock::setEstimate(
    const okvis::kinematics::Transformation& T_WS) {
  const Eigen::Vector3d r = T_WS.r();
  const Eigen::Vector4d q = T_WS.q().coeffs();
  parameters_[0] = r[0];
  parameters_[1] = r[1];
  parameters_[2] = r[2];
  parameters_[3] = q[0];
  parameters_[4] = q[1];
  parameters_[5] = q[2];
  parameters_[6] = q[3];

  if (!linPointFixed_) {
    pLinPoint_ = r;
    qLinPoint_ = T_WS.q();
  }
}

void PoseParameterBlock::setLinPoint(const okvis::kinematics::Transformation& T_WS) {
  pLinPoint_ = T_WS.r();
  qLinPoint_ = T_WS.q();
  linPointFixed_ = true;
}

// getters
// Get estimate.
okvis::kinematics::Transformation PoseParameterBlock::estimate() const {
  return okvis::kinematics::Transformation(
      Eigen::Vector3d(parameters_[0], parameters_[1], parameters_[2]),
      Eigen::Quaterniond(parameters_[6], parameters_[3], parameters_[4],
                         parameters_[5]));
}

okvis::kinematics::Transformation PoseParameterBlock::linPoint() const {
  return okvis::kinematics::Transformation(pLinPoint_, qLinPoint_);
}

}  // namespace swift_vio
