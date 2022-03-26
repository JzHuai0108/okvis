#include <okvis/Parameters.hpp>
#include <unordered_map>

namespace okvis {
CameraNoiseParameters::CameraNoiseParameters()
    : sigma_absolute_translation(0.0),
      sigma_absolute_orientation(0.0),
      sigma_c_relative_translation(0.0),
      sigma_c_relative_orientation(0.0),
      sigma_focal_length(0.0),
      sigma_principal_point(0.0),
      sigma_td(0.0),
      sigma_tr(0.0),
      sigma_observation(1.0) {
}

CameraNoiseParameters::CameraNoiseParameters(
    double sigma_absolute_translation, double sigma_absolute_orientation,
    double sigma_c_relative_translation, double sigma_c_relative_orientation)
    : sigma_absolute_translation(sigma_absolute_translation),
      sigma_absolute_orientation(sigma_absolute_orientation),
      sigma_c_relative_translation(sigma_c_relative_translation),
      sigma_c_relative_orientation(sigma_c_relative_orientation) {
}

ImuParameters::ImuParameters()
    : T_BS(),
      a_max(200.0),
      g_max(10),
      sigma_g_c(1.2e-3),
      sigma_a_c(8e-3),
      sigma_bg(0.03),
      sigma_ba(0.1),
      sigma_gw_c(4e-6),
      sigma_aw_c(4e-5),
      tau(3600.0),
      g(9.80665),
      rate(100),
      sigma_Mg_element(5e-3),
      sigma_Ts_element(1e-3),
      sigma_Ma_element(5e-3),
      imuIdx(0u),
      model_type("BG_BA_MG_TS_MA"),
      estimateGravityDirection(false),
      sigmaGravityDirection(0.0),
      g0(0, 0, 0),
      a0(0, 0, 0),
      normalGravity(0, 0, -1) {
  Eigen::Matrix<double, 9, 1> eye;
  eye << 1, 0, 0, 0, 1, 0, 0, 0, 1;
  Mg0 = eye;
  Ts0.setZero();
  Ma0 << 1, 0, 1, 0, 0, 1;
}

const Eigen::Vector3d &ImuParameters::gravityDirection() const {
  return normalGravity;
}

Eigen::Vector3d ImuParameters::gravity() const {
  return g * normalGravity;
}

void ImuParameters::setGravityDirection(
    const Eigen::Vector3d &gravityDirection) {
  normalGravity = gravityDirection;
}

EstimatorOptions::EstimatorOptions(int _max_iterations, int _min_iterations,
                           double _timeLimitForMatchingAndOptimization,
                           okvis::Duration _timeReserve, int _numKeyframes,
                           int _numImuFrames,
                           swift_vio::EstimatorAlgorithm _algorithm,
                           bool _useEpipolarConstraint,
                           int _cameraObservationModelId,
                           bool _computeOkvisNees, bool _useMahalanobisGating,
                           double _maxProjectionErrorTol,
                           int _delayInitByFrames,
                           int _numThreads, bool _verbose)
    : max_iterations(_max_iterations), min_iterations(_min_iterations),
      timeLimitForMatchingAndOptimization(_timeLimitForMatchingAndOptimization),
      timeReserve(_timeReserve), numKeyframes(_numKeyframes),
      numImuFrames(_numImuFrames), algorithm(_algorithm),
      useEpipolarConstraint(_useEpipolarConstraint),
      cameraObservationModelId(_cameraObservationModelId),
      computeOkvisNees(_computeOkvisNees),
      useMahalanobisGating(_useMahalanobisGating),
      maxProjectionErrorTol(_maxProjectionErrorTol),
      delayFilterInitByFrames(_delayInitByFrames),
      numThreads(_numThreads), verbose(_verbose) {}

std::string EstimatorOptions::toString(std::string lead) const {
  std::stringstream ss(lead);
  ss << "Algorithm " << algorithm << ", numKeyframes " << numKeyframes
     << ", numImuFrames " << numImuFrames << ".\nUse epipolar constraint? "
     << useEpipolarConstraint << ", camera observation model Id "
     << cameraObservationModelId << ", compute OKVIS NEES? " << computeOkvisNees
     << ".\nMahalanobis gating? " << useMahalanobisGating
     << ", max projection error " << maxProjectionErrorTol
     << " (px).\nDelay filter initialization by #frames: "
     << delayFilterInitByFrames << ".";
  return ss.str();
}
}  // namespace okvis
