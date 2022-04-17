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
      sigma_observation(1.0),
      intrinsics_fixed_(true), extrinsics_fixed_(true) {
}

CameraNoiseParameters::CameraNoiseParameters(
    double sigma_absolute_translation, double sigma_absolute_orientation,
    double sigma_c_relative_translation, double sigma_c_relative_orientation)
    : sigma_absolute_translation(sigma_absolute_translation),
      sigma_absolute_orientation(sigma_absolute_orientation),
      sigma_c_relative_translation(sigma_c_relative_translation),
      sigma_c_relative_orientation(sigma_c_relative_orientation),
      sigma_focal_length(0.0), sigma_principal_point(0.0), sigma_td(0.0),
      sigma_tr(0.0), sigma_observation(1.0), intrinsics_fixed_(true) {
  extrinsics_fixed_ = isExtrinsicsFixed();
}

std::string CameraNoiseParameters::toString() const {
  std::stringstream ss;
  ss << "sigma_absolute_translation " << sigma_absolute_translation
     << ", sigma_absolute_orientation " << sigma_absolute_orientation
     << ", sigma_c_relative_translation " << sigma_c_relative_translation
     << ", sigma_c_relative_orientation " << sigma_c_relative_orientation
     << ".\nsigma_focal_length " << sigma_focal_length
     << ", sigma_principal_point " << sigma_principal_point << ".\n";
  if (sigma_distortion.size()) {
    ss << "sigma_distortion [" << sigma_distortion[0];
    for (size_t i = 1; i < sigma_distortion.size(); ++i) {
      ss << ", " << sigma_distortion[i];
    }
    ss << "].\n";
  }
  ss << "sigma_td " << sigma_td << ", sigma_tr " << sigma_tr
     << ", sigma_observaiton " << sigma_observation << ".\n";
  return ss.str();
}

bool CameraNoiseParameters::isIntrinsicsFixed() const {
  bool projIntrinsicsFixed = sigma_focal_length == 0.0 && sigma_principal_point == 0.0;
  bool distortionIntrinsicsFixed = true;
  for (size_t i = 0; i < sigma_distortion.size(); ++i) {
    if (sigma_distortion[i] > 0) {
      distortionIntrinsicsFixed = false;
      break;
    }
  }
  return projIntrinsicsFixed && distortionIntrinsicsFixed;
}

bool CameraNoiseParameters::isExtrinsicsFixed() const {
  return sigma_absolute_translation == 0.0 &&
      sigma_absolute_orientation == 0.0;
}

void CameraNoiseParameters::updateParameterStatus() {
  intrinsics_fixed_ = isIntrinsicsFixed();
  extrinsics_fixed_ = isExtrinsicsFixed();
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
      sigma_Mg_element(0.0),
      sigma_Ts_element(0.0),
      sigma_Ma_element(0.0),
      imuIdx(0),
      model_name("BG_BA_MG_TS_MA"),
      sigma_gravity_direction(0.0),
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

std::string ImuParameters::toString() const {
  std::stringstream ss;
  ss << "a max " << a_max << ", g max " << g_max << ", sigma_g_c " << sigma_g_c
            << ", sigma_a_c " << sigma_a_c << ", sigma_gw_c " << sigma_gw_c << ", sigma_aw_c "
            << sigma_aw_c << ".\n";
  ss << "sigma_bg " << sigma_bg << ", sigma ba " << sigma_ba << ", g " << g << " unit gravity "
     << normalGravity.transpose() << ",\nsigma gravity direction " << sigma_gravity_direction
     << ".\n";

  ss << "rate " << rate << ", imu idx " << imuIdx << ", imu model " << model_name << ".\n";
  ss << "sigma_Mg_element " << sigma_Mg_element << ", sigma_Ts_element " << sigma_Ts_element
     << ", sigma_Ma_element " << sigma_Ma_element << ".\n";
  ss << "g0 " << g0.transpose() << ", a0 " << a0.transpose() << ".\nMg0 " << Mg0.transpose()
     << ".\nTs0 " << Ts0.transpose() << ".\nMa0 " << Ma0.transpose() << ".\n";
  return ss.str();
}

EstimatorOptions::EstimatorOptions(
    swift_vio::EstimatorAlgorithm _algorithm,
    swift_vio::EstimatorAlgorithm _initializer,
    int _max_iterations,
    int _min_iterations, double _timeLimitForMatchingAndOptimization,
    okvis::Duration _timeReserve, int _numKeyframes, int _numImuFrames,
    bool _constantBias, bool _useEpipolarConstraint,
    int _cameraObservationModelId, bool _computeOkvisNees,
    bool _useMahalanobisGating, double _maxProjectionErrorTol,
    int _delayInitByFrames, int _numThreads, bool _verbose)
    : algorithm(_algorithm),
      initializer(_initializer),
      max_iterations(_max_iterations),
      min_iterations(_min_iterations),
      timeLimitForMatchingAndOptimization(_timeLimitForMatchingAndOptimization),
      timeReserve(_timeReserve), numKeyframes(_numKeyframes),
      numImuFrames(_numImuFrames), constantBias(_constantBias),
      useEpipolarConstraint(_useEpipolarConstraint),
      cameraObservationModelId(_cameraObservationModelId),
      computeOkvisNees(_computeOkvisNees),
      useMahalanobisGating(_useMahalanobisGating),
      maxProjectionErrorTol(_maxProjectionErrorTol),
      delayFilterInitByFrames(_delayInitByFrames), numThreads(_numThreads),
      verbose(_verbose) {}

std::string EstimatorOptions::toString(std::string lead) const {
  std::stringstream ss(lead);
  ss << "Algorithm " << algorithm << ", initializer " << initializer
     << ", numKeyframes " << numKeyframes
     << ", numImuFrames " << numImuFrames << ".\nConstant bias? "
     << constantBias << ", use epipolar constraint? " << useEpipolarConstraint
     << ", camera observation model Id " << cameraObservationModelId
     << ", compute OKVIS NEES? " << computeOkvisNees
     << ".\nMahalanobis gating? " << useMahalanobisGating
     << ", max projection error " << maxProjectionErrorTol
     << " (px).\nDelay filter initialization by #frames: "
     << delayFilterInitByFrames << ".";
  return ss.str();
}
}  // namespace okvis
