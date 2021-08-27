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
      sigma_tr(0.0) {}

CameraNoiseParameters::CameraNoiseParameters(
    double sigma_absolute_translation, double sigma_absolute_orientation,
    double sigma_c_relative_translation, double sigma_c_relative_orientation)
    : sigma_absolute_translation(sigma_absolute_translation),
      sigma_absolute_orientation(sigma_absolute_orientation),
      sigma_c_relative_translation(sigma_c_relative_translation),
      sigma_c_relative_orientation(sigma_c_relative_orientation) {}

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
      g0(0, 0, 0),
      a0(0, 0, 0),
      rate(100),
      sigma_TGElement(5e-3),
      sigma_TSElement(1e-3),
      sigma_TAElement(5e-3),
      model_type("BG_BA_TG_TS_TA"),
      estimateGravityDirection(false),
      sigmaGravityDirection(0.05),
      normalGravity(0, 0, -1) {
  Eigen::Matrix<double, 9, 1> eye;
  eye << 1, 0, 0, 0, 1, 0, 0, 0, 1;
  Tg0 = eye;
  Ts0.setZero();
  Ta0 = eye;
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

Optimization::Optimization()
    : max_iterations(10), min_iterations(3),
      timeLimitForMatchingAndOptimization(0.035), timeReserve(0.005),
      detectionThreshold(40), useMedianFilter(false), detectionOctaves(0),
      maxNoKeypoints(400), numKeyframes(5), numImuFrames(3),
      keyframeInsertionOverlapThreshold(0.6),
      keyframeInsertionMatchingRatioThreshold(0.2),
      algorithm(swift_vio::EstimatorAlgorithm::OKVIS),
      translationThreshold(0.4), rotationThreshold(0.2618),
      trackingRateThreshold(0.5), triangulationTranslationThreshold(-1.0),
      triangulationMaxDepth(1000.0), useEpipolarConstraint(false),
      cameraObservationModelId(0), computeOkvisNees(false),
      useMahalanobisGating(true), maxProjectionErrorTol(7) {}

std::string Optimization::toString(std::string lead) const {
  std::stringstream ss(lead);
  ss << "Algorithm " << EstimatorAlgorithmIdToName(algorithm)
     << " numKeyframes " << numKeyframes << " numImuFrames " << numImuFrames
     << "\nTranslation threshold for feature triangulation "
     << triangulationTranslationThreshold << " Max depth in triangulation "
     << triangulationMaxDepth << "\nUse epipolar constraint? "
     << useEpipolarConstraint << " Camera observation model Id "
     << cameraObservationModelId << " compute OKVIS NEES? " << computeOkvisNees
     << " Mahalanobis gating? " << useMahalanobisGating
     << " Max projection error " << maxProjectionErrorTol << " (px)";
  return ss.str();
}
}  // namespace okvis
