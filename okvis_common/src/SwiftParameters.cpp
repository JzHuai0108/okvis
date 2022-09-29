#include <swift_vio/SwiftParameters.hpp>
#include <algorithm>
#include <sstream>
#include <unordered_map>

namespace swift_vio {
bool EnumFromString(std::string description, EstimatorAlgorithm *e) {
  std::transform(description.begin(), description.end(), description.begin(),
                 ::toupper);
  std::unordered_map<std::string, EstimatorAlgorithm> descriptionToId{
      {"SLIDINGWINDOWSMOOTHER", EstimatorAlgorithm::SlidingWindowSmoother},
      {"FIXEDLAGSMOOTHER", EstimatorAlgorithm::FixedLagSmoother},
      {"RIFIXEDLAGSMOOTHER", EstimatorAlgorithm::RiFixedLagSmoother},
      {"OKVISESTIMATOR", EstimatorAlgorithm::OkvisEstimator},
      {"SLIDINGWINDOWFILTER", EstimatorAlgorithm::SlidingWindowFilter},
      {"IMUINITIALIZER", EstimatorAlgorithm::ImuInitializer},
      {"VIOINITIALIZER", EstimatorAlgorithm::VioInitializer}};
  auto iter = descriptionToId.find(description);
  if (iter == descriptionToId.end()) {
    *e = EstimatorAlgorithm::SlidingWindowSmoother;
    return false;
  } else {
    *e = iter->second;
  }
  return true;
}

std::string EnumToString(EstimatorAlgorithm a) {
  const std::string names[] = {"SlidingWindowSmoother", "FixedLagSmoother",
                               "RiFixedLagSmoother",    "OkvisEstimator",
                               "SlidingWindowFilter",   "ImuInitializer",
                               "VioInitializer"};
  return names[static_cast<int>(a)];
}

std::string EnumToString(FeatureTrackingScheme s) {
  const std::string names[] = {"KeyframeDescriptorMatching", "FramewiseKLT",
                               "FramewiseDescriptorMatching"};
  return names[static_cast<int>(s)];
}

std::string BriskOptions::toString(std::string hint) const {
  std::stringstream ss(hint);
  ss << "DetectionAbsoluteThreshold " << detectionAbsoluteThreshold
     << ", DescriptionRotationInvariance "
     << descriptionRotationInvariance
     << ", DescriptionScaleInvariance " << descriptionScaleInvariance
     << ", MatchingThreshold " << matchingThreshold << ".\n";
  return ss.str();
}

FrontendOptions::FrontendOptions(
    FeatureTrackingScheme _featureTrackingMethod, BriskOptions _brisk,
    bool _useMedianFilter, HistogramMethod hm,
    int _detectionOctaves, double _detectionThreshold,
    int _maxNoKeypoints, float _keyframeInsertionOverlapThreshold,
    float _keyframeInsertionMatchingRatioThreshold,
    bool _stereoWithEpipolarCheck, double _epipolarDistanceThreshold,
    size_t _numOldKeyframesToMatch, size_t _numKeyframesToMatch, int _numThreads, bool allKeyframes)
    : featureTrackingMethod(_featureTrackingMethod), brisk(_brisk),
      useMedianFilter(_useMedianFilter), histogramMethod(hm),
      detectionOctaves(_detectionOctaves),
      detectionThreshold(_detectionThreshold), maxNoKeypoints(_maxNoKeypoints),
      keyframeInsertionOverlapThreshold(_keyframeInsertionOverlapThreshold),
      keyframeInsertionMatchingRatioThreshold(
          _keyframeInsertionMatchingRatioThreshold),
      stereoMatchWithEpipolarCheck(_stereoWithEpipolarCheck),
      epipolarDistanceThreshold(_epipolarDistanceThreshold),
      numOldKeyframesToMatch(_numOldKeyframesToMatch),
      numKeyframesToMatch(_numKeyframesToMatch), numThreads(_numThreads),
      allAreKeyframes(allKeyframes) {}

std::string FrontendOptions::toString(std::string hint) const {
  std::stringstream ss(hint);
  ss << "Feature tracking method " << featureTrackingMethod << ".\n"
     << brisk.toString("Brisk options ") << "useMedianFilter "
     << useMedianFilter << ", histogram method " << histogramMethod
     << ", detectionOctaves " << detectionOctaves << ", detectionThreshold "
     << detectionThreshold << ", maxNoKeypoints " << maxNoKeypoints
     << ".\nkeyframeInsertionOverlapThreshold "
     << keyframeInsertionOverlapThreshold
     << ", keyframeInsertionMatchingRatioThreshold "
     << keyframeInsertionMatchingRatioThreshold
     << ". stereoMatchWithEpipolarCheck " << stereoMatchWithEpipolarCheck
     << ", epipolarDistanceThreshold " << epipolarDistanceThreshold
     << ".\nnumOldKeyframesToMatch " << numOldKeyframesToMatch
     << ", numKeyframesToMatch " << numKeyframesToMatch << ", numThreads "
     << numThreads << ".\nAll frames are keyframes? " << allAreKeyframes << ".\n";
  return ss.str();
}

PoseGraphOptions::PoseGraphOptions(int _maxOdometryConstraintForAKeyframe,
                                   double _minDistance, double _minAngle)
    : maxOdometryConstraintForAKeyframe(_maxOdometryConstraintForAKeyframe),
      minDistance(_minDistance), minAngle(_minAngle) {}

PointLandmarkOptions::PointLandmarkOptions(
    int lmkModelId, size_t minMsckfTrackLength, size_t hibernationFrames,
    size_t minSlamTrackLength, int maxStateLandmarks, int maxMargedLandmarks,
    double _triangulationMaxDepth)
    : landmarkModelId(lmkModelId), minTrackLengthForMsckf(minMsckfTrackLength),
      maxHibernationFrames(hibernationFrames),
      minTrackLengthForSlam(minSlamTrackLength),
      maxInStateLandmarks(maxStateLandmarks),
      maxMarginalizedLandmarks(maxMargedLandmarks),
      triangulationMaxDepth(_triangulationMaxDepth) {}

std::string PointLandmarkOptions::toString(std::string hint) const {
  std::stringstream ss(hint);
  ss << "Landmark model id " << landmarkModelId << "\n#hibernation frames "
     << maxHibernationFrames << " track length for MSCKF "
     << minTrackLengthForMsckf << " for SLAM " << minTrackLengthForSlam
     << ".\nMax landmarks in state " << maxInStateLandmarks
     << ", max landmarks marginalized in one update step "
     << maxMarginalizedLandmarks << ", max depth in triangulation "
     << triangulationMaxDepth << ".";
  return ss.str();
}
}  // namespace swift_vio
