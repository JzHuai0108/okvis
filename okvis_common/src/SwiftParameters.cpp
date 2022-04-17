#include <swift_vio/SwiftParameters.hpp>
#include <algorithm>
#include <sstream>
#include <unordered_map>

namespace swift_vio {
EstimatorAlgorithm EstimatorAlgorithmNameToId(std::string description) {
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
    return EstimatorAlgorithm::SlidingWindowSmoother;
  } else {
    return iter->second;
  }
}

std::ostream &operator<<(std::ostream &strm, EstimatorAlgorithm a) {
  const std::string names[] = {"SlidingWindowSmoother", "FixedLagSmoother",
                               "RiFixedLagSmoother",    "OkvisEstimator",
                               "SlidingWindowFilter",   "ImuInitializer",
                               "VioInitializer"};
  return strm << names[static_cast<int>(a)];
}

struct EstimatorAlgorithmHash {
  template <typename T> std::size_t operator()(T t) const {
    return static_cast<std::size_t>(t);
  }
};

std::string EstimatorAlgorithmIdToName(EstimatorAlgorithm id) {
  std::unordered_map<EstimatorAlgorithm, std::string, EstimatorAlgorithmHash>
      idToDescription{
          {EstimatorAlgorithm::SlidingWindowSmoother, "SlidingWindowSmoother"},
          {EstimatorAlgorithm::FixedLagSmoother, "FixedLagSmoother"},
          {EstimatorAlgorithm::RiFixedLagSmoother, "RiFixedLagSmoother"},
          {EstimatorAlgorithm::OkvisEstimator, "OkvisEstimator"},
          {EstimatorAlgorithm::SlidingWindowFilter, "SlidingWindowFilter"},
          {EstimatorAlgorithm::ImuInitializer, "ImuInitializer"},
          {EstimatorAlgorithm::VioInitializer, "VioInitializer"}};
  auto iter = idToDescription.find(id);
  if (iter == idToDescription.end()) {
    return "SlidingWindowSmoother";
  } else {
    return iter->second;
  }
}

std::ostream &operator<<(std::ostream &strm, FeatureTrackingScheme s) {
  const std::string names[] = {"KeyframeDescriptorMatching", "FramewiseKLT",
                               "FramewiseDescriptorMatching"};
  return strm << names[static_cast<int>(s)];
}

std::string BriskOptions::toString(std::string hint) const {
  std::stringstream ss(hint);
  ss << "DetectionAbsoluteThreshold " << detectionAbsoluteThreshold
     << " DescriptionRotationInvariance "
     << descriptionRotationInvariance
     << " DescriptionScaleInvariance " << descriptionScaleInvariance
     << " MatchingThreshold " << matchingThreshold;
  return ss.str();
}

FrontendOptions::FrontendOptions(
    FeatureTrackingScheme _featureTrackingMethod, BriskOptions _brisk,
    bool _useMedianFilter, int _detectionOctaves, double _detectionThreshold,
    int _maxNoKeypoints, float _keyframeInsertionOverlapThreshold,
    float _keyframeInsertionMatchingRatioThreshold,
    bool _stereoWithEpipolarCheck, double _epipolarDistanceThreshold,
    size_t _numOldKeyframesToMatch, size_t _numKeyframesToMatch, int _numThreads)
    : featureTrackingMethod(_featureTrackingMethod), brisk(_brisk),
      useMedianFilter(_useMedianFilter), detectionOctaves(_detectionOctaves),
      detectionThreshold(_detectionThreshold), maxNoKeypoints(_maxNoKeypoints),
      keyframeInsertionOverlapThreshold(_keyframeInsertionOverlapThreshold),
      keyframeInsertionMatchingRatioThreshold(
          _keyframeInsertionMatchingRatioThreshold),
      stereoMatchWithEpipolarCheck(_stereoWithEpipolarCheck),
      epipolarDistanceThreshold(_epipolarDistanceThreshold),
      numOldKeyframesToMatch(_numOldKeyframesToMatch),
      numKeyframesToMatch(_numKeyframesToMatch), numThreads(_numThreads) {}

std::string FrontendOptions::toString(std::string hint) const {
  std::stringstream ss(hint);
  ss << "Feature tracking method " << featureTrackingMethod
     << "\nBrisk options " << brisk.toString("Brisk options ")
     << "\nuseMedianFilter " << useMedianFilter << " detectionOctaves "
     << detectionOctaves << " detectionThreshold " << detectionThreshold
     << " maxNoKeypoints " << maxNoKeypoints
     << "\nkeyframeInsertionOverlapThreshold "
     << keyframeInsertionOverlapThreshold
     << " keyframeInsertionMatchingRatioThreshold "
     << keyframeInsertionMatchingRatioThreshold
     << " stereoMatchWithEpipolarCheck " << stereoMatchWithEpipolarCheck
     << " epipolarDistanceThreshold " << epipolarDistanceThreshold
     << "\nnumOldKeyframesToMatch " << numOldKeyframesToMatch
     << " numKeyframesToMatch " << numKeyframesToMatch
     << " numThreads " << numThreads;
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
