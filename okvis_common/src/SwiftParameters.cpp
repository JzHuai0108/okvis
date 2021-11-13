#include <swift_vio/SwiftParameters.hpp>
#include <algorithm>
#include <sstream>
#include <unordered_map>

namespace swift_vio {
EstimatorAlgorithm EstimatorAlgorithmNameToId(std::string description) {
  std::transform(description.begin(), description.end(), description.begin(),
                 ::toupper);
  std::unordered_map<std::string, EstimatorAlgorithm> descriptionToId{
      {"OKVIS", EstimatorAlgorithm::OKVIS},
      {"MSCKF", EstimatorAlgorithm::MSCKF},
      {"TFVIO", EstimatorAlgorithm::TFVIO},
      {"SLIDINGWINDOWSMOOTHER", EstimatorAlgorithm::SlidingWindowSmoother},
      {"RISLIDINGWINDOWSMOOTHER", EstimatorAlgorithm::RiSlidingWindowSmoother},
      {"HYBRIDFILTER", EstimatorAlgorithm::HybridFilter},
      {"CALIBRATIONFILTER", EstimatorAlgorithm::CalibrationFilter},
  };

  auto iter = descriptionToId.find(description);
  if (iter == descriptionToId.end()) {
    return EstimatorAlgorithm::OKVIS;
  } else {
    return iter->second;
  }
}

struct EstimatorAlgorithmHash {
  template <typename T>
  std::size_t operator()(T t) const {
    return static_cast<std::size_t>(t);
  }
};

std::string EstimatorAlgorithmIdToName(EstimatorAlgorithm id) {
  std::unordered_map<EstimatorAlgorithm, std::string, EstimatorAlgorithmHash>
      idToDescription{
          {EstimatorAlgorithm::OKVIS, "OKVIS"},
          {EstimatorAlgorithm::MSCKF, "MSCKF"},
          {EstimatorAlgorithm::TFVIO, "TFVIO"},
          {EstimatorAlgorithm::SlidingWindowSmoother, "SlidingWindowSmoother"},
          {EstimatorAlgorithm::RiSlidingWindowSmoother,
           "RiSlidingWindowSmoother"},
          {EstimatorAlgorithm::HybridFilter, "HybridFilter"},
          {EstimatorAlgorithm::CalibrationFilter, "CalibrationFilter"},
      };
  auto iter = idToDescription.find(id);
  if (iter == idToDescription.end()) {
    return "OKVIS";
  } else {
    return iter->second;
  }
}

FrontendOptions::FrontendOptions(int _featureTrackingMethod,
                                 bool _useMedianFilter, int _detectionOctaves,
                                 double _detectionThreshold,
                                 int _maxNoKeypoints,
                                 double _triangulationTranslationThreshold,
                                 double _triangulationMaxDepth,
                                 float _keyframeInsertionOverlapThreshold,
                                 float _keyframeInsertionMatchingRatioThreshold,
                                 bool _stereoWithEpipolarCheck,
                                 double _epipolarDistanceThreshold)
    : featureTrackingMethod(_featureTrackingMethod),
      useMedianFilter(_useMedianFilter), detectionOctaves(_detectionOctaves),
      detectionThreshold(_detectionThreshold), maxNoKeypoints(_maxNoKeypoints),
      triangulationTranslationThreshold(_triangulationTranslationThreshold),
      triangulationMaxDepth(_triangulationMaxDepth),
      keyframeInsertionOverlapThreshold(_keyframeInsertionOverlapThreshold),
      keyframeInsertionMatchingRatioThreshold(
          _keyframeInsertionMatchingRatioThreshold),
      stereoMatchWithEpipolarCheck(_stereoWithEpipolarCheck),
      epipolarDistanceThreshold(_epipolarDistanceThreshold) {}

PoseGraphOptions::PoseGraphOptions()
    : maxOdometryConstraintForAKeyframe(3), minDistance(0.1), minAngle(0.1) {}

PointLandmarkOptions::PointLandmarkOptions()
    : landmarkModelId(0), minTrackLengthForMsckf(3u),
      maxHibernationFrames(3u),
      minTrackLengthForSlam(11u), maxInStateLandmarks(50),
      maxMarginalizedLandmarks(50) {}

PointLandmarkOptions::PointLandmarkOptions(
    int lmkModelId, size_t minMsckfTrackLength,
    size_t hibernationFrames, size_t minSlamTrackLength, int maxStateLandmarks,
    int maxMargedLandmarks)
    : landmarkModelId(lmkModelId), minTrackLengthForMsckf(minMsckfTrackLength),
      maxHibernationFrames(hibernationFrames),
      minTrackLengthForSlam(minSlamTrackLength),
      maxInStateLandmarks(maxStateLandmarks),
      maxMarginalizedLandmarks(maxMargedLandmarks) {}

std::string PointLandmarkOptions::toString(std::string lead) const {
  std::stringstream ss(lead);
  ss << "Landmark model id " << landmarkModelId << "\n#hibernation frames "
     << maxHibernationFrames << " track length for MSCKF "
     << minTrackLengthForMsckf << " for SLAM " << minTrackLengthForSlam
     << ". Max landmarks in state " << maxInStateLandmarks
     << ", max landmarks marginalized in one update step "
     << maxMarginalizedLandmarks << ".";
  return ss.str();
}
}  // namespace swift_vio
