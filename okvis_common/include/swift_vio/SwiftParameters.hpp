
/**
 * @file Parameters.hpp
 * @brief This file contains struct definitions that encapsulate parameters and settings for swift_vio.
 * @author Jianzhu Huai
 */

#ifndef INCLUDE_SWIFT_VIO_PARAMETERS_HPP_
#define INCLUDE_SWIFT_VIO_PARAMETERS_HPP_

#include <string>
#include <iostream>

namespace swift_vio {
enum class EstimatorAlgorithm {
  SlidingWindowSmoother = 0,  ///< keyframe-based smoother.
  FixedLagSmoother, ///< Gtsam::FixedLagSmoother.
  RiFixedLagSmoother, ///< Gtsam::FixedLagSmoother with right invariant errors.
  SlidingWindowFilter, ///< keyframe-based filter.
};

EstimatorAlgorithm EstimatorAlgorithmNameToId(std::string description);

std::ostream &operator<<(std::ostream &strm, EstimatorAlgorithm a);

std::string EstimatorAlgorithmIdToName(EstimatorAlgorithm id);

enum class FeatureTrackingScheme {
  KeyframeDescriptorMatching = 0,   ///< default, keyframe and back-to-back frame matching
  FramewiseKLT,  ///< KLT back-to-back frame matching,
  FramewiseDescriptorMatching, ///< back-to-back descriptor-based frame matching
  SingleThreadKeyframeDescMatching
};

std::ostream &operator<<(std::ostream &strm, FeatureTrackingScheme s);

struct BriskOptions {
  double detectionAbsoluteThreshold;
  bool descriptionRotationInvariance;
  bool descriptionScaleInvariance;
  double matchingThreshold;

  /**
   * @brief BriskOptions
   * @param detectionThreshold
   * @param rotationInvariance false following maplab setting, see maplab/algorithms/feature-tracking/src/feature-tracking-types.cc,
   * note it is true in okvis where the descriptor is computed relative to given gravity direction.
   * @param scaleInvariance true following maplab setting, note it is false in okvis.
   * @param threshold
   */
  BriskOptions(double detectionThreshold = 800.0,
               bool rotationInvariance = true,
               bool scaleInvariance = false,
               double threshold = 60.0)
      : detectionAbsoluteThreshold(detectionThreshold),
        descriptionRotationInvariance(rotationInvariance),
        descriptionScaleInvariance(scaleInvariance),
        matchingThreshold(threshold) {}
  std::string toString(std::string hint) const;
};

struct FrontendOptions {
  FeatureTrackingScheme featureTrackingMethod;

  BriskOptions brisk;
  bool useMedianFilter;     ///< Use a Median filter over captured image?
  int detectionOctaves;     ///< Number of keypoint detection octaves.
  double detectionThreshold;  ///< Keypoint detection threshold.
  int maxNoKeypoints;       ///< Restrict to a maximum of this many keypoints per image (strongest ones).

  /**
   * @brief If the hull-area around all matched keypoints of the current frame (with existing landmarks)
   *        divided by the hull-area around all keypoints in the current frame is lower than
   *        this threshold it should be a new keyframe.
   * @see   doWeNeedANewKeyframe()
   */
  float keyframeInsertionOverlapThreshold;
  /**
   * @brief If the number of matched keypoints of the current frame with an older frame
   *        divided by the amount of points inside the convex hull around all keypoints
   *        is lower than the threshold it should be a keyframe.
   * @see   doWeNeedANewKeyframe()
   */
  float keyframeInsertionMatchingRatioThreshold;

  ///< stereo matching with epipolar check and landmark fusion or
  /// the okvis stereo matching 2d-2d + 3d-2d + 3d-2d?
  bool stereoMatchWithEpipolarCheck;

  double epipolarDistanceThreshold;

  size_t numOldKeyframesToMatch;
  size_t numKeyframesToMatch;

  int numThreads;

  FrontendOptions(FeatureTrackingScheme featureTrackingMethod =
                      FeatureTrackingScheme::KeyframeDescriptorMatching,
                  BriskOptions brisk = BriskOptions(),
                  bool useMedianFilter = false, int detectionOctaves = 0,
                  double detectionThreshold = 40, int maxNoKeypoints = 400,
                  float keyframeInsertionOverlapThreshold = 0.6,
                  float keyframeInsertionMatchingRatioThreshold = 0.2,
                  bool stereoWithEpipolarCheck = true,
                  double epipolarDistanceThreshold = 2.5,
                  size_t numOldKeyframesToMatch = 2,
                  size_t numKeyframesToMatch = 3,
                  int numThreads = 4);

  std::string toString(std::string hint) const;
};

struct PointLandmarkOptions {
  int landmarkModelId;
  size_t minTrackLengthForMsckf;
  size_t maxHibernationFrames;   ///< max number of miss frames, each frame has potentially many images.
  size_t minTrackLengthForSlam;  ///< min track length of a landmark to be included in state.
  int maxInStateLandmarks;       ///< max number of landmarks in the state vector.
  int maxMarginalizedLandmarks;  ///< max number of marginalized landmarks in one update step.
  double triangulationMaxDepth;

  PointLandmarkOptions(int lmkModelId = 0, size_t minMsckfTrackLength = 3u,
                       size_t hibernationFrames = 3u, size_t minSlamTrackLength = 11u,
                       int maxInStateLandmarks = 50, int maxMarginalizedLandmarks = 50,
                       double triangulationMaxDepth = 1000);
  std::string toString(std::string hint = "") const;
};

struct PoseGraphOptions {
  int maxOdometryConstraintForAKeyframe;
  double minDistance;
  double minAngle;
  PoseGraphOptions(int maxOdometryConstraintForAKeyframe = 3,
                   double minDistance = 0.1, double minAngle = 0.1);
};

struct InputData {
  std::string imageFolder;
  std::string timeFile;
  std::string videoFile;
  std::string imuFile;
  int startIndex;
  int finishIndex;
};
}  // namespace swift_vio

#endif // INCLUDE_SWIFT_VIO_PARAMETERS_HPP_
