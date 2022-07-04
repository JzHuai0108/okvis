#ifndef INCLUDE_SWIFT_VIO_POINT_LANDMARK_SIMULATION_HPP
#define INCLUDE_SWIFT_VIO_POINT_LANDMARK_SIMULATION_HPP

#include <random>

#include <okvis/MultiFrame.hpp>
#include <okvis/cameras/NCameraSystem.hpp>
#include <okvis/kinematics/Transformation.hpp>

namespace simul {
class PointLandmarkSimulation
{
 public:
  /**
   * @brief projectLandmarksToNFrame
   * @param homogeneousPoints
   * @param T_WS_ref
   * @param cameraSystemRef
   * @param[in, out] nframes the keypoints for every frame are created from
   * observations of successfully projected landmarks.
   * @param frameLandmarkIndices {{landmark index of every keypoint} in every
   * frame}, every entry >= 0
   * @param keypointIndices {map from landmark index to keypoint index in every frame}
   * @param imageNoiseMag
   */
  template <typename CameraSystemT, typename MultiFrameT>
  static void projectLandmarksToNFrame(
      const std::vector<Eigen::Vector4d,
                        Eigen::aligned_allocator<Eigen::Vector4d>>&
          homogeneousPoints,
      okvis::kinematics::Transformation& T_WS_ref,
      const CameraSystemT &cameraSystemRef,
      std::shared_ptr<MultiFrameT> nframes,
      std::vector<std::vector<size_t>>* frameLandmarkIndices,
      std::vector<std::unordered_map<size_t, size_t>>* keypointIndices,
      const double* imageNoiseMag) {
    size_t numFrames = nframes->numFrames();
    // project landmarks onto frames of nframes
    for (size_t i = 0; i < numFrames; ++i) {
      std::vector<size_t> lmk_indices;
      std::vector<cv::KeyPoint> keypoints;
      std::unordered_map<size_t, size_t> frameKeypointIndices;
      for (size_t j = 0; j < homogeneousPoints.size(); ++j) {
        Eigen::Vector2d projection;
        Eigen::Vector4d point_C = cameraSystemRef.T_SC(i)->inverse() *
                                  T_WS_ref.inverse() * homogeneousPoints[j];
        okvis::cameras::CameraBase::ProjectionStatus status =
            cameraSystemRef.cameraGeometry(i)->projectHomogeneous(point_C,
                                                                  &projection);
        if (status ==
            okvis::cameras::CameraBase::ProjectionStatus::Successful) {
          Eigen::Vector2d measurement(projection);
          if (imageNoiseMag) {
            std::random_device rd{};
            std::mt19937 gen{rd()};
            std::normal_distribution<> d{0, *imageNoiseMag};
            measurement[0] += d(gen);
            measurement[1] += d(gen);
          }
          frameKeypointIndices[j] = keypoints.size();
          keypoints.emplace_back(measurement[0], measurement[1], 8.0);
          lmk_indices.emplace_back(j);
        }
      }
      nframes->resetKeypoints(i, keypoints);
      if (frameLandmarkIndices) {
        frameLandmarkIndices->emplace_back(lmk_indices);
      }
      if (keypointIndices) {
        keypointIndices->emplace_back(frameKeypointIndices);
      }
    }
  }
};
}  // namespace simul

#endif // INCLUDE_SWIFT_VIO_POINT_LANDMARK_SIMULATION_HPP
