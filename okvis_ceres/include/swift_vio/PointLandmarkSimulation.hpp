#ifndef INCLUDE_SWIFT_VIO_POINT_LANDMARK_SIMULATION_HPP
#define INCLUDE_SWIFT_VIO_POINT_LANDMARK_SIMULATION_HPP

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
   * @param keypointIndices {{every landmark's keypoint index} in every frame},
   * valid entry >= 0, void entry -1
   * @param imageNoiseMag
   */
  static void projectLandmarksToNFrame(
      const std::vector<Eigen::Vector4d,
                        Eigen::aligned_allocator<Eigen::Vector4d>>&
          homogeneousPoints,
      okvis::kinematics::Transformation& T_WS_ref,
      const okvis::cameras::NCameraSystem &cameraSystemRef,
      std::shared_ptr<okvis::MultiFrame> nframes,
      std::vector<std::vector<size_t>>* frameLandmarkIndices,
      std::vector<std::vector<int>>* keypointIndices,
      const double* imageNoiseMag);
};
}  // namespace simul

#endif // INCLUDE_SWIFT_VIO_POINT_LANDMARK_SIMULATION_HPP
