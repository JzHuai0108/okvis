#include "swift_vio/PointLandmarkSimulation.hpp"

#include <random>
#include <Eigen/Core>

namespace simul {
void PointLandmarkSimulation::projectLandmarksToNFrame(
    const std::vector<Eigen::Vector4d,
                      Eigen::aligned_allocator<Eigen::Vector4d>>&
        homogeneousPoints,
    okvis::kinematics::Transformation& T_WS_ref,
    const okvis::cameras::NCameraSystem &cameraSystemRef,
    std::shared_ptr<okvis::MultiFrame> nframes,
    std::vector<std::vector<size_t>>* frameLandmarkIndices,
    std::vector<std::vector<int>>* keypointIndices,
    const double* imageNoiseMag) {
  size_t numFrames = nframes->numFrames();
  std::vector<std::vector<cv::KeyPoint>> frame_keypoints;
  // project landmarks onto frames of nframes
  for (size_t i = 0; i < numFrames; ++i) {
    std::vector<size_t> lmk_indices;
    std::vector<cv::KeyPoint> keypoints;
    std::vector<int> frameKeypointIndices(homogeneousPoints.size(), -1);
    for (size_t j = 0; j < homogeneousPoints.size(); ++j) {
      Eigen::Vector2d projection;
      Eigen::Vector4d point_C = cameraSystemRef.T_SC(i)->inverse() *
                                T_WS_ref.inverse() * homogeneousPoints[j];
      okvis::cameras::CameraBase::ProjectionStatus status =
          cameraSystemRef.cameraGeometry(i)->projectHomogeneous(point_C,
                                                                 &projection);
      if (status == okvis::cameras::CameraBase::ProjectionStatus::Successful) {
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
    frameLandmarkIndices->emplace_back(lmk_indices);
    frame_keypoints.emplace_back(keypoints);
    nframes->resetKeypoints(i, keypoints);
    keypointIndices->emplace_back(frameKeypointIndices);
  }
}
}  // namespace simul
