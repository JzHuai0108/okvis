
/**
 * @file implementation/Estimator.hpp
 * @brief Header implementation file for the Estimator class.
 * @author Jianzhu Huai
 */

#include <swift_vio/ProjectionIntrinsicReps.h>

/// \brief okvis Main namespace of this package.
namespace okvis {
template <class GEOMETRY_TYPE>
::ceres::ResidualBlockId Estimator::addPointFrameResidual(
    uint64_t landmarkId, const KeypointIdentifier& kpi) {
  // get the keypoint measurement
  okvis::MultiFramePtr multiFramePtr = multiFramePtrMap_.at(kpi.frameId);
  Eigen::Vector2d measurement;
  multiFramePtr->getKeypoint(kpi.cameraIndex, kpi.keypointIndex, measurement);
  Eigen::Matrix2d information = Eigen::Matrix2d::Identity();
  double size = 1.0;
  multiFramePtr->getKeypointSize(kpi.cameraIndex, kpi.keypointIndex, size);
  information *= 64.0 / (size * size);

  std::shared_ptr<const GEOMETRY_TYPE> cameraGeometry =
      cameraRig_.template geometryAs<GEOMETRY_TYPE>(kpi.cameraIndex);

  std::shared_ptr<ceres::ReprojectionError<GEOMETRY_TYPE>> reprojectionError(
      new ceres::ReprojectionError<GEOMETRY_TYPE>(cameraGeometry, kpi.cameraIndex,
                                                  measurement, information));

  ::ceres::ResidualBlockId retVal = mapPtr_->addResidualBlock(
      reprojectionError,
      cauchyLossFunctionPtr_ ? cauchyLossFunctionPtr_.get() : NULL,
      mapPtr_->parameterBlockPtr(kpi.frameId),
      mapPtr_->parameterBlockPtr(landmarkId),
      mapPtr_->parameterBlockPtr(statesMap_.at(kpi.frameId)
                                     .sensors.at(SensorStates::Camera)
                                     .at(kpi.cameraIndex)
                                     .at(CameraSensorStates::T_XCi)
                                     .id));
  return retVal;
}
}  // namespace okvis
