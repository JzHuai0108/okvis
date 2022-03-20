#include "swift_vio/PointSharedData.hpp"
#include <iterator>
#include <glog/logging.h>

#include <swift_vio/CameraRig.hpp>
#include <swift_vio/FrameTypedefs.hpp>
#include <swift_vio/imu/ImuRig.hpp>
#include <swift_vio/imu/ImuOdometry.h>
#include <okvis/ceres/PoseParameterBlock.hpp>
#include <okvis/ceres/SpeedAndBiasParameterBlock.hpp>

namespace swift_vio {
void PointSharedData::computePoseAndVelocityAtObservation() {
  CHECK(status_ >= PointSharedDataState::ImuInfoReady)
      << "Set IMU data, params, camera time params before calling this method.";
  int imuModelId = ImuModelNameToId(imuParameters_->model_type);
  Eigen::Matrix<double, -1, 1> imuAugmentedParams;
  getImuAugmentedStatesEstimate(imuAugmentedParamBlockPtrs_,
                                &imuAugmentedParams, imuModelId);
  if (0) {
    // naive approach, ignoring the rolling shutter effect and the time offset.
    for (auto& item : stateInfoForObservations_) {
      std::shared_ptr<const okvis::ceres::ParameterBlock> b = item.T_WBj_ptr;
      item.T_WBtij =
          std::static_pointer_cast<const PoseParameterBlock>(b)
              ->estimate();
      item.v_WBtij =
          std::static_pointer_cast<
              const okvis::ceres::SpeedParameterBlock>(
              item.v_WBj_ptr)
              ->estimate();
      Eigen::Matrix<double, 6, 1> bj =
          std::static_pointer_cast<
              const okvis::ceres::BiasParameterBlock>(
              item.biasPtr)
              ->estimate();
      Imu_BG_BA_TG_TS_TA iem;
      iem.updateParameters(bj.data(), imuAugmentedParams.data());
      okvis::ImuMeasurement interpolatedInertialData;
      ImuOdometry::interpolateInertialData(*item.imuMeasurementPtr, iem,
                                           item.stateEpoch,
                                           interpolatedInertialData);
      item.omega_Btij = interpolatedInertialData.measurement.gyroscopes;
    }
    status_ = PointSharedDataState::NavStateReady;
    return;
  }
  for (auto& item : stateInfoForObservations_) {
    okvis::kinematics::Transformation T_WB = item.T_WBj_ptr->estimate();

    Eigen::Vector3d sj =
        std::static_pointer_cast<
            const okvis::ceres::SpeedParameterBlock>(
            item.v_WBj_ptr)
            ->estimate();
    Eigen::Matrix<double, 6, 1> bj =
        std::static_pointer_cast<
            const okvis::ceres::BiasParameterBlock>(
            item.biasPtr)->estimate();

    okvis::Duration featureTime(normalizedFeatureTime(item));
    okvis::ImuMeasurement interpolatedInertialData;
    poseAndVelocityAtObservation(*item.imuMeasurementPtr, imuAugmentedParams,
                                        *imuParameters_, item.stateEpoch,
                                        featureTime, &T_WB, &sj, &bj,
                                        &interpolatedInertialData, false);
    item.T_WBtij = T_WB;
    item.v_WBtij = sj;
    item.omega_Btij = interpolatedInertialData.measurement.gyroscopes;
  }
  status_ = PointSharedDataState::NavStateReady;
}

void PointSharedData::computePoseAndVelocityForJacobians() {
  CHECK(status_ == PointSharedDataState::NavStateReady);
  Eigen::Matrix<double, -1, 1> imuAugmentedParams;
  getImuAugmentedStatesEstimate(
      imuAugmentedParamBlockPtrs_, &imuAugmentedParams,
      ImuModelNameToId(imuParameters_->model_type));
  for (auto& item : stateInfoForObservations_) {
    okvis::kinematics::Transformation T_WB_lin = item.T_WBj_ptr->linPoint();
    Eigen::Vector3d speedLinPoint = item.v_WBj_ptr->linPoint();
    okvis::Duration featureTime(normalizedFeatureTime(item));
    poseAndLinearVelocityAtObservation(
        *item.imuMeasurementPtr, imuAugmentedParams, *imuParameters_,
        item.stateEpoch, featureTime, &T_WB_lin, &speedLinPoint, item.biasPtr->estimate());
    item.v_WBtij_lin = speedLinPoint;
    item.T_WBtij_lin = T_WB_lin;
  }

  status_ = PointSharedDataState::NavStateForJacReady;
}

void PointSharedData::computeSharedJacobians(int /*cameraObservationModelId*/) {
  CHECK(status_ == PointSharedDataState::NavStateForJacReady);
}

void PointSharedData::removeExtraObservations(
    const std::vector<uint64_t>& orderedSelectedFrameIds) {
  auto stateIter = stateInfoForObservations_.begin();
  auto keepStateIter = stateInfoForObservations_.begin();
  auto selectedFrameIter = orderedSelectedFrameIds.begin();

  for (; selectedFrameIter != orderedSelectedFrameIds.end();
       ++selectedFrameIter) {
    while (stateIter->frameId != *selectedFrameIter) {
      ++stateIter;
    }
    *keepStateIter = *stateIter;
    ++stateIter;
    ++keepStateIter;
  }
  size_t keepSize = orderedSelectedFrameIds.size();
  size_t foundSize =
      std::distance(stateInfoForObservations_.begin(), keepStateIter);
  CHECK_EQ(orderedSelectedFrameIds.size(), foundSize);
  stateInfoForObservations_.resize(keepSize);

  // Also update thAnchorFrameIdentifieror frames.
  for (std::vector<AnchorFrameIdentifier>::iterator anchorIdIter =
           anchorIds_.begin();
       anchorIdIter != anchorIds_.end(); ++anchorIdIter) {
    bool found = false;
    int index = (int)stateInfoForObservations_.size() - 1;
    for (auto riter = stateInfoForObservations_.rbegin();
         riter != stateInfoForObservations_.rend(); ++riter, --index) {
      if (riter->frameId == anchorIdIter->frameId_ &&
          riter->cameraId == anchorIdIter->cameraIndex_) {
        anchorIdIter->observationIndex_ = index;
        found = true;
        break;
      }
    }
    LOG_IF(WARNING, !found)
        << "Observation for anchor frame is not found in stateInfo list!";
  }
}

void PointSharedData::removeExtraObservations(
    const std::vector<uint64_t>& orderedSelectedFrameIds,
    std::vector<double>* imageNoise2dStdList) {
  CHECK_EQ(imageNoise2dStdList->size(), 2u * stateInfoForObservations_.size());
  auto stateIter = stateInfoForObservations_.begin();
  auto keepStateIter = stateInfoForObservations_.begin();
  auto selectedFrameIter = orderedSelectedFrameIds.begin();
  auto noiseIter = imageNoise2dStdList->begin();
  auto keepNoiseIter = imageNoise2dStdList->begin();

  for (; selectedFrameIter != orderedSelectedFrameIds.end();
       ++selectedFrameIter) {
    while (stateIter->frameId != *selectedFrameIter) {
      ++stateIter;
      noiseIter += 2;
    }
    *keepStateIter = *stateIter;
    ++stateIter;
    ++keepStateIter;

    *keepNoiseIter = *noiseIter;
    ++keepNoiseIter;
    ++noiseIter;
    *keepNoiseIter = *noiseIter;
    ++keepNoiseIter;
    ++noiseIter;
  }
  size_t keepSize = orderedSelectedFrameIds.size();
  size_t foundSize =
      std::distance(stateInfoForObservations_.begin(), keepStateIter);
  CHECK_EQ(orderedSelectedFrameIds.size(), foundSize);
  stateInfoForObservations_.resize(keepSize);
  imageNoise2dStdList->resize(keepSize * 2);

  // Also update thAnchorFrameIdentifieror frames.
  for (std::vector<AnchorFrameIdentifier>::iterator anchorIdIter =
           anchorIds_.begin();
       anchorIdIter != anchorIds_.end(); ++anchorIdIter) {
    bool found = false;
    int index = (int)stateInfoForObservations_.size() - 1;
    for (auto riter = stateInfoForObservations_.rbegin();
         riter != stateInfoForObservations_.rend(); ++riter, --index) {
      if (riter->frameId == anchorIdIter->frameId_ &&
          riter->cameraId == anchorIdIter->cameraIndex_) {
        anchorIdIter->observationIndex_ = index;
        found = true;
        break;
      }
    }
    LOG_IF(WARNING, !found)
        << "Observation for anchor frame is not found in stateInfo list!";
  }
}

void PointSharedData::removeExtraObservationsLegacy(
    const std::vector<uint64_t>& orderedSelectedFrameIds,
    std::vector<double>* imageNoise2dStdList) {
  auto itFrameIds = stateInfoForObservations_.begin();
  auto itRoi = imageNoise2dStdList->begin();
  size_t numPoses = stateInfoForObservations_.size();
  for (size_t poseIndex = 0u; poseIndex < numPoses; ++poseIndex) {
    uint64_t poseId = itFrameIds->frameId;
    if (std::find(orderedSelectedFrameIds.begin(),
                  orderedSelectedFrameIds.end(),
                  poseId) == orderedSelectedFrameIds.end()) {
      itFrameIds = stateInfoForObservations_.erase(itFrameIds);
      itRoi = imageNoise2dStdList->erase(itRoi);
      itRoi = imageNoise2dStdList->erase(itRoi);
      continue;
    } else {
      ++itFrameIds;
      ++itRoi;
      ++itRoi;
    }
  }
}

std::vector<int> PointSharedData::anchorObservationIds() const {
  std::vector<int> anchorObservationIds;
  anchorObservationIds.reserve(2);
  for (auto identifier : anchorIds_) {
    int index = 0;
    for (auto& stateInfo : stateInfoForObservations_) {
      if (identifier.frameId_ == stateInfo.frameId) {
        anchorObservationIds.push_back(index);
        break;
      }
      ++index;
    }
  }
  return anchorObservationIds;
}

std::shared_ptr<const PoseParameterBlock> PointSharedData::poseParameterBlockPtr(
    int observationIndex) const {
  return stateInfoForObservations_.at(observationIndex).T_WBj_ptr;
}
} // namespace swift_vio
