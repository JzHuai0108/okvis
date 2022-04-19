#include "swift_vio/PointSharedData.hpp"
#include <iterator>
#include <glog/logging.h>

#include <okvis/ceres/HomogeneousPointLocalParameterization.hpp>

#include <swift_vio/CameraRig.hpp>
#include <swift_vio/FrameTypedefs.hpp>
#include <swift_vio/imu/ImuRig.hpp>
#include <swift_vio/imu/ImuOdometry.h>
#include <swift_vio/PointLandmarkModels.hpp>

namespace swift_vio {
void PointSharedData::computePoseAndVelocityAtObservation() {
  CHECK(status_ >= PointSharedDataState::ImuInfoReady)
      << "Set IMU data, params, camera time params before calling this method.";
  int imuModelId = ImuModelNameToId(imuParameters_->model_name);
  CHECK(imuModelId != Imu_BG_BA_TG_TS_TA::kModelId) << "Imu_BG_BA_TG_TS_TA deprecated!";
  Eigen::Matrix<double, -1, 1> imuAugmentedParams;
  getImuAugmentedStatesEstimate(imuAugmentedParamBlockPtrs_,
                                &imuAugmentedParams, imuModelId);
  if (0) {
    // naive approach, ignoring the rolling shutter effect and the time offset.
    for (auto& item : stateInfoForObservations_) {
      item.T_WBtij = item.T_WBj_ptr->estimate();
      item.v_WBtij = item.v_WBj_ptr->estimate();
      Eigen::Matrix<double, 6, 1> bj = item.biasPtr->estimate();
      Imu_BG_BA_MG_TS_MA iem;
      if (imuModelId == Imu_BG_BA::kModelId) {
        iem.updateParameters(bj.data());
      } else {
        iem.updateParameters(bj.data(), imuAugmentedParams.data());
      }
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
    Eigen::Vector3d sj = item.v_WBj_ptr->estimate();
    okvis::Duration featureTime(normalizedFeatureTime(item));
    okvis::ImuMeasurement interpolatedInertialData;
    Imu_BG_BA_MG_TS_MA iem;
    if (imuModelId == Imu_BG_BA::kModelId) {
      iem.updateParameters(item.biasPtr->parameters());
    } else {
      iem.updateParameters(item.biasPtr->parameters(), imuAugmentedParams.data());
    }
    poseAndVelocityAtObservation(*item.imuMeasurementPtr, iem, *imuParameters_,
                                 item.stateEpoch, featureTime, &T_WB, &sj,
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
  int imuModelId = ImuModelNameToId(imuParameters_->model_name);
  getImuAugmentedStatesEstimate(
      imuAugmentedParamBlockPtrs_, &imuAugmentedParams,
      imuModelId);
  for (auto& item : stateInfoForObservations_) {
    okvis::kinematics::Transformation T_WB_lin = item.T_WBj_ptr->linPoint();
    Eigen::Vector3d speedLinPoint = item.v_WBj_ptr->linPoint();
    okvis::Duration featureTime(normalizedFeatureTime(item));
    Imu_BG_BA_MG_TS_MA iem;
    if (imuModelId == Imu_BG_BA::kModelId) {
      iem.updateParameters(item.biasPtr->parameters());
    } else {
      iem.updateParameters(item.biasPtr->parameters(), imuAugmentedParams.data());
    }
    poseAndLinearVelocityAtObservation(
        *item.imuMeasurementPtr, iem, *imuParameters_,
        item.stateEpoch, featureTime, &T_WB_lin, &speedLinPoint);
    item.v_WBtij_lin = speedLinPoint;
    item.T_WBtij_lin = T_WB_lin;
  }
  status_ = PointSharedDataState::NavStateForJacReady;
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

  // Also update the anchor frame index.
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

  // Also update the anchor frame indices.
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

std::vector<size_t> PointSharedData::anchorObservationIds() const {
  std::vector<size_t> anchorObservationIds;
  anchorObservationIds.reserve(2);
  for (auto identifier : anchorIds_) {
    anchorObservationIds.push_back(identifier.observationIndex_);
  }
  return anchorObservationIds;
}

std::shared_ptr<const okvis::ceres::PoseParameterBlock> PointSharedData::poseParameterBlockPtr(
    int observationIndex) const {
  return stateInfoForObservations_.at(observationIndex).T_WBj_ptr;
}

bool PointSharedData::decideAnchors(const std::vector<uint64_t>& orderedFrameIdsToUse,
                   int landmarkModelId, bool anchorInKeyframe) {
  bool anchorFound = true;
  std::vector<uint64_t> anchorFrameIds;
  anchorFrameIds.reserve(2);
  switch (landmarkModelId) {
    case swift_vio::ParallaxAngleParameterization::kModelId:
      // greedily choose the head and tail frames as main and associated anchors.
      anchorFrameIds.push_back(orderedFrameIdsToUse.front());
      [[fallthrough]];
    case swift_vio::InverseDepthParameterization::kModelId: {
      auto rit = orderedFrameIdsToUse.rbegin();
      if (anchorInKeyframe) {
        for (; rit != orderedFrameIdsToUse.rend(); ++rit) {
          uint64_t fid = *rit;
          Eigen::AlignedVector<StateInfoForOneKeypoint>::const_iterator obsIt =
              std::find_if(stateInfoForObservations_.begin(),
                           stateInfoForObservations_.end(),
                           [fid](const StateInfoForOneKeypoint &s) {
                             return s.frameId == fid;
                           });
          if (obsIt->isKeyframe) {
            break;
          }
        }
        if (rit == orderedFrameIdsToUse.rend()) {
          return false;
        }
      }
      anchorFrameIds.push_back(*rit);
    } break;
    case okvis::ceres::HomogeneousPointLocalParameterization::kModelId:
    default:
      break;
  }

  anchorIds_.clear();
  anchorIds_.reserve(anchorFrameIds.size());
  for (auto fid : anchorFrameIds) {
    Eigen::AlignedVector<StateInfoForOneKeypoint>::const_iterator anchorIter =
        std::find_if(stateInfoForObservations_.begin(), stateInfoForObservations_.end(),
                     [fid](const StateInfoForOneKeypoint& s) {
                       return s.frameId == fid;
                     });
    size_t anchorSeqId = std::distance(stateInfoForObservations_.cbegin(), anchorIter);
    anchorIds_.emplace_back(anchorIter->frameId, anchorIter->cameraId, anchorSeqId);
  }
  return anchorFound;
}

bool PointSharedData::decideAnchors(int landmarkModelId, bool anchorInKeyframe) {
  bool anchorFound = true;
  anchorIds_.clear();
  anchorIds_.reserve(2);
  switch (landmarkModelId) {
    case swift_vio::ParallaxAngleParameterization::kModelId:
      CHECK(!anchorInKeyframe) << "Enforcing keyframe anchors is not implemented for a PAP landmark!";
      // greedily choose the frames of the head and tail observations as main and associate anchors.
      anchorIds_.emplace_back(stateInfoForObservations_.front().frameId, stateInfoForObservations_.front().cameraId,
                              0u);
      [[fallthrough]];
    case swift_vio::InverseDepthParameterization::kModelId: {
      auto rit = stateInfoForObservations_.rbegin();
      if (anchorInKeyframe) {
        for (; rit != stateInfoForObservations_.rend(); ++rit) {
          if (rit->isKeyframe) {
            break;
          }
        }
        if (rit == stateInfoForObservations_.rend()) {
          return false;
        }
      }
      // find the camera of the minimal index as the anchor camera.
      auto nextrit = rit;
      ++nextrit;
      while (nextrit != stateInfoForObservations_.rend() &&
             nextrit->frameId == rit->frameId) {
        rit = nextrit;
        ++nextrit;
      }
      size_t obsIndex = stateInfoForObservations_.size() -
                     std::distance(stateInfoForObservations_.rbegin(), rit) - 1;
      anchorIds_.emplace_back(rit->frameId, rit->cameraId, obsIndex);
    } break;
    case okvis::ceres::HomogeneousPointLocalParameterization::kModelId:
    default:
      break;
  }
  return anchorFound;
}

} // namespace swift_vio
