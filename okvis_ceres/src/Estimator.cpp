/*********************************************************************************
 *  OKVIS - Open Keyframe-based Visual-Inertial SLAM
 *  Copyright (c) 2015, Autonomous Systems Lab / ETH Zurich
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 * 
 *   * Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *   * Neither the name of Autonomous Systems Lab / ETH Zurich nor the names of
 *     its contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 *  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 *  Created on: Dec 30, 2014
 *      Author: Stefan Leutenegger (s.leutenegger@imperial.ac.uk)
 *    Modified: Andreas Forster (an.forster@gmail.com)
 *********************************************************************************/

/**
 * @file Estimator.cpp
 * @brief Source file for the Estimator class.
 * @author Stefan Leutenegger
 * @author Andreas Forster
 */

#include <glog/logging.h>
#include <okvis/Estimator.hpp>
#include <okvis/CameraModelSwitch.hpp>
#include <okvis/ceres/PoseParameterBlock.hpp>
#include <okvis/ceres/ImuError.hpp>
#include <okvis/ceres/PoseError.hpp>
#include <okvis/ceres/RelativePoseError.hpp>
#include <okvis/ceres/SpeedAndBiasError.hpp>
#include <okvis/IdProvider.hpp>
#include <okvis/MultiFrame.hpp>
#include <okvis/assert_macros.hpp>

#include <swift_vio/ceres/CameraTimeParamBlock.hpp>
#include <swift_vio/ceres/EuclideanParamBlock.hpp>
#include <swift_vio/ceres/EuclideanParamBlockSized.hpp>
#include <swift_vio/ExtrinsicReps.hpp>
#include <swift_vio/IoUtil.hpp>
#include <swift_vio/VectorOperations.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {

// Constructor if a ceres map is already available.
Estimator::Estimator(
    std::shared_ptr<okvis::ceres::Map> mapPtr)
    : EstimatorBase(mapPtr),
      cauchyLossFunctionPtr_(new ::ceres::CauchyLoss(1)),
      huberLossFunctionPtr_(new ::ceres::HuberLoss(1)),
      marginalizationResidualId_(0)
{
}

// The default constructor.
Estimator::Estimator()
    : EstimatorBase(),
      cauchyLossFunctionPtr_(new ::ceres::CauchyLoss(1)),
      huberLossFunctionPtr_(new ::ceres::HuberLoss(1)),
      marginalizationResidualId_(0)
{
}

Estimator::~Estimator()
{
}

// Add a pose to the state.
bool Estimator::addStates(
    okvis::MultiFramePtr multiFrame,
    const okvis::ImuMeasurementDeque & imuMeasurements,
    bool asKeyframe)
{
  // note: this is before matching...
  // TODO !!
  okvis::kinematics::Transformation T_WS;
  okvis::SpeedAndBias speedAndBias;
  if (statesMap_.empty()) {
    // in case this is the first frame ever, let's initialize the pose:
    if (initialNavState_.initializeToCustomPose)
      T_WS = okvis::kinematics::Transformation(initialNavState_.p_WS, initialNavState_.q_WS);
    else {
      bool success0 = swift_vio::initPoseFromImu(imuMeasurements, multiFrame->timestamp(), T_WS);
      OKVIS_ASSERT_TRUE_DBG(
          Exception, success0,
          "pose could not be initialized from imu measurements.");
      if (!success0) return false;
      initialNavState_.updatePose(T_WS, multiFrame->timestamp());
    }
    speedAndBias.setZero();
    speedAndBias.head<3>() = initialNavState_.v_WS;
    speedAndBias.segment<3>(3) = imuParametersVec_.at(0)->initialGyroBias();
    speedAndBias.tail<3>() = imuParametersVec_.at(0)->initialAccelBias();
  } else {
    // get the previous states
    uint64_t T_WS_id = statesMap_.rbegin()->second.id;
    uint64_t speedAndBias_id = statesMap_.rbegin()->second.sensors.at(SensorStates::Imu)
        .at(0).at(ImuSensorStates::SpeedAndBias).id;
    OKVIS_ASSERT_TRUE_DBG(Exception, mapPtr_->parameterBlockExists(T_WS_id),
                       "this is an okvis bug. previous pose does not exist.");
    T_WS = std::static_pointer_cast<ceres::PoseParameterBlock>(
        mapPtr_->parameterBlockPtr(T_WS_id))->estimate();
    //OKVIS_ASSERT_TRUE_DBG(
    //    Exception, speedAndBias_id,
    //    "this is an okvis bug. previous speedAndBias does not exist.");
    speedAndBias =
        std::static_pointer_cast<ceres::SpeedAndBiasParameterBlock>(
            mapPtr_->parameterBlockPtr(speedAndBias_id))->estimate();

    // propagate pose and speedAndBias
    int numUsedImuMeasurements = ceres::ImuError::propagation(
        imuMeasurements, *imuParametersVec_.at(0), T_WS, speedAndBias,
        statesMap_.rbegin()->second.timestamp, multiFrame->timestamp());
    OKVIS_ASSERT_TRUE_DBG(Exception, numUsedImuMeasurements > 1,
                       "propagation failed");
    if (numUsedImuMeasurements < 1){
      LOG(INFO) << "numUsedImuMeasurements=" << numUsedImuMeasurements;
      return false;
    }
  }


  // create a states object:
  States states(asKeyframe, multiFrame->id(), multiFrame->timestamp());

  // check if id was used before
  OKVIS_ASSERT_TRUE_DBG(Exception,
      statesMap_.find(states.id)==statesMap_.end(),
      "pose ID" <<states.id<<" was used before!");

  // create global states
  std::shared_ptr<okvis::ceres::PoseParameterBlock> poseParameterBlock(
      new okvis::ceres::PoseParameterBlock(T_WS, states.id,
                                           multiFrame->timestamp()));
  states.global.at(GlobalStates::T_WS).exists = true;
  states.global.at(GlobalStates::T_WS).id = states.id;

  if(statesMap_.empty())
  {
    referencePoseId_ = states.id; // set this as reference pose
    if (!mapPtr_->addParameterBlock(poseParameterBlock,ceres::Map::Pose6d)) {
      return false;
    }
  } else {
    if (!mapPtr_->addParameterBlock(poseParameterBlock,ceres::Map::Pose6d)) {
      return false;
    }
  }

  // add to buffer
  statesMap_.insert(std::pair<uint64_t, States>(states.id, states));
  multiFramePtrMap_.insert(std::pair<uint64_t, okvis::MultiFramePtr>(states.id, multiFrame));

  // the following will point to the last states:
  std::map<uint64_t, States>::reverse_iterator lastElementIterator = statesMap_.rbegin();
  lastElementIterator++;

  // initialize new sensor states
  // cameras:
  for (size_t i = 0; i < cameraNoiseParametersVec_.size(); ++i) {

    SpecificSensorStatesContainer cameraInfos(2);
    cameraInfos.at(CameraSensorStates::T_XCi).exists=true;
    cameraInfos.at(CameraSensorStates::Intrinsics).exists=false;
    if(((cameraNoiseParametersVec_.at(i).sigma_c_relative_translation<1e-12)||
        (cameraNoiseParametersVec_.at(i).sigma_c_relative_orientation<1e-12))&&
        (statesMap_.size() > 1)){
      // use the same block...
      cameraInfos.at(CameraSensorStates::T_XCi).id =
          lastElementIterator->second.sensors.at(SensorStates::Camera).at(i).at(CameraSensorStates::T_XCi).id;
    } else {
      const okvis::kinematics::Transformation T_SC = *multiFrame->T_SC(i);
      uint64_t id = IdProvider::instance().newId();
      std::shared_ptr<okvis::ceres::PoseParameterBlock> extrinsicsParameterBlockPtr(
          new okvis::ceres::PoseParameterBlock(T_SC, id,
                                               multiFrame->timestamp()));
      if(!mapPtr_->addParameterBlock(extrinsicsParameterBlockPtr,ceres::Map::Pose6d)){
        return false;
      }
      cameraInfos.at(CameraSensorStates::T_XCi).id = id;
    }
    // update the states info
    statesMap_.rbegin()->second.sensors.at(SensorStates::Camera).push_back(cameraInfos);
    states.sensors.at(SensorStates::Camera).push_back(cameraInfos);
  }

  // IMU states are automatically propagated.
  for (size_t i=0; i<imuParametersVec_.size(); ++i){
    SpecificSensorStatesContainer imuInfo(2);
    imuInfo.at(ImuSensorStates::SpeedAndBias).exists = true;
    uint64_t id = IdProvider::instance().newId();
    std::shared_ptr<okvis::ceres::SpeedAndBiasParameterBlock> speedAndBiasParameterBlock(
        new okvis::ceres::SpeedAndBiasParameterBlock(speedAndBias, id, multiFrame->timestamp()));

    if(!mapPtr_->addParameterBlock(speedAndBiasParameterBlock)){
      return false;
    }
    imuInfo.at(ImuSensorStates::SpeedAndBias).id = id;
    statesMap_.rbegin()->second.sensors.at(SensorStates::Imu).push_back(imuInfo);
    states.sensors.at(SensorStates::Imu).push_back(imuInfo);
  }

  addPriorAndRelativeTerms(imuMeasurements);
  return true;
}

void Estimator::addPriorAndRelativeTerms(const okvis::ImuMeasurementDeque &imuMeasurements) {
  std::map<uint64_t, States>::reverse_iterator lastElementIterator = statesMap_.rbegin();
  const States &states = lastElementIterator->second;
  lastElementIterator++;
  uint64_t id = states.global.at(GlobalStates::T_WS).id;
  std::shared_ptr<okvis::ceres::PoseParameterBlock> poseParameterBlock =
      std::static_pointer_cast<okvis::ceres::PoseParameterBlock>(
          mapPtr_->parameterBlockPtr(id));
  kinematics::Transformation T_WS = poseParameterBlock->estimate();

  uint64_t sbid = states.sensors.at(SensorStates::Imu).at(0).at(ImuSensorStates::SpeedAndBias).id;
  std::shared_ptr<okvis::ceres::SpeedAndBiasParameterBlock> speedAndBiasParameterBlock =
      std::static_pointer_cast<okvis::ceres::SpeedAndBiasParameterBlock>(
          mapPtr_->parameterBlockPtr(sbid));
  SpeedAndBiases speedAndBias = speedAndBiasParameterBlock->estimate();

  // depending on whether or not this is the very beginning, we will add priors or relative terms to the last state:
  if (statesMap_.size() == 1) {
    // let's add a prior
    Eigen::Matrix<double, 6, 6> information;
    initialNavState_.toInformation(&information);
    std::shared_ptr<ceres::PoseError > poseError(new ceres::PoseError(T_WS, information));
    /*auto id2= */ mapPtr_->addResidualBlock(poseError,NULL,poseParameterBlock);
    //mapPtr_->isJacobianCorrect(id2,1.0e-6);

    // sensor states
    for (size_t i = 0; i < cameraNoiseParametersVec_.size(); ++i) {
      double translationStdev = cameraNoiseParametersVec_.at(i).sigma_absolute_translation;
      double translationVariance = translationStdev*translationStdev;
      double rotationStdev = cameraNoiseParametersVec_.at(i).sigma_absolute_orientation;
      double rotationVariance = rotationStdev*rotationStdev;
      if(translationVariance>1.0e-16 && rotationVariance>1.0e-16){
        const okvis::kinematics::Transformation T_SC = cameraRig_.getCameraExtrinsic(i);
        std::shared_ptr<ceres::PoseError > cameraPoseError(
              new ceres::PoseError(T_SC, translationVariance, rotationVariance));
        // add to map
        mapPtr_->addResidualBlock(
            cameraPoseError,
            NULL,
            mapPtr_->parameterBlockPtr(
                states.sensors.at(SensorStates::Camera).at(i).at(CameraSensorStates::T_XCi).id));
        //mapPtr_->isJacobianCorrect(id,1.0e-6);
      }
      else {
        mapPtr_->setParameterBlockConstant(
            states.sensors.at(SensorStates::Camera).at(i).at(CameraSensorStates::T_XCi).id);
      }
    }
    for (size_t i = 0; i < imuParametersVec_.size(); ++i) {
      Eigen::Matrix<double,6,1> variances;
      // get these from parameter file
      const double sigma_bg = imuParametersVec_.at(0)->sigma_bg;
      const double sigma_ba = imuParametersVec_.at(0)->sigma_ba;
      std::shared_ptr<ceres::SpeedAndBiasError > speedAndBiasError(
            new ceres::SpeedAndBiasError(
                speedAndBias, initialNavState_.std_v_WS[0]*initialNavState_.std_v_WS[0],
                sigma_bg*sigma_bg, sigma_ba*sigma_ba));
      // add to map
      mapPtr_->addResidualBlock(
          speedAndBiasError,
          NULL,
          mapPtr_->parameterBlockPtr(
              states.sensors.at(SensorStates::Imu).at(i).at(ImuSensorStates::SpeedAndBias).id));
      //mapPtr_->isJacobianCorrect(id,1.0e-6);
    }
  } else {
    // add IMU error terms
    for (size_t i = 0; i < imuParametersVec_.size(); ++i) {
      std::shared_ptr<ceres::ImuError> imuError(
          new ceres::ImuError(imuMeasurements, *imuParametersVec_.at(i),
                              lastElementIterator->second.timestamp,
                              states.timestamp));
      /*::ceres::ResidualBlockId id = */mapPtr_->addResidualBlock(
          imuError,
          NULL,
          mapPtr_->parameterBlockPtr(lastElementIterator->second.id),
          mapPtr_->parameterBlockPtr(
              lastElementIterator->second.sensors.at(SensorStates::Imu).at(i).at(
                  ImuSensorStates::SpeedAndBias).id),
          mapPtr_->parameterBlockPtr(states.id),
          mapPtr_->parameterBlockPtr(
              states.sensors.at(SensorStates::Imu).at(i).at(
                  ImuSensorStates::SpeedAndBias).id));
      //imuError->setRecomputeInformation(false);
      //mapPtr_->isJacobianCorrect(id,1.0e-9);
      //imuError->setRecomputeInformation(true);
    }

    // add relative sensor state errors
    for (size_t i = 0; i < cameraNoiseParametersVec_.size(); ++i) {
      if(lastElementIterator->second.sensors.at(SensorStates::Camera).at(i).at(CameraSensorStates::T_XCi).id !=
          states.sensors.at(SensorStates::Camera).at(i).at(CameraSensorStates::T_XCi).id){
        // i.e. they are different estimated variables, so link them with a temporal error term
        double dt = (states.timestamp - lastElementIterator->second.timestamp)
            .toSec();
        double translationSigmaC = cameraNoiseParametersVec_.at(i)
            .sigma_c_relative_translation;
        double translationVariance = translationSigmaC * translationSigmaC * dt;
        double rotationSigmaC = cameraNoiseParametersVec_.at(i)
            .sigma_c_relative_orientation;
        double rotationVariance = rotationSigmaC * rotationSigmaC * dt;
        std::shared_ptr<ceres::RelativePoseError> relativeExtrinsicsError(
            new ceres::RelativePoseError(translationVariance,
                                         rotationVariance));
        mapPtr_->addResidualBlock(
            relativeExtrinsicsError,
            NULL,
            mapPtr_->parameterBlockPtr(
                lastElementIterator->second.sensors.at(SensorStates::Camera).at(
                    i).at(CameraSensorStates::T_XCi).id),
            mapPtr_->parameterBlockPtr(
                states.sensors.at(SensorStates::Camera).at(i).at(
                    CameraSensorStates::T_XCi).id));
        //mapPtr_->isJacobianCorrect(id,1.0e-6);
      }
    }
    // only camera. this is slightly inconsistent, since the IMU error term contains both
    // a term for global states as well as for the sensor-internal ones (i.e. biases).
    // TODO: magnetometer, pressure, ...
  }
}

// Remove an observation from a landmark.
bool Estimator::removeObservationAndResidual(::ceres::ResidualBlockId residualBlockId) {
  const ceres::Map::ParameterBlockCollection parameters = mapPtr_->parameters(residualBlockId);
  const uint64_t landmarkId = parameters.at(1).first;
  // remove in landmarksMap
  MapPoint& mapPoint = landmarksMap_.at(landmarkId);
  for(std::map<okvis::KeypointIdentifier, uint64_t >::iterator it = mapPoint.observations.begin();
      it!= mapPoint.observations.end(); ){
    if(it->second == uint64_t(residualBlockId)){

      it = mapPoint.observations.erase(it);
      break;
    } else {
      it++;
    }
  }
  // remove residual block
  mapPtr_->removeResidualBlock(residualBlockId);
  return true;
}

// Applies the dropping/marginalization strategy according to the RSS'13/IJRR'14 paper.
// The new number of frames in the window will be numKeyframes+numImuFrames.
bool Estimator::applyMarginalizationStrategy(okvis::MapPointVector& removedLandmarks) {
  size_t numKeyframes = estimatorOptions_.numKeyframes;
  size_t numImuFrames = estimatorOptions_.numImuFrames;
  // keep the newest numImuFrames
  std::map<uint64_t, States>::reverse_iterator rit = statesMap_.rbegin();
  for(size_t k=0; k<numImuFrames; k++){
    rit++;
    if(rit==statesMap_.rend()){
      // nothing to do.
      return true;
    }
  }

  // remove linear marginalizationError, if existing
  if (marginalizationErrorPtr_ && marginalizationResidualId_) {
    bool success = mapPtr_->removeResidualBlock(marginalizationResidualId_);
    OKVIS_ASSERT_TRUE_DBG(Exception, success,
                       "could not remove marginalization error");
    marginalizationResidualId_ = 0;
    if (!success)
      return false;
  }

  // these will keep track of what we want to marginalize out.
  std::vector<uint64_t> paremeterBlocksToBeMarginalized;
  std::vector<bool> keepParameterBlocks;

  if (!marginalizationErrorPtr_) {
    marginalizationErrorPtr_.reset(
        new ceres::MarginalizationError(*mapPtr_.get()));
  }

  // distinguish if we marginalize everything or everything but pose
  std::vector<uint64_t> removeFrames;
  std::vector<uint64_t> removeAllButPose;
  std::vector<uint64_t> allLinearizedFrames;
  size_t countedKeyframes = 0;
  while (rit != statesMap_.rend()) {
    if (!rit->second.isKeyframe || countedKeyframes >= numKeyframes) {
      removeFrames.push_back(rit->second.id);
    } else {
      countedKeyframes++;
    }
    removeAllButPose.push_back(rit->second.id);
    allLinearizedFrames.push_back(rit->second.id);
    ++rit;// check the next frame
  }

  // marginalize everything but pose:
  for(size_t k = 0; k<removeAllButPose.size(); ++k){
    std::map<uint64_t, States>::iterator it = statesMap_.find(removeAllButPose[k]);
    for (size_t i = 0; i < it->second.global.size(); ++i) {
      if (i == GlobalStates::T_WS) {
        continue; // we do not remove the pose here.
      }
      if (!it->second.global[i].exists) {
        continue; // if it doesn't exist, we don't do anything.
      }
      if (mapPtr_->parameterBlockPtr(it->second.global[i].id)->fixed()) {
        continue;  // we never eliminate fixed blocks.
      }
      std::map<uint64_t, States>::iterator checkit = it;
      checkit++;
      // only get rid of it, if it's different
      if(checkit->second.global[i].exists &&
          checkit->second.global[i].id == it->second.global[i].id){
        continue;
      }
      it->second.global[i].exists = false; // remember we removed
      paremeterBlocksToBeMarginalized.push_back(it->second.global[i].id);
      keepParameterBlocks.push_back(false);
      ceres::Map::ResidualBlockCollection residuals = mapPtr_->residuals(
          it->second.global[i].id);
      for (size_t r = 0; r < residuals.size(); ++r) {
        std::shared_ptr<ceres::ReprojectionErrorBase> reprojectionError =
            std::dynamic_pointer_cast<ceres::ReprojectionErrorBase>(
            residuals[r].errorInterfacePtr);
        if(!reprojectionError){   // we make sure no reprojection errors are yet included.
          marginalizationErrorPtr_->addResidualBlock(residuals[r].residualBlockId);
        }
      }
    }
    // add all error terms of the sensor states.
    for (size_t i = 0; i < it->second.sensors.size(); ++i) {
      for (size_t j = 0; j < it->second.sensors[i].size(); ++j) {
        for (size_t k = 0; k < it->second.sensors[i][j].size(); ++k) {
          if (i == SensorStates::Camera && k == CameraSensorStates::T_XCi) {
            continue; // we do not remove the extrinsics pose here.
          }
          if (!it->second.sensors[i][j][k].exists) {
            continue;
          }
          if (mapPtr_->parameterBlockPtr(it->second.sensors[i][j][k].id)
              ->fixed()) {
            continue;  // we never eliminate fixed blocks.
          }
          std::map<uint64_t, States>::iterator checkit = it;
          checkit++;
          // only get rid of it, if it's different
          if(checkit->second.sensors[i][j][k].exists &&
              checkit->second.sensors[i][j][k].id == it->second.sensors[i][j][k].id){
            continue;
          }
          it->second.sensors[i][j][k].exists = false; // remember we removed
          paremeterBlocksToBeMarginalized.push_back(it->second.sensors[i][j][k].id);
          keepParameterBlocks.push_back(false);
          ceres::Map::ResidualBlockCollection residuals = mapPtr_->residuals(
              it->second.sensors[i][j][k].id);
          for (size_t r = 0; r < residuals.size(); ++r) {
            std::shared_ptr<ceres::ReprojectionErrorBase> reprojectionError =
                std::dynamic_pointer_cast<ceres::ReprojectionErrorBase>(
                residuals[r].errorInterfacePtr);
            if(!reprojectionError){   // we make sure no reprojection errors are yet included.
              marginalizationErrorPtr_->addResidualBlock(residuals[r].residualBlockId);
            }
          }
        }
      }
    }
  }
  // marginalize ONLY pose now:
  bool reDoFixation = false;
  for(size_t k = 0; k<removeFrames.size(); ++k){
    std::map<uint64_t, States>::iterator it = statesMap_.find(removeFrames[k]);

    // schedule removal - but always keep the very first frame.
    //if(it != statesMap_.begin()){
    if(true){ /////DEBUG
      it->second.global[GlobalStates::T_WS].exists = false; // remember we removed
      paremeterBlocksToBeMarginalized.push_back(it->second.global[GlobalStates::T_WS].id);
      keepParameterBlocks.push_back(false);
    }

    // add remaing error terms
    ceres::Map::ResidualBlockCollection residuals = mapPtr_->residuals(
        it->second.global[GlobalStates::T_WS].id);

    for (size_t r = 0; r < residuals.size(); ++r) {
      // jhuai: redo fixation leads to inconsistent covariance.
      if(!estimatorOptions_.computeOkvisNees && std::dynamic_pointer_cast<ceres::PoseError>(
           residuals[r].errorInterfacePtr)){ // avoids linearising initial pose error
        mapPtr_->removeResidualBlock(residuals[r].residualBlockId);
        reDoFixation = true;
        continue;
      }
      std::shared_ptr<ceres::ReprojectionErrorBase> reprojectionError =
          std::dynamic_pointer_cast<ceres::ReprojectionErrorBase>(
          residuals[r].errorInterfacePtr);
      if(!reprojectionError){   // we make sure no reprojection errors are yet included.
        marginalizationErrorPtr_->addResidualBlock(residuals[r].residualBlockId);
      }
    }

    // add remaining error terms of the sensor states.
    size_t i = SensorStates::Camera;
    for (size_t j = 0; j < it->second.sensors[i].size(); ++j) {
      size_t k = CameraSensorStates::T_XCi;
      if (!it->second.sensors[i][j][k].exists) {
        continue;
      }
      if (mapPtr_->parameterBlockPtr(it->second.sensors[i][j][k].id)
          ->fixed()) {
        continue;  // we never eliminate fixed blocks.
      }
      std::map<uint64_t, States>::iterator checkit = it;
      checkit++;
      // only get rid of it, if it's different
      if(checkit->second.sensors[i][j][k].exists &&
          checkit->second.sensors[i][j][k].id == it->second.sensors[i][j][k].id){
        continue;
      }
      it->second.sensors[i][j][k].exists = false; // remember we removed
      paremeterBlocksToBeMarginalized.push_back(it->second.sensors[i][j][k].id);
      keepParameterBlocks.push_back(false);
      ceres::Map::ResidualBlockCollection residuals = mapPtr_->residuals(
          it->second.sensors[i][j][k].id);
      for (size_t r = 0; r < residuals.size(); ++r) {
        std::shared_ptr<ceres::ReprojectionErrorBase> reprojectionError =
            std::dynamic_pointer_cast<ceres::ReprojectionErrorBase>(
            residuals[r].errorInterfacePtr);
        if(!reprojectionError){   // we make sure no reprojection errors are yet included.
          marginalizationErrorPtr_->addResidualBlock(residuals[r].residualBlockId);
        }
      }
    }

    // now finally we treat all the observations.
    OKVIS_ASSERT_TRUE_DBG(Exception, allLinearizedFrames.size()>0, "bug");
    uint64_t currentKfId = allLinearizedFrames.at(0);

    {
      for(PointMap::iterator pit = landmarksMap_.begin();
          pit != landmarksMap_.end(); ){

        ceres::Map::ResidualBlockCollection residuals = mapPtr_->residuals(pit->first);

        // first check if we can skip
        bool skipLandmark = true;
        bool hasNewObservations = false;
        bool justDelete = false;
        bool marginalize = true;
        bool errorTermAdded = false;
        std::map<uint64_t,bool> visibleInFrame;
        size_t obsCount = 0;
        for (size_t r = 0; r < residuals.size(); ++r) {
          std::shared_ptr<ceres::ReprojectionErrorBase> reprojectionError =
              std::dynamic_pointer_cast<ceres::ReprojectionErrorBase>(
                  residuals[r].errorInterfacePtr);
          if (reprojectionError) {
            uint64_t poseId = mapPtr_->parameters(residuals[r].residualBlockId).at(0).first;
            // since we have implemented the linearisation to account for robustification,
            // we don't kick out bad measurements here any more like
            // if(swift_vio::vectorContains(allLinearizedFrames,poseId)){ ...
            //   if (error.transpose() * error > 6.0) { ... removeObservation ... }
            // }
            if(swift_vio::vectorContains(removeFrames,poseId)){
              skipLandmark = false;
            }
            if(poseId>=currentKfId){
              marginalize = false;
              hasNewObservations = true;
            }
            if(swift_vio::vectorContains(allLinearizedFrames, poseId)){
              visibleInFrame.insert(std::pair<uint64_t,bool>(poseId,true));
              obsCount++;
            }
          }
        }

        if(residuals.size()==0){
          mapPtr_->removeParameterBlock(pit->first);
          removedLandmarks.push_back(pit->second);
          pit = landmarksMap_.erase(pit);
          continue;
        }

        if(skipLandmark) {
          pit++;
          continue;
        }

        // so, we need to consider it.
        for (size_t r = 0; r < residuals.size(); ++r) {
          std::shared_ptr<ceres::ReprojectionErrorBase> reprojectionError =
              std::dynamic_pointer_cast<ceres::ReprojectionErrorBase>(
                  residuals[r].errorInterfacePtr);
          if (reprojectionError) {
            uint64_t poseId = mapPtr_->parameters(residuals[r].residualBlockId).at(0).first;
            if((swift_vio::vectorContains(removeFrames,poseId) && hasNewObservations) ||
                (!swift_vio::vectorContains(allLinearizedFrames,poseId) && marginalize)){
              // ok, let's ignore the observation.
              removeObservationAndResidual(residuals[r].residualBlockId);
              residuals.erase(residuals.begin() + r);
              r--;
            } else if(marginalize && swift_vio::vectorContains(allLinearizedFrames,poseId)) {
              // TODO: consider only the sensible ones for marginalization
              if(obsCount<2){ //visibleInFrame.size()
                removeObservationAndResidual(residuals[r].residualBlockId);
                residuals.erase(residuals.begin() + r);
                r--;
              } else {
                // add information to be considered in marginalization later.
                errorTermAdded = true;
                marginalizationErrorPtr_->addResidualBlock(
                    residuals[r].residualBlockId, false);
              }
            }
            // check anything left
            if (residuals.size() == 0) {
              justDelete = true;
              marginalize = false;
            }
          }
        }

        if(justDelete){
          mapPtr_->removeParameterBlock(pit->first);
          removedLandmarks.push_back(pit->second);
          pit = landmarksMap_.erase(pit);
          continue;
        }
        if(marginalize&&errorTermAdded){
          paremeterBlocksToBeMarginalized.push_back(pit->first);
          keepParameterBlocks.push_back(false);
          removedLandmarks.push_back(pit->second);
          pit = landmarksMap_.erase(pit);
          continue;
        }

        pit++;
      }
    }

    // update book-keeping and go to the next frame
    //if(it != statesMap_.begin()){ // let's remember that we kept the very first pose
    if(true) { ///// DEBUG
      multiFramePtrMap_.erase(it->second.id);
      statesMap_.erase(it->second.id);
    }
  }

  // now apply the actual marginalization
  if(paremeterBlocksToBeMarginalized.size()>0){
    std::vector< ::ceres::ResidualBlockId> addedPriors;
    marginalizationErrorPtr_->marginalizeOut(paremeterBlocksToBeMarginalized, keepParameterBlocks);
  }

  // update error computation
  if(paremeterBlocksToBeMarginalized.size()>0){
    marginalizationErrorPtr_->updateErrorComputation();
  }

  // add the marginalization term again
  if(marginalizationErrorPtr_->num_residuals()==0){
    marginalizationErrorPtr_.reset();
  }
  if (marginalizationErrorPtr_) {
  std::vector<std::shared_ptr<okvis::ceres::ParameterBlock> > parameterBlockPtrs;
  marginalizationErrorPtr_->getParameterBlockPtrs(parameterBlockPtrs);
  marginalizationResidualId_ = mapPtr_->addResidualBlock(
      marginalizationErrorPtr_, NULL, parameterBlockPtrs);
  OKVIS_ASSERT_TRUE_DBG(Exception, marginalizationResidualId_,
                     "could not add marginalization error");
  if (!marginalizationResidualId_)
    return false;
  }
	
	if(reDoFixation){
	  // finally fix the first pose properly
		//mapPtr_->resetParameterization(statesMap_.begin()->first, ceres::Map::Pose3d);
		okvis::kinematics::Transformation T_WS_0;
		get_T_WS(statesMap_.begin()->first, T_WS_0);
		Eigen::Matrix<double, 6, 6> information;
		initialNavState_.toInformation(&information);
	  std::shared_ptr<ceres::PoseError > poseError(new ceres::PoseError(T_WS_0, information));
	  mapPtr_->addResidualBlock(poseError,NULL,mapPtr_->parameterBlockPtr(statesMap_.begin()->first));
	}

  return true;
}

// Prints state information to buffer.
void Estimator::printStates(uint64_t poseId, std::ostream & buffer) const {
  buffer << "GLOBAL: ";
  for(size_t i = 0; i<statesMap_.at(poseId).global.size(); ++i){
    if(statesMap_.at(poseId).global.at(i).exists) {
      uint64_t id = statesMap_.at(poseId).global.at(i).id;
      if(mapPtr_->parameterBlockPtr(id)->fixed())
        buffer << "(";
      buffer << "id="<<id<<":";
      buffer << mapPtr_->parameterBlockPtr(id)->typeInfo();
      if(mapPtr_->parameterBlockPtr(id)->fixed())
        buffer << ")";
      buffer <<", ";
    }
  }
  buffer << "SENSOR: ";
  for(size_t i = 0; i<statesMap_.at(poseId).sensors.size(); ++i){
    for(size_t j = 0; j<statesMap_.at(poseId).sensors.at(i).size(); ++j){
      for(size_t k = 0; k<statesMap_.at(poseId).sensors.at(i).at(j).size(); ++k){
        if(statesMap_.at(poseId).sensors.at(i).at(j).at(k).exists) {
          uint64_t id = statesMap_.at(poseId).sensors.at(i).at(j).at(k).id;
          if(mapPtr_->parameterBlockPtr(id)->fixed())
            buffer << "(";
          buffer << "id="<<id<<":";
          buffer << mapPtr_->parameterBlockPtr(id)->typeInfo();
          if(mapPtr_->parameterBlockPtr(id)->fixed())
            buffer << ")";
          buffer <<", ";
        }
      }
    }
  }
  buffer << std::endl;
}

std::vector<std::string> Estimator::variableLabels() const {
  std::vector<std::string> varList{
      "p_WB_W_x(m)",   "p_WB_W_y(m)",   "p_WB_W_z(m)",  "q_WB_x",
      "q_WB_y",        "q_WB_z",        "q_WB_w",       "v_WB_W_x(m/s)",
      "v_WB_W_y(m/s)", "v_WB_W_z(m/s)", "b_g_x(rad/s)", "b_g_y(rad/s)",
      "b_g_z(rad/s)",  "b_a_x(m/s^2)",  "b_a_y(m/s^2)", "b_a_z(m/s^2)"};

  size_t numCameras = cameraNoiseParametersVec_.size();
  for (size_t j = 0u; j < numCameras; ++j) {
    if (!fixCameraExtrinsicParams_[j]) {
      std::vector<std::string> camExtrinsicLabels;
      swift_vio::ExtrinsicRepToDimensionLabels(
          cameraRig_.getExtrinsicRepId(j), &camExtrinsicLabels);
      varList.insert(varList.end(), camExtrinsicLabels.begin(),
                     camExtrinsicLabels.end());
    }
  }
  return varList;
}

std::vector<std::string> Estimator::perturbationLabels() const {
  return std::vector<std::string>{
      "p_WB_W_x(m)",   "p_WB_W_y(m)",  "p_WB_W_z(m)",   "theta_WB_x",
      "theta_WB_y",    "theta_WB_z",   "v_WB_W_x(m/s)", "v_WB_W_y(m/s)",
      "v_WB_W_z(m/s)", "b_g_x(rad/s)", "b_g_y(rad/s)",  "b_g_z(rad/s)",
      "b_a_x(m/s^2)",  "b_a_y(m/s^2)", "b_a_z(m/s^2)"};
}

// Start ceres optimization.
#ifdef USE_OPENMP
void Estimator::optimize(size_t numIter, size_t numThreads,
                                 bool verbose)
#else
void Estimator::optimize(size_t numIter, size_t /*numThreads*/,
                                 bool verbose) // avoid warning since numThreads unused
#warning openmp not detected, your system may be slower than expected
#endif

{
  // assemble options
  mapPtr_->options.linear_solver_type = ::ceres::SPARSE_SCHUR;
  //mapPtr_->options.initial_trust_region_radius = 1.0e4;
  //mapPtr_->options.initial_trust_region_radius = 2.0e6;
  //mapPtr_->options.preconditioner_type = ::ceres::IDENTITY;
  mapPtr_->options.trust_region_strategy_type = ::ceres::DOGLEG;
  //mapPtr_->options.trust_region_strategy_type = ::ceres::LEVENBERG_MARQUARDT;
  //mapPtr_->options.use_nonmonotonic_steps = true;
  //mapPtr_->options.max_consecutive_nonmonotonic_steps = 10;
  //mapPtr_->options.function_tolerance = 1e-12;
  //mapPtr_->options.gradient_tolerance = 1e-12;
  //mapPtr_->options.jacobi_scaling = false;
#ifdef USE_OPENMP
    mapPtr_->options.num_threads = numThreads;
#endif
  mapPtr_->options.max_num_iterations = numIter;

  if (verbose) {
    mapPtr_->options.minimizer_progress_to_stdout = true;
  } else {
    mapPtr_->options.minimizer_progress_to_stdout = false;
  }
  addReprojectionFactors();
  // call solver
  mapPtr_->solve();

  // update landmarks
  {
    for(auto it = landmarksMap_.begin(); it!=landmarksMap_.end(); ++it){
      if (it->second.inState()) {
        Eigen::MatrixXd H(3, 3);
        mapPtr_->getLhs(it->first, H);
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(H);
        Eigen::Vector3d eigenvalues = saes.eigenvalues();
        const double smallest = (eigenvalues[0]);
        const double largest = (eigenvalues[2]);
        if (smallest < 1.0e-12) {
          // this means, it has a non-observable depth
          it->second.quality = 0.0;
        } else {
          // OK, well constrained
          it->second.quality = sqrt(smallest) / sqrt(largest);
        }

        // update coordinates
        it->second.pointHomog =
            std::static_pointer_cast<
                okvis::ceres::HomogeneousPointParameterBlock>(
                mapPtr_->parameterBlockPtr(it->first))
                ->estimate();
      }
    }
  }

  updateSensorRigs();


  // summary output
  if (verbose) {
    LOG(INFO) << mapPtr_->summary.FullReport();
  }
}

// Set a time limit for the optimization process.
bool Estimator::setOptimizationTimeLimit(double timeLimit, int minIterations) {
  if(ceresCallback_ != nullptr) {
    if(timeLimit < 0.0) {
      // no time limit => set minimum iterations to maximum iterations
      ceresCallback_->setMinimumIterations(mapPtr_->options.max_num_iterations);
      return true;
    }
    ceresCallback_->setTimeLimit(timeLimit);
    ceresCallback_->setMinimumIterations(minIterations);
    return true;
  }
  else if(timeLimit >= 0.0) {
    ceresCallback_ = std::unique_ptr<okvis::ceres::CeresIterationCallback>(
          new okvis::ceres::CeresIterationCallback(timeLimit,minIterations));
    mapPtr_->options.callbacks.push_back(ceresCallback_.get());
    return true;
  }
  // no callback yet registered with ceres.
  // but given time limit is lower than 0, so no callback needed
  return true;
}

bool Estimator::addReprojectionFactors() {
  okvis::cameras::NCameraSystem::DistortionType distortionType =
      cameraRig_.distortionType(0);

  for (PointMap::iterator pit = landmarksMap_.begin();
       pit != landmarksMap_.end(); ++pit) {
    if (!pit->second.inState()) {
      std::shared_ptr<okvis::ceres::HomogeneousPointParameterBlock>
          pointParameterBlock(new okvis::ceres::HomogeneousPointParameterBlock(
              pit->second.pointHomog, pit->first));
      if (!mapPtr_->addParameterBlock(pointParameterBlock,
                                      okvis::ceres::Map::HomogeneousPoint)) {
        LOG(WARNING) << "Failed to add block for landmark " << pit->first;
        continue;
      }
      pit->second.setInState(true);
    }
    // examine starting from the rear of a landmark's observations, add
    // reprojection factors for those with null residual pointers, terminate
    // until a valid residual pointer is hit.
    MapPoint& mp = pit->second;
    for (std::map<okvis::KeypointIdentifier, uint64_t>::reverse_iterator riter =
             mp.observations.rbegin();
         riter != mp.observations.rend(); ++riter) {
      ::ceres::ResidualBlockId retVal = 0u;
      if (riter->second == 0u) {
#define DISTORTION_MODEL_CASE(camera_geometry_t)                               \
  retVal = addPointFrameResidual<camera_geometry_t>(pit->first, riter->first); \
  riter->second = reinterpret_cast<uint64_t>(retVal);

        switch (distortionType) { DISTORTION_MODEL_SWITCH_CASES }

#undef DISTORTION_MODEL_CASE
      }
    }
  }
  return true;
}

bool Estimator::computeCovariance(Eigen::MatrixXd* cov) const {
  if (!estimatorOptions_.computeOkvisNees) {
    *cov = Eigen::Matrix<double, 15, 15>::Identity();
    return false;
  }
  uint64_t poseId = statesMap_.rbegin()->second.id;
  uint64_t speedAndBiasId = statesMap_.rbegin()
                                ->second.sensors.at(SensorStates::Imu)
                                .at(0)
                                .at(ImuSensorStates::SpeedAndBias)
                                .id;
  return mapPtr_->computeNavStateCovariance(poseId, {speedAndBiasId},
                                            marginalizationResidualId_, cov);
}

bool Estimator::computeCovarianceCeres(
    Eigen::MatrixXd *cov, ::ceres::CovarianceAlgorithmType covAlgorithm) const {
  uint64_t poseId = statesMap_.rbegin()->second.id;
  uint64_t speedAndBiasId = statesMap_.rbegin()
                                ->second.sensors.at(SensorStates::Imu)
                                .at(0)
                                .at(ImuSensorStates::SpeedAndBias)
                                .id;
  std::vector<
      Eigen::Matrix<double, -1, -1, Eigen::RowMajor>,
      Eigen::aligned_allocator<Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>>
      covarianceBlockList;
  bool status = mapPtr_->getParameterBlockMinimalCovariance(
      {poseId, speedAndBiasId}, mapPtr_->problemUnsafe(), &covarianceBlockList,
      covAlgorithm);
  if (status) {
    cov->resize(6 + 9, 6 + 9);
    cov->topLeftCorner<6, 6>() = covarianceBlockList[0];
    cov->topRightCorner<6, 9>() = covarianceBlockList[1];
    cov->bottomLeftCorner<9, 6>() = covarianceBlockList[1].transpose();
    cov->bottomRightCorner<9, 9>() = covarianceBlockList[2];
  } else {
    LOG(INFO)
        << "The ceres::Covariance with SPARSE_QR often raises rank deficient "
           "Jacobian exception because there are low-disparity landmarks.";
  }
  return status;
}

// getters
bool Estimator::getStateStd(
    Eigen::Matrix<double, Eigen::Dynamic, 1>* stateStd) const {
  // skip computing covariance in processing real world data.
  *stateStd = Eigen::MatrixXd::Constant(15, 1, 1.0);
  return true;
}

void Estimator::updateSensorRigs() {
  size_t numCameras = cameraRig_.numCameras();
  const uint64_t currFrameId = currentFrameId();
  for (size_t camIdx = 0u; camIdx < numCameras; ++camIdx) {
    okvis::kinematics::Transformation T_XCi;
    getCameraSensorStates(currFrameId, camIdx, T_XCi);
    cameraRig_.setCameraExtrinsic(camIdx, T_XCi);
  }
}

}  // namespace okvis


