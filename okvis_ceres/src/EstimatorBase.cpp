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
 * @file EstimatorBase.cpp
 * @brief Source file for the EstimatorBase class.
 * @author Stefan Leutenegger
 * @author Andreas Forster
 */

#include <glog/logging.h>
#include <okvis/EstimatorBase.hpp>
#include <okvis/CameraModelSwitch.hpp>
#include <okvis/ceres/PoseParameterBlock.hpp>
#include <okvis/ceres/ImuError.hpp>
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
EstimatorBase::EstimatorBase(
    std::shared_ptr<okvis::ceres::Map> mapPtr)
    : mapPtr_(mapPtr),
      referencePoseId_(0), 
      initStatus_(InitializationStatus::Ongoing) {
}

// The default constructor.
EstimatorBase::EstimatorBase()
    : mapPtr_(new okvis::ceres::Map()),
      referencePoseId_(0), 
      initStatus_(InitializationStatus::Ongoing) {
}

EstimatorBase::~EstimatorBase() {
}

// Add a camera to the configuration. Sensors can only be added and never removed.
int EstimatorBase::addCameraParameterStds(
    const CameraNoiseParameters & cameraNoiseParameters)
{
  cameraNoiseParametersVec_.push_back(cameraNoiseParameters);
  return cameraNoiseParametersVec_.size() - 1;
}

void EstimatorBase::addCameraSystem(const okvis::cameras::NCameraSystem& cameras) {
  cameraRig_ = swift_vio::CameraRig::deepCopy(cameras);
}

// Add an IMU to the configuration.
int EstimatorBase::addImu(const ImuParameters & imuParameters)
{
  if(imuParametersVec_.size()>0u){
    LOG(ERROR) << "only one IMU currently supported";
    return -1;
  }
  imuParametersVec_.emplace_back(new ImuParameters(imuParameters));
  imuRig_.addImu(imuParameters);
  return imuParametersVec_.size() - 1;
}

// Remove all cameras from the configuration
void EstimatorBase::clearCameras(){
  cameraNoiseParametersVec_.clear();
}

// Remove all IMUs from the configuration.
void EstimatorBase::clearImus(){
  imuParametersVec_.clear();
}

// Add a landmark.
bool EstimatorBase::addLandmark(uint64_t landmarkId,
                            const Eigen::Vector4d & landmark) {
  // Landmark parameter blocks will be added by the derived estimators.
//  std::shared_ptr<okvis::ceres::HomogeneousPointParameterBlock> pointParameterBlock(
//      new okvis::ceres::HomogeneousPointParameterBlock(landmark, landmarkId));
//  if (!mapPtr_->addParameterBlock(pointParameterBlock,
//                                  okvis::ceres::Map::HomogeneousPoint)) {
//    return false;
//  }

  // remember
  double dist = std::numeric_limits<double>::max();
  if(fabs(landmark[3])>1.0e-8){
    dist = (landmark/landmark[3]).head<3>().norm(); // euclidean distance
  }
  auto result = landmarksMap_.insert(
      std::pair<uint64_t, MapPoint>(
          landmarkId, MapPoint(landmarkId, landmark, 0.0, dist)));
  OKVIS_ASSERT_TRUE_DBG(Exception, isLandmarkAdded(landmarkId), "bug: inconsistend landmarkdMap_ with mapPtr_.");
  return result.second;
}

// Remove an observation from a landmark, if available.
bool EstimatorBase::removeObservation(uint64_t landmarkId, uint64_t poseId,
                                  size_t camIdx, size_t keypointIdx) {
  if(landmarksMap_.find(landmarkId) == landmarksMap_.end()){
    for (PointMap::iterator it = landmarksMap_.begin(); it!= landmarksMap_.end(); ++it) {
      LOG(INFO) << it->first<<", no. obs = "<<it->second.observations.size();
    }
    LOG(INFO) << landmarksMap_.at(landmarkId).id;
  }
  OKVIS_ASSERT_TRUE_DBG(Exception, isLandmarkAdded(landmarkId),
                     "landmark not added");

  okvis::KeypointIdentifier kid(poseId,camIdx,keypointIdx);
  MapPoint& mapPoint = landmarksMap_.at(landmarkId);
  std::map<okvis::KeypointIdentifier, uint64_t >::iterator it = mapPoint.observations.find(kid);
  if(it == landmarksMap_.at(landmarkId).observations.end()){
    return false; // observation not present
  }

  // remove residual block
  if (it->second) { // nullptr becomes possible since we separated feature matching and adding residuals.
    mapPtr_->removeResidualBlock(
        reinterpret_cast<::ceres::ResidualBlockId>(it->second));
  }
  // remove also in local map
  mapPoint.observations.erase(it);

  return true;
}

void EstimatorBase::printNavStateAndBiases(std::ostream& stream, uint64_t poseId) const {
  std::shared_ptr<ceres::PoseParameterBlock> poseParamBlockPtr =
      std::static_pointer_cast<ceres::PoseParameterBlock>(
          mapPtr_->parameterBlockPtr(poseId));
  kinematics::Transformation T_WS = poseParamBlockPtr->estimate();

  const States& stateInQuestion = statesMap_.at(poseId);
  okvis::Time currentTime = stateInQuestion.timestamp;

  Eigen::Quaterniond q_WS = T_WS.q();
  if (q_WS.w() < 0) {
    q_WS.coeffs() *= -1;
  }
  stream << currentTime << " " << poseId
         << " " << T_WS.parameters().transpose().format(swift_vio::kSpaceInitFmt);

  // imu sensor states
  const int imuIdx = 0;
  uint64_t SBId = stateInQuestion.sensors.at(SensorStates::Imu)
                      .at(imuIdx)
                      .at(ImuSensorStates::SpeedAndBias)
                      .id;
  std::shared_ptr<ceres::SpeedAndBiasParameterBlock> sbParamBlockPtr =
      std::static_pointer_cast<ceres::SpeedAndBiasParameterBlock>(
          mapPtr_->parameterBlockPtr(SBId));
  SpeedAndBiases sb = sbParamBlockPtr->estimate();
  stream << " " << sb.transpose().format(swift_vio::kSpaceInitFmt);
}

bool EstimatorBase::printStatesAndStdevs(std::ostream& stream, const Eigen::MatrixXd *cov) const {
  uint64_t poseId = statesMap_.rbegin()->first;
  printNavStateAndBiases(stream, poseId);

  size_t numCameras = cameraNoiseParametersVec_.size();
  for (size_t camIdx = 0u; camIdx < numCameras; ++camIdx) {
    Eigen::VectorXd extrinsicValues;
    getVariableCameraExtrinsics(poseId, camIdx, &extrinsicValues);
    stream << " " << extrinsicValues.transpose().format(swift_vio::kSpaceInitFmt);
  }
  Eigen::VectorXd stateStd;
  if (cov) {
    stateStd = cov->diagonal().cwiseSqrt();
  } else {
    Eigen::MatrixXd covariance;
    computeCovariance(&covariance);
    stateStd = covariance.diagonal().cwiseSqrt();
  }
  stream << " " << stateStd.transpose().format(swift_vio::kSpaceInitFmt) << "\n";
  return true;
}

std::string EstimatorBase::headerLine(const std::string delimiter) const {
  std::stringstream stream;
  stream << "timestamp(sec)" << delimiter << "frameId" << delimiter;
  std::vector<std::string> variableList = variableLabels();
  for (const auto& variable : variableList) {
    stream << variable << delimiter;
  }
  std::vector<std::string> minVarList = perturbationLabels();
  for (const auto &variable : minVarList) {
    stream << "std_" << variable << delimiter;
  }
  return stream.str();
}

std::string EstimatorBase::rmseHeaderLine(const std::string delimiter) const {
  std::stringstream ss;
  std::vector<std::string> minVarList = perturbationLabels();
  ss << "timestamp(sec)" << delimiter;
  for (const auto &variable : minVarList) {
    ss << variable << delimiter;
  }
  return ss.str();
}

// getters
// Get a specific landmark.
bool EstimatorBase::getLandmark(uint64_t landmarkId,
                                    MapPoint& mapPoint) const
{
  std::lock_guard<std::mutex> l(statesMutex_);
  if (landmarksMap_.find(landmarkId) == landmarksMap_.end()) {
    OKVIS_THROW_DBG(Exception,"landmark with id = "<<landmarkId<<" does not exist.")
    return false;
  }
  mapPoint = landmarksMap_.at(landmarkId);
  return true;
}

bool EstimatorBase::getLandmark(uint64_t landmarkId,
                                swift_vio::MapPoint& mapPoint) const
{
  std::lock_guard<std::mutex> l(statesMutex_);
  if (landmarksMap_.find(landmarkId) == landmarksMap_.end()) {
    OKVIS_THROW_DBG(Exception,"landmark with id = "<<landmarkId<<" does not exist.")
    return false;
  }
  mapPoint = swift_vio::MapPoint(landmarksMap_.at(landmarkId));
  return true;
}

// Checks whether the landmark is initialized.
bool EstimatorBase::isLandmarkInitialized(uint64_t landmarkId) const {
  OKVIS_ASSERT_TRUE_DBG(Exception, isLandmarkAdded(landmarkId),
                     "landmark not added");
//  return std::static_pointer_cast<okvis::ceres::HomogeneousPointParameterBlock>(
//      mapPtr_->parameterBlockPtr(landmarkId))->initialized();
  return landmarksMap_.at(landmarkId).isInitialized();
}

size_t EstimatorBase::getLandmarks(PointMap & landmarks) const
{
  std::lock_guard<std::mutex> l(statesMutex_);
  landmarks = landmarksMap_;
  return landmarksMap_.size();
}

// Get a copy of all the landmark in a MapPointVector. This is for legacy support.
// Use getLandmarks(okvis::PointMap&) if possible.
size_t EstimatorBase::getLandmarks(MapPointVector & landmarks) const
{
  std::lock_guard<std::mutex> l(statesMutex_);
  landmarks.clear();
  landmarks.reserve(landmarksMap_.size());
  for(PointMap::const_iterator it=landmarksMap_.begin(); it!=landmarksMap_.end(); ++it){
    landmarks.push_back(it->second);
  }
  return landmarksMap_.size();
}

size_t EstimatorBase::getCurrentlyObservedLandmarks(swift_vio::MapPointVector *landmarks) const {
  std::lock_guard<std::mutex> l(statesMutex_);
  landmarks->clear();
  landmarks->reserve(landmarksMap_.size());
  for(PointMap::const_iterator it=landmarksMap_.begin(); it!=landmarksMap_.end(); ++it){
    landmarks->emplace_back(it->second);
  }
  return landmarksMap_.size();
}

size_t EstimatorBase::getMarginalizedLandmarks(
    swift_vio::MapPointVector *landmarks) const {
  landmarks->clear();
  for (const auto &p : marginalizedLandmarks_) {
    landmarks->emplace_back(p);
  }
  return marginalizedLandmarks_.size();
}

// Get pose for a given pose ID.
bool EstimatorBase::get_T_WS(uint64_t poseId,
                                 okvis::kinematics::Transformation & T_WS) const
{
  if (!getGlobalStateEstimateAs<ceres::PoseParameterBlock>(poseId,
                                                           GlobalStates::T_WS,
                                                           T_WS)) {
    return false;
  }

  return true;
}

// Feel free to implement caching for them...
// Get speeds and IMU biases for a given pose ID.
bool EstimatorBase::getSpeedAndBias(uint64_t poseId, uint64_t imuIdx,
                                okvis::SpeedAndBias & speedAndBias) const
{
  if (!getSensorStateEstimateAs<ceres::SpeedAndBiasParameterBlock>(
      poseId, imuIdx, SensorStates::Imu, ImuSensorStates::SpeedAndBias,
      speedAndBias)) {
    return false;
  }
  return true;
}

bool EstimatorBase::getSpeed(uint64_t poseId, Eigen::Vector3d &speed) const
{
  Eigen::Matrix<double, 9, 1> speedAndBias;
  if (!getSensorStateEstimateAs<ceres::SpeedAndBiasParameterBlock>(
      poseId, 0, SensorStates::Imu, ImuSensorStates::SpeedAndBias,
      speedAndBias)) {
    return false;
  }
  speed = speedAndBias.head<3>();
  return true;
}

bool EstimatorBase::getImuBiases(uint64_t poseId, uint64_t imuIdx,
                                 Eigen::Matrix<double, 6, 1> &bgba) const {
  Eigen::Matrix<double, 9, 1> speedAndBias;
  if (!getSensorStateEstimateAs<ceres::SpeedAndBiasParameterBlock>(
          poseId, imuIdx, SensorStates::Imu, ImuSensorStates::SpeedAndBias,
          speedAndBias)) {
    return false;
  }
  bgba = speedAndBias.tail<6>();
  return true;
}

// Get camera states for a given pose ID.
bool EstimatorBase::getCameraSensorStates(
    uint64_t poseId, size_t cameraIdx,
    okvis::kinematics::Transformation & T_XCi) const
{
  return getSensorStateEstimateAs<ceres::PoseParameterBlock>(
      poseId, cameraIdx, SensorStates::Camera, CameraSensorStates::T_XCi, T_XCi);
}

// Get the ID of the current keyframe.
uint64_t EstimatorBase::currentKeyframeId() const {
  for (std::map<uint64_t, States>::const_reverse_iterator rit = statesMap_.rbegin();
      rit != statesMap_.rend(); ++rit) {
    if (rit->second.isKeyframe) {
      return rit->first;
    }
  }
  OKVIS_THROW_DBG(Exception, "no keyframes existing...");
  return 0;
}

// Get the ID of an older frame.
uint64_t EstimatorBase::frameIdByAge(size_t age) const {
  std::map<uint64_t, States>::const_reverse_iterator rit = statesMap_.rbegin();
  for(size_t i=0; i<age; ++i){
    ++rit;
    OKVIS_ASSERT_TRUE_DBG(Exception, rit != statesMap_.rend(),
                       "requested age " << age << " out of range.");
  }
  return rit->first;
}

// Get the ID of the newest frame added to the state.
uint64_t EstimatorBase::currentFrameId() const {
  OKVIS_ASSERT_TRUE_DBG(Exception, statesMap_.size()>0, "no frames added yet.")
  return statesMap_.rbegin()->first;
}

okvis::Time EstimatorBase::currentFrameTimestamp() const {
  OKVIS_ASSERT_TRUE_DBG(Exception, statesMap_.size() > 0,
                        "no frames added yet.")
  return statesMap_.rbegin()->second.timestamp;
}

uint64_t EstimatorBase::oldestFrameId() const {
  OKVIS_ASSERT_TRUE_DBG(Exception, statesMap_.size() > 0,
                        "no frames added yet.")
  return statesMap_.begin()->first;
}

okvis::Time EstimatorBase::oldestFrameTimestamp() const {
  return statesMap_.begin()->second.timestamp;
}

size_t EstimatorBase::statesMapSize() const {
  return statesMap_.size();
}

// Checks if a particular frame is still in the IMU window
bool EstimatorBase::isInImuWindow(uint64_t frameId) const {
  if(statesMap_.at(frameId).sensors.at(SensorStates::Imu).size()==0){
    return false; // no IMU added
  }
  return statesMap_.at(frameId).sensors.at(SensorStates::Imu).at(0).at(ImuSensorStates::SpeedAndBias).exists;
}

// Set pose for a given pose ID.
bool EstimatorBase::set_T_WS(uint64_t poseId,
                                 const okvis::kinematics::Transformation & T_WS)
{
  auto iter = statesMap_.find(poseId);
  if (iter != statesMap_.end() && iter->second.positionVelocityLin) {
    iter->second.positionVelocityLin->head<3>() = T_WS.r();
  }
  if (!setGlobalStateEstimateAs<ceres::PoseParameterBlock>(poseId,
                                                           GlobalStates::T_WS,
                                                           T_WS)) {
    return false;
  }

  return true;
}

// Set the speeds and IMU biases for a given pose ID.
bool EstimatorBase::setSpeedAndBias(uint64_t poseId, size_t imuIdx, const okvis::SpeedAndBias & speedAndBias)
{
  auto iter = statesMap_.find(poseId);
  if (iter != statesMap_.end() && iter->second.positionVelocityLin) {
    iter->second.positionVelocityLin->tail<3>() = speedAndBias.head<3>();
  }
  return setSensorStateEstimateAs<ceres::SpeedAndBiasParameterBlock>(
      poseId, imuIdx, SensorStates::Imu, ImuSensorStates::SpeedAndBias, speedAndBias);
}

// Set the homogeneous coordinates for a landmark.
bool EstimatorBase::setLandmark(
    uint64_t landmarkId, const Eigen::Vector4d & landmark)
{
  if (landmarksMap_.at(landmarkId).inState()) {
  std::shared_ptr<ceres::ParameterBlock> parameterBlockPtr = mapPtr_
      ->parameterBlockPtr(landmarkId);
#ifndef NDEBUG
  std::shared_ptr<ceres::HomogeneousPointParameterBlock> derivedParameterBlockPtr =
  std::dynamic_pointer_cast<ceres::HomogeneousPointParameterBlock>(parameterBlockPtr);
  if(!derivedParameterBlockPtr) {
    OKVIS_THROW_DBG(Exception,"wrong pointer type requested.")
    return false;
  }
  derivedParameterBlockPtr->setEstimate(landmark);;
#else
  std::static_pointer_cast<ceres::HomogeneousPointParameterBlock>(
      parameterBlockPtr)->setEstimate(landmark);
#endif
  }
  // also update in map
  landmarksMap_.at(landmarkId).pointHomog = landmark;
  return true;
}

// Set the landmark initialization state.
void EstimatorBase::setLandmarkInitialized(uint64_t landmarkId,
                                               bool initialized) {
  OKVIS_ASSERT_TRUE_DBG(Exception, isLandmarkAdded(landmarkId),
                     "landmark not added");
//  std::static_pointer_cast<okvis::ceres::HomogeneousPointParameterBlock>(
//      mapPtr_->parameterBlockPtr(landmarkId))->setInitialized(initialized);
  landmarksMap_.at(landmarkId).setInitialized(initialized);
}

// private stuff
// getters
bool EstimatorBase::getGlobalStateParameterBlockPtr(
    uint64_t poseId, int stateType,
    std::shared_ptr<ceres::ParameterBlock>& stateParameterBlockPtr) const
{
  // check existence in states set
  if (statesMap_.find(poseId) == statesMap_.end()) {
    OKVIS_THROW(Exception,"pose with id = "<<poseId<<" does not exist.")
    return false;
  }

  // obtain the parameter block ID
  uint64_t id = statesMap_.at(poseId).global.at(stateType).id;
  if (!mapPtr_->parameterBlockExists(id)) {
    OKVIS_THROW(Exception,"pose with id = "<<id<<" does not exist.")
    return false;
  }

  stateParameterBlockPtr = mapPtr_->parameterBlockPtr(id);
  return true;
}
template<class PARAMETER_BLOCK_T>
bool EstimatorBase::getGlobalStateParameterBlockAs(
    uint64_t poseId, int stateType,
    PARAMETER_BLOCK_T & stateParameterBlock) const
{
  // convert base class pointer with various levels of checking
  std::shared_ptr<ceres::ParameterBlock> parameterBlockPtr;
  if (!getGlobalStateParameterBlockPtr(poseId, stateType, parameterBlockPtr)) {
    return false;
  }
#ifndef NDEBUG
  std::shared_ptr<PARAMETER_BLOCK_T> derivedParameterBlockPtr =
  std::dynamic_pointer_cast<PARAMETER_BLOCK_T>(parameterBlockPtr);
  if(!derivedParameterBlockPtr) {
    LOG(INFO) << "--"<<parameterBlockPtr->typeInfo();
    std::shared_ptr<PARAMETER_BLOCK_T> info(new PARAMETER_BLOCK_T);
    OKVIS_THROW_DBG(Exception,"wrong pointer type requested: requested "
                 <<info->typeInfo()<<" but is of type"
                 <<parameterBlockPtr->typeInfo())
    return false;
  }
  stateParameterBlock = *derivedParameterBlockPtr;
#else
  stateParameterBlock = *std::static_pointer_cast<PARAMETER_BLOCK_T>(
      parameterBlockPtr);
#endif
  return true;
}
template<class PARAMETER_BLOCK_T>
bool EstimatorBase::getGlobalStateEstimateAs(
    uint64_t poseId, int stateType,
    typename PARAMETER_BLOCK_T::estimate_t & state) const
{
  PARAMETER_BLOCK_T stateParameterBlock;
  if (!getGlobalStateParameterBlockAs(poseId, stateType, stateParameterBlock)) {
    return false;
  }
  state = stateParameterBlock.estimate();
  return true;
}

bool EstimatorBase::getSensorStateParameterBlockPtr(
    uint64_t poseId, int sensorIdx, int sensorType, int stateType,
    std::shared_ptr<ceres::ParameterBlock>& stateParameterBlockPtr) const
{
  // check existence in states set
  if (statesMap_.find(poseId) == statesMap_.end()) {
    OKVIS_THROW_DBG(Exception,"pose with id = "<<poseId<<" does not exist.")
    return false;
  }

  // obtain the parameter block ID
  uint64_t id = statesMap_.at(poseId).sensors.at(sensorType).at(sensorIdx).at(
      stateType).id;
  if (!mapPtr_->parameterBlockExists(id)) {
    OKVIS_THROW_DBG(Exception,"pose with id = "<<poseId<<" does not exist.")
    return false;
  }
  stateParameterBlockPtr = mapPtr_->parameterBlockPtr(id);
  return true;
}
template<class PARAMETER_BLOCK_T>
bool EstimatorBase::getSensorStateParameterBlockAs(
    uint64_t poseId, int sensorIdx, int sensorType, int stateType,
    PARAMETER_BLOCK_T & stateParameterBlock) const
{
  // convert base class pointer with various levels of checking
  std::shared_ptr<ceres::ParameterBlock> parameterBlockPtr;
  if (!getSensorStateParameterBlockPtr(poseId, sensorIdx, sensorType, stateType,
                                       parameterBlockPtr)) {
    return false;
  }
#ifndef NDEBUG
  std::shared_ptr<PARAMETER_BLOCK_T> derivedParameterBlockPtr =
  std::dynamic_pointer_cast<PARAMETER_BLOCK_T>(parameterBlockPtr);
  if(!derivedParameterBlockPtr) {
    std::shared_ptr<PARAMETER_BLOCK_T> info(new PARAMETER_BLOCK_T);
    OKVIS_THROW_DBG(Exception,"wrong pointer type requested: requested "
                     <<info->typeInfo()<<" but is of type"
                     <<parameterBlockPtr->typeInfo())
    return false;
  }
  stateParameterBlock = *derivedParameterBlockPtr;
#else
  stateParameterBlock = *std::static_pointer_cast<PARAMETER_BLOCK_T>(
      parameterBlockPtr);
#endif
  return true;
}

template<class PARAMETER_BLOCK_T>
bool EstimatorBase::setGlobalStateEstimateAs(
    uint64_t poseId, int stateType,
    const typename PARAMETER_BLOCK_T::estimate_t & state)
{
  // check existence in states set
  if (statesMap_.find(poseId) == statesMap_.end()) {
    OKVIS_THROW_DBG(Exception,"pose with id = "<<poseId<<" does not exist.")
    return false;
  }

  // obtain the parameter block ID
  uint64_t id = statesMap_.at(poseId).global.at(stateType).id;
  if (!mapPtr_->parameterBlockExists(id)) {
    OKVIS_THROW_DBG(Exception,"pose with id = "<<poseId<<" does not exist.")
    return false;
  }

  std::shared_ptr<ceres::ParameterBlock> parameterBlockPtr = mapPtr_
      ->parameterBlockPtr(id);
#ifndef NDEBUG
  std::shared_ptr<PARAMETER_BLOCK_T> derivedParameterBlockPtr =
  std::dynamic_pointer_cast<PARAMETER_BLOCK_T>(parameterBlockPtr);
  if(!derivedParameterBlockPtr) {
    OKVIS_THROW_DBG(Exception,"wrong pointer type requested.")
    return false;
  }
  derivedParameterBlockPtr->setEstimate(state);
#else
  std::static_pointer_cast<PARAMETER_BLOCK_T>(parameterBlockPtr)->setEstimate(
      state);
#endif
  return true;
}

template<class PARAMETER_BLOCK_T>
bool EstimatorBase::setSensorStateEstimateAs(
    uint64_t poseId, int sensorIdx, int sensorType, int stateType,
    const typename PARAMETER_BLOCK_T::estimate_t & state)
{
  // check existence in states set
  if (statesMap_.find(poseId) == statesMap_.end()) {
    OKVIS_THROW_DBG(Exception,"pose with id = "<<poseId<<" does not exist.")
    return false;
  }

  // obtain the parameter block ID
  uint64_t id = statesMap_.at(poseId).sensors.at(sensorType).at(sensorIdx).at(
      stateType).id;
  if (!mapPtr_->parameterBlockExists(id)) {
    OKVIS_THROW_DBG(Exception,"pose with id = "<<poseId<<" does not exist.")
    return false;
  }

  std::shared_ptr<ceres::ParameterBlock> parameterBlockPtr = mapPtr_
      ->parameterBlockPtr(id);
#ifndef NDEBUG
  std::shared_ptr<PARAMETER_BLOCK_T> derivedParameterBlockPtr =
  std::dynamic_pointer_cast<PARAMETER_BLOCK_T>(parameterBlockPtr);
  if(!derivedParameterBlockPtr) {
    OKVIS_THROW_DBG(Exception,"wrong pointer type requested.")
    return false;
  }
  derivedParameterBlockPtr->setEstimate(state);
#else
  std::static_pointer_cast<PARAMETER_BLOCK_T>(parameterBlockPtr)->setEstimate(
      state);
#endif
  return true;
}

size_t EstimatorBase::numObservations(uint64_t landmarkId) const {
  PointMap::const_iterator it = landmarksMap_.find(landmarkId);
  if (it != landmarksMap_.end())
    return it->second.observations.size();
  else
    return 0;
}

bool EstimatorBase::getLandmarkHeadObs(uint64_t landmarkId,
                                      okvis::KeypointIdentifier* kpId) const {
  auto lmIt = landmarksMap_.find(landmarkId);
  if (lmIt == landmarksMap_.end()) {
    OKVIS_THROW_DBG(Exception,
                    "landmark with id = " << landmarkId << " does not exist.")
    return false;
  }
  *kpId = lmIt->second.observations.begin()->first;
  return true;
}

const okvis::MapPoint &EstimatorBase::getLandmarkUnsafe(uint64_t landmarkId) const {
  return landmarksMap_.at(landmarkId);
}

uint64_t EstimatorBase::mergeTwoLandmarks(uint64_t lmIdA, uint64_t lmIdB) {
  PointMap::iterator lmItA = landmarksMap_.find(lmIdA);
  PointMap::iterator lmItB = landmarksMap_.find(lmIdB);
  if (lmItB->second.observations.size() > lmItA->second.observations.size()) {
    std::swap(lmIdA, lmIdB);
    std::swap(lmItA, lmItB);
  }
  std::map<okvis::KeypointIdentifier, uint64_t>& obsMapA =
      lmItA->second.observations;
  std::map<okvis::KeypointIdentifier, uint64_t>& obsMapB =
      lmItB->second.observations;
  for (std::map<okvis::KeypointIdentifier, uint64_t>::iterator obsIter =
           obsMapB.begin();
       obsIter != obsMapB.end(); ++obsIter) {
    if (obsIter->second != 0u) {
      mapPtr_->removeResidualBlock(
          reinterpret_cast<::ceres::ResidualBlockId>(obsIter->second));
      obsIter->second = 0u;
    }
    // reset landmark ids for relevant keypoints in multiframe.
    const KeypointIdentifier& kpi = obsIter->first;
    okvis::MultiFramePtr multiFramePtr = multiFramePtrMap_.at(kpi.frameId);
    auto iterA = std::find_if(obsMapA.begin(), obsMapA.end(),
                              IsObservedInFrame(kpi.frameId, kpi.cameraIndex));
    if (iterA != obsMapA.end()) {
      multiFramePtr->setLandmarkId(kpi.cameraIndex, kpi.keypointIndex, 0u);
      continue;
    }
    multiFramePtr->setLandmarkId(kpi.cameraIndex, kpi.keypointIndex, lmIdA);

    // jhuai: we will add reprojection factors in the optimize() step, hence, less
    // coupling between feature tracking frontend and estimator, and flexibility
    // in choosing which landmark's observations to use in the opt problem.
    obsMapA.emplace(kpi, 0u);
  }
  mapPtr_->removeParameterBlock(lmIdB);
  landmarksMap_.erase(lmItB);
  return lmIdB;
}

okvis::Time EstimatorBase::removeState(uint64_t stateId) {
  std::map<uint64_t, States>::iterator it = statesMap_.find(stateId);
  okvis::Time removedStateTime = it->second.timestamp;
  it->second.global[GlobalStates::T_WS].exists = false;  // remember we removed
  it->second.sensors.at(SensorStates::Imu)
      .at(0)
      .at(ImuSensorStates::SpeedAndBias)
      .exists = false;  // remember we removed
  mapPtr_->removeParameterBlock(it->second.global[GlobalStates::T_WS].id);
  mapPtr_->removeParameterBlock(it->second.sensors.at(SensorStates::Imu)
                                    .at(0)
                                    .at(ImuSensorStates::SpeedAndBias)
                                    .id);


  multiFramePtrMap_.erase(stateId);
  statesMap_.erase(it);
  return removedStateTime;
}

bool EstimatorBase::computeErrors(
    const okvis::kinematics::Transformation &ref_T_WS,
    const Eigen::Vector3d &ref_v_WS, const Eigen::Matrix<double, 6, 1> &biasRef,
    const okvis::ImuParameters &/*refImuParams*/,
    std::shared_ptr<const swift_vio::CameraRig> /*refCameraSystem*/,
    Eigen::VectorXd *errors) const {
  errors->resize(15);
  okvis::kinematics::Transformation est_T_WS;
  uint64_t currFrameId = currentFrameId();
  get_T_WS(currFrameId, est_T_WS);
  errors->head<3>() = ref_T_WS.r() - est_T_WS.r();
  Eigen::Matrix3d dR = ref_T_WS.C() * est_T_WS.C().transpose();
  errors->segment<3>(3) = okvis::kinematics::vee(dR);

  okvis::SpeedAndBias speedAndBiasEstimate;
  getSpeedAndBias(currFrameId, 0, speedAndBiasEstimate);
  errors->segment<3>(6) = speedAndBiasEstimate.head<3>() - ref_v_WS;
  errors->segment<3>(9) = speedAndBiasEstimate.segment<3>(3) - biasRef.head<3>();
  errors->segment<3>(12) = speedAndBiasEstimate.tail<3>() - biasRef.tail<3>();;
  return true;
}

bool EstimatorBase::getDesiredStdevs(Eigen::VectorXd *desiredStdevs) const {
  desiredStdevs->resize(15, 1);
  (*desiredStdevs) << 0.3, 0.3, 0.3, 0.08, 0.08, 0.08, 0.1, 0.1, 0.1, 0.002,
      0.002, 0.002, 0.02, 0.02, 0.02;
  return true;
}

size_t EstimatorBase::gatherMapPointObservations(
    const MapPoint& mp,
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>*
        obsDirections,
    std::vector<okvis::kinematics::Transformation,
                Eigen::aligned_allocator<okvis::kinematics::Transformation>>*
        T_CWs,
    std::vector<double>* imageNoiseStd) const {
  T_CWs->clear();
  obsDirections->clear();
  imageNoiseStd->clear();

  const std::map<okvis::KeypointIdentifier, uint64_t>& observations =
      mp.observations;
  size_t numPotentialObs = mp.observations.size();
  T_CWs->reserve(numPotentialObs);
  obsDirections->reserve(numPotentialObs);
  imageNoiseStd->reserve(numPotentialObs);

  uint64_t minValidStateId = statesMap_.begin()->first;
  for (auto itObs = observations.begin(), iteObs = observations.end();
       itObs != iteObs; ++itObs) {
    uint64_t poseId = itObs->first.frameId;

    if (poseId < minValidStateId) {
      continue;
    }
    Eigen::Vector2d measurement;
    auto multiFrameIter = multiFramePtrMap_.find(poseId);
    //    OKVIS_ASSERT_TRUE(Exception, multiFrameIter !=
    //    multiFramePtrMap_.end(), "multiframe not found");
    okvis::MultiFramePtr multiFramePtr = multiFrameIter->second;
    multiFramePtr->getKeypoint(itObs->first.cameraIndex,
                               itObs->first.keypointIndex, measurement);

    // use the latest estimates for camera intrinsic parameters
    Eigen::Vector3d backProjectionDirection;
    std::shared_ptr<const cameras::CameraBase> cameraGeometry =
        cameraRig_.cameraGeometry(itObs->first.cameraIndex);
    bool validDirection =
        cameraGeometry->backProject(measurement, &backProjectionDirection);
    if (!validDirection) {
      continue;
    }
    obsDirections->push_back(backProjectionDirection);

    okvis::kinematics::Transformation T_WB;
    get_T_WS(poseId, T_WB);
    okvis::kinematics::Transformation T_BC =
        cameraRig_.getCameraExtrinsic(itObs->first.cameraIndex);
    T_CWs->emplace_back((T_WB * T_BC).inverse());

    double kpSize = 1.0;
    multiFramePtr->getKeypointSize(itObs->first.cameraIndex,
                                   itObs->first.keypointIndex, kpSize);
    imageNoiseStd->push_back(kpSize / 8);
    imageNoiseStd->push_back(kpSize / 8);
  }
  return obsDirections->size();
}

bool EstimatorBase::getCameraSensorExtrinsics(
    size_t cameraIdx, okvis::kinematics::Transformation &T_BCi) const {
  T_BCi = cameraRig_.getCameraExtrinsic(cameraIdx);
  return true;
}

void EstimatorBase::getVariableCameraExtrinsics(
    uint64_t poseId, size_t camIdx,
    Eigen::Matrix<double, Eigen::Dynamic, 1> *extrinsicParams) const {
  const States &currentState = statesMap_.at(poseId);
  if (!cameraNoiseParametersVec_.at(camIdx).isExtrinsicsFixed()) {
    uint64_t extrinsicId = currentState.sensors.at(SensorStates::Camera)
                               .at(camIdx)
                               .at(okvis::EstimatorBase::CameraSensorStates::T_XCi)
                               .id;
    std::shared_ptr<okvis::ceres::PoseParameterBlock> extrinsicParamBlockPtr =
        std::static_pointer_cast<okvis::ceres::PoseParameterBlock>(
            mapPtr_->parameterBlockPtr(extrinsicId));
    okvis::kinematics::Transformation T_XC = extrinsicParamBlockPtr->estimate();
    swift_vio::ExtrinsicRepToParamValues(
        cameraRig_.getExtrinsicRepId(camIdx), T_XC, extrinsicParams);
  } else {
    extrinsicParams->resize(0);
  }
}

void EstimatorBase::getVariableCameraIntrinsics(
    uint64_t /*poseId*/,
    size_t /*camIdx*/,
    Eigen::Matrix<double, Eigen::Dynamic, 1>* intrinsicParams) const {
  intrinsicParams->resize(0);
}

void EstimatorBase::getImuAugmentedStatesEstimate(
    uint64_t /*poseId*/,
    size_t /*imuId*/,
    Eigen::Matrix<double, Eigen::Dynamic, 1>* extraParams) const {
  extraParams->resize(0);
}

void EstimatorBase::getEstimatedCameraSystem(okvis::cameras::NCameraSystem *cameraSystem) const {
  cameraRig_.assignTo(cameraSystem);
}

bool EstimatorBase::getOdometryConstraintsForKeyframe(
    std::shared_ptr<swift_vio::LoopQueryKeyframeMessage<okvis::MultiFrame>> queryKeyframe) const {
  int j = 0;
  auto& odometryConstraintList = queryKeyframe->odometryConstraintListMutable();
  odometryConstraintList.reserve(poseGraphOptions_.maxOdometryConstraintForAKeyframe);
  okvis::kinematics::Transformation T_WBr = queryKeyframe->T_WB_;
  queryKeyframe->setZeroCovariance();
  auto riter = statesMap_.rbegin();
  for (++riter;  // skip the last frame which is queryKeyframe.
       riter != statesMap_.rend() && j < poseGraphOptions_.maxOdometryConstraintForAKeyframe;
       ++riter) {
    if (riter->second.isKeyframe) {
      okvis::kinematics::Transformation T_WBn;
      get_T_WS(riter->first, T_WBn);
      okvis::kinematics::Transformation T_BnBr = T_WBn.inverse() * T_WBr;
      std::shared_ptr<swift_vio::NeighborConstraintMessage> odometryConstraint(
          new swift_vio::NeighborConstraintMessage(
              riter->first, riter->second.timestamp, T_BnBr, T_WBn));
      odometryConstraint->core_.squareRootInfo_.setIdentity();
      odometryConstraintList.emplace_back(odometryConstraint);
      ++j;
    }
  }
  return true;
}

// TODO(jhuai): Add heuristic rules to throttle loop query keyframes.
// 1, minimum time gap, 2, minimum distance, 3, minimum number of keypoints
// while keeping the keyframe of the previous message in the sliding window.
bool EstimatorBase::getLoopQueryKeyframeMessage(
    okvis::MultiFramePtr multiFrame,
    std::shared_ptr<swift_vio::LoopQueryKeyframeMessage<okvis::MultiFrame>>* queryKeyframe) const {
  auto riter = statesMap_.rbegin();
  if (!riter->second.isKeyframe) {
    return false;
  }
  okvis::kinematics::Transformation T_WBr;
  get_T_WS(riter->first, T_WBr);

  uint64_t queryKeyframeId = riter->first;
  queryKeyframe->reset(new swift_vio::LoopQueryKeyframeMessage<okvis::MultiFrame>(
      queryKeyframeId, riter->second.timestamp, T_WBr, multiFrame));

  getOdometryConstraintsForKeyframe(*queryKeyframe);

  // add 3d landmarks observed in query keyframe's first frame,
  // and corresponding indices into the 2d keypoint list.
  // The local camera frame will be used as their coordinate frame.
  const std::vector<uint64_t>& landmarkIdList =
      multiFrame->getLandmarkIds(swift_vio::LoopQueryKeyframeMessage::kQueryCameraIndex);
  size_t numKeypoints = landmarkIdList.size();
  auto& keypointIndexForLandmarkList =
      (*queryKeyframe)->keypointIndexForLandmarkListMutable();
  keypointIndexForLandmarkList.reserve(numKeypoints / 4);
  auto& landmarkPositionList = (*queryKeyframe)->landmarkPositionListMutable();
  landmarkPositionList.reserve(numKeypoints / 4);
  int keypointIndex = 0;

  okvis::kinematics::Transformation T_BrW = T_WBr.inverse();
  for (const uint64_t landmarkId : landmarkIdList) {
    if (landmarkId != 0) {
      auto result = landmarksMap_.find(landmarkId);
      if (result != landmarksMap_.end() && result->second.quality > 1e-6) {
        keypointIndexForLandmarkList.push_back(keypointIndex);
        Eigen::Vector4d hp_W = result->second.pointHomog;
        Eigen::Vector4d hp_B = T_BrW * hp_W;
        landmarkPositionList.push_back(hp_B);
      }
    }
    ++keypointIndex;
  }
  return true;
}

const okvis::Duration EstimatorBase::half_window_(2, 0);

}  // namespace okvis


