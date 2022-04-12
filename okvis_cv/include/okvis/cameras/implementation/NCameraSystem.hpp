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
 *  Created on: Apr 1, 2015
 *      Author: Stefan Leutenegger (s.leutenegger@imperial.ac.uk)
 *    Modified: Andreas Forster (an.forster@gmail.com)
 *********************************************************************************/

/**
 * @file implementation/NCameraSystem.hpp
 * @brief Header implementation file for the NCameraSystem class.
 * @author Stefan Leutenegger
 * @author Andreas Forster
 */


/// \brief okvis Main namespace of this package.
namespace okvis {
/// \brief cameras Namespace for camera-related functionality.
namespace cameras {

// Default constructor
NCameraSystem::NCameraSystem()
{
}
// Construct with vector of extrinsics and geometries
NCameraSystem::NCameraSystem(
    const std::vector<std::shared_ptr<okvis::kinematics::Transformation>> & T_SC,
    const std::vector<std::shared_ptr<cameras::CameraBase>> & cameraGeometries,
    const std::vector<DistortionType>& distortionTypes,
    bool computeOverlaps)
    : T_SC_(T_SC),
      cameraGeometries_(cameraGeometries),
      distortionTypes_(distortionTypes)
{

  OKVIS_ASSERT_TRUE_DBG(
      Exception, T_SC.size() == cameraGeometries.size(),
      "Number of extrinsics must match number of camera models!");
  OKVIS_ASSERT_TRUE_DBG(
      Exception, T_SC.size() == distortionTypes.size(),
      "Number of distortiontypes must match number of extrinsics!");
  extrinsicRepNames_.resize(numCameras());
  projectionIntrinsicRepNames_.resize(numCameras());
  if (computeOverlaps) {
    this->computeOverlaps();
  }
}
NCameraSystem::~NCameraSystem()
{
}

// Reset with vector of extrinsics and geometries
void NCameraSystem::reset(
    const std::vector<std::shared_ptr<okvis::kinematics::Transformation>> & T_SC,
    const std::vector<std::shared_ptr<cameras::CameraBase>> & cameraGeometries,
    const std::vector<DistortionType>& distortionTypes,
    bool computeOverlaps)
{
  OKVIS_ASSERT_TRUE_DBG(
      Exception, T_SC.size() == cameraGeometries.size(),
      "Number of extrinsics must match number of camera models!");
  OKVIS_ASSERT_TRUE_DBG(
      Exception, T_SC.size() == distortionTypes.size(),
      "Number of distortiontypes must match number of extrinsics!");

  T_SC_ = T_SC;
  cameraGeometries_ = cameraGeometries;
  distortionTypes_ = distortionTypes;
  extrinsicRepNames_.resize(numCameras());
  projectionIntrinsicRepNames_.resize(numCameras());
  // recompute overlaps if requested
  if (computeOverlaps) {
    this->computeOverlaps();
  }
}

// Reset with vector of extrinsics and geometries
void NCameraSystem::addCamera(
    std::shared_ptr<okvis::kinematics::Transformation> T_SC,
    std::shared_ptr<cameras::CameraBase> cameraGeometry,
    DistortionType distortionType,
    std::string projectionIntrinsicRepName,
    std::string extrinsicRepName,
    bool computeOverlaps)
{
  T_SC_.push_back(T_SC);
  cameraGeometries_.push_back(cameraGeometry);
  distortionTypes_.push_back(distortionType);
  projectionIntrinsicRepNames_.emplace_back(projectionIntrinsicRepName);
  extrinsicRepNames_.emplace_back(extrinsicRepName);
  // recompute overlaps if requested
  if (computeOverlaps) {
    this->computeOverlaps();
  }
}

void NCameraSystem::removeCamera(size_t camId) {
  T_SC_.erase(T_SC_.begin() + camId);
  cameraGeometries_.erase(cameraGeometries_.begin() + camId);
  distortionTypes_.erase(distortionTypes_.begin() + camId);
  if (overlapMats_.size()) {
    overlapMats_.erase(overlapMats_.begin() + camId);
    for (std::vector<cv::Mat> &matlist : overlapMats_) {
      matlist.erase(matlist.begin() + camId);
    }
  }
  if (overlaps_.size()) {
    overlaps_.erase(overlaps_.begin() + camId);
    for (std::vector<bool> &overlaplist : overlaps_) {
      overlaplist.erase(overlaplist.begin() + camId);
    }
  }
  projectionIntrinsicRepNames_.erase(projectionIntrinsicRepNames_.begin() +
                                     camId);
  extrinsicRepNames_.erase(extrinsicRepNames_.begin() + camId);
}

// get the pose of the IMU frame S with respect to the camera cameraIndex
std::shared_ptr<const okvis::kinematics::Transformation> NCameraSystem::T_SC(
    size_t cameraIndex) const
{
  OKVIS_ASSERT_TRUE_DBG(Exception, cameraIndex < T_SC_.size(),
                        "Camera index " << cameraIndex << "out of range.");
  return T_SC_[cameraIndex];
}

//get the camera geometry of camera cameraIndex
std::shared_ptr<const cameras::CameraBase> NCameraSystem::cameraGeometry(
    size_t cameraIndex) const
{
  OKVIS_ASSERT_TRUE_DBG(Exception, cameraIndex < cameraGeometries_.size(),
                        "Camera index " << cameraIndex << "out of range.");
  return cameraGeometries_[cameraIndex];
}

std::shared_ptr<cameras::CameraBase> NCameraSystem::cameraGeometryMutable(
    size_t cameraIndex)
{
  OKVIS_ASSERT_TRUE_DBG(Exception, cameraIndex < cameraGeometries_.size(),
                        "Camera index " << cameraIndex << "out of range.");
  return cameraGeometries_[cameraIndex];
}

// get the distortion type of cmaera cameraIndex
inline DistortionType NCameraSystem::distortionType(size_t cameraIndex) const
{
  OKVIS_ASSERT_TRUE_DBG(Exception, cameraIndex < cameraGeometries_.size(),
                        "Camera index " << cameraIndex << "out of range.");
  return distortionTypes_[cameraIndex];
}

// Get the overlap mask
//const cv::Mat NCameraSystem::overlap(size_t cameraIndexSeenBy,
//                                      size_t cameraIndex) const
//{
//  OKVIS_ASSERT_TRUE_DBG(
//      Exception, cameraIndexSeenBy < T_SC_.size(),
//      "Camera index " << cameraIndexSeenBy << "out of range.");
//  OKVIS_ASSERT_TRUE_DBG(Exception, cameraIndex < T_SC_.size(),
//                        "Camera index " << cameraIndex << "out of range.");

//  OKVIS_ASSERT_TRUE_DBG(Exception, overlapComputationValid(),
//                            "Overlap computation not performed or incorrectly computed!");

//  return overlapMats_[cameraIndexSeenBy][cameraIndex];
//}

const std::vector<std::vector<bool>> &NCameraSystem::overlaps() const {
  return overlaps_;
}

// Can the first camera see parts of the FOV of the second camera?
bool NCameraSystem::hasOverlap(size_t cameraIndexSeenBy,
                                      size_t cameraIndex) const
{
  OKVIS_ASSERT_TRUE_DBG(
      Exception, cameraIndexSeenBy < T_SC_.size(),
      "Camera index " << cameraIndexSeenBy << "out of range.");
  OKVIS_ASSERT_TRUE_DBG(Exception, cameraIndex < T_SC_.size(),
                        "Camera index " << cameraIndex << "out of range.");
  OKVIS_ASSERT_TRUE_DBG(Exception, cameraIndex < T_SC_.size(),
                        "Camera index " << cameraIndex << "out of range.");

  OKVIS_ASSERT_TRUE_DBG(Exception, overlapComputationValid(),
                          "Overlap computation not performed or incorrectly computed!");

  return overlaps_[cameraIndexSeenBy][cameraIndex];
}

bool NCameraSystem::overlapComputationValid() const {
  OKVIS_ASSERT_TRUE_DBG(
      Exception, T_SC_.size() == cameraGeometries_.size(),
      "Number of extrinsics must match number of camera models!");

  if(overlaps_.size() != cameraGeometries_.size()) {
    return false;
  }
  if(overlapMats_.size() != cameraGeometries_.size()) {
    return false;
  }

  // also check for each element
  for(size_t i= 0; i<overlaps_.size(); ++i){
    if(overlaps_[i].size() != cameraGeometries_.size()) {
      return false;
    }
    if(overlapMats_[i].size() != cameraGeometries_.size()) {
      return false;
    }
  }
  return true;
}

size_t NCameraSystem::numCameras() const {
  return cameraGeometries_.size();
}

std::string NCameraSystem::projectionIntrinsicRep(size_t cameraIndex) const {
  return projectionIntrinsicRepNames_[cameraIndex];
}

std::string NCameraSystem::extrinsicRep(size_t cameraIndex) const {
  return extrinsicRepNames_[cameraIndex];
}

void NCameraSystem::set_T_SC(
    size_t camIdx,
    std::shared_ptr<const okvis::kinematics::Transformation> T_SC) {
  *T_SC_[camIdx] = *T_SC;
}

void NCameraSystem::setCameraIntrinsics(int camera_id,
                                        const Eigen::VectorXd &intrinsic_vec) {
  cameraGeometries_[camera_id]->setIntrinsics(intrinsic_vec);
}

void NCameraSystem::setImageDelay(int camera_id, double td) {
  cameraGeometries_[camera_id]->setImageDelay(td);
}

void NCameraSystem::setReadoutTime(int camera_id, double tr) {
  cameraGeometries_[camera_id]->setReadoutTime(tr);
}

void NCameraSystem::setProjectionIntrinsicRepName(int camera_id, const std::string& rep_name) {
  projectionIntrinsicRepNames_[camera_id] = rep_name;
}

void NCameraSystem::setExtrinsicRepName(int camera_id, const std::string& rep_name) {
  extrinsicRepNames_[camera_id] = rep_name;
}

void NCameraSystem::setOverlaps(const std::vector<std::vector<bool>> &overlaps) {
  overlaps_ = overlaps;
}

void NCameraSystem::setOverlapMats(const std::vector<std::vector<cv::Mat>> &overlapMats) {
  overlapMats_ = overlapMats;
}
}  // namespace cameras
}  // namespace okvis
