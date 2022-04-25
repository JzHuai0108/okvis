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
 * @file okvis/EstimatorBase.hpp
 * @brief Header file for the EstimatorBase class. This does all the backend work.
 * @author Stefan Leutenegger
 * @author Andreas Forster
 */

#ifndef INCLUDE_OKVIS_ESTIMATOR_BASE_HPP_
#define INCLUDE_OKVIS_ESTIMATOR_BASE_HPP_

#include <memory>
#include <mutex>
#include <array>

#include <ceres/ceres.h>
#include <okvis/kinematics/Transformation.hpp>

#include <okvis/assert_macros.hpp>
#include <loop_closure/KeyframeForLoopDetection.hpp>
#include <loop_closure/LoopFrameAndMatches.hpp>
#include <okvis/VioBackendInterface.hpp>
#include <okvis/MultiFrame.hpp>
#include <okvis/FrameTypedefs.hpp>
#include <okvis/Measurements.hpp>
#include <okvis/Variables.hpp>
#include <okvis/ceres/PoseParameterBlock.hpp>
#include <okvis/ceres/SpeedAndBiasParameterBlock.hpp>
#include <okvis/ceres/HomogeneousPointParameterBlock.hpp>
#include <okvis/ceres/Map.hpp>
#include <okvis/ceres/MarginalizationError.hpp>
#include <okvis/ceres/ReprojectionError.hpp>
#include <okvis/ceres/CeresIterationCallback.hpp>

#include <swift_vio/imu/BoundedImuDeque.hpp>
#include <swift_vio/CameraRig.hpp>
#include <swift_vio/imu/ImuRig.hpp>
#include <swift_vio/InitialNavState.hpp>
#include <swift_vio/MapPoint.h>

/// \brief okvis Main namespace of this package.
namespace okvis {

//! The estimator class
/*!
 The estimator class. This does all the backend work.
 Frames:
 W: World
 B: Body
 C: Camera
 S: Sensor (IMU)
 */
class EstimatorBase : public VioBackendInterface
{
 public:
  OKVIS_DEFINE_EXCEPTION(Exception, std::runtime_error)
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /**
   * @brief The default constructor.
   */
  EstimatorBase();

  /**
   * @brief Constructor if a ceres map is already available.
   * @param mapPtr Shared pointer to ceres map.
   */
  EstimatorBase(std::shared_ptr<okvis::ceres::Map> mapPtr);
  virtual ~EstimatorBase();

  /// @name Sensor configuration related
  ///@{
  /**
   * @brief Add a camera to the configuration. Sensors can only be added and never removed.
   * @param cameraNoiseParameters The parameters that tell how to estimate extrinsics.
   * @return Index of new camera.
   */
  int addCameraParameterStds(const okvis::CameraNoiseParameters&
                                     cameraNoiseParameters) final;
  /**
   * @brief addCameraSystem add the cameraSystem to the estimator.
   * @param cameras
   */
  void addCameraSystem(const okvis::cameras::NCameraSystem& cameras) override;

  void getEstimatedCameraSystem(okvis::cameras::NCameraSystem *cameraSystem) const;

  /**
   * @brief Add an IMU to the configuration.
   * @warning Currently there is only one IMU supported.
   * @param imuParameters The IMU parameters.
   * @return index of IMU.
   */
  int addImu(const okvis::ImuParameters &imuParameters) override;

  /**
   * @brief Remove all cameras from the configuration
   */
  void clearCameras() final;

  /**
   * @brief Remove all IMUs from the configuration.
   */
  void clearImus() final;

  /// @}

  /**
   * @brief Add a pose to the state.
   * @param multiFrame Matched multiFrame.
   * @param imuMeasurements IMU measurements from last state to new one.
   * @param asKeyframe Is this new frame a keyframe?
   * @return True if successful.
   */
  bool addStates(okvis::MultiFramePtr /*multiFrame*/,
                 const okvis::ImuMeasurementDeque &/*imuMeasurements*/,
                 bool /*asKeyframe*/) override {
    return true;
  }

  /**
   * @brief Add a landmark.
   * @param landmarkId ID of the new landmark.
   * @param landmark Homogeneous coordinates of landmark in W-frame.
   * @return True if successful.
   */
  bool addLandmark(uint64_t landmarkId,
                   const Eigen::Vector4d & landmark) final;

  /**
   * @brief Add an observation to a landmark.
   * \tparam GEOMETRY_TYPE The camera geometry type for this observation.
   * @param landmarkId ID of landmark.
   * @param poseId ID of pose where the landmark was observed.
   * @param camIdx ID of camera frame where the landmark was observed.
   * @param keypointIdx ID of keypoint corresponding to the landmark.
   * @return Residual block ID for that observation.
   */
  template<class GEOMETRY_TYPE>
  ::ceres::ResidualBlockId addObservation(uint64_t landmarkId, uint64_t poseId,
                                          size_t camIdx, size_t keypointIdx);

  /**
   * @brief Remove an observation from a landmark, if available.
   * @param landmarkId ID of landmark.
   * @param poseId ID of pose where the landmark was observed.
   * @param camIdx ID of camera frame where the landmark was observed.
   * @param keypointIdx ID of keypoint corresponding to the landmark.
   * @return True if observation was present and successfully removed.
   */
  bool removeObservation(uint64_t landmarkId, uint64_t poseId,  size_t camIdx,
                         size_t keypointIdx) final;

  /**
   * @brief Applies the dropping/marginalization strategy according to the RSS'13/IJRR'14 paper.
   *        The new number of frames in the window will be numKeyframes+numImuFrames.
   * @param removedLandmarks Get the landmarks that were removed by this operation.
   * @return True if successful.
   */
  bool applyMarginalizationStrategy(okvis::MapPointVector &/*removedLandmarks*/) override {
    return true;
  }

  void optimize(size_t /*numIter*/, size_t /*numThreads*/ = 1, bool /*verbose*/ = false) override {}

  /**
   * @brief Set a time limit for the optimization process.
   * @param[in] timeLimit Time limit in seconds. If timeLimit < 0 the time limit is removed.
   * @param[in] minIterations minimum iterations the optimization process should do
   *            disregarding the time limit.
   * @return True if successful.
   */
  bool setOptimizationTimeLimit(double /*timeLimit*/, int /*minIterations*/) override {
    return true;
  }

  /// @name Getters for landmarks.
  ///\{
  /**
   * @brief Checks whether the landmark is added to the estimator.
   * @param landmarkId The ID.
   * @return True if added.
   */
  bool isLandmarkAdded(uint64_t landmarkId) const final {
    bool isAdded = landmarksMap_.find(landmarkId) != landmarksMap_.end();
//    OKVIS_ASSERT_TRUE_DBG(Exception, isAdded == mapPtr_->parameterBlockExists(landmarkId),
//                   "id="<<landmarkId<<" inconsistent. isAdded = " << isAdded);
    return isAdded;
  }

  /**
   * @brief Checks whether the landmark is initialized.
   * @param landmarkId The ID.
   * @return True if initialised.
   */
  bool isLandmarkInitialized(uint64_t landmarkId) const final;

  /**
   * @brief Get a specific landmark's first observation.
   *     thread unsafe, only call it when the estimator is locked by estimator_mutex_
   * @param[in]  landmarkId ID of desired landmark.
   * @param[out] kpId keypoint identifier of the first observation thanks to
   *     the fact that the MapPoint::observations is an ordered map
   * @return True if successful.
   */
  bool getLandmarkHeadObs(uint64_t landmarkId, okvis::KeypointIdentifier* kpId) const;

  /**
   * @brief Get a specific landmark.
   * @param[in]  landmarkId ID of desired landmark.
   * @param[out] mapPoint Landmark information, such as quality, coordinates etc.
   * @return True if successful.
   */
  bool getLandmark(uint64_t landmarkId, okvis::MapPoint& mapPoint) const final;

  bool getLandmark(uint64_t landmarkId, swift_vio::MapPoint& mapPoint) const;

  /**
   * @brief Get a copy of all the landmarks as a PointMap.
   * @param[out] landmarks The landmarks.
   * @return number of landmarks.
   */
  size_t getLandmarks(okvis::PointMap & landmarks) const;

  /**
   * @brief Get a copy of all the landmark in a MapPointVector. This is for legacy support.
   *        Use getLandmarks(okvis::PointMap&) if possible.
   * @param[out] landmarks A vector of all landmarks.
   * @see getLandmarks().
   * @return number of landmarks.
   */
  size_t getLandmarks(okvis::MapPointVector & landmarks) const override;

  size_t getCurrentlyObservedLandmarks(swift_vio::MapPointVector *landmarks) const;

  size_t getMarginalizedLandmarks(swift_vio::MapPointVector *landmarks) const;

  /// @brief Get the number of landmarks in the estimator
  /// \return The number of landmarks.
  size_t numLandmarks() const final {
    return landmarksMap_.size();
  }

  // return number of observations for a landmark in the landmark map
  size_t numObservations(uint64_t landmarkId) const;

  const okvis::MapPoint &getLandmarkUnsafe(uint64_t landmarkId) const;

  void frameLandmarkIds(uint64_t nframeId, size_t camId, std::vector<uint64_t> *landmarkIds) const {
    *landmarkIds = multiFramePtrMap_.at(nframeId)->getLandmarkIds(camId);
  }

  ///@}

  /// @name Operations on landmarks.
  ///\{
  /// @brief Set the homogeneous coordinates for a landmark.
  /// @param[in] landmarkId The landmark ID.
  /// @param[in] landmark Homogeneous coordinates of landmark in W-frame.
  /// @return True if successful.
  bool setLandmark(uint64_t landmarkId, const Eigen::Vector4d & landmark) final;

  /// @brief Set the landmark initialization state.
  /// @param[in] landmarkId The landmark ID.
  /// @param[in] initialized Whether or not initialised.
  void setLandmarkInitialized(uint64_t landmarkId, bool initialized) final;


  /**
   * @brief merge two landmarks: fuse the shorter feature track to the longer,
   * replace residuals of the shorter feature track,
   * also reset landmark ids for relevant keypoints in multiframes.
   */
  virtual uint64_t mergeTwoLandmarks(uint64_t lmIdA, uint64_t lmIdB);
  ///@}

  /// @name Getters
  ///\{
  /**
   * @brief Get a multiframe.
   * @param frameId ID of desired multiframe.
   * @return Shared pointer to multiframe.
   */
  okvis::MultiFramePtr multiFrame(uint64_t frameId) const final {
    OKVIS_ASSERT_TRUE_DBG(Exception, multiFramePtrMap_.find(frameId)!=multiFramePtrMap_.end(),
                       "Requested multi-frame does not exist in estimator.");
    return multiFramePtrMap_.at(frameId);
  }

  /**
   * @brief Get pose for a given pose ID.
   * @param[in]  poseId ID of desired pose.
   * @param[out] T_WS Homogeneous transformation of this pose.
   * @return True if successful.
   */
  bool get_T_WS(uint64_t poseId, okvis::kinematics::Transformation & T_WS) const final;

  // the following access the optimization graph, so are not very fast.
  // Feel free to implement caching for them...
  /**
   * @brief Get speeds and IMU biases for a given pose ID.
   * @warning This accesses the optimization graph, so not very fast.
   * @param[in]  poseId ID of pose to get speeds and biases for.
   * @param[in]  imuIdx index of IMU to get biases for. As only one IMU is supported this is always 0.
   * @param[out] speedAndBias Speed And bias requested.
   * @return True if successful.
   */
  bool getSpeedAndBias(uint64_t poseId, uint64_t imuIdx, okvis::SpeedAndBias & speedAndBias) const final;

  bool getSpeed(uint64_t poseId, Eigen::Vector3d &speed) const;

  bool getImuBiases(uint64_t poseId, size_t imuIdx, Eigen::Matrix<double, 6, 1> &biases) const;

  /**
   * @brief Get camera states for a given pose ID.
   * @warning This accesses the optimization graph, so not very fast.
   * @param[in]  poseId ID of pose to get camera state for.
   * @param[in]  cameraIdx index of camera to get state for.
   * @param[out] T_XCi Homogeneous transformation from a reference sensor frame to camera frame.
   * For the main camera, T_XC0 is always T_BC0. For other cameras, T_XCi can be T_BCi or T_C0Ci
   * depending on the camera extrinsic configuration.
   * @return True if successful.
   */
  bool getCameraSensorStates(uint64_t poseId, size_t cameraIdx,
                             okvis::kinematics::Transformation & T_XCi) const final;

  /// @brief Get the number of states/frames in the estimator.
  /// \return The number of frames.
  size_t numFrames() const final {
    return statesMap_.size();
  }

  /// @brief Get the ID of the current keyframe.
  /// \return The ID of the current keyframe.
  uint64_t currentKeyframeId() const;

  /**
   * @brief Get the ID of an older frame.
   * @param[in] age age of desired frame. 0 would be the newest frame added to the state.
   * @return ID of the desired frame or 0 if parameter age was out of range.
   */
  uint64_t frameIdByAge(size_t age) const;

  /// @brief Get the ID of the newest frame added to the state.
  /// \return The ID of the current frame.
  uint64_t currentFrameId() const final;

  okvis::Time currentFrameTimestamp() const;

  uint64_t oldestFrameId() const;

  okvis::Time oldestFrameTimestamp() const;

  size_t statesMapSize() const;

  ///@}

  virtual bool computeErrors(
      const okvis::kinematics::Transformation &ref_T_WS,
      const Eigen::Vector3d &ref_v_WS, const Eigen::Matrix<double, 6, 1> &biasRef,
      const okvis::ImuParameters &refImuParams,
      std::shared_ptr<const swift_vio::CameraRig> refCameraSystem,
      Eigen::VectorXd *errors) const;

  /**
   * @brief computeCovariance compute covariance by okvis marginalization module
   * which handles rank deficiency caused by low-disparity landmarks.
   * @param cov covariance of p_WS, q_WS, v_WS, b_g, b_a.
   * @return true if covariance is computed successfully, false otherwise.
   */
  virtual bool computeCovariance(Eigen::MatrixXd */*cov*/) const {
    return false;
  }

  bool printStatesAndStdevs(std::ostream& stream, const Eigen::MatrixXd *cov) const override;

  std::vector<std::string> variableLabels() const override {
    return std::vector<std::string>();
  }

  std::vector<std::string> perturbationLabels() const override {
    return std::vector<std::string>();
  }

  std::string headerLine(const std::string delimiter=" ") const final;

  std::string rmseHeaderLine(const std::string delimiter=" ") const final;

  void printNavStateAndBiases(std::ostream& stream, uint64_t poseId) const;

  virtual void printTrackLengthHistogram(std::ostream& /*stream*/) const {}

  /// @name Getters
  /// @{
  /**
   * @brief Get the timestamp for a particular frame.
   * @param[in] frameId ID of frame.
   * @return Timestamp of frame.
   */
  okvis::Time timestamp(uint64_t frameId) const final {
    return statesMap_.at(frameId).timestamp;
  }

  /**
   * @brief Checks if a particular frame is a keyframe.
   * @param[in] frameId ID of frame to check.
   * @return True if the frame is a keyframe.
   */
  bool isKeyframe(uint64_t frameId) const final {
    return statesMap_.at(frameId).isKeyframe;
  }

  /**
   * @brief Checks if a particular frame is still in the IMU window.
   * @param[in] frameId ID of frame to check.
   * @return True if the frame is in IMU window.
   */
  bool isInImuWindow(uint64_t frameId) const;
  /**
   * @brief get std. dev. of state for nav state (p,q,v), imu(bg ba), and optionally
   * imu augmented intrinsic parameters, camera extrinsic, intrinsic, td, tr.
   * @param stateStd
   * @return true if std. dev. of states are computed successfully.
   */
  virtual bool getStateStd(Eigen::Matrix<double, Eigen::Dynamic, 1> */*stateStd*/) const {
    return true;
  }

  virtual bool getDesiredStdevs(Eigen::VectorXd *desiredStdevs) const;

  /**
   * @brief minimal dim of parameters of camera of index camIdx.
   * @param camIdx cameraIndex.
   * @return minimal dim
   */
  size_t cameraParamsMinimalDim(size_t camIdx = 0) const {
    return (cameraNoiseParametersVec_.at(camIdx).isExtrinsicsFixed() ? 0 : cameraRig_.getMinimalExtrinsicDim(camIdx)) +
           (cameraNoiseParametersVec_.at(camIdx).isIntrinsicsFixed() ? 0 : cameraRig_.getMinimalProjectionIntrinsicDim(camIdx) +
           cameraRig_.getDistortionDim(camIdx)) + 2u;  // 2 for td and tr
  }

  /**
   * @brief gatherMapPointObservations gather all observations of a landmark
   * @param mp
   * @param obsDirections undistorted observation directions, [x, y, 1]
   * @param T_WCs  T_WB * T_BC for each observation.
   * @param obsInPixel observation in pixels
   * @param imageNoiseStd the std dev of image noise at both x and y direction.
   * twice as long as obsDirections.
   * @return
   */
  size_t gatherMapPointObservations(
      const MapPoint& mp,
      std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>*
          obsDirections,
      std::vector<okvis::kinematics::Transformation,
                  Eigen::aligned_allocator<okvis::kinematics::Transformation>>*
          T_CWs,
      std::vector<double>* imageNoiseStd) const;

  /**
   * @brief getCameraSensorExtrinsics get extrinsic parameters T_BC for a camera.
   * @param cameraIdx
   * @param T_BCi
   * @return
   */
  bool
  getCameraSensorExtrinsics(size_t cameraIdx,
                            okvis::kinematics::Transformation &T_BCi) const;

  /**
   * @brief get variable extrinsic parameters of camIdx.
   * @param[out] extrinsicParams variable extrinsic parameters refined by the estimator.
   * @param camIdx
   * @return
   */
  void getVariableCameraExtrinsics(
      uint64_t poseId, size_t camIdx,
      Eigen::Matrix<double, Eigen::Dynamic, 1> *extrinsicParams) const;

  /**
   * @brief get variable intrinsic parameters of camIdx.
   * @param cameraParams[out] variable intrinsic parameters refined by the
   * estimator, may include time offset and readout time.
   * @param camIdx
   * @return
   */
  virtual void getVariableCameraIntrinsics(uint64_t poseId, size_t camIdx,
      Eigen::Matrix<double, Eigen::Dynamic, 1> *cameraParams) const;

  /**
   * @brief getImuAugmentedStatesEstimate get the lastest estimate of IMU augmented params.
   * @param extraParams excluding biases.
   * @return
   */
  virtual void getImuAugmentedStatesEstimate(uint64_t poseId, size_t imuIdx,
      Eigen::Matrix<double, Eigen::Dynamic, 1>* extraParams) const;

  /**
   * @brief get the latest keyframe and its info which is used for loop detection.
   */
  bool getLoopQueryKeyframeMessage(
      okvis::MultiFramePtr multiFrame,
      std::shared_ptr<swift_vio::LoopQueryKeyframeMessage>* queryKeyframe) const;
  ///@}

  /// @name Setters
  ///@{
  /**
   * @brief Set pose for a given pose ID and the linearization point for position.
   * @warning Currently this function is only used at initialization stage, because
   * it accesses the optimization graph, so not very fast.
   * @param[in] poseId ID of the pose that should be changed.
   * @param[in] T_WS new homogeneous transformation.
   * @return True if successful.
   */
  bool set_T_WS(uint64_t poseId, const okvis::kinematics::Transformation & T_WS) final;

  /**
   * @brief Set the speeds and IMU biases for a given pose ID and the linearization point for velocity.
   * @warning Currently this function is only used at initialization stage because
   * it accesses the optimization graph, so not very fast.
   * @param[in] poseId ID of the pose to change corresponding speeds and biases for.
   * @param[in] imuIdx index of IMU to get biases for. As only one IMU is supported this is always 0.
   * @param[in] speedAndBias new speeds and biases.
   * @return True if successful.
   */
  bool setSpeedAndBias(uint64_t poseId, size_t imuIdx, const okvis::SpeedAndBias & speedAndBias) final;

  /// @brief Set whether a frame is a keyframe or not.
  /// @param[in] frameId The frame ID.
  /// @param[in] isKeyframe Whether or not keyrame.
  void setKeyframe(uint64_t frameId, bool isKeyframe) final {
    statesMap_.at(frameId).isKeyframe = isKeyframe;
  }

  /// @brief set ceres map
  /// @param[in] mapPtr The pointer to the okvis::ceres::Map.
  void setMap(std::shared_ptr<okvis::ceres::Map> mapPtr) final {
    mapPtr_ = mapPtr;
  }

  void setInitialNavState(const swift_vio::InitialNavState& rhs) {
     initialNavState_ = rhs;
  }

  void setEstimatorOptions(const EstimatorOptions& estimatorOptions) {
    estimatorOptions_ = estimatorOptions;
  }

  void setPointLandmarkOptions(const swift_vio::PointLandmarkOptions& plOptions) override {
    pointLandmarkOptions_ = plOptions;
  }

  void setPoseGraphOptions(const swift_vio::PoseGraphOptions& pgp) {
    poseGraphOptions_ = pgp;
  }

  /**
   * @brief setLoopFrameAndMatchesList
   * @warning This method is thread unsafe so do not call
   * setLoopFrameAndMatchesList() in one thread and call optimize() in another.
   * @param loopFrameAndMatchesList
   */
  void setLoopFrameAndMatchesList(
      const std::vector<std::shared_ptr<swift_vio::LoopFrameAndMatches>>&
          loopFrameAndMatchesList) {
    loopFrameAndMatchesList_ = loopFrameAndMatchesList;
  }

  void setInitializationStatus(InitializationStatus status) {
    initStatus_ = status;
  }
  ///@}

  template<class PARAMETER_BLOCK_T>
  bool getSensorStateEstimateAs(uint64_t poseId, int sensorIdx, int sensorType,
                                int stateType, typename PARAMETER_BLOCK_T::estimate_t & state) const;

  // Remove state parameter blocks and all of their related residuals
  okvis::Time removeState(uint64_t stateId);

  /// \brief SensorStates The sensor-internal states enumerated
  enum SensorStates
  {
    Camera = 0, ///< Camera
    Imu = 1, ///< IMU
    Position = 2, ///< Position, currently unused
    Gps = 3, ///< GPS, currently unused
    Magnetometer = 4, ///< Magnetometer, currently unused
    StaticPressure = 5, ///< Static pressure, currently unused
    DynamicPressure = 6 ///< Dynamic pressure, currently unused
  };

  /// \brief CameraSensorStates The camera-internal states enumerated
  enum CameraSensorStates
  {
    T_XCi = 0, ///< Extrinsics as T_SCi or T_C0Ci
    Intrinsics = 1, ///< Intrinsics, eg., for pinhole camera, fx ,fy, cx, cy
    Distortion = 2,  ///< Distortion coefficients, eg., for radial tangential distoriton
                 /// of pinhole cameras, k1, k2, p1, p2, [k3], this ordering is
                 /// OpenCV style
    TD = 3,      ///< time delay of the image timestamp with respect to the IMU
                 /// timescale, Raw t_Ci + t_d = t_Ci in IMU time,
    TR = 4       ///< t_r is the read out time of a whole frames of a rolling shutter camera
  };

  /// \brief ImuSensorStates The IMU-internal states enumerated
  /// \warning This is slightly inconsistent, since the velocity should be global.
  enum ImuSensorStates
  {
    SpeedAndBias = 0, ///< Speed and biases as v in S-frame, then b_g and b_a
    TG,               ///< gyro T_g, eg., SM or SMR_{GS}
    TS,               ///< g sensitivity
    TA                ///< accelerometer T_a, eg, SM or SMR_{AS}
  };

  // The window centered at a stateEpoch for retrieving the inertial data
  // which is used for propagating the camera pose to epochs in the window,
  // i.e., timestamps of observations in a rolling shutter image.
  // A value greater than (t_d + t_r)/2 is recommended.
  // Note camera observations in MSCKF will not occur at the latest frame.
  static const okvis::Duration half_window_;

  static const size_t kMainCameraIndex = 0u;

 protected:
  /**
   * @brief getOdometryConstraintsForKeyframe
   * @pre T_WB in queryKeyframe is set properly.
   * @param queryKeyframe
   * @return
   */
  virtual bool getOdometryConstraintsForKeyframe(
      std::shared_ptr<swift_vio::LoopQueryKeyframeMessage> /*queryKeyframe*/) const;

  /// \brief StateInfo This configures the state vector ordering
  struct StateInfo
  {
    /// \brief Constructor
    /// @param[in] id The Id.
    /// @param[in] isRequired Whether or not we require the state.
    /// @param[in] exists Whether or not this exists in the ceres problem.
    StateInfo(uint64_t id = 0, bool isRequired = true,
              bool exists = false, size_t startIndex = 0u)
        : id(id),
          isRequired(isRequired),
          exists(exists),
          startIndexInCov(startIndex)
    {
    }
    uint64_t id; ///< The ID.
    bool isRequired; ///< Whether or not we require the state.
    bool exists; ///< Whether or not this exists in the ceres problem.
    size_t startIndexInCov; ///< start index in the covariance matrix.
  };

  /// \brief GlobalStates The global states enumerated
  enum GlobalStates
  {
    T_WS = 0, ///< Pose.
    GravityDirection,
    MagneticZBias, ///< Magnetometer z-bias, currently unused
    Qff, ///< QFF (pressure at sea level), currently unused
    T_GW, ///< Alignment of global frame, currently unused
  };

  /// \brief PositionSensorStates, currently unused
  enum PositionSensorStates
  {
    T_PiW = 0,  ///< position sensor frame to world, currently unused
    PositionSensorB_t_BA = 1  ///< antenna offset, currently unused
  };

  /// \brief GpsSensorStates, currently unused
  enum GpsSensorStates
  {
    GpsB_t_BA = 0  ///< antenna offset, currently unused
  };

  /// \brief MagnetometerSensorStates, currently unused
  enum MagnetometerSensorStates
  {
    MagnetometerBias = 0 ///< currently unused
  };

  /// \brief GpsSensorStates, currently unused
  enum StaticPressureSensorStates
  {
    StaticPressureBias = 0 ///< currently unused
  };

  /// \brief GpsSensorStates, currently unused
  enum DynamicPressureSensorStates
  {
    DynamicPressureBias = 0 ///< currently unused
  };

  // getters
  bool getGlobalStateParameterBlockPtr(uint64_t poseId, int stateType,
                                    std::shared_ptr<ceres::ParameterBlock>& stateParameterBlockPtr) const;
  template<class PARAMETER_BLOCK_T>
  bool getGlobalStateParameterBlockAs(uint64_t poseId, int stateType,
                                      PARAMETER_BLOCK_T & stateParameterBlock) const;
  template<class PARAMETER_BLOCK_T>
  bool getGlobalStateEstimateAs(uint64_t poseId, int stateType,
                                typename PARAMETER_BLOCK_T::estimate_t & state) const;

  bool getSensorStateParameterBlockPtr(uint64_t poseId, int sensorIdx,
                                    int sensorType, int stateType,
                                    std::shared_ptr<ceres::ParameterBlock>& stateParameterBlockPtr) const;
  template<class PARAMETER_BLOCK_T>
  bool getSensorStateParameterBlockAs(uint64_t poseId, int sensorIdx,
                                      int sensorType, int stateType,
                                      PARAMETER_BLOCK_T & stateParameterBlock) const;

  // setters
  template<class PARAMETER_BLOCK_T>
  bool setGlobalStateEstimateAs(uint64_t poseId, int stateType,
                                const typename PARAMETER_BLOCK_T::estimate_t & state);
  template<class PARAMETER_BLOCK_T>
  bool setSensorStateEstimateAs(uint64_t poseId, int sensorIdx, int sensorType,
                                int stateType, const typename PARAMETER_BLOCK_T::estimate_t & state);

  // the following are just fixed-size containers for related parameterBlockIds:
  typedef std::array<StateInfo, 6> GlobalStatesContainer; ///< Container for global states.
  typedef std::vector<StateInfo> SpecificSensorStatesContainer;  ///< Container for sensor states. The dimension can vary from sensor to sensor...
  typedef std::array<std::vector<SpecificSensorStatesContainer>, 7> AllSensorStatesContainer; ///< Union of all sensor states.

  /// \brief States This summarizes all the possible states -- i.e. their ids:
  /// \f$ t_j = t_{j_0} - imageDelay + t_{d_j} \f$
  /// here \f$ t_{j_0} \f$ is the raw timestamp of image j,
  /// \f$ t_{d_j} \f$ is the current estimated time offset between the visual and
  /// inertial data, after correcting the initial time offset imageDelay.
  /// Therefore, \f$ t_{d_j} \f$ is usually 0 at the beginning of the algorithm.
  /// \f$ t_j \f$ is the timestamp of the state, remains constant after initialization.
  /// \f$ t_{f_{i,j}} = t_{j_0} - imageDelay + t_d + (v-N/2)t_r/N \f$ here \f$ t_d \f$ and \f$ t_r \f$ are the time
  /// offset and image readout time, \f$ t_{f_{i, j}} \f$ is the time feature i is observed in frame j.
  struct States {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    States() : isKeyframe(false), id(0) {}
    States(bool isKeyframe, uint64_t id, okvis::Time _timestamp)
        : isKeyframe(isKeyframe),
          id(id),
          timestamp(_timestamp) {}
    GlobalStatesContainer global;
    AllSensorStatesContainer sensors;
    bool isKeyframe;
    uint64_t id;
    const okvis::Time timestamp;         // t_j, fixed once initialized
    // linearization points of position r_WB and velocity v_WB.
    std::shared_ptr<Eigen::Matrix<double, 6, 1>> positionVelocityLin;
    // IMU measurements centering at state timestamp.
    std::shared_ptr<okvis::ImuMeasurementDeque> imuReadingWindow;
  };

  // the following keeps track of all the states at different time instances (key=poseId)
  std::map<uint64_t, States, std::less<uint64_t>,
           Eigen::aligned_allocator<std::pair<const uint64_t, States>>> statesMap_; ///< Buffer for currently considered states.
  std::map<uint64_t, okvis::MultiFramePtr> multiFramePtrMap_; ///< remember all needed okvis::MultiFrame.
  std::shared_ptr<okvis::ceres::Map> mapPtr_; ///< The underlying okvis::Map.

  // this is the reference pose
  uint64_t referencePoseId_; ///< The pose ID of the reference (currently not changing)

  // the following are updated after the optimization
  okvis::PointMap landmarksMap_; ///< Contains all the current landmarks (synched after optimisation).
  mutable std::mutex statesMutex_;  ///< Regulate access of landmarksMap_.

  // parameters
  std::vector<okvis::CameraNoiseParameters,
      Eigen::aligned_allocator<okvis::CameraNoiseParameters> > cameraNoiseParametersVec_; ///< Extrinsics parameters.
  std::vector<std::shared_ptr<okvis::ImuParameters>> imuParametersVec_; ///< IMU parameters.

  // An evolving camera rig to store the optimized camera
  // parameters and interface with the camera models.
  swift_vio::CameraRig cameraRig_;

  // An evolving imu rig to store the optimized imu parameters and
  // interface with the IMU models.
  swift_vio::ImuRig imuRig_;

  // sequential imu measurements covering states in the estimator
  swift_vio::BoundedImuDeque inertialMeasForStates_;

  // initial nav state, (position, orientation, and velocity), and their stds.
  swift_vio::InitialNavState initialNavState_;

  std::vector<std::shared_ptr<swift_vio::LoopFrameAndMatches>> loopFrameAndMatchesList_;

  EstimatorOptions estimatorOptions_;

  swift_vio::PointLandmarkOptions pointLandmarkOptions_; // see PointLandmarkModels.hpp

  swift_vio::PoseGraphOptions poseGraphOptions_;

  InitializationStatus initStatus_;

  okvis::MapPointVector marginalizedLandmarks_;
};
}  // namespace okvis
#include "okvis/implementation/EstimatorBase.hpp"

#endif /* INCLUDE_OKVIS_ESTIMATOR_BASE_HPP_ */
