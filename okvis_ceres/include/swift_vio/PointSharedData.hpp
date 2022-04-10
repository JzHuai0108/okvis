#ifndef INCLUDE_SWIFT_VIO_POINT_SHARED_DATA_HPP_
#define INCLUDE_SWIFT_VIO_POINT_SHARED_DATA_HPP_

#include <memory>
#include <unordered_map>
#include <Eigen/StdVector>

#include <okvis/FrameTypedefs.hpp>
#include <okvis/Measurements.hpp>
#include <okvis/Parameters.hpp>
#include <okvis/ceres/PoseParameterBlock.hpp>

#include <swift_vio/CameraIdentifier.h>
#include <swift_vio/ceres/EuclideanParamBlockSized.hpp>
#include <swift_vio/ceres/EuclideanParamBlockSizedLin.hpp>
#include <swift_vio/VectorOperations.hpp>

namespace swift_vio {
// The state info for one keypoint relevant to computing the pose (T_WB) and
// (linear and angular) velocity (v_WB, omega_WB_B) at keypoint observation epoch.
struct StateInfoForOneKeypoint {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    StateInfoForOneKeypoint() {

    }

    StateInfoForOneKeypoint(
        uint64_t _frameId, size_t _camIdx,
        std::shared_ptr<const okvis::ceres::PoseParameterBlock> T_WB_ptr,
        double _normalizedRow, okvis::Time imageStamp)
        : frameId(_frameId),
          cameraId(_camIdx),
          T_WBj_ptr(T_WB_ptr),
          normalizedRow(_normalizedRow),
          imageTimestamp(imageStamp) {}

    uint64_t frameId;
    size_t cameraId;
    std::shared_ptr<const okvis::ceres::PoseParameterBlock> T_WBj_ptr;
    std::shared_ptr<const okvis::ceres::SpeedParameterBlock> v_WBj_ptr;
    std::shared_ptr<const okvis::ceres::BiasParameterBlock> biasPtr;
    // IMU measurements covering the state epoch.
    std::shared_ptr<const okvis::ImuMeasurementDeque> imuMeasurementPtr;
    okvis::Time stateEpoch;
    double normalizedRow; // v / imageHeight - 0.5.
    okvis::Time imageTimestamp; // raw image frame timestamp may be different for cameras in NFrame.

    // Pose of the body frame in the world frame at the feature observation epoch.
    // It should be computed with IMU propagation for RS cameras.
    okvis::kinematics::Transformation T_WBtij;
    Eigen::Vector3d v_WBtij;
    Eigen::Vector3d omega_Btij;
    okvis::kinematics::Transformation  T_WBtij_lin;
    Eigen::Vector3d v_WBtij_lin;
};

enum class PointSharedDataState {
  Barebones = 0,
  ImuInfoReady = 1,
  NavStateReady = 2,
  NavStateForJacReady = 3,
};

// Data shared by observations of a point landmark in computing Jacobians
// relative to pose (T_WB) and velocity (v_WB) and camera time parameters.
// The data of the class members may be updated in ceres EvaluationCallback.
class PointSharedData {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef std::vector<StateInfoForOneKeypoint,
                      Eigen::aligned_allocator<StateInfoForOneKeypoint>>
      StateInfoForObservationsType;

  PointSharedData() : status_(PointSharedDataState::Barebones) {}

  void addKeypointObservation(
      const okvis::KeypointIdentifier& kpi,
      std::shared_ptr<const okvis::ceres::PoseParameterBlock> T_WBj_ptr,
      double normalizedRow, okvis::Time imageTimestamp) {
    stateInfoForObservations_.emplace_back(kpi.frameId, kpi.cameraIndex,
                                           T_WBj_ptr, normalizedRow, imageTimestamp);
  }

  /// @name Setters for data for IMU propagation.
  /// @{
  /**
   * @brief setVelocityAndBiasParameterBlockPtr
   * @deprecated
   */
  void setVelocityAndBiasParameterBlockPtr(
      int /*index*/,
      std::shared_ptr<const okvis::ceres::ParameterBlock> /*speedAndBiasPtr*/) {
    OKVIS_ASSERT_TRUE(std::runtime_error, false, "This function is broken!");
  }

  void setVelocityAndBiasParameterBlockPtr(
      int index,
      std::shared_ptr<const okvis::ceres::SpeedParameterBlock> speedPtr,
      std::shared_ptr<const okvis::ceres::BiasParameterBlock> biasPtr) {
    stateInfoForObservations_[index].v_WBj_ptr = speedPtr;
    stateInfoForObservations_[index].biasPtr = biasPtr;
  }

  void setImuInfo(
      int index, const okvis::Time stateEpoch,
      std::shared_ptr<const okvis::ImuMeasurementDeque> imuMeasurements) {
    stateInfoForObservations_[index].stateEpoch = stateEpoch;
    stateInfoForObservations_[index].imuMeasurementPtr = imuMeasurements;
  }

  void setImuAugmentedParameterPtrs(
      const std::vector<std::shared_ptr<const okvis::ceres::ParameterBlock>>&
          imuAugmentedParamBlockPtrs,
      std::shared_ptr<const okvis::ImuParameters> imuParams) {
    imuAugmentedParamBlockPtrs_ = imuAugmentedParamBlockPtrs;
    imuParameters_ = imuParams;
  }

  void setCameraTimeParameterPtrs(
      const std::vector<std::shared_ptr<const okvis::ceres::ParameterBlock>>&
          tdParamBlockPtrs,
      const std::vector<std::shared_ptr<const okvis::ceres::ParameterBlock>>&
          trParamBlockPtrs) {
    tdParamBlockPtrs_ = tdParamBlockPtrs;
    trParamBlockPtrs_ = trParamBlockPtrs;
    status_ = PointSharedDataState::ImuInfoReady;
  }
  /// @}

  const std::vector<std::shared_ptr<const okvis::ceres::ParameterBlock>> &
  imuAugmentedParameterPtrs() const {
    return imuAugmentedParamBlockPtrs_;
  }

  /// @name functions for IMU propagation.
  /// @{
  /**
   * @brief computePoseAndVelocityAtObservation.
   *     for feature i, estimate $p_B^G(t_{f_i})$, $R_B^G(t_{f_i})$,
   *     $v_B^G(t_{f_i})$, and $\omega_{GB}^B(t_{f_i})$ with the corresponding
   *     states' LATEST ESTIMATES and imu measurements.
   * @warning Call this function after setImuAugmentedParameterPtrs().
   */
  void computePoseAndVelocityAtObservation();

  /**
   * @brief computePoseAndVelocityForJacobians
   * @warning Only call this function after
   * computePoseAndVelocityAtObservation() has finished.
   */
  void computePoseAndVelocityForJacobians();
  /// @}

  /// @name Functions for anchors.
  /// @{
  void setAnchors(const std::vector<AnchorFrameIdentifier>& anchorIds) {
    anchorIds_ = anchorIds;    
  }

  const std::vector<AnchorFrameIdentifier>& anchorIds() const {
    return anchorIds_;
  }

  const CameraIdentifier anchorCameraId(size_t anchorId) const {
    return CameraIdentifier(anchorIds_.at(anchorId).frameId_,
                            anchorIds_.at(anchorId).cameraIndex_);
  }

  /**
   * @brief Get index of observations from the anchor frames in the observation sequence.
   * @warning This only support a monocular camera.
   * @return
   */
  std::vector<int> anchorObservationIds() const;

  okvis::kinematics::Transformation T_WB_mainAnchorStateEpoch() const {
    const StateInfoForOneKeypoint& mainAnchorItem =
        stateInfoForObservations_.at(anchorIds_[0].observationIndex_);
    return mainAnchorItem.T_WBj_ptr->estimate();
  }

  okvis::kinematics::Transformation T_WB_mainAnchorStateEpochForJacobian() const {
    const StateInfoForOneKeypoint& mainAnchorItem =
        stateInfoForObservations_.at(anchorIds_[0].observationIndex_);
   return mainAnchorItem.T_WBj_ptr->linPoint();
  }
  /// @}

  /// @name functions for managing the main stateInfo list.
  /// @{
  StateInfoForObservationsType::iterator begin() {
    return stateInfoForObservations_.begin();
  }

  StateInfoForObservationsType::iterator end() {
      return stateInfoForObservations_.end();
  }

  void removeBadObservations(const std::vector<bool>& projectStatus) {
      removeUnsetMatrices<StateInfoForOneKeypoint>(&stateInfoForObservations_, projectStatus);
  }

  void removeExtraObservations(const std::vector<uint64_t>& orderedSelectedFrameIds);

  /**
   * @brief removeExtraObservations
   * @deprecated
   * @warning orderedSelectedFrameIds must be a subsets of stateInfoForObservations_
   * @param orderedSelectedFrameIds
   * @param imageNoise2dStdList
   */
  void removeExtraObservations(const std::vector<uint64_t>& orderedSelectedFrameIds,
                               std::vector<double>* imageNoise2dStdList);

  void removeExtraObservationsLegacy(
      const std::vector<uint64_t>& orderedSelectedFrameIds,
      std::vector<double>* imageNoise2dStdList);
  /// @}

  /// @name Getters for frameIds.
  /// @{
  size_t numObservations() const {
      return stateInfoForObservations_.size();
  }

  std::vector<std::pair<uint64_t, size_t>> frameIds() const {
    std::vector<std::pair<uint64_t, size_t>> frameIds;
    frameIds.reserve(stateInfoForObservations_.size());
    for (auto item : stateInfoForObservations_) {
      frameIds.emplace_back(item.frameId, item.cameraId);
    }
    return frameIds;
  }

  uint64_t frameId(int index) const {
    return stateInfoForObservations_[index].frameId;
  }

  uint64_t lastFrameId() const {
    return stateInfoForObservations_.back().frameId;
  }
  /// @}

  /// @name Getters
  /// @{
  double normalizedFeatureTime(int observationIndex) const {
    return normalizedFeatureTime(stateInfoForObservations_[observationIndex]);
  }

  double normalizedFeatureTime(const StateInfoForOneKeypoint& item) const {
    size_t cameraIdx = item.cameraId;
    return tdParamBlockPtrs_[cameraIdx]->parameters()[0] +
           trParamBlockPtrs_[cameraIdx]->parameters()[0] * item.normalizedRow +
        (item.imageTimestamp - item.stateEpoch).toSec();
  }

  size_t cameraIndex(size_t observationIndex) const {
    return stateInfoForObservations_[observationIndex].cameraId;
  }

  double normalizedRow(int index) const {
    return stateInfoForObservations_[index].normalizedRow;
  }

  okvis::Time imageTime(int index) const {
    return stateInfoForObservations_[index].imageTimestamp;
  }

  std::vector<okvis::kinematics::Transformation,
              Eigen::aligned_allocator<okvis::kinematics::Transformation>>
  poseAtObservationList() const {
    std::vector<okvis::kinematics::Transformation,
                Eigen::aligned_allocator<okvis::kinematics::Transformation>>
        T_WBtij_list;
    T_WBtij_list.reserve(stateInfoForObservations_.size());
    for (auto item : stateInfoForObservations_) {
      T_WBtij_list.push_back(item.T_WBtij);
    }
    return T_WBtij_list;
  }

  std::vector<okvis::kinematics::Transformation,
              Eigen::aligned_allocator<okvis::kinematics::Transformation>>
  poseAtFrameList() const {
    std::vector<okvis::kinematics::Transformation,
                Eigen::aligned_allocator<okvis::kinematics::Transformation>>
        T_WBj_list;
    T_WBj_list.reserve(stateInfoForObservations_.size());
    for (auto item : stateInfoForObservations_) {
      T_WBj_list.push_back(item.T_WBj_ptr->estimate());
    }
    return T_WBj_list;
  }

  std::vector<size_t> cameraIndexList() const {
    std::vector<size_t> camIndices;
    camIndices.reserve(stateInfoForObservations_.size());
    for (auto item : stateInfoForObservations_) {
      camIndices.push_back(item.cameraId);
    }
    return camIndices;
  }

  void poseAtObservation(int index, okvis::kinematics::Transformation* T_WBtij) const {
    *T_WBtij = stateInfoForObservations_[index].T_WBtij;
  }

  okvis::kinematics::Transformation T_WBtij(int index) const {
    return stateInfoForObservations_[index].T_WBtij;
  }

  Eigen::Vector3d omega_Btij(int index) const {
    return stateInfoForObservations_[index].omega_Btij;
  }

  Eigen::Vector3d v_WBtij(int index) const {
    return stateInfoForObservations_[index].v_WBtij;
  }

  okvis::kinematics::Transformation T_WBtij_ForJacobian(int index) const {
    return stateInfoForObservations_[index].T_WBtij_lin;
  }

  Eigen::Matrix3d Phi_pq_feature(int observationIndex) const {
    const StateInfoForOneKeypoint& item =
        stateInfoForObservations_[observationIndex];
    double relFeatureTime = normalizedFeatureTime(item);
    Eigen::Vector3d gW = imuParameters_->gravity();
    Eigen::Vector3d dr =
        -(item.T_WBtij_lin.r() - item.T_WBj_ptr->positionLinPoint() -
          item.v_WBj_ptr->linPoint() * relFeatureTime -
          0.5 * gW * relFeatureTime * relFeatureTime);
    return okvis::kinematics::crossMx(dr);
  }

  Eigen::Vector3d v_WBtij_ForJacobian(int index) const {
    return stateInfoForObservations_[index].v_WBtij_lin;
  }

  Eigen::Matrix<double, 6, 1> posVelLinPoint(int index) const {
    Eigen::Matrix<double, 6, 1> linPoint;
    linPoint << stateInfoForObservations_[index].T_WBj_ptr->positionLinPoint(),
        stateInfoForObservations_[index].v_WBj_ptr->linPoint();
    return linPoint;
  }

  okvis::kinematics::Transformation poseLinPoint(int index) const {
    return stateInfoForObservations_[index].T_WBj_ptr->linPoint();
  }

  Eigen::Vector3d velLinPoint(int index) const {
    return stateInfoForObservations_[index].v_WBj_ptr->linPoint();
  }

  PointSharedDataState status() const {
    return status_;
  }

  double gravityNorm() const {
    return imuParameters_->g;
  }
  /// @}

  /// @name Getters for parameter blocks
  /// @{
  std::shared_ptr<const okvis::ceres::PoseParameterBlock> poseParameterBlockPtr(
      int observationIndex) const;

  std::shared_ptr<const okvis::ceres::ParameterBlock>
  speedAndBiasParameterBlockPtr(int observationIndex) const {
    OKVIS_ASSERT_TRUE(std::runtime_error, false, "This function is broken!");
    return stateInfoForObservations_.at(observationIndex).v_WBj_ptr;
  }

  std::shared_ptr<const okvis::ceres::ParameterBlock>
  speedParameterBlockPtr(int observationIndex) const {
    return stateInfoForObservations_.at(observationIndex).v_WBj_ptr;
  }

  std::shared_ptr<const okvis::ceres::ParameterBlock>
  biasParameterBlockPtr(int observationIndex) const {
    return stateInfoForObservations_.at(observationIndex).biasPtr;
  }

  std::shared_ptr<const okvis::ceres::ParameterBlock>
  cameraTimeDelayParameterBlockPtr(size_t cameraIndex) const {
    return tdParamBlockPtrs_[cameraIndex];
  }

  std::shared_ptr<const okvis::ceres::ParameterBlock>
  frameReadoutTimeParameterBlockPtr(size_t cameraIndex) const {
    return trParamBlockPtrs_[cameraIndex];
  }

  size_t imuIdx() const {
    return imuIdx_;
  }

  /// @}

 private:
  // The items of stateInfoForObservations_ are added in an ordered manner
  // by sequentially examining the ordered elements of MapPoint.observations.
  std::vector<StateInfoForOneKeypoint,
              Eigen::aligned_allocator<StateInfoForOneKeypoint>>
      stateInfoForObservations_;

  std::vector<AnchorFrameIdentifier> anchorIds_;

  std::vector<std::shared_ptr<const okvis::ceres::ParameterBlock>>
      tdParamBlockPtrs_;
  std::vector<std::shared_ptr<const okvis::ceres::ParameterBlock>>
      trParamBlockPtrs_;
  std::vector<std::shared_ptr<const okvis::ceres::ParameterBlock>>
      imuAugmentedParamBlockPtrs_;
  std::shared_ptr<const okvis::ImuParameters> imuParameters_;
  size_t imuIdx_;
  // The structure of sharedJacobians is determined by an external cameraObservationModelId.
  std::vector<
      Eigen::Matrix<double, -1, -1, Eigen::RowMajor>,
      Eigen::aligned_allocator<Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>>
      sharedJacobians_;
  PointSharedDataState status_;
};
} // namespace swift_vio

#endif // INCLUDE_SWIFT_VIO_POINT_SHARED_DATA_HPP_
