#ifndef INCLUDE_OKVIS_KEYFRAME_FOR_LOOP_DETECTION_HPP_
#define INCLUDE_OKVIS_KEYFRAME_FOR_LOOP_DETECTION_HPP_
#include <vector>

#include <Eigen/Core>
#include <Eigen/StdVector>

#include <opencv2/core.hpp>

#include <okvis/assert_macros.hpp>
#include <okvis/cameras/CameraBase.hpp>
#include <okvis/cameras/NCameraSystem.hpp>

#include <okvis/class_macros.hpp>
#include <okvis/MultiFrame.hpp>
#include <okvis/Time.hpp>

#include <loop_closure/InverseTransformMultiplyJacobian.hpp>
#include <swift_vio/MultiFrame.hpp>

namespace swift_vio {
enum class PoseConstraintType {
  Odometry = 0,
  LoopClosure = 1,
};

class NeighborConstraintInDatabase {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  NeighborConstraintInDatabase();
  NeighborConstraintInDatabase(uint64_t id, okvis::Time stamp,
                               const okvis::kinematics::Transformation& T_BnBr,
                               PoseConstraintType type);
  ~NeighborConstraintInDatabase();

  uint64_t id_;
  okvis::Time stamp_;

  // Br is a body frame for reference, B body frame of this neighbor.
  okvis::kinematics::Transformation T_BBr_;

  PoseConstraintType type_;

  // square root info L' of the inverse of the covariance of the between factor
  // unwhitened/raw error due to measurement noise. LL' = \Lambda = inv(cov)
  // It depends on definitions of the between factor and errors of the
  // measurement. e.g., gtsam::BetweenFactor<Pose3>==log(T_z^{-1}T_x^{-1}T_y)
  // and error of T_z is defined by T_z = Pose3::Retraction(\hat{T}_z, \delta).
  Eigen::Matrix<double, 6, 6> squareRootInfo_;
};

class NeighborConstraintMessage {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  NeighborConstraintMessage();
  /**
   * @brief NeighborConstraintMessage
   * @param id
   * @param stamp
   * @param T_BnBr Bn the body frame associated with this neighbor,
   * Br the body frame associated with the reference frame of this neighbor.
   * @param T_WB pose of this neighbor.
   * @param type
   */
  NeighborConstraintMessage(
      uint64_t id, okvis::Time stamp,
      const okvis::kinematics::Transformation& T_BnBr,
      const okvis::kinematics::Transformation& T_WB,
      PoseConstraintType type = PoseConstraintType::Odometry);
  ~NeighborConstraintMessage();

  /**
   * @deprecated
   * @brief compute the covariance of error in $T_BnBr$ given the covariance of errors in $T_WBr$ and $T_WBn$
   * $T_BnBr = T_WBn^{-1} T_WBr$
   * The error(perturbation) of  $T_WBr$ $T_WBn$ and $T_BnBr$ are defined by
   * okvis::Transformation::oplus and ominus.
   * @param T_WBr
   * @param cov_T_WBr
   * @param[out] cov_T_BnBr cov for error in $T_BnBr$.
   * @return
   */
  void computeRelativePoseCovariance(
      const okvis::kinematics::Transformation& T_WBr,
      const Eigen::Matrix<double, 6, 6>& cov_T_WBr,
      Eigen::Matrix<double, 6, 6>* cov_T_BnBr);

  NeighborConstraintInDatabase core_;

  // variables used for computing the weighting covariance for the constraint
  // in the case of odometry pose constraint. In the case of loop constraint,
  // the covariance is computed inside PnP solver.
  okvis::kinematics::Transformation T_WB_; // pose of this neighbor keyframe.
  // cov of T_WB
  Eigen::Matrix<double, 6, 6> cov_T_WB_;
  // cov(T_WBr, T_WB)
  Eigen::Matrix<double, 6, 6> cov_T_WBr_T_WB_;
};

/**
 * @brief The KeyframeInDatabase class is stored keyframe info in loop closure keyframe database.
 */
class KeyframeInDatabase {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  DELETE_COPY_CONSTRUCTORS(KeyframeInDatabase);

  KeyframeInDatabase();

  KeyframeInDatabase(size_t dbowId, uint64_t vioId, okvis::Time stamp,
                     const okvis::kinematics::Transformation& vio_T_WB,
                     const Eigen::Matrix<double, 6, 6>& cov_T_WB);

  void setOdometryConstraints(
      const std::vector<std::shared_ptr<NeighborConstraintMessage>>&
          odometryConstraintList) {
    constraintList_.reserve(odometryConstraintList.size());
    for (auto constraint : odometryConstraintList) {
      std::shared_ptr<NeighborConstraintInDatabase> dbConstraint(
          new NeighborConstraintInDatabase(constraint->core_));
//      dbConstraint->squareRootInfo_ will be set later on.
      constraintList_.push_back(dbConstraint);
    }
  }

  inline std::vector<size_t> convertToLandmarkIndices() const {
    std::vector<size_t> landmarkIdForKeypoints(keypointList_.size(), 0u);
    size_t lmId = 0u;
    for (auto index : keypointIndexForLandmarkList_) {
      landmarkIdForKeypoints[index] = lmId;
      ++lmId;
    }
    return landmarkIdForKeypoints;
  }

  const std::vector<std::shared_ptr<NeighborConstraintInDatabase>>&
  constraintList() const {
    return constraintList_;
  }

  void addLoopConstraint(
      std::shared_ptr<NeighborConstraintInDatabase>& loopConstraint) {
    loopConstraintList_.push_back(loopConstraint);
  }

  const cv::Mat frontendDescriptors() const {
     return frontendDescriptors_;
  }

  const cv::Mat frontendDescriptorsWithLandmarks() const {
    return okvis::selectDescriptors(frontendDescriptors_,
                             keypointIndexForLandmarkList_);
  }

  const std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>&
  landmarkPositionList() const {
    return landmarkPositionList_;
  }

  const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>&
  keypointList() const {
    return keypointList_;
  }

  void setSquareRootInfo(size_t j,
                      const Eigen::Matrix<double, 6, 6>& squareRootInfo) {
    constraintList_.at(j)->squareRootInfo_ = squareRootInfo;
  }

  void setSquareRootInfoFromCovariance(size_t j,
                      const Eigen::Matrix<double, 6, 6>& covRawError);

  void setFrontendDescriptors(cv::Mat frontendDescriptors) {
    frontendDescriptors_ = frontendDescriptors;
  }

  void setLandmarkPositionList(
      const std::vector<Eigen::Vector4d,
                        Eigen::aligned_allocator<Eigen::Vector4d>>&
          landmarkPositionList) {
    landmarkPositionList_ = landmarkPositionList;
  }

  /**
   * @brief setKeypointIndexForLandmarkList
   * @param kpIndexForLandmarks each entry is index in keypoint list of a
   * keypoint corresponding to every landmark in landmark list.
   */
  void setKeypointIndexForLandmarkList(const std::vector<int>& kpIndexForLandmarks) {
    keypointIndexForLandmarkList_ = kpIndexForLandmarks;
  }

  void setKeypointList(
      const std::vector<Eigen::Vector3f,
                        Eigen::aligned_allocator<Eigen::Vector3f>>&
          keypointList) {
    keypointList_ = keypointList;
  }

 public:
  size_t dbowId_; ///< id used in DBoW vocabulary. Determined by the size of the KeyframeInDatabase list.
  uint64_t id_; ///< frontend keyframe id.
  okvis::Time stamp_;

  const okvis::kinematics::Transformation vio_T_WB_; ///< original vio estimated T_WB;
  const Eigen::Matrix<double, 6, 6> cov_vio_T_WB_;  ///< cov of $[\delta p, \delta \theta]$ provided by VIO.

 private:
  ///< If we do not construct the pose graph solver from scratches once in a
  /// while as in VINS Mono, then we do not need the constraint list.
  std::vector<std::shared_ptr<NeighborConstraintInDatabase>> constraintList_; ///< odometry constraints.
  std::vector<std::shared_ptr<NeighborConstraintInDatabase>> loopConstraintList_; ///< loop constraints.
  // The below variables are used to find correspondence between a loop frame
  // and a query frame and estimate the relative pose.
  cv::Mat frontendDescriptors_; ///< descriptors for every keypoint from VIO frontend. #columns is the descriptor size, #rows is for landmarks.
  std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>
      landmarkPositionList_;  ///< landmark positions expressed in the body frame of this keyframe passed in by a VIO estimator.
  std::vector<int> keypointIndexForLandmarkList_; ///< index in keypointList of keypoints associated with landmarks.
  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>
      keypointList_; ///< locations and size of every keypoint in left camera.
};

constexpr size_t kQueryCameraIndex = 0u;

/**
  * @brief setNFrame copy essential parts from frontend NFrame to avoid
  * read/write at the same time by VIO estimator and loop closure module.
  */
template <typename MultiFrameT>
void setNFrame(std::shared_ptr<const MultiFrameT> multiframe, std::shared_ptr<okvis::MultiFrame> nframe) {
  // shallow copy one image.
  nframe->setImage(kQueryCameraIndex, multiframe->image(kQueryCameraIndex));
  nframe->setTimestamp(multiframe->timestamp());
  nframe->setTimestamp(kQueryCameraIndex, multiframe->timestamp(kQueryCameraIndex));
  nframe->resetKeypoints(kQueryCameraIndex,
                          multiframe->getKeypoints(kQueryCameraIndex));
  cv::Mat descriptors;
  cv::Mat rawDescriptors = multiframe->getDescriptors(kQueryCameraIndex);
  // With motion blurred images, rawDescriptors may be empty.
  rawDescriptors.copyTo(descriptors);
  nframe->resetDescriptors(kQueryCameraIndex, descriptors);
}

/**
 * @brief The LoopQueryKeyframeMessage class
 * The internal C++ keyframe message for loop closure.
 * Only one frame out of nframe will be used for querying keyframe database and
 * computing loop constraint. As a result, from the NCameraSystem, we only
 * need the camera intrinsic parameters, but not the extrinsic parameters.
 * We may reset the NCameraSystem for nframe_ when intrinsic parameters are
 * estimated online by the estimator. This should not disturb the frontend
 * feature matching which locks the estimator in matching features.
 */
class LoopQueryKeyframeMessage {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  DELETE_COPY_CONSTRUCTORS(LoopQueryKeyframeMessage);

  LoopQueryKeyframeMessage() {}

  ~LoopQueryKeyframeMessage() {}

  LoopQueryKeyframeMessage(uint64_t id, okvis::Time stamp,
                         const okvis::kinematics::Transformation& T_WB,
                         const std::shared_ptr<const okvis::MultiFrame>& multiframe)
    : id_(id), stamp_(stamp), T_WB_(T_WB) {
    std::shared_ptr<okvis::MultiFrame> nframe(new okvis::MultiFrame(
        multiframe->cameraSystem(), multiframe->timestamp(), multiframe->id()));
    setNFrame(multiframe, nframe);
    nframe_ = nframe;
  }

  LoopQueryKeyframeMessage(uint64_t id, okvis::Time stamp,
                         const okvis::kinematics::Transformation& T_WB,
                         const std::shared_ptr<const swift_vio::MultiFrame>& multiframe,
                         const okvis::cameras::NCameraSystem & cameraSystem)
    : id_(id), stamp_(stamp), T_WB_(T_WB) {
    std::shared_ptr<okvis::MultiFrame> nframe(new okvis::MultiFrame(
        cameraSystem, multiframe->timestamp(), multiframe->id()));
    setNFrame(multiframe, nframe);
    nframe_ = nframe;
  }

  std::shared_ptr<KeyframeInDatabase> toKeyframeInDatabase(size_t dbowId) const {
    std::shared_ptr<KeyframeInDatabase> keyframeInDB(
        new KeyframeInDatabase(dbowId, id_, stamp_, T_WB_, cov_T_WB_));
    keyframeInDB->setOdometryConstraints(odometryConstraintList_);
    keyframeInDB->setLandmarkPositionList(landmarkPositionList_);
    keyframeInDB->setFrontendDescriptors(getFrontendDescriptors());
    keyframeInDB->setKeypointList(getFrontendKeypoints());
    keyframeInDB->setKeypointIndexForLandmarkList(keypointIndexForLandmarkList_);
    return keyframeInDB;
  }

  bool hasValidCovariance() const {
    return cov_T_WB_(0, 0) > 1e-7;
  }

  const Eigen::Matrix<double, 6, 6>& getCovariance() const {
    return cov_T_WB_;
  }

  void setCovariance(const Eigen::Matrix<double, 6, 6>& cov_T_WB) {
    cov_T_WB_ = cov_T_WB;
  }

  void setZeroCovariance() {
    cov_T_WB_.setZero();
  }

  std::shared_ptr<const okvis::MultiFrame> NFrame() const {
    return nframe_;
  }

  const cv::Mat queryImage() const {
    return nframe_->image(kQueryCameraIndex);
  }

  std::shared_ptr<const okvis::cameras::CameraBase> cameraGeometry() const {
    return nframe_->geometry(kQueryCameraIndex);
  }

  const std::vector<std::shared_ptr<NeighborConstraintMessage>>&
  odometryConstraintList() const {
    return odometryConstraintList_;
  }

  const std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>&
  landmarkPositionList() const {
    return landmarkPositionList_;
  }

  cv::Mat getFrontendDescriptors() const {
    // deep copy is unneeded because nframe's descriptors are newly allocated.
    return nframe_->getDescriptors(kQueryCameraIndex);
  }

  std::vector<okvis::KeypointReduced, Eigen::aligned_allocator<okvis::KeypointReduced>>
  getFrontendKeypoints() const {
    return nframe_->copyKeypoints(kQueryCameraIndex);
  }

  ///< \brief get all descriptors for a view in nframe.
  cv::Mat getDescriptors() const {
    return nframe_->getDescriptors(kQueryCameraIndex);
  }

  std::vector<std::shared_ptr<NeighborConstraintMessage>>&
  odometryConstraintListMutable() {
    return odometryConstraintList_;
  }

  const std::vector<int>& keypointIndexForLandmarkList() const {
    return keypointIndexForLandmarkList_;
  }

  std::vector<int>& keypointIndexForLandmarkListMutable() {
    return keypointIndexForLandmarkList_;
  }

  std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>&
  landmarkPositionListMutable() {
    return landmarkPositionList_;
  }

  void setLandmarkPositionList(
      const std::vector<Eigen::Vector4d,
                        Eigen::aligned_allocator<Eigen::Vector4d>>&
          landmarkPositionList) {
    landmarkPositionList_ = landmarkPositionList;
  }

  uint64_t id_;
  okvis::Time stamp_;
  okvis::kinematics::Transformation T_WB_;

 private:
  Eigen::Matrix<double, 6, 6>
      cov_T_WB_;  ///< cov of $[\delta p, \delta \theta]$. An estimator
                  ///< that does not provide covariance for poses should zero
                  ///< cov_T_WB_.

  /// @warn Do not hold on to nframe_ which has many images. 
  /// We use okvis::MultiFrame here for FrameNoncentralAbsoluteAdapter which is used for geometric verification.
  std::shared_ptr<const okvis::MultiFrame> nframe_; ///< nframe contains the list of keypoints for each subframe, and the camera system info.

  std::vector<std::shared_ptr<NeighborConstraintMessage>> odometryConstraintList_; ///< The most adjacent neighbor is at the front.

  std::vector<int> keypointIndexForLandmarkList_; ///< Index of the keypoints with landmark positions.
  std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>
      landmarkPositionList_;  ///< landmark positions expressed in the body frame of this keyframe.
}; // LoopQueryKeyframeMessage

struct PgoResult {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  okvis::Time stamp_;
  okvis::kinematics::Transformation T_WB_;
};
}  // namespace swift_vio
#endif  // INCLUDE_OKVIS_KEYFRAME_FOR_LOOP_DETECTION_HPP_
