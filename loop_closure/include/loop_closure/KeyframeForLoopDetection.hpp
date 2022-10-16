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

  KeyframeInDatabase(uint64_t vioId, okvis::Time stamp,
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

  void addLoopConstraint(
      std::shared_ptr<NeighborConstraintInDatabase>& loopConstraint) {
    loopConstraintList_.push_back(loopConstraint);
  }

  const cv::Mat frontendDescriptorsWithLandmarks(size_t camId) const {
    return okvis::selectDescriptors(nFrameWithoutImages_->getDescriptors(camId),
                                    keypointIndexForLandmarkList_.at(camId));
  }

  void setSquareRootInfo(size_t j,
                      const Eigen::Matrix<double, 6, 6>& squareRootInfo) {
    constraintList_.at(j)->squareRootInfo_ = squareRootInfo;
  }

  void setSquareRootInfoFromCovariance(size_t j,
                      const Eigen::Matrix<double, 6, 6>& covRawError);

 public:
  uint64_t id_; ///< frontend keyframe id.
  okvis::Time stamp_;

  const okvis::kinematics::Transformation vio_T_WB_; ///< original vio estimated T_WB;
  const Eigen::Matrix<double, 6, 6> cov_vio_T_WB_;  ///< cov of $[\delta p, \delta \theta]$ provided by VIO.

  ///< If we do not construct the pose graph solver from scratches once in a
  /// while as in VINS Mono, then we do not need the constraint list.
  std::vector<std::shared_ptr<NeighborConstraintInDatabase>> constraintList_; ///< odometry constraints.
  std::vector<std::shared_ptr<NeighborConstraintInDatabase>> loopConstraintList_; ///< loop constraints.

  std::vector<size_t> dbowIds_; ///< dbow descriptor for each frame.
  std::shared_ptr<const swift_vio::MultiFrame> nFrameWithoutImages_; ///< nframe contains the list of keypoints and descriptors for each frame.

  // The below variables are used to find correspondence between a loop frame
  // and a query frame and estimate the relative pose.
  std::vector<std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>>
      landmarkPositionList_;  ///< landmark positions expressed in the body frame of this keyframe passed in by a VIO estimator.
  std::vector<std::vector<int>> keypointIndexForLandmarkList_;
};

/**
  * @brief copyNFrame shallow copy essential parts from frontend NFrame.
  * @param selectedCamIds Ids of cameras within the multiframe to be passed to queryNFrame.
  */
template <typename MultiFrameT>
void copyNFrame(std::shared_ptr<MultiFrameT> multiframe, const std::vector<size_t>& selectedCamIds,
    std::shared_ptr<swift_vio::MultiFrame> queryNFrame, bool copyImages) {
  queryNFrame->setTimestamp(multiframe->timestamp());
  for (size_t i = 0u; i < selectedCamIds.size(); ++i) {
    size_t origId = selectedCamIds[i];
    // shallow copy iamges
    if (copyImages)
      queryNFrame->setImage(i, multiframe->image(origId));
    queryNFrame->setTimestamp(i, multiframe->timestamp(origId));
    queryNFrame->resetKeypoints(i, multiframe->getKeypoints(origId));
    // With motion blurred images, rawDescriptors may be empty.
    queryNFrame->resetDescriptors(i, multiframe->getDescriptors(origId));
  }
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
                         const std::shared_ptr<const okvis::MultiFrame>& multiframe,
                         const std::vector<size_t>& selectedCamIds)
    : id_(id), stamp_(stamp), T_WB_(T_WB), use_uniform_cov_(true) {
    cameraSystem_ = multiframe->cameraSystem().selectedNCameraSystem(selectedCamIds);
    std::shared_ptr<swift_vio::MultiFrame> nframe(new swift_vio::MultiFrame(
        selectedCamIds.size(), multiframe->timestamp(), multiframe->id()));
    copyNFrame(multiframe, selectedCamIds, nframe, true);
    nframe_ = nframe;
  }

  LoopQueryKeyframeMessage(uint64_t id, okvis::Time stamp,
                         const okvis::kinematics::Transformation& T_WB,
                         const std::shared_ptr<const swift_vio::MultiFrame>& multiframe,
                         const std::vector<size_t>& selectedCamIds,
                         const okvis::cameras::NCameraSystem& cameraSystem)
    : id_(id), stamp_(stamp), T_WB_(T_WB), use_uniform_cov_(true) {
    cameraSystem_ = cameraSystem.selectedNCameraSystem(selectedCamIds);
    std::shared_ptr<swift_vio::MultiFrame> nframe(new swift_vio::MultiFrame(
        selectedCamIds.size(), multiframe->timestamp(), multiframe->id()));
    copyNFrame(multiframe, selectedCamIds, nframe, true);
    nframe_ = nframe;
  }

  std::shared_ptr<KeyframeInDatabase> toKeyframeInDatabase() const {
    std::shared_ptr<KeyframeInDatabase> keyframeInDB(
        new KeyframeInDatabase(id_, stamp_, T_WB_, cov_T_WB_));
    keyframeInDB->setOdometryConstraints(odometryConstraintList_);
    keyframeInDB->landmarkPositionList_ = landmarkPositionList_;
    keyframeInDB->keypointIndexForLandmarkList_ = keypointIndexForLandmarkList_;
    std::shared_ptr<swift_vio::MultiFrame> nframe(new swift_vio::MultiFrame(
        cameraSystem_.numCameras(), nframe_->timestamp(), nframe_->id()));
    size_t numcameras = nframe_->numFrames();
    std::vector<size_t> selectedCamIds;
    selectedCamIds.resize(numcameras);
    for (size_t i = 0; i < numcameras; ++i) {
      selectedCamIds[i] = i;
    }
    copyNFrame(nframe_, selectedCamIds, nframe, false);
    keyframeInDB->nFrameWithoutImages_ = nframe;
    return keyframeInDB;
  }

  bool useUniformCov() const {
    return use_uniform_cov_;
  }

  const Eigen::Matrix<double, 6, 6>& getCovariance() const {
    return cov_T_WB_;
  }

  void setDefaultCovariance() {
    use_uniform_cov_ = true;
    cov_T_WB_.setIdentity();
  }

  void setCovariance(const Eigen::Matrix<double, 6, 6>& cov_T_WB, bool uniform) {
    use_uniform_cov_ = uniform;
    cov_T_WB_ = cov_T_WB;
  }

public:
  uint64_t id_;
  okvis::Time stamp_;
  okvis::kinematics::Transformation T_WB_;
private:
  bool use_uniform_cov_;
  Eigen::Matrix<double, 6, 6> cov_T_WB_;  ///< cov of $[\delta p, \delta \theta]$.
public:
  std::shared_ptr<swift_vio::MultiFrame> nframe_; ///< nframe contains the image, keypoints, and descriptors, for each frame.
  okvis::cameras::NCameraSystem cameraSystem_;  ///< the camera system info.
  // We use okvis::MultiFrame here for FrameNoncentralAbsoluteAdapter which is used for geometric verification.
  std::vector<std::shared_ptr<NeighborConstraintMessage>> odometryConstraintList_; ///< The most adjacent neighbor is at the front.

  std::vector<std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>>
      landmarkPositionList_;  ///< landmark positions expressed in the body frame of this keyframe for selected frames.
  std::vector<std::vector<int>> keypointIndexForLandmarkList_;  ///< The index of the keypoint within the nframe keypoint list for a landmark.
}; // LoopQueryKeyframeMessage

struct PgoResult {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  okvis::Time stamp_;
  okvis::kinematics::Transformation T_WB_;
};
}  // namespace swift_vio
#endif  // INCLUDE_OKVIS_KEYFRAME_FOR_LOOP_DETECTION_HPP_
