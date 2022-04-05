#ifndef INCLUDE_SWIFT_VIO_CAMERA_RIG_HPP_
#define INCLUDE_SWIFT_VIO_CAMERA_RIG_HPP_

#include <map>

#include <swift_vio/ExtrinsicReps.hpp>
#include <swift_vio/ProjectionIntrinsicReps.h>

#include <okvis/assert_macros.hpp>
#include <okvis/cameras/CameraBase.hpp>
#include <okvis/cameras/EquidistantDistortion.hpp>
#include <okvis/cameras/EUCM.hpp>
#include <okvis/cameras/FovDistortion.hpp>
#include <okvis/cameras/NCameraSystem.hpp>
#include <okvis/cameras/NoDistortion.hpp>
#include <okvis/cameras/PinholeCamera.hpp>
#include <okvis/cameras/RadialTangentialDistortion.hpp>
#include <okvis/cameras/RadialTangentialDistortion8.hpp>
#include <okvis/Parameters.hpp>

#include <okvis/kinematics/Transformation.hpp>
#include <glog/logging.h>

namespace swift_vio {

void DistortionTypeToDimensionLabels(
    const okvis::cameras::NCameraSystem::DistortionType dtype,
    std::vector<std::string> *dimensionLabels);

void DistortionTypeToDesiredStdevs(
    const okvis::cameras::NCameraSystem::DistortionType dtype,
    Eigen::VectorXd *desiredStdevs);

okvis::cameras::NCameraSystem::DistortionType
DistortionNameToTypeId(const std::string& distortionName);

class CameraRig {
 private:
  ///< Mounting transformations from IMU
  std::vector<std::shared_ptr<okvis::kinematics::Transformation>> T_SC_;
  ///< Camera geometries
  std::vector<std::shared_ptr<okvis::cameras::CameraBase>> cameraGeometries_;

  std::vector<okvis::cameras::NCameraSystem::DistortionType> distortionTypes_;

  ///< This indicates for each camera which subset of the extrinsic parameters are variable in estimation.
  std::vector<int> extrinsicRepIds_;

  ///< This indicates for each camera which subset of the projection intrinsic parameters are variable in estimation.
  std::vector<int> projectionIntrinsicRepIds_;

  std::vector<std::vector<bool>> overlaps_;

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  inline CameraRig() {}

  inline size_t numCameras() const {
    return cameraGeometries_.size();
  }

  void clear();

  inline double getImageDelay(int camera_id) const {
    return cameraGeometries_[camera_id]->imageDelay();
  }

  inline double getReadoutTime(int camera_id) const {
    return cameraGeometries_[camera_id]->readoutTime();
  }

  inline uint32_t getImageWidth(int camera_id) const {
    return cameraGeometries_[camera_id]->imageWidth();
  }

  inline uint32_t getImageHeight(int camera_id) const {
    return cameraGeometries_[camera_id]->imageHeight();
  }

  inline const okvis::kinematics::Transformation &getCameraExtrinsic(
      int camera_id) const {
    return *(T_SC_[camera_id]);
  }

  inline std::shared_ptr<const okvis::kinematics::Transformation> getCameraExtrinsicPtr(
      int camera_id) const {
    return T_SC_[camera_id];
  }

  inline std::shared_ptr<const okvis::cameras::CameraBase> cameraGeometry(
      int camera_id) const {
    return cameraGeometries_[camera_id];
  }

  inline std::shared_ptr<okvis::cameras::CameraBase> cameraGeometryMutable(
      size_t cameraIndex) {
    return cameraGeometries_[cameraIndex];
  }

  // get the specific geometry (will be fast to use)
  template<class GEOMETRY_T>
  std::shared_ptr<const GEOMETRY_T> geometryAs(int camera_id) const {
  #ifndef NDEBUG
    OKVIS_ASSERT_TRUE(
        std::runtime_error, std::dynamic_pointer_cast<const GEOMETRY_T>(cameraGeometries_[camera_id]),
        "incorrect pointer cast requested. " << cameraGeometries_[camera_id]->distortionType());
  #endif
    return std::static_pointer_cast<const GEOMETRY_T>(cameraGeometries_[camera_id]);
  }

  inline std::shared_ptr<okvis::cameras::CameraBase> getMutableCameraGeometry(
      int camera_id) const {
    return cameraGeometries_[camera_id];
  }

  inline int getIntrinsicDim(int camera_id) const {
    return cameraGeometries_[camera_id]->noIntrinsicsParameters();
  }

  inline std::string getProjIntrinsicRepName(int camera_id) const {
    return ProjectionIntrinsicRepIdToName(projectionIntrinsicRepIds_[camera_id]);
  }

  inline int getProjectionIntrinsicRepId(int camera_id) const {
    return projectionIntrinsicRepIds_[camera_id];
  }

  inline int getExtrinsicRepId(int camera_id) const {
    if (camera_id >= (int)extrinsicRepIds_.size()) {
      return Extrinsic_p_BC_q_BC::kModelId;
    } else {
      return extrinsicRepIds_[camera_id];
    }
  }

  inline std::string extrinsicRepName(int camera_id) const {
    return ExtrinsicRepIdToName(extrinsicRepIds_[camera_id]);
  }

  inline int getDistortionDim(int camera_id) const {
    return cameraGeometries_[camera_id]->noDistortionParameters();
  }

  inline okvis::cameras::NCameraSystem::DistortionType
      distortionType(int camera_id) const {
    return distortionTypes_[camera_id];
  }

  inline int getMinimalExtrinsicDim(int camera_id) const {
    return ExtrinsicRepGetMinimalDim(extrinsicRepIds_[camera_id]);
  }

  inline int getMinimalProjectionIntrinsicDim(int camera_id) const {
    return ProjIntrinsicRepGetMinimalDim(projectionIntrinsicRepIds_[camera_id]);
  }

  inline int getCameraParamsVariableDim(
      int camera_id, const okvis::CameraNoiseParameters &camNoise) const {
    std::shared_ptr<okvis::cameras::CameraBase> camera =
        cameraGeometries_[camera_id];
    return (camNoise.isExtrinsicsFixed() ? 0
                                         : getMinimalExtrinsicDim(camera_id)) +
           (camNoise.isIntrinsicsFixed() ? 0 : getIntrinsicDim(camera_id)) +
           (camNoise.isTimeDelayFixed() ? 0 : 1) +
           (camNoise.isReadoutTimeFixed() ? 0 : 1);
  }

  inline void setImageDelay(int camera_id, double td) {
    cameraGeometries_[camera_id]->setImageDelay(td);
  }

  inline void setReadoutTime(int camera_id, double tr) {
    cameraGeometries_[camera_id]->setReadoutTime(tr);
  }

  inline void setCameraExtrinsic(
      int camera_id, const okvis::kinematics::Transformation& T_SC) {
    *(T_SC_[camera_id]) = T_SC;
  }

  inline void setProjectionIntrinsicRepId(int camera_id, int rep_id) {
    projectionIntrinsicRepIds_[camera_id] = rep_id;
  }

  inline void setExtrinsicRepId(int camera_id, int rep_id) {
    extrinsicRepIds_[camera_id] = rep_id;
  }

  inline void setProjectionIntrinsicRepId(int camera_id, const std::string & rep_name) {
    projectionIntrinsicRepIds_[camera_id] = ProjIntrinsicRepNameToId(rep_name);
  }

  inline void setExtrinsicRepId(int camera_id, const std::string & rep_name) {
    extrinsicRepIds_[camera_id] = ExtrinsicRepNameToId(rep_name);
  }

  inline void setCameraIntrinsics(int camera_id,
                                  const Eigen::VectorXd& intrinsic_vec) {
    cameraGeometries_[camera_id]->setIntrinsics(intrinsic_vec);
  }

  void setCameraIntrinsics(int camera_id,
                                  const Eigen::VectorXd& projection_vec,
                                  const Eigen::VectorXd& distortion_vec);

  inline void setOverlaps(const std::vector<std::vector<bool>> &overlaps) {
    overlaps_ = overlaps;
  }

  inline bool hasOverlap(size_t cameraIndexSeenBy, size_t cameraIndex) const {
      return overlaps_[cameraIndexSeenBy][cameraIndex];
  }

  inline int
  addCamera(std::shared_ptr<okvis::kinematics::Transformation> T_SC,
            std::shared_ptr<okvis::cameras::CameraBase> cameraGeometry,
            std::string projectionIntrinsicRepName, std::string extrinsicRepName) {
    int projIntrinsicRepId = ProjIntrinsicRepNameToId(projectionIntrinsicRepName);
    int extrinsicRepId = ExtrinsicRepNameToId(extrinsicRepName);
    return addCamera(T_SC, cameraGeometry, projIntrinsicRepId, extrinsicRepId);
  }

  int addCamera(std::shared_ptr<okvis::kinematics::Transformation> T_SC,
            std::shared_ptr<okvis::cameras::CameraBase> cameraGeometry,
            int projIntrinsicRepId, int extrinsicRepId);

  inline int
  addCameraDeep(std::shared_ptr<const okvis::kinematics::Transformation> T_SC,
            std::shared_ptr<const okvis::cameras::CameraBase> cameraGeometry,
            std::string projectionIntrinsicRepName, std::string extrinsicRepName) {
    int projIntrinsicRepId = ProjIntrinsicRepNameToId(projectionIntrinsicRepName);
    int extrinsicRepId = ExtrinsicRepNameToId(extrinsicRepName);
    return addCameraDeep(T_SC, cameraGeometry, projIntrinsicRepId, extrinsicRepId);
  }

  CameraRig deepCopy() const;

  std::shared_ptr<CameraRig> deepCopyPtr() const;

  static CameraRig deepCopy(const okvis::cameras::NCameraSystem &ncameraSystem);

  static std::shared_ptr<CameraRig> deepCopyPtr(const okvis::cameras::NCameraSystem &ncameraSystem);

  void initializeTo(okvis::cameras::NCameraSystem *rig) const;

  void assignTo(CameraRig *rig) const;

  void assignTo(okvis::cameras::NCameraSystem *rig) const;

  void computeOverlaps();

private:
  int addCameraDeep(std::shared_ptr<const okvis::kinematics::Transformation> T_SC,
            std::shared_ptr<const okvis::cameras::CameraBase> cameraGeometry,
            int projIntrinsicRepId, int extrinsicRepId);

};
}  // namespace swift_vio
#endif  // INCLUDE_SWIFT_VIO_CAMERA_RIG_HPP_
