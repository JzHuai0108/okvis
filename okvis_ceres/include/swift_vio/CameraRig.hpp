#ifndef INCLUDE_SWIFT_VIO_CAMERA_RIG_HPP_
#define INCLUDE_SWIFT_VIO_CAMERA_RIG_HPP_

#include <map>

#include <swift_vio/ExtrinsicModels.hpp>
#include <swift_vio/ProjParamOptModels.hpp>

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
  std::vector<int> extrinsic_opt_rep_;

  ///< This indicates for each camera which subset of the projection intrinsic parameters are variable in estimation.
  std::vector<int> proj_opt_rep_;

  ///< for each camera, is the intrinsic parameters fixed?
  std::vector<bool> fixCameraIntrinsicParams_;

  ///< for each camera, is the extrinsic parameters fixed?
  std::vector<bool> fixCameraExtrinsicParams_;

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

  inline int getProjectionOptMode(int camera_id) const {
    return proj_opt_rep_[camera_id];
  }

  inline int getExtrinsicOptMode(int camera_id) const {
    if (camera_id >= (int)extrinsic_opt_rep_.size()) {
      return Extrinsic_p_BC_q_BC::kModelId;
    } else {
      return extrinsic_opt_rep_[camera_id];
    }
  }

  inline bool fixCameraIntrinsics(int camId) const {
    return fixCameraExtrinsicParams_[camId];
  }

  inline bool fixCameraExtrinsics(int camId) const {
    return fixCameraExtrinsicParams_[camId];
  }

  inline int getDistortionDim(int camera_id) const {
    return cameraGeometries_[camera_id]->noDistortionParameters();
  }

  inline okvis::cameras::NCameraSystem::DistortionType
      distortionType(int camera_id) const {
    return distortionTypes_[camera_id];
  }

  inline int getMinimalExtrinsicDim(int camera_id) const {
    return ExtrinsicModelGetMinimalDim(extrinsic_opt_rep_[camera_id]);
  }

  inline int getMinimalProjectionDim(int camera_id) const {
    return ProjectionOptGetMinimalDim(proj_opt_rep_[camera_id]);
  }

  inline int getCameraParamsMinimalDim(int camera_id) const {
    std::shared_ptr<okvis::cameras::CameraBase> camera =
        cameraGeometries_[camera_id];
    return (fixCameraExtrinsicParams_[camera_id] ? 0 : getMinimalExtrinsicDim(camera_id)) +
           (fixCameraIntrinsicParams_[camera_id] ? 0 : getIntrinsicDim(camera_id)) + 2;  // 2 for td and tr
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

  inline void setProjectionOptMode(int camera_id, int opt_mode) {
    proj_opt_rep_[camera_id] = opt_mode;
  }

  inline void setExtrinsicOptMode(int camera_id, int opt_mode) {
    extrinsic_opt_rep_[camera_id] = opt_mode;
  }

  inline void setProjectionOptMode(int camera_id, const std::string & opt_rep) {
    bool fixIntrinsics = false;
    proj_opt_rep_[camera_id] = ProjectionOptNameToId(opt_rep, &fixIntrinsics);
    fixCameraIntrinsicParams_[camera_id] = fixIntrinsics;
  }

  inline void setExtrinsicOptMode(int camera_id, const std::string & opt_rep) {
    bool fix = false;
    extrinsic_opt_rep_[camera_id] = ExtrinsicModelNameToId(opt_rep, &fix);
    fixCameraExtrinsicParams_[camera_id] = fix;
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
            std::string proj_opt_rep, std::string extrinsic_opt_rep) {
    bool fixIntrinsics = false;
    bool fixExtrinsics = false;
    int proj_opt_id = ProjectionOptNameToId(proj_opt_rep, &fixIntrinsics);
    int extrinsic_opt_id = ExtrinsicModelNameToId(extrinsic_opt_rep, &fixExtrinsics);
    return addCamera(T_SC, cameraGeometry, proj_opt_id, extrinsic_opt_id,
                      fixIntrinsics, fixExtrinsics);
  }

  int addCamera(std::shared_ptr<okvis::kinematics::Transformation> T_SC,
            std::shared_ptr<okvis::cameras::CameraBase> cameraGeometry,
            int proj_opt_id, int extrinsic_opt_id, bool fixIntrinsics, bool fixExtrinsics);

  inline int
  addCameraDeep(std::shared_ptr<const okvis::kinematics::Transformation> T_SC,
            std::shared_ptr<const okvis::cameras::CameraBase> cameraGeometry,
            std::string proj_opt_rep, std::string extrinsic_opt_rep) {
    bool fixIntrinsics = false;
    bool fixExtrinsics = false;
    int proj_opt_id = ProjectionOptNameToId(proj_opt_rep, &fixIntrinsics);
    int extrinsic_opt_id = ExtrinsicModelNameToId(extrinsic_opt_rep, &fixExtrinsics);
    return addCameraDeep(T_SC, cameraGeometry, proj_opt_id, extrinsic_opt_id, fixIntrinsics, fixExtrinsics);
  }

  CameraRig deepCopy() const;

  std::shared_ptr<CameraRig> deepCopyPtr() const;

  static CameraRig deepCopy(const okvis::cameras::NCameraSystem &ncameraSystem);

  static std::shared_ptr<CameraRig> deepCopyPtr(const okvis::cameras::NCameraSystem &ncameraSystem);

  void assignTo(CameraRig *rig) const;

  void assignTo(okvis::cameras::NCameraSystem *rig) const;

  void computeOverlaps();

private:
  int addCameraDeep(std::shared_ptr<const okvis::kinematics::Transformation> T_SC,
            std::shared_ptr<const okvis::cameras::CameraBase> cameraGeometry,
            int proj_opt_id, int extrinsic_opt_id, bool fixIntrinsics, bool fixExtrinsics);

};
}  // namespace swift_vio
#endif  // INCLUDE_SWIFT_VIO_CAMERA_RIG_HPP_
