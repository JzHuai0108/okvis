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
namespace cameras {

static inline void DistortionTypeToDimensionLabels(
    const okvis::cameras::NCameraSystem::DistortionType dtype,
    std::vector<std::string> *dimensionLabels) {
  std::map<okvis::cameras::NCameraSystem::DistortionType,
           std::vector<std::string>>
      distortionNameList{
          {okvis::cameras::NCameraSystem::Equidistant,
           {"k1", "k2", "k3", "k4"}},
          {okvis::cameras::NCameraSystem::RadialTangential,
           {"k1", "k2", "p1", "p2"}},
          {okvis::cameras::NCameraSystem::NoDistortion, {}},
          {okvis::cameras::NCameraSystem::RadialTangential8,
           {"k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"}},
          {okvis::cameras::NCameraSystem::FOV, {"omega"}},
          {okvis::cameras::NCameraSystem::EUCM, {"alpha", "beta"}}};

  std::map<okvis::cameras::NCameraSystem::DistortionType,
           std::vector<std::string>>::iterator it =
      std::find_if(
          distortionNameList.begin(), distortionNameList.end(),
          [&dtype](
              const std::pair<okvis::cameras::NCameraSystem::DistortionType,
                              std::vector<std::string>> &val) {
            if (val.first == dtype)
              return true;
            return false;
          });
  if (it == distortionNameList.end()) {
    dimensionLabels->clear();
  } else {
    *dimensionLabels = it->second;
  }
}

static inline void DistortionTypeToDesiredStdevs(
    const okvis::cameras::NCameraSystem::DistortionType dtype,
    Eigen::VectorXd *desiredStdevs) {
  switch (dtype) {
  case okvis::cameras::NCameraSystem::Equidistant:
    desiredStdevs->resize(4);
    desiredStdevs->setConstant(0.002);
    break;
  case okvis::cameras::NCameraSystem::RadialTangential:
    desiredStdevs->resize(4);
    desiredStdevs->setConstant(0.002);
    break;
  case okvis::cameras::NCameraSystem::NoDistortion:
    desiredStdevs->resize(0);
    break;
  case okvis::cameras::NCameraSystem::RadialTangential8:
    desiredStdevs->resize(8);
    desiredStdevs->head<4>().setConstant(0.002);
    desiredStdevs->tail<4>().setConstant(0.0002);
    break;
  case okvis::cameras::NCameraSystem::FOV:
    desiredStdevs->resize(1);
    desiredStdevs->setConstant(0.002);
    break;
  case okvis::cameras::NCameraSystem::EUCM:
    desiredStdevs->resize(2);
    desiredStdevs->setConstant(0.01);
    break;
  }
}

static inline okvis::cameras::NCameraSystem::DistortionType
DistortionNameToTypeId(const std::string& distortionName) {
  std::map<std::string, okvis::cameras::NCameraSystem::DistortionType> distortionNameList{
      {okvis::cameras::EquidistantDistortion().type(),
       okvis::cameras::NCameraSystem::Equidistant},
      {okvis::cameras::RadialTangentialDistortion().type(),
       okvis::cameras::NCameraSystem::RadialTangential},
      {okvis::cameras::NoDistortion().type(),
       okvis::cameras::NCameraSystem::NoDistortion},
      {okvis::cameras::RadialTangentialDistortion8().type(),
       okvis::cameras::NCameraSystem::RadialTangential8},
      {okvis::cameras::FovDistortion().type(), okvis::cameras::NCameraSystem::FOV},
      {okvis::cameras::EUCM().type(), okvis::cameras::NCameraSystem::EUCM}};

  std::map<std::string, okvis::cameras::NCameraSystem::DistortionType>::iterator
      it = std::find_if(
      distortionNameList.begin(), distortionNameList.end(),
      [&distortionName](const std::pair<std::string, okvis::cameras::NCameraSystem::DistortionType>& val) {
        if (val.first.compare(distortionName) == 0) return true;
        return false;
      });
  if (it == distortionNameList.end()) {
    return okvis::cameras::NCameraSystem::NoDistortion;
  } else {
    return it->second;
  }
}

class CameraRig {
 private:
  ///< Mounting transformations from IMU
  std::vector<std::shared_ptr<okvis::kinematics::Transformation>> T_SC_;
  ///< Camera geometries
  std::vector<std::shared_ptr<okvis::cameras::CameraBase>> camera_geometries_;

  std::vector<okvis::cameras::NCameraSystem::DistortionType> distortionTypes_;

  ///< which subset of the extrinsic parameters are variable in estimation.
  std::vector<int> extrinsic_opt_rep_;

  ///< which subset of the projection intrinsic parameters are variable in estimation.
  std::vector<int> proj_opt_rep_;

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  inline CameraRig() {}
  inline size_t numberCameras() const {
    return camera_geometries_.size();
  }
  inline void clear() {
    T_SC_.clear();
    camera_geometries_.clear();
    distortionTypes_.clear();
    extrinsic_opt_rep_.clear();
    proj_opt_rep_.clear();
  }
  inline double getImageDelay(int camera_id) const {
    return camera_geometries_[camera_id]->imageDelay();
  }
  inline double getReadoutTime(int camera_id) const {
    return camera_geometries_[camera_id]->readoutTime();
  }
  inline uint32_t getImageWidth(int camera_id) const {
    return camera_geometries_[camera_id]->imageWidth();
  }
  inline uint32_t getImageHeight(int camera_id) const {
    return camera_geometries_[camera_id]->imageHeight();
  }
  inline okvis::kinematics::Transformation getCameraExtrinsic(
      int camera_id) const {
    return *(T_SC_[camera_id]);
  }
  inline std::shared_ptr<const okvis::kinematics::Transformation> getCameraExtrinsicPtr(
      int camera_id) const {
    return T_SC_[camera_id];
  }
  inline std::shared_ptr<const okvis::cameras::CameraBase> getCameraGeometry(
      int camera_id) const {
    return camera_geometries_[camera_id];
  }

  // get the specific geometry (will be fast to use)
  template<class GEOMETRY_T>
  std::shared_ptr<const GEOMETRY_T> geometryAs(int camera_id) const {
  #ifndef NDEBUG
    OKVIS_ASSERT_TRUE(
        std::runtime_error, std::dynamic_pointer_cast<const GEOMETRY_T>(camera_geometries_[camera_id]),
        "incorrect pointer cast requested. " << camera_geometries_[camera_id]->distortionType());
  #endif
    return std::static_pointer_cast<const GEOMETRY_T>(camera_geometries_[camera_id]);
  }

  inline std::shared_ptr<okvis::cameras::CameraBase> getMutableCameraGeometry(
      int camera_id) const {
    return camera_geometries_[camera_id];
  }
  inline int getIntrinsicDimen(int camera_id) const {
    return camera_geometries_[camera_id]->noIntrinsicsParameters();
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
  inline int getDistortionDimen(int camera_id) const {
    return camera_geometries_[camera_id]->noDistortionParameters();
  }
  inline okvis::cameras::NCameraSystem::DistortionType
      getDistortionType(int camera_id) const {
    return distortionTypes_[camera_id];
  }
  inline int getMinimalExtrinsicDimen(int camera_id) const {
    return ExtrinsicModelGetMinimalDim(extrinsic_opt_rep_[camera_id]);
  }
  inline int getMinimalProjectionDimen(int camera_id) const {
    return ProjectionOptGetMinimalDim(proj_opt_rep_[camera_id]);
  }
  inline int getCameraParamsMinimalDim(int camera_id) const {
    std::shared_ptr<okvis::cameras::CameraBase> camera =
        camera_geometries_[camera_id];
    return getMinimalExtrinsicDimen(camera_id) +
           getMinimalProjectionDimen(camera_id) +
           camera->noDistortionParameters() + 2;  // 2 for td and tr
  }
  inline void setImageDelay(int camera_id, double td) {
    camera_geometries_[camera_id]->setImageDelay(td);
  }
  inline void setReadoutTime(int camera_id, double tr) {
    camera_geometries_[camera_id]->setReadoutTime(tr);
  }
  inline void setCameraExtrinsic(
      int camera_id, const okvis::kinematics::Transformation& T_SC) {
    *(T_SC_[camera_id]) = T_SC;
  }
  inline void setProjectionOptMode(int opt_mode, int camera_id) {
    proj_opt_rep_[camera_id] = opt_mode;
  }
  inline void setExtrinsicOptMode(int opt_mode, int camera_id) {
    extrinsic_opt_rep_[camera_id] = opt_mode;
  }
  inline void setCameraIntrinsics(int camera_id,
                                  const Eigen::VectorXd& intrinsic_vec) {
    camera_geometries_[camera_id]->setIntrinsics(intrinsic_vec);
  }
  inline void setCameraIntrinsics(int camera_id,
                                  const Eigen::VectorXd& projection_vec,
                                  const Eigen::VectorXd& distortion_vec) {
    Eigen::VectorXd intrinsicParameters;
    camera_geometries_[camera_id]->getIntrinsics(intrinsicParameters);
    const int distortionDim =
        camera_geometries_[camera_id]->noDistortionParameters();
    intrinsicParameters.tail(distortionDim) = distortion_vec;
    ProjectionOptLocalToGlobal(proj_opt_rep_[camera_id], projection_vec,
                               &intrinsicParameters);

    camera_geometries_[camera_id]->setIntrinsics(intrinsicParameters);
  }
  inline int
  addCamera(std::shared_ptr<const okvis::kinematics::Transformation> T_SC,
            std::shared_ptr<const okvis::cameras::CameraBase> cameraGeometry,
            std::string pinhole_opt_rep, std::string extrinsic_opt_rep) {
    return addCamera(T_SC, cameraGeometry,
                     ProjectionOptNameToId(pinhole_opt_rep),
                     ExtrinsicModelNameToId(extrinsic_opt_rep));
  }

  inline int
  addCamera(std::shared_ptr<const okvis::kinematics::Transformation> T_SC,
            std::shared_ptr<const okvis::cameras::CameraBase> cameraGeometry,
            int pinhole_opt_rep_id, int extrinsic_opt_rep_id) {
    T_SC_.emplace_back(
        std::make_shared<okvis::kinematics::Transformation>(*T_SC));
    camera_geometries_.emplace_back(okvis::cameras::cloneCameraGeometry(cameraGeometry));
    distortionTypes_.emplace_back(
        DistortionNameToTypeId(cameraGeometry->distortionType()));
    proj_opt_rep_.emplace_back(pinhole_opt_rep_id);
    extrinsic_opt_rep_.emplace_back(extrinsic_opt_rep_id);
    return static_cast<int>(T_SC_.size()) - 1;
  }

  CameraRig deepCopy() const {
    CameraRig rig;
    for (size_t i = 0u; i < T_SC_.size(); ++i) {
      rig.addCamera(T_SC_[i], camera_geometries_[i], proj_opt_rep_[i],
                    extrinsic_opt_rep_[i]);
    }
    return rig;
  }

  void assignTo(std::shared_ptr<okvis::cameras::NCameraSystem> rig) const {
    for (size_t i = 0u; i < T_SC_.size(); ++i) {
      rig->set_T_SC(i, T_SC_[i]);
      rig->setCameraIntrinsics(i, camera_geometries_[i]->getIntrinsics());
      rig->setImageDelay(i, getImageDelay(i));
      rig->setReadoutTime(i, getReadoutTime(i));
    }
  }
};
}  // namespace cameras
}  // namespace swift_vio
#endif  // INCLUDE_SWIFT_VIO_CAMERA_RIG_HPP_
