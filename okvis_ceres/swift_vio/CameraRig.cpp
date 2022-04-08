#include "swift_vio/CameraRig.hpp"

namespace swift_vio {
void DistortionTypeToDimensionLabels(
    const okvis::cameras::DistortionType dtype,
    std::vector<std::string> *dimensionLabels) {
  std::map<okvis::cameras::DistortionType,
           std::vector<std::string>>
      distortionNameList{
          {okvis::cameras::DistortionType::Equidistant,
           {"k1", "k2", "k3", "k4"}},
          {okvis::cameras::DistortionType::RadialTangential,
           {"k1", "k2", "p1", "p2"}},
          {okvis::cameras::DistortionType::No, {}},
          {okvis::cameras::DistortionType::RadialTangential8,
           {"k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"}},
          {okvis::cameras::DistortionType::Fov, {"omega"}},
          {okvis::cameras::DistortionType::Eucm, {"alpha", "beta"}}};

  std::map<okvis::cameras::DistortionType,
           std::vector<std::string>>::iterator it =
      std::find_if(
          distortionNameList.begin(), distortionNameList.end(),
          [&dtype](
              const std::pair<okvis::cameras::DistortionType,
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

void DistortionTypeToDesiredStdevs(
    const okvis::cameras::DistortionType dtype,
    Eigen::VectorXd *desiredStdevs) {
  switch (dtype) {
  case okvis::cameras::DistortionType::Equidistant:
    desiredStdevs->resize(4);
    desiredStdevs->setConstant(0.002);
    break;
  case okvis::cameras::DistortionType::RadialTangential:
    desiredStdevs->resize(4);
    desiredStdevs->setConstant(0.002);
    break;
  case okvis::cameras::DistortionType::No:
    desiredStdevs->resize(0);
    break;
  case okvis::cameras::DistortionType::RadialTangential8:
    desiredStdevs->resize(8);
    desiredStdevs->head<4>().setConstant(0.002);
    desiredStdevs->tail<4>().setConstant(0.0002);
    break;
  case okvis::cameras::DistortionType::Fov:
    desiredStdevs->resize(1);
    desiredStdevs->setConstant(0.002);
    break;
  case okvis::cameras::DistortionType::Eucm:
    desiredStdevs->resize(2);
    desiredStdevs->setConstant(0.01);
    break;
  }
}

okvis::cameras::DistortionType
DistortionNameToTypeId(const std::string& distortionName) {
  std::map<std::string, okvis::cameras::DistortionType> distortionNameList{
      {okvis::cameras::EquidistantDistortion().type(),
       okvis::cameras::DistortionType::Equidistant},
      {okvis::cameras::RadialTangentialDistortion().type(),
       okvis::cameras::DistortionType::RadialTangential},
      {okvis::cameras::NoDistortion().type(),
       okvis::cameras::DistortionType::No},
      {okvis::cameras::RadialTangentialDistortion8().type(),
       okvis::cameras::DistortionType::RadialTangential8},
      {okvis::cameras::FovDistortion().type(), okvis::cameras::DistortionType::Fov},
      {okvis::cameras::EUCM().type(), okvis::cameras::DistortionType::Eucm}};

  std::map<std::string, okvis::cameras::DistortionType>::iterator
      it = std::find_if(
      distortionNameList.begin(), distortionNameList.end(),
      [&distortionName](const std::pair<std::string, okvis::cameras::DistortionType>& val) {
        if (val.first.compare(distortionName) == 0) return true;
        return false;
      });
  if (it == distortionNameList.end()) {
    return okvis::cameras::DistortionType::No;
  } else {
    return it->second;
  }
}

std::string DistortionTypeToKalibrModel(okvis::cameras::DistortionType dt) {
  switch (dt) {
  case okvis::cameras::Equidistant:
    return "equidistant";
  case okvis::cameras::RadialTangential:
    return "radtan";
  case okvis::cameras::RadialTangential8:
    return "radtan8";
  case okvis::cameras::Fov:
    return "fov";
  case okvis::cameras::Eucm:
    return "eucm";
  case okvis::cameras::No:
    return "no";
  default:
    return "no";
  }
}

void CameraRig::clear() {
  T_SC_.clear();
  cameraGeometries_.clear();
  distortionTypes_.clear();
  extrinsicRepIds_.clear();
  projectionIntrinsicRepIds_.clear();
  overlaps_.clear();
}

void CameraRig::setCameraIntrinsics(int camera_id,
                                const Eigen::VectorXd& projection_vec,
                                const Eigen::VectorXd& distortion_vec) {
  Eigen::VectorXd intrinsicParameters;
  cameraGeometries_[camera_id]->getIntrinsics(intrinsicParameters);
  const int distortionDim =
      cameraGeometries_[camera_id]->noDistortionParameters();
  intrinsicParameters.tail(distortionDim) = distortion_vec;
  ProjIntrinsicRepLocalToGlobal(projectionIntrinsicRepIds_[camera_id], projection_vec,
                             &intrinsicParameters);

  cameraGeometries_[camera_id]->setIntrinsics(intrinsicParameters);
}

int CameraRig::addCamera(std::shared_ptr<okvis::kinematics::Transformation> T_SC,
          std::shared_ptr<okvis::cameras::CameraBase> cameraGeometry,
          int projIntrinsicRepId, int extrinsicRepId) {
  T_SC_.emplace_back(T_SC);
  cameraGeometries_.emplace_back(cameraGeometry);
  distortionTypes_.emplace_back(DistortionNameToTypeId(cameraGeometry->distortionType()));
  projectionIntrinsicRepIds_.emplace_back(projIntrinsicRepId);
  extrinsicRepIds_.emplace_back(extrinsicRepId);
  return static_cast<int>(T_SC_.size()) - 1;
}

int CameraRig::addCameraDeep(std::shared_ptr<const okvis::kinematics::Transformation> T_SC,
          std::shared_ptr<const okvis::cameras::CameraBase> cameraGeometry,
          int projIntrinsicRepId, int extrinsicRepId) {
  T_SC_.emplace_back(
      std::make_shared<okvis::kinematics::Transformation>(*T_SC));
  cameraGeometries_.emplace_back(okvis::cameras::cloneCameraGeometry(cameraGeometry));
  distortionTypes_.emplace_back(
      DistortionNameToTypeId(cameraGeometry->distortionType()));
  projectionIntrinsicRepIds_.emplace_back(projIntrinsicRepId);
  extrinsicRepIds_.emplace_back(extrinsicRepId);
  return static_cast<int>(T_SC_.size()) - 1;
}

CameraRig CameraRig::deepCopy() const {
  CameraRig rig;
  for (size_t i = 0u; i < T_SC_.size(); ++i) {
    rig.addCamera(T_SC_[i], cameraGeometries_[i], projectionIntrinsicRepIds_[i],
                   extrinsicRepIds_[i]);
  }
  rig.setOverlaps(overlaps_);
  return rig;
}

std::shared_ptr<CameraRig> CameraRig::deepCopyPtr() const {
  std::shared_ptr<CameraRig> rig(new CameraRig());
  for (size_t i = 0u; i < T_SC_.size(); ++i) {
    rig->addCamera(T_SC_[i], cameraGeometries_[i], projectionIntrinsicRepIds_[i],
                   extrinsicRepIds_[i]);
  }
  rig->setOverlaps(overlaps_);
  return rig;
}

CameraRig CameraRig::deepCopy(const okvis::cameras::NCameraSystem &ncameraSystem) {
  CameraRig rig;
  for (size_t i = 0u; i < ncameraSystem.numCameras(); ++i) {
    rig.addCameraDeep(ncameraSystem.T_SC(i), ncameraSystem.cameraGeometry(i), ncameraSystem.projectionIntrinsicRep(i),
                   ncameraSystem.extrinsicRep(i));
  }
  rig.setOverlaps(ncameraSystem.overlaps());
  return rig;
}

std::shared_ptr<CameraRig> CameraRig::deepCopyPtr(const okvis::cameras::NCameraSystem &ncameraSystem) {
  std::shared_ptr<CameraRig> rig(new CameraRig());
  for (size_t i = 0u; i < ncameraSystem.numCameras(); ++i) {
    rig->addCameraDeep(ncameraSystem.T_SC(i), ncameraSystem.cameraGeometry(i), ncameraSystem.projectionIntrinsicRep(i),
                   ncameraSystem.extrinsicRep(i));
  }
  rig->setOverlaps(ncameraSystem.overlaps());
  return rig;
}

void CameraRig::initializeTo(okvis::cameras::NCameraSystem *rig) const {
  rig->reset(this->T_SC_, this->cameraGeometries_, this->distortionTypes_, false);
  rig->setOverlaps(this->overlaps_);
  for (size_t i = 0 ; i < cameraGeometries_.size(); ++i) {
    rig->setExtrinsicRepName(i, extrinsicRepName(i));
    rig->setProjectionIntrinsicRepName(i, getProjIntrinsicRepName(i));
  }
}

void CameraRig::assignTo(CameraRig *rig) const {
  for (size_t i = 0u; i < T_SC_.size(); ++i) {
    rig->setCameraExtrinsic(i, *T_SC_[i]);
    rig->setCameraIntrinsics(i, cameraGeometries_[i]->getIntrinsics());
    rig->setImageDelay(i, getImageDelay(i));
    rig->setReadoutTime(i, getReadoutTime(i));
    rig->setExtrinsicRepId(i, extrinsicRepIds_.at(i));
    rig->setProjectionIntrinsicRepId(i, projectionIntrinsicRepIds_.at(i));
  }
  rig->setOverlaps(overlaps_);
}

void CameraRig::assignTo(okvis::cameras::NCameraSystem *rig) const {
  for (size_t i = 0u; i < T_SC_.size(); ++i) {
    rig->set_T_SC(i, T_SC_[i]);
    rig->setCameraIntrinsics(i, cameraGeometries_[i]->getIntrinsics());
    rig->setImageDelay(i, getImageDelay(i));
    rig->setReadoutTime(i, getReadoutTime(i));
    rig->setExtrinsicRepName(i, extrinsicRepName(i));
    rig->setProjectionIntrinsicRepName(i, getProjIntrinsicRepName(i));
  }
  rig->setOverlaps(overlaps_);
}

/// \brief compute all the overlaps of fields of view. Attention: can be expensive.
void CameraRig::computeOverlaps()
{
  std::cout << "CameraRig is computing overlaps between camera views. It may take a few seconds!" << std::endl;
  std::vector<std::vector<cv::Mat>> overlapMats;  ///< Overlaps between cameras: mats
  overlapMats.resize(cameraGeometries_.size());
  overlaps_.resize(cameraGeometries_.size());
  for (size_t cameraIndexSeenBy = 0; cameraIndexSeenBy < overlapMats.size();
      ++cameraIndexSeenBy) {
    overlapMats[cameraIndexSeenBy].resize(cameraGeometries_.size());
    overlaps_[cameraIndexSeenBy].resize(cameraGeometries_.size());
    for (size_t cameraIndex = 0; cameraIndex < overlapMats.size();
        ++cameraIndex) {
      std::shared_ptr<const okvis::cameras::CameraBase> camera = cameraGeometries_[cameraIndex];
      // self-visibility is trivial:
      if (cameraIndex == cameraIndexSeenBy) {
        // sizing the overlap map:
        overlapMats[cameraIndexSeenBy][cameraIndex] = cv::Mat::ones(
            camera->imageHeight(), camera->imageWidth(), CV_8UC1);
        overlaps_[cameraIndexSeenBy][cameraIndex] = true;
      } else {
        // sizing the overlap map:
        const size_t height = camera->imageHeight();
        const size_t width = camera->imageWidth();
        cv::Mat& overlapMat = overlapMats[cameraIndexSeenBy][cameraIndex];
        overlapMat = cv::Mat::zeros(height, width, CV_8UC1);
        // go through all the pixels:
        std::shared_ptr<const okvis::cameras::CameraBase> otherCamera =
            cameraGeometries_[cameraIndexSeenBy];
        const okvis::kinematics::Transformation T_Cother_C =
            T_SC_[cameraIndexSeenBy]->inverse() * (*T_SC_[cameraIndex]);

        int numOverlapPixels = 0;
        for (size_t u = 0; u < width; ++u) {
          for (size_t v = 0; v < height; ++v) {
            // backproject
            Eigen::Vector3d ray_C;
            camera->backProject(Eigen::Vector2d(double(u), double(v)), &ray_C);
            // project into other camera
            Eigen::Vector3d ray_Cother = T_Cother_C.C() * ray_C;  // points at infinity, i.e. we only do rotation
            Eigen::Vector2d imagePointInOtherCamera;
            okvis::cameras::CameraBase::ProjectionStatus status = otherCamera->project(
                ray_Cother, &imagePointInOtherCamera);

            // check the result
            if (status == okvis::cameras::CameraBase::ProjectionStatus::Successful) {
              Eigen::Vector3d verificationRay;
              otherCamera->backProject(imagePointInOtherCamera,&verificationRay);

              // to avoid an artefact of some distortion models, check again
              // note: (this should be fixed in the distortion implementation)
              if(fabs(ray_Cother.normalized().transpose()*verificationRay.normalized()-1.0)<1.0e-10) {
                // fill in the matrix:
                overlapMat.at<uchar>(v,u) = 1;
                ++numOverlapPixels;
              }
            }
          }
        }
        if (numOverlapPixels > (int)(height * width / 16)) {
          overlaps_[cameraIndexSeenBy][cameraIndex] = true;
        }
      }
      //std::stringstream name;
      //name << (cameraIndexSeenBy)<<"+"<<(cameraIndex);
      //cv::imshow(name.str().c_str(),255*overlapMats[cameraIndexSeenBy][cameraIndex]);
    }
  }
  //cv::waitKey();
}

}  // namespace swift_vio
