
/**
 * @file EUCM.cpp
 * @brief Source file for the extended unified camera model class.
 * @author Jianzhu Huai
 */

#include <okvis/cameras/EUCM.hpp>
#include <okvis/cameras/extended_camera.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {
/// \brief cameras Namespace for camera-related functionality.
namespace cameras {

EUCM::EUCM() : CameraBase(752, 480, 0.1, 0.02, 0) {
  Eigen::Matrix<double, NumIntrinsics, 1> intrinsics;
  intrinsics[0] = 460; //< focalLengthU
  intrinsics[1] = 460; //< focalLengthV
  intrinsics[2] = 376; //< imageCenterU
  intrinsics[3] = 240; //< imageCenterV
  intrinsics[4] = 0.6;
  intrinsics[5] = 1.0;
  eucm_ = ExtendedUnifiedCamera<double>(intrinsics);
}

EUCM::EUCM(int imageWidth, int imageHeight, double focalLengthU,
           double focalLengthV, double imageCenterU, double imageCenterV,
           double alpha, double beta, double delayTime, double readoutTime,
           uint64_t id)
    : CameraBase(imageWidth, imageHeight, delayTime, readoutTime, id) {
  Eigen::Matrix<double, NumIntrinsics, 1> intrinsics;
  intrinsics[0] = focalLengthU; //< focalLengthU
  intrinsics[1] = focalLengthV; //< focalLengthV
  intrinsics[2] = imageCenterU; //< imageCenterU
  intrinsics[3] = imageCenterV; //< imageCenterV
  intrinsics[4] = alpha;
  intrinsics[5] = beta;
  eucm_ = ExtendedUnifiedCamera<double>(intrinsics);
}

void EUCM::getIntrinsics(Eigen::VectorXd &intrinsics) const {
  intrinsics = eucm_.getParam();
}

const Eigen::VectorXd &EUCM::getIntrinsics() const {
  return eucm_.getParam();
}

bool EUCM::setIntrinsics(const Eigen::VectorXd &intrinsics) {
  if (intrinsics.size() != NumIntrinsics) {
    return false;
  }
  eucm_ = ExtendedUnifiedCamera<double>(intrinsics);
  return true;
}

CameraBase::ProjectionStatus EUCM::project(const Eigen::Vector3d &point,
                                           Eigen::Vector2d *imagePoint) const {
  bool status = eucm_.project(point, *imagePoint);
  if (status) {
    return okvis::cameras::CameraBase::ProjectionStatus::Successful;
  }
  return okvis::cameras::CameraBase::ProjectionStatus::Invalid;
}

CameraBase::ProjectionStatus
EUCM::project(const Eigen::Vector3d &point, Eigen::Vector2d *imagePoint,
              Eigen::Matrix<double, 2, 3> *pointJacobian,
              Eigen::Matrix2Xd *intrinsicsJacobian) const {
  bool status;
  if (intrinsicsJacobian) {
    intrinsicsJacobian->resize(2, NumIntrinsics);
    status =
        eucm_.project(point, *imagePoint, pointJacobian, intrinsicsJacobian);
  } else {
    status = eucm_.project(point, *imagePoint, pointJacobian);
  }
  if (status) {
    return okvis::cameras::CameraBase::ProjectionStatus::Successful;
  }
  return okvis::cameras::CameraBase::ProjectionStatus::Invalid;
}

CameraBase::ProjectionStatus
EUCM::projectHomogeneous(const Eigen::Vector4d &point,
                         Eigen::Vector2d *imagePoint) const {
  Eigen::Vector3d head = point.head<3>();
  okvis::cameras::CameraBase::ProjectionStatus status;
  if (point[3] < 0) {
    status = project(-head, imagePoint);
  } else {
    status = project(head, imagePoint);
  }
  return status;
}

CameraBase::ProjectionStatus
EUCM::projectHomogeneous(const Eigen::Vector4d &point, Eigen::Vector2d *imagePoint,
                   Eigen::Matrix<double, 2, 4> *pointJacobian,
                   Eigen::Matrix2Xd *intrinsicsJacobian) const {
  Eigen::Vector3d head = point.head<3>();
  Eigen::Matrix<double, 2, 3> pointJacobian3;
  okvis::cameras::CameraBase::ProjectionStatus status;
  if (point[3] < 0) {
    status = project(-head, imagePoint, &pointJacobian3, intrinsicsJacobian);
  } else {
    status = project(head, imagePoint, &pointJacobian3, intrinsicsJacobian);
  }
  pointJacobian->template bottomRightCorner<2, 1>() = Eigen::Vector2d::Zero();
  pointJacobian->template topLeftCorner<2, 3>() = pointJacobian3;
  return status;
}

CameraBase::ProjectionStatus EUCM::projectHomogeneousWithExternalParameters(
    const Eigen::Vector4d &point, const Eigen::VectorXd &parameters,
    Eigen::Vector2d *imagePoint, Eigen::Matrix<double, 2, 4> *pointJacobian,
    Eigen::Matrix2Xd *intrinsicsJacobian) const {
  Eigen::Vector3d head = point.head<3>();
  Eigen::Matrix<double, 2, 3> pointJacobian3;
  CameraBase::ProjectionStatus status;
  if (point[3] < 0) {
    status = projectWithExternalParameters(-head, parameters, imagePoint,
                                                  &pointJacobian3,
                                                  intrinsicsJacobian);
  } else {
    status = projectWithExternalParameters(head, parameters, imagePoint,
                                                  &pointJacobian3,
                                                  intrinsicsJacobian);
  }
  if (pointJacobian) {
    pointJacobian->template bottomRightCorner<2, 1>() = Eigen::Vector2d::Zero();
    pointJacobian->template topLeftCorner<2, 3>() = pointJacobian3;
  }
  return status;
}

CameraBase::ProjectionStatus EUCM::projectWithExternalParameters(
    const Eigen::Vector3d &point, const Eigen::VectorXd &parameters,
    Eigen::Vector2d *imagePoint, Eigen::Matrix<double, 2, 3> *pointJacobian,
    Eigen::Matrix2Xd *intrinsicsJacobian) const {
  ExtendedUnifiedCamera<double> eucm(parameters);
  bool status = false;
  if (pointJacobian) {
    status = eucm.project(point, *imagePoint, pointJacobian, intrinsicsJacobian);
  } else {
    status = eucm.project(point, *imagePoint);
  }
  if (status) {
    return okvis::cameras::CameraBase::ProjectionStatus::Successful;
  }
  return okvis::cameras::CameraBase::ProjectionStatus::Invalid;
}

bool EUCM::backProject(const Eigen::Vector2d &imagePoint,
                       Eigen::Vector3d *direction) const {
  return eucm_.unproject(imagePoint, *direction);
}
} // namespace cameras
} // namespace okvis
