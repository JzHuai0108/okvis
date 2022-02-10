
/**
 * @file cameras/EUCM.hpp
 * @brief Header file for the extended unified camera model class.
 * @author Jianzhu Huai
 */

#ifndef INCLUDE_OKVIS_CAMERAS_EUCM_HPP_
#define INCLUDE_OKVIS_CAMERAS_EUCM_HPP_

#include <Eigen/Core>
#include <memory>
#include <stdint.h>
#include <vector>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#include <opencv2/core/core.hpp> // Code that causes warning goes here
#pragma GCC diagnostic pop
#include "okvis/cameras/CameraBase.hpp"
#include "okvis/cameras/extended_camera.hpp"
#include <okvis/assert_macros.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {
/// \brief cameras Namespace for camera-related functionality.
namespace cameras {

class EUCM : public CameraBase {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef EUCM distortion_t; ///< To be compatible with other distortion models.

  EUCM();

  /// \brief Constructor
  /// @param[in] imageWidth The width in pixels.
  /// @param[in] imageHeight The height in pixels.
  /// @param[in] focalLengthU The horizontal focal length in pixels.
  /// @param[in] focalLengthV The vertical focal length in pixels.
  /// @param[in] imageCenterU The horizontal centre in pixels.
  /// @param[in] imageCenterV The vertical centre in pixels.
  /// @param[in] id Assign a generic ID, if desired.
  EUCM(int imageWidth, int imageHeight, double focalLengthU,
       double focalLengthV, double imageCenterU, double imageCenterV,
       double alpha, double beta, double delayTime = 0.0,
       double readoutTime = 0.0, uint64_t id = -1);

  /// \brief Destructor.
  ~EUCM() {}

  static const int NumProjectionIntrinsics =
      4;                              ///< optimisable projection intrinsics
  static const int NumIntrinsics = 6; ///< total number of intrinsics
  static const int NumDistortionIntrinsics = 2; ///< To be compatible with other distortion models.

  /// \brief Get the focal length along the u-dimension.
  /// \return The horizontal focal length in pixels.
  double focalLengthU() const { return eucm_.getParam()[0]; }

  /// \brief Get the focal length along the v-dimension.
  /// \return The vertical focal length in pixels.
  double focalLengthV() const { return eucm_.getParam()[1]; }

  /// \brief Get the image centre along the u-dimension.
  /// \return The horizontal centre in pixels.
  double imageCenterU() const { return eucm_.getParam()[2]; }

  /// \brief Get the focal image centre along the v-dimension.
  /// \return The vertical centre in pixels.
  double imageCenterV() const { return eucm_.getParam()[3]; }

  /// \brief Get the intrinsics as a concatenated vector.
  /// \return The intrinsics as a concatenated vector.
  void getIntrinsics(Eigen::VectorXd &intrinsics) const final;

  const Eigen::VectorXd &getIntrinsics() const final;

  /// \brief overwrite all intrinsics - use with caution !
  /// \param[in] intrinsics The intrinsics as a concatenated vector.
  bool setIntrinsics(const Eigen::VectorXd &intrinsics);

  /// \brief Get the total number of intrinsics.
  /// \return Number of intrinsics parameters.
  inline int noIntrinsicsParameters() const { return NumIntrinsics; }

  inline int noDistortionParameters() const { return 2; }
  //////////////////////////////////////////////////////////////
  /// \name Methods to project points
  /// @{

  /// \brief Projects a Euclidean point to a 2d image point (projection).
  /// @param[in]  point      The point in Euclidean coordinates.
  /// @param[out] imagePoint The image point.
  /// @return     Get information about the success of the projection. See
  ///             \ref ProjectionStatus for more information.
  CameraBase::ProjectionStatus
  project(const Eigen::Vector3d &point, Eigen::Vector2d *imagePoint) const;

  /// \brief Projects a Euclidean point to a 2d image point (projection).
  /// @param[in]  point              The point in Euclidean coordinates.
  /// @param[out] imagePoint         The image point.
  /// @param[out] pointJacobian      The Jacobian of the projection function
  /// w.r.t. the point..
  /// @param[out] intrinsicsJacobian The Jacobian of the projection function
  /// w.r.t. the intinsics.
  /// @return     Get information about the success of the projection. See
  ///             \ref ProjectionStatus for more information.
  CameraBase::ProjectionStatus
  project(const Eigen::Vector3d &point, Eigen::Vector2d *imagePoint,
          Eigen::Matrix<double, 2, 3> *pointJacobian,
          Eigen::Matrix2Xd *intrinsicsJacobian = NULL) const;

  /// \brief Projects a Euclidean point to a 2d image point (projection).
  /// @param[in]  point              The point in Euclidean coordinates.
  /// @param[in]  parameters         The intrinsics.
  /// @param[out] imagePoint         The image point.
  /// @param[out] pointJacobian      The Jacobian of the projection function
  /// w.r.t. the point..
  /// @param[out] intrinsicsJacobian The Jacobian of the projection function
  /// w.r.t. the intinsics.
  /// @return     Get information about the success of the projection. See
  ///             \ref ProjectionStatus for more information.
  CameraBase::ProjectionStatus projectWithExternalParameters(
      const Eigen::Vector3d &point, const Eigen::VectorXd &parameters,
      Eigen::Vector2d *imagePoint, Eigen::Matrix<double, 2, 3> *pointJacobian,
      Eigen::Matrix2Xd *intrinsicsJacobian = NULL) const;

  /// \brief Projects Euclidean points to 2d image points (projection) in a
  /// batch.
  /// @param[in]  points      The points in Euclidean coordinates (one point per
  /// column).
  /// @param[out] imagePoints The image points (one point per column).
  /// @param[out] stati       Get information about the success of the
  /// projections. See
  ///                         \ref ProjectionStatus for more information.
  inline void
  projectBatch(const Eigen::Matrix3Xd &/*points*/, Eigen::Matrix2Xd */*imagePoints*/,
               std::vector<CameraBase::ProjectionStatus> */*stati*/) const {
    OKVIS_THROW(std::runtime_error, "Not implemented.");
  }

  /// \brief Projects a point in homogenous coordinates to a 2d image point
  /// (projection).
  /// @param[in]  point      The point in Homogeneous coordinates.
  /// @param[out] imagePoint The image point.
  /// @return     Get information about the success of the projection. See
  ///             \ref ProjectionStatus for more information.
  CameraBase::ProjectionStatus
  projectHomogeneous(const Eigen::Vector4d &point, Eigen::Vector2d *imagePoint) const;

  /// \brief Projects a point in homogenous coordinates to a 2d image point
  /// (projection).
  /// @param[in]  point              The point in Homogeneous coordinates.
  /// @param[out] imagePoint         The image point.
  /// @param[out] pointJacobian      The Jacobian of the projection function
  /// w.r.t. the point.
  /// @param[out] intrinsicsJacobian The Jacobian of the projection function
  /// w.r.t. the intrinsics.
  /// @return     Get information about the success of the projection. See
  ///             \ref ProjectionStatus for more information.
  CameraBase::ProjectionStatus
  projectHomogeneous(const Eigen::Vector4d &point, Eigen::Vector2d *imagePoint,
                     Eigen::Matrix<double, 2, 4> *pointJacobian,
                     Eigen::Matrix2Xd *intrinsicsJacobian = NULL) const;

  /// \brief Projects a point in homogenous coordinates to a 2d image point
  /// (projection).
  /// @param[in]  point              The point in Homogeneous coordinates.
  /// @param[in]  parameters         The intrinsics.
  /// @param[out] imagePoint         The image point.
  /// @param[out] pointJacobian      The Jacobian of the projection function
  /// w.r.t. the point.
  /// @param[out] intrinsicsJacobian The Jacobian of the projection function
  /// w.r.t. the intrinsics.
  /// @return     Get information about the success of the projection. See
  ///             \ref ProjectionStatus for more information.
  CameraBase::ProjectionStatus projectHomogeneousWithExternalParameters(
      const Eigen::Vector4d &point, const Eigen::VectorXd &parameters,
      Eigen::Vector2d *imagePoint,
      Eigen::Matrix<double, 2, 4> *pointJacobian = NULL,
      Eigen::Matrix2Xd *intrinsicsJacobian = NULL) const;

  /// \brief Projects points in homogenous coordinates to 2d image points
  /// (projection) in a batch.
  /// @param[in]  points      The points in homogeneous coordinates (one point
  /// per column).
  /// @param[out] imagePoints The image points (one point per column).
  /// @param[out] stati       Get information about the success of the
  /// projections. See
  ///                         \ref ProjectionStatus for more information.
  inline void projectHomogeneousBatch(
      const Eigen::Matrix4Xd &/*points*/, Eigen::Matrix2Xd */*imagePoints*/,
      std::vector<CameraBase::ProjectionStatus> */*stati*/) const {
    OKVIS_THROW(std::runtime_error, "Not implemented.");
  }

  /// @}

  //////////////////////////////////////////////////////////////
  /// \name Methods to backproject points
  /// @{

  /// \brief Back-project a 2d image point into Euclidean space (direction
  /// vector).
  /// @param[in]  imagePoint The image point.
  /// @param[out] direction  The Euclidean direction vector.
  /// @return     true on success.
  bool backProject(const Eigen::Vector2d &imagePoint,
                          Eigen::Vector3d *direction) const;

  /// \brief Back-project a 2d image point into Euclidean space (direction
  /// vector).
  /// @param[in]  imagePoint         The image point.
  /// @param[out] direction          The Euclidean direction vector.
  /// @param[out] pointJacobian      Jacobian of the back-projection function
  /// w.r.t. the point.
  /// @return     true on success.
  inline bool backProject(const Eigen::Vector2d &/*imagePoint*/,
                          Eigen::Vector3d */*direction*/,
                          Eigen::Matrix<double, 3, 2> */*pointJacobian*/) const {
    OKVIS_THROW(std::runtime_error, "Not implemented.");
    return false;
  }

  /// \brief Back-project 2d image points into Euclidean space (direction
  /// vectors).
  /// @param[in]  imagePoints The image points (one point per column).
  /// @param[out] directions  The Euclidean direction vectors (one point per
  /// column).
  /// @param[out] success     Success of each of the back-projection
  inline bool backProjectBatch(const Eigen::Matrix2Xd &/*imagePoints*/,
                               Eigen::Matrix3Xd */*directions*/,
                               std::vector<bool> */*success*/) const {
    OKVIS_THROW(std::runtime_error, "Not implemented.");
    return false;
  }

  /// \brief Back-project a 2d image point into homogeneous point (direction
  /// vector).
  /// @param[in]  imagePoint The image point.
  /// @param[out] direction  The homogeneous point as direction vector.
  /// @return     true on success.
  inline bool backProjectHomogeneous(const Eigen::Vector2d &/*imagePoint*/,
                                     Eigen::Vector4d */*direction*/) const {
    OKVIS_THROW(std::runtime_error, "Not implemented.");
    return false;
  }

  /// \brief Back-project a 2d image point into homogeneous point (direction
  /// vector).
  /// @param[in]  imagePoint         The image point.
  /// @param[out] direction          The homogeneous point as direction vector.
  /// @param[out] pointJacobian      Jacobian of the back-projection function.
  /// @return     true on success.
  inline bool
  backProjectHomogeneous(const Eigen::Vector2d &/*imagePoint*/,
                         Eigen::Vector4d */*direction*/,
                         Eigen::Matrix<double, 4, 2> */*pointJacobian*/) const {
    OKVIS_THROW(std::runtime_error, "Not implemented.");
    return false;
  }

  /// \brief Back-project 2d image points into homogeneous points (direction
  /// vectors).
  /// @param[in]  imagePoints The image points (one point per column).
  /// @param[out] directions  The homogeneous points as direction vectors (one
  /// point per column).
  /// @param[out] success     Success of each of the back-projection
  inline bool backProjectHomogeneousBatch(const Eigen::Matrix2Xd &/*imagePoints*/,
                                          Eigen::Matrix4Xd */*directions*/,
                                          std::vector<bool> */*success*/) const {
    OKVIS_THROW(std::runtime_error, "Not implemented.");
    return false;
  }

  /// @}

  /// \brief get a test instance
  static std::shared_ptr<CameraBase> createTestObject() {
    return std::shared_ptr<CameraBase>(new EUCM(
        752, 480, 460.76484651566468, 459.4051018049483, 365.8937161309615,
        249.33499869752445, 0.5903365915227143, 1.127468196965374, 0.1, 0.02));
  }

  /// \brief get a test instance
  static EUCM testObject() {
    return EUCM(752, 480, 460.76484651566468, 459.4051018049483,
                365.8937161309615, 249.33499869752445, 0.5903365915227143,
                1.127468196965374, 0.1, 0.02);
  }

  /// \brief Obtain the projection type
  std::string type() const { return "EUCM"; }

  /// \brief Obtain the projection type
  const std::string distortionType() const { return "EUCM"; }

protected:

  ExtendedUnifiedCamera<double> eucm_;
};

} // namespace cameras
} // namespace okvis

#endif /* INCLUDE_OKVIS_CAMERAS_EUCM_HPP_ */
