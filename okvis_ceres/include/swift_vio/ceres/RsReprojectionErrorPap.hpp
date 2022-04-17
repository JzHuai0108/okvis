
/**
 * @file ceres/RsReprojectionErrorPap.hpp
 * @brief Header file for the Rolling Shutter ReprojectionError 
 * with Parallax Angle Parameterization class.
 * @author Jianzhu Huai
 */

#ifndef INCLUDE_SWIFT_VIO_RS_REPROJECTION_ERROR_PAP_HPP_
#define INCLUDE_SWIFT_VIO_RS_REPROJECTION_ERROR_PAP_HPP_

#include <vector>
#include <memory>
#include <ceres/ceres.h>
#include <okvis/Measurements.hpp>

#include <okvis/assert_macros.hpp>
#include <okvis/ceres/ErrorInterface.hpp>

#include <swift_vio/imu/ImuModels.hpp>
#include <swift_vio/ExtrinsicReps.hpp>
#include <swift_vio/PointSharedData.hpp>
#include <swift_vio/PointLandmarkModels.hpp>

namespace okvis {
namespace ceres {

class RsReprojectionErrorPapBase : public ErrorInterface {
public:
  static const int kModelId = 3;
  static const int kNumResiduals = 2;
};

/// \brief The reprojection error with Parallax angle parameterization
/// \f$ \pi(R_{C(t_{i,j})} * N_{i,j}) - z_{i,j} \f$ accounting
/// for rolling shutter skew and time offset and camera intrinsics.
/// \warning A potential problem with this error term happens when
///     the provided IMU measurements do not cover camera observations to the
///     extent of the rolling shutter effect. This is most likely to occur with
///     observations in the most recent frame.
/// \tparam GEOMETRY_TYPE The camera gemetry type.
template <class GEOMETRY_TYPE>
class RsReprojectionErrorPap
    : public ::ceres::SizedCostFunction<
          2 /* residuals */, 7 /* observing frame pose */, 7 /* main anchor */,
          7 /* associate anchor */,
          swift_vio::ParallaxAngleParameterization::kGlobalDim /* landmark */,
          7 /* camera extrinsic */,
          GEOMETRY_TYPE::NumIntrinsics /* camera intrinsic */,
          1 /* frame readout time */, 1 /* camera time delay */,
          9 /* velocity and biases of observing frame */,
          9 /* velocity and biases of main anchor */,
          9 /* velocity and biases of associate anchor */>,
      public RsReprojectionErrorPapBase {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  OKVIS_DEFINE_EXCEPTION(Exception,std::runtime_error)

  enum Index {
    T_WBt = 0,
    T_WBm,
    T_WBa,
    PAP,
    T_BC,
    Intrinsics,
    TR,
    TD,
    SpeedAndBiast,
    SpeedAndBiasm,
    SpeedAndBiasa
  };

  /// \brief Make the camera geometry type accessible.
  typedef GEOMETRY_TYPE camera_geometry_t;
  typedef swift_vio::ParallaxAngleParameterization LANDMARK_MODEL;
  static const int kIntrinsicDim = GEOMETRY_TYPE::NumIntrinsics;

  /// \brief The base class type.
  typedef ::ceres::SizedCostFunction<
      kNumResiduals, 7, 7, 7, LANDMARK_MODEL::kGlobalDim, 7, kIntrinsicDim,
      1, 1, 9, 9, 9>
      base_t;

  typedef Eigen::Matrix<double, kNumResiduals, kIntrinsicDim, Eigen::RowMajor> IntrinsicJacType;

  /// \brief Default constructor.
  RsReprojectionErrorPap();

  /**
   * @brief RsReprojectionErrorPap Construct with measurement and information matrix
   * @param cameraGeometry
   * @warning The camera geometry will be modified in evaluating Jacobians.
   * @param cameraId The id of the camera in the okvis::cameras::NCameraSystem.
   * @param measurement
   * @param pointDataPtr shared data of the landmark to compute propagated
   * poses and velocities at observation epochs.
   */
  RsReprojectionErrorPap(
      std::shared_ptr<const camera_geometry_t> cameraGeometry,
      const Eigen::Vector2d& imageObservation,
      const Eigen::Matrix2d& observationCovariance,
      size_t observationIndex,
      const swift_vio::PointSharedData *pointDataPtr);

  /// \brief Trivial destructor.
  virtual ~RsReprojectionErrorPap()
  {
  }

  /// \brief Set the underlying camera model.
  /// @param[in] cameraGeometry The camera geometry.
  void setCameraGeometry(
      std::shared_ptr<const camera_geometry_t> cameraGeometry)
  {
    cameraGeometryBase_ = cameraGeometry;
  }

  // error term and Jacobian implementation
  /**
   * @brief This evaluates the error term and additionally computes the Jacobians.
   * @param parameters Pointer to the parameters (see ceres)
   * @param residuals Pointer to the residual vector (see ceres)
   * @param jacobians Pointer to the Jacobians (see ceres)
   * @return success of th evaluation.
   */
  virtual bool Evaluate(double const* const * parameters, double* residuals,
                        double** jacobians) const;

  /**
   * @brief This evaluates the error term and additionally computes
   *        the Jacobians in the minimal internal representation.
   * @param parameters Pointer to the parameters (see ceres)
   * @param residuals Pointer to the residual vector (see ceres)
   * @param jacobians Pointer to the Jacobians (see ceres)
   * @param jacobiansMinimal Pointer to the minimal Jacobians (equivalent to jacobians).
   * @return Success of the evaluation.
   */
  virtual bool EvaluateWithMinimalJacobians(double const* const * parameters,
                                            double* residuals,
                                            double** jacobians,
                                            double** jacobiansMinimal) const;

  void setJacobiansZero(double** jacobians, double** jacobiansMinimal) const;

  // sizes
  /// \brief Residual dimension.
  size_t residualDim() const
  {
    return kNumResiduals;
  }

  /// \brief Number of parameter blocks.
  size_t parameterBlocks() const
  {
    return base_t::parameter_block_sizes().size();
  }

  /// \brief Dimension of an individual parameter block.
  /// @param[in] parameterBlockId ID of the parameter block of interest.
  /// \return The dimension.
  size_t parameterBlockDim(size_t parameterBlockId) const
  {
    return base_t::parameter_block_sizes().at(parameterBlockId);
  }

  /// @brief Residual block type as string
  virtual std::string typeInfo() const
  {
    return "RsReprojectionErrorPap";
  }

 protected:
  Eigen::Vector2d measurement_; ///< The image observation.

  /// Warn: cameraGeometryBase_ may be updated with
  /// a ceres EvaluationCallback prior to Evaluate().
  std::shared_ptr<const camera_geometry_t> cameraGeometryBase_;

  // weighting related
  Eigen::Matrix2d squareRootInformation_; // updated in Evaluate()
  Eigen::Matrix2d covariance_;

  size_t observationIndex_; ///< Index of the observation in the map point shared data.
  const swift_vio::PointSharedData *pointDataPtr_;
};

}  // namespace ceres
}  // namespace okvis

#include "implementation/RsReprojectionErrorPap.hpp"
#endif /* INCLUDE_SWIFT_VIO_RS_REPROJECTION_ERROR_PAP_HPP_ */
