
/**
 * @file ceres/RSCameraReprojectionError.hpp
 * @brief Header file for the Rolling Shutter camera ReprojectionError class.
 * @author Jianzhu Huai
 */

#ifndef INCLUDE_SWIFT_VIO_RSCAMERA_REPROJECTION_ERROR_HPP_
#define INCLUDE_SWIFT_VIO_RSCAMERA_REPROJECTION_ERROR_HPP_

#include <vector>
#include <memory>
#include <ceres/ceres.h>

#include <okvis/assert_macros.hpp>
#include <okvis/cameras/CameraBase.hpp>
#include <okvis/ceres/PoseLocalParameterization.hpp>
#include <okvis/ceres/ErrorInterface.hpp>
#include <okvis/Measurements.hpp>
#include <okvis/Parameters.hpp>

#include <swift_vio/imu/ImuModels.hpp>
#include <swift_vio/ExtrinsicModels.hpp>
#include <swift_vio/PointLandmarkModels.hpp>

// Frame notation:
// B: body frame
// Ci: camera i's frame relates to B by T_BCi
// Ai: accelerometer triad i's frame relates to B by T_BAi
// Gi: gyroscope triad i's frame relates to B by T_BGi = T_BAi * T_AiGi.
// W: world frame
// H: used as subscript to denote the host or anchor frame.

// States
// T_WBi stamped by the base IMU clock.

// landmark hp_Ch = [alpha, beta, 1, rho] = [X/Z, Y/Z, 1, 1/Z] where X, Y, Z are the coordinates of the landmark in the host camera frame Ch.

// IMU intrinsic parameters have three blocks
// Tgi: a fully populated matrix for R_AiGi, scale factors, and misalignment.
// Tsi: a fully populated matrix for g-sensitivity.
// Tai: 6 parameters for scale factors, and misalignment.

// IMU measurements timestamped by the IMU itself.
// w_m = T_gi * R_AiB * w_B + T_s * R_AiB * a_B + b_g
// a_m = T_ai * R_AiB * a_B + b_a
// Comparing to the IMU model in extending Kalibr, this model ignores the size effect.

// Camera intrinsic parameters
// For pinhole cameras, these parameters include projection intrinsics and distortion intrinsics.

// Camera readout time.

// Camera extrinsic parameters
// T_BCi, extrinsics for the target camera
// T_BCh, extrinsics for the host camera

// Camera time offset relative to the base IMU.

// Camera measurements
// z = h((T_WB(t_{ij}) * T_BCi)^{-1} * T_WBh(t_h) * T_BCh * hp_Ch, camera intrinsics)

// Error definitions.
// R_{WB} = Exp(\theta) \hat{R}_{WB}
// p = dp + \hat{p}

namespace okvis {
namespace ceres {
class RSCameraReprojectionErrorBase : public ErrorInterface {
public:
  static const int kModelId = 6;
  static const int kNumResiduals = 2;
};

template <class GEOMETRY_TYPE>
class RS_LocalBearingVector;

/// \brief The 2D keypoint reprojection error accounting for rolling shutter
///     skew and time offset and camera intrinsics.
/// \warning A potential problem with this reprojection error happens when
///     the provided IMU measurements do not cover camera observations to the
///     extent of the rolling shutter effect. This is most likely to occur with
///     observations in the most recent frame.
///     Because MSCKF uses observations up to the second most recent frame,
///     this problem should only happen to optimization-based estimator with
///     undelayed observations.
/// \tparam GEOMETRY_TYPE The camera gemetry type.
template <class GEOMETRY_TYPE>
class RSCameraReprojectionError
    : public ::ceres::SizedCostFunction<
          2 /* number of residuals */, 
          7 /* T_WBt */, 
          4 /* hp_Ch */,
          7 /* T_WBh */,
          7 /* T_BCt */,
          7 /* T_BCh */,
          GEOMETRY_TYPE::NumIntrinsics,
          1 /* frame readout time */,
          1 /* camera time offset */,
          9 /* speed, bg_i and ba_i */,
          9 /* T_gi */,
          9 /* T_si */,
          6 /* T_ai */>,
      public RSCameraReprojectionErrorBase {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  OKVIS_DEFINE_EXCEPTION(Exception,std::runtime_error)

  /// \brief Make the camera geometry type accessible.
  typedef GEOMETRY_TYPE camera_geometry_t;

  static const int kDistortionDim = GEOMETRY_TYPE::distortion_t::NumDistortionIntrinsics;
  static const int kIntrinsicsDim = GEOMETRY_TYPE::NumIntrinsics;

  /// \brief The base class type.
  typedef ::ceres::SizedCostFunction<
          2 /* number of residuals */, 
          7 /* T_WBt */, 
          4 /* hp_Ch */,
          7 /* T_WBh */,
          7 /* T_BCt */,
          7 /* T_BCh */,
          GEOMETRY_TYPE::NumIntrinsics,
          1 /* frame readout time */,
          1 /* camera time offset */,
          9 /* velocity, bg_i and ba_i */,
          9 /* T_gi */,
          9 /* T_si */,
          6 /* T_ai */> base_t;

  enum Index
  {
    T_WBt = 0,
    hp_Ch,
    T_WBh,
    T_BCt,
    T_BCh,
    Intrinsics,
    ReadoutTime,
    CameraTd,
    SpeedAndBiases,
    T_gi,
    T_si,
    T_ai
  };

  /// \brief The keypoint type (measurement type).
  typedef Eigen::Vector2d keypoint_t;

  /// \brief Measurement type (2D).
  typedef Eigen::Vector2d measurement_t;

  /// \brief Covariance / information matrix type (2x2).
  typedef Eigen::Matrix2d covariance_t;

  /// \brief Default constructor.
  RSCameraReprojectionError();

  /**
   * @brief RSCameraReprojectionError Construct with measurement and information matrix
   * @param measurement
   * @param information The information (weight) matrix.
   * @param imuMeasCanopy imu meas in the neighborhood of stateEpoch for
   *     compensating the rolling shutter effect.
   * @param stateEpoch epoch of the pose state and speed and biases
   */
  RSCameraReprojectionError(
      const measurement_t& measurement,
      const covariance_t& covariance,
      std::shared_ptr<const camera_geometry_t> targetCamera,
      std::shared_ptr<const okvis::ImuMeasurementDeque> imuMeasurementCanopy,
      std::shared_ptr<const okvis::ImuParameters> imuParameters,
      okvis::Time targetStateTime, okvis::Time targetImageTime);

  /// \brief Trivial destructor.
  virtual ~RSCameraReprojectionError()
  {
  }

  /// \brief Set the information.
  /// @param[in] information The information (weight) matrix.
  virtual void setCovariance(const covariance_t& information);

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

  bool EvaluateWithMinimalJacobiansAnalytic(double const* const * parameters,
                                            double* residuals,
                                            double** jacobians,
                                            double** jacobiansMinimal) const;

  bool EvaluateWithMinimalJacobiansAutoDiff(double const* const * parameters,
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
    return "RSCameraReprojectionError";
  }

  void assignJacobians(
      double const *const *parameters, double **jacobians,
      double **jacobiansMinimal,
      const Eigen::Matrix<double, 2, 4> &Jh_weighted,
      const Eigen::Matrix<double, 2, Eigen::Dynamic> &Jpi_weighted,
      const Eigen::Matrix<double, 4, 6> &dhC_deltaTWSt,
      const Eigen::Matrix<double, 4, 6> &dhC_deltaTWSh,
      const Eigen::Matrix<double, 4, 4> &dhC_deltahpCh,
      const Eigen::Matrix<double, 4, 6> &dhC_dExtrinsict,
      const Eigen::Matrix<double, 4, 6> &dhC_dExtrinsich,
      const Eigen::Vector4d &dhC_td, double kpN,
      const Eigen::Matrix<double, 4, 9> &dhC_sb) const;

  friend class RS_LocalBearingVector<GEOMETRY_TYPE>;
 protected:
  measurement_t measurement_; ///< The (2D) measurement.

  std::shared_ptr<const camera_geometry_t> cameraGeometryBase_;

  std::shared_ptr<const okvis::ImuMeasurementDeque> imuMeasCanopy_;
  std::shared_ptr<const okvis::ImuParameters> imuParameters_;

  std::shared_ptr<const camera_geometry_t> targetCamera_;

  // weighting related
  covariance_t information_; ///< The 2x2 information matrix.
  covariance_t squareRootInformation_; ///< The 2x2 square root information matrix.
  covariance_t covariance_; ///< The 2x2 covariance matrix.

  okvis::Time targetStateTime_; ///< Timestamp of the target pose, T_WBt.
  okvis::Time targetImageTime_; ///< Raw timestamp of the target image.
};

template <class GEOMETRY_TYPE>
class RS_LocalBearingVector
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  RS_LocalBearingVector(const RSCameraReprojectionError<GEOMETRY_TYPE> &rsre);
  template <typename Scalar>
  bool operator()(const Scalar *const T_WS, const Scalar *const hp_W, const Scalar *const T_WSh,
                  const Scalar *const extrinsic, const Scalar *const extrinsich, const Scalar *const t_r,
                  const Scalar *const t_d, const Scalar *const speedAndBiases,
                  const Scalar *const deltaT_WS, const Scalar *const deltaT_WSh,
                  const Scalar *const deltaExtrinsic, const Scalar *const deltaExtrinsich,
                  const Scalar *const T_g, const Scalar *const T_s, const Scalar *const T_a,
                  Scalar *hp_C) const;

private:
  const RSCameraReprojectionError<GEOMETRY_TYPE> &rsre_;
};

}  // namespace ceres
}  // namespace okvis

#include "implementation/RSCameraReprojectionError.hpp"
#endif /* INCLUDE_SWIFT_VIO_RSCAMERA_REPROJECTION_ERROR_HPP_ */
