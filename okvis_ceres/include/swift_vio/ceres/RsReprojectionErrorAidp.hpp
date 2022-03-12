
/**
 * @file ceres/RsReprojectionErrorAidp.hpp
 * @brief Header file for the Rolling Shutter ReprojectionError with 
 * Anchored Inverse Depth Parameterization class.
 * @author Jianzhu Huai
 */

#ifndef INCLUDE_SWIFT_VIO_RS_REPROJECTION_ERROR_AIDP_HPP_
#define INCLUDE_SWIFT_VIO_RS_REPROJECTION_ERROR_AIDP_HPP_

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
// Gi: gyroscope triad i's frame relates to B by T_BGi = T_BAi * M_aiGi.
// W: world frame
// H: used as subscript to denote the host (anchor) frame.

// States
// T_WBi stamped by the base IMU clock.

// landmark hp_Ch = [alpha, beta, 1, rho] = [X/Z, Y/Z, 1, 1/Z] where X, Y, Z are the coordinates of the landmark in the host camera frame Ch.

// IMU intrinsic parameters have three blocks
// Mg: a fully populated matrix for R_AiGi, scale factors, and misalignment.
// Ms: a fully populated matrix for g-sensitivity.
// Ma: 6 parameters for scale factors, and misalignment.

// Camera intrinsic parameters
// For pinhole cameras, these parameters include projection intrinsics and distortion intrinsics.

// Camera readout time.

// Camera extrinsic parameters
// T_BCi, extrinsics for the target camera
// T_BCh, extrinsics for the host camera

// Camera time offset relative to the base IMU.

// Camera measurements of feature j in camera i.
// z = h((T_WB(t_{ij}) * T_BCi)^{-1} * T_WBh(t_h) * T_BCh * hp_Ch, camera intrinsics)

// Error definitions.
// R_{WB} = Exp(\theta) \hat{R}_{WB}
// p = dp + \hat{p}

namespace okvis {
namespace ceres {
class RsReprojectionErrorAidpBase : public ErrorInterface {
public:
  static const int kModelId = 6;
  static const int kNumResiduals = 2;
};

template <class GEOMETRY_TYPE>
class LocalBearingVectorAidp;

/// \brief The 2D keypoint reprojection error accounting for rolling shutter
///     skew and time offset and camera intrinsics, using anchored inverse depth parameterization,
///     and BG_BA_MG_MS_MA model.
/// This factor works whether the host camera and target camera are different or not.
///
/// The IMU data is used to predict camera positions at different image rows.
/// This factor works with the case where the host and target poses are the same.
/// \warning A potential problem with this reprojection error happens when
///     the provided IMU measurements do not cover camera observations to the
///     extent of the rolling shutter effect. This is most likely to occur with
///     observations in the most recent frame.
/// \tparam GEOMETRY_TYPE The camera gemetry type.
template <class GEOMETRY_TYPE>
class RsReprojectionErrorAidp
    : public ::ceres::SizedCostFunction<
          2 /* number of residuals */, 
          7 /* T_WBt with PoseLocalParameterization */,
          4 /* hp_Ch with InverseDepthParameterization */,
          7 /* T_WBh with PoseLocalParameterization */,
          7 /* T_BCt with PoseLocalParameterization */,
          7 /* T_BCh with PoseLocalParameterization */,
          GEOMETRY_TYPE::NumIntrinsics,
          1 /* frame readout time */,
          1 /* camera time offset */,
          3 /* speed */,
          6 /* bg_i and ba_i */,
          9 /* M_gi */,
          9 /* M_si */,
          6 /* M_ai */>,
      public RsReprojectionErrorAidpBase {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  OKVIS_DEFINE_EXCEPTION(Exception,std::runtime_error)

  /// \brief Make the camera geometry type accessible.
  typedef GEOMETRY_TYPE camera_geometry_t;

  static const int kDistortionDim = GEOMETRY_TYPE::distortion_t::NumDistortionIntrinsics;
  static const int kIntrinsicDim = GEOMETRY_TYPE::NumIntrinsics;

  /// \brief The base class type.
  typedef ::ceres::SizedCostFunction<
          2 /* number of residuals */, 
          7 /* T_WBt */, 
          4 /* AIDP */,
          7 /* T_WBh */,
          7 /* T_BCt */,
          7 /* T_BCh */,
          GEOMETRY_TYPE::NumIntrinsics,
          1 /* frame readout time */,
          1 /* camera time offset */,
          3 /* vW */,
          6 /* bg_i ba_i */,
          9,
          9,
          6> base_t;

  enum Index
  {
    T_WBt = 0,
    AIDP,
    T_WBh,
    T_BCt,
    T_BCh,
    Intrinsics,
    ReadoutTime,
    CameraTd,
    Speed,
    Biases,
    M_gi,
    M_si,
    M_ai
  };

  /// \brief The keypoint type (measurement type).
  typedef Eigen::Vector2d keypoint_t;

  /// \brief Measurement type (2D).
  typedef Eigen::Vector2d measurement_t;

  /// \brief Covariance / information matrix type (2x2).
  typedef Eigen::Matrix2d covariance_t;

  /// \brief Default constructor.
  RsReprojectionErrorAidp();

  /**
   * @brief RsReprojectionErrorAidp Construct with measurement and information matrix
   * @param measurement
   * @param information The information (weight) matrix.
   * @param imuMeasCanopy imu meas in the neighborhood of stateEpoch for
   *     compensating the rolling shutter effect.
   * @param stateEpoch epoch of the pose state and speed and biases
   */
  RsReprojectionErrorAidp(
      const measurement_t& measurement,
      const covariance_t& covariance,
      std::shared_ptr<const camera_geometry_t> targetCamera,
      std::shared_ptr<const okvis::ImuMeasurementDeque> imuMeasurementCanopy,
      std::shared_ptr<const okvis::ImuParameters> imuParameters,
      okvis::Time targetStateTime, okvis::Time targetImageTime);

  /// \brief Trivial destructor.
  virtual ~RsReprojectionErrorAidp()
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
    return "RsReprojectionErrorAidp";
  }

  void assignJacobians(
      double const *const *parameters, double **jacobians,
      double **jacobiansMinimal,
      const Eigen::Matrix<double, 2, 4> &Jh_weighted,
      const Eigen::Matrix<double, 2, Eigen::Dynamic> &Jpi_weighted,
      const Eigen::Matrix<double, 4, 6> &dhC_deltaTWSt,
      const Eigen::Matrix<double, 4, 6> &dhC_deltaTWSh,
      const Eigen::Matrix<double, 4, 4> &dhC_dlCh,
      const Eigen::Matrix<double, 4, 6> &dhC_dExtrinsict,
      const Eigen::Matrix<double, 4, 6> &dhC_dExtrinsich,
      const Eigen::Vector4d &dhC_td, double kpN,
      const Eigen::Matrix<double, 4, 3> &dhC_speed,
      const Eigen::Matrix<double, 4, 6> &dhC_biases) const;

  friend class LocalBearingVectorAidp<GEOMETRY_TYPE>;
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
class LocalBearingVectorAidp
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  LocalBearingVectorAidp(const RsReprojectionErrorAidp<GEOMETRY_TYPE> &rsre);
  template <typename Scalar>
  bool operator()(const Scalar *const T_WS, const Scalar *const hp_W, const Scalar *const T_WSh,
                  const Scalar *const extrinsic, const Scalar *const extrinsich, const Scalar *const t_r,
                  const Scalar *const t_d, const Scalar *const speed, const Scalar *const biases,
                  const Scalar *const deltaT_WS, const Scalar *const deltaT_WSh,
                  const Scalar *const deltaExtrinsic, const Scalar *const deltaExtrinsich,
                  const Scalar *const T_g, const Scalar *const T_s, const Scalar *const T_a,
                  Scalar *hp_C) const;

private:
  const RsReprojectionErrorAidp<GEOMETRY_TYPE> &rsre_;
};

}  // namespace ceres
}  // namespace okvis

#include "implementation/RsReprojectionErrorAidp.hpp"
#endif /* INCLUDE_SWIFT_VIO_RS_REPROJECTION_ERROR_AIDP_HPP_ */
