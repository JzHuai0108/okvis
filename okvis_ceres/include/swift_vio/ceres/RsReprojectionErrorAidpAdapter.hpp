
/**
 * @file ceres/RsReprojectionErrorAidpAdapter.hpp
 * @brief Header file for the Rolling Shutter ReprojectionError adapter with 
 * Anchored Inverse Depth Parameterization class.
 * @author Jianzhu Huai
 */

#ifndef INCLUDE_SWIFT_VIO_RS_REPROJECTION_ERROR_AIDP_ADAPTER_HPP_
#define INCLUDE_SWIFT_VIO_RS_REPROJECTION_ERROR_AIDP_ADAPTER_HPP_

#include <vector>
#include <memory>

#include <okvis/ceres/ParameterBlock.hpp>
#include <swift_vio/CameraIdentifier.h>

#include <swift_vio/ceres/RsReprojectionErrorAidp.hpp>

namespace okvis {
namespace ceres {

template <class GEOMETRY_TYPE>
class RsReprojectionErrorAidpAdapter
    : public ::ceres::DynamicCostFunction,
      public RsReprojectionErrorAidpBase {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  OKVIS_DEFINE_EXCEPTION(Exception,std::runtime_error)

  /// \brief Make the camera geometry type accessible.
  typedef GEOMETRY_TYPE camera_geometry_t;
  typedef RsReprojectionErrorAidp<GEOMETRY_TYPE> kernel_t;

  static const int kDistortionDim = GEOMETRY_TYPE::distortion_t::NumDistortionIntrinsics;
  static const int kIntrinsicDim = GEOMETRY_TYPE::NumIntrinsics;

  /// \brief The keypoint type (measurement type).
  typedef Eigen::Vector2d keypoint_t;

  /// \brief Measurement type (2D).
  typedef Eigen::Vector2d measurement_t;

  /// \brief Covariance / information matrix type (2x2).
  typedef Eigen::Matrix2d covariance_t;

  /// \brief Default constructor.
  RsReprojectionErrorAidpAdapter();

  /**
   * @brief RsReprojectionErrorAidpAdapter Construct with measurement and information matrix
   * @param measurement
   * @param information The information (weight) matrix.
   * @param imuMeasCanopy imu meas in the neighborhood of stateEpoch for
   *     compensating the rolling shutter effect.
   * @param stateEpoch epoch of the pose state and speed and biases
   */
  RsReprojectionErrorAidpAdapter(
      const swift_vio::CameraIdentifier &targetCamera,
      const swift_vio::CameraIdentifier &hostCamera,
      const measurement_t& measurement,
      const covariance_t& covariance,
      std::shared_ptr<const camera_geometry_t> targetCameraGeometry,
      std::shared_ptr<const okvis::ImuMeasurementDeque> imuMeasurementCanopy,
      std::shared_ptr<const okvis::ImuParameters> imuParameters,
      okvis::Time targetStateTime, okvis::Time targetImageTime);

  virtual ~RsReprojectionErrorAidpAdapter()
  {
  }

  virtual void setCovariance(const covariance_t& covariance);

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

  // sizes
  /// \brief Residual dimension.
  size_t residualDim() const
  {
    return kNumResiduals;
  }

  /// \brief Number of parameter blocks.
  size_t parameterBlocks() const
  {
    return kernel_t::numParameterBlocks() -
            (targetCamera_.frameId == hostCamera_.frameId ? 1:0) -
            (targetCamera_.cameraIndex == hostCamera_.cameraIndex ? 1:0);
  }

  void setParameterBlockAndResidualSizes();

  void uniqueBlocks(std::vector<double *> *ambientBlocks) const {
    if (targetCamera_.cameraIndex == hostCamera_.cameraIndex) {
      ambientBlocks->erase(ambientBlocks->begin() + kernel_t::Index::T_BCh);
    }
    if (targetCamera_.frameId == hostCamera_.frameId) {
      ambientBlocks->erase(ambientBlocks->begin() + kernel_t::Index::T_WBh);
    }
  }

  void uniqueBlocks(std::vector<std::shared_ptr<okvis::ceres::ParameterBlock>> *ambientBlocks) const {
    if (targetCamera_.cameraIndex == hostCamera_.cameraIndex) {
      ambientBlocks->erase(ambientBlocks->begin() + kernel_t::Index::T_BCh);
    }
    if (targetCamera_.frameId == hostCamera_.frameId) {
      ambientBlocks->erase(ambientBlocks->begin() + kernel_t::Index::T_WBh);
    }
  }

  /// \brief Dimension of an individual parameter block.
  /// @param[in] parameterBlockId ID of the parameter block of interest.
  /// \return The dimension.
  size_t parameterBlockDim(size_t /*parameterBlockId*/) const
  {
    throw std::runtime_error("parameterBlockDim not implemented!");
    return 0;
  }

  /// @brief Residual block type as string
  virtual std::string typeInfo() const
  {
    return "RsReprojectionErrorAidpAdapter";
  }

 protected:
  swift_vio::CameraIdentifier targetCamera_;
  swift_vio::CameraIdentifier hostCamera_;

  kernel_t costFunction_;

  void fullParameterList(double const *const *parameters,
      std::vector<double const *> *fullparameters) const;

  void fullParameterList2(double const *const *parameters,
      std::vector<double const *> *fullparameters) const;

  void fullJacobianList(double **jacobians, double *j_T_WBh, double *j_T_BCh,
                        std::vector<double *> *fullJacobians) const;

  template <int ParamDim>
  void uniqueJacobians(const std::vector<double *> &fullJacobians) const;
};

}  // namespace ceres
}  // namespace okvis

#include "implementation/RsReprojectionErrorAidpAdapter.hpp"
#endif /* INCLUDE_SWIFT_VIO_RS_REPROJECTION_ERROR_AIDP_ADAPTER_HPP_ */
