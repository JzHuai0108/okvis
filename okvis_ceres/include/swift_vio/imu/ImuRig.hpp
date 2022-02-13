#ifndef INCLUDE_SWIFT_VIO_IMU_RIG_HPP_
#define INCLUDE_SWIFT_VIO_IMU_RIG_HPP_

#include <memory>

#include <Eigen/Core>
#include <Eigen/StdVector>

#include <swift_vio/imu/ImuModels.hpp>
#include <swift_vio/ceres/EuclideanParamBlockSized.hpp>

#include <okvis/Parameters.hpp>

namespace swift_vio {
class ImuModel {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ImuModel() {}
  /**
   * @brief ImuModel
   * @param modelId
   * @param bg
   * @param ba
   * @param extraParams
   */
  ImuModel(int modelId, const Eigen::Vector3d bg, Eigen::Vector3d ba,
           const Eigen::VectorXd &extraParams)
      : modelId_(modelId), gyroBias_(bg), accelBias_(ba),
        extraParams_(extraParams) {}

  inline Eigen::Vector3d getGyroBias() const {
    return gyroBias_;
  }

  inline Eigen::Vector3d getAccelBias() const {
    return accelBias_;
  }

  inline Eigen::VectorXd getImuAugmentedParams() const {
    return extraParams_;
  }

  inline int modelId() const {
    return modelId_;
  }

  Eigen::VectorXd computeImuAugmentedParamsError() const {
    return ImuModelComputeAugmentedParamsError(modelId_, extraParams_);
  }

  void assignTo(okvis::ImuParameters *imuParams) {
    ImuModelAssignTo(modelId_, gyroBias_, accelBias_, extraParams_, imuParams);
  }

  inline void setGyroBias(const Eigen::Vector3d &bias) {
    gyroBias_ = bias;
  }

  inline void setAccelBias(const Eigen::Vector3d &bias) {
    accelBias_ = bias;
  }

  inline void setImuAugmentedParams(const Eigen::VectorXd& extraParams) {
    extraParams_ = extraParams;
  }

private:
  int modelId_;
  Eigen::Vector3d gyroBias_;
  Eigen::Vector3d accelBias_;
  Eigen::VectorXd extraParams_;
};

class ImuRig {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  inline ImuRig() {}

  inline int getModelId(int index) const {
    if ((int)imus_.size() > index) {
      return imus_[index].modelId();
    } else {
      return -1;
    }
  }

  int addImu(const okvis::ImuParameters& imuParams);

  inline const ImuModel &at(int imuId) const {
    return imus_.at(imuId);
  }

  inline ImuModel &at(int imuId) {
    return imus_.at(imuId);
  }

  inline int getImuParamsMinimalDim(int imu_id=0) const {
    return ImuModelGetMinimalDim(imus_[imu_id].modelId());
  }

  inline Eigen::VectorXd getImuAugmentedParams(int imu_id=0) const {
    return imus_.at(imu_id).getImuAugmentedParams();
  }

  inline Eigen::VectorXd computeImuAugmentedParamsError(int imu_id=0) const {
    return imus_.at(imu_id).computeImuAugmentedParamsError();
  }

  inline void setImuAugmentedParams(int imu_id, const Eigen::VectorXd& euclideanParams) {
    imus_.at(imu_id).setImuAugmentedParams(euclideanParams);
  }

  inline int getAugmentedMinimalDim(int imu_id) const {
    return ImuModelGetAugmentedMinimalDim(imus_[imu_id].modelId());
  }

  inline int getAugmentedDim(int imu_id) const {
    return ImuModelGetAugmentedDim(imus_[imu_id].modelId());
  }
private:
  std::vector<ImuModel> imus_;
};

/**
 * @brief getImuAugmentedStatesEstimate get augmented IMU parameters except for
 *     biases from parameter blocks.
 * @param imuAugmentedParameterPtrs[in]
 * @param extraParams[out]
 * @param imuModelId[in]
 */
void getImuAugmentedStatesEstimate(
     std::vector<std::shared_ptr<const okvis::ceres::ParameterBlock>> imuAugmentedParameterPtrs,
    Eigen::Matrix<double, Eigen::Dynamic, 1>* extraParams, int imuModelId);

}  // namespace swift_vio
#endif  // INCLUDE_SWIFT_VIO_IMU_RIG_HPP_
