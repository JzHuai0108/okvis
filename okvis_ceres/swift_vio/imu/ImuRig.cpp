#include <swift_vio/imu/ImuRig.hpp>

namespace swift_vio {
int ImuRig::addImu(const okvis::ImuParameters& imuParams) {
  int modelId = ImuModelNameToId(imuParams.model_name);
  Eigen::Matrix<double, Eigen::Dynamic, 1> extraParams;
  std::stringstream ss;
  switch (modelId) {
    case Imu_BG_BA_MG_TS_MA::kModelId:
      extraParams.resize(Imu_BG_BA_MG_TS_MA::kAugmentedDim, 1);
      extraParams.head<9>() = imuParams.gyroCorrectionMatrix();
      extraParams.segment<9>(9) = imuParams.gyroGSensitivity();
      extraParams.segment<6>(18) = imuParams.accelCorrectionMatrix();
      break;
    case Imu_BG_BA::kModelId:
      extraParams.resize(0);
      break;
    default:
      ss << "Conversion from ImuParameters to IMU model " << modelId << " is not supported yet!";
      std::runtime_error(ss.str());
      break;
  }
  imus_.emplace_back(modelId, imuParams.initialGyroBias(), imuParams.initialAccelBias(), extraParams);
  return static_cast<int>(imus_.size()) - 1;
}

void getImuAugmentedStatesEstimate(
    std::vector<std::shared_ptr<const okvis::ceres::ParameterBlock>>
        imuAugmentedParameterPtrs,
    Eigen::Matrix<double, Eigen::Dynamic, 1>* extraParams, int imuModelId) {
  const int augmentedDim = ImuModelGetAugmentedDim(imuModelId);
  extraParams->resize(augmentedDim);
  size_t offset = 0u;
  for (const auto paramPtr : imuAugmentedParameterPtrs) {
    memcpy(extraParams->data() + offset, paramPtr->parameters(),
           sizeof(double) * paramPtr->dimension());
    offset += paramPtr->dimension();
  }
}
}  // namespace swift_vio
