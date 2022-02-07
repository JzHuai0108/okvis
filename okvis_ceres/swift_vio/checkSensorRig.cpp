#include <swift_vio/checkSensorRig.hpp>

namespace swift_vio {
bool doesExtrinsicModelFitImuModel(const std::string& extrinsicModel,
                                   const std::string& imuModel) {
  int extrinsicModelId = ExtrinsicModelNameToId(extrinsicModel, nullptr);
  int imuModelId = ImuModelNameToId(imuModel);
  switch (imuModelId) {
    case Imu_BG_BA_TG_TS_TA::kModelId:
      if (extrinsicModelId != Extrinsic_p_CB::kModelId) {
        LOG(ERROR) << "When IMU model is BG_BA_TG_TS_TA, the first camera's "
                        "extrinsic model should be P_CB!";
        return false;
      }
      break;
    case Imu_BG_BA::kModelId:
    case ScaledMisalignedImu::kModelId:
      if (extrinsicModelId != Extrinsic_p_BC_q_BC::kModelId) {
        LOG(ERROR) << "When IMU model is BG_BA or ScaledMisalignedImu, the "
                        "first camera's extrinsic model should be P_BC_Q_BC!";
        return false;
      }
      break;
    default:
      break;
  }
  return true;
}

bool doesExtrinsicModelFitOkvisBackend(
    const okvis::cameras::NCameraSystem& cameraSystem,
    EstimatorAlgorithm algorithm) {
  size_t numCameras = cameraSystem.numCameras();

  if (algorithm == EstimatorAlgorithm::SlidingWindowSmoother) {
    for (size_t index = 1u; index < numCameras; ++index) {
      std::string extrinsicModel = cameraSystem.extrinsicOptRep(index);
      int extrinsicModelId =
          ExtrinsicModelNameToId(extrinsicModel, nullptr);
      if (extrinsicModelId == Extrinsic_p_C0C_q_C0C::kModelId) {
        LOG(FATAL) << "When the OKVIS backend is used, the second camera's "
                      "extrinsic model should be P_BC_Q_BC instead of "
                      "P_C0C_Q_C0C which leads "
                      "to wrong extrinsics in frontend!";
        return false;
      }
    }
  }
  return true;
}
}  // namespace swift_vio
