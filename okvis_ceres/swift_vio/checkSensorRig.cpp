#include <swift_vio/checkSensorRig.hpp>

namespace swift_vio {
bool doesExtrinsicRepFitImuModel(const std::string& extrinsicRepName,
                                   const std::string& imuModel) {
  int extrinsicRepId = ExtrinsicRepNameToId(extrinsicRepName);
  int imuModelId = ImuModelNameToId(imuModel);
  switch (imuModelId) {
    case Imu_BG_BA_TG_TS_TA::kModelId:
      if (extrinsicRepId != Extrinsic_p_CB::kModelId) {
        LOG(ERROR) << "When IMU model is BG_BA_TG_TS_TA, the first camera's "
                        "extrinsic model should be P_CB!";
        return false;
      }
      break;
    case Imu_BG_BA::kModelId:
    case Imu_BG_BA_MG_TS_MA::kModelId:
    case ScaledMisalignedImu::kModelId:
      if (extrinsicRepId != Extrinsic_p_BC_q_BC::kModelId) {
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

bool doesExtrinsicRepFitOkvisBackend(
    const okvis::cameras::NCameraSystem& cameraSystem,
    EstimatorAlgorithm algorithm) {
  size_t numCameras = cameraSystem.numCameras();

  if (algorithm == EstimatorAlgorithm::OkvisEstimator || algorithm == EstimatorAlgorithm::SlidingWindowSmoother) {
    for (size_t index = 1u; index < numCameras; ++index) {
      std::string extrinsicRepName = cameraSystem.extrinsicRep(index);
      int extrinsicRepId =
          ExtrinsicRepNameToId(extrinsicRepName);
      if (extrinsicRepId == Extrinsic_p_C0C_q_C0C::kModelId) {
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
