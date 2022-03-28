#ifndef INCLUDE_SWIFT_VIO_CHECK_SENSOR_RIG_HPP_
#define INCLUDE_SWIFT_VIO_CHECK_SENSOR_RIG_HPP_

#include <swift_vio/imu/ImuRig.hpp>
#include <swift_vio/CameraRig.hpp>

namespace swift_vio {
bool doesExtrinsicRepFitImuModel(const std::string& extrinsicRepName,
                                   const std::string& imuModel);

bool doesExtrinsicRepFitOkvisBackend(
    const okvis::cameras::NCameraSystem& cameraSystem,
    EstimatorAlgorithm algorithm);
}  // namespace swift_vio

#endif // INCLUDE_SWIFT_VIO_CHECK_SENSOR_RIG_HPP_
