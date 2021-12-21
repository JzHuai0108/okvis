#include "swift_vio/CameraObservationModel.h"

#include <okvis/ceres/ReprojectionErrorBase.hpp>
#include <swift_vio/ceres/ChordalDistance.hpp>
#include <swift_vio/ceres/EpipolarFactor.hpp>
#include <swift_vio/ceres/RsReprojectionErrorAidp.hpp>
#include <swift_vio/ceres/RsReprojectionErrorPap.hpp>
#include <swift_vio/ceres/RsReprojectionError.hpp>

namespace swift_vio {
int CameraObservationModel::residualDim(int modelId) {
  switch (modelId) {
#define RESIDUAL_MODEL_CASE(Model)                                             \
  case Model::kModelId:                                                        \
    return Model::kNumResiduals;

    RESIDUAL_MODEL_SWITCH_CASES

#undef RESIDUAL_MODEL_CASE
  }
}

int CameraObservationModel::toLandmarkModelId(int modelId) {
  switch (modelId) {
  case okvis::ceres::ReprojectionError2dBase::kModelId:
  case okvis::ceres::RsReprojectionErrorBase::kModelId:
    return okvis::ceres::HomogeneousPointLocalParameterization::kModelId;
  case okvis::ceres::EpipolarFactorBase::kModelId:
    return -1;
  case okvis::ceres::ChordalDistanceBase::kModelId:
  case okvis::ceres::RsReprojectionErrorPapBase::kModelId:
    return swift_vio::ParallaxAngleParameterization::kModelId;
  case okvis::ceres::RsReprojectionErrorAidpBase::kModelId:
    return swift_vio::InverseDepthParameterization::kModelId;
  }
  return -1;
}

} // namespace swift_vio
