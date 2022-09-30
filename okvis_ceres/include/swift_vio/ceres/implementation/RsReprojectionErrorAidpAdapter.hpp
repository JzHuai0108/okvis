
/**
 * @file implementation/RsReprojectionErrorAidpAdapter.hpp
 * @brief Header implementation file for the RsReprojectionErrorAidpAdapter
 * adapter class.
 * @author Jianzhu Huai
 */
#include "ceres/internal/autodiff.h"

#include <okvis/kinematics/Transformation.hpp>
#include <okvis/kinematics/operators.hpp>

#include <swift_vio/Measurements.hpp>
#include <swift_vio/ceres/JacobianHelpers.hpp>
#include <swift_vio/imu/SimpleImuOdometry.hpp>
#include <swift_vio/imu/SimpleImuPropagationJacobian.hpp>

namespace okvis {
namespace ceres {
template <class GEOMETRY_TYPE>
RsReprojectionErrorAidpAdapter<
    GEOMETRY_TYPE>::RsReprojectionErrorAidpAdapter() {}

template <class GEOMETRY_TYPE>
RsReprojectionErrorAidpAdapter<GEOMETRY_TYPE>::RsReprojectionErrorAidpAdapter(
    const swift_vio::CameraIdentifier &targetCamera,
    const swift_vio::CameraIdentifier &hostCamera,
    const measurement_t &measurement, const covariance_t &covariance,
    std::shared_ptr<const camera_geometry_t> targetCameraGeometry,
    std::shared_ptr<const okvis::ImuMeasurementDeque> imuMeasCanopy,
    std::shared_ptr<const okvis::ImuParameters> imuParameters,
    okvis::Time targetStateTime, okvis::Time targetImageTime)
    : targetCamera_(targetCamera), hostCamera_(hostCamera),
      costFunction_(measurement, covariance, targetCameraGeometry,
                    imuMeasCanopy, imuParameters, targetStateTime,
                    targetImageTime) {
  if (imuParameters->model_name == "BG_BA") {
    Mg0_ = costFunction_.ImuParameters()->gyroCorrectionMatrix().data();
    Ts0_ = costFunction_.ImuParameters()->gyroGSensitivity().data();
    Ma0_ = costFunction_.ImuParameters()->accelCorrectionMatrix().data();
  } else {
    Mg0_ = nullptr;
    Ts0_ = nullptr;
    Ma0_ = nullptr;
  }
}

template <class GEOMETRY_TYPE>
void RsReprojectionErrorAidpAdapter<GEOMETRY_TYPE>::setCovariance(
    const covariance_t &covariance) {
  costFunction_.setCovariance(covariance);
}

template <class GEOMETRY_TYPE>
bool RsReprojectionErrorAidpAdapter<GEOMETRY_TYPE>::Evaluate(
    double const *const *parameters, double *residuals,
    double **jacobians) const {
  return EvaluateWithMinimalJacobians(parameters, residuals, jacobians, NULL);
}

template <class GEOMETRY_TYPE>
bool RsReprojectionErrorAidpAdapter<GEOMETRY_TYPE>::
    EvaluateWithMinimalJacobians(double const *const *parameters,
                                 double *residuals, double **jacobians,
                                 double **jacobiansMinimal) const {
  return EvaluateWithMinimalJacobiansAnalytic(parameters, residuals, jacobians,
                                              jacobiansMinimal);
}

template <class GEOMETRY_TYPE>
void RsReprojectionErrorAidpAdapter<
    GEOMETRY_TYPE>::setParameterBlockAndResidualSizes() {
  AddParameterBlock(7);
  AddParameterBlock(4);
  if (targetCamera_.frameId != hostCamera_.frameId) {
    AddParameterBlock(7);
  }
  AddParameterBlock(7);
  if (targetCamera_.cameraIndex != hostCamera_.cameraIndex) {
    AddParameterBlock(7);
  }
  AddParameterBlock(GEOMETRY_TYPE::NumIntrinsics);
  AddParameterBlock(1);
  AddParameterBlock(1);
  AddParameterBlock(3);
  AddParameterBlock(6);
  if (Mg0_ == nullptr) {
    AddParameterBlock(9);
    AddParameterBlock(9);
    AddParameterBlock(6);
  }
  SetNumResiduals(2);
}

template <class GEOMETRY_TYPE>
void RsReprojectionErrorAidpAdapter<GEOMETRY_TYPE>::fullParameterList(
    double const *const *parameters,
    std::vector<double const *> *fullparameters) const {
  fullparameters->reserve(kernel_t::numParameterBlocks());
  for (int i = 0; i < kernel_t::Index::T_WBh; ++i) {
    fullparameters->push_back(parameters[i]);
  }
  int indexShift = 0;
  if (targetCamera_.frameId == hostCamera_.frameId) {
    fullparameters->push_back(parameters[kernel_t::Index::T_WBt]);
    indexShift = -1;
  } else {
    fullparameters->push_back(parameters[kernel_t::Index::T_WBh]);
  }
  fullparameters->push_back(parameters[kernel_t::Index::T_BCt + indexShift]);
  if (targetCamera_.cameraIndex == hostCamera_.cameraIndex) {
    fullparameters->push_back(parameters[kernel_t::Index::T_BCt + indexShift]);
    --indexShift;
  } else {
    fullparameters->push_back(parameters[kernel_t::Index::T_BCh + indexShift]);
  }
  for (int i = kernel_t::Index::Intrinsics; i < kernel_t::Index::M_gi;
       ++i) {
    fullparameters->push_back(parameters[i + indexShift]);
  }
  if (Mg0_) { // BG_BA model
    fullparameters->push_back(Mg0_);
    fullparameters->push_back(Ts0_);
    fullparameters->push_back(Ma0_);
  } else {
    for (int i = kernel_t::Index::M_gi; i < kernel_t::Index::M_ai + 1; ++i) {
      fullparameters->push_back(parameters[i + indexShift]);
    }
  }
}

template <class GEOMETRY_TYPE>
void RsReprojectionErrorAidpAdapter<GEOMETRY_TYPE>::fullParameterList2(
    double const *const *parameters,
    std::vector<double const *> *fullparameters) const {
  fullparameters->reserve(kernel_t::numParameterBlocks());
  for (size_t i = 0; i < parameterBlocks(); ++i) {
    fullparameters->push_back(parameters[i]);
  }
  if (Mg0_) { // BG_BA model
    fullparameters->push_back(Mg0_);
    fullparameters->push_back(Ts0_);
    fullparameters->push_back(Ma0_);
  }
  int indexShift = 0;
  if (targetCamera_.frameId == hostCamera_.frameId) {
    fullparameters->insert(fullparameters->begin() + kernel_t::Index::T_WBh,
                           parameters[kernel_t::Index::T_WBt]);
    indexShift = -1;
  }

  if (targetCamera_.cameraIndex == hostCamera_.cameraIndex) {
    fullparameters->insert(fullparameters->begin() + kernel_t::Index::T_BCh,
                           parameters[kernel_t::Index::T_BCt + indexShift]);
  }
}


template <class GEOMETRY_TYPE>
void RsReprojectionErrorAidpAdapter<GEOMETRY_TYPE>::fullJacobianList(
    double **jacobians, double *j_T_WBh, double *j_T_BCh,
    std::vector<double *> *fullJacobians) const {
  size_t numJacBlocks = parameterBlocks();
  fullJacobians->reserve(kernel_t::numParameterBlocks());
  for (size_t i = 0; i < numJacBlocks; ++i) {
    fullJacobians->push_back(jacobians[i]);
  }
  if (Mg0_) { // BG_BA model
    fullJacobians->push_back(nullptr);
    fullJacobians->push_back(nullptr);
    fullJacobians->push_back(nullptr);
  }
  if (targetCamera_.frameId == hostCamera_.frameId) {
    fullJacobians->insert(fullJacobians->begin() + kernel_t::Index::T_WBh, j_T_WBh);
  }
  if (targetCamera_.cameraIndex == hostCamera_.cameraIndex) {
    fullJacobians->insert(fullJacobians->begin() + kernel_t::Index::T_BCh, j_T_BCh);
  }
}

template <class GEOMETRY_TYPE>
template <int ParamDim>
void RsReprojectionErrorAidpAdapter<GEOMETRY_TYPE>::uniqueJacobians(
    const std::vector<double *> &fullJacobians) const {
  if (targetCamera_.frameId == hostCamera_.frameId &&
          fullJacobians[kernel_t::Index::T_WBt]) {
    Eigen::Map<Eigen::Matrix<double, kNumResiduals, ParamDim, Eigen::RowMajor>> j_T_WBt(
        fullJacobians[kernel_t::Index::T_WBt]);
    Eigen::Map<const Eigen::Matrix<double, kNumResiduals, ParamDim, Eigen::RowMajor>> j_T_WBh(
        fullJacobians[kernel_t::Index::T_WBh]);
    j_T_WBt += j_T_WBh;
  }
  if (targetCamera_.cameraIndex == hostCamera_.cameraIndex &&
          fullJacobians[kernel_t::Index::T_BCt]) {
    Eigen::Map<Eigen::Matrix<double, kNumResiduals, ParamDim, Eigen::RowMajor>> j_T_BCt(
        fullJacobians[kernel_t::Index::T_BCt]);
    Eigen::Map<const Eigen::Matrix<double, kNumResiduals, ParamDim, Eigen::RowMajor>> j_T_BCh(
        fullJacobians[kernel_t::Index::T_BCh]);
    j_T_BCt += j_T_BCh;
  }
}

template <class GEOMETRY_TYPE>
bool RsReprojectionErrorAidpAdapter<GEOMETRY_TYPE>::
    EvaluateWithMinimalJacobiansAnalytic(double const *const *parameters,
                                         double *residuals, double **jacobians,
                                         double **jacobiansMinimal) const {
  std::vector<double const *> fullparameters;
  fullParameterList(parameters, &fullparameters);

  // Check
//  std::vector<double const *> fullparameters2;
//  fullParameterList2(parameters, &fullparameters2);
//  for (size_t i = 0; i < fullparameters.size(); ++i) {
//    CHECK_EQ(fullparameters[i], fullparameters2[i]);
//  }

  Eigen::Matrix<double, kNumResiduals, 7, Eigen::RowMajor> j_T_WBh, j_T_BCh;
  Eigen::Matrix<double, kNumResiduals, 6, Eigen::RowMajor> j_T_WBh_min,
      j_T_BCh_min;
  std::vector<double *> fullJacobians, fullJacobiansMinimal;
  if (jacobians)
    fullJacobianList(jacobians, j_T_WBh.data(), j_T_BCh.data(), &fullJacobians);
  if (jacobiansMinimal)
    fullJacobianList(jacobiansMinimal, j_T_WBh_min.data(), j_T_BCh_min.data(),
                     &fullJacobiansMinimal);
  bool result = costFunction_.EvaluateWithMinimalJacobiansAnalytic(
      fullparameters.data(), residuals, fullJacobians.data(), fullJacobiansMinimal.data());
  if (jacobians)
    uniqueJacobians<7>(fullJacobians);
  if (jacobiansMinimal)
    uniqueJacobians<6>(fullJacobiansMinimal);
  return result;
}

// This evaluates the error term and additionally computes
// the Jacobians in the minimal internal representation via autodiff
template <class GEOMETRY_TYPE>
bool RsReprojectionErrorAidpAdapter<GEOMETRY_TYPE>::
    EvaluateWithMinimalJacobiansAutoDiff(double const *const *parameters,
                                         double *residuals, double **jacobians,
                                         double **jacobiansMinimal) const {
  std::vector<double const *> fullparameters;
  fullParameterList(parameters, &fullparameters);
  Eigen::Matrix<double, kNumResiduals, 7> j_T_WBh, j_T_BCh;
  Eigen::Matrix<double, kNumResiduals, 6> j_T_WBh_min, j_T_BCh_min;
  std::vector<double *> fullJacobians, fullJacobiansMinimal;
  fullJacobianList(jacobians, j_T_WBh.data(), j_T_BCh.data(), &fullJacobians);
  fullJacobianList(jacobiansMinimal, j_T_WBh_min.data(), j_T_BCh_min.data(),
                   &fullJacobiansMinimal);
  bool result = costFunction_.EvaluateWithMinimalJacobiansAutoDiff(
      fullparameters.data(), residuals, fullJacobians.data(), fullJacobiansMinimal.data());
  uniqueJacobians<7>(fullJacobians);
  uniqueJacobians<6>(fullJacobiansMinimal);
  return result;
}
} // namespace ceres
} // namespace okvis
