
/**
 * @file implementation/RsReprojectionErrorAidp.hpp
 * @brief Header implementation file for the RsReprojectionErrorAidp class.
 * @author Jianzhu Huai
 */
#include "ceres/internal/autodiff.h"

#include <okvis/kinematics/Transformation.hpp>
#include <okvis/kinematics/operators.hpp>

#include <swift_vio/ceres/JacobianHelpers.hpp>
#include <swift_vio/Measurements.hpp>
#include <swift_vio/MultipleTransformPointJacobian.hpp>
#include <swift_vio/imu/SimpleImuOdometry.hpp>
#include <swift_vio/imu/SimpleImuPropagationJacobian.hpp>

namespace okvis {
namespace ceres {
template <class GEOMETRY_TYPE>
RsReprojectionErrorAidp<GEOMETRY_TYPE>::RsReprojectionErrorAidp() {}

template <class GEOMETRY_TYPE>
RsReprojectionErrorAidp<GEOMETRY_TYPE>::RsReprojectionErrorAidp(
    const measurement_t& measurement,
    const covariance_t& covariance,
    std::shared_ptr<const camera_geometry_t> targetCameraGeometry,
    std::shared_ptr<const okvis::ImuMeasurementDeque> imuMeasCanopy,
    std::shared_ptr<const okvis::ImuParameters> imuParameters,
    okvis::Time targetStateTime, okvis::Time targetImageTime)
    : imuMeasCanopy_(imuMeasCanopy),
      imuParameters_(imuParameters),
      targetCameraGeometry_(targetCameraGeometry),
      targetStateTime_(targetStateTime),
      targetImageTime_(targetImageTime) {
  measurement_ = measurement;
  setCovariance(covariance);
}

template <class GEOMETRY_TYPE>
void RsReprojectionErrorAidp<GEOMETRY_TYPE>::
    setCovariance(const covariance_t& covariance) {
  information_ = covariance.inverse();
  covariance_ = covariance;
  // perform the Cholesky decomposition on order to obtain the correct error
  // weighting
  Eigen::LLT<Eigen::Matrix2d> lltOfInformation(information_);
  squareRootInformation_ = lltOfInformation.matrixL().transpose();
}

template <class GEOMETRY_TYPE>
bool RsReprojectionErrorAidp<GEOMETRY_TYPE>::
    Evaluate(double const* const* parameters, double* residuals,
             double** jacobians) const {
  return EvaluateWithMinimalJacobians(parameters, residuals, jacobians, NULL);
}

template <class GEOMETRY_TYPE>
bool RsReprojectionErrorAidp<GEOMETRY_TYPE>::
    EvaluateWithMinimalJacobians(double const* const* parameters,
                                 double* residuals, double** jacobians,
                                 double** jacobiansMinimal) const {
  return EvaluateWithMinimalJacobiansAnalytic(parameters, residuals, jacobians,
                                              jacobiansMinimal);
}

template <class GEOMETRY_TYPE>
bool RsReprojectionErrorAidp<GEOMETRY_TYPE>::
    EvaluateWithMinimalJacobiansAnalytic(double const* const* parameters,
                                 double* residuals, double** jacobians,
                                 double** jacobiansMinimal) const {
  Eigen::Map<const Eigen::Vector3d> p_WBt0(parameters[Index::T_WBt]);
  Eigen::Map<const Eigen::Quaterniond> q_WBt0(parameters[Index::T_WBt] + 3);

  Eigen::Matrix<double, 4, 4, Eigen::RowMajor> dh_dl;

  Eigen::Map<const Eigen::Matrix<double, 4, 1>> hp_Ch(parameters[Index::AIDP]);

  Eigen::Map<const Eigen::Vector3d> p_BCt(parameters[Index::T_BCt]);
  Eigen::Map<const Eigen::Quaterniond> q_BCt(parameters[Index::T_BCt] + 3);
  okvis::kinematics::Transformation T_BCt(p_BCt, q_BCt);

  Eigen::Map<const Eigen::Vector3d> p_BCh(parameters[Index::T_BCh]);
  Eigen::Map<const Eigen::Quaterniond> q_BCh(parameters[Index::T_BCh] + 3);
  okvis::kinematics::Transformation T_BCh(p_BCh, q_BCh);

  Eigen::Map<const Eigen::Vector3d> p_WBh(parameters[Index::T_WBh]);
  Eigen::Map<const Eigen::Quaterniond> q_WBh(parameters[Index::T_WBh] + 3);
  okvis::kinematics::Transformation T_WBh(p_WBh, q_WBh);

  Eigen::Matrix<double, -1, 1> intrinsics =
      Eigen::Map<const Eigen::Matrix<double, kIntrinsicDim, 1>>(
          parameters[Index::Intrinsics]);

  double readoutTime = parameters[Index::ReadoutTime][0]; //tr
  double cameraTd = parameters[Index::CameraTd][0];       //td

  Eigen::Map<const Eigen::Matrix<double, 3, 1>> v_WB0(parameters[Index::Speed]);
  Eigen::Matrix<double, 6, 1> bgBa = Eigen::Map<const Eigen::Matrix<double, 6, 1>>(parameters[Index::Biases]);
  Eigen::Matrix<double, 3, 1> speed = v_WB0;

  Eigen::Map<const Eigen::Matrix<double, 9, 1>> Tg(parameters[Index::M_gi]); // not used for now.
  Eigen::Map<const Eigen::Matrix<double, 9, 1>> Ts(parameters[Index::M_si]); // not used for now.
  Eigen::Map<const Eigen::Matrix<double, 6, 1>> Ta(parameters[Index::M_ai]); // not used for now.

  double ypixel(measurement_[1]);
  uint32_t height = targetCameraGeometry_->imageHeight();
  double kpN = ypixel / height - 0.5;
  double relativeFeatureTime = cameraTd + readoutTime * kpN + ( targetImageTime_- targetStateTime_).toSec(); 
  std::pair<Eigen::Matrix<double, 3, 1>, Eigen::Quaternion<double>> pair_T_WBt(p_WBt0, q_WBt0);

  const okvis::Time t_start = targetStateTime_;
  const okvis::Time t_end = targetStateTime_ + okvis::Duration(relativeFeatureTime);
  const double wedge = 5e-8;
  if (relativeFeatureTime >= wedge){
    swift_vio::ode::predictStates(*imuMeasCanopy_, imuParameters_->gravity(), pair_T_WBt,
                                  speed, bgBa, t_start, t_end);
  } else if (relativeFeatureTime <= -wedge){
    swift_vio::ode::predictStatesBackward(*imuMeasCanopy_, imuParameters_->gravity(), pair_T_WBt,
                                          speed, bgBa, t_start, t_end);
  }
  okvis::kinematics::Transformation T_WBt(pair_T_WBt.first, pair_T_WBt.second);
  Eigen::Quaterniond q_WBt = pair_T_WBt.second;
  Eigen::Matrix3d C_WBt = q_WBt.toRotationMatrix();

  swift_vio::MultipleTransformPointJacobian mtpj({T_BCt, T_WBt, T_WBh, T_BCh}, {-1, -1, 1, 1}, hp_Ch);
  Eigen::Vector4d hp_Ct = mtpj.evaluate();

  // calculate the reprojection error
  measurement_t kp;
  Eigen::Matrix<double, 2, 3> Jh;
  Eigen::Matrix<double, 2, 3> Jh_weighted;
  Eigen::Matrix<double, 2, Eigen::Dynamic> Jpi;
  Eigen::Matrix<double, 2, Eigen::Dynamic> Jpi_weighted;
  if (jacobians != NULL){
    targetCameraGeometry_->projectWithExternalParameters(hp_Ct.head<3>(), intrinsics, &kp, &Jh, &Jpi);
    Jh_weighted = squareRootInformation_ * Jh;
    Jpi_weighted = squareRootInformation_ * Jpi;
  } else {
    targetCameraGeometry_->projectWithExternalParameters(hp_Ct.head<3>(), intrinsics, &kp, &Jh, &Jpi);
  }
  measurement_t error = kp - measurement_;

  // weight:
  measurement_t weighted_error = squareRootInformation_ * error;

  residuals[0] = weighted_error[0];
  residuals[1] = weighted_error[1];

  bool valid = true;
  if (fabs(hp_Ct[3]) > 1.0e-8){
    Eigen::Vector3d p_C = hp_Ct.template head<3>() / hp_Ct[3];
    if (p_C[2] < 0.2){ // 20 cm - not very generic... but reasonable
      valid = false;
    }
  }

  if (jacobians != NULL){
    if (!valid){
      setJacobiansZero(jacobians, jacobiansMinimal);
      return true;
    }

    mtpj.computeJacobians();

    okvis::ImuMeasurement queryValue;
    swift_vio::ode::interpolateInertialData(*imuMeasCanopy_, t_end, queryValue);
    queryValue.measurement.gyroscopes -= bgBa.head<3>();
    swift_vio::SimpleImuPropagationJacobian sipj(t_start, t_end,
                                                 T_WBt,
                                                 speed,
                                                 queryValue.measurement.gyroscopes);
    Eigen::Matrix<double, 6, 1> dT_WB_dt;
    Eigen::Vector3d dp_WB_dt;
    sipj.dp_dt(&dp_WB_dt);
    Eigen::Vector3d dq_WB_dt;
    sipj.dtheta_dt(&dq_WB_dt);
    dT_WB_dt.head<3>() = dp_WB_dt;
    dT_WB_dt.tail<3>() = dq_WB_dt;

    Eigen::Matrix3d dp_WB_dvW;
    sipj.dp_dv_WB(&dp_WB_dvW);

    Eigen::Matrix3d Phi_pq;
    sipj.Phi_pq(p_WBt0, v_WB0, imuParameters_->gravity(), &Phi_pq);

    Eigen::Vector3d dhC_td = mtpj.dp_dT(1).topRows<3>() * dT_WB_dt;
    Eigen::Matrix3d dhC_vW = mtpj.dp_dT(1).topRows<3>().leftCols<3>() * dp_WB_dvW;
    Eigen::Matrix3d dtheta_dbg = - C_WBt * relativeFeatureTime;
    Eigen::Matrix3d dhC_bg = mtpj.dp_dT(1).topRows<3>().rightCols<3>() * dtheta_dbg;
    Eigen::Matrix<double, 3, 6> dhC_biases;
    dhC_biases.rightCols<3>().setZero();
    dhC_biases.leftCols<3>() = dhC_bg;

    if (jacobians[0] != NULL)
    {
      Eigen::Matrix<double, 2, 6, Eigen::RowMajor> J0_minimal;
      J0_minimal = Jh_weighted * mtpj.dp_dT(1).topRows<3>();
      J0_minimal.rightCols<3>() += J0_minimal.leftCols<3>() * Phi_pq;

      Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> J0(jacobians[0]);
      J0.leftCols<6>() = J0_minimal;
      J0.col(6).setZero();

      if (jacobiansMinimal != NULL)
      {
        if (jacobiansMinimal[0] != NULL)
        {
          Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>>
              J0_minimal_mapped(jacobiansMinimal[0]);
          J0_minimal_mapped = J0_minimal;
        }
      }
    }

    if (jacobians[1] != NULL)
    {
      Eigen::Map<Eigen::Matrix<double, 2, 4, Eigen::RowMajor>> J1(jacobians[1]);
      J1 = Jh_weighted * mtpj.dp_dpoint().topRows<3>();
      if (jacobiansMinimal != NULL)
      {
        if (jacobiansMinimal[1] != NULL)
        {
          Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>>
              J1_minimal_mapped(jacobiansMinimal[1]);

          Eigen::Matrix<double, 4, 3, Eigen::RowMajor> S;
          swift_vio::InverseDepthParameterization::plusJacobian(nullptr, S.data());
          J1_minimal_mapped = J1 * S;
        }
      }
    }

    if (jacobians[2] != NULL)
    {
      Eigen::Matrix<double, 2, 6, Eigen::RowMajor> J2_minimal;
      J2_minimal = Jh_weighted * mtpj.dp_dT(2).topRows<3>();
      Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> J2(jacobians[2]);

      J2.leftCols<6>() = J2_minimal;
      J2.col(6).setZero();

      if (jacobiansMinimal != NULL)
      {
        if (jacobiansMinimal[2] != NULL)
        {
          Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>>
              J2_minimal_mapped(jacobiansMinimal[2]);
          J2_minimal_mapped = J2_minimal;
        }
      }
    }

    if (jacobians[3] != NULL)
    {
      Eigen::Matrix<double, 2, 6, Eigen::RowMajor>
          J3_minimal = Jh_weighted * mtpj.dp_dT(0).topRows<3>();
      Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> J3(jacobians[3]);

      J3.leftCols<6>() = J3_minimal;
      J3.col(6).setZero();
      if (jacobiansMinimal != NULL)
      {
        if (jacobiansMinimal[3] != NULL)
        {
          Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>>
              J3_minimal_mapped(jacobiansMinimal[3]);
          J3_minimal_mapped = J3_minimal;
        }
      }
    }

    if (jacobians[4] != NULL)
    {
      Eigen::Matrix<double, 2, 6, Eigen::RowMajor>
          J4_minimal = Jh_weighted * mtpj.dp_dT(3).topRows<3>();
      Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> J4(jacobians[4]);
      J4.leftCols<6>() = J4_minimal;
      J4.col(6).setZero();
      if (jacobiansMinimal != NULL)
      {
        if (jacobiansMinimal[4] != NULL)
        {
          Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>>
              J4_minimal_mapped(jacobiansMinimal[4]);
          J4_minimal_mapped = J4_minimal;
        }
      }
    }

    // camera intrinsics
    if (jacobians[5] != NULL)
    {
      Eigen::Map<Eigen::Matrix<double, 2, GEOMETRY_TYPE::NumIntrinsics, Eigen::RowMajor>> J5(jacobians[5]);
      J5 = Jpi_weighted;
      if (jacobiansMinimal != NULL)
      {
        if (jacobiansMinimal[5] != NULL)
        {
          Eigen::Map<Eigen::Matrix<double, 2, GEOMETRY_TYPE::NumIntrinsics, Eigen::RowMajor>> J5_minimal_mapped(jacobiansMinimal[5]);
          J5_minimal_mapped = J5;
        }
      }
    }

    //tr
    if (jacobians[6] != NULL)
    {
      Eigen::Map<Eigen::Matrix<double, 2, 1>> J6(jacobians[6]);
      J6 = Jh_weighted * dhC_td * kpN;
      if (jacobiansMinimal != NULL)
      {
        if (jacobiansMinimal[6] != NULL)
        {
          Eigen::Map<Eigen::Matrix<double, 2, 1>>
              J6_minimal_mapped(jacobiansMinimal[6]);
          J6_minimal_mapped = J6;
        }
      }
    }

    // t_d
    if (jacobians[7] != NULL)
    {
      Eigen::Map<Eigen::Matrix<double, 2, 1>> J7(jacobians[7]);
      J7 = Jh_weighted * dhC_td;
      if (jacobiansMinimal != NULL)
      {
        if (jacobiansMinimal[7] != NULL)
        {
          Eigen::Map<Eigen::Matrix<double, 2, 1>> J7_minimal_mapped(
              jacobiansMinimal[7]);
          J7_minimal_mapped = J7;
        }
      }
    }

    // speed and gyro biases and accel biases
    if (jacobians[Index::Speed] != NULL)
    {
      Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> J(jacobians[Index::Speed]);
      J = Jh_weighted * dhC_vW;
      if (jacobiansMinimal != NULL)
      {
        if (jacobiansMinimal[Index::Speed] != NULL)
        {
          Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>>
              J_minimal_mapped(jacobiansMinimal[Index::Speed]);
          J_minimal_mapped = J;
        }
      }
    }

    if (jacobians[Index::Biases] != NULL)
    {
      Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> J(jacobians[Index::Biases]);
      J = Jh_weighted * dhC_biases;
      if (jacobiansMinimal != NULL)
      {
        if (jacobiansMinimal[Index::Biases] != NULL)
        {
          Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>>
              J_minimal_mapped(jacobiansMinimal[Index::Biases]);
          J_minimal_mapped = J;
        }
      }
    }

    // M_gi
    if (jacobians[Index::M_gi] != NULL)
    {
      Eigen::Map<Eigen::Matrix<double, 2, 9, Eigen::RowMajor>> J(jacobians[Index::M_gi]);
      J.setZero();
      if (jacobiansMinimal != NULL)
      {
        if (jacobiansMinimal[Index::M_gi] != NULL)
        {
          Eigen::Map<Eigen::Matrix<double, 2, 9, Eigen::RowMajor>>
              J_minimal_mapped(jacobiansMinimal[Index::M_gi]);
          J_minimal_mapped = J;
        }
      }
    }

    // M_si
    if (jacobians[Index::M_si] != NULL)
    {
      Eigen::Map<Eigen::Matrix<double, 2, 9, Eigen::RowMajor>> J(jacobians[Index::M_si]);
      J.setZero();
      if (jacobiansMinimal != NULL)
      {
        if (jacobiansMinimal[Index::M_si] != NULL)
        {
          Eigen::Map<Eigen::Matrix<double, 2, 9, Eigen::RowMajor>>
              J_minimal_mapped(jacobiansMinimal[Index::M_si]);
          J_minimal_mapped = J;
        }
      }
    }

    // M_ai
    if (jacobians[Index::M_ai] != NULL)
    {
      Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> J(jacobians[Index::M_ai]);
      J.setZero();
      if (jacobiansMinimal != NULL)
      {
        if (jacobiansMinimal[Index::M_ai] != NULL)
        {
          Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>>
              J_minimal_mapped(jacobiansMinimal[Index::M_ai]);
          J_minimal_mapped = J;
        }
      }
    }
  }
  return true;
}

// This evaluates the error term and additionally computes
// the Jacobians in the minimal internal representation via autodiff
template <class GEOMETRY_TYPE>
bool RsReprojectionErrorAidp<GEOMETRY_TYPE>::
    EvaluateWithMinimalJacobiansAutoDiff(double const* const* parameters,
                                         double* residuals, double** jacobians,
                                         double** jacobiansMinimal) const
{
  const int numOutputs = 4;
  double deltaTWSt[6] = {0};
  double deltaTWSh[6] = {0};
  double deltaTSCt[6] = {0};
  double deltaTSCh[6] = {0};
  double const *const expandedParams[] = {
      parameters[Index::T_WBt],
      parameters[Index::AIDP],
      parameters[Index::T_WBh],
      parameters[Index::T_BCt],
      parameters[Index::T_BCh],
      parameters[Index::ReadoutTime],
      parameters[Index::CameraTd],
      parameters[Index::Speed],
      parameters[Index::Biases],
      parameters[Index::M_gi], parameters[Index::M_si], parameters[Index::M_ai],
      deltaTWSt, deltaTWSh,
      deltaTSCt, deltaTSCh};

  double php_C[numOutputs];
  Eigen::Matrix<double, numOutputs, 7, Eigen::RowMajor> dhC_deltaTWSt_full;
  Eigen::Matrix<double, numOutputs, 7, Eigen::RowMajor> dhC_deltaTWSh_full;
  Eigen::Matrix<double, numOutputs, 4, Eigen::RowMajor> dhC_dlCh;
  Eigen::Matrix<double, numOutputs, 7, Eigen::RowMajor> dhC_dExtrinsict_full;
  Eigen::Matrix<double, numOutputs, 7, Eigen::RowMajor> dhC_dExtrinsich_full;

  Eigen::Matrix<double, numOutputs, GEOMETRY_TYPE::NumIntrinsics, Eigen::RowMajor> dhC_Intrinsic;
  Eigen::Matrix<double, numOutputs, 1> dhC_tr;
  Eigen::Matrix<double, numOutputs, 1> dhC_td;
  Eigen::Matrix<double, numOutputs, 3, Eigen::RowMajor> dhC_speed;
  Eigen::Matrix<double, numOutputs, 6, Eigen::RowMajor> dhC_biases;
  Eigen::Matrix<double, numOutputs, 6, Eigen::RowMajor> dhC_deltaTWSt;
  Eigen::Matrix<double, numOutputs, 6, Eigen::RowMajor> dhC_deltaTWSh;
  Eigen::Matrix<double, numOutputs, 6, Eigen::RowMajor> dhC_dExtrinsict;
  Eigen::Matrix<double, numOutputs, 6, Eigen::RowMajor> dhC_dExtrinsich;

  Eigen::Matrix<double, numOutputs, 9, Eigen::RowMajor> dhC_tgi;
  Eigen::Matrix<double, numOutputs, 9, Eigen::RowMajor> dhC_tsi;
  Eigen::Matrix<double, numOutputs, 6, Eigen::RowMajor> dhC_tai;

  dhC_Intrinsic.setZero();
  double *dpC_deltaAll[] = {
      dhC_deltaTWSt_full.data(),
      dhC_dlCh.data(),
      dhC_deltaTWSh_full.data(),
      dhC_dExtrinsict_full.data(),
      dhC_dExtrinsich_full.data(),
      dhC_tr.data(),
      dhC_td.data(),
      dhC_speed.data(), dhC_biases.data(),
      dhC_tgi.data(), dhC_tsi.data(), dhC_tai.data(),
      dhC_deltaTWSt.data(), dhC_deltaTWSh.data(),
      dhC_dExtrinsict.data(), dhC_dExtrinsich.data()};

  LocalBearingVectorAidp<GEOMETRY_TYPE>
      rsre(*this);

  bool diffState =
      ::ceres::internal::AutoDifferentiate<
          ::ceres::internal::StaticParameterDims<7, 4, 7, 7, 7, 1, 1, 3, 6, 9, 9, 6,
                                                 6, 6, 6, 6>>(rsre, expandedParams, numOutputs, php_C, dpC_deltaAll);

  if (!diffState)
    std::cerr << "Potentially wrong Jacobians in autodiff " << std::endl;

  Eigen::Map<const Eigen::Vector4d> hp_C(&php_C[0]);
  // calculate the reprojection error
  Eigen::Map<const Eigen::Matrix<double, GEOMETRY_TYPE::NumIntrinsics, 1>>
      Intrinsics(parameters[Index::Intrinsics]);

  measurement_t kp;
  Eigen::Matrix<double, 2, 4> Jh;
  Eigen::Matrix<double, 2, 4> Jh_weighted;
  Eigen::Matrix<double, 2, Eigen::Dynamic> Jpi;
  Eigen::Matrix<double, 2, Eigen::Dynamic> Jpi_weighted;
  if (jacobians != NULL)
  {
    targetCameraGeometry_->projectHomogeneousWithExternalParameters(hp_C, Intrinsics, &kp, &Jh, &Jpi);
    Jh_weighted = squareRootInformation_ * Jh;
    Jpi_weighted = squareRootInformation_ * Jpi;
  }
  else
  {
    targetCameraGeometry_->projectHomogeneousWithExternalParameters(hp_C, Intrinsics, &kp);
  }

  measurement_t error = kp - measurement_;

  // weight:
  measurement_t weighted_error = squareRootInformation_ * error;

  // assign:
  residuals[0] = weighted_error[0];
  residuals[1] = weighted_error[1];

  // check validity:
  bool valid = true;
  if (fabs(hp_C[3]) > 1.0e-8)
  {
    Eigen::Vector3d p_C = hp_C.template head<3>() / hp_C[3];
    if (p_C[2] < 0.2)
    { // 20 cm - not very generic... but reasonable
      // std::cout<<"INVALID POINT"<<std::endl;
      valid = false;
    }
  }

  // calculate jacobians, if required
  if (jacobians != NULL)
  {
    if (!valid)
    {
      setJacobiansZero(jacobians, jacobiansMinimal);
      return true;
    }
    uint32_t height = targetCameraGeometry_->imageHeight();
    double ypixel(measurement_[1]);
    double kpN = ypixel / height - 0.5;

    assignJacobians(
        jacobians,
        jacobiansMinimal,
        Jh_weighted,
        Jpi_weighted,
        dhC_deltaTWSt, dhC_deltaTWSh,
        dhC_dlCh,
        dhC_dExtrinsict, dhC_dExtrinsich,
        dhC_td,
        kpN,
        dhC_speed, dhC_biases);
  }
  return true;
}

template <class GEOMETRY_TYPE>
void RsReprojectionErrorAidp<GEOMETRY_TYPE>::
    setJacobiansZero(double** jacobians, double** jacobiansMinimal) const {
  zeroJacobian<7, 6, 2>(Index::T_WBt, jacobians, jacobiansMinimal);
  zeroJacobian<4, 3, 2>(Index::AIDP, jacobians, jacobiansMinimal);

  zeroJacobian<7, 6, 2>(Index::T_WBh, jacobians, jacobiansMinimal);
  zeroJacobian<7, 6, 2>(Index::T_BCt, jacobians, jacobiansMinimal);
  zeroJacobian<7, 6, 2>(Index::T_BCh, jacobians, jacobiansMinimal);
  zeroJacobian<GEOMETRY_TYPE::NumIntrinsics, GEOMETRY_TYPE::NumIntrinsics, 2>(Index::Intrinsics, jacobians, jacobiansMinimal);
  zeroJacobian<1, 1, 2>(Index::ReadoutTime, jacobians, jacobiansMinimal);
  zeroJacobian<1, 1, 2>(Index::CameraTd, jacobians, jacobiansMinimal);
  zeroJacobian<3, 3, 2>(Index::Speed, jacobians, jacobiansMinimal);
  zeroJacobian<6, 6, 2>(Index::Biases, jacobians, jacobiansMinimal);

  zeroJacobian<9, 9, 2>(Index::M_gi, jacobians, jacobiansMinimal);
  zeroJacobian<9, 9, 2>(Index::M_si, jacobians, jacobiansMinimal);
  zeroJacobian<6, 6, 2>(Index::M_ai, jacobians, jacobiansMinimal);
}

template <class GEOMETRY_TYPE>
void RsReprojectionErrorAidp<GEOMETRY_TYPE>::
    assignJacobians(
        double **jacobians,
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
        const Eigen::Matrix<double, 4, 6> &dhC_biases) const {
  if (jacobians != NULL)
  {
    if (jacobians[0] != NULL)
    {
      Eigen::Matrix<double, 2, 6, Eigen::RowMajor> J0_minimal;
      J0_minimal = Jh_weighted * dhC_deltaTWSt;

      Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> J0(jacobians[0]);
      J0.leftCols<6>() = J0_minimal;
      J0.col(6).setZero();
      if (jacobiansMinimal != NULL)
      {
        if (jacobiansMinimal[0] != NULL)
        {
          Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>>
              J0_minimal_mapped(jacobiansMinimal[0]);
          J0_minimal_mapped = J0_minimal;
        }
      }
    }

    if (jacobians[1] != NULL)
    {
      Eigen::Map<Eigen::Matrix<double, 2, 4, Eigen::RowMajor>> J1(jacobians[1]);
      J1 = Jh_weighted * dhC_dlCh;
      if (jacobiansMinimal != NULL)
      {
        if (jacobiansMinimal[1] != NULL)
        {
          Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>>
              J1_minimal_mapped(jacobiansMinimal[1]);
          Eigen::Matrix<double, 4, 3, Eigen::RowMajor> S;
          swift_vio::InverseDepthParameterization::plusJacobian(nullptr, S.data());
          J1_minimal_mapped = J1 * S;
        }
      }
    }

    if (jacobians[2] != NULL)
    {
      Eigen::Matrix<double, 2, 6, Eigen::RowMajor> J2_minimal;
      J2_minimal = Jh_weighted * dhC_deltaTWSh;

      Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> J2(jacobians[2]);
      J2.leftCols<6>() = J2_minimal;
      J2.col(6).setZero();

      if (jacobiansMinimal != NULL)
      {
        if (jacobiansMinimal[2] != NULL)
        {
          Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>>
              J2_minimal_mapped(jacobiansMinimal[2]);
          J2_minimal_mapped = J2_minimal;
        }
      }
    }

    if (jacobians[3] != NULL)
    {
      Eigen::Matrix<double, 2, 6, Eigen::RowMajor>
          J3_minimal = Jh_weighted * dhC_dExtrinsict;
      Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> J3(jacobians[3]);
      J3.leftCols<6>() = J3_minimal;
      J3.col(6).setZero();
      if (jacobiansMinimal != NULL)
      {
        if (jacobiansMinimal[3] != NULL)
        {
          Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>>
              J3_minimal_mapped(jacobiansMinimal[3]);
          J3_minimal_mapped = J3_minimal;
        }
      }
    }

    if (jacobians[4] != NULL)
    {
      Eigen::Matrix<double, 2, 6, Eigen::RowMajor>
          J4_minimal = Jh_weighted * dhC_dExtrinsich;
      Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> J4(jacobians[4]);
      J4.leftCols<6>() = J4_minimal;
      J4.col(6).setZero();

      if (jacobiansMinimal != NULL)
      {
        if (jacobiansMinimal[4] != NULL)
        {
          Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>>
              J4_minimal_mapped(jacobiansMinimal[4]);
          J4_minimal_mapped = J4_minimal;
        }
      }
    }

    // camera intrinsics
    if (jacobians[5] != NULL)
    {
      Eigen::Map<Eigen::Matrix<double, 2, GEOMETRY_TYPE::NumIntrinsics, Eigen::RowMajor>> J5(jacobians[5]);
      J5 = Jpi_weighted;
      if (jacobiansMinimal != NULL)
      {
        if (jacobiansMinimal[5] != NULL)
        {
          Eigen::Map<Eigen::Matrix<double, 2, GEOMETRY_TYPE::NumIntrinsics, Eigen::RowMajor>> J5_minimal_mapped(jacobiansMinimal[5]);
          J5_minimal_mapped = J5;
        }
      }
    }

    //tr
    if (jacobians[6] != NULL)
    {
      Eigen::Map<Eigen::Matrix<double, 2, 1>> J6(jacobians[6]);
      J6 = Jh_weighted * dhC_td * kpN;
      if (jacobiansMinimal != NULL)
      {
        if (jacobiansMinimal[6] != NULL)
        {
          Eigen::Map<Eigen::Matrix<double, 2, 1>>
              J6_minimal_mapped(jacobiansMinimal[6]);
          J6_minimal_mapped = J6;
        }
      }
    }

    // t_d
    if (jacobians[7] != NULL)
    {
      Eigen::Map<Eigen::Matrix<double, 2, 1>> J7(jacobians[7]);
      J7 = Jh_weighted * dhC_td;
      if (jacobiansMinimal != NULL)
      {
        if (jacobiansMinimal[7] != NULL)
        {
          Eigen::Map<Eigen::Matrix<double, 2, 1>> J7_minimal_mapped(
              jacobiansMinimal[7]);
          J7_minimal_mapped = J7;
        }
      }
    }

    // speed
    if (jacobians[Index::Speed] != NULL)
    {
      Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> J(jacobians[Index::Speed]);
      J = Jh_weighted * dhC_speed;
      if (jacobiansMinimal != NULL)
      {
        if (jacobiansMinimal[Index::Speed] != NULL)
        {
          Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>>
              J_minimal_mapped(jacobiansMinimal[Index::Speed]);
          J_minimal_mapped = J;
        }
      }
    }

    if (jacobians[Index::Biases] != NULL)
    {
      Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> J(jacobians[Index::Biases]);
      J = Jh_weighted * dhC_biases;
      if (jacobiansMinimal != NULL)
      {
        if (jacobiansMinimal[Index::Biases] != NULL)
        {
          Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>>
              J_minimal_mapped(jacobiansMinimal[Index::Biases]);
          J_minimal_mapped = J;
        }
      }
    }

    // M_gi
    if (jacobians[Index::M_gi] != NULL)
    {
      Eigen::Map<Eigen::Matrix<double, 2, 9, Eigen::RowMajor>> J(jacobians[Index::M_gi]);
      J.setZero();
      if (jacobiansMinimal != NULL)
      {
        if (jacobiansMinimal[Index::M_gi] != NULL)
        {
          Eigen::Map<Eigen::Matrix<double, 2, 9, Eigen::RowMajor>>
              J_minimal_mapped(jacobiansMinimal[Index::M_gi]);
          J_minimal_mapped = J;
        }
      }
    }

    // M_si
    if (jacobians[Index::M_si] != NULL)
    {
      Eigen::Map<Eigen::Matrix<double, 2, 9, Eigen::RowMajor>> J(jacobians[Index::M_si]);
      J.setZero();
      if (jacobiansMinimal != NULL)
      {
        if (jacobiansMinimal[Index::M_si] != NULL)
        {
          Eigen::Map<Eigen::Matrix<double, 2, 9, Eigen::RowMajor>>
              J_minimal_mapped(jacobiansMinimal[Index::M_si]);
          J_minimal_mapped = J;
        }
      }
    }

    // M_ai
    if (jacobians[Index::M_ai] != NULL)
    {
      Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> J(jacobians[Index::M_ai]);
      J.setZero();
      if (jacobiansMinimal != NULL)
      {
        if (jacobiansMinimal[11] != NULL)
        {
          Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>>
              J_minimal_mapped(jacobiansMinimal[Index::M_ai]);
          J_minimal_mapped = J;
        }
      }
    }
  }
}

template <class GEOMETRY_TYPE>
LocalBearingVectorAidp<GEOMETRY_TYPE>::
    LocalBearingVectorAidp(
        const RsReprojectionErrorAidp<GEOMETRY_TYPE> &
            rsre)
    : rsre_(rsre) {}

template <class GEOMETRY_TYPE>
template <typename Scalar>
bool LocalBearingVectorAidp<GEOMETRY_TYPE>::
operator()(const Scalar *const T_WBt,
           const Scalar *const l_Ch,
           const Scalar *const T_WBh,
           const Scalar *const T_BCt,
           const Scalar *const T_BCh,
           const Scalar *const t_r,
           const Scalar *const t_d,
           const Scalar *const speed,
           const Scalar *const biases,
           const Scalar *const /*M_g*/, const Scalar *const /*M_s*/, const Scalar *const /*M_a*/,
           const Scalar *const deltaT_WSt, const Scalar *const deltaT_WSh,
           const Scalar *const deltaExtrinsict, const Scalar *const deltaExtrinsich,
           Scalar residuals[4]) const
{
  //T_WBt
  Eigen::Map<const Eigen::Matrix<Scalar, 3, 1>> p_WB_t_temp(T_WBt);
  const Eigen::Quaternion<Scalar> q_WB_t_temp(T_WBt[6], T_WBt[3], T_WBt[4], T_WBt[5]);
  Eigen::Map<const Eigen::Matrix<Scalar, 6, 1>> deltaT_WB_t(deltaT_WSt);
  Eigen::Matrix<Scalar, 3, 1> p_WB_t = p_WB_t_temp + deltaT_WB_t.template head<3>();

  Eigen::Matrix<Scalar, 3, 1> omega1 = deltaT_WB_t.template tail<3>();
  Eigen::Quaternion<Scalar> dqWSt = okvis::kinematics::expAndTheta(omega1);
  Eigen::Quaternion<Scalar> q_WB_t = dqWSt * q_WB_t_temp;

  //T_WBh
  Eigen::Map<const Eigen::Matrix<Scalar, 3, 1>> p_WB_h0(T_WBh);
  const Eigen::Quaternion<Scalar> q_WB_h0(T_WBh[6], T_WBh[3], T_WBh[4], T_WBh[5]);
  Eigen::Map<const Eigen::Matrix<Scalar, 6, 1>> deltaT_WB_h(deltaT_WSh);
  Eigen::Matrix<Scalar, 3, 1> p_WB_h = p_WB_h0 + deltaT_WB_h.template head<3>();

  Eigen::Matrix<Scalar, 3, 1> omega2 = deltaT_WB_h.template tail<3>();
  Eigen::Quaternion<Scalar> dqWSh = okvis::kinematics::expAndTheta(omega2);
  Eigen::Quaternion<Scalar> q_WB_h = dqWSh * q_WB_h0;

  Eigen::Matrix<Scalar, 4, 4> T_WB_h = Eigen::Matrix<Scalar, 4, 4>::Identity();
  T_WB_h.template topLeftCorner<3, 3>() = q_WB_h.toRotationMatrix();
  T_WB_h.template topRightCorner<3, 1>() = p_WB_h;

  Eigen::Map<const Eigen::Matrix<Scalar, 4, 1>> l_Ch_t(l_Ch);

  //T_BCt
  Eigen::Matrix<Scalar, 3, 1> p_BC_t0(T_BCt[0], T_BCt[1], T_BCt[2]);
  Eigen::Quaternion<Scalar> q_BC_t0(T_BCt[6], T_BCt[3], T_BCt[4], T_BCt[5]);
  Eigen::Map<const Eigen::Matrix<Scalar, 6, 1>> deltaT_BC_t(deltaExtrinsict);

  Eigen::Matrix<Scalar, 3, 1> p_BC_t = p_BC_t0 + deltaT_BC_t.template head<3>();
  Eigen::Matrix<Scalar, 3, 1> omega3 = deltaT_BC_t.template tail<3>();
  Eigen::Quaternion<Scalar> dqBCt = okvis::kinematics::expAndTheta(omega3);
  Eigen::Quaternion<Scalar> q_BC_t = dqBCt * q_BC_t0;

  //T_BCh
  Eigen::Matrix<Scalar, 3, 1> p_BC_h0(T_BCh[0], T_BCh[1], T_BCh[2]);
  Eigen::Quaternion<Scalar> q_BC_h0(T_BCh[6], T_BCh[3], T_BCh[4], T_BCh[5]);
  Eigen::Map<const Eigen::Matrix<Scalar, 6, 1>> deltaT_BC_h(deltaExtrinsich);
  Eigen::Matrix<Scalar, 3, 1> p_BC_h = p_BC_h0 + deltaT_BC_h.template head<3>();

  Eigen::Matrix<Scalar, 3, 1> omega4 = deltaT_BC_h.template tail<3>();
  Eigen::Quaternion<Scalar> dqBCh = okvis::kinematics::expAndTheta(omega4);
  Eigen::Quaternion<Scalar> q_BC_h = dqBCh * q_BC_h0;

  Eigen::Matrix<Scalar, 4, 4> T_BC_h = Eigen::Matrix<Scalar, 4, 4>::Identity();
  T_BC_h.template topLeftCorner<3, 3>() = q_BC_h.toRotationMatrix();
  T_BC_h.template topRightCorner<3, 1>() = p_BC_h;

  Scalar trLatestEstimate = t_r[0];

  uint32_t height = rsre_.targetCameraGeometry_->imageHeight();
  double ypixel(rsre_.measurement_[1]);
  Scalar kpN = (Scalar)(ypixel / height - 0.5);
  Scalar tdLatestEstimate = t_d[0];
  Scalar relativeFeatureTime =
      tdLatestEstimate + trLatestEstimate * kpN - (Scalar)(rsre_.targetStateTime_.toSec() - rsre_.targetImageTime_.toSec());

  std::pair<Eigen::Matrix<Scalar, 3, 1>, Eigen::Quaternion<Scalar>> pairT_WB_t(
      p_WB_t, q_WB_t);

  Eigen::Matrix<Scalar, 3, 1> vW =
      Eigen::Map<const Eigen::Matrix<Scalar, 3, 1>>(speed);

  Eigen::Matrix<Scalar, 6, 1> bgba =
      Eigen::Map<const Eigen::Matrix<Scalar, 6, 1>>(biases);

  Scalar t_start = (Scalar)rsre_.targetStateTime_.toSec();
  Scalar t_end = t_start + relativeFeatureTime;
  swift_vio::GenericImuMeasurementDeque<Scalar> imuMeasurements;
  for (size_t jack = 0; jack < rsre_.imuMeasCanopy_->size(); ++jack)
  {
    swift_vio::GenericImuMeasurement<Scalar> imuMeas(
        (Scalar)(rsre_.imuMeasCanopy_->at(jack).timeStamp.toSec()),
        rsre_.imuMeasCanopy_->at(jack).measurement.gyroscopes.template cast<Scalar>(),
        rsre_.imuMeasCanopy_->at(jack).measurement.accelerometers.template cast<Scalar>());
    imuMeasurements.push_back(imuMeas);
  }

  Eigen::Matrix<Scalar, 3, 1> gW = rsre_.imuParameters_->gravity().template cast<Scalar>();
  if (relativeFeatureTime >= Scalar(5e-8))
  {
    swift_vio::ode::predictStates(imuMeasurements, gW, pairT_WB_t,
                                  vW, bgba, t_start, t_end);
  } else if (relativeFeatureTime <= Scalar(-5e-8)) {
    swift_vio::ode::predictStatesBackward(imuMeasurements, gW,
                                          pairT_WB_t, vW, bgba, t_start, t_end);
  }

  p_WB_t = pairT_WB_t.first;
  q_WB_t = pairT_WB_t.second;

  // transform the point into the camera:
  Eigen::Matrix<Scalar, 3, 3> C_BC_t = q_BC_t.toRotationMatrix();
  Eigen::Matrix<Scalar, 3, 3> C_CB_t = C_BC_t.transpose();
  Eigen::Matrix<Scalar, 4, 4> T_CB_t = Eigen::Matrix<Scalar, 4, 4>::Identity();
  T_CB_t.template topLeftCorner<3, 3>() = C_CB_t;
  T_CB_t.template topRightCorner<3, 1>() = -C_CB_t * p_BC_t;
  Eigen::Matrix<Scalar, 3, 3> C_WB_t = q_WB_t.toRotationMatrix();
  Eigen::Matrix<Scalar, 3, 3> C_BW_t = C_WB_t.transpose();
  Eigen::Matrix<Scalar, 4, 4> T_BW_t = Eigen::Matrix<Scalar, 4, 4>::Identity();
  T_BW_t.template topLeftCorner<3, 3>() = C_BW_t;
  T_BW_t.template topRightCorner<3, 1>() = -C_BW_t * p_WB_t;
  Eigen::Matrix<Scalar, 4, 1> hp_W_t = (T_WB_h * T_BC_h) * l_Ch_t;
  Eigen::Matrix<Scalar, 4, 1> hp_B_t = T_BW_t * hp_W_t;
  Eigen::Matrix<Scalar, 4, 1> hp_C_t = T_CB_t * hp_B_t;

  residuals[0] = hp_C_t[0];
  residuals[1] = hp_C_t[1];
  residuals[2] = hp_C_t[2];
  residuals[3] = hp_C_t[3];

  return true;
}
}  // namespace ceres
}  // namespace okvis
