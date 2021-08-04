
/**
 * @file implementation/RSCameraReprojectionError.hpp
 * @brief Header implementation file for the RSCameraReprojectionError class.
 * @author Jianzhu Huai
 */
#include "ceres/internal/autodiff.h"

#include <okvis/kinematics/Transformation.hpp>
#include <okvis/kinematics/operators.hpp>

#include <swift_vio/ceres/JacobianHelpers.hpp>
#include <swift_vio/Measurements.hpp>
#include <swift_vio/imu/SimpleImuOdometry.hpp>

namespace okvis {
namespace ceres {
template <class GEOMETRY_TYPE>
RSCameraReprojectionError<GEOMETRY_TYPE>::RSCameraReprojectionError() {}

template <class GEOMETRY_TYPE>
RSCameraReprojectionError<GEOMETRY_TYPE>::RSCameraReprojectionError(
    const measurement_t& measurement,
    const covariance_t& covariance,
    std::shared_ptr<const camera_geometry_t> targetCamera,
    std::shared_ptr<const okvis::ImuMeasurementDeque> imuMeasCanopy,
    okvis::ImuParameters imuParameters,
    okvis::Time targetStateTime, okvis::Time targetImageTime)
    : imuMeasCanopy_(imuMeasCanopy),
      imuParameters_(imuParameters),
      targetCamera_(targetCamera),
      targetStateTime_(targetStateTime),
      targetImageTime_(targetImageTime) {
  measurement_ = measurement;
  setCovariance(covariance);
  cameraGeometryBase_ = targetCamera;
}

template <class GEOMETRY_TYPE>
void RSCameraReprojectionError<GEOMETRY_TYPE>::
    setCovariance(const covariance_t& covariance) {
  information_ = covariance.inverse();
  covariance_ = covariance;
  // perform the Cholesky decomposition on order to obtain the correct error
  // weighting
  Eigen::LLT<Eigen::Matrix2d> lltOfInformation(information_);
  squareRootInformation_ = lltOfInformation.matrixL().transpose();
}

template <class GEOMETRY_TYPE>
bool RSCameraReprojectionError<GEOMETRY_TYPE>::
    Evaluate(double const* const* parameters, double* residuals,
             double** jacobians) const {
  return EvaluateWithMinimalJacobians(parameters, residuals, jacobians, NULL);
}

template <class GEOMETRY_TYPE>
bool RSCameraReprojectionError<GEOMETRY_TYPE>::
    EvaluateWithMinimalJacobians(double const* const* parameters,
                                 double* residuals, double** jacobians,
                                 double** jacobiansMinimal) const {
  return EvaluateWithMinimalJacobiansAnalytic(parameters, residuals, jacobians,
                                              jacobiansMinimal);
}

template <class GEOMETRY_TYPE>
bool RSCameraReprojectionError<GEOMETRY_TYPE>::
    EvaluateWithMinimalJacobiansAnalytic(double const* const* parameters,
                                 double* residuals, double** jacobians,
                                 double** jacobiansMinimal) const {
  Eigen::Map<const Eigen::Vector3d> p_WBt0(parameters[Index::T_WBt]);
  Eigen::Map<const Eigen::Quaterniond> q_WBt0(parameters[Index::T_WBt] + 3);

  Eigen::Map<const Eigen::Vector4d> scaled_p_Ch(parameters[Index::hp_Ch]);
  Eigen::Vector4d hp_Ch = scaled_p_Ch / scaled_p_Ch[3]; // homogeneous representation.

  Eigen::Map<const Eigen::Vector3d> p_BCt(parameters[Index::T_BCt]);
  Eigen::Map<const Eigen::Quaterniond> q_BCt(parameters[Index::T_BCt] + 3);
  okvis::kinematics::Transformation T_BCt(p_BCt, q_BCt);

  Eigen::Map<const Eigen::Vector3d> p_BCh(parameters[Index::T_BCh]);
  Eigen::Map<const Eigen::Quaterniond> q_BCh(parameters[Index::T_BCh] + 3);
  okvis::kinematics::Transformation T_BCh(p_BCh, q_BCh);

  Eigen::Map<const Eigen::Vector3d> p_WBh(parameters[Index::T_WBh]);
  Eigen::Map<const Eigen::Quaterniond> q_WBh(parameters[Index::T_WBh] + 3);
  okvis::kinematics::Transformation T_WBh(p_WBh, q_WBh);

  Eigen::Matrix<double, -1, 1> intrinsics = Eigen::Map<const Eigen::Matrix<double, kIntrinsicsDim, 1>>(parameters[Index::Intrinsics]); //proj intrinsics+DistortionIntrinsics

  double readoutTime = parameters[Index::ReadoutTime][0]; //tr
  double cameraTd = parameters[Index::CameraTd][0];       //td

  Eigen::Matrix<double, 9, 1> speedBgBa = Eigen::Map<const Eigen::Matrix<double, 9, 1>>(parameters[Index::SpeedAndBiases]);

  Eigen::Map<const Eigen::Matrix<double, 9, 1>> Tg(parameters[Index::T_gi]); // not used for now.
  Eigen::Map<const Eigen::Matrix<double, 9, 1>> Ts(parameters[Index::T_si]); // not used for now.
  Eigen::Map<const Eigen::Matrix<double, 6, 1>> Ta(parameters[Index::T_ai]); // not used for now.

  double ypixel(measurement_[1]);
  uint32_t height = cameraGeometryBase_->imageHeight();
  double kpN = ypixel / height - 0.5;
  double relativeFeatureTime = cameraTd + readoutTime * kpN + ( targetImageTime_- targetStateTime_).toSec(); 
  std::pair<Eigen::Matrix<double, 3, 1>, Eigen::Quaternion<double>> pair_T_WBt(p_WBt0, q_WBt0);

  const okvis::Time t_start = targetStateTime_;
  const okvis::Time t_end = targetStateTime_ + okvis::Duration(relativeFeatureTime);
  const double wedge = 5e-8;
  if (relativeFeatureTime >= wedge){
    swift_vio::ode::predictStates(*imuMeasCanopy_, imuParameters_.g, pair_T_WBt,
                                  speedBgBa, t_start, t_end);
  }else if (relativeFeatureTime <= -wedge){
    swift_vio::ode::predictStatesBackward(*imuMeasCanopy_, imuParameters_.g, pair_T_WBt,
                                          speedBgBa, t_start, t_end);
  }
  okvis::kinematics::Transformation T_WBt(pair_T_WBt.first, pair_T_WBt.second);
  Eigen::Quaterniond q_WBt = pair_T_WBt.second;
  Eigen::Vector3d p_WBt = pair_T_WBt.first;

  //Eigen::Vector4d hp_Ct = (T_WBt * T_BCt).inverse() * (T_WBh * T_BCh) * hp_Ch; //(T_WBh * T_BCh) * hp_Ch=hp_W
  // transform the point into the camera:
  Eigen::Matrix3d C_BCt = q_BCt.toRotationMatrix();
  Eigen::Matrix3d C_CBt = C_BCt.transpose();
  Eigen::Matrix4d T_CBt = Eigen::Matrix4d::Identity();
  T_CBt.topLeftCorner<3, 3>() = C_CBt;
  T_CBt.topRightCorner<3, 1>() = -C_CBt * p_BCt; //_xiugai
  Eigen::Matrix3d C_WBt = q_WBt.toRotationMatrix();
  Eigen::Matrix3d C_BWt = C_WBt.transpose();
  Eigen::Matrix4d T_BWt = Eigen::Matrix4d::Identity();
  T_BWt.topLeftCorner<3, 3>() = C_BWt;
  T_BWt.topRightCorner<3, 1>() = -C_BWt * p_WBt;
  Eigen::Vector4d hp_Wt = (T_WBh * T_BCh).T() * hp_Ch;
  Eigen::Vector4d hp_Bt = T_BWt * hp_Wt;
  Eigen::Vector4d hp_Ct = T_CBt * hp_Bt;
  Eigen::Vector4d hp_Bh = (T_BCh).T() * hp_Ch;

  // calculate the reprojection error
  measurement_t kp;
  Eigen::Matrix<double, 2, 3> Jh;
  Eigen::Matrix<double, 2, 3> Jh_weighted;
  Eigen::Matrix<double, 2, Eigen::Dynamic> Jpi;
  Eigen::Matrix<double, 2, Eigen::Dynamic> Jpi_weighted;
  if (jacobians != NULL){
    cameraGeometryBase_->projectWithExternalParameters(hp_Ct.head<3>(), intrinsics, &kp, &Jh, &Jpi);
    Jh_weighted = squareRootInformation_ * Jh;
    Jpi_weighted = squareRootInformation_ * Jpi;
  }
  else{
    cameraGeometryBase_->projectWithExternalParameters(hp_Ct.head<3>(), intrinsics, &kp, &Jh, &Jpi);
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

    // TODO(jhuai): binliang: correct the below to compute the Jacobians analytically.
    Eigen::Matrix<double, 3, 6> dhC_deltaTWSt;   //T_WB
    Eigen::Matrix<double, 3, 6> dhC_dExtrinsict; //T_BC
    Eigen::Matrix<double, 3, 4> dhC_deltahpCh;
    Eigen::Matrix<double, 3, 6> dhC_deltaTWSh;
    Eigen::Matrix<double, 3, 6> dhC_dExtrinsich;
    Eigen::Vector3d dhC_td;
    Eigen::Matrix<double, 3, 9> dhC_sb;

    //t
    Eigen::Vector3d p_BP_Wt = hp_Wt.head<3>() - p_WBt * hp_Wt[3];
    Eigen::Matrix<double, 4, 6> dhS_deltaTWSt;
    dhS_deltaTWSt.topLeftCorner<3, 3>() = -C_BWt * hp_Wt[3];                            //p
    dhS_deltaTWSt.topRightCorner<3, 3>() = C_BWt * okvis::kinematics::crossMx(p_BP_Wt); //q
    dhS_deltaTWSt.row(3).setZero();

    dhC_deltaTWSt = (T_CBt * dhS_deltaTWSt).topRows<3>();

    Eigen::Matrix<double, 4, 6> dhC_dExtrinsict_temp;
    dhC_dExtrinsict_temp.block<3, 3>(0, 0) = -C_CBt * hp_Ct[3];
    //dhC_dExtrinsict_temp.block<3, 3>(0, 3) = -okvis::kinematics::crossMx(hp_Ct.head<3>()) * C_CBt;
    dhC_dExtrinsict_temp.block<3, 3>(0, 3) = okvis::kinematics::crossMx(C_CBt * (T_WBt.inverse() * (T_WBh * T_BCh) * hp_Ch).head<3>());
    dhC_dExtrinsict_temp.row(3).setZero();
    dhC_dExtrinsict = dhC_dExtrinsict_temp.topRows<3>();

    //h
    //hp_Ct = (T_WBt * T_BCt).inverse() * (T_WBh * T_BCh) * hp_Ch;
    Eigen::Vector3d p_BP_Wh = (T_BCh.T() * hp_Ch).head<3>();
    Eigen::Matrix3d C_WBh = q_WBh.toRotationMatrix();
    Eigen::Matrix<double, 4, 6> dhW_deltaTWSh;
    dhW_deltaTWSh.topLeftCorner<3, 3>() = Eigen::Matrix3d::Identity() * (T_BCh.T() * hp_Ch)[3]; //p
    //dhW_deltaTWSh.topRightCorner<3, 3>() = -C_WBh * okvis::kinematics::crossMx(p_BP_Wh); //q
    dhW_deltaTWSh.topRightCorner<3, 3>() = -1 * okvis::kinematics::crossMx(C_WBh * p_BP_Wh); //q
    dhW_deltaTWSh.row(3).setZero();

    dhC_deltaTWSh = ((T_WBt * T_BCt).inverse().T() * dhW_deltaTWSh).topRows<3>();
    dhC_deltahpCh = ((T_WBt * T_BCt).inverse() * (T_WBh * T_BCh)).T().topRows<3>(); //_xiugai(budui)

    Eigen::Vector3d p_BP_Ch = hp_Ch.head<3>();
    Eigen::Matrix3d C_BCh = q_BCh.toRotationMatrix();
    Eigen::Matrix<double, 4, 6> dhW_dExtrinsich;
    dhW_dExtrinsich.topLeftCorner<3, 3>() = Eigen::Matrix3d::Identity() * hp_Ch[3];        //p
    dhW_dExtrinsich.topRightCorner<3, 3>() = -C_BCh * okvis::kinematics::crossMx(p_BP_Ch); //q
    //dhW_dExtrinsich.topRightCorner<3, 3>() = -1* okvis::kinematics::crossMx(C_BCh * p_BP_Ch); //q
    dhW_dExtrinsich.row(3).setZero();
    dhW_dExtrinsich = T_WBh.T() * dhW_dExtrinsich;

    dhC_dExtrinsich = ((T_WBt * T_BCt).inverse().T() * dhW_dExtrinsich).topRows<3>();

    //other
    okvis::ImuMeasurement queryValue;
    swift_vio::ode::interpolateInertialData(*imuMeasCanopy_, t_end, queryValue);
    queryValue.measurement.gyroscopes -= speedBgBa.segment<3>(3);
    Eigen::Vector3d p =
        okvis::kinematics::crossMx(queryValue.measurement.gyroscopes) *
            hp_Bt.head<3>() +
        C_BWt * speedBgBa.head<3>() * hp_Wt[3];
    dhC_td = -C_CBt * p;

    Eigen::Matrix3d dhC_vW = -C_CBt * C_BWt * relativeFeatureTime * hp_Wt[3];
    Eigen::Matrix3d dhC_bg =
        -C_CBt * C_BWt *
        okvis::kinematics::crossMx(hp_Wt.head<3>() - hp_Wt[3] * p_WBt) *
        relativeFeatureTime * q_WBt0.toRotationMatrix();

    //dhC_sb.row(3).setZero();
    dhC_sb.topRightCorner<3, 3>().setZero();
    dhC_sb.topLeftCorner<3, 3>() = dhC_vW;
    dhC_sb.block<3, 3>(0, 3) = dhC_bg;

    //assignJacobians
    if (jacobians[0] != NULL)
    {
      Eigen::Matrix<double, 2, 6, Eigen::RowMajor> J0_minimal;
      J0_minimal = Jh_weighted * dhC_deltaTWSt;
      // pseudo inverse of the local parametrization Jacobian
      Eigen::Matrix<double, 6, 7, Eigen::RowMajor> J_lift;
      PoseLocalParameterization::liftJacobian(parameters[Index::T_WBt], J_lift.data());

      // hallucinate Jacobian w.r.t. state
      Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> J0(jacobians[0]);
      J0 = J0_minimal * J_lift;
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
      J1 = Jh_weighted * dhC_deltahpCh;
      if (jacobiansMinimal != NULL)
      {
        if (jacobiansMinimal[1] != NULL)
        {
          Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>>
              J1_minimal_mapped(jacobiansMinimal[1]);
          Eigen::Matrix<double, 4, 3> S;
          S.setZero();
          S.topLeftCorner<3, 3>().setIdentity();
          J1_minimal_mapped = J1 * S;
        }
      }
    }

    if (jacobians[2] != NULL)
    {
      Eigen::Matrix<double, 2, 6, Eigen::RowMajor> J2_minimal;
      J2_minimal = Jh_weighted * dhC_deltaTWSh;
      // hallucinate Jacobian w.r.t. state
      Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> J2(jacobians[2]);
      J2.setZero();
      J2.topLeftCorner<2, 6>() = J2_minimal;
      //J0 = J0_minimal * J_lift;
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
      // compute the minimal version
      Eigen::Matrix<double, 2, 6, Eigen::RowMajor>
          J3_minimal = Jh_weighted * dhC_dExtrinsict;
      Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> J3(jacobians[3]);
      Eigen::Matrix<double, 6, 7, Eigen::RowMajor> J_lift;
      PoseLocalParameterization::liftJacobian(parameters[Index::T_BCt], J_lift.data());
      J3 = J3_minimal * J_lift;
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
      // compute the minimal version
      Eigen::Matrix<double, 2, 6, Eigen::RowMajor>
          J4_minimal = Jh_weighted * dhC_dExtrinsich;
      Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> J4(jacobians[4]);
      J4.setZero();
      J4.topLeftCorner<2, 6>() = J4_minimal; // Warn: This assumes that the custom PoseLocalParameterization assigns identity to the PlusJacobian.
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
      Eigen::Matrix<double, 2, Eigen::Dynamic> Jpi_weighted_copy = Jpi_weighted;
      J5 = Jpi_weighted_copy
               .template topLeftCorner<2, GEOMETRY_TYPE::NumIntrinsics>();
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
    if (jacobians[8] != NULL)
    {
      Eigen::Map<Eigen::Matrix<double, 2, 9, Eigen::RowMajor>> J8(jacobians[8]);
      J8 = Jh_weighted * dhC_sb;
      if (jacobiansMinimal != NULL)
      {
        if (jacobiansMinimal[8] != NULL)
        {
          Eigen::Map<Eigen::Matrix<double, 2, 9, Eigen::RowMajor>>
              J8_minimal_mapped(jacobiansMinimal[8]);
          J8_minimal_mapped = J8;
        }
      }
    }

    // T_gi
    if (jacobians[9] != NULL)
    {
      Eigen::Map<Eigen::Matrix<double, 2, 9, Eigen::RowMajor>> J9(jacobians[9]);
      J9.setZero();
      if (jacobiansMinimal != NULL)
      {
        if (jacobiansMinimal[9] != NULL)
        {
          Eigen::Map<Eigen::Matrix<double, 2, 9, Eigen::RowMajor>>
              J9_minimal_mapped(jacobiansMinimal[9]);
          J9_minimal_mapped = J9;
        }
      }
    }

    // T_si
    if (jacobians[10] != NULL)
    {
      Eigen::Map<Eigen::Matrix<double, 2, 9, Eigen::RowMajor>> J10(jacobians[10]);
      J10.setZero();
      if (jacobiansMinimal != NULL)
      {
        if (jacobiansMinimal[10] != NULL)
        {
          Eigen::Map<Eigen::Matrix<double, 2, 9, Eigen::RowMajor>>
              J10_minimal_mapped(jacobiansMinimal[10]);
          J10_minimal_mapped = J10;
        }
      }
    }

    // T_ai
    if (jacobians[11] != NULL)
    {
      Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> J11(jacobians[11]);
      J11.setZero();
      if (jacobiansMinimal != NULL)
      {
        if (jacobiansMinimal[11] != NULL)
        {
          Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>>
              J11_minimal_mapped(jacobiansMinimal[11]);
          J11_minimal_mapped = J11;
        }
      }
    }
  }
  return true;
}

// This evaluates the error term and additionally computes
// the Jacobians in the minimal internal representation via autodiff
template <class GEOMETRY_TYPE>
bool RSCameraReprojectionError<GEOMETRY_TYPE>::
    EvaluateWithMinimalJacobiansAutoDiff(double const* const* parameters,
                                         double* residuals, double** jacobians,
                                         double** jacobiansMinimal) const
{
  // TODO(jhuai): binliang: either implement auto diff here or use autodiff in the test.
  const int numOutputs = 4;
  double deltaTWSt[6] = {0};
  double deltaTWSh[6] = {0};
  double deltaTSCt[6] = {0};
  double deltaTSCh[6] = {0};
  double const *const expandedParams[] = {
      parameters[Index::T_WBt],
      parameters[Index::hp_Ch],
      parameters[Index::T_WBh],
      parameters[Index::T_BCt],
      parameters[Index::T_BCh],
      parameters[Index::ReadoutTime],
      parameters[Index::CameraTd],
      parameters[Index::SpeedAndBiases],
      parameters[Index::T_gi], parameters[Index::T_si], parameters[Index::T_ai],
      deltaTWSt, deltaTWSh,
      deltaTSCt, deltaTSCh};

  double php_C[numOutputs];
  Eigen::Matrix<double, numOutputs, 7, Eigen::RowMajor> dhC_deltaTWSt_full;
  Eigen::Matrix<double, numOutputs, 7, Eigen::RowMajor> dhC_deltaTWSh_full;
  Eigen::Matrix<double, numOutputs, 4, Eigen::RowMajor> dhC_deltahpCh;
  Eigen::Matrix<double, numOutputs, 7, Eigen::RowMajor> dhC_dExtrinsict_full;
  Eigen::Matrix<double, numOutputs, 7, Eigen::RowMajor> dhC_dExtrinsich_full;

  //ProjectionIntrinsicJacType4 dhC_projIntrinsic;
  //Eigen::Matrix<double, numOutputs, kDistortionDim, Eigen::RowMajor> dhC_distortion;
  Eigen::Matrix<double, numOutputs, GEOMETRY_TYPE::NumIntrinsics, Eigen::RowMajor> dhC_Intrinsic;
  Eigen::Matrix<double, numOutputs, 1> dhC_tr;
  Eigen::Matrix<double, numOutputs, 1> dhC_td;
  Eigen::Matrix<double, numOutputs, 9, Eigen::RowMajor> dhC_sb;
  Eigen::Matrix<double, numOutputs, 6, Eigen::RowMajor> dhC_deltaTWSt;
  Eigen::Matrix<double, numOutputs, 6, Eigen::RowMajor> dhC_deltaTWSh;
  Eigen::Matrix<double, numOutputs, 6, Eigen::RowMajor> dhC_dExtrinsict;
  Eigen::Matrix<double, numOutputs, 6, Eigen::RowMajor> dhC_dExtrinsich;

  Eigen::Matrix<double, numOutputs, 9, Eigen::RowMajor> dhC_tgi;
  Eigen::Matrix<double, numOutputs, 9, Eigen::RowMajor> dhC_tsi;
  Eigen::Matrix<double, numOutputs, 6, Eigen::RowMajor> dhC_tai;

  //dhC_projIntrinsic.setZero();
  //dhC_distortion.setZero();
  dhC_Intrinsic.setZero();
  double *dpC_deltaAll[] = {
      dhC_deltaTWSt_full.data(),
      dhC_deltahpCh.data(),
      dhC_deltaTWSh_full.data(),
      dhC_dExtrinsict_full.data(),
      dhC_dExtrinsich_full.data(),
      dhC_tr.data(),
      dhC_td.data(),
      dhC_sb.data(),
      dhC_tgi.data(), dhC_tsi.data(), dhC_tai.data(),
      dhC_deltaTWSt.data(), dhC_deltaTWSh.data(),
      dhC_dExtrinsict.data(), dhC_dExtrinsich.data()};

  RS_LocalBearingVector<GEOMETRY_TYPE>
      rsre(*this);

  bool diffState =
      ::ceres::internal::AutoDifferentiate<
          ::ceres::internal::StaticParameterDims<7, 4, 7, 7, 7, 1, 1, 9, 9, 9, 6,
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
    cameraGeometryBase_->projectHomogeneousWithExternalParameters(hp_C, Intrinsics, &kp, &Jh, &Jpi);
    Jh_weighted = squareRootInformation_ * Jh;
    Jpi_weighted = squareRootInformation_ * Jpi;
  }
  else
  {
    cameraGeometryBase_->projectHomogeneousWithExternalParameters(hp_C, Intrinsics, &kp);
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
  // This is pretty close to Paul Furgale's thesis. eq. 3.100 on page 40
  if (jacobians != NULL)
  {
    if (!valid)
    {
      setJacobiansZero(jacobians, jacobiansMinimal);
      return true;
    }
    uint32_t height = cameraGeometryBase_->imageHeight();
    double ypixel(measurement_[1]);
    double kpN = ypixel / height - 0.5;

    assignJacobians(
        parameters, jacobians,
        jacobiansMinimal,
        Jh_weighted,
        Jpi_weighted,
        dhC_deltaTWSt, dhC_deltaTWSh,
        dhC_deltahpCh,
        dhC_dExtrinsict, dhC_dExtrinsich,
        dhC_td,
        kpN,
        dhC_sb);
  }
  return true;
}

template <class GEOMETRY_TYPE>
void RSCameraReprojectionError<GEOMETRY_TYPE>::
    setJacobiansZero(double** jacobians, double** jacobiansMinimal) const {
  zeroJacobian<7, 6, 2>(Index::T_WBt, jacobians, jacobiansMinimal);
  zeroJacobian<4, 3, 2>(Index::hp_Ch, jacobians, jacobiansMinimal);

  // TODO(jhuai): binliang: correct the below for other parameters.
  zeroJacobian<7, 6, 2>(Index::T_WBh, jacobians, jacobiansMinimal);
  zeroJacobian<7, 6, 2>(Index::T_BCt, jacobians, jacobiansMinimal);
  zeroJacobian<7, 6, 2>(Index::T_BCh, jacobians, jacobiansMinimal);
  zeroJacobian<GEOMETRY_TYPE::NumIntrinsics, GEOMETRY_TYPE::NumIntrinsics, 2>(Index::Intrinsics, jacobians, jacobiansMinimal);
  zeroJacobian<1, 1, 2>(Index::ReadoutTime, jacobians, jacobiansMinimal);
  zeroJacobian<1, 1, 2>(Index::CameraTd, jacobians, jacobiansMinimal);
  zeroJacobian<9, 9, 2>(Index::SpeedAndBiases, jacobians, jacobiansMinimal);

  zeroJacobian<9, 9, 2>(Index::T_gi, jacobians, jacobiansMinimal);
  zeroJacobian<9, 9, 2>(Index::T_si, jacobians, jacobiansMinimal);
  zeroJacobian<6, 6, 2>(Index::T_ai, jacobians, jacobiansMinimal);
}

template <class GEOMETRY_TYPE>
void RSCameraReprojectionError<GEOMETRY_TYPE>::
    assignJacobians(
        double const *const *parameters, double **jacobians,
        double **jacobiansMinimal,
        const Eigen::Matrix<double, 2, 4> &Jh_weighted,
        const Eigen::Matrix<double, 2, Eigen::Dynamic> &Jpi_weighted,
        const Eigen::Matrix<double, 4, 6> &dhC_deltaTWSt,
        const Eigen::Matrix<double, 4, 6> &dhC_deltaTWSh,
        const Eigen::Matrix<double, 4, 4> &dhC_deltahpCh,
        const Eigen::Matrix<double, 4, 6> &dhC_dExtrinsict,
        const Eigen::Matrix<double, 4, 6> &dhC_dExtrinsich,
        const Eigen::Vector4d &dhC_td, double kpN,
        const Eigen::Matrix<double, 4, 9> &dhC_sb) const
{
  if (jacobians != NULL)
  {
    if (jacobians[0] != NULL)
    {
      Eigen::Matrix<double, 2, 6, Eigen::RowMajor> J0_minimal;
      J0_minimal = Jh_weighted * dhC_deltaTWSt;
      // pseudo inverse of the local parametrization Jacobian
      Eigen::Matrix<double, 6, 7, Eigen::RowMajor> J_lift;
      PoseLocalParameterization::liftJacobian(parameters[Index::T_WBt], J_lift.data());

      // hallucinate Jacobian w.r.t. state
      Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> J0(jacobians[0]);
      J0 = J0_minimal * J_lift;
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
      J1 = Jh_weighted * dhC_deltahpCh;
      if (jacobiansMinimal != NULL)
      {
        if (jacobiansMinimal[1] != NULL)
        {
          Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>>
              J1_minimal_mapped(jacobiansMinimal[1]);
          Eigen::Matrix<double, 4, 3> S;
          S.setZero();
          S.topLeftCorner<3, 3>().setIdentity();
          J1_minimal_mapped = J1 * S;
        }
      }
    }

    if (jacobians[2] != NULL)
    {
      Eigen::Matrix<double, 2, 6, Eigen::RowMajor> J2_minimal;
      J2_minimal = Jh_weighted * dhC_deltaTWSh;
      // pseudo inverse of the local parametrization Jacobian
      //Eigen::Matrix<double, 6, 7, Eigen::RowMajor> J_lift;
      //PoseLocalParameterization::liftJacobian(parameters[Index::T_WBh], J_lift.data());

      // hallucinate Jacobian w.r.t. state
      Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> J2(jacobians[2]);
      J2.setZero();
      J2.topLeftCorner<2, 6>() = J2_minimal;
      //J0 = J0_minimal * J_lift;
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
      // compute the minimal version
      Eigen::Matrix<double, 2, 6, Eigen::RowMajor>
          J3_minimal = Jh_weighted * dhC_dExtrinsict;
      Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> J3(jacobians[3]);
      Eigen::Matrix<double, 6, 7, Eigen::RowMajor> J_lift;
      PoseLocalParameterization::liftJacobian(parameters[Index::T_BCt], J_lift.data());
      J3 = J3_minimal * J_lift;
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
      // compute the minimal version
      Eigen::Matrix<double, 2, 6, Eigen::RowMajor>
          J4_minimal = Jh_weighted * dhC_dExtrinsich;
      Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> J4(jacobians[4]);
      //Eigen::Matrix<double, 6, 7, Eigen::RowMajor> J_lift;
      //PoseLocalParameterization::liftJacobian(parameters[Index::T_BCh], J_lift.data());

      //J4 = J4_minimal * J_lift;
      J4.setZero();
      J4.topLeftCorner<2, 6>() = J4_minimal;
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
      Eigen::Matrix<double, 2, Eigen::Dynamic> Jpi_weighted_copy = Jpi_weighted;
      //const int resultCols = Jpi_weighted_copy->cols() - 1;
      //Jpi_weighted_copy->col(0) += Jpi_weighted_copy->col(1);
      //Jpi_weighted_copy->block(0, 1, 2, resultCols - 1) = Jpi_weighted_copy->block(0, 2, 2, resultCols - 1);
      //Jpi_weighted_copy->conservativeResize(Eigen::NoChange, resultCols);
      J5 = Jpi_weighted_copy
               .template topLeftCorner<2, GEOMETRY_TYPE::NumIntrinsics>();
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
    if (jacobians[8] != NULL)
    {
      Eigen::Map<Eigen::Matrix<double, 2, 9, Eigen::RowMajor>> J8(jacobians[8]);
      J8 = Jh_weighted * dhC_sb;
      if (jacobiansMinimal != NULL)
      {
        if (jacobiansMinimal[8] != NULL)
        {
          Eigen::Map<Eigen::Matrix<double, 2, 9, Eigen::RowMajor>>
              J8_minimal_mapped(jacobiansMinimal[8]);
          J8_minimal_mapped = J8;
        }
      }
    }

    // T_gi
    if (jacobians[9] != NULL)
    {
      Eigen::Map<Eigen::Matrix<double, 2, 9, Eigen::RowMajor>> J9(jacobians[9]);
      J9.setZero();
      if (jacobiansMinimal != NULL)
      {
        if (jacobiansMinimal[9] != NULL)
        {
          Eigen::Map<Eigen::Matrix<double, 2, 9, Eigen::RowMajor>>
              J9_minimal_mapped(jacobiansMinimal[9]);
          J9_minimal_mapped = J9;
        }
      }
    }

    // T_si
    if (jacobians[10] != NULL)
    {
      Eigen::Map<Eigen::Matrix<double, 2, 9, Eigen::RowMajor>> J10(jacobians[10]);
      J10.setZero();
      if (jacobiansMinimal != NULL)
      {
        if (jacobiansMinimal[10] != NULL)
        {
          Eigen::Map<Eigen::Matrix<double, 2, 9, Eigen::RowMajor>>
              J10_minimal_mapped(jacobiansMinimal[10]);
          J10_minimal_mapped = J10;
        }
      }
    }

    // T_ai
    if (jacobians[11] != NULL)
    {
      Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> J11(jacobians[11]);
      J11.setZero();
      if (jacobiansMinimal != NULL)
      {
        if (jacobiansMinimal[11] != NULL)
        {
          Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>>
              J11_minimal_mapped(jacobiansMinimal[11]);
          J11_minimal_mapped = J11;
        }
      }
    }
  }
}

template <class GEOMETRY_TYPE>
RS_LocalBearingVector<GEOMETRY_TYPE>::
    RS_LocalBearingVector(
        const RSCameraReprojectionError<GEOMETRY_TYPE> &
            rsre)
    : rsre_(rsre) {}

template <class GEOMETRY_TYPE>
template <typename Scalar>
bool RS_LocalBearingVector<GEOMETRY_TYPE>::
operator()(const Scalar *const T_WBt,
           const Scalar *const hp_Ch,
           const Scalar *const T_WBh,
           const Scalar *const T_BCt,
           const Scalar *const T_BCh,
           const Scalar *const t_r,
           const Scalar *const t_d,
           const Scalar *const speedAndBiases,
           const Scalar *const T_g, const Scalar *const T_s, const Scalar *const T_a,
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
  q_WB_t.normalize();

  //T_WBh
  Eigen::Map<const Eigen::Matrix<Scalar, 3, 1>> p_WB_h0(T_WBh);
  const Eigen::Quaternion<Scalar> q_WB_h0(T_WBh[6], T_WBh[3], T_WBh[4], T_WBh[5]);
  Eigen::Map<const Eigen::Matrix<Scalar, 6, 1>> deltaT_WB_h(deltaT_WSh);
  Eigen::Matrix<Scalar, 3, 1> p_WB_h = p_WB_h0 + deltaT_WB_h.template head<3>();

  Eigen::Matrix<Scalar, 3, 1> omega2 = deltaT_WB_h.template tail<3>();
  Eigen::Quaternion<Scalar> dqWSh = okvis::kinematics::expAndTheta(omega2);
  Eigen::Quaternion<Scalar> q_WB_h = dqWSh * q_WB_h0;
  q_WB_h.normalize();

  //okvis::kinematics::Transformation T_WB_h(p_WB_h, q_WB_h);
  Eigen::Matrix<Scalar, 4, 4> T_WB_h = Eigen::Matrix<Scalar, 4, 4>::Identity();
  T_WB_h.template topLeftCorner<3, 3>() = q_WB_h.toRotationMatrix();
  T_WB_h.template topRightCorner<3, 1>() = p_WB_h;

  //hp_Ch
  Eigen::Map<const Eigen::Matrix<Scalar, 4, 1>> hp_Ch_t(hp_Ch);

  //T_BCt
  Eigen::Matrix<Scalar, 3, 1> p_BC_t0(T_BCt[0], T_BCt[1], T_BCt[2]);
  Eigen::Quaternion<Scalar> q_BC_t0(T_BCt[6], T_BCt[3], T_BCt[4], T_BCt[5]);
  Eigen::Map<const Eigen::Matrix<Scalar, 6, 1>> deltaT_BC_t(deltaExtrinsict);

  Eigen::Matrix<Scalar, 3, 1> p_BC_t = p_BC_t0 + deltaT_BC_t.template head<3>();
  Eigen::Matrix<Scalar, 3, 1> omega3 = deltaT_BC_t.template tail<3>();
  Eigen::Quaternion<Scalar> dqBCt = okvis::kinematics::expAndTheta(omega3);
  //Eigen::Quaternion<Scalar> q_BC_t = dqBCt * q_BC_t0;
  Eigen::Quaternion<Scalar> q_BC_t = q_BC_t0 * dqBCt;
  q_BC_t.normalize();

  //T_BCh
  Eigen::Matrix<Scalar, 3, 1> p_BC_h0(T_BCh[0], T_BCh[1], T_BCh[2]);
  Eigen::Quaternion<Scalar> q_BC_h0(T_BCh[6], T_BCh[3], T_BCh[4], T_BCh[5]);
  Eigen::Map<const Eigen::Matrix<Scalar, 6, 1>> deltaT_BC_h(deltaExtrinsich);
  Eigen::Matrix<Scalar, 3, 1> p_BC_h = p_BC_h0 + deltaT_BC_h.template head<3>();

  Eigen::Matrix<Scalar, 3, 1> omega4 = deltaT_BC_h.template tail<3>();
  Eigen::Quaternion<Scalar> dqBCh = okvis::kinematics::expAndTheta(omega4);
  //Eigen::Quaternion<Scalar> q_BC_h = dqBCh * q_BC_h0;
  Eigen::Quaternion<Scalar> q_BC_h = q_BC_h0 * dqBCh;
  q_BC_h.normalize();

  //okvis::kinematics::Transformation T_BC_h(p_BC_h, q_BC_h);
  Eigen::Matrix<Scalar, 4, 4> T_BC_h = Eigen::Matrix<Scalar, 4, 4>::Identity();
  T_BC_h.template topLeftCorner<3, 3>() = q_BC_h.toRotationMatrix();
  T_BC_h.template topRightCorner<3, 1>() = p_BC_h;

  Scalar trLatestEstimate = t_r[0];

  uint32_t height = rsre_.cameraGeometryBase_->imageHeight();
  double ypixel(rsre_.measurement_[1]);
  Scalar kpN = (Scalar)(ypixel / height - 0.5);
  Scalar tdLatestEstimate = t_d[0];
  Scalar relativeFeatureTime =
      tdLatestEstimate + trLatestEstimate * kpN - (Scalar)(rsre_.targetStateTime_.toSec() - rsre_.targetImageTime_.toSec());

  std::pair<Eigen::Matrix<Scalar, 3, 1>, Eigen::Quaternion<Scalar>> pairT_WB_t(
      p_WB_t, q_WB_t);
  Eigen::Matrix<Scalar, 9, 1> speedBgBa =
      Eigen::Map<const Eigen::Matrix<Scalar, 9, 1>>(speedAndBiases);

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

  if (relativeFeatureTime >= Scalar(5e-8))
  {
    swift_vio::ode::predictStates(imuMeasurements, (Scalar)(rsre_.imuParameters_.g), pairT_WB_t,
                                  speedBgBa, t_start, t_end);
  }
  else if (relativeFeatureTime <= Scalar(-5e-8))
  {
    swift_vio::ode::predictStatesBackward(imuMeasurements, (Scalar)(rsre_.imuParameters_.g),
                                          pairT_WB_t, speedBgBa, t_start, t_end);
  }

  p_WB_t = pairT_WB_t.first;
  q_WB_t = pairT_WB_t.second;
  //Eigen::Matrix<Scalar, 3, 1> p_WB_t = pairT_WB_t.first;
  //Eigen::Quaterniond<Scalar> q_WB_t = pairT_WB_t.second;

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
  Eigen::Matrix<Scalar, 4, 1> hp_W_t = (T_WB_h * T_BC_h) * hp_Ch_t; //
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