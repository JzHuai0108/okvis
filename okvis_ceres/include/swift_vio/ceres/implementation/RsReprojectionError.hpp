
/**
 * @file implementation/RsReprojectionError.hpp
 * @brief Header implementation file for the RsReprojectionError class.
 * @author Jianzhu Huai
 */
#include "ceres/internal/autodiff.h"

#include <okvis/kinematics/Transformation.hpp>
#include <okvis/kinematics/operators.hpp>

#include <swift_vio/ceres/JacobianHelpers.hpp>
#include <swift_vio/ExtrinsicReps.hpp>
#include <swift_vio/ParallaxAnglePoint.hpp>
#include <swift_vio/Measurements.hpp>
#include <swift_vio/imu/SimpleImuOdometry.hpp>
#include <swift_vio/imu/SimpleImuPropagationJacobian.hpp>

namespace okvis {
/// \brief ceres Namespace for ceres-related functionality implemented in okvis.
namespace ceres {
template <class GEOMETRY_TYPE>
RsReprojectionError<GEOMETRY_TYPE>::RsReprojectionError()
    : gravityMag_(9.80665) {}

template <class GEOMETRY_TYPE>
RsReprojectionError<GEOMETRY_TYPE>::
    RsReprojectionError(
        std::shared_ptr<const camera_geometry_t> cameraGeometry,
        const measurement_t& measurement,
        const covariance_t& covariance,
        std::shared_ptr<const okvis::ImuMeasurementDeque> imuMeasCanopy,
        std::shared_ptr<const Eigen::Matrix<double, 6, 1>> positionVelocityLin,
        okvis::Time stateEpoch, okvis::Time imageTime, double gravityMag)
    : imuMeasCanopy_(imuMeasCanopy),
      positionVelocityLin_(positionVelocityLin),
      stateEpoch_(stateEpoch),
      imageTime_(imageTime),
      gravityMag_(gravityMag) {
  setMeasurement(measurement);
  setCovariance(covariance);
  setCameraGeometry(cameraGeometry);
}

template <class GEOMETRY_TYPE>
void RsReprojectionError<GEOMETRY_TYPE>::
    setCovariance(const covariance_t& covariance) {
  information_ = covariance.inverse();
  covariance_ = covariance;
  // perform the Cholesky decomposition on order to obtain the correct error
  // weighting
  Eigen::LLT<Eigen::Matrix2d> lltOfInformation(information_);
  squareRootInformation_ = lltOfInformation.matrixL().transpose();
}

template <class GEOMETRY_TYPE>
bool RsReprojectionError<GEOMETRY_TYPE>::
    Evaluate(double const* const* parameters, double* residuals,
             double** jacobians) const {
  return EvaluateWithMinimalJacobians(parameters, residuals, jacobians, NULL);
}

template <class GEOMETRY_TYPE>
bool RsReprojectionError<GEOMETRY_TYPE>::
    EvaluateWithMinimalJacobians(double const* const* parameters,
                                 double* residuals, double** jacobians,
                                 double** jacobiansMinimal) const {
  return EvaluateWithMinimalJacobiansAnalytic(parameters, residuals, jacobians,
                                              jacobiansMinimal);
}

template <class GEOMETRY_TYPE>
bool RsReprojectionError<GEOMETRY_TYPE>::
    EvaluateWithMinimalJacobiansAnalytic(double const* const* parameters,
                                 double* residuals, double** jacobians,
                                 double** jacobiansMinimal) const {
  Eigen::Map<const Eigen::Vector3d> t_WB_W0(parameters[Index::T_WBt]);
  Eigen::Map<const Eigen::Quaterniond> q_WB0(parameters[Index::T_WBt] + 3);

  // the point in world coordinates
  Eigen::Map<const Eigen::Vector4d> hp_W(parameters[Index::HPP]);

  Eigen::Map<const Eigen::Matrix<double, 3, 1>> t_BC_B(parameters[Index::T_BCt]);
  Eigen::Map<const Eigen::Quaterniond> q_BC(parameters[Index::T_BCt] + 3);
  double trLatestEstimate = parameters[Index::ReadoutTime][0];
  double tdLatestEstimate = parameters[Index::CameraTd][0];
  Eigen::Map<const Eigen::Vector3d> unitgW(parameters[Index::GravityDirection]);

  double ypixel(measurement_[1]);
  uint32_t height = cameraGeometryBase_->imageHeight();
  double kpN = ypixel / height - 0.5;
  double relativeFeatureTime = tdLatestEstimate + trLatestEstimate * kpN + (imageTime_ - stateEpoch_).toSec();
  std::pair<Eigen::Matrix<double, 3, 1>, Eigen::Quaternion<double>> pairT_WB(
      t_WB_W0, q_WB0);
  Eigen::Map<const Eigen::Matrix<double, 3, 1>> v_WB0(parameters[Index::SpeedAndBiases]);
  Eigen::Matrix<double, 3, 1> speed = v_WB0;

  Eigen::Matrix<double, 6, 1> bgBa =
      Eigen::Map<const Eigen::Matrix<double, 6, 1>>(parameters[Index::SpeedAndBiases] + 3);

  const okvis::Time t_start = stateEpoch_;
  const okvis::Time t_end = stateEpoch_ + okvis::Duration(relativeFeatureTime);
  const double wedge = 5e-8;
  Eigen::Vector3d gW = gravityMag_ * unitgW;
  if (relativeFeatureTime >= wedge) {
    swift_vio::ode::predictStates(*imuMeasCanopy_, gW, pairT_WB,
                                speed, bgBa, t_start, t_end);
  } else if (relativeFeatureTime <= -wedge) {
    swift_vio::ode::predictStatesBackward(*imuMeasCanopy_, gW, pairT_WB,
                                        speed, bgBa, t_start, t_end);
  }

  Eigen::Quaterniond q_WB = pairT_WB.second;
  Eigen::Vector3d t_WB_W = pairT_WB.first;

  // transform the point into the camera:
  Eigen::Matrix3d C_BC = q_BC.toRotationMatrix();
  Eigen::Matrix3d C_CB = C_BC.transpose();
  Eigen::Matrix4d T_CB = Eigen::Matrix4d::Identity();
  T_CB.topLeftCorner<3, 3>() = C_CB;
  T_CB.topRightCorner<3, 1>() = -C_CB * t_BC_B;
  Eigen::Matrix3d C_WB = q_WB.toRotationMatrix();
  Eigen::Matrix3d C_BW = C_WB.transpose();
  Eigen::Matrix4d T_BW = Eigen::Matrix4d::Identity();
  T_BW.topLeftCorner<3, 3>() = C_BW;
  T_BW.topRightCorner<3, 1>() = -C_BW * t_WB_W;
  Eigen::Vector4d hp_B = T_BW * hp_W;
  Eigen::Vector4d hp_C = T_CB * hp_B;

  // calculate the reprojection error
  Eigen::Map<const Eigen::Matrix<double, kIntrinsicDim, 1>> intrinsics(parameters[Index::Intrinsics]);

  measurement_t kp;
  Eigen::Matrix<double, 2, 4> Jh;
  Eigen::Matrix<double, 2, 4> Jh_weighted;
  Eigen::Matrix<double, 2, Eigen::Dynamic> Jpi;
  Eigen::Matrix<double, 2, Eigen::Dynamic> Jpi_weighted;
  if (jacobians != NULL) {
    cameraGeometryBase_->projectHomogeneousWithExternalParameters(hp_C, intrinsics, &kp, &Jh, &Jpi);
    Jh_weighted = squareRootInformation_ * Jh;
    Jpi_weighted = squareRootInformation_ * Jpi;
  } else {
    cameraGeometryBase_->projectHomogeneousWithExternalParameters(hp_C, intrinsics, &kp);
  }

  measurement_t error = kp - measurement_;

  // weight:
  measurement_t weighted_error = squareRootInformation_ * error;

  // assign:
  residuals[0] = weighted_error[0];
  residuals[1] = weighted_error[1];

  // check validity:
  bool valid = true;
  if (fabs(hp_C[3]) > 1.0e-8) {
    Eigen::Vector3d p_C = hp_C.template head<3>() / hp_C[3];
    if (p_C[2] < 0.2) {  // 20 cm - not very generic... but reasonable
      // std::cout<<"INVALID POINT"<<std::endl;
      valid = false;
    }
  }

  // calculate jacobians, if required
  if (jacobians != NULL) {
    if (!valid) {
      setJacobiansZero(jacobians, jacobiansMinimal);
      return true;
    }
    std::pair<Eigen::Matrix<double, 3, 1>, Eigen::Quaternion<double>> T_WB_lin = pairT_WB;
    Eigen::Vector3d speedLin = speed;
    if (positionVelocityLin_) {
      // compute position and velocity at t_{f_i,j} with first estimates of
      // position and velocity at t_j.
      T_WB_lin = std::make_pair(positionVelocityLin_->head<3>(), q_WB0);
      speedLin = positionVelocityLin_->tail<3>();
      if (relativeFeatureTime >= wedge) {
        swift_vio::ode::predictStates(*imuMeasCanopy_, gW, T_WB_lin,
                                      speedLin, bgBa, t_start, t_end);
      } else if (relativeFeatureTime <= -wedge) {
        swift_vio::ode::predictStatesBackward(*imuMeasCanopy_, gW, T_WB_lin,
                                              speedLin, bgBa, t_start, t_end);
      }
      C_BW = T_WB_lin.second.toRotationMatrix().transpose();
      t_WB_W = T_WB_lin.first;
      T_BW.topLeftCorner<3, 3>() = C_BW;
      T_BW.topRightCorner<3, 1>() = -C_BW * t_WB_W;
      hp_B = T_BW * hp_W;
      hp_C = T_CB * hp_B;
    }

    Eigen::Matrix<double, 4, 6> dhC_deltaTWS;
    Eigen::Matrix<double, 4, 4> dhC_deltahpW;
    Eigen::Matrix<double, 4, 6> dhC_dExtrinsic;
    Eigen::Vector4d dhC_td;
    Eigen::Matrix<double, 4, 9> dhC_sb;

    Eigen::Vector3d p_BP_W = hp_W.head<3>() - t_WB_W * hp_W[3];
    Eigen::Matrix<double, 4, 6> dhS_deltaTWS;
    dhS_deltaTWS.topLeftCorner<3, 3>() = -C_BW * hp_W[3];
    dhS_deltaTWS.topRightCorner<3, 3>() =
        C_BW * okvis::kinematics::crossMx(p_BP_W);

    Eigen::Matrix3d phi;
    swift_vio::Phi_pq(t_WB_W0, t_WB_W, v_WB0, gW, relativeFeatureTime, &phi);
    dhS_deltaTWS.rightCols<3>() += dhS_deltaTWS.leftCols<3>() * phi;

    dhS_deltaTWS.row(3).setZero();
    dhC_deltaTWS = T_CB * dhS_deltaTWS;
    dhC_deltahpW = T_CB * T_BW;

    dhC_dExtrinsic.block<3, 3>(0, 0) = -C_CB * hp_C[3];
    dhC_dExtrinsic.block<3, 3>(0, 3) =
        okvis::kinematics::crossMx(hp_C.head<3>()) * C_CB;
    dhC_dExtrinsic.row(3).setZero();

    okvis::ImuMeasurement queryValue;
    swift_vio::ode::interpolateInertialData(*imuMeasCanopy_, t_end, queryValue);
    queryValue.measurement.gyroscopes -= bgBa.head<3>();
    Eigen::Vector3d p =
        okvis::kinematics::crossMx(queryValue.measurement.gyroscopes) *
            hp_B.head<3>() +
        C_BW * speedLin * hp_W[3];
    dhC_td.head<3>() = -C_CB * p;
    dhC_td[3] = 0;

    Eigen::Matrix<double, 3, 3> dpC_dp_WBt = -C_CB * C_BW * hp_W[3];
    Eigen::Matrix3d dhC_vW = dpC_dp_WBt * relativeFeatureTime;
    Eigen::Matrix<double, 3, 3> dpC_dq_WBt = C_CB * C_BW * okvis::kinematics::crossMx(hp_W.head<3>() - hp_W[3] * t_WB_W);
    Eigen::Matrix3d dhC_bg = dpC_dq_WBt * C_BW.transpose() * (-relativeFeatureTime);

    dhC_sb.row(3).setZero();
    dhC_sb.topRightCorner<3, 3>().setZero();
    dhC_sb.topLeftCorner<3, 3>() = dhC_vW;
    dhC_sb.block<3, 3>(0, 3) = dhC_bg;

    assignJacobians(parameters, jacobians, jacobiansMinimal, Jh_weighted,
                    Jpi_weighted, dhC_deltaTWS, dhC_deltahpW, dhC_dExtrinsic,
                    dhC_td, dhC_sb, relativeFeatureTime, kpN);
  }
  return true;
}

// This evaluates the error term and additionally computes
// the Jacobians in the minimal internal representation via autodiff
template <class GEOMETRY_TYPE>
bool RsReprojectionError<GEOMETRY_TYPE>::
    EvaluateWithMinimalJacobiansAutoDiff(double const* const* parameters,
                                         double* residuals, double** jacobians,
                                         double** jacobiansMinimal) const {
  const int numOutputs = 4;
  double deltaTWS[6] = {0};
  double deltaTSC[6] = {0};
  double const* const expandedParams[] = {
      parameters[Index::T_WBt], parameters[Index::HPP], parameters[Index::T_BCt],
      parameters[Index::ReadoutTime], parameters[Index::CameraTd],
      parameters[Index::SpeedAndBiases], parameters[Index::GravityDirection],
      deltaTWS, deltaTSC};

  double php_C[numOutputs];
  Eigen::Matrix<double, numOutputs, 7, Eigen::RowMajor> dhC_deltaTWS_full;
  Eigen::Matrix<double, numOutputs, 4, Eigen::RowMajor> dhC_deltahpW;
  Eigen::Matrix<double, numOutputs, 7, Eigen::RowMajor> dhC_dExtrinsic_full;

  Eigen::Matrix<double, 4, kIntrinsicDim,
                Eigen::RowMajor> dhC_dIntrinsic;

  Eigen::Matrix<double, numOutputs, 1> dhC_tr;
  Eigen::Matrix<double, numOutputs, 1> dhC_td;
  Eigen::Matrix<double, numOutputs, 9, Eigen::RowMajor> dhC_sb;
  Eigen::Matrix<double, numOutputs, 3, Eigen::RowMajor> dhC_dunitgW;
  Eigen::Matrix<double, numOutputs, 6, Eigen::RowMajor> dhC_deltaTWS;
  Eigen::Matrix<double, numOutputs, 6, Eigen::RowMajor> dhC_dExtrinsic;

  dhC_dIntrinsic.setZero();
  double* dpC_deltaAll[] = {dhC_deltaTWS_full.data(),
                            dhC_deltahpW.data(),
                            dhC_dExtrinsic_full.data(),
                            dhC_tr.data(),
                            dhC_td.data(),
                            dhC_sb.data(),
                            dhC_dunitgW.data(),
                            dhC_deltaTWS.data(),
                            dhC_dExtrinsic.data()};
  LocalBearingVector<GEOMETRY_TYPE>
      rsre(*this);
  bool diffState =
          ::ceres::internal::AutoDifferentiate<
              ::ceres::internal::StaticParameterDims<7, 4, 7, 1, 1, 9, 3, 6, 6>
             >(rsre, expandedParams, numOutputs, php_C, dpC_deltaAll);
  if (!diffState)
    std::cerr << "Potentially wrong Jacobians in autodiff " << std::endl;

  Eigen::Map<const Eigen::Vector4d> hp_C(&php_C[0]);
  // calculate the reprojection error
  Eigen::Map<const Eigen::Matrix<double, kIntrinsicDim, 1>> intrinsics(parameters[Index::Intrinsics]);
  measurement_t kp;
  Eigen::Matrix<double, 2, 4> Jh;
  Eigen::Matrix<double, 2, 4> Jh_weighted;
  Eigen::Matrix<double, 2, Eigen::Dynamic> Jpi;
  Eigen::Matrix<double, 2, Eigen::Dynamic> Jpi_weighted;
  if (jacobians != NULL) {
    cameraGeometryBase_->projectHomogeneousWithExternalParameters(hp_C, intrinsics, &kp, &Jh, &Jpi);
    Jh_weighted = squareRootInformation_ * Jh;
    Jpi_weighted = squareRootInformation_ * Jpi;
  } else {
    cameraGeometryBase_->projectHomogeneousWithExternalParameters(hp_C, intrinsics, &kp);
  }

  measurement_t error = kp - measurement_;
  measurement_t weighted_error = squareRootInformation_ * error;
  residuals[0] = weighted_error[0];
  residuals[1] = weighted_error[1];

  // check validity:
  bool valid = true;
  if (fabs(hp_C[3]) > 1.0e-8) {
    Eigen::Vector3d p_C = hp_C.template head<3>() / hp_C[3];
    if (p_C[2] < 0.2) {  // 20 cm - not very generic... but reasonable
      // std::cout<<"INVALID POINT"<<std::endl;
      valid = false;
    }
  }

  // calculate jacobians, if required
  // This is pretty close to Paul Furgale's thesis. eq. 3.100 on page 40
  if (jacobians != NULL) {
    if (!valid) {
      setJacobiansZero(jacobians, jacobiansMinimal);
      return true;
    }

    double trLatestEstimate = parameters[Index::ReadoutTime][0];
    double tdLatestEstimate = parameters[Index::CameraTd][0];
    uint32_t height = cameraGeometryBase_->imageHeight();
    double ypixel(measurement_[1]);
    double kpN = ypixel / height - 0.5;
    double relativeFeatureTime =
        tdLatestEstimate + trLatestEstimate * kpN + (imageTime_ - stateEpoch_).toSec();
    assignJacobians(parameters, jacobians, jacobiansMinimal, Jh_weighted,
                    Jpi_weighted, dhC_deltaTWS, dhC_deltahpW, dhC_dExtrinsic,
                    dhC_td, dhC_sb, relativeFeatureTime, kpN);
  }
  return true;
}

template <class GEOMETRY_TYPE>
void RsReprojectionError<GEOMETRY_TYPE>::
    setJacobiansZero(double** jacobians, double** jacobiansMinimal) const {
  zeroJacobian<7, 6, 2>(Index::T_WBt, jacobians, jacobiansMinimal);
  zeroJacobian<4, 3, 2>(Index::HPP, jacobians, jacobiansMinimal);
  zeroJacobian<7, 6, 2>(Index::T_BCt, jacobians, jacobiansMinimal);
  zeroJacobian<kIntrinsicDim, kIntrinsicDim, 2>(Index::Intrinsics, jacobians, jacobiansMinimal);
  zeroJacobian<1, 1, 2>(Index::ReadoutTime, jacobians, jacobiansMinimal);
  zeroJacobian<1, 1, 2>(Index::CameraTd, jacobians, jacobiansMinimal);
  zeroJacobian<9, 9, 2>(Index::SpeedAndBiases, jacobians, jacobiansMinimal);
  zeroJacobian<3, 2, 2>(Index::GravityDirection, jacobians, jacobiansMinimal);
}

template <class GEOMETRY_TYPE>
void RsReprojectionError<GEOMETRY_TYPE>::
    assignJacobians(
        double const* const* parameters, double** jacobians,
        double** jacobiansMinimal,
        const Eigen::Matrix<double, 2, 4>& Jh_weighted,
        const Eigen::Matrix<double, 2, Eigen::Dynamic>& Jpi_weighted,
        const Eigen::Matrix<double, 4, 6>& dhC_deltaTWS,
        const Eigen::Matrix<double, 4, 4>& dhC_deltahpW,
        const Eigen::Matrix<double, 4, 6>& dhC_dExtrinsic,
        const Eigen::Vector4d& dhC_td, const Eigen::Matrix<double, 4, 9>& dhC_sb,
        double relativeFeatureTime, double kpN) const {
  if (jacobians[0] != NULL) {
    Eigen::Matrix<double, 2, 6, Eigen::RowMajor> J0_minimal;
    J0_minimal = Jh_weighted * dhC_deltaTWS;
    // pseudo inverse of the local parametrization Jacobian
    Eigen::Matrix<double, 6, 7, Eigen::RowMajor> J_lift;
    swift_vio::PoseLocalParameterizationSimplified::liftJacobian(parameters[Index::T_WBt], J_lift.data());

    // hallucinate Jacobian w.r.t. state
    Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> J0(jacobians[0]);
    J0 = J0_minimal * J_lift;
    if (jacobiansMinimal != NULL) {
      if (jacobiansMinimal[0] != NULL) {
        Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>>
            J0_minimal_mapped(jacobiansMinimal[0]);
        J0_minimal_mapped = J0_minimal;
      }
    }
  }

  if (jacobians[1] != NULL) {
    Eigen::Map<Eigen::Matrix<double, 2, 4, Eigen::RowMajor>> J1(jacobians[1]);
    J1 = Jh_weighted * dhC_deltahpW;
    if (jacobiansMinimal != NULL) {
      if (jacobiansMinimal[1] != NULL) {
        Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>>
            J1_minimal_mapped(jacobiansMinimal[1]);
        Eigen::Matrix<double, 4, 3> S;
        S.setZero();
        S.topLeftCorner<3, 3>().setIdentity();
        J1_minimal_mapped = J1 * S;
      }
    }
  }

  if (jacobians[2] != NULL) {
    // compute the minimal version
    Eigen::Matrix<double, 2, 6, Eigen::RowMajor>
        J2_minimal = Jh_weighted * dhC_dExtrinsic;
    Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> J2(jacobians[2]);
    Eigen::Matrix<double, 6, 7, Eigen::RowMajor> J_lift;
    swift_vio::PoseLocalParameterizationSimplified::liftJacobian(parameters[Index::T_BCt], J_lift.data());
    J2 = J2_minimal * J_lift;

    if (jacobiansMinimal != NULL) {
      if (jacobiansMinimal[2] != NULL) {
        Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>>
            J2_minimal_mapped(jacobiansMinimal[2]);
        J2_minimal_mapped = J2_minimal;
      }
    }
  }

  // camera intrinsics
  if (jacobians[3] != NULL) {
    Eigen::Map<Eigen::Matrix<double, 2, kIntrinsicDim,
        Eigen::RowMajor>> J1(jacobians[3]);
    J1 = Jpi_weighted;
    if (jacobiansMinimal != NULL) {
      if (jacobiansMinimal[3] != NULL) {
        Eigen::Map<Eigen::Matrix<double, 2, kIntrinsicDim,
            Eigen::RowMajor>> J1_minimal_mapped(jacobiansMinimal[3]);
        J1_minimal_mapped = J1;
      }
    }
  }

  if (jacobians[Index::ReadoutTime] != NULL) {
    Eigen::Map<Eigen::Matrix<double, 2, 1>> J1(jacobians[Index::ReadoutTime]);
    J1 = Jh_weighted * dhC_td * kpN;
    if (jacobiansMinimal != NULL) {
      if (jacobiansMinimal[Index::ReadoutTime] != NULL) {
        Eigen::Map<Eigen::Matrix<double, 2, 1>>
            J1_minimal_mapped(jacobiansMinimal[Index::ReadoutTime]);
        J1_minimal_mapped = J1;
      }
    }
  }

  // t_d
  if (jacobians[Index::CameraTd] != NULL) {
    Eigen::Map<Eigen::Matrix<double, 2, 1>> J1(jacobians[Index::CameraTd]);
    J1 = Jh_weighted * dhC_td;
    if (jacobiansMinimal != NULL) {
      if (jacobiansMinimal[Index::CameraTd] != NULL) {
        Eigen::Map<Eigen::Matrix<double, 2, 1>> J1_minimal_mapped(
            jacobiansMinimal[Index::CameraTd]);
        J1_minimal_mapped = J1;
      }
    }
  }

  // speed and gyro biases and accel biases
  if (jacobians[Index::SpeedAndBiases] != NULL) {
    Eigen::Map<Eigen::Matrix<double, 2, 9, Eigen::RowMajor>> J1(jacobians[Index::SpeedAndBiases]);
    J1 = Jh_weighted * dhC_sb;
    if (jacobiansMinimal != NULL) {
      if (jacobiansMinimal[Index::SpeedAndBiases] != NULL) {
        Eigen::Map<Eigen::Matrix<double, 2, 9, Eigen::RowMajor>>
            J1_minimal_mapped(jacobiansMinimal[Index::SpeedAndBiases]);
        J1_minimal_mapped = J1;
      }
    }
  }

  if (jacobians[Index::GravityDirection] != NULL) {
    Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> J1(jacobians[Index::GravityDirection]);
    J1 = Jh_weighted * dhC_sb.topLeftCorner<4, 3>() * 0.5 * relativeFeatureTime * gravityMag_;
    if (jacobiansMinimal != NULL) {
      if (jacobiansMinimal[Index::GravityDirection] != NULL) {
        Eigen::Map<Eigen::Matrix<double, 2, 2, Eigen::RowMajor>>
            J1_minimal_mapped(jacobiansMinimal[Index::GravityDirection]);

        Eigen::Matrix<double, 3, 2, Eigen::RowMajor> dunitgW_du;
        swift_vio::NormalVectorParameterization::plusJacobian(
              parameters[Index::GravityDirection], dunitgW_du.data());
        J1_minimal_mapped = J1 * dunitgW_du;
      }
    }
  }
}


template <class GEOMETRY_TYPE>
LocalBearingVector<GEOMETRY_TYPE>::
    LocalBearingVector(const RsReprojectionError<GEOMETRY_TYPE>& rsre)
    : rsre_(rsre) {}

template <class GEOMETRY_TYPE>
template <typename Scalar>
bool LocalBearingVector<GEOMETRY_TYPE>::
operator()(const Scalar* const T_WB, const Scalar* const php_W,
           const Scalar* const T_BC_params,
           const Scalar* const t_r,
           const Scalar* const t_d, const Scalar* const speedAndBiases,
           const Scalar* const unitgW,
           const Scalar* const deltaT_WB, const Scalar* const deltaT_BC,
           Scalar residuals[4]) const {
  Eigen::Map<const Eigen::Matrix<Scalar, 3, 1>> t_WB_W0(T_WB);
  const Eigen::Quaternion<Scalar> q_WB0(T_WB[6], T_WB[3], T_WB[4], T_WB[5]);
  Eigen::Map<const Eigen::Matrix<Scalar, 6, 1>> deltaT_WBe(deltaT_WB);
  Eigen::Matrix<Scalar, 3, 1> t_WB_W = t_WB_W0 + deltaT_WBe.template head<3>();
  Eigen::Matrix<Scalar, 3, 1> omega = deltaT_WBe.template tail<3>();
  Eigen::Quaternion<Scalar> dqWS = okvis::kinematics::expAndTheta(omega);
  Eigen::Quaternion<Scalar> q_WB = dqWS * q_WB0;

  Eigen::Map<const Eigen::Matrix<Scalar, 4, 1>> hp_W(php_W);

  Eigen::Matrix<Scalar, 7, 1> T_BC_temp;
  swift_vio::PoseLocalParameterizationSimplified::oplus(T_BC_params, deltaT_BC, T_BC_temp.data());
  Eigen::Matrix<Scalar, 3, 1> t_BC_B = T_BC_temp.template head<3>();
  Eigen::Map<Eigen::Quaternion<Scalar>> q_BC(T_BC_temp.data() + 3);

  Scalar trLatestEstimate = t_r[0];
  uint32_t height = rsre_.cameraGeometryBase_->imageHeight();
  double ypixel(rsre_.measurement_[1]);
  Scalar kpN = (Scalar)(ypixel / height - 0.5);
  Scalar tdLatestEstimate = t_d[0];
  Scalar relativeFeatureTime =
      tdLatestEstimate + trLatestEstimate * kpN + (Scalar)(rsre_.imageTime_ - rsre_.stateEpoch_).toSec();

  std::pair<Eigen::Matrix<Scalar, 3, 1>, Eigen::Quaternion<Scalar>> pairT_WB(
      t_WB_W, q_WB);
  Eigen::Matrix<Scalar, 3, 1> speed =
      Eigen::Map<const Eigen::Matrix<Scalar, 3, 1>>(speedAndBiases);
  Eigen::Matrix<Scalar, 6, 1> bgBa =
      Eigen::Map<const Eigen::Matrix<Scalar, 6, 1>>(speedAndBiases + 3);
  Eigen::Map<const Eigen::Matrix<Scalar, 3, 1>> gravityDirection(unitgW);

  Scalar t_start = (Scalar)rsre_.stateEpoch_.toSec();
  Scalar t_end = t_start + relativeFeatureTime;
  swift_vio::GenericImuMeasurementDeque<Scalar> imuMeasurements;
  for (size_t jack = 0; jack < rsre_.imuMeasCanopy_->size(); ++jack) {
    swift_vio::GenericImuMeasurement<Scalar> imuMeas(
        (Scalar)(rsre_.imuMeasCanopy_->at(jack).timeStamp.toSec()),
        rsre_.imuMeasCanopy_->at(jack).measurement.gyroscopes.template cast<Scalar>(),
        rsre_.imuMeasCanopy_->at(jack).measurement.accelerometers.template cast<Scalar>());
    imuMeasurements.push_back(imuMeas);
  }

  Eigen::Matrix<Scalar, 3, 1> gW = ((Scalar)rsre_.gravityMag_) * gravityDirection;
  if (relativeFeatureTime >= Scalar(5e-8)) {
    swift_vio::ode::predictStates(imuMeasurements, gW, pairT_WB,
                                  speed, bgBa, t_start, t_end);
  } else if (relativeFeatureTime <= Scalar(-5e-8)) {
    swift_vio::ode::predictStatesBackward(imuMeasurements, gW, pairT_WB,
                                          speed, bgBa, t_start, t_end);
  }

  q_WB = pairT_WB.second;
  t_WB_W = pairT_WB.first;

  // transform the point into the camera:
  Eigen::Matrix<Scalar, 3, 3> C_BC = q_BC.toRotationMatrix();
  Eigen::Matrix<Scalar, 3, 3> C_CB = C_BC.transpose();
  Eigen::Matrix<Scalar, 4, 4> T_CB = Eigen::Matrix<Scalar, 4, 4>::Identity();
  T_CB.template topLeftCorner<3, 3>() = C_CB;
  T_CB.template topRightCorner<3, 1>() = -C_CB * t_BC_B;
  Eigen::Matrix<Scalar, 3, 3> C_WB = q_WB.toRotationMatrix();
  Eigen::Matrix<Scalar, 3, 3> C_BW = C_WB.transpose();
  Eigen::Matrix<Scalar, 4, 4> T_BW = Eigen::Matrix<Scalar, 4, 4>::Identity();
  T_BW.template topLeftCorner<3, 3>() = C_BW;
  T_BW.template topRightCorner<3, 1>() = -C_BW * t_WB_W;
  Eigen::Matrix<Scalar, 4, 1> hp_B = T_BW * hp_W;
  Eigen::Matrix<Scalar, 4, 1> hp_C = T_CB * hp_B;

  residuals[0] = hp_C[0];
  residuals[1] = hp_C[1];
  residuals[2] = hp_C[2];
  residuals[3] = hp_C[3];

  return true;
}
}  // namespace ceres
}  // namespace okvis
