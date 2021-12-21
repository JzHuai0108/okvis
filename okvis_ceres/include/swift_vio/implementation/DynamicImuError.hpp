#include <swift_vio/memory.h>
#include <swift_vio/ParallaxAnglePoint.hpp>

//#include <chrono>
#include <thread>

#include <glog/logging.h>

#include <okvis/kinematics/operators.hpp>
#include <okvis/Parameters.hpp>
#include <okvis/ceres/PoseLocalParameterization.hpp>
#include <okvis/assert_macros.hpp>
#include <okvis/kinematics/Transformation.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {
/// \brief ceres Namespace for ceres-related functionality implemented in okvis.
namespace ceres {

// Construct with measurements and parameters.
template <typename ImuModelT>
DynamicImuError<ImuModelT>::DynamicImuError(const okvis::ImuMeasurementDeque & imuMeasurements,
                   const okvis::ImuParameters & imuParameters,
                   const okvis::Time& t_0, const okvis::Time& t_1) {
  setImuMeasurements(imuMeasurements);
  setImuParameters(imuParameters);
  setT0(t_0);
  setT1(t_1);

  OKVIS_ASSERT_TRUE_DBG(Exception,
                     t_0 >= imuMeasurements.front().timeStamp,
                     "First IMU measurement included in DynamicImuError is not old enough!");
  OKVIS_ASSERT_TRUE_DBG(Exception,
                     t_1 <= imuMeasurements.back().timeStamp,
                     "Last IMU measurement included in DynamicImuError is not new enough!");
}

// Propagates pose, speeds and biases with given IMU measurements.
template <typename ImuModelT>
int DynamicImuError<ImuModelT>::redoPreintegration(const okvis::kinematics::Transformation& /*T_WS*/,
                                 const okvis::SpeedAndBias & speedAndBiases) const {
//  auto start = std::chrono::high_resolution_clock::now();
  // ensure unique access
  std::lock_guard<std::mutex> lock(preintegrationMutex_);

  // now the propagation
  okvis::Time time = t0_;
  okvis::Time end = t1_;

  // sanity check:
  assert(imuMeasurements_.front().timeStamp<=time);
  if (!(imuMeasurements_.back().timeStamp >= end))
    return -1;  // nothing to do...

  imuModel_.resetPreintegration();

  double Delta_t = 0;
  bool hasStarted = false;
  int i = 0;
  for (okvis::ImuMeasurementDeque::const_iterator it = imuMeasurements_.begin();
      it != imuMeasurements_.end(); ++it) {

    Eigen::Vector3d omega_S_0 = it->measurement.gyroscopes;
    Eigen::Vector3d acc_S_0 = it->measurement.accelerometers;
    Eigen::Vector3d omega_S_1 = (it + 1)->measurement.gyroscopes;
    Eigen::Vector3d acc_S_1 = (it + 1)->measurement.accelerometers;

    // time delta
    okvis::Time nexttime;
    if ((it + 1) == imuMeasurements_.end()) {
      nexttime = t1_;
    } else
      nexttime = (it + 1)->timeStamp;
    double dt = (nexttime - time).toSec();

    if (end < nexttime) {
      double interval = (nexttime - it->timeStamp).toSec();
      nexttime = t1_;
      dt = (nexttime - time).toSec();
      const double r = dt / interval;
      omega_S_1 = ((1.0 - r) * omega_S_0 + r * omega_S_1).eval();
      acc_S_1 = ((1.0 - r) * acc_S_0 + r * acc_S_1).eval();
    }

    if (dt <= 0.0) {
      continue;
    }
    Delta_t += dt;

    if (!hasStarted) {
      hasStarted = true;
      const double r = dt / (nexttime - it->timeStamp).toSec();
      omega_S_0 = (r * omega_S_0 + (1.0 - r) * omega_S_1).eval();
      acc_S_0 = (r * acc_S_0 + (1.0 - r) * acc_S_1).eval();
    }

    // ensure integrity
    double sigma_g_c = imuParameters_.sigma_g_c;
    double sigma_a_c = imuParameters_.sigma_a_c;

    if (fabs(omega_S_0[0]) > imuParameters_.g_max
        || fabs(omega_S_0[1]) > imuParameters_.g_max
        || fabs(omega_S_0[2]) > imuParameters_.g_max
        || fabs(omega_S_1[0]) > imuParameters_.g_max
        || fabs(omega_S_1[1]) > imuParameters_.g_max
        || fabs(omega_S_1[2]) > imuParameters_.g_max) {
      sigma_g_c *= 100;
      LOG(WARNING)<< "gyr saturation";
    }

    if (fabs(acc_S_0[0]) > imuParameters_.a_max || fabs(acc_S_0[1]) > imuParameters_.a_max
        || fabs(acc_S_0[2]) > imuParameters_.a_max
        || fabs(acc_S_1[0]) > imuParameters_.a_max
        || fabs(acc_S_1[1]) > imuParameters_.a_max
        || fabs(acc_S_1[2]) > imuParameters_.a_max) {
      sigma_a_c *= 100;
      LOG(WARNING)<< "acc saturation";
    }

    Eigen::Vector3d omega_est_0;
    Eigen::Vector3d acc_est_0;
    imuModel_.correct(omega_S_0, acc_S_0, &omega_est_0, &acc_est_0);

    Eigen::Vector3d omega_est_1;
    Eigen::Vector3d acc_est_1;
    imuModel_.correct(omega_S_1, acc_S_1, &omega_est_1, &acc_est_1);

    imuModel_.propagate(dt, omega_est_0, acc_est_0, omega_est_1, acc_est_1,
                        sigma_g_c, sigma_a_c, imuParameters_.sigma_gw_c,
                        imuParameters_.sigma_aw_c);

    time = nexttime;

    ++i;

    if (nexttime == t1_)
      break;

  }

  // store the reference (linearisation) point
  speedAndBiases_ref_ = speedAndBiases;

  if (reweight_) {
    imuModel_.getWeight(&information_);
  }

  // square root
  Eigen::LLT<information_t> lltOfInformation(information_);
  squareRootInformation_ = lltOfInformation.matrixL().transpose();

//  auto stop = std::chrono::high_resolution_clock::now();
//  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
//  std::cout << "Preintegration costs " << duration.count() << " ms.\n";
  return i;
}

// This evaluates the error term and additionally computes the Jacobians.
template <typename ImuModelT>
bool DynamicImuError<ImuModelT>::Evaluate(double const* const * parameters, double* residuals,
                        double** jacobians) const {
  return EvaluateWithMinimalJacobians(parameters, residuals, jacobians, NULL);
}

template <typename ImuModelT>
template <size_t Start, size_t End>
void DynamicImuError<ImuModelT>::fillAnalyticJacLoop(
    double **jacobians, double **jacobiansMinimal,
    const Eigen::Matrix<double, 3, 3> &derot_dDrot, const ImuModelT &imuModel) const {
  if constexpr (Start < End) {
    if (jacobians != NULL && jacobians[5 + Start] != NULL) {
      Eigen::Map<Eigen::Matrix<double, 15, ImuModelT::kXBlockDims[Start],
                               Eigen::RowMajor>>
          J(jacobians[5 + Start]);
      J.template block<3, ImuModelT::kXBlockDims[Start]>(0, 0) =
          imuModel.template dDp_dx<Start>();
      J.template block<3, ImuModelT::kXBlockDims[Start]>(3, 0) =
          derot_dDrot * imuModel.template dDrot_dx<Start>();
      J.template block<3, ImuModelT::kXBlockDims[Start]>(6, 0) =
          imuModel.template dDv_dx<Start>();
      J.template bottomRows<6>().setZero();
      J = squareRootInformation_ * J;
      if (jacobiansMinimal != NULL && jacobiansMinimal[5 + Start] != NULL) {
        Eigen::Map<Eigen::Matrix<double, 15, ImuModelT::kXBlockDims[Start],
                                 Eigen::RowMajor>>
            Jm(jacobiansMinimal[5 + Start]);
        Jm.template block<3, ImuModelT::kXBlockDims[Start]>(0, 0) =
            imuModel.template dDp_dminx<Start>();
        Jm.template block<3, ImuModelT::kXBlockDims[Start]>(3, 0) =
            derot_dDrot * imuModel.template dDrot_dminx<Start>();
        Jm.template block<3, ImuModelT::kXBlockDims[Start]>(6, 0) =
            imuModel.template dDv_dminx<Start>();
        Jm.template bottomRows<6>().setZero();
        Jm = squareRootInformation_ * Jm;
      }
    }
    fillAnalyticJacLoop<Start + 1, End>(jacobians, jacobiansMinimal,
                                        derot_dDrot, imuModel);
  }
}

// This evaluates the error term and additionally computes
// the Jacobians in the minimal internal representation.
template <typename ImuModelT>
bool DynamicImuError<ImuModelT>::EvaluateWithMinimalJacobians(double const* const * parameters,
                                            double* residuals,
                                            double** jacobians,
                                            double** jacobiansMinimal) const {

  // get poses
  const okvis::kinematics::Transformation T_WS_0(
      Eigen::Vector3d(parameters[0][0], parameters[0][1], parameters[0][2]),
      Eigen::Quaterniond(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]));

  const okvis::kinematics::Transformation T_WS_1(
      Eigen::Vector3d(parameters[2][0], parameters[2][1], parameters[2][2]),
      Eigen::Quaterniond(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]));

  // get speed and bias
  okvis::SpeedAndBias speedAndBiases_0;
  okvis::SpeedAndBias speedAndBiases_1;
  for (size_t i = 0; i < 9; ++i) {
    speedAndBiases_0[i] = parameters[1][i];
    speedAndBiases_1[i] = parameters[3][i];
  }

  imuModel_.updateParameters(parameters[1] + 3, parameters + 5);

  // this will NOT be changed:
  const Eigen::Matrix3d C_WS_0 = T_WS_0.C();
  const Eigen::Matrix3d C_S0_W = C_WS_0.transpose();

  // call the propagation
  const double Delta_t = (t1_ - t0_).toSec();
  Eigen::Matrix<double, 6, 1> Delta_b;
  // ensure unique access
  {
    std::lock_guard<std::mutex> lock(preintegrationMutex_);
    Delta_b = speedAndBiases_0.tail<6>()
          - speedAndBiases_ref_.tail<6>();
  }
  redo_ = redo_ || (Delta_b.head<3>().norm() * Delta_t > 0.0001);
  if (redo_) {
    redoPreintegration(T_WS_0, speedAndBiases_0);
    redoCounter_++;
    Delta_b.setZero();
    redo_ = false;
    /*if (redoCounter_ > 1) {
      std::cout << "pre-integration no. " << redoCounter_ << std::endl;
    }*/
  }

  // actual propagation output:
  {
    std::lock_guard<std::mutex> lock(preintegrationMutex_); // this is a bit stupid, but shared read-locks only come in C++14

    Eigen::Map<const Eigen::Vector3d> gravityDirection(parameters[4]);
    const Eigen::Vector3d g_W = imuParameters_.g * gravityDirection;

    // assign Jacobian w.r.t. x0
    Eigen::Matrix<double, 3, 3> dDp_dbg = imuModel_.dDp_dbg();
    Eigen::Matrix<double, 3, 3> dDrot_dbg = imuModel_.dDrot_dbg();
    Eigen::Matrix<double, 3, 3> dDv_dbg = imuModel_.dDv_dbg();
    Eigen::Matrix<double, 3, 3> dDp_dba = imuModel_.dDp_dba();
//    Eigen::Matrix<double, 3, 3> dDrot_dba = imuModel_.dDrot_dba();
    Eigen::Matrix<double, 3, 3> dDv_dba = imuModel_.dDv_dba();

    Eigen::Matrix<double,15,15> F0 =
        Eigen::Matrix<double,15,15>::Identity(); // holds for d/db_g, d/db_a
    const Eigen::Vector3d delta_p_est_W =
        T_WS_0.r() - T_WS_1.r() + speedAndBiases_0.head<3>()*Delta_t + 0.5*g_W*Delta_t*Delta_t;
    const Eigen::Vector3d delta_v_est_W =
        speedAndBiases_0.head<3>() - speedAndBiases_1.head<3>() + g_W*Delta_t;
    const Eigen::Quaterniond Dq = okvis::kinematics::deltaQ(dDrot_dbg*Delta_b.head<3>())*imuModel_.Delta_q();
    F0.block<3,3>(0,0) = C_S0_W;
    F0.block<3,3>(0,3) = C_S0_W * okvis::kinematics::crossMx(delta_p_est_W);
    F0.block<3,3>(0,6) = C_S0_W * Eigen::Matrix3d::Identity()*Delta_t;
    F0.block<3,3>(0,9) = dDp_dbg;
    F0.block<3,3>(0,12) = dDp_dba;
    F0.block<3,3>(3,3) = (okvis::kinematics::plus(Dq*T_WS_1.q().inverse()) *
        okvis::kinematics::oplus(T_WS_0.q())).topLeftCorner<3,3>();
    F0.block<3,3>(3,9) = (okvis::kinematics::oplus(T_WS_1.q().inverse()*T_WS_0.q())*
        okvis::kinematics::oplus(Dq)).topLeftCorner<3,3>()*(dDrot_dbg);
    F0.block<3,3>(6,3) = C_S0_W * okvis::kinematics::crossMx(delta_v_est_W);
    F0.block<3,3>(6,6) = C_S0_W;
    F0.block<3,3>(6,9) = dDv_dbg;
    F0.block<3,3>(6,12) = dDv_dba;

    // assign Jacobian w.r.t. x1
    Eigen::Matrix<double,15,15> F1 =
        -Eigen::Matrix<double,15,15>::Identity(); // holds for the biases
    F1.block<3,3>(0,0) = -C_S0_W;
    F1.block<3,3>(3,3) = -(okvis::kinematics::plus(Dq) *
        okvis::kinematics::oplus(T_WS_0.q()) *
        okvis::kinematics::plus(T_WS_1.q().inverse())).topLeftCorner<3,3>();
    F1.block<3,3>(6,6) = -C_S0_W;

    // the overall error vector
    Eigen::Matrix<double, 15, 1> error;
    error.segment<3>(0) =  C_S0_W * delta_p_est_W + imuModel_.Delta_p() + F0.block<3,6>(0,9)*Delta_b;
    error.segment<3>(3) = 2*(Dq*(T_WS_1.q().inverse()*T_WS_0.q())).vec(); //2*T_WS_0.q()*Dq*T_WS_1.q().inverse();//
    error.segment<3>(6) = C_S0_W * delta_v_est_W + imuModel_.Delta_v() + F0.block<3,6>(6,9)*Delta_b;
    error.tail<6>() = speedAndBiases_0.tail<6>() - speedAndBiases_1.tail<6>();

    // error weighting
    Eigen::Map<Eigen::Matrix<double, 15, 1> > weighted_error(residuals);
    weighted_error = squareRootInformation_ * error;

    Eigen::Matrix<double, 3, 3> derot_dDrot = (okvis::kinematics::oplus(T_WS_1.q().inverse()*T_WS_0.q())*
                                               okvis::kinematics::oplus(Dq)).topLeftCorner<3,3>();
    // get the Jacobians
    if (jacobians != NULL) {
      if (jacobians[0] != NULL) {
        // Jacobian w.r.t. minimal perturbance
        Eigen::Matrix<double, 15, 6> J0_minimal = squareRootInformation_
            * F0.block<15, 6>(0, 0);

        // pseudo inverse of the local parametrization Jacobian:
        Eigen::Matrix<double, 6, 7, Eigen::RowMajor> J_lift;
        PoseLocalParameterization::liftJacobian(parameters[0], J_lift.data());

        // hallucinate Jacobian w.r.t. state
        Eigen::Map<Eigen::Matrix<double, 15, 7, Eigen::RowMajor> > J0(
            jacobians[0]);
        J0 = J0_minimal * J_lift;

        // if requested, provide minimal Jacobians
        if (jacobiansMinimal != NULL) {
          if (jacobiansMinimal[0] != NULL) {
            Eigen::Map<Eigen::Matrix<double, 15, 6, Eigen::RowMajor> > J0_minimal_mapped(
                jacobiansMinimal[0]);
            J0_minimal_mapped = J0_minimal;
          }
        }
      }
      if (jacobians[1] != NULL) {
        Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor> > J1(
            jacobians[1]);
        J1 = squareRootInformation_ * F0.block<15, 9>(0, 6);

        // if requested, provide minimal Jacobians
        if (jacobiansMinimal != NULL) {
          if (jacobiansMinimal[1] != NULL) {
            Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor> > J1_minimal_mapped(
                jacobiansMinimal[1]);
            J1_minimal_mapped = J1;
          }
        }
      }
      if (jacobians[2] != NULL) {
        // Jacobian w.r.t. minimal perturbance
        Eigen::Matrix<double, 15, 6> J2_minimal = squareRootInformation_
                    * F1.block<15, 6>(0, 0);

        // pseudo inverse of the local parametrization Jacobian:
        Eigen::Matrix<double, 6, 7, Eigen::RowMajor> J_lift;
        PoseLocalParameterization::liftJacobian(parameters[2], J_lift.data());

        // hallucinate Jacobian w.r.t. state
        Eigen::Map<Eigen::Matrix<double, 15, 7, Eigen::RowMajor> > J2(
            jacobians[2]);
        J2 = J2_minimal * J_lift;

        // if requested, provide minimal Jacobians
        if (jacobiansMinimal != NULL) {
          if (jacobiansMinimal[2] != NULL) {
            Eigen::Map<Eigen::Matrix<double, 15, 6, Eigen::RowMajor> > J2_minimal_mapped(
                jacobiansMinimal[2]);
            J2_minimal_mapped = J2_minimal;
          }
        }
      }
      if (jacobians[3] != NULL) {
        Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor> > J3(jacobians[3]);
        J3 = squareRootInformation_ * F1.block<15, 9>(0, 6);

        // if requested, provide minimal Jacobians
        if (jacobiansMinimal != NULL) {
          if (jacobiansMinimal[3] != NULL) {
            Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor> > J3_minimal_mapped(
                jacobiansMinimal[3]);
            J3_minimal_mapped = J3;
          }
        }
      }
      if (jacobians[4] != NULL) {
        Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> J4(
            jacobians[4]);
        Eigen::Matrix<double, 15, 3> de_dunitgW =
            Eigen::Matrix<double, 15, 3>::Zero();
        de_dunitgW.topLeftCorner<3, 3>() = 0.5 * Delta_t * Delta_t *
                                           imuParameters_.g *
                                           C_S0_W;
        de_dunitgW.block<3, 3>(6, 0) = Delta_t * imuParameters_.g * C_S0_W;
        J4 = squareRootInformation_ * de_dunitgW;

        // if requested, provide minimal Jacobians
        if (jacobiansMinimal != NULL) {
          if (jacobiansMinimal[4] != NULL) {
            Eigen::Map<Eigen::Matrix<double, 15, 2, Eigen::RowMajor>>
                J4_minimal_mapped(jacobiansMinimal[4]);
            Eigen::Matrix<double, 3, 2, Eigen::RowMajor> dunitgW_du;
            swift_vio::NormalVectorParameterization::plusJacobian(
                gravityDirection.data(), dunitgW_du.data());
            J4_minimal_mapped = J4 * dunitgW_du;
          }
        }
      }

      fillAnalyticJacLoop<0, ImuModelT::kXBlockDims.size()>(jacobians, jacobiansMinimal, derot_dDrot, imuModel_);
    }
  }
  return true;
}

template <typename ImuModelT>
template <size_t Start, size_t End>
void DynamicImuError<ImuModelT>::fillNumericJacLoop(double *const * parameters,
                                  double **jacobians,
                                  double **jacobiansMinimal,
                                  const ImuModelT &imuModel) const {
  if constexpr (Start < End) {
    double dx = 1e-6;
    if (jacobians != NULL && jacobians[5 + Start] != NULL) {
      constexpr size_t blockDim = ImuModelT::kXBlockDims[Start];
      Eigen::Map<Eigen::Matrix<double, 15, blockDim,
                               Eigen::RowMajor>>
          J(jacobians[5 + Start]);
      double originalParams[ImuModelT::kXBlockDims[Start]];
      memcpy(originalParams, parameters[5 + Start], sizeof(double) * blockDim);
      Eigen::Map<Eigen::Matrix<double, blockDim, 1>> parameterBlock(parameters[5 + Start]);
      for (size_t i = 0; i <blockDim; ++i) {
        Eigen::Matrix<double, blockDim, 1> ds_0;
        Eigen::Matrix<double, 15, 1> residuals_p;
        Eigen::Matrix<double, 15, 1> residuals_m;
        ds_0.setZero();
        ds_0[i] = dx;
        parameterBlock += ds_0;
        redo_ = true;
        Evaluate(parameters, residuals_p.data(), NULL);
        memcpy(parameters[5 + Start], originalParams, sizeof(double) * blockDim); // reset
        ds_0[i] = -dx;
        parameterBlock += ds_0;
        redo_ = true;
        Evaluate(parameters, residuals_m.data(), NULL);
        memcpy(parameters[5 + Start], originalParams, sizeof(double) * blockDim); // reset
        J.col(i) = (residuals_p - residuals_m) * (1.0 / (2 * dx));
      }

      if (jacobiansMinimal != NULL && jacobiansMinimal[5 + Start] != NULL) {
        constexpr size_t minBlockDim = ImuModelT::kXBlockMinDims[Start];
        Eigen::Map<Eigen::Matrix<double, 15, minBlockDim,
                                 Eigen::RowMajor>>
            Jm(jacobiansMinimal[5 + Start]);
        for (size_t i = 0; i < minBlockDim; ++i) {
          Eigen::Matrix<double, minBlockDim, 1> ds_0;
          Eigen::Matrix<double, 15, 1> residuals_p;
          Eigen::Matrix<double, 15, 1> residuals_m;
          ds_0.setZero();
          ds_0[i] = dx;
          ImuModelT::template plus<Start>(parameters[5 + Start], ds_0.data(), parameters[5 + Start]);
          redo_ = true;
          Evaluate(parameters, residuals_p.data(), NULL);
          memcpy(parameters[5 + Start], originalParams, sizeof(double) * blockDim); // reset
          ds_0[i] = -dx;
          ImuModelT::template plus<Start>(parameters[5 + Start], ds_0.data(), parameters[5 + Start]);
          redo_ = true;
          Evaluate(parameters, residuals_m.data(), NULL);
          memcpy(parameters[5 + Start], originalParams, sizeof(double) * blockDim); // reset
          Jm.col(i) = (residuals_p - residuals_m) * (1.0 / (2 * dx));
        }
      }
    }
    fillNumericJacLoop<Start + 1, End>(parameters, jacobians, jacobiansMinimal, imuModel);
  }
}

template <typename ImuModelT>
bool DynamicImuError<ImuModelT>::EvaluateWithMinimalJacobiansNumeric(
    double *const *parameters, double *residuals, double **jacobians,
    double **jacobiansMinimal) const {
  double dx = 1e-6;
  if (jacobiansMinimal) {
    Eigen::Map<Eigen::Matrix<double, 15, 6, Eigen::RowMajor>> J0minNumeric(jacobiansMinimal[0]);
    double T_WS[7];
    memcpy(T_WS, parameters[0], sizeof(double) * 7);
    for (size_t i = 0; i < 6; ++i) {
      Eigen::Matrix<double, 6, 1> dp_0;
      Eigen::Matrix<double, 15, 1> residuals_p;
      Eigen::Matrix<double, 15, 1> residuals_m;
      dp_0.setZero();
      dp_0[i] = dx;
      PoseLocalParameterization::plus(parameters[0], dp_0.data(),
                                      parameters[0]);
      // std::cout<<poseParameterBlock_0.estimate().T()<<std::endl;
      Evaluate(parameters, residuals_p.data(), NULL);
      // std::cout<<residuals_p.transpose()<<std::endl;
      memcpy(parameters[0], T_WS, sizeof(double) * 7); // reset
      dp_0[i] = -dx;
      // std::cout<<residuals.transpose()<<std::endl;
      PoseLocalParameterization::plus(parameters[0], dp_0.data(),
                                      parameters[0]);
      // std::cout<<poseParameterBlock_0.estimate().T()<<std::endl;
      Evaluate(parameters, residuals_m.data(), NULL);
      // std::cout<<residuals_m.transpose()<<std::endl;
      memcpy(parameters[0], T_WS, sizeof(double) * 7); // reset
      J0minNumeric.col(i) = (residuals_p - residuals_m) * (1.0 / (2 * dx));
    }
    if (jacobians) {
      Eigen::Matrix<double, 6, 7, Eigen::RowMajor> J_lift;
      PoseLocalParameterization::liftJacobian(parameters[0], J_lift.data());
      Eigen::Map<Eigen::Matrix<double, 15, 7, Eigen::RowMajor>> J0Numeric(jacobians[0]);
      J0Numeric = J0minNumeric * J_lift;
    }
  }

  if (jacobians) {
    Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> J1_numDiff(jacobians[1]);
    Eigen::Map<Eigen::Matrix<double, 9, 1>> speedAndBiasParameterBlock_0(parameters[1]);
    Eigen::Matrix<double, 9, 1> speedAndBias_0 = speedAndBiasParameterBlock_0;
    for (size_t i = 0; i < 9; ++i) {
      Eigen::Matrix<double, 9, 1> ds_0;
      Eigen::Matrix<double, 15, 1> residuals_p;
      Eigen::Matrix<double, 15, 1> residuals_m;
      ds_0.setZero();
      ds_0[i] = dx;
      Eigen::Matrix<double, 9, 1> plussed = speedAndBias_0 + ds_0;
      speedAndBiasParameterBlock_0 = plussed;
      redo_ = true;
      Evaluate(parameters, residuals_p.data(), NULL);
      ds_0[i] = -dx;
      plussed = speedAndBias_0 + ds_0;
      speedAndBiasParameterBlock_0 = plussed;
      redo_ = true;
      Evaluate(parameters, residuals_m.data(), NULL);
      speedAndBiasParameterBlock_0 = speedAndBias_0; // reset
      J1_numDiff.col(i) = (residuals_p - residuals_m) * (1.0 / (2 * dx));
    }
    if (jacobiansMinimal) {
      Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> J1(jacobiansMinimal[1]);
      J1 = J1_numDiff;
    }
  }

  if (jacobiansMinimal) {
    Eigen::Map<Eigen::Matrix<double, 15, 6, Eigen::RowMajor>> J2minNumeric(jacobiansMinimal[2]);
    double T_WS[7];
    memcpy(T_WS, parameters[2], sizeof(double) * 7);
    for (size_t i = 0; i < 6; ++i) {
      Eigen::Matrix<double, 6, 1> dp_1;
      Eigen::Matrix<double, 15, 1> residuals_p;
      Eigen::Matrix<double, 15, 1> residuals_m;
      dp_1.setZero();
      dp_1[i] = dx;
      PoseLocalParameterization::plus(parameters[2], dp_1.data(),
                                      parameters[2]);
      Evaluate(parameters, residuals_p.data(), NULL);
      memcpy(parameters[2], T_WS, sizeof(double) * 7); // reset
      dp_1[i] = -dx;
      PoseLocalParameterization::plus(parameters[2], dp_1.data(),
                                      parameters[2]);
      Evaluate(parameters, residuals_m.data(), NULL);
      memcpy(parameters[2], T_WS, sizeof(double) * 7); // reset
      J2minNumeric.col(i) = (residuals_p - residuals_m) * (1.0 / (2 * dx));
    }

    if (jacobians) {
      Eigen::Matrix<double, 6, 7, Eigen::RowMajor> J_lift;
      PoseLocalParameterization::liftJacobian(parameters[2], J_lift.data());
      Eigen::Map<Eigen::Matrix<double, 15, 7, Eigen::RowMajor>> J2(jacobians[2]);
      J2 = J2minNumeric * J_lift;
    }
  }

  if (jacobians) {
    Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> J3_numDiff(jacobians[3]);
    Eigen::Map<Eigen::Matrix<double, 9, 1>> speedAndBiasParameterBlock_1(parameters[3]);
    Eigen::Matrix<double, 9, 1> speedAndBias_1 = speedAndBiasParameterBlock_1;
    for (size_t i = 0; i < 9; ++i) {
      Eigen::Matrix<double, 9, 1> ds_1;
      Eigen::Matrix<double, 15, 1> residuals_p;
      Eigen::Matrix<double, 15, 1> residuals_m;
      ds_1.setZero();
      ds_1[i] = dx;
      Eigen::Matrix<double, 9, 1> plussed = speedAndBias_1 + ds_1;
      speedAndBiasParameterBlock_1 = plussed;
      redo_ = true;
      Evaluate(parameters, residuals_p.data(), NULL);
      ds_1[i] = -dx;
      plussed = speedAndBias_1 + ds_1;
      speedAndBiasParameterBlock_1 = plussed;
      redo_ = true;
      Evaluate(parameters, residuals_m.data(), NULL);
      speedAndBiasParameterBlock_1 = speedAndBias_1; // reset
      J3_numDiff.col(i) = (residuals_p - residuals_m) * (1.0 / (2 * dx));
    }
    if (jacobiansMinimal) {
      Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> J3(jacobiansMinimal[3]);
      J3 = J3_numDiff;
    }
  }

  if (jacobiansMinimal) {
    Eigen::Map<Eigen::Matrix<double, 15, 2, Eigen::RowMajor>> J4_minNumDiff(jacobiansMinimal[4]);
    double gravityDirectionBlock[3];
    memcpy(gravityDirectionBlock, parameters[4], sizeof(double) * 3);
    for (size_t i = 0; i < 2; ++i) {
      Eigen::Matrix<double, 2, 1> du;
      Eigen::Matrix<double, 15, 1> residuals_p;
      Eigen::Matrix<double, 15, 1> residuals_m;
      du.setZero();
      du[i] = dx;
      swift_vio::NormalVectorParameterization::plus(parameters[4], du.data(),
                                         parameters[4]);
      Evaluate(parameters, residuals_p.data(), NULL);
      memcpy(parameters[4], gravityDirectionBlock, sizeof(double) * 3); // reset
      du[i] = -dx;
      swift_vio::NormalVectorParameterization::plus(parameters[4], du.data(),
                                         parameters[4]);
      Evaluate(parameters, residuals_m.data(), NULL);
      memcpy(parameters[4], gravityDirectionBlock, sizeof(double) * 3); // reset
      J4_minNumDiff.col(i) = (residuals_p - residuals_m) * (1.0 / (2 * dx));
    }
  }

  if (jacobians) {
    Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> J4_numDiff(jacobians[4]);
    double gravityDirectionBlock[3];
    memcpy(gravityDirectionBlock, parameters[4], sizeof(double) * 3);
    for (size_t i = 0; i < 3; ++i) {
      Eigen::Matrix<double, 3, 1> du;
      Eigen::Matrix<double, 15, 1> residuals_p;
      Eigen::Matrix<double, 15, 1> residuals_m;
      du.setZero();
      du[i] = dx;
      Eigen::Map<Eigen::Vector3d> val(parameters[4]);
      val += du;
      Evaluate(parameters, residuals_p.data(), NULL);
      memcpy(parameters[4], gravityDirectionBlock, sizeof(double) * 3); // reset
      du[i] = -dx;
      val += du;
      Evaluate(parameters, residuals_m.data(), NULL);
      memcpy(parameters[4], gravityDirectionBlock, sizeof(double) * 3); // reset
      J4_numDiff.col(i) = (residuals_p - residuals_m) * (1.0 / (2 * dx));
    }
  }

  fillNumericJacLoop<0, ImuModelT::kXBlockDims.size()>(
      parameters, jacobians, jacobiansMinimal, imuModel_);
  return true;
}

template <typename ImuModelT>
bool DynamicImuError<ImuModelT>::checkJacobians(double *const * parameters) {
  double* jacobians[5 + ImuModelT::kXBlockDims.size()];
  Eigen::Matrix<double,15,7,Eigen::RowMajor> J0;
  Eigen::Matrix<double,15,9,Eigen::RowMajor> J1;
  Eigen::Matrix<double,15,7,Eigen::RowMajor> J2;
  Eigen::Matrix<double,15,9,Eigen::RowMajor> J3;
  Eigen::Matrix<double,15,3,Eigen::RowMajor> J4;
  jacobians[0]=J0.data();
  jacobians[1]=J1.data();
  jacobians[2]=J2.data();
  jacobians[3]=J3.data();
  jacobians[4]=J4.data();
  double* jacobiansMinimal[5 + ImuModelT::kXBlockDims.size()];
  Eigen::Matrix<double,15,6,Eigen::RowMajor> J0min;
  Eigen::Matrix<double,15,9,Eigen::RowMajor> J1min;
  Eigen::Matrix<double,15,6,Eigen::RowMajor> J2min;
  Eigen::Matrix<double,15,9,Eigen::RowMajor> J3min;
  Eigen::Matrix<double,15,2,Eigen::RowMajor> J4min;
  jacobiansMinimal[0]=J0min.data();
  jacobiansMinimal[1]=J1min.data();
  jacobiansMinimal[2]=J2min.data();
  jacobiansMinimal[3]=J3min.data();
  jacobiansMinimal[4]=J4min.data();

  std::vector<std::shared_ptr<Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>> jacPtrs;
  std::vector<std::shared_ptr<Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>> jacMinPtrs;
  for (size_t i = 0u; i < ImuModelT::kXBlockDims.size(); ++i)  {
    std::shared_ptr<Eigen::Matrix<double, -1, -1, Eigen::RowMajor>> jacPtr(
          new Eigen::Matrix<double, -1, -1, Eigen::RowMajor>(15, ImuModelT::kXBlockDims[i]));
    jacobians[i + 5] = jacPtr->data();

    std::shared_ptr<Eigen::Matrix<double, -1, -1, Eigen::RowMajor>> jacMinPtr(
          new Eigen::Matrix<double, -1, -1, Eigen::RowMajor>(15, ImuModelT::kXBlockMinDims[i]));
    jacobiansMinimal[i + 5] = jacMinPtr->data();
    jacPtrs.push_back(jacPtr);
    jacMinPtrs.push_back(jacMinPtr);
  }

  Eigen::Matrix<double,15,1> residuals;

  // evaluate twice to be sure that we will be using the linearisation of the biases (i.e. no preintegrals redone)
  EvaluateWithMinimalJacobians(parameters,residuals.data(),jacobians,jacobiansMinimal);
  EvaluateWithMinimalJacobians(parameters,residuals.data(),jacobians,jacobiansMinimal);

  double* jacobiansNumeric[5 + ImuModelT::kXBlockDims.size()];
  Eigen::Matrix<double,15,7,Eigen::RowMajor> J0Numeric;
  Eigen::Matrix<double,15,9,Eigen::RowMajor> J1Numeric;
  Eigen::Matrix<double,15,7,Eigen::RowMajor> J2Numeric;
  Eigen::Matrix<double,15,9,Eigen::RowMajor> J3Numeric;
  Eigen::Matrix<double,15,3,Eigen::RowMajor> J4Numeric;
  jacobiansNumeric[0]=J0Numeric.data();
  jacobiansNumeric[1]=J1Numeric.data();
  jacobiansNumeric[2]=J2Numeric.data();
  jacobiansNumeric[3]=J3Numeric.data();
  jacobiansNumeric[4]=J4Numeric.data();
  double* jacobiansMinimalNumeric[5 + ImuModelT::kXBlockDims.size()];
  Eigen::Matrix<double,15,6,Eigen::RowMajor> J0minNumeric;
  Eigen::Matrix<double,15,9,Eigen::RowMajor> J1minNumeric;
  Eigen::Matrix<double,15,6,Eigen::RowMajor> J2minNumeric;
  Eigen::Matrix<double,15,9,Eigen::RowMajor> J3minNumeric;
  Eigen::Matrix<double,15,2,Eigen::RowMajor> J4minNumeric;
  jacobiansMinimalNumeric[0]=J0minNumeric.data();
  jacobiansMinimalNumeric[1]=J1minNumeric.data();
  jacobiansMinimalNumeric[2]=J2minNumeric.data();
  jacobiansMinimalNumeric[3]=J3minNumeric.data();
  jacobiansMinimalNumeric[4]=J4minNumeric.data();

  std::vector<std::shared_ptr<Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>> jacNumericPtrs;
  std::vector<std::shared_ptr<Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>> jacMinNumericPtrs;
  for (size_t i = 0u; i < ImuModelT::kXBlockDims.size(); ++i)  {
    std::shared_ptr<Eigen::Matrix<double, -1, -1, Eigen::RowMajor>> jacPtr(
          new Eigen::Matrix<double, -1, -1, Eigen::RowMajor>(15, ImuModelT::kXBlockDims[i]));
    jacobiansNumeric[i + 5] = jacPtr->data();

    std::shared_ptr<Eigen::Matrix<double, -1, -1, Eigen::RowMajor>> jacMinPtr(
          new Eigen::Matrix<double, -1, -1, Eigen::RowMajor>(15, ImuModelT::kXBlockMinDims[i]));
    jacobiansMinimalNumeric[i + 5] = jacMinPtr->data();
    jacNumericPtrs.push_back(jacPtr);
    jacMinNumericPtrs.push_back(jacMinPtr);
  }

  reweight_ = false; // disable weighting update.
  Eigen::Matrix<double,15,1> residualsNumeric;
  EvaluateWithMinimalJacobiansNumeric(parameters, residualsNumeric.data(),
                                      jacobiansNumeric, jacobiansMinimalNumeric);

  constexpr double jacobianTolerance = ImuModelT::kJacobianTolerance;
  OKVIS_ASSERT_TRUE(Exception,(J0min-J0minNumeric).norm()<jacobianTolerance,
                    "minimal Jacobian 0 = \n"<<J0min<<std::endl<<
                    "numDiff minimal Jacobian 0 = \n"<<J0minNumeric);
  OKVIS_ASSERT_TRUE(Exception,(J0-J0Numeric).norm()<jacobianTolerance,
                    "Jacobian 0 = \n"<<J0<<std::endl<<
                    "numDiff Jacobian 0 = \n"<<J0Numeric);
  //std::cout << "minimal Jacobian 0 = \n"<<J0min<<std::endl;
  //std::cout << "numDiff minimal Jacobian 0 = \n"<<J0minNumeric<<std::endl;

  double diffNorm = (J1min - J1minNumeric).lpNorm<Eigen::Infinity>();
  OKVIS_ASSERT_TRUE(Exception, diffNorm < jacobianTolerance,
                    "minimal Jacobian 1 = \n" << J1min << std::endl
                    << "numDiff minimal Jacobian 1 = \n" << J1minNumeric << "\nDiff inf norm " << diffNorm);

// std::cout << "minimal Jacobian 1 = \n"<<J1min<<std::endl;
// std::cout << "numDiff minimal Jacobian 1 =\n"<<J1minNumeric<<std::endl;

  OKVIS_ASSERT_TRUE(Exception,(J2min-J2minNumeric).norm()<jacobianTolerance,
                      "minimal Jacobian 2 = \n"<<J2min<<std::endl<<
                      "numDiff minimal Jacobian 2 = \n"<<J2minNumeric);

  OKVIS_ASSERT_TRUE(Exception,(J2-J2Numeric).norm()<jacobianTolerance,
                      "Jacobian 2 = \n"<<J2<<std::endl<<
                      "numDiff Jacobian 2 = \n"<<J2Numeric);

  diffNorm = (J3min-J3minNumeric).lpNorm<Eigen::Infinity>();
  OKVIS_ASSERT_TRUE(Exception, diffNorm < jacobianTolerance, "minimal Jacobian 1 = \n"<<J3min<<std::endl<<
                    "numDiff minimal Jacobian 1 = \n"<<J3minNumeric << "\nDiff inf norm " << diffNorm);

//  std::cout << "minimal Jacobian 3 = \n"<<J3min<<std::endl;
//  std::cout << "numDiff minimal Jacobian 3 = \n"<<J3minNumeric<<std::endl;

  OKVIS_ASSERT_TRUE(Exception,(J4min-J4minNumeric).norm()<jacobianTolerance,
                      "minimal Jacobian 4 = \n"<<J4min<<std::endl<<
                      "numDiff minimal Jacobian 4 = \n"<<J4minNumeric);
//  std::cout << "minimal Jacobian 4 = \n"<<J4min<<std::endl;
//  std::cout << "numDiff minimal Jacobian 4 = \n"<<J4minNumeric<<std::endl;

  OKVIS_ASSERT_TRUE(Exception,(J4-J4Numeric).norm()<jacobianTolerance,
                      "Jacobian 4 = \n"<<J4<<std::endl<<
                      "numDiff Jacobian 4 = \n"<<J4Numeric);

  //  std::cout << "Jacobian 4 = \n"<<J4<<std::endl;
  //  std::cout << "numDiff Jacobian 4 = \n"<<J4Numeric<<std::endl;
  for (size_t i = 0u; i < ImuModelT::kXBlockDims.size(); ++i) {
    double diffNorm = (*jacPtrs[i] - *jacNumericPtrs[i]).lpNorm<Eigen::Infinity>();
    OKVIS_ASSERT_TRUE(Exception, diffNorm < jacobianTolerance, "XParam " << i << " Jacobian =\n"
                      << *jacPtrs[i] << "\nnumDiff Jacobian =\n"
                      << *jacNumericPtrs[i] << "\nDiff inf norm " << diffNorm);

    diffNorm = (*jacMinPtrs[i] - *jacMinNumericPtrs[i]).lpNorm<Eigen::Infinity>();
    OKVIS_ASSERT_TRUE(Exception, diffNorm < jacobianTolerance,
                      "Minimal XParam " << i << " Jacobian =\n"
                      << *jacMinPtrs[i] << "\nnumDiff Jacobian =\n"
                      << *jacMinNumericPtrs[i] << "\nDiff inf norm " << diffNorm);
  }
  return true;
}

}  // namespace ceres
}  // namespace okvis
