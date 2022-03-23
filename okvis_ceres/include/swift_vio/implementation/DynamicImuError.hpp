#include <swift_vio/memory.h>
#include <swift_vio/ParallaxAnglePoint.hpp>

//#include <chrono>
#include <thread>

#include <glog/logging.h>

#include <okvis/kinematics/operators.hpp>
#include <okvis/Parameters.hpp>
#include <swift_vio/ExtrinsicModels.hpp>
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
int DynamicImuError<ImuModelT>::redoPreintegration(const Eigen::Matrix<double, 6, 1> &biases) const {
//  auto start = std::chrono::high_resolution_clock::now();
  // ensure unique access
  std::lock_guard<std::mutex> lock(preintegrationMutex_);

  // now the propagation
  okvis::Time time = t0_;
  okvis::Time end = t1_;

  // sanity check:
  if (imuMeasurements_.front().timeStamp > time || imuMeasurements_.back().timeStamp < end) {
    LOG(WARNING) << "IMU measurements time interval ["
              << imuMeasurements_.front().timeStamp << ", "
              << imuMeasurements_.back().timeStamp
              << "] does not cover integration interval [" << t0_ << ", " << t1_
              << "].";
  }

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

    imuModel_.propagate(dt, omega_S_0, acc_S_0, omega_S_1, acc_S_1,
                        sigma_g_c, sigma_a_c, imuParameters_.sigma_gw_c,
                        imuParameters_.sigma_aw_c);

    time = nexttime;

    ++i;

    if (nexttime == t1_)
      break;

  }

  // store the reference (linearisation) point
  biases_ref_ = biases;

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
    if (jacobians != NULL && jacobians[Index::extra + Start] != NULL) {
      Eigen::Map<Eigen::Matrix<double, 15, ImuModelT::kXBlockDims[Start],
                               Eigen::RowMajor>>
          J(jacobians[Index::extra + Start]);
      J.template block<3, ImuModelT::kXBlockDims[Start]>(0, 0) =
          imuModel.template dDp_dx<Start>();
      J.template block<3, ImuModelT::kXBlockDims[Start]>(3, 0) =
          derot_dDrot * imuModel.template dDrot_dx<Start>();
      J.template block<3, ImuModelT::kXBlockDims[Start]>(6, 0) =
          imuModel.template dDv_dx<Start>();
      J.template bottomRows<6>().setZero();
      J = squareRootInformation_ * J;
      if (jacobiansMinimal != NULL && jacobiansMinimal[Index::extra + Start] != NULL) {
        Eigen::Map<Eigen::Matrix<double, 15, ImuModelT::kXBlockDims[Start],
                                 Eigen::RowMajor>>
            Jm(jacobiansMinimal[Index::extra + Start]);
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
      Eigen::Vector3d(parameters[Index::T_WB1][0], parameters[Index::T_WB1][1], parameters[Index::T_WB1][2]),
      Eigen::Quaterniond(parameters[Index::T_WB1][6], parameters[Index::T_WB1][3], parameters[Index::T_WB1][4], parameters[Index::T_WB1][5]));

  // get speed and bias
  Eigen::Vector3d v_WS0 = Eigen::Map<const Eigen::Vector3d>(parameters[Index::v_WB0]);
  Eigen::Matrix<double, 6, 1> biases0 = Eigen::Map<const Eigen::Matrix<double, 6, 1>>(parameters[Index::bgBa0]);
  Eigen::Vector3d v_WS1 = Eigen::Map<const Eigen::Vector3d>(parameters[Index::v_WB1]);
  Eigen::Matrix<double, 6, 1> biases1 = Eigen::Map<const Eigen::Matrix<double, 6, 1>>(parameters[Index::bgBa1]);

  imuModel_.updateParameters(parameters[Index::bgBa0], parameters + Index::extra);

  // this will NOT be changed:
  const Eigen::Matrix3d C_WS_0 = T_WS_0.C();
  const Eigen::Matrix3d C_S0_W = C_WS_0.transpose();

  // call the propagation
  const double Delta_t = (t1_ - t0_).toSec();
  Eigen::Matrix<double, 6, 1> Delta_b;
  // ensure unique access
  {
    std::lock_guard<std::mutex> lock(preintegrationMutex_);
    Delta_b = biases0 - biases_ref_;
  }
  redo_ = redo_ || (Delta_b.head<3>().norm() * Delta_t > 0.0001);
  if (redo_) {
    redoPreintegration(biases0);
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

    Eigen::Map<const Eigen::Vector3d> gravityDirection(parameters[Index::unitgW]);
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
        T_WS_0.r() - T_WS_1.r() + v_WS0*Delta_t + 0.5*g_W*Delta_t*Delta_t;
    const Eigen::Vector3d delta_v_est_W =
        v_WS0 - v_WS1 + g_W*Delta_t;
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
    error.tail<6>() = biases0 - biases1;

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

        // hallucinate Jacobian w.r.t. state
        Eigen::Map<Eigen::Matrix<double, 15, 7, Eigen::RowMajor> > J0(
            jacobians[0]);
        J0.leftCols<6>() = J0_minimal;
        J0.col(6).setZero();

        // if requested, provide minimal Jacobians
        if (jacobiansMinimal != NULL) {
          if (jacobiansMinimal[0] != NULL) {
            Eigen::Map<Eigen::Matrix<double, 15, 6, Eigen::RowMajor> > J0_minimal_mapped(
                jacobiansMinimal[0]);
            J0_minimal_mapped = J0_minimal;
          }
        }
      }
      if (jacobians[Index::v_WB0] != NULL) {
        Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor> > J(
            jacobians[Index::v_WB0]);
        J = squareRootInformation_ * F0.block<15, 3>(0, 6);

        // if requested, provide minimal Jacobians
        if (jacobiansMinimal != NULL) {
          if (jacobiansMinimal[Index::v_WB0] != NULL) {
            Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor> > J_minimal_mapped(
                jacobiansMinimal[Index::v_WB0]);
            J_minimal_mapped = J;
          }
        }
      }

      if (jacobians[Index::bgBa0] != NULL) {
        Eigen::Map<Eigen::Matrix<double, 15, 6, Eigen::RowMajor> > J(
            jacobians[Index::bgBa0]);
        J = squareRootInformation_ * F0.block<15, 6>(0, 9);

        // if requested, provide minimal Jacobians
        if (jacobiansMinimal != NULL) {
          if (jacobiansMinimal[Index::bgBa0] != NULL) {
            Eigen::Map<Eigen::Matrix<double, 15, 6, Eigen::RowMajor> > J_minimal_mapped(
                jacobiansMinimal[Index::bgBa0]);
            J_minimal_mapped = J;
          }
        }
      }
      if (jacobians[Index::T_WB1] != NULL) {
        Eigen::Matrix<double, 15, 6> J_minimal = squareRootInformation_
                    * F1.block<15, 6>(0, 0);
        Eigen::Map<Eigen::Matrix<double, 15, 7, Eigen::RowMajor> > J(
            jacobians[Index::T_WB1]);
        J.leftCols<6>() = J_minimal;
        J.col(6).setZero();

        if (jacobiansMinimal != NULL) {
          if (jacobiansMinimal[Index::T_WB1] != NULL) {
            Eigen::Map<Eigen::Matrix<double, 15, 6, Eigen::RowMajor> > J_minimal_mapped(
                jacobiansMinimal[Index::T_WB1]);
            J_minimal_mapped = J_minimal;
          }
        }
      }
      if (jacobians[Index::v_WB1] != NULL) {
        Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor> > J(jacobians[Index::v_WB1]);
        J = squareRootInformation_ * F1.block<15, 3>(0, 6);

        // if requested, provide minimal Jacobians
        if (jacobiansMinimal != NULL) {
          if (jacobiansMinimal[Index::v_WB1] != NULL) {
            Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor> > J_minimal_mapped(
                jacobiansMinimal[Index::v_WB1]);
            J_minimal_mapped = J;
          }
        }
      }
      if (jacobians[Index::bgBa1] != NULL) {
        Eigen::Map<Eigen::Matrix<double, 15, 6, Eigen::RowMajor> > J(jacobians[Index::bgBa1]);
        J = squareRootInformation_ * F1.block<15, 6>(0, 9);

        // if requested, provide minimal Jacobians
        if (jacobiansMinimal != NULL) {
          if (jacobiansMinimal[Index::bgBa1] != NULL) {
            Eigen::Map<Eigen::Matrix<double, 15, 6, Eigen::RowMajor> > J_minimal_mapped(
                jacobiansMinimal[Index::bgBa1]);
            J_minimal_mapped = J;
          }
        }
      }
      if (jacobians[Index::unitgW] != NULL) {
        Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> J(
            jacobians[Index::unitgW]);
        Eigen::Matrix<double, 15, 3> de_dunitgW =
            Eigen::Matrix<double, 15, 3>::Zero();
        de_dunitgW.topLeftCorner<3, 3>() = 0.5 * Delta_t * Delta_t *
                                           imuParameters_.g *
                                           C_S0_W;
        de_dunitgW.block<3, 3>(6, 0) = Delta_t * imuParameters_.g * C_S0_W;
        J = squareRootInformation_ * de_dunitgW;

        // if requested, provide minimal Jacobians
        if (jacobiansMinimal != NULL) {
          if (jacobiansMinimal[Index::unitgW] != NULL) {
            Eigen::Map<Eigen::Matrix<double, 15, 2, Eigen::RowMajor>>
                J_minimal_mapped(jacobiansMinimal[Index::unitgW]);
            Eigen::Matrix<double, 3, 2, Eigen::RowMajor> dunitgW_du;
            swift_vio::NormalVectorParameterization::plusJacobian(
                gravityDirection.data(), dunitgW_du.data());
            J_minimal_mapped = J * dunitgW_du;
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
    if (jacobians != NULL && jacobians[Index::extra + Start] != NULL) {
      constexpr size_t blockDim = ImuModelT::kXBlockDims[Start];
      Eigen::Map<Eigen::Matrix<double, 15, blockDim,
                               Eigen::RowMajor>>
          J(jacobians[Index::extra + Start]);
      double originalParams[ImuModelT::kXBlockDims[Start]];
      memcpy(originalParams, parameters[Index::extra + Start], sizeof(double) * blockDim);
      Eigen::Map<Eigen::Matrix<double, blockDim, 1>> parameterBlock(parameters[Index::extra + Start]);
      for (size_t i = 0; i <blockDim; ++i) {
        Eigen::Matrix<double, blockDim, 1> ds_0;
        Eigen::Matrix<double, 15, 1> residuals_p;
        Eigen::Matrix<double, 15, 1> residuals_m;
        ds_0.setZero();
        ds_0[i] = dx;
        parameterBlock += ds_0;
        redo_ = true;
        Evaluate(parameters, residuals_p.data(), NULL);
        memcpy(parameters[Index::extra + Start], originalParams, sizeof(double) * blockDim); // reset
        ds_0[i] = -dx;
        parameterBlock += ds_0;
        redo_ = true;
        Evaluate(parameters, residuals_m.data(), NULL);
        memcpy(parameters[Index::extra + Start], originalParams, sizeof(double) * blockDim); // reset
        J.col(i) = (residuals_p - residuals_m) * (1.0 / (2 * dx));
      }

      if (jacobiansMinimal != NULL && jacobiansMinimal[Index::extra + Start] != NULL) {
        constexpr size_t minBlockDim = ImuModelT::kXBlockMinDims[Start];
        Eigen::Map<Eigen::Matrix<double, 15, minBlockDim,
                                 Eigen::RowMajor>>
            Jm(jacobiansMinimal[Index::extra + Start]);
        for (size_t i = 0; i < minBlockDim; ++i) {
          Eigen::Matrix<double, minBlockDim, 1> ds_0;
          Eigen::Matrix<double, 15, 1> residuals_p;
          Eigen::Matrix<double, 15, 1> residuals_m;
          ds_0.setZero();
          ds_0[i] = dx;
          ImuModelT::template plus<Start>(parameters[Index::extra + Start], ds_0.data(), parameters[Index::extra + Start]);
          redo_ = true;
          Evaluate(parameters, residuals_p.data(), NULL);
          memcpy(parameters[Index::extra + Start], originalParams, sizeof(double) * blockDim); // reset
          ds_0[i] = -dx;
          ImuModelT::template plus<Start>(parameters[Index::extra + Start], ds_0.data(), parameters[Index::extra + Start]);
          redo_ = true;
          Evaluate(parameters, residuals_m.data(), NULL);
          memcpy(parameters[Index::extra + Start], originalParams, sizeof(double) * blockDim); // reset
          Jm.col(i) = (residuals_p - residuals_m) * (1.0 / (2 * dx));
        }
      }
    }
    fillNumericJacLoop<Start + 1, End>(parameters, jacobians, jacobiansMinimal, imuModel);
  }
}

template <typename ImuModelT>
bool DynamicImuError<ImuModelT>::EvaluateWithMinimalJacobiansNumeric(
    double *const *parameters, double */*residuals*/, double **jacobians,
    double **jacobiansMinimal) const {
  double dx = 1e-6;
  if (jacobiansMinimal) {
    Eigen::Map<Eigen::Matrix<double, 15, 6, Eigen::RowMajor>> Jp0minNumeric(jacobiansMinimal[0]);
    double T_WS[7];
    memcpy(T_WS, parameters[0], sizeof(double) * 7);
    for (size_t i = 0; i < 6; ++i) {
      Eigen::Matrix<double, 6, 1> dp_0;
      Eigen::Matrix<double, 15, 1> residuals_p;
      Eigen::Matrix<double, 15, 1> residuals_m;
      dp_0.setZero();
      dp_0[i] = dx;
      swift_vio::PoseLocalParameterizationSimplified::oplus(parameters[0], dp_0.data(),
                                      parameters[0]);
      // std::cout<<poseParameterBlock_0.estimate().T()<<std::endl;
      Evaluate(parameters, residuals_p.data(), NULL);
      // std::cout<<residuals_p.transpose()<<std::endl;
      memcpy(parameters[0], T_WS, sizeof(double) * 7); // reset
      dp_0[i] = -dx;
      // std::cout<<residuals.transpose()<<std::endl;
      swift_vio::PoseLocalParameterizationSimplified::oplus(parameters[0], dp_0.data(),
                                      parameters[0]);
      // std::cout<<poseParameterBlock_0.estimate().T()<<std::endl;
      Evaluate(parameters, residuals_m.data(), NULL);
      // std::cout<<residuals_m.transpose()<<std::endl;
      memcpy(parameters[0], T_WS, sizeof(double) * 7); // reset
      Jp0minNumeric.col(i) = (residuals_p - residuals_m) * (1.0 / (2 * dx));
    }
    if (jacobians) {
      Eigen::Map<Eigen::Matrix<double, 15, 7, Eigen::RowMajor>> Jp0Numeric(jacobians[0]);
      Jp0Numeric.leftCols<6>() = Jp0minNumeric;
      Jp0Numeric.col(6).setZero();
    }
  }

  if (jacobians) {
    Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> J1_numDiff(
        jacobians[Index::v_WB0]);
    Eigen::Map<Eigen::Matrix<double, 3, 1>> speedParameterBlock_0(
        parameters[Index::v_WB0]);
    Eigen::Matrix<double, 3, 1> speed0 = speedParameterBlock_0;

    for (size_t i = 0; i < 3; ++i) {
      Eigen::Matrix<double, 3, 1> ds_0;
      Eigen::Matrix<double, 15, 1> residuals_p;
      Eigen::Matrix<double, 15, 1> residuals_m;
      ds_0.setZero();
      ds_0[i] = dx;
      Eigen::Matrix<double, 3, 1> plussed = speed0 + ds_0;
      speedParameterBlock_0 = plussed;
      redo_ = true;
      Evaluate(parameters, residuals_p.data(), NULL);
      ds_0[i] = -dx;
      plussed = speed0 + ds_0;
      speedParameterBlock_0 = plussed;
      redo_ = true;
      Evaluate(parameters, residuals_m.data(), NULL);
      speedParameterBlock_0 = speed0; // reset
      J1_numDiff.col(i) = (residuals_p - residuals_m) * (1.0 / (2 * dx));
    }
    if (jacobiansMinimal) {
      Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> J1(
          jacobiansMinimal[Index::v_WB0]);
      J1 = J1_numDiff;
    }
  }

  if (jacobians) {
    Eigen::Map<Eigen::Matrix<double, 15, 6, Eigen::RowMajor>> J2_numDiff(
        jacobians[Index::bgBa0]);
    Eigen::Map<Eigen::Matrix<double, 6, 1>> biasParameterBlock_0(
        parameters[Index::bgBa0]);
    Eigen::Matrix<double, 6, 1> bias0 = biasParameterBlock_0;
    for (size_t i = 0; i < 6; ++i) {
      Eigen::Matrix<double, 6, 1> ds_0;
      Eigen::Matrix<double, 15, 1> residuals_p;
      Eigen::Matrix<double, 15, 1> residuals_m;
      ds_0.setZero();
      ds_0[i] = dx;
      Eigen::Matrix<double, 6, 1> plussed = bias0 + ds_0;
      biasParameterBlock_0 = plussed;
      redo_ = true;
      Evaluate(parameters, residuals_p.data(), NULL);
      ds_0[i] = -dx;
      plussed = bias0 + ds_0;
      biasParameterBlock_0 = plussed;
      redo_ = true;
      Evaluate(parameters, residuals_m.data(), NULL);
      biasParameterBlock_0 = bias0; // reset
      J2_numDiff.col(i) = (residuals_p - residuals_m) * (1.0 / (2 * dx));
    }
    if (jacobiansMinimal) {
      Eigen::Map<Eigen::Matrix<double, 15, 6, Eigen::RowMajor>> J2(
          jacobiansMinimal[Index::bgBa0]);
      J2 = J2_numDiff;
    }
  }

  if (jacobiansMinimal) {
    Eigen::Map<Eigen::Matrix<double, 15, 6, Eigen::RowMajor>> JminNumeric(jacobiansMinimal[Index::T_WB1]);
    double T_WS[7];
    memcpy(T_WS, parameters[Index::T_WB1], sizeof(double) * 7);
    for (size_t i = 0; i < 6; ++i) {
      Eigen::Matrix<double, 6, 1> dp_1;
      Eigen::Matrix<double, 15, 1> residuals_p;
      Eigen::Matrix<double, 15, 1> residuals_m;
      dp_1.setZero();
      dp_1[i] = dx;
      swift_vio::PoseLocalParameterizationSimplified::oplus(parameters[Index::T_WB1], dp_1.data(),
                                      parameters[Index::T_WB1]);
      Evaluate(parameters, residuals_p.data(), NULL);
      memcpy(parameters[Index::T_WB1], T_WS, sizeof(double) * 7); // reset
      dp_1[i] = -dx;
      swift_vio::PoseLocalParameterizationSimplified::oplus(parameters[Index::T_WB1], dp_1.data(),
                                      parameters[Index::T_WB1]);
      Evaluate(parameters, residuals_m.data(), NULL);
      memcpy(parameters[Index::T_WB1], T_WS, sizeof(double) * 7); // reset
      JminNumeric.col(i) = (residuals_p - residuals_m) * (1.0 / (2 * dx));
    }

    if (jacobians) {
      Eigen::Map<Eigen::Matrix<double, 15, 7, Eigen::RowMajor>> J(jacobians[Index::T_WB1]);
      J.leftCols<6>() = JminNumeric;
      J.col(6).setZero();
    }
  }

  if (jacobians) {
    Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> Jv_numDiff(jacobians[Index::v_WB1]);
    Eigen::Map<Eigen::Matrix<double, 3, 1>> speedParameterBlock_1(parameters[Index::v_WB1]);
    Eigen::Matrix<double, 3, 1> speed1 = speedParameterBlock_1;
    for (size_t i = 0; i < 3; ++i) {
      Eigen::Matrix<double, 3, 1> ds_1;
      Eigen::Matrix<double, 15, 1> residuals_p;
      Eigen::Matrix<double, 15, 1> residuals_m;
      ds_1.setZero();
      ds_1[i] = dx;
      Eigen::Matrix<double, 3, 1> plussed = speed1 + ds_1;
      speedParameterBlock_1 = plussed;
      redo_ = true;
      Evaluate(parameters, residuals_p.data(), NULL);
      ds_1[i] = -dx;
      plussed = speed1 + ds_1;
      speedParameterBlock_1 = plussed;
      redo_ = true;
      Evaluate(parameters, residuals_m.data(), NULL);
      speedParameterBlock_1 = speed1; // reset
      Jv_numDiff.col(i) = (residuals_p - residuals_m) * (1.0 / (2 * dx));
    }
    if (jacobiansMinimal) {
      Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> Jv(jacobiansMinimal[Index::v_WB1]);
      Jv = Jv_numDiff;
    }
  }

  if (jacobians) {
    Eigen::Map<Eigen::Matrix<double, 15, 6, Eigen::RowMajor>> Jb_numDiff(jacobians[Index::bgBa1]);
    Eigen::Map<Eigen::Matrix<double, 6, 1>> biasParameterBlock_1(parameters[Index::bgBa1]);
        Eigen::Matrix<double, 6, 1> bias1 = biasParameterBlock_1;
    for (size_t i = 0; i < 6; ++i) {
      Eigen::Matrix<double, 6, 1> ds_1;
      Eigen::Matrix<double, 15, 1> residuals_p;
      Eigen::Matrix<double, 15, 1> residuals_m;
      ds_1.setZero();
      ds_1[i] = dx;
      Eigen::Matrix<double, 6, 1> plussed = bias1 + ds_1;
      biasParameterBlock_1 = plussed;
      redo_ = true;
      Evaluate(parameters, residuals_p.data(), NULL);
      ds_1[i] = -dx;
      plussed = bias1 + ds_1;
      biasParameterBlock_1 = plussed;
      redo_ = true;
      Evaluate(parameters, residuals_m.data(), NULL);
      biasParameterBlock_1 = bias1; // reset
      Jb_numDiff.col(i) = (residuals_p - residuals_m) * (1.0 / (2 * dx));
    }

    if (jacobiansMinimal) {
      Eigen::Map<Eigen::Matrix<double, 15, 6, Eigen::RowMajor>> Jb(jacobiansMinimal[Index::bgBa1]);
      Jb = Jb_numDiff;
    }
  }

  if (jacobiansMinimal) {
    Eigen::Map<Eigen::Matrix<double, 15, 2, Eigen::RowMajor>> J_minNumDiff(jacobiansMinimal[Index::unitgW]);
    double gravityDirectionBlock[3];
    memcpy(gravityDirectionBlock, parameters[Index::unitgW], sizeof(double) * 3);
    for (size_t i = 0; i < 2; ++i) {
      Eigen::Matrix<double, 2, 1> du;
      Eigen::Matrix<double, 15, 1> residuals_p;
      Eigen::Matrix<double, 15, 1> residuals_m;
      du.setZero();
      du[i] = dx;
      swift_vio::NormalVectorParameterization::plus(parameters[Index::unitgW], du.data(),
                                         parameters[Index::unitgW]);
      Evaluate(parameters, residuals_p.data(), NULL);
      memcpy(parameters[Index::unitgW], gravityDirectionBlock, sizeof(double) * 3); // reset
      du[i] = -dx;
      swift_vio::NormalVectorParameterization::plus(parameters[Index::unitgW], du.data(),
                                         parameters[Index::unitgW]);
      Evaluate(parameters, residuals_m.data(), NULL);
      memcpy(parameters[Index::unitgW], gravityDirectionBlock, sizeof(double) * 3); // reset
      J_minNumDiff.col(i) = (residuals_p - residuals_m) * (1.0 / (2 * dx));
    }
  }

  if (jacobians) {
    Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> J_numDiff(jacobians[Index::unitgW]);
    double gravityDirectionBlock[3];
    memcpy(gravityDirectionBlock, parameters[Index::unitgW], sizeof(double) * 3);
    for (size_t i = 0; i < 3; ++i) {
      Eigen::Matrix<double, 3, 1> du;
      Eigen::Matrix<double, 15, 1> residuals_p;
      Eigen::Matrix<double, 15, 1> residuals_m;
      du.setZero();
      du[i] = dx;
      Eigen::Map<Eigen::Vector3d> val(parameters[Index::unitgW]);
      val += du;
      Evaluate(parameters, residuals_p.data(), NULL);
      memcpy(parameters[Index::unitgW], gravityDirectionBlock, sizeof(double) * 3); // reset
      du[i] = -dx;
      val += du;
      Evaluate(parameters, residuals_m.data(), NULL);
      memcpy(parameters[Index::unitgW], gravityDirectionBlock, sizeof(double) * 3); // reset
      J_numDiff.col(i) = (residuals_p - residuals_m) * (1.0 / (2 * dx));
    }
  }

  fillNumericJacLoop<0, ImuModelT::kXBlockDims.size()>(
      parameters, jacobians, jacobiansMinimal, imuModel_);
  return true;
}

template <typename ImuModelT>
bool DynamicImuError<ImuModelT>::checkJacobians(double *const * parameters) {
  double* jacobians[Index::extra + ImuModelT::kXBlockDims.size()];
  Eigen::Matrix<double,15,7,Eigen::RowMajor> Jp0;
  Eigen::Matrix<double,15,3,Eigen::RowMajor> Jv0;
  Eigen::Matrix<double,15,6,Eigen::RowMajor> Jb0;
  Eigen::Matrix<double,15,7,Eigen::RowMajor> Jp1;
  Eigen::Matrix<double,15,3,Eigen::RowMajor> Jv1;
  Eigen::Matrix<double,15,6,Eigen::RowMajor> Jb1;
  Eigen::Matrix<double,15,3,Eigen::RowMajor> Jg;
  jacobians[0]=Jp0.data();
  jacobians[1]=Jv0.data();
  jacobians[2]=Jb0.data();
  jacobians[3]=Jp1.data();
  jacobians[4]=Jv1.data();
  jacobians[5]=Jb1.data();
  jacobians[6]=Jg.data();

  double* jacobiansMinimal[Index::extra + ImuModelT::kXBlockDims.size()];
  Eigen::Matrix<double,15,6,Eigen::RowMajor> Jp0min;
  Eigen::Matrix<double,15,3,Eigen::RowMajor> Jv0min;
  Eigen::Matrix<double,15,6,Eigen::RowMajor> Jb0min;

  Eigen::Matrix<double,15,6,Eigen::RowMajor> Jp1min;
  Eigen::Matrix<double,15,3,Eigen::RowMajor> Jv1min;
  Eigen::Matrix<double,15,6,Eigen::RowMajor> Jb1min;
  Eigen::Matrix<double,15,2,Eigen::RowMajor> Jgmin;
  jacobiansMinimal[0]=Jp0min.data();
  jacobiansMinimal[1]=Jv0min.data();
  jacobiansMinimal[2]=Jb0min.data();
  jacobiansMinimal[3]=Jp1min.data();
  jacobiansMinimal[4]=Jv1min.data();
  jacobiansMinimal[5]=Jb1min.data();
  jacobiansMinimal[6]=Jgmin.data();

  std::vector<std::shared_ptr<Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>> jacPtrs;
  std::vector<std::shared_ptr<Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>> jacMinPtrs;
  for (size_t i = 0u; i < ImuModelT::kXBlockDims.size(); ++i)  {
    std::shared_ptr<Eigen::Matrix<double, -1, -1, Eigen::RowMajor>> jacPtr(
          new Eigen::Matrix<double, -1, -1, Eigen::RowMajor>(15, ImuModelT::kXBlockDims[i]));
    jacobians[i + Index::extra] = jacPtr->data();

    std::shared_ptr<Eigen::Matrix<double, -1, -1, Eigen::RowMajor>> jacMinPtr(
          new Eigen::Matrix<double, -1, -1, Eigen::RowMajor>(15, ImuModelT::kXBlockMinDims[i]));
    jacobiansMinimal[i + Index::extra] = jacMinPtr->data();
    jacPtrs.push_back(jacPtr);
    jacMinPtrs.push_back(jacMinPtr);
  }

  Eigen::Matrix<double,15,1> residuals;

  // evaluate twice to be sure that we will be using the linearisation of the biases (i.e. no preintegrals redone)
  EvaluateWithMinimalJacobians(parameters,residuals.data(),jacobians,jacobiansMinimal);
  EvaluateWithMinimalJacobians(parameters,residuals.data(),jacobians,jacobiansMinimal);

  double* jacobiansNumeric[Index::extra + ImuModelT::kXBlockDims.size()];
  Eigen::Matrix<double,15,7,Eigen::RowMajor> Jp0Numeric;
  Eigen::Matrix<double,15,3,Eigen::RowMajor> Jv0Numeric;
  Eigen::Matrix<double,15,6,Eigen::RowMajor> Jb0Numeric;
  Eigen::Matrix<double,15,7,Eigen::RowMajor> Jp1Numeric;
  Eigen::Matrix<double,15,3,Eigen::RowMajor> Jv1Numeric;
  Eigen::Matrix<double,15,6,Eigen::RowMajor> Jb1Numeric;
  Eigen::Matrix<double,15,3,Eigen::RowMajor> JgNumeric;
  jacobiansNumeric[0]=Jp0Numeric.data();
  jacobiansNumeric[1]=Jv0Numeric.data();
  jacobiansNumeric[2]=Jb0Numeric.data();
  jacobiansNumeric[3]=Jp1Numeric.data();
  jacobiansNumeric[4]=Jv1Numeric.data();
  jacobiansNumeric[5]=Jb1Numeric.data();
  jacobiansNumeric[6]=JgNumeric.data();

  double* jacobiansMinimalNumeric[Index::extra + ImuModelT::kXBlockDims.size()];
  Eigen::Matrix<double,15,6,Eigen::RowMajor> Jp0minNumeric;
  Eigen::Matrix<double,15,3,Eigen::RowMajor> Jv0minNumeric;
  Eigen::Matrix<double,15,6,Eigen::RowMajor> Jb0minNumeric;

  Eigen::Matrix<double,15,6,Eigen::RowMajor> Jp1minNumeric;
  Eigen::Matrix<double,15,3,Eigen::RowMajor> Jv1minNumeric;
  Eigen::Matrix<double,15,6,Eigen::RowMajor> Jb1minNumeric;

  Eigen::Matrix<double,15,2,Eigen::RowMajor> JgminNumeric;
  jacobiansMinimalNumeric[0]=Jp0minNumeric.data();
  jacobiansMinimalNumeric[1]=Jv0minNumeric.data();
  jacobiansMinimalNumeric[2]=Jb0minNumeric.data();
  jacobiansMinimalNumeric[3]=Jp1minNumeric.data();
  jacobiansMinimalNumeric[4]=Jv1minNumeric.data();
  jacobiansMinimalNumeric[5]=Jb1minNumeric.data();
  jacobiansMinimalNumeric[6]=JgminNumeric.data();

  std::vector<std::shared_ptr<Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>> jacNumericPtrs;
  std::vector<std::shared_ptr<Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>> jacMinNumericPtrs;
  for (size_t i = 0u; i < ImuModelT::kXBlockDims.size(); ++i) {
    std::shared_ptr<Eigen::Matrix<double, -1, -1, Eigen::RowMajor>> jacPtr(
          new Eigen::Matrix<double, -1, -1, Eigen::RowMajor>(15, ImuModelT::kXBlockDims[i]));
    jacobiansNumeric[i + Index::extra] = jacPtr->data();

    std::shared_ptr<Eigen::Matrix<double, -1, -1, Eigen::RowMajor>> jacMinPtr(
          new Eigen::Matrix<double, -1, -1, Eigen::RowMajor>(15, ImuModelT::kXBlockMinDims[i]));
    jacobiansMinimalNumeric[i + Index::extra] = jacMinPtr->data();
    jacNumericPtrs.push_back(jacPtr);
    jacMinNumericPtrs.push_back(jacMinPtr);
  }

  reweight_ = false; // disable weighting update.
  Eigen::Matrix<double,15,1> residualsNumeric;
  EvaluateWithMinimalJacobiansNumeric(parameters, residualsNumeric.data(),
                                      jacobiansNumeric, jacobiansMinimalNumeric);

  constexpr double jacobianTolerance = ImuModelT::kJacobianTolerance;
  OKVIS_ASSERT_TRUE(Exception,(Jp0min-Jp0minNumeric).norm()<jacobianTolerance,
                    "minimal Jacobian 0 = \n"<<Jp0min<<std::endl<<
                    "numDiff minimal Jacobian 0 = \n"<<Jp0minNumeric);
  OKVIS_ASSERT_TRUE(Exception,(Jp0-Jp0Numeric).norm()<jacobianTolerance,
                    "Jacobian 0 = \n"<<Jp0<<std::endl<<
                    "numDiff Jacobian 0 = \n"<<Jp0Numeric);
  //std::cout << "minimal Jacobian 0 = \n"<<Jp0min<<std::endl;
  //std::cout << "numDiff minimal Jacobian 0 = \n"<<Jp0minNumeric<<std::endl;

  double diffNorm = (Jv0min - Jv0minNumeric).lpNorm<Eigen::Infinity>();
  OKVIS_ASSERT_TRUE(Exception, diffNorm < jacobianTolerance,
                    "minimal Jacobian v0 = \n" << Jv0min << std::endl
                    << "numDiff minimal Jacobian v0 = \n" << Jv0minNumeric << "\nDiff inf norm " << diffNorm);

// std::cout << "minimal Jacobian v0 = \n"<<Jv0min<<std::endl;
// std::cout << "numDiff minimal Jacobian v0 =\n"<<Jv0minNumeric<<std::endl;

  diffNorm = (Jb0min - Jb0minNumeric).lpNorm<Eigen::Infinity>();
  OKVIS_ASSERT_TRUE(Exception, diffNorm < jacobianTolerance,
                    "minimal Jacobian bias0 = \n" << Jb0min << std::endl
                    << "numDiff minimal Jacobian bias0 = \n" << Jb0minNumeric << "\nDiff inf norm " << diffNorm);

  OKVIS_ASSERT_TRUE(Exception,(Jp1min-Jp1minNumeric).norm()<jacobianTolerance,
                      "minimal Jacobian p1 = \n"<<Jp1min<<std::endl<<
                      "numDiff minimal Jacobian p1 = \n"<<Jp1minNumeric);

  OKVIS_ASSERT_TRUE(Exception,(Jp1-Jp1Numeric).norm()<jacobianTolerance,
                      "Jacobian p1 = \n"<<Jp1<<std::endl<<
                      "numDiff Jacobian p1 = \n"<<Jp1Numeric);

  diffNorm = (Jv1min-Jv1minNumeric).lpNorm<Eigen::Infinity>();
  OKVIS_ASSERT_TRUE(Exception, diffNorm < jacobianTolerance, "minimal Jacobian v1 = \n"<<Jv1min<<std::endl<<
                    "numDiff minimal Jacobian v1 = \n"<<Jv1minNumeric << "\nDiff inf norm " << diffNorm);

//  std::cout << "minimal Jacobian v1 = \n"<<Jv1min<<std::endl;
//  std::cout << "numDiff minimal Jacobian v1 = \n"<<Jv1minNumeric<<std::endl;

  diffNorm = (Jb1min-Jb1minNumeric).lpNorm<Eigen::Infinity>();
  OKVIS_ASSERT_TRUE(Exception, diffNorm < jacobianTolerance, "minimal Jacobian b1 = \n"<<Jb1min<<std::endl<<
                    "numDiff minimal Jacobian b1 = \n"<<Jb1minNumeric << "\nDiff inf norm " << diffNorm);

  OKVIS_ASSERT_TRUE(Exception,(Jgmin-JgminNumeric).norm()<jacobianTolerance,
                      "minimal Jacobian g = \n"<<Jgmin<<std::endl<<
                      "numDiff minimal Jacobian g = \n"<<JgminNumeric);
//  std::cout << "minimal Jacobian 4 = \n"<<Jgmin<<std::endl;
//  std::cout << "numDiff minimal Jacobian 4 = \n"<<JgminNumeric<<std::endl;

  OKVIS_ASSERT_TRUE(Exception,(Jg-JgNumeric).norm()<jacobianTolerance,
                      "Jacobian g = \n"<<Jg<<std::endl<<
                      "numDiff Jacobian g = \n"<<JgNumeric);

  //  std::cout << "Jacobian 4 = \n"<<Jg<<std::endl;
  //  std::cout << "numDiff Jacobian 4 = \n"<<JgNumeric<<std::endl;
  for (size_t i = 0u; i < ImuModelT::kXBlockDims.size(); ++i) {
    double diffNorm = (*jacPtrs[i] - *jacNumericPtrs[i]).lpNorm<Eigen::Infinity>();
    if (diffNorm > jacobianTolerance)
      std::cerr << "XParam " << i << " Jacobian =\n"
                << *jacPtrs[i] << "\nnumDiff Jacobian =\n"
                << *jacNumericPtrs[i] << "\nDiff inf norm " << diffNorm
                << "\ndiff\n"
                << (*jacPtrs[i] - *jacNumericPtrs[i]) << "\n";

    OKVIS_ASSERT_LT(Exception, diffNorm, 1e-2,
                    "For XParam " << i << ", numeric Jacobian differs by "
                                  << diffNorm << " from the analytic one.");

    diffNorm = (*jacMinPtrs[i] - *jacMinNumericPtrs[i]).lpNorm<Eigen::Infinity>();
    if (diffNorm > jacobianTolerance)
      std::cerr << "Minimal XParam " << i << " Jacobian =\n"
                << *jacMinPtrs[i] << "\nnumDiff Jacobian =\n"
                << *jacMinNumericPtrs[i] << "\nDiff inf norm " << diffNorm << "\n";
    OKVIS_ASSERT_LT(Exception, diffNorm, 1e-2, "For XParam " << i << ", numeric Jacobian differs by "
                    << diffNorm << " from the analytic one.");
  }
  return true;
}

}  // namespace ceres
}  // namespace okvis
