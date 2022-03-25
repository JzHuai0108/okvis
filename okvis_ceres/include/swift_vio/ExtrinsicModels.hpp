#ifndef INCLUDE_SWIFT_VIO_EXTRINSIC_MODELS_HPP_
#define INCLUDE_SWIFT_VIO_EXTRINSIC_MODELS_HPP_

#include <Eigen/Core>

#include <ceres/local_parameterization.h>

#include <okvis/ModelSwitch.hpp>
#include <swift_vio/ceres/JacobianHelpers.hpp>

// #include <okvis/ceres/PoseLocalParameterization.hpp>
#include <okvis/ceres/LocalParamizationAdditionalInterfaces.hpp>
#include <okvis/kinematics/Transformation.hpp>
#include <okvis/kinematics/sophus_operators.hpp>

namespace swift_vio {
class PoseLocalParameterizationInvTranslation
    : public okvis::ceres::LocalParamizationAdditionalInterfaces {
  // T_AB is parameterized by the variable p_BA.
public:
  static const size_t kNumParams = 3;
  static const size_t kGlobalDim = 7;
  static inline int getMinimalDim() { return kNumParams; }

  virtual ~PoseLocalParameterizationInvTranslation() {}

  template <typename Scalar>
  static void
  oplus(const Scalar *const delta,
        std::pair<Eigen::Matrix<Scalar, 3, 1>, Eigen::Quaternion<Scalar>> *T) {
    Eigen::Map<const Eigen::Matrix<Scalar, 3, 1>> delta_t(delta);
    T->first -= T->second * delta_t;
  }

  template <typename Scalar>
  static void oplus(const Scalar *x, const Scalar *delta,
                    Scalar *x_plus_delta) {
    Eigen::Map<const Eigen::Matrix<Scalar, 3, 1>> dr(delta);
    Eigen::Map<const Eigen::Matrix<Scalar, 3, 1>> r(x);
    Eigen::Map<const Eigen::Quaternion<Scalar>> q(x + 3);
    Eigen::Map<Eigen::Matrix<Scalar, 3, 1>> rplus(x_plus_delta);
    rplus = r - q * dr;
  }

  template <typename Scalar>
  static void ominus(const Scalar *T, const Scalar *T_plus, Scalar *delta) {
    Eigen::Map<const Eigen::Quaternion<Scalar>> q(T + 3);
    Eigen::Map<Eigen::Vector3d> deltaVec(delta);
    deltaVec = q.conjugate() * (Eigen::Map<const Eigen::Vector3d>(T) -
                                Eigen::Map<const Eigen::Vector3d>(T_plus));
  }

  bool Minus(const double *x, const double *x_plus_delta,
             double *delta) const final {
    ominus(x, x_plus_delta, delta);
    return true;
  }

  /// \brief Computes the Jacobian from minimal space to naively
  /// overparameterised space as used by ceres.
  /// @param[in] x Variable.
  /// @param[out] jacobian the Jacobian (dimension minDim x dim).
  /// \return True on success.
  static bool liftJacobian(const double * /*x*/, double *jacobian) {
    Eigen::Map<Eigen::Matrix<double, kNumParams, kGlobalDim, Eigen::RowMajor>>
        J_lift(jacobian);
    J_lift.setIdentity();
    return true;
  }

  /// \brief Computes the Jacobian from minimal space to naively
  /// overparameterised space as used by ceres.
  /// @param[in] x Variable.
  /// @param[out] jacobian the Jacobian (dimension minDim x dim).
  /// \return True on success.
  bool ComputeLiftJacobian(const double *x, double *jacobian) const final {
    return liftJacobian(x, jacobian);
  }

  bool Plus(const double *x, const double *delta,
            double *x_plus_delta) const final {
    // transform to okvis::kinematics framework
    std::pair<Eigen::Matrix<double, 3, 1>, Eigen::Quaternion<double>> T(
        Eigen::Vector3d(x[0], x[1], x[2]),
        Eigen::Quaterniond(x[6], x[3], x[4], x[5]));

    oplus(delta, &T);

    // copy back
    const Eigen::Vector3d &r = T.first;
    x_plus_delta[0] = r[0];
    x_plus_delta[1] = r[1];
    x_plus_delta[2] = r[2];
    return true;
  }

  // https://www.jianzhuhuai.com/post/lift-jacobian/
  bool ComputeJacobian(const double * /*x*/, double *jacobian) const final {
    Eigen::Map<Eigen::Matrix<double, kGlobalDim, kNumParams, Eigen::RowMajor>>
        j(jacobian);
    j.setIdentity();
    return true;
  }

  int GlobalSize() const final { return kGlobalDim; }
  int LocalSize() const final { return kNumParams; }
};

class Extrinsic_p_CB final : public PoseLocalParameterizationInvTranslation {
public:
  static const int kModelId = 1;
  virtual ~Extrinsic_p_CB() {}
  static inline Eigen::MatrixXd initCov(double sigma_translation,
                                        double /*sigma_orientation*/) {
    return Eigen::MatrixXd::Identity(3, 3) *
           (sigma_translation * sigma_translation);
  }

  /**
   * @brief dT_BC_dExtrinsic Jacobian of perturbation in the tangent space of
   * \f$ T_{BC} \f$ relative to perturbation on the parameters \f$ p_{CB} \f$.
   * The perturbation in the tangent space of \f$ T_{BC} \f$ is \f$ [\delta t,
   * \delta \theta] \f$. \f$ R_{BC} = Exp(\delta\theta) \hat{R}_{BC} \f$ \f$
   * t_{BC} = \delta t + \hat{t}_{BC} \f$
   * @param T_BCi current camera relative pose.
   * @param j Jacobian
   */
  static void
  dT_BC_dExtrinsic(const okvis::kinematics::Transformation &T_BCi,
                   const okvis::kinematics::Transformation * /*T_BC0*/,
                   Eigen::Matrix<double, 6, kNumParams> *j) {
    j->topRows<3>() = -T_BCi.C();
    j->bottomRows<3>().setZero();
  }

  static void dT_BC_dT_BC0(const okvis::kinematics::Transformation & /*T_BCi*/,
                           const okvis::kinematics::Transformation * /*T_BC0*/,
                           Eigen::Matrix<double, 6, 6> *j) {
    j->setZero();
  }

  // the error state is $\delta p_B^C$ or $\delta p_S^C$
  //  static void updateState(const Eigen::Vector3d &r, const Eigen::Quaterniond
  //  &q,
  //                          const Eigen::VectorXd &delta,
  //                          Eigen::Vector3d *r_delta,
  //                          Eigen::Quaterniond *q_delta) {
  //    *r_delta = r - q * delta;
  //    *q_delta = q;
  //  }

  static void toDimensionLabels(std::vector<std::string> *extrinsicLabels) {
    *extrinsicLabels = {"p_CS_C_x[m]", "p_CS_C_y", "p_CS_C_z"};
  }

  static void toMinDimensionLabels(std::vector<std::string> *extrinsicLabels) {
    *extrinsicLabels = {"p_CS_C_x[m]", "p_CS_C_y", "p_CS_C_z"};
  }

  static void toDesiredStdevs(Eigen::VectorXd *desiredStdevs) {
    desiredStdevs->resize(3);
    desiredStdevs->setConstant(0.04);
  }

  static void toParamValues(const okvis::kinematics::Transformation &T_BC,
                            Eigen::VectorXd *extrinsic_opt_coeffs) {
    *extrinsic_opt_coeffs = T_BC.q().conjugate() * (-T_BC.r());
  }

  //  template <class Scalar>
  //  static std::pair<Eigen::Matrix<Scalar, 3, 1>, Eigen::Quaternion<Scalar>>
  //  get_T_BC(const okvis::kinematics::Transformation &T_BC_base,
  //           const Scalar *parameters) {
  //    Eigen::Matrix<Scalar, 3, 1> t_CB_C(parameters[0], parameters[1],
  //                                       parameters[2]);
  //    Eigen::Quaternion<Scalar> q_BC = T_BC_base.q().cast<Scalar>();
  //    return std::make_pair(q_BC * (-t_CB_C), q_BC);
  //  }
};

// This is different from PoseLocalParameterization in liftJacobian() and
// ComputeJacobian() in that both are set to identity here.
class PoseLocalParameterizationSimplified
    : public okvis::ceres::LocalParamizationAdditionalInterfaces {
public:
  static const size_t kNumParams = 6;
  static const size_t kGlobalDim = 7;
  virtual ~PoseLocalParameterizationSimplified() {}

  static inline int getMinimalDim() { return kNumParams; }

  //  static void updateState(const Eigen::Vector3d &r, const Eigen::Quaterniond
  //  &q,
  //                          const Eigen::VectorXd &delta,
  //                          Eigen::Vector3d *r_delta,
  //                          Eigen::Quaterniond *q_delta) {
  //    Eigen::Vector3d deltaAlpha = delta.segment<3>(3);
  //    *r_delta = r + delta.head<3>();
  //    *q_delta = okvis::kinematics::expAndTheta(deltaAlpha) * q;
  //    q_delta->normalize();
  //  }

  template <typename Scalar>
  static void
  oplus(const Scalar *const deltaT,
        std::pair<Eigen::Matrix<Scalar, 3, 1>, Eigen::Quaternion<Scalar>> *T) {
    Eigen::Map<const Eigen::Matrix<Scalar, 6, 1>> delta(deltaT);
    T->first += delta.template head<3>();
    Eigen::Matrix<Scalar, 3, 1> omega = delta.template tail<3>();
    Eigen::Quaternion<Scalar> dq = okvis::kinematics::expAndTheta(omega);
    T->second = dq * T->second;
  }

  template <typename Scalar>
  static void oplus(const Scalar *x, const Scalar *delta,
                    Scalar *x_plus_delta) {
    Eigen::Map<const Eigen::Matrix<Scalar, 3, 1>> dr(delta);
    Eigen::Map<const Eigen::Matrix<Scalar, 3, 1>> dq(delta + 3);
    Eigen::Map<const Eigen::Matrix<Scalar, 3, 1>> r(x);
    Eigen::Map<const Eigen::Quaternion<Scalar>> q(x + 3);
    Eigen::Map<Eigen::Matrix<Scalar, 3, 1>> rplus(x_plus_delta);
    Eigen::Map<Eigen::Quaternion<Scalar>> qplus(x_plus_delta + 3);
    rplus = r + dr;
    qplus = okvis::kinematics::expAndTheta(Eigen::Matrix<Scalar, 3, 1>(dq)) * q;
  }

  template <typename Scalar>
  static void ominus(const Scalar *T, const Scalar *T_plus, Scalar *delta) {
    delta[0] = T_plus[0] - T[0];
    delta[1] = T_plus[1] - T[1];
    delta[2] = T_plus[2] - T[2];
    const Eigen::Quaterniond q_plus(T_plus[6], T_plus[3], T_plus[4], T_plus[5]);
    const Eigen::Quaterniond q(T[6], T[3], T[4], T[5]);
    Eigen::Map<Eigen::Vector3d> delta_q(&delta[3]);
    double theta;
    delta_q = okvis::kinematics::logAndTheta(q_plus * q.inverse(), &theta);
  }

  bool Minus(const double *x, const double *x_plus_delta,
             double *delta) const final {
    ominus(x, x_plus_delta, delta);
    return true;
  }

  /**
   * @brief toParamValues
   * @param T
   * @param extrinsic_opt_coeffs
   */
  static void toParamValues(const okvis::kinematics::Transformation &T,
                            Eigen::VectorXd *extrinsic_opt_coeffs) {
    *extrinsic_opt_coeffs = T.coeffs();
  }

  /// \brief Computes the Jacobian from minimal space to naively
  /// overparameterised space as used by ceres.
  ///     It must be the pseudo inverse of the local parametrization Jacobian as
  ///     obtained by COmputeJacobian.
  /// @param[in] x Variable.
  /// @param[out] jacobian the Jacobian (dimension minDim x dim).
  /// \return True on success.
  static bool liftJacobian(const double * /*x*/, double *jacobian) {
    Eigen::Map<Eigen::Matrix<double, kNumParams, kGlobalDim, Eigen::RowMajor>>
        J_lift(jacobian);
    J_lift.setIdentity();
    return true;
  }

  /// \brief Computes the Jacobian from minimal space to naively
  /// overparameterised space as used by ceres.
  /// @param[in] x Variable.
  /// @param[out] jacobian the Jacobian (dimension minDim x dim).
  /// \return True on success.
  bool ComputeLiftJacobian(const double *x, double *jacobian) const final {
    return liftJacobian(x, jacobian);
  }

  bool Plus(const double *x, const double *delta,
            double *x_plus_delta) const final {
    // transform to okvis::kinematics framework
    std::pair<Eigen::Matrix<double, 3, 1>, Eigen::Quaternion<double>> T(
        Eigen::Vector3d(x[0], x[1], x[2]),
        Eigen::Quaterniond(x[6], x[3], x[4], x[5]));
    // call oplus operator in okvis::kinematis
    oplus(delta, &T);

    // copy back
    const Eigen::Vector3d &r = T.first;
    x_plus_delta[0] = r[0];
    x_plus_delta[1] = r[1];
    x_plus_delta[2] = r[2];
    const Eigen::Vector4d &q = T.second.coeffs();
    x_plus_delta[3] = q[0];
    x_plus_delta[4] = q[1];
    x_plus_delta[5] = q[2];
    x_plus_delta[6] = q[3];
    return true;
  }

  bool ComputeJacobian(const double * /*x*/, double *jacobian) const final {
    Eigen::Map<Eigen::Matrix<double, kGlobalDim, kNumParams, Eigen::RowMajor>>
        j(jacobian);
    j.setIdentity();
    return true;
  }

  int GlobalSize() const final { return kGlobalDim; }
  int LocalSize() const final { return kNumParams; }
};

class Extrinsic_p_BC_q_BC final : public PoseLocalParameterizationSimplified {
public:
  static const int kModelId = 2;
  virtual ~Extrinsic_p_BC_q_BC() {}

  static void toDimensionLabels(std::vector<std::string> *extrinsicLabels) {
    *extrinsicLabels = {"p_SC_S_x[m]", "p_SC_S_y", "p_SC_S_z", "q_SC_x",
                        "q_SC_y",      "q_SC_z",   "q_SC_w"};
  }

  static void toMinDimensionLabels(std::vector<std::string> *extrinsicLabels) {
    *extrinsicLabels = {"p_SC_S_x[m]", "p_SC_S_y",   "p_SC_S_z",
                        "theta_SC_x",  "theta_SC_y", "theta_SC_z"};
  }

  static void toDesiredStdevs(Eigen::VectorXd *desiredStdevs) {
    desiredStdevs->resize(6);
    desiredStdevs->head<3>().setConstant(0.04);
    desiredStdevs->tail<3>().setConstant(0.01);
  }

  static inline Eigen::MatrixXd initCov(double sigma_translation,
                                        double sigma_orientation) {
    Eigen::Matrix<double, 6, 6> cov = Eigen::Matrix<double, 6, 6>::Identity();
    cov.topLeftCorner<3, 3>() *= (sigma_translation * sigma_translation);
    cov.bottomRightCorner<3, 3>() *= (sigma_orientation * sigma_orientation);
    return cov;
  }

  //  template <class Scalar>
  //  static std::pair<Eigen::Matrix<Scalar, 3, 1>, Eigen::Quaternion<Scalar>>
  //  get_T_BC(const okvis::kinematics::Transformation & /*T_BC_base*/,
  //           const Scalar *parameters) {
  //    Eigen::Matrix<Scalar, 3, 1> t_BC_B(parameters[0], parameters[1],
  //                                       parameters[2]);
  //    Eigen::Quaternion<Scalar> q_BC(parameters[6], parameters[3],
  //    parameters[4],
  //                                   parameters[5]);
  //    return std::make_pair(t_BC_B, q_BC);
  //  }

  static void
  dT_BC_dExtrinsic(const okvis::kinematics::Transformation & /*T_BCi*/,
                   const okvis::kinematics::Transformation * /*T_BC0*/,
                   Eigen::Matrix<double, 6, kNumParams> *j) {
    j->setIdentity();
  }

  static void dT_BC_dT_BC0(const okvis::kinematics::Transformation & /*T_BCi*/,
                           const okvis::kinematics::Transformation * /*T_BC0*/,
                           Eigen::Matrix<double, 6, 6> *j) {
    j->setZero();
  }

  /**
   * @brief dpC_dExtrinsic_AIDP anchored inverse depth
   * @param pC pC = (T_BC^{-1} * T_WB_{f_i}^{-1} * T_WB_a * T_BC * [a, b, 1,
   * \rho]^T)_{1:3} R_BC = exp(\delta\theta) \hat{R}_BC t_BC = \delta t +
   * \hat{t}_BC
   * @param R_CB
   * @param dpC_dT
   * @param R_CfCa
   * @param ab1rho
   */
  //  static void dpC_dExtrinsic_AIDP(const Eigen::Vector3d &pC,
  //                                  const Eigen::Matrix3d &R_CB,
  //                                  Eigen::MatrixXd *dpC_dT,
  //                                  const Eigen::Matrix3d *R_CfCa,
  //                                  const Eigen::Vector4d *ab1rho) {
  //    dpC_dT->resize(3, 6);
  //    dpC_dT->block<3, 3>(0, 0) =
  //        ((*R_CfCa) - Eigen::Matrix3d::Identity()) * R_CB * (*ab1rho)[3];
  //    dpC_dT->block<3, 3>(0, 3) =
  //        (okvis::kinematics::crossMx(pC) -
  //         (*R_CfCa) * okvis::kinematics::crossMx(ab1rho->head<3>())) *
  //        R_CB;
  //  }

  /**
   * @brief dpC_dExtrinsic_HPP homogeneous point
   * @param hpC hpC = T_BC^{-1} * [x,y,z,w]_B^T
   *     hpC = [pC, w]
   *     R_BC = exp(\delta\theta) \hat{R}_BC
   *     t_BC = \delta t + \hat{t}_BC
   * @param R_CB
   * @param dpC_dT 3x6
   */
  //  static void dpC_dExtrinsic_HPP(const Eigen::Vector4d &hpC,
  //                                 const Eigen::Matrix3d &R_CB,
  //                                 Eigen::MatrixXd *dpC_dT) {
  //    dpC_dT->resize(3, 6);
  //    dpC_dT->block<3, 3>(0, 0) = -R_CB * hpC[3];
  //    dpC_dT->block<3, 3>(0, 3) =
  //        okvis::kinematics::crossMx(hpC.head<3>()) * R_CB;
  //  }
};

class Extrinsic_p_C0C_q_C0C final : public PoseLocalParameterizationSimplified {
public:
  static const int kModelId = 3;

  virtual ~Extrinsic_p_C0C_q_C0C() {}

  static void toDimensionLabels(std::vector<std::string> *extrinsicLabels) {
    *extrinsicLabels = {"p_C0C_C0_x[m]", "p_C0C_C0_y", "p_C0C_C0_z", "q_C0C_x",
                        "q_C0C_y",       "q_C0C_z",    "q_C0C_w"};
  }

  static void toMinDimensionLabels(std::vector<std::string> *extrinsicLabels) {
    *extrinsicLabels = {"p_C0C_C0_x[m]", "p_C0C_C0_y",  "p_C0C_C0_z",
                        "theta_C0C_x",   "theta_C0C_y", "theta_C0C_z"};
  }

  static void toDesiredStdevs(Eigen::VectorXd *desiredStdevs) {
    desiredStdevs->resize(6);
    desiredStdevs->head<3>().setConstant(0.04);
    desiredStdevs->tail<3>().setConstant(0.01);
  }

  static inline Eigen::MatrixXd initCov(double sigma_translation,
                                        double sigma_orientation) {
    Eigen::Matrix<double, 6, 6> cov = Eigen::Matrix<double, 6, 6>::Identity();
    cov.topLeftCorner<3, 3>() *= (sigma_translation * sigma_translation);
    cov.bottomRightCorner<3, 3>() *= (sigma_orientation * sigma_orientation);
    return cov;
  }

  /**
   * @brief dT_BC_dExtrinsic
   * @param T_BCi current camera relative pose
   * @param T_BC0 main camera relative pose
   * @param j_C0Ci Jacobian of T_BCi relative to T_C0Ci
   */
  static void dT_BC_dExtrinsic(const okvis::kinematics::Transformation &T_BCi,
                               const okvis::kinematics::Transformation *T_BC0,
                               Eigen::Matrix<double, 6, kNumParams> *j_C0Ci);

  /**
   * @brief dT_BC_dT_BC0
   * @param T_BCi current camera relative pose
   * @param T_BC0 main camera relative pose
   * @param j_BC0 Jacobian of T_BCi relative to T_BC0.
   */
  static void dT_BC_dT_BC0(const okvis::kinematics::Transformation &T_BCi,
                           const okvis::kinematics::Transformation *T_BC0,
                           Eigen::Matrix<double, 6, 6> *j_BC0);

  //  static void dpC_dExtrinsic_AIDP(const Eigen::Vector3d & /*pC*/,
  //                                  const Eigen::Matrix3d & /*R_CB*/,
  //                                  Eigen::MatrixXd * /*dpC_dT*/,
  //                                  const Eigen::Matrix3d * /*R_CfCa*/,
  //                                  const Eigen::Vector4d * /*ab1rho*/) {
  //    throw std::runtime_error("Method not implemented!");
  //  }

  //  static void dpC_dExtrinsic_HPP(const Eigen::Vector4d & /*hpC*/,
  //                                 const Eigen::Matrix3d & /*R_CB*/,
  //                                 Eigen::MatrixXd * /*dpC_dT*/) {
  //    throw std::runtime_error("Method not implemented!");
  //  }
};

#ifndef EXTRINSIC_MODEL_CASES
#define EXTRINSIC_MODEL_CASES                                                  \
  EXTRINSIC_MODEL_CASE(Extrinsic_p_CB)                                         \
  EXTRINSIC_MODEL_CASE(Extrinsic_p_BC_q_BC)                                    \
  EXTRINSIC_MODEL_CASE(Extrinsic_p_C0C_q_C0C)
#endif

inline int ExtrinsicModelGetMinimalDim(int model_id) {
  switch (model_id) {
#define MODEL_CASES EXTRINSIC_MODEL_CASES
#define EXTRINSIC_MODEL_CASE(ExtrinsicModel)                                   \
  case ExtrinsicModel::kModelId:                                               \
    return ExtrinsicModel::getMinimalDim();

    MODEL_SWITCH_CASES

#undef EXTRINSIC_MODEL_CASE
#undef MODEL_CASES
  }
  return 0;
}

inline Eigen::MatrixXd ExtrinsicModelInitCov(int model_id,
                                             double sigma_translation,
                                             double sigma_orientation) {
  switch (model_id) {
#define MODEL_CASES EXTRINSIC_MODEL_CASES
#define EXTRINSIC_MODEL_CASE(ExtrinsicModel)                                   \
  case ExtrinsicModel::kModelId:                                               \
    return ExtrinsicModel::initCov(sigma_translation, sigma_orientation);

    MODEL_SWITCH_CASES

#undef EXTRINSIC_MODEL_CASE
#undef MODEL_CASES
  }
}

inline int ExtrinsicModelNameToId(std::string extrinsic_opt_rep,
                                  bool *isFixed = nullptr) {
  std::transform(extrinsic_opt_rep.begin(), extrinsic_opt_rep.end(),
                 extrinsic_opt_rep.begin(),
                 [](unsigned char c) { return std::toupper(c); });
  if (extrinsic_opt_rep.compare("P_BC_Q_BC") == 0) {
    if (isFixed) {
      *isFixed = false;
    }
    return Extrinsic_p_BC_q_BC::kModelId;
  } else if (extrinsic_opt_rep.compare("P_C0C_Q_C0C") == 0) {
    if (isFixed) {
      *isFixed = false;
    }
    return Extrinsic_p_C0C_q_C0C::kModelId;
  } else if (extrinsic_opt_rep.compare("P_CB") == 0) {
    if (isFixed) {
      *isFixed = false;
    }
    return Extrinsic_p_CB::kModelId;
  } else {
    if (isFixed) {
      *isFixed = true;
    }
    return Extrinsic_p_BC_q_BC::kModelId;
  }
}

inline void ExtrinsicModelOplus(int model_id, const double *const deltaT_XC,
                                okvis::kinematics::Transformation *T_XC) {
  std::pair<Eigen::Matrix<double, 3, 1>, Eigen::Quaternion<double>> pair_T_XC =
      std::make_pair(T_XC->r(), T_XC->q());
  switch (model_id) {
#define MODEL_CASES EXTRINSIC_MODEL_CASES
#define EXTRINSIC_MODEL_CASE(ExtrinsicModel)                                   \
  case ExtrinsicModel::kModelId:                                               \
    ExtrinsicModel::oplus(deltaT_XC, &pair_T_XC);                              \
    T_XC->set(pair_T_XC.first, pair_T_XC.second);                              \
    break;

    MODEL_SWITCH_CASES

#undef EXTRINSIC_MODEL_CASE
#undef MODEL_CASES
  }
}

inline void ExtrinsicModelOminus(int model_id, const double *rq,
                                 const double *rq_delta, double *delta) {
  switch (model_id) {
#define MODEL_CASES EXTRINSIC_MODEL_CASES
#define EXTRINSIC_MODEL_CASE(ExtrinsicModel)                                   \
  case ExtrinsicModel::kModelId:                                               \
    return ExtrinsicModel::ominus(rq, rq_delta, delta);

    MODEL_SWITCH_CASES

#undef EXTRINSIC_MODEL_CASE
#undef MODEL_CASES
  }
}

inline void
ExtrinsicModelToDimensionLabels(int model_id,
                                std::vector<std::string> *dimensionLabels) {
  switch (model_id) {
#define MODEL_CASES EXTRINSIC_MODEL_CASES
#define EXTRINSIC_MODEL_CASE(ExtrinsicModel)                                   \
  case ExtrinsicModel::kModelId:                                               \
    return ExtrinsicModel::toDimensionLabels(dimensionLabels);

    MODEL_SWITCH_CASES

#undef EXTRINSIC_MODEL_CASE
#undef MODEL_CASES
  }
}

inline void
ExtrinsicModelToMinDimensionLabels(int model_id,
                                   std::vector<std::string> *dimensionLabels) {
  switch (model_id) {
#define MODEL_CASES EXTRINSIC_MODEL_CASES
#define EXTRINSIC_MODEL_CASE(ExtrinsicModel)                                   \
  case ExtrinsicModel::kModelId:                                               \
    return ExtrinsicModel::toMinDimensionLabels(dimensionLabels);

    MODEL_SWITCH_CASES

#undef EXTRINSIC_MODEL_CASE
#undef MODEL_CASES
  }
}

inline void
ExtrinsicModelToParamValues(int model_id,
                            const okvis::kinematics::Transformation &T_XC,
                            Eigen::VectorXd *extrinsic_opt_coeffs) {
  switch (model_id) {
#define MODEL_CASES EXTRINSIC_MODEL_CASES
#define EXTRINSIC_MODEL_CASE(ExtrinsicModel)                                   \
  case ExtrinsicModel::kModelId:                                               \
    return ExtrinsicModel::toParamValues(T_XC, extrinsic_opt_coeffs);

    MODEL_SWITCH_CASES

#undef EXTRINSIC_MODEL_CASE
#undef MODEL_CASES
  }
}

inline void ExtrinsicModelToDesiredStdevs(int model_id,
                                          Eigen::VectorXd *desiredStdevs) {
  switch (model_id) {
#define MODEL_CASES EXTRINSIC_MODEL_CASES
#define EXTRINSIC_MODEL_CASE(ExtrinsicModel)                                   \
  case ExtrinsicModel::kModelId:                                               \
    return ExtrinsicModel::toDesiredStdevs(desiredStdevs);

    MODEL_SWITCH_CASES

#undef EXTRINSIC_MODEL_CASE
#undef MODEL_CASES
  }
}
} // namespace swift_vio
#endif // INCLUDE_SWIFT_VIO_EXTRINSIC_MODELS_HPP_
