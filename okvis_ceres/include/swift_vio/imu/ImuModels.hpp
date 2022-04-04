#ifndef INCLUDE_SWIFT_VIO_IMU_ERROR_MODELS_HPP_
#define INCLUDE_SWIFT_VIO_IMU_ERROR_MODELS_HPP_

// Generic methods specific to each IMU model is encapsulated in the following classes.
// These models are to be used in constructing ceres::SizedCostFunction or
// ceres::AutoDiffCostFunction.

#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <okvis/kinematics/sophus_operators.hpp>
#include <okvis/ModelSwitch.hpp>
#include <okvis/Parameters.hpp>
#include <okvis/Time.hpp>
#include <okvis/kinematics/MatrixPseudoInverse.hpp>

#include <swift_vio/matrixUtilities.h>
#include <swift_vio/ParallaxAnglePoint.hpp>

#define IMU_MODEL_SHARED_MEMBERS                                               \
public:                                                                        \
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW                                              \
  template <size_t XBlockId>                                                   \
  Eigen::Matrix<double, 3, kXBlockDims[XBlockId]> dDp_dx() const {             \
    return dDp_dx_.middleCols<kXBlockDims[XBlockId]>(                          \
        kCumXBlockDims[XBlockId]);                                             \
  }                                                                            \
  template <size_t XBlockId>                                                   \
  Eigen::Matrix<double, 3, kXBlockDims[XBlockId]> dDrot_dx() const {           \
    return dDrot_dx_.middleCols<kXBlockDims[XBlockId]>(                        \
        kCumXBlockDims[XBlockId]);                                             \
  }                                                                            \
  template <size_t XBlockId>                                                   \
  Eigen::Matrix<double, 3, kXBlockDims[XBlockId]> dDv_dx() const {             \
    return dDv_dx_.middleCols<kXBlockDims[XBlockId]>(                          \
        kCumXBlockDims[XBlockId]);                                             \
  }                                                                            \
  template <size_t XBlockId>                                                   \
  Eigen::Matrix<double, 3, kXBlockMinDims[XBlockId]> dDp_dminx() const {       \
    return dDp_dminx_.middleCols<kXBlockMinDims[XBlockId]>(                    \
        kCumXBlockMinDims[XBlockId]);                                          \
  }                                                                            \
  template <size_t XBlockId>                                                   \
  Eigen::Matrix<double, 3, kXBlockMinDims[XBlockId]> dDrot_dminx() const {     \
    return dDrot_dminx_.middleCols<kXBlockMinDims[XBlockId]>(                  \
        kCumXBlockMinDims[XBlockId]);                                          \
  }                                                                            \
  template <size_t XBlockId>                                                   \
  Eigen::Matrix<double, 3, kXBlockMinDims[XBlockId]> dDv_dminx() const {       \
    return dDv_dminx_.middleCols<kXBlockMinDims[XBlockId]>(                    \
        kCumXBlockMinDims[XBlockId]);                                          \
  }                                                                            \
  Eigen::Matrix<double, 3, 3> dDp_dbg() const {                                \
    return dDp_db_.leftCols<3>();                                              \
  }                                                                            \
  Eigen::Matrix<double, 3, 3> dDp_dba() const {                                \
    return dDp_db_.rightCols<3>();                                             \
  }                                                                            \
  Eigen::Matrix<double, 3, 3> dDrot_dbg() const {                              \
    return dDrot_db_.leftCols<3>();                                            \
  }                                                                            \
  Eigen::Matrix<double, 3, 3> dDrot_dba() const {                              \
    return dDrot_db_.rightCols<3>();                                           \
  }                                                                            \
  Eigen::Matrix<double, 3, 3> dDv_dbg() const {                                \
    return dDv_db_.leftCols<3>();                                              \
  }                                                                            \
  Eigen::Matrix<double, 3, 3> dDv_dba() const {                                \
    return dDv_db_.rightCols<3>();                                             \
  }                                                                            \                                                                  \
private:                                                                       \
  Eigen::Matrix<double, 3, 1> bg_;                                             \
  Eigen::Matrix<double, 3, 1> ba_;                                             \
  Eigen::Matrix<double, 3, kAugmentedDim> dDp_dx_, dDrot_dx_, dDv_dx_;         \
  Eigen::Matrix<double, 3, kAugmentedMinDim> dDp_dminx_, dDrot_dminx_,         \
      dDv_dminx_;                                                              \
  Eigen::Matrix<double, 3, 6> dDp_db_, dDrot_db_, dDv_db_;                     \
  Eigen::Quaterniond Delta_q_ = Eigen::Quaterniond(1, 0, 0, 0);                \
  Eigen::Matrix3d C_integral_ = Eigen::Matrix3d::Zero();                       \
  Eigen::Matrix3d C_doubleintegral_ = Eigen::Matrix3d::Zero();                 \
  Eigen::Vector3d acc_integral_ = Eigen::Vector3d::Zero();                     \
  Eigen::Vector3d acc_doubleintegral_ = Eigen::Vector3d::Zero();               \
  Eigen::Matrix<double, 15, 15> P_delta_ =                                     \
      Eigen::Matrix<double, 15, 15>::Zero();

namespace swift_vio {
static const int kBgBaDim = 6; // bg ba

/**
 * @brief The Imu_BG_BA class
 * The body frame coincides the IMU sensor frame realized by the accelerometers.
 * This model assumes that the accelerometer triad and the gyroscope triad are free of
 * scaling error and misalignment, and the two triads realize the same coordinate frame.
 */
class Imu_BG_BA {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  static const int kModelId = 0;
  static const size_t kGlobalDim = kBgBaDim;
  static const size_t kAugmentedDim = 0;
  static const size_t kAugmentedMinDim = 0;
  static constexpr std::array<int, 0> kXBlockDims{};
  static constexpr std::array<int, 0> kXBlockMinDims{};
  static constexpr std::array<int, 0> kCumXBlockDims{};
  static constexpr std::array<int, 0> kCumXBlockMinDims{};
  static constexpr double kJacobianTolerance = 1.0e-3;

  /**
   * @brief getAugmentedDim
   * @return dim of all the augmented params.
   */
  static inline int getAugmentedDim() { return kAugmentedDim; }
  /**
   * @brief getMinimalDim
   * @return minimal dim of all the params.
   */
  static inline int getMinimalDim() { return kGlobalDim; }
  /**
   * @brief getAugmentedMinimalDim
   * @return minimal dim of all augmented params.
   */
  static inline int getAugmentedMinimalDim() { return kAugmentedDim; }
  /**
   * get nominal values for augmented params.
   */
  template <typename T>
  static Eigen::Matrix<T, kAugmentedDim, 1> getNominalAugmentedParams() {
    return Eigen::Matrix<T, kAugmentedDim, 1>::Zero();
  }
  /**
   * predict IMU measurement from values in the body frame.
   * This function is used for testing purposes.
   */
  template <typename T>
  static void
  predict(const Eigen::Matrix<T, 3, 1> &bg, const Eigen::Matrix<T, 3, 1> &ba,
          const Eigen::Matrix<T, Eigen::Dynamic, 1> & /*extraParams*/,
          const Eigen::Matrix<T, 3, 1> &w_b, const Eigen::Matrix<T, 3, 1> &a_b,
          Eigen::Matrix<T, 3, 1> *w, Eigen::Matrix<T, 3, 1> *a) {
    *a = a_b + ba;
    *w = w_b + bg;
  }
  /**
   * correct IMU measurement to the body frame.
   * This function is used by the ceres::CostFunction.
   * @param[in] params bg ba and augmented Euclidean params.
   * @param[in] q_gyro_i orientation from the accelerometer triad input reference frame,
   *     i.e., the IMU sensor frame to the gyro triad input reference frame.
   * @param[in] w, a angular velocity and linear acceleration measured by the IMU.
   * @param[out] w_b, a_b angular velocity and linear acceleration in the body frame.
   */
  template <typename T>
  static void
  correct(const Eigen::Matrix<T, 3, 1> &bg, const Eigen::Matrix<T, 3, 1> &ba,
          const Eigen::Matrix<T, Eigen::Dynamic, 1> & /*params*/,
          const Eigen::Matrix<T, 3, 1> &w, const Eigen::Matrix<T, 3, 1> &a,
          Eigen::Matrix<T, 3, 1> *w_b, Eigen::Matrix<T, 3, 1> *a_b) {
    *a_b = a - ba;
    *w_b = w - bg;
  }

  static Eigen::VectorXd computeAugmentedParamsError(
      const Eigen::VectorXd& /*params*/) {
      return Eigen::VectorXd(0);
  }

  template <typename T>
  void correct(const Eigen::Matrix<T, 3, 1> &w, const Eigen::Matrix<T, 3, 1> &a,
               Eigen::Matrix<T, 3, 1> *w_b, Eigen::Matrix<T, 3, 1> *a_b,
               Eigen::Matrix<T, 6, kGlobalDim> *jacobian = nullptr) const {
    *a_b = a - ba_;
    *w_b = w - bg_;
    if (jacobian) {
      jacobian->setZero();
      for (size_t i = 0u; i < kGlobalDim; ++i)
        (*jacobian)(i, i) = -1;
    }
  }

  void updateParameters(const double * bgba, double const * const * /*xparams*/) {
    bg_ = Eigen::Map<const Eigen::Matrix<double, 3, 1>>(bgba);
    ba_ = Eigen::Map<const Eigen::Matrix<double, 3, 1>>(bgba + 3);
  }

  void updateParameters(const double * bgba, double const * /*xparams*/ = nullptr) {
    bg_ = Eigen::Map<const Eigen::Matrix<double, 3, 1>>(bgba);
    ba_ = Eigen::Map<const Eigen::Matrix<double, 3, 1>>(bgba + 3);
  }

  void propagate(double dt, const Eigen::Vector3d &omega_S_0,
                 const Eigen::Vector3d &acc_S_0,
                 const Eigen::Vector3d &omega_S_1,
                 const Eigen::Vector3d &acc_S_1, double sigma_g_c,
                 double sigma_a_c, double sigma_gw_c, double sigma_aw_c);

  void resetPreintegration();

  template <typename DerivedT>
  void getWeight(Eigen::MatrixBase<DerivedT> *information,
                 Eigen::MatrixBase<DerivedT> *rootInformation = nullptr) {
    P_delta_ = 0.5 * (P_delta_ + P_delta_.transpose().eval());
    if (rootInformation) {
      okvis::MatrixPseudoInverse::inverseSymmSqrt(
          P_delta_.topLeftCorner<DerivedT::RowsAtCompileTime,
                                 DerivedT::ColsAtCompileTime>(),
          *rootInformation, *information);
      *rootInformation = rootInformation->transpose().eval();
    } else {
      okvis::MatrixPseudoInverse::inverseSymm(
          P_delta_.topLeftCorner<DerivedT::RowsAtCompileTime,
                                 DerivedT::ColsAtCompileTime>(),
          *information);
    }
  }

  Eigen::Matrix<double, 3, 3> dDp_dbg() const { return dp_db_g_; }
  Eigen::Matrix<double, 3, 3> dDp_dba() const { return -C_doubleintegral_; }
  Eigen::Matrix<double, 3, 3> dDrot_dbg() const { return -dalpha_db_g_; }
  Eigen::Matrix<double, 3, 3> dDrot_dba() const {
    return Eigen::Matrix3d::Zero();
  }
  Eigen::Matrix<double, 3, 3> dDv_dbg() const { return dv_db_g_; }
  Eigen::Matrix<double, 3, 3> dDv_dba() const { return -C_integral_; }
  const Eigen::Quaterniond &Delta_q() const { return Delta_q_; }
  const Eigen::Vector3d &Delta_p() const { return acc_doubleintegral_; }
  const Eigen::Vector3d &Delta_v() const { return acc_integral_; }

private:
  Eigen::Matrix<double, 3, 1> bg_;
  Eigen::Matrix<double, 3, 1> ba_;

  // increments (initialise with identity)
  Eigen::Quaterniond Delta_q_ = Eigen::Quaterniond(1,0,0,0); ///< Intermediate result
  Eigen::Matrix3d C_integral_ = Eigen::Matrix3d::Zero(); ///< Intermediate result
  Eigen::Matrix3d C_doubleintegral_ = Eigen::Matrix3d::Zero(); ///< Intermediate result
  Eigen::Vector3d acc_integral_ = Eigen::Vector3d::Zero(); ///< Intermediate result
  Eigen::Vector3d acc_doubleintegral_ = Eigen::Vector3d::Zero(); ///< Intermediate result

  // cross matrix accumulatrion
  Eigen::Matrix3d cross_ = Eigen::Matrix3d::Zero(); ///< Intermediate result

  // sub-Jacobians
  Eigen::Matrix3d dalpha_db_g_ = Eigen::Matrix3d::Zero(); ///< Intermediate result
  Eigen::Matrix3d dv_db_g_ = Eigen::Matrix3d::Zero(); ///< Intermediate result
  Eigen::Matrix3d dp_db_g_ = Eigen::Matrix3d::Zero(); ///< Intermediate result

  Eigen::Matrix<double, 15, 15> P_delta_ = Eigen::Matrix<double, 15, 15>::Zero();
};

/**
 * @brief The Imu_BG_BA_TG_TS_TA class
 * The body frame is defined relative to an external sensor, e.g., the camera. Its
 * orientation is fixed to the nominal value of the orientation between the IMU
 * sensor frame and the camera frame \f$ R_{SC0} \f$ and its origin is at
 * the accelerometer intersection. Thus both accelerometer triad and 
 * gyroscope triad need to account for scaling effect (3), misalignment (3),
 * relative orientation (4, minimal 3) to the body frame.
 * This model also considers the g-sensitivity (9) of the gyroscope triad.
 *
 * IMU model
 * w_m = T_g * w_B + T_s * a_B + b_w + n_w
 * a_m = T_a * a_B + b_a + n_a = S * M * R_AB * a_B + b_a + n_a
 *
 * The accelerometer input frame A is affixed to
 * the accelerometer triad, and its x-axis aligned to the accelerometer in the x direction.
 * Its origin is at the intersection of the three accelerometers.
 * Tts y-axis in the plane spanned by the two accelerometers at x and y
 * direction while being close to the accelerometer at y-direction.
 */
class Imu_BG_BA_TG_TS_TA {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  static const int kModelId = 3;
  static const size_t kAugmentedDim = 27;
  static const size_t kAugmentedMinDim = 27;
  static const size_t kGlobalDim = kAugmentedDim + kBgBaDim;

  static constexpr std::array<int, 3> kXBlockDims{9, 9, 9};  // Tg, Ts, Ta
  static constexpr std::array<int, 3> kXBlockMinDims{9, 9, 9};
  static constexpr std::array<int, 4> kCumXBlockDims{0, 9, 18, 27};  // Tg, Ts, Ta
  static constexpr std::array<int, 4> kCumXBlockMinDims{0, 9, 18, 27};
  static constexpr double kJacobianTolerance = 5.0e-3;

  static inline int getAugmentedDim() { return kAugmentedDim; }
  static inline int getMinimalDim() { return kGlobalDim; }
  static inline int getAugmentedMinimalDim() { return kAugmentedDim; }
  template <typename T>
  static Eigen::Matrix<T, kAugmentedDim, 1> getNominalAugmentedParams() {
    Eigen::Matrix<T, 9, 1> eye;
    eye << 1, 0, 0, 0, 1, 0, 0, 0, 1;
    Eigen::Matrix<T, kAugmentedDim, 1> augmentedParams;
    augmentedParams.template head<9>() = eye;
    augmentedParams.template segment<9>(9).setZero();
    augmentedParams.template tail<9>() = eye;
    return augmentedParams;
  }

  template <typename T>
  static void
  predict(const Eigen::Matrix<T, 3, 1> &bg, const Eigen::Matrix<T, 3, 1> &ba,
          const Eigen::Matrix<T, Eigen::Dynamic, 1> &params,
          const Eigen::Matrix<T, 3, 1> &w_b, const Eigen::Matrix<T, 3, 1> &a_b,
          Eigen::Matrix<T, 3, 1> *w, Eigen::Matrix<T, 3, 1> *a) {
    Eigen::Matrix<T, 3, 3> T_g;
    vectorToMatrix<T>(params.data(), 0, &T_g);
    Eigen::Matrix<T, 3, 3> T_s;
    vectorToMatrix<T>(params.data(), 9, &T_s);
    Eigen::Matrix<T, 3, 3> T_a;
    vectorToMatrix<T>(params.data(), 18, &T_a);
    *a = T_a * a_b + ba;
    *w = T_g * w_b + T_s * a_b + bg;
  }

  template <typename T>
  static void
  correct(const Eigen::Matrix<T, 3, 1> &bg, const Eigen::Matrix<T, 3, 1> &ba,
          const Eigen::Matrix<T, Eigen::Dynamic, 1> &params,
          const Eigen::Matrix<T, 3, 1> &w, const Eigen::Matrix<T, 3, 1> &a,
          Eigen::Matrix<T, 3, 1> *w_b, Eigen::Matrix<T, 3, 1> *a_b) {
    Eigen::Matrix<T, 3, 3> T_g;
    vectorToMatrix<T>(params.data(), 0, &T_g);
    Eigen::Matrix<T, 3, 3> T_s;
    vectorToMatrix<T>(params.data(), 9, &T_s);
    Eigen::Matrix<T, 3, 3> T_a;
    vectorToMatrix<T>(params.data(), 18, &T_a);
    Eigen::Matrix<T, 3, 3> inv_T_g = T_g.inverse();
    Eigen::Matrix<T, 3, 3> inv_T_a = T_a.inverse();
    *a_b = inv_T_a * (a - ba);
    *w_b = inv_T_g * (w - bg - T_s * (*a_b));
  }

  static Eigen::VectorXd computeAugmentedParamsError(
      const Eigen::VectorXd& params) {
      Eigen::VectorXd residual = params;
      Eigen::Matrix<double, 9, 1> eye;
      eye << 1, 0, 0, 0, 1, 0, 0, 0, 1;
      residual.head<9>() -= eye;
      residual.tail<9>() -= eye;
      return residual;
  }

  template<size_t XBlockId>
  static void plus(const double *x, const double *inc, double *xplus) {
    Eigen::Map<const Eigen::Matrix<double, kXBlockDims[XBlockId], 1>> xm(x);
    Eigen::Map<const Eigen::Matrix<double, kXBlockMinDims[XBlockId], 1>> incm(inc);
    Eigen::Map<Eigen::Matrix<double, kXBlockDims[XBlockId], 1>> xplusm(xplus);
    xplusm = xm + incm;
  }

  template <typename T>
  void correct(const Eigen::Matrix<T, 3, 1> &w, const Eigen::Matrix<T, 3, 1> &a,
               Eigen::Matrix<T, 3, 1> *w_b, Eigen::Matrix<T, 3, 1> *a_b,
               Eigen::Matrix<T, 6, kGlobalDim> *jacobian = nullptr) const {
    *a_b = invTa_ * (a - ba_);
    *w_b = invTg_ * (w - bg_ - Ts_ * (*a_b));
    if (jacobian) {
      dwa_B_dbgbaSTS(*w_b, *a_b, *jacobian);
    }
  }

  void updateParameters(const double * bgba, double const *const * xparams) {
    bg_ = Eigen::Map<const Eigen::Matrix<double, 3, 1>>(bgba);
    ba_ = Eigen::Map<const Eigen::Matrix<double, 3, 1>>(bgba + 3);
    Eigen::Matrix<double, 3, 3> T_g;
    vectorToMatrix<double>(xparams[0], 0, &T_g);
    vectorToMatrix<double>(xparams[1], 0, &Ts_);
    Eigen::Matrix<double, 3, 3> T_a;
    vectorToMatrix<double>(xparams[2], 0, &T_a);
    invTg_ = T_g.inverse();
    invTa_ = T_a.inverse();
    invTgsa_ = invTg_ * Ts_ * invTa_;
  }

  void updateParameters(const double * bgba) {
    bg_ = Eigen::Map<const Eigen::Matrix<double, 3, 1>>(bgba);
    ba_ = Eigen::Map<const Eigen::Matrix<double, 3, 1>>(bgba + 3);

    Ts_.setZero();
    invTg_.setIdentity();
    invTa_.setIdentity();
    invTgsa_.setZero();
  }

  void updateParameters(const double * bgba, double const * xparams) {
    std::vector<const double *> xparamPtrs{xparams, xparams + 9, xparams + 18};
    updateParameters(bgba, xparamPtrs.data());
  }

  template <typename DerivedT>
  void getWeight(Eigen::MatrixBase<DerivedT> *information,
                 Eigen::MatrixBase<DerivedT> *rootInformation = nullptr) {
    P_delta_ = 0.5 * (P_delta_ + P_delta_.transpose().eval());
    information->setIdentity();
    P_delta_
        .topLeftCorner<DerivedT::RowsAtCompileTime,
                       DerivedT::ColsAtCompileTime>()
        .llt()
        .solveInPlace(*information);
    if (rootInformation) {
      Eigen::LLT<DerivedT> lltOfInfo(information->derived());
      rootInformation->derived() = lltOfInfo.matrixL().transpose();
    }
  }

  void propagate(double dt, const Eigen::Vector3d &omega_S_0,
                 const Eigen::Vector3d &acc_S_0,
                 const Eigen::Vector3d &omega_S_1,
                 const Eigen::Vector3d &acc_S_1, double sigma_g_c,
                 double sigma_a_c, double sigma_gw_c, double sigma_aw_c);

  void resetPreintegration();

  const Eigen::Quaterniond &Delta_q() const { return Delta_q_; }
  const Eigen::Vector3d &Delta_p() const { return acc_doubleintegral_; }
  const Eigen::Vector3d &Delta_v() const { return acc_integral_; }

  template <size_t XBlockId>
  Eigen::Matrix<double, 3, kXBlockDims[XBlockId]> dDp_dx() const {
    if constexpr (XBlockId == 0)
      return dp_dT_g_;
    if constexpr (XBlockId == 1)
      return dp_dT_s_;
    if constexpr (XBlockId == 2)
      return -dp_dT_a_;
  }

  template <size_t XBlockId>
  Eigen::Matrix<double, 3, kXBlockDims[XBlockId]> dDrot_dx() const {
    if constexpr (XBlockId == 0)
      return -dalpha_dT_g_;
    if constexpr (XBlockId == 1)
      return -dalpha_dT_s_;
    if constexpr (XBlockId == 2)
      return dalpha_dT_a_;
  }
  template <size_t XBlockId>
  Eigen::Matrix<double, 3, kXBlockDims[XBlockId]> dDv_dx() const {
    if constexpr (XBlockId == 0)
      return dv_dT_g_;
    if constexpr (XBlockId == 1)
      return dv_dT_s_;
    if constexpr (XBlockId == 2)
      return -dv_dT_a_;
  }

  template <size_t XBlockId>
  Eigen::Matrix<double, 3, kXBlockMinDims[XBlockId]> dDp_dminx() const {
    return dDp_dx<XBlockId>();
  }
  template <size_t XBlockId>
  Eigen::Matrix<double, 3, kXBlockMinDims[XBlockId]> dDrot_dminx() const {
    return dDrot_dx<XBlockId>();
  }
  template <size_t XBlockId>
  Eigen::Matrix<double, 3, kXBlockMinDims[XBlockId]> dDv_dminx() const {
    return dDv_dx<XBlockId>();
  }

  Eigen::Matrix<double, 3, 3> dDp_dbg() const { return dp_db_g_; }
  Eigen::Matrix<double, 3, 3> dDp_dba() const {
    return -(C_doubleintegral_ * invTa_ + dp_db_g_ * Ts_ * invTa_);
  }
  Eigen::Matrix<double, 3, 3> dDrot_dbg() const {
    return -(C_integral_ * invTg_);
  }
  Eigen::Matrix<double, 3, 3> dDrot_dba() const {
    return C_integral_ * invTgsa_;
  }
  Eigen::Matrix<double, 3, 3> dDv_dbg() const { return dv_db_g_; }
  Eigen::Matrix<double, 3, 3> dDv_dba() const {
    return -(C_integral_ * invTa_ + dv_db_g_ * Ts_ * invTa_);
  }

  const Eigen::Matrix3d &Ts() const { return Ts_; }
  const Eigen::Matrix3d &invTg() const { return invTg_; }
  const Eigen::Matrix3d &invTa() const { return invTa_; }
  const Eigen::Matrix3d &invTgsa() const { return invTgsa_; }

private:

  template <class Scalar>
  Eigen::Matrix<Scalar, 3, 6> domega_B_dbgba() const {
    Eigen::Matrix<Scalar, 3, 6> dwB_dbgba = Eigen::Matrix<Scalar, 3, 6>::Zero();
    dwB_dbgba.template block<3, 3>(0, 0) = -invTg_;
    dwB_dbgba.template block<3, 3>(0, 3) = invTg_ * Ts_ * invTa_;
    return dwB_dbgba;
  }

  template <class Scalar>
  Eigen::Matrix<Scalar, 3, 27> domega_B_dSgTsSa(
      const Eigen::Matrix<Scalar, 3, 1>& w_est,
      const Eigen::Matrix<Scalar, 3, 1>& a_est) const {
    Eigen::Matrix<Scalar, 3, 27> dwB_dSgTsSa =
        Eigen::Matrix<Scalar, 3, 27>::Zero();
    dwB_dSgTsSa.template block<3, 9>(0, 0) =
        -invTg_ * dmatrix3_dvector9_multiply(w_est);
    dwB_dSgTsSa.template block<3, 9>(0, 9) =
        -invTg_ * dmatrix3_dvector9_multiply(a_est);
    dwB_dSgTsSa.template block<3, 9>(0, 18) =
        invTg_ * Ts_ * invTa_ * dmatrix3_dvector9_multiply(a_est);
    return dwB_dSgTsSa;
  }

  template <class Scalar>
  Eigen::Matrix<Scalar, 3, 6> dacc_B_dbgba() const {
    Eigen::Matrix<Scalar, 3, 6> daB_dbgba = Eigen::Matrix<Scalar, 3, 6>::Zero();
    daB_dbgba.template block<3, 3>(0, 3) = -invTa_;
    return daB_dbgba;
  }

  template <class Scalar>
  Eigen::Matrix<Scalar, 3, 27> dacc_B_dSgTsSa(
      const Eigen::Matrix<Scalar, 3, 1>& a_est) const {
    Eigen::Matrix<Scalar, 3, 27> daB_dSgTsSa =
        Eigen::Matrix<Scalar, 3, 27>::Zero();
    daB_dSgTsSa.template block<3, 9>(0, 18) =
        -invTa_ * dmatrix3_dvector9_multiply(a_est);
    return daB_dSgTsSa;
  }

  /**
   * w_est ideal gyroscope measurement
   * a_est ideal accelerometer measurement
   */
  template <class Scalar>
  void dwa_B_dbgbaSTS(
      const Eigen::Matrix<Scalar, 3, 1>& w_est,
      const Eigen::Matrix<Scalar, 3, 1>& a_est,
      Eigen::Matrix<Scalar, 6, 6 + 27>& output) const {
    output.template block<3, 6>(0, 0) = domega_B_dbgba<Scalar>();
    output.template block<3, 6>(3, 0) = dacc_B_dbgba<Scalar>();
    output.template block<3, 27>(0, 6) = domega_B_dSgTsSa(w_est, a_est);
    output.template block<3, 27>(3, 6) = dacc_B_dSgTsSa(a_est);
  }

  Eigen::Vector3d bg_;
  Eigen::Vector3d ba_;
  Eigen::Matrix3d Ts_;
  Eigen::Matrix3d invTg_;
  Eigen::Matrix3d invTa_;
  Eigen::Matrix3d invTgsa_;

  Eigen::Quaterniond Delta_q_ = Eigen::Quaterniond(1, 0, 0, 0);  // quaternion of DCM from Si to S0
  //$\int_{t_0}^{t_i} R_S^{S_0} dt$
  Eigen::Matrix3d C_integral_ =
      Eigen::Matrix3d::Zero();  // integrated DCM up to Si expressed in S0 frame
  // $\int_{t_0}^{t_i} \int_{t_0}^{s} R_S^{S_0} dt ds$
  Eigen::Matrix3d C_doubleintegral_ =
      Eigen::Matrix3d::Zero();  // double integrated DCM up to Si expressed in
                                // S0 frame
  // $\int_{t_0}^{t_i} R_S^{S_0} a^S dt$
  Eigen::Vector3d acc_integral_ =
      Eigen::Vector3d::Zero();  // integrated acceleration up to Si expressed in
                                // S0 frame
  // $\int_{t_0}^{t_i} \int_{t_0}^{s} R_S^{S_0} a^S dt ds$
  Eigen::Vector3d acc_doubleintegral_ =
      Eigen::Vector3d::Zero();  // double integrated acceleration up to Si
                                // expressed in S0 frame
  // sub-Jacobians
  // $R_{S_0}^W \frac{d^{S_0}\alpha_{l+1}}{d b_g_{l}} = \frac{d^W\alpha_{l+1}}{d
  // b_g_{l}} $ for ease of implementation, we compute all the Jacobians with
  // positive increment, thus there can be a sign difference between, for
  // instance, dalpha_db_g and \frac{d^{S_0}\alpha_{l+1}}{d b_g_{(l)}}. This
  // difference is adjusted when putting all Jacobians together
//  Eigen::Matrix3d dalpha_db_g_ = Eigen::Matrix3d::Zero();
//  Eigen::Matrix3d dalpha_db_a_ = Eigen::Matrix3d::Zero();
  Eigen::Matrix<double, 3, 9> dalpha_dT_g_ = Eigen::Matrix<double, 3, 9>::Zero();
  Eigen::Matrix<double, 3, 9> dalpha_dT_s_ = Eigen::Matrix<double, 3, 9>::Zero();
  Eigen::Matrix<double, 3, 9> dalpha_dT_a_ = Eigen::Matrix<double, 3, 9>::Zero();

  Eigen::Matrix3d dv_db_g_ = Eigen::Matrix3d::Zero();
//  Eigen::Matrix3d dv_db_a_ = Eigen::Matrix3d::Zero();
  Eigen::Matrix<double, 3, 9> dv_dT_g_ = Eigen::Matrix<double, 3, 9>::Zero();
  Eigen::Matrix<double, 3, 9> dv_dT_s_ = Eigen::Matrix<double, 3, 9>::Zero();
  Eigen::Matrix<double, 3, 9> dv_dT_a_ = Eigen::Matrix<double, 3, 9>::Zero();

  Eigen::Matrix3d dp_db_g_ = Eigen::Matrix3d::Zero();
//  Eigen::Matrix3d dp_db_a_ = Eigen::Matrix3d::Zero();
  Eigen::Matrix<double, 3, 9> dp_dT_g_ = Eigen::Matrix<double, 3, 9>::Zero();
  Eigen::Matrix<double, 3, 9> dp_dT_s_ = Eigen::Matrix<double, 3, 9>::Zero();
  Eigen::Matrix<double, 3, 9> dp_dT_a_ = Eigen::Matrix<double, 3, 9>::Zero();

  // the Jacobian of the increment (w/o biases)
  Eigen::Matrix<double, 15, 15> P_delta_ = Eigen::Matrix<double, 15, 15>::Zero();
};

/**
 * @brief The Imu_BG_BA_MG_TS_MA class
 * The accelerometer frame is realized by the accelerometers.
 * The body frame coincides the IMU frame.
 * The gyro frame realized by the gyros has a relative rotation to the accelerometer frame.
 *
 * w_m = M_g^{-1} * w_B + T_s * a_B + b_w + n_w
 * a_m = M_a^{-1} * a_B + b_a + n_a
 * M_a is a lower triangular matrix, M_g and T_s are fully populated 3x3 matrices.
 * Therefore,
 * w_B = M_g * (w_m - b_w - T_s * (a_m - b_a))
 * a_B = M_a * (a_m - b_a)
 */
class Imu_BG_BA_MG_TS_MA {
public:
 EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 static const int kModelId = 1;
 static const size_t kAugmentedDim = 24;
 static const size_t kAugmentedMinDim = 24;
 static const size_t kGlobalDim = kAugmentedDim + kBgBaDim;

 static constexpr std::array<int, 3> kXBlockDims{9, 9, 6};  // Mg, Ts, Ma
 static constexpr std::array<int, 3> kXBlockMinDims{9, 9, 6};
 static constexpr std::array<int, 4> kCumXBlockDims{0, 9, 18, 24};
 static constexpr std::array<int, 4> kCumXBlockMinDims{0, 9, 18, 24};
 static constexpr double kJacobianTolerance = 7.0e-3;

 static inline int getAugmentedDim() { return kAugmentedDim; }
 static inline int getMinimalDim() { return kGlobalDim; }
 static inline int getAugmentedMinimalDim() { return kAugmentedDim; }
 template <typename T>
 static Eigen::Matrix<T, kAugmentedDim, 1> getNominalAugmentedParams() {
   Eigen::Matrix<T, 9, 1> eye;
   eye << 1, 0, 0, 0, 1, 0, 0, 0, 1;
   Eigen::Matrix<T, kAugmentedDim, 1> augmentedParams;
   augmentedParams.template head<9>() = eye;
   augmentedParams.template segment<9>(9).setZero();
   Eigen::Matrix<T, 3, 3> identity = Eigen::Matrix<T, 3, 3>::Identity();
   lowerTriangularMatrixToVector(identity, augmentedParams.data(), 18);
   return augmentedParams;
 }

 static Eigen::VectorXd computeAugmentedParamsError(
     const Eigen::VectorXd& params) {
     Eigen::VectorXd residual = params;
     Eigen::Matrix<double, 9, 1> eye;
     eye << 1, 0, 0, 0, 1, 0, 0, 0, 1;
     residual.head<9>() -= eye;
     Eigen::Matrix<double, 6, 1> lowerTriangularMat;
     Eigen::Matrix3d identity = Eigen::Matrix3d::Identity();
     lowerTriangularMatrixToVector(identity, lowerTriangularMat.data(), 0);
     residual.tail<6>() -= lowerTriangularMat;
     return residual;
 }

 template<size_t XBlockId>
 static void plus(const double *x, const double *inc, double *xplus) {
   Eigen::Map<const Eigen::Matrix<double, kXBlockDims[XBlockId], 1>> xm(x);
   Eigen::Map<const Eigen::Matrix<double, kXBlockMinDims[XBlockId], 1>> incm(inc);
   Eigen::Map<Eigen::Matrix<double, kXBlockDims[XBlockId], 1>> xplusm(xplus);
   xplusm = xm + incm;
 }

 template <typename T>
 void correct(const Eigen::Matrix<T, 3, 1> &w, const Eigen::Matrix<T, 3, 1> &a,
              Eigen::Matrix<T, 3, 1> *w_b, Eigen::Matrix<T, 3, 1> *a_b,
              Eigen::Matrix<T, 6, kGlobalDim> *jacobian = nullptr) const {
   Eigen::Matrix<T, 3, 1> alternatea = a - ba_;
   *a_b = Ma_ * alternatea;
   Eigen::Matrix<T, 3, 1> alternatew = w - bg_ - Ts_ * (a - ba_);
   *w_b = Mg_ * alternatew;
   if (jacobian) {
     // w_b relative to bg ba Mg Ts Ma
     jacobian->template block<3, 3>(0, 0) = -Mg_;
     jacobian->template block<3, 3>(0, 3) = Mg_ * Ts_;
     jacobian->template block<3, 9>(0, 6) = dmatrix3_dvector9_multiply(alternatew);
     jacobian->template block<3, 9>(0, 15) = -Mg_ * dmatrix3_dvector9_multiply(alternatea);
     jacobian->template block<3, 6>(0, 24).setZero();

     // a_b relative to bg ba Mg Ts Ma
     jacobian->template block<3, 3>(3, 0).setZero();
     jacobian->template block<3, 3>(3, 3) = -Ma_;
     jacobian->template block<3, 18>(3, 6).setZero();
     jacobian->template block<3, 6>(3, 24) = dltm3_dvector6_multiply(alternatea);
   }
 }

 void updateParameters(const double * bgba, double const *const * xparams) {
   bg_ = Eigen::Map<const Eigen::Matrix<double, 3, 1>>(bgba);
   ba_ = Eigen::Map<const Eigen::Matrix<double, 3, 1>>(bgba + 3);
   vectorToMatrix<double>(xparams[0], 0, &Mg_);
   vectorToMatrix<double>(xparams[1], 0, &Ts_);
   vectorToLowerTriangularMatrix<double>(xparams[2], 0, &Ma_);
   MgTs_ = Mg_ * Ts_;
 }

 void updateParameters(const double * bgba, double const * xparams) {
   std::vector<const double *> xparamPtrs{xparams, xparams + 9, xparams + 18};
   updateParameters(bgba, xparamPtrs.data());
 }

 void updateParameters(const double * bgba) {
   bg_ = Eigen::Map<const Eigen::Matrix<double, 3, 1>>(bgba);
   ba_ = Eigen::Map<const Eigen::Matrix<double, 3, 1>>(bgba + 3);
   Mg_.setIdentity();
   Ts_.setZero();
   Ma_.setIdentity();
   MgTs_.setZero();
 }

 template <typename DerivedT>
 void getWeight(Eigen::MatrixBase<DerivedT> *information,
                Eigen::MatrixBase<DerivedT> *rootInformation = nullptr) {
   P_delta_ = 0.5 * (P_delta_ + P_delta_.transpose().eval());
   information->setIdentity();
   P_delta_
       .topLeftCorner<DerivedT::RowsAtCompileTime,
                      DerivedT::ColsAtCompileTime>()
       .llt()
       .solveInPlace(*information);
   if (rootInformation) {
     Eigen::LLT<DerivedT> lltOfInfo(information->derived());
     rootInformation->derived() = lltOfInfo.matrixL().transpose();
   }
 }

 /**
  * @brief one step preintegration of IMU measurements.
  * @param dt
  * @param omega_S_0
  * @param acc_S_0
  * @param omega_S_1
  * @param acc_S_1
  * @param sigma_g_c
  * @param sigma_a_c
  * @param sigma_gw_c
  * @param sigma_aw_c
  */
 void propagate(double dt, const Eigen::Vector3d &omega_S_0,
                const Eigen::Vector3d &acc_S_0,
                const Eigen::Vector3d &omega_S_1,
                const Eigen::Vector3d &acc_S_1, double sigma_g_c,
                double sigma_a_c, double sigma_gw_c, double sigma_aw_c);

 /// @name methods for propagation starting from a known state and covariance.
 /// @{
 void setInitialStateAndCov(const okvis::kinematics::Transformation &T_WS0,
                            const Eigen::Vector3d &v_WS0, const Eigen::Vector3d &g_W,
                            const Eigen::MatrixXd &cov) {
   T_WS0_ = T_WS0;
   v_WS0_ = v_WS0;
   g_W_ = g_W;
   P_ = cov;
 }

 void setPosVelLinPoint(const Eigen::Matrix<double, 6, 1> &posVelLinPoint) {
   posVelLinPoint_ = posVelLinPoint;
 }

 /**
  * @brief computeDeltaState
  * @param dt
  * @param omega_S_0
  * @param acc_S_0
  * @param omega_S_1
  * @param acc_S_1
  */
 void computeDeltaState(double dt, const Eigen::Vector3d &omega_S_0,
                        const Eigen::Vector3d &acc_S_0,
                        const Eigen::Vector3d &omega_S_1,
                        const Eigen::Vector3d &acc_S_1);

 void computeFdelta(double Delta_t, double dt,
                    const NormalVectorElement &normalGravity, double gravityNorm,
                    bool estimate_gravity_direction);

 void updatePdelta(double dt, double sigma_g_c, double sigma_a_c, double sigma_gw_c, double sigma_aw_c);

 void shiftVariables();

 void getFinalState(double Delta_t, okvis::kinematics::Transformation *T_WS, Eigen::Vector3d *v_WS);

 void getFinalJacobian(Eigen::MatrixXd *jacobian, double Delta_t, const Eigen::Matrix<double, 6, 1> &posVelLinPoint,
                       const NormalVectorElement &normalGravity, double gravityNorm,
                       bool estimate_gravity_direction);

 void getFinalCovariance(Eigen::MatrixXd *covariance, const Eigen::Matrix3d &C_WS0);

 /// @}

 void resetPreintegration();

 const Eigen::Quaterniond &Delta_q() const { return Delta_q_; }
 const Eigen::Vector3d &Delta_p() const { return acc_doubleintegral_; }
 const Eigen::Vector3d &Delta_v() const { return acc_integral_; }

 template <size_t XBlockId>
 Eigen::Matrix<double, 3, kXBlockDims[XBlockId]> dDp_dx() const {
   if constexpr (XBlockId == 0)
     return dp_dM_g_;
   if constexpr (XBlockId == 1)
     return dp_dT_s_;
   if constexpr (XBlockId == 2)
     return dp_dM_a_;
 }

 template <size_t XBlockId>
 Eigen::Matrix<double, 3, kXBlockDims[XBlockId]> dDrot_dx() const {
   if constexpr (XBlockId == 0)
     return dalpha_dM_g_;
   if constexpr (XBlockId == 1)
     return dalpha_dT_s_;
   if constexpr (XBlockId == 2)
     return dalpha_dM_a_;
 }

 template <size_t XBlockId>
 Eigen::Matrix<double, 3, kXBlockDims[XBlockId]> dDv_dx() const {
   if constexpr (XBlockId == 0)
     return dv_dM_g_;
   if constexpr (XBlockId == 1)
     return dv_dT_s_;
   if constexpr (XBlockId == 2)
     return dv_dM_a_;
 }

 template <size_t XBlockId>
 Eigen::Matrix<double, 3, kXBlockMinDims[XBlockId]> dDp_dminx() const {
   return dDp_dx<XBlockId>();
 }
 template <size_t XBlockId>
 Eigen::Matrix<double, 3, kXBlockMinDims[XBlockId]> dDrot_dminx() const {
   return dDrot_dx<XBlockId>();
 }
 template <size_t XBlockId>
 Eigen::Matrix<double, 3, kXBlockMinDims[XBlockId]> dDv_dminx() const {
   return dDv_dx<XBlockId>();
 }

 Eigen::Matrix<double, 3, 3> dDp_dbg() const { return dp_db_g_; }
 Eigen::Matrix<double, 3, 3> dDp_dba() const {
   return -(C_doubleintegral_ * Ma_ + dp_db_g_ * Ts_);
 }
 Eigen::Matrix<double, 3, 3> dDrot_dbg() const {
   return -(C_integral_ * Mg_);
 }
 Eigen::Matrix<double, 3, 3> dDrot_dba() const {
   return C_integral_ * MgTs_;
 }
 Eigen::Matrix<double, 3, 3> dDv_dbg() const { return dv_db_g_; }
 Eigen::Matrix<double, 3, 3> dDv_dba() const {
   return -(C_integral_ * Ma_ + dv_db_g_ * Ts_);
 }

private:
 Eigen::Vector3d bg_;
 Eigen::Vector3d ba_;
 Eigen::Matrix3d Ts_;
 Eigen::Matrix3d Mg_;
 Eigen::Matrix3d Ma_;
 Eigen::Matrix3d MgTs_;

 Eigen::Quaterniond Delta_q_ = Eigen::Quaterniond(1, 0, 0, 0);  // quaternion of DCM from Si to S0
 //$\int_{t_0}^{t_i} R_S^{S_0} dt$
 Eigen::Matrix3d C_integral_ =
     Eigen::Matrix3d::Zero();  // integrated DCM up to Si expressed in S0 frame
 // $\int_{t_0}^{t_i} \int_{t_0}^{s} R_S^{S_0} dt ds$
 Eigen::Matrix3d C_doubleintegral_ =
     Eigen::Matrix3d::Zero();  // double integrated DCM up to Si expressed in
                               // S0 frame
 // $\int_{t_0}^{t_i} R_S^{S_0} a^S dt$
 Eigen::Vector3d acc_integral_ =
     Eigen::Vector3d::Zero();  // integrated acceleration up to Si expressed in
                               // S0 frame
 // $\int_{t_0}^{t_i} \int_{t_0}^{s} R_S^{S_0} a^S dt ds$
 Eigen::Vector3d acc_doubleintegral_ =
     Eigen::Vector3d::Zero();  // double integrated acceleration up to Si
                               // expressed in S0 frame
 // sub-Jacobians
 // $R_{S_0}^W \frac{d^{S_0}\alpha_{l+1}}{d b_g_{l}} = \frac{d^W\alpha_{l+1}}{d
 // b_g_{l}} $
//  Eigen::Matrix3d dalpha_db_g_ = Eigen::Matrix3d::Zero();
//  Eigen::Matrix3d dalpha_db_a_ = Eigen::Matrix3d::Zero();
 Eigen::Matrix<double, 3, 9> dalpha_dM_g_ = Eigen::Matrix<double, 3, 9>::Zero();
 Eigen::Matrix<double, 3, 9> dalpha_dT_s_ = Eigen::Matrix<double, 3, 9>::Zero();
 Eigen::Matrix<double, 3, 6> dalpha_dM_a_ = Eigen::Matrix<double, 3, 6>::Zero();

 Eigen::Matrix3d dv_db_g_ = Eigen::Matrix3d::Zero();
//  Eigen::Matrix3d dv_db_a_ = Eigen::Matrix3d::Zero();
 Eigen::Matrix<double, 3, 9> dv_dM_g_ = Eigen::Matrix<double, 3, 9>::Zero();
 Eigen::Matrix<double, 3, 9> dv_dT_s_ = Eigen::Matrix<double, 3, 9>::Zero();
 Eigen::Matrix<double, 3, 6> dv_dM_a_ = Eigen::Matrix<double, 3, 6>::Zero();

 Eigen::Matrix3d dp_db_g_ = Eigen::Matrix3d::Zero();
//  Eigen::Matrix3d dp_db_a_ = Eigen::Matrix3d::Zero();
 Eigen::Matrix<double, 3, 9> dp_dM_g_ = Eigen::Matrix<double, 3, 9>::Zero();
 Eigen::Matrix<double, 3, 9> dp_dT_s_ = Eigen::Matrix<double, 3, 9>::Zero();
 Eigen::Matrix<double, 3, 6> dp_dM_a_ = Eigen::Matrix<double, 3, 6>::Zero();
 // the Jacobian of the increment (w/o biases)
 Eigen::Matrix<double, 15, 15> P_delta_ = Eigen::Matrix<double, 15, 15>::Zero();

 /// @name members for IMU propagation from a known state and covariance.
 /// @{
 Eigen::Matrix<double, 6, 1> posVelLinPoint_;
 Eigen::Matrix<double, 6, 1> posVelLinPoint_1_;
 okvis::kinematics::Transformation T_WS0_;
 Eigen::Vector3d v_WS0_;
 Eigen::Vector3d g_W_;
 Eigen::Vector3d acc_integral_1_;
 Eigen::Matrix3d C_;   // DCM from Si to S0
 Eigen::Matrix3d C_1_;  // DCM from S_{i+1} to S0
 Eigen::Quaterniond Delta_q_1_;
 Eigen::Matrix3d C_integral_1_;

 Eigen::Vector3d acc_est_1_;
 Eigen::Vector3d acc_est_;
 Eigen::Vector3d omega_est_1_;
 Eigen::Vector3d omega_est_;

 Eigen::Vector3d acc_nobias_0_;
 Eigen::Vector3d acc_nobias_1_;
 Eigen::Vector3d omega_nobias_0_;
 Eigen::Vector3d omega_nobias_1_;

 Eigen::Matrix<double, 3, 9> dalpha_dM_g_1_;
 Eigen::Matrix<double, 3, 9> dalpha_dT_s_1_;

 Eigen::Matrix3d dv_db_g_1_;
 Eigen::Matrix<double, 3, 9> dv_dM_g_1_;
 Eigen::Matrix<double, 3, 9> dv_dT_s_1_;
 Eigen::Matrix<double, 3, 6> dv_dM_a_1_;

 Eigen::MatrixXd F_delta_; // \f$ \frac{\partial (p_{S0S_{i+1}}, q_{S0S_{i+1}}, v_{S0S_{i+1}}, b_g, b_a, M_g, T_s, M_a)}{\partial
 // (p_{S0S_{i}}, q_{S0S_{i}}, v_{S0S_{i}}, b_g, b_a, M_g, T_s, M_a)} \f$
 Eigen::MatrixXd P_; // \f$ cov(p_{S0S_{i+1}}, q_{S0S_{i+1}}, v_{S0S_{i+1}}, b_g, b_a, M_g, T_s, M_a) \f$
 bool computeJacobian = true;
 const bool use_first_estimate = true;
 /// @}
};


/**
 * @brief The ScaledMisalignedImu class
 * The body frame is the same as the classic IMU sensor frame. So the gyroscope triad
 * needs to consider scaling effect(3), misalignment(3), and relative 
 * orientation(4, minimal 3) to the IMU sensor frame, and g-sensitivity (9)
 * whereas the accelerometer triad needs to consider scaling effect (3) and
 * misalignment (3). The lever arm(size) effects are ignored.
 * Implemented according to "Extending Kalibr".
 */
class ScaledMisalignedImu {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  static const int kModelId = 2;
  static const size_t kSMDim = 6;
  static const size_t kSensitivityDim = 9;
  static const size_t kAugmentedDim = kSMDim + kSensitivityDim + kSMDim + 4;
  static const size_t kAugmentedMinDim = kSMDim + kSensitivityDim + kSMDim + 3;
  static const size_t kGlobalDim = kAugmentedDim + kBgBaDim;
  static constexpr std::array<int, 4> kXBlockDims{kSMDim, kSensitivityDim, kSMDim, 4};  // M_gyro, M_accel_gyro, M_accel, q_gyro_i
  static constexpr std::array<int, 4> kXBlockMinDims{kSMDim, kSensitivityDim, kSMDim, 3};
  static constexpr std::array<int, 5> kCumXBlockDims{0, kSMDim, kSMDim + kSensitivityDim, kSMDim + kSensitivityDim + kSMDim, kSMDim + kSensitivityDim + kSMDim + 4};
  static constexpr std::array<int, 5> kCumXBlockMinDims{0, kSMDim, kSMDim + kSensitivityDim, kSMDim + kSensitivityDim + kSMDim, kSMDim + kSensitivityDim + kSMDim + 3};
  static constexpr double kJacobianTolerance = 5.0e-3;

  static inline int getAugmentedDim() { return kAugmentedDim; }
  static inline int getMinimalDim() { return kGlobalDim - 1; }
  static inline int getAugmentedMinimalDim() { return kAugmentedDim - 1; }
  template <typename T>
  static Eigen::Matrix<T, kAugmentedDim, 1> getNominalAugmentedParams() {
    Eigen::Matrix<T, kAugmentedDim, 1> nominalValues = Eigen::Matrix<T, kAugmentedDim, 1>::Zero();
    nominalValues[0] = T(1.0);
    nominalValues[2] = T(1.0);
    nominalValues[5] = T(1.0);
    nominalValues[6 + 9] = T(1.0);
    nominalValues[6 + 9 + 2] = T(1.0);
    nominalValues[6 + 9 + 5] = T(1.0);
    nominalValues[kAugmentedDim - 1] = T(1.0);  // quaternion in xyzw format for R_gyro_i.
    return nominalValues;
  }

  /**
   * nearly 1:1 implementation of
   * https://github.com/ethz-asl/kalibr/blob/master/aslam_offline_calibration/kalibr/python/kalibr_imu_camera_calibration/IccSensors.py#L1033-L1049
   * w_b angular velocity in body frame at time tk.
   * w_dot_b angular acceleration in body frame at time tk.
   * a_w linear acceleration at tk.
   * r_b acceleration triad origin, i.e., the sensor frame origin, coordinates expressed in the body frame.
   * params gyro bias, accelerometer bias, gyro Scaling*Misalignment, gyro g-sensitivity, accelerometer Scaling*Misalignment.
   * C_gyro_i the relative orientation from the accelerometer triad frame, i.e., the IMU sensor frame to the gyro triad frame.
   */
  template <typename T>
  static void predictAngularVelocity(
      const Eigen::Matrix<T, 3, 1> &bg, const Eigen::Matrix<T, 3, 1> & /*ba*/,
      const Eigen::Matrix<T, Eigen::Dynamic, 1> &params,
      const Eigen::Quaternion<T> &q_w_b, const Eigen::Matrix<T, 3, 1> &w_b,
      const Eigen::Matrix<T, 3, 1> &a_w, const Eigen::Matrix<T, 3, 1> &g_w,
      Eigen::Matrix<T, 3, 1> *w) {
    Eigen::Matrix<T, 3, 1> w_dot_b = Eigen::Matrix<T, 3, 1>::Zero();
    Eigen::Matrix<T, 3, 3> C_b_w =
        q_w_b.template toRotationMatrix().transpose();
    Eigen::Matrix<T, 3, 1> r_b =
        Eigen::Matrix<T, 3, 1>::Zero(); // Assume the 3 accelerometers are at
                                        // the origin of the body frame.
    Eigen::Matrix<T, 3, 1> a_b =
        C_b_w * (a_w - g_w) + w_dot_b.cross(r_b) + w_b.cross(w_b.cross(r_b));

    Eigen::Matrix<T, 3, 3> C_i_b =
        Eigen::Matrix<T, 3, 3>::Identity(); // The IMU sensor frame coincides
                                            // the accelerometer triad frame.
    Eigen::Map<const Eigen::Quaternion<T>> q_gyro_i(params.data() +
                                                    kAugmentedDim - 4);
    Eigen::Matrix<T, 3, 3> C_gyro_i = q_gyro_i.template toRotationMatrix();
    Eigen::Matrix<T, 3, 3> C_gyro_b = C_gyro_i * C_i_b;

    Eigen::Matrix<T, 3, 3> M_gyro;
    vectorToLowerTriangularMatrix<T>(params.data(), 0, &M_gyro);
    Eigen::Matrix<T, 3, 3> M_accel_gyro;
    vectorToMatrix<T>(params.data(), kSMDim, &M_accel_gyro);
    *w = M_gyro * (C_gyro_b * w_b) + M_accel_gyro * (C_gyro_b * a_b) + bg;
  }

  /**
   * nearly 1:1 implementation of
   * https://github.com/ethz-asl/kalibr/blob/master/aslam_offline_calibration/kalibr/python/kalibr_imu_camera_calibration/IccSensors.py#L989-L1000
   */
  template <typename T>
  static void predictLinearAcceleration(
      const Eigen::Matrix<T, 3, 1> &/*bg*/, const Eigen::Matrix<T, 3, 1> &ba,
      const Eigen::Matrix<T, Eigen::Dynamic, 1> &params,
      const Eigen::Quaternion<T> &q_w_b, const Eigen::Matrix<T, 3, 1> &w_b,
      const Eigen::Matrix<T, 3, 1> &a_w, const Eigen::Matrix<T, 3, 1> &g_w,
      Eigen::Matrix<T, 3, 1> *a) {
    Eigen::Matrix<T, 3, 1> w_dot_b = Eigen::Matrix<T, 3, 1>::Zero();
    Eigen::Matrix<T, 3, 3> C_b_w =
        q_w_b.template toRotationMatrix().transpose();

    Eigen::Matrix<T, 3, 3> M_accel;
    vectorToLowerTriangularMatrix<T>(params.data(), kSMDim + kSensitivityDim,
                                     &M_accel);

    Eigen::Matrix<T, 3, 1> r_b =
        Eigen::Matrix<T, 3, 1>::Zero(); // Assume the 3 accelerometers are at
                                        // the origin of the body frame.
    Eigen::Matrix<T, 3, 1> a_b =
        C_b_w * (a_w - g_w) + w_dot_b.cross(r_b) + w_b.cross(w_b.cross(r_b));

    Eigen::Matrix<T, 3, 3> C_i_b =
        Eigen::Matrix<T, 3, 3>::Identity(); // The IMU sensor frame coincides
                                            // the accelerometer triad frame.
    *a = M_accel * (C_i_b * a_b) + ba;
  }

  template <typename T>
  static void
  predict(const Eigen::Matrix<T, 3, 1> &bg, const Eigen::Matrix<T, 3, 1> &ba,
          const Eigen::Matrix<T, Eigen::Dynamic, 1> &params,
          const Eigen::Matrix<T, 3, 1> &w_b, const Eigen::Matrix<T, 3, 1> &a_b,
          Eigen::Matrix<T, 3, 1> *w, Eigen::Matrix<T, 3, 1> *a) {
    Eigen::Matrix<T, 3, 3> M_accel;
    vectorToLowerTriangularMatrix<T>(params.data(), kSMDim + kSensitivityDim,
                                     &M_accel);
    Eigen::Matrix<T, 3, 3> C_i_b =
        Eigen::Matrix<T, 3, 3>::Identity(); // The IMU sensor frame coincides
                                            // the accelerometer triad frame.
    *a = M_accel * (C_i_b * a_b) + ba;

    Eigen::Map<const Eigen::Quaternion<T>> q_gyro_i(params.data() +
                                                    kAugmentedDim - 4);
    Eigen::Matrix<T, 3, 3> C_gyro_i = q_gyro_i.toRotationMatrix();
    Eigen::Matrix<T, 3, 3> C_gyro_b = C_gyro_i * C_i_b;

    Eigen::Matrix<T, 3, 3> M_gyro;
    vectorToLowerTriangularMatrix<T>(params.data(), 0, &M_gyro);
    Eigen::Matrix<T, 3, 3> M_accel_gyro;
    vectorToMatrix<T>(params.data(), kSMDim, &M_accel_gyro);
    *w = M_gyro * (C_gyro_b * w_b) + M_accel_gyro * (C_gyro_b * a_b) + bg;
  }

  template <typename T>
  static void
  correct(const Eigen::Matrix<T, 3, 1> &bg, const Eigen::Matrix<T, 3, 1> &ba,
          const Eigen::Matrix<T, Eigen::Dynamic, 1> &params,
          const Eigen::Matrix<T, 3, 1> &w, const Eigen::Matrix<T, 3, 1> &a,
          Eigen::Matrix<T, 3, 1> *w_b, Eigen::Matrix<T, 3, 1> *a_b) {
    Eigen::Matrix<T, 3, 3> M_accel_inv;
    invertLowerTriangularMatrix<T>(params.data(), kSMDim + kSensitivityDim,
                                   &M_accel_inv);
    Eigen::Matrix<T, 3, 3> C_b_i =
        Eigen::Matrix<T, 3, 3>::Identity(); // The IMU sensor frame coincides
                                            // the accelerometer triad frame.
    *a_b = C_b_i * M_accel_inv * (a - ba);

    Eigen::Matrix<T, 3, 3> C_i_b = C_b_i.transpose();
    Eigen::Map<const Eigen::Quaternion<T>> q_gyro_i(params.data() +
                                                    kAugmentedDim - 4);
    Eigen::Matrix<T, 3, 3> C_gyro_i = q_gyro_i.toRotationMatrix();
    Eigen::Matrix<T, 3, 3> C_gyro_b = C_gyro_i * C_i_b;

    Eigen::Matrix<T, 3, 3> M_gyro_inv;
    invertLowerTriangularMatrix<T>(params.data(), 0, &M_gyro_inv);
    Eigen::Matrix<T, 3, 3> M_accel_gyro;
    vectorToMatrix<T>(params.data(), kSMDim, &M_accel_gyro);
    *w_b = C_gyro_b.transpose() *
           (M_gyro_inv * (w - bg - M_accel_gyro * (C_gyro_b * (*a_b))));
  }

  template <typename T>
  void correct(const Eigen::Matrix<T, 3, 1> &w, const Eigen::Matrix<T, 3, 1> &a,
               Eigen::Matrix<T, 3, 1> *w_b, Eigen::Matrix<T, 3, 1> *a_b,
               Eigen::Matrix<T, 6, 6 + kAugmentedMinDim> *jacobian = nullptr) const {
    *a_b = C_b_i_ * M_accel_inv_ * (a - ba_);
    Eigen::Matrix<T, 3, 3> C_gyro_i = q_gyro_i_.toRotationMatrix();
    Eigen::Matrix<T, 3, 3> C_gyro_b = C_gyro_i * C_i_b_;
    *w_b = C_gyro_b.transpose() *
           (M_gyro_inv_ * (w - bg_ - M_accel_gyro_ * (C_gyro_b * (*a_b))));
    if (jacobian) {
      throw std::runtime_error("Jacobian for ScaledMisalignedImu::correct() is not implemented!");
    }
  }

  static Eigen::VectorXd
  computeAugmentedParamsError(const Eigen::VectorXd &params) {
    Eigen::VectorXd residual(getAugmentedMinimalDim());
    Eigen::VectorXd nominalValues = getNominalAugmentedParams<double>();
    constexpr int kAugmentedEuclideanDim = kAugmentedDim - 4;
    Eigen::Map<const Eigen::Quaterniond> q_g_i(nominalValues.data() +
                                               kAugmentedEuclideanDim);
    residual.head<kAugmentedEuclideanDim>() =
        params.head<kAugmentedEuclideanDim>() -
        nominalValues.head<kAugmentedEuclideanDim>();
    Eigen::Map<const Eigen::Quaterniond> q_g_i_hat(params.data() + kAugmentedEuclideanDim);
    residual.tail<3>() = (q_g_i * q_g_i_hat.conjugate()).coeffs().head<3>() * 2;
    return residual;
  }

  template<size_t XBlockId, std::enable_if_t<XBlockId != 4, bool> = true>
  static void plus(const double *x, const double *inc, double *xplus) {
    Eigen::Map<const Eigen::Matrix<double, kXBlockDims[XBlockId], 1>> xm(x);
    Eigen::Map<const Eigen::Matrix<double, kXBlockMinDims[XBlockId], 1>> incm(inc);
    Eigen::Map<Eigen::Matrix<double, kXBlockDims[XBlockId], 1>> xplusm(xplus);
    xplusm = xm + incm;
  }

  template<size_t XBlockId, std::enable_if_t<XBlockId == 4, bool> = true>
  static void plus(const double *x, const double *inc, double *xplus) {
    Eigen::Map<const Eigen::Quaterniond> xm(x);
    Eigen::Map<const Eigen::Matrix<double, kXBlockMinDims[XBlockId], 1>> incm(inc);
    Eigen::Map<Eigen::Quaterniond> xplusm(xplus);
    Eigen::Quaterniond dq = okvis::kinematics::expAndTheta(incm);
    xplusm = dq * xm;
  }

  void updateParameters(const double *bgba, double const *const *xparams) {
    bg_ = Eigen::Map<const Eigen::Matrix<double, 3, 1>>(bgba);
    ba_ = Eigen::Map<const Eigen::Matrix<double, 3, 1>>(bgba + 3);
    vectorToLowerTriangularMatrix(xparams[0], 0, &M_gyro_);
    vectorToMatrix(xparams[1], 0, &M_accel_gyro_);
    vectorToLowerTriangularMatrix(xparams[2], 0, &M_accel_);
    q_gyro_i_ = Eigen::Map<const Eigen::Quaterniond>(xparams[3]);
    invertLowerTriangularMatrix(xparams[0], 0, &M_gyro_inv_);
    invertLowerTriangularMatrix(xparams[2], 0, &M_accel_inv_);
  }

  void updateParameters(const double * bgba, double const * xparams) {
    std::vector<const double *> xparamPtrs{xparams, xparams + 6, xparams + 15, xparams + 21};
    updateParameters(bgba, xparamPtrs.data());
  }

  void propagate(double dt, const Eigen::Vector3d &omega_S_0,
                 const Eigen::Vector3d &acc_S_0,
                 const Eigen::Vector3d &omega_S_1,
                 const Eigen::Vector3d &acc_S_1, double sigma_g_c,
                 double sigma_a_c, double sigma_gw_c, double sigma_aw_c);

  void resetPreintegration();

 private:
  Eigen::Vector3d bg_;
  Eigen::Vector3d ba_;
  Eigen::Matrix3d M_gyro_;
  Eigen::Matrix3d M_gyro_inv_;
  Eigen::Matrix3d M_accel_gyro_;
  Eigen::Matrix3d M_accel_;
  Eigen::Matrix3d M_accel_inv_;
  Eigen::Quaterniond q_gyro_i_;

  Eigen::Matrix<double, 3, 3> C_b_i_ = Eigen::Matrix<double, 3, 3>::Identity();
  Eigen::Matrix<double, 3, 3> C_i_b_ = Eigen::Matrix<double, 3, 3>::Identity();

};

#ifndef IMU_ERROR_MODEL_CASES
#define IMU_ERROR_MODEL_CASES                                                  \
  IMU_ERROR_MODEL_CASE(Imu_BG_BA)                                              \
  IMU_ERROR_MODEL_CASE(Imu_BG_BA_MG_TS_MA)                                     \
  IMU_ERROR_MODEL_CASE(Imu_BG_BA_TG_TS_TA)                                     \
  IMU_ERROR_MODEL_CASE(ScaledMisalignedImu)
#endif

inline int ImuModelGetMinimalDim(int model_id) {
  switch (model_id) {
#define MODEL_CASES IMU_ERROR_MODEL_CASES
#define IMU_ERROR_MODEL_CASE(ImuModel) \
  case ImuModel::kModelId:             \
    return ImuModel::getMinimalDim();

    MODEL_SWITCH_CASES

#undef IMU_ERROR_MODEL_CASE
#undef MODEL_CASES
  }
  return 0;
}

inline int ImuModelGetAugmentedDim(int model_id) {
  switch (model_id) {
#define MODEL_CASES IMU_ERROR_MODEL_CASES
#define IMU_ERROR_MODEL_CASE(ImuModel) \
  case ImuModel::kModelId:             \
    return ImuModel::getAugmentedDim();

    MODEL_SWITCH_CASES

#undef IMU_ERROR_MODEL_CASE
#undef MODEL_CASES
  }
  return 0;
}

inline int ImuModelGetAugmentedMinimalDim(int model_id) {
  switch (model_id) {
#define MODEL_CASES IMU_ERROR_MODEL_CASES
#define IMU_ERROR_MODEL_CASE(ImuModel) \
  case ImuModel::kModelId:             \
    return ImuModel::getAugmentedMinimalDim();

    MODEL_SWITCH_CASES

#undef IMU_ERROR_MODEL_CASE
#undef MODEL_CASES
  }
  return 0;
}

inline void ImuModelToAugmentedDesiredStdevs(const int imu_model,
                                             Eigen::VectorXd *stdevs) {
  int index = 0;
  switch (imu_model) {
  case Imu_BG_BA_TG_TS_TA::kModelId:
    stdevs->resize(27);
    for (int i = 0; i < 9; ++i) {
      (*stdevs)[i] = 4e-3;
    }
    index = 9;
    for (int i = 0; i < 9; ++i) {
      (*stdevs)[i + index] = 1e-3;
    }
    index += 9;
    for (int i = 0; i < 9; ++i) {
      (*stdevs)[i + index] = 5e-3;
    }
    break;
  case Imu_BG_BA_MG_TS_MA::kModelId:
    stdevs->resize(24);
    for (int i = 0; i < 9; ++i) {
      (*stdevs)[i] = 4e-3;
    }
    index = 9;
    for (int i = 0; i < 9; ++i) {
      (*stdevs)[i + index] = 1e-3;
    }
    index += 9;
    for (int i = 0; i < 6; ++i) {
      (*stdevs)[i + index] = 5e-3;
    }
    break;
  case ScaledMisalignedImu::kModelId:
    stdevs->resize(24);
    for (int i = 0; i < 6; ++i) {
      (*stdevs)[i] = 4e-3;
    }
    index += 6;
    for (int i = 0; i < 9; ++i) {
      (*stdevs)[i + index] = 1e-3;
    }
    index += 9;
    for (int i = 0; i < 6; ++i) {
      (*stdevs)[i + index] = 5e-3;
    }
    index += 6;
    for (int i = 0; i < 3; ++i) {
      (*stdevs)[i + index] = 5e-3;
    }
    index += 3;
    break;
  case Imu_BG_BA::kModelId:
  default:
    stdevs->resize(0);
    break;
  }
}

inline void
ImuModelToMinimalAugmentedDimensionLabels(const int imu_model,
                                          std::vector<std::string> *labels) {
  std::vector<std::string> extraLabels;
  switch (imu_model) {
  case Imu_BG_BA_MG_TS_MA::kModelId:
    extraLabels = {"Mg_1", "Mg_2", "Mg_3", "Mg_4", "Mg_5", "Mg_6", "Mg_7",
                   "Mg_8", "Mg_9", "Ts_1", "Ts_2", "Ts_3", "Ts_4", "Ts_5",
                   "Ts_6", "Ts_7", "Ts_8", "Ts_9", "Ma_1", "Ma_2", "Ma_3",
                   "Ma_4", "Ma_5", "Ma_6"};
    break;
  case Imu_BG_BA_TG_TS_TA::kModelId:
    extraLabels = {"Tg_1", "Tg_2", "Tg_3", "Tg_4", "Tg_5", "Tg_6", "Tg_7",
                   "Tg_8", "Tg_9", "Ts_1", "Ts_2", "Ts_3", "Ts_4", "Ts_5",
                   "Ts_6", "Ts_7", "Ts_8", "Ts_9", "Ta_1", "Ta_2", "Ta_3",
                   "Ta_4", "Ta_5", "Ta_6", "Ta_7", "Ta_8", "Ta_9"};
    break;
  case ScaledMisalignedImu::kModelId:
    extraLabels = {"Mg_11", "Mg_21",       "Mg_22",       "Mg_31",      "Mg_32",
                   "Mg_33", "A_11",        "A_12",        "A_13",       "A_21",
                   "A_22",  "A_23",        "A_31",        "A_32",       "A_33",
                   "Ma_11", "Ma_21",       "Ma_22",       "Ma_31",      "Ma_32",
                   "Ma_33", "theta_g_a_x", "theta_g_a_y", "theta_g_a_z"};
    break;
  case Imu_BG_BA::kModelId:
  default:
    break;
  }
  *labels = extraLabels;
}

inline void
ImuModelToAugmentedDimensionLabels(const int imu_model,
                                   std::vector<std::string> *labels) {
  std::vector<std::string> extraLabels;
  switch (imu_model) {
  case Imu_BG_BA_MG_TS_MA::kModelId:
    extraLabels = {"Mg_1", "Mg_2", "Mg_3", "Mg_4", "Mg_5", "Mg_6", "Mg_7",
                   "Mg_8", "Mg_9", "Ts_1", "Ts_2", "Ts_3", "Ts_4", "Ts_5",
                   "Ts_6", "Ts_7", "Ts_8", "Ts_9", "Ma_1", "Ma_2", "Ma_3",
                   "Ma_4", "Ma_5", "Ma_6"};
    break;
  case Imu_BG_BA_TG_TS_TA::kModelId:
    extraLabels = {"Tg_1", "Tg_2", "Tg_3", "Tg_4", "Tg_5", "Tg_6", "Tg_7",
                   "Tg_8", "Tg_9", "Ts_1", "Ts_2", "Ts_3", "Ts_4", "Ts_5",
                   "Ts_6", "Ts_7", "Ts_8", "Ts_9", "Ta_1", "Ta_2", "Ta_3",
                   "Ta_4", "Ta_5", "Ta_6", "Ta_7", "Ta_8", "Ta_9"};
    break;
  case ScaledMisalignedImu::kModelId:
    extraLabels = {"Mg_11", "Mg_21",   "Mg_22",   "Mg_31",   "Mg_32",
                   "Mg_33", "A_11",    "A_12",    "A_13",    "A_21",
                   "A_22",  "A_23",    "A_31",    "A_32",    "A_33",
                   "Ma_11", "Ma_21",   "Ma_22",   "Ma_31",   "Ma_32",
                   "Ma_33", "q_g_a_x", "q_g_a_y", "q_g_a_z", "q_g_a_w"};
    break;
  case Imu_BG_BA::kModelId:
  default:
    break;
  }
  *labels = extraLabels;
}

inline void ImuModelToDimensionLabels(const int imu_model,
                                      std::vector<std::string> *labels) {
  *labels = {"b_g_x[rad/s]", "b_g_y", "b_g_z",
             "b_a_x[m/s^2]", "b_a_y", "b_a_z"};
  std::vector<std::string> extraLabels;
  ImuModelToAugmentedDimensionLabels(imu_model, &extraLabels);
  labels->insert(labels->end(), extraLabels.begin(), extraLabels.end());
}

inline int ImuModelNameToId(std::string imu_error_model_descrip) {
  std::transform(imu_error_model_descrip.begin(), imu_error_model_descrip.end(),
                 imu_error_model_descrip.begin(),
                 [](unsigned char c) { return std::toupper(c); });
  if (imu_error_model_descrip.compare("SCALEDMISALIGNED") == 0) {
    return ScaledMisalignedImu::kModelId;
  } else if (imu_error_model_descrip.compare("BG_BA_TG_TS_TA") == 0) {
    return Imu_BG_BA_TG_TS_TA::kModelId;
  } else if (imu_error_model_descrip.compare("BG_BA") == 0) {
    return Imu_BG_BA::kModelId;
  } else {
    return Imu_BG_BA_MG_TS_MA::kModelId;
  }
}

inline Eigen::Matrix<double, Eigen::Dynamic, 1> ImuModelNominalAugmentedParams(int model_id) {
  switch (model_id) {
#define MODEL_CASES IMU_ERROR_MODEL_CASES
#define IMU_ERROR_MODEL_CASE(ImuModel) \
    case ImuModel::kModelId:             \
  return ImuModel::getNominalAugmentedParams<double>();

    MODEL_SWITCH_CASES

    #undef IMU_ERROR_MODEL_CASE
    #undef MODEL_CASES
    }
}

inline Eigen::VectorXd ImuModelComputeAugmentedParamsError(
    int model_id, const Eigen::VectorXd& parameters) {
  switch (model_id) {
#define MODEL_CASES IMU_ERROR_MODEL_CASES
#define IMU_ERROR_MODEL_CASE(ImuModel) \
    case ImuModel::kModelId:             \
  return ImuModel::computeAugmentedParamsError(parameters);

    MODEL_SWITCH_CASES

    #undef IMU_ERROR_MODEL_CASE
    #undef MODEL_CASES
  }
}
}  // namespace swift_vio
#endif  // INCLUDE_SWIFT_VIO_IMU_ERROR_MODELS_HPP_
