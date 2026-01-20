#ifndef COVPROPCONFIG_HPP
#define COVPROPCONFIG_HPP

#include <gtest/gtest.h>
#include <iostream>
#include <Eigen/Geometry>

#include <vio/Sample.h>

#include <okvis/Measurements.hpp>
#include <okvis/Parameters.hpp>

inline void check_q_near(const Eigen::Quaterniond& q_WS0,
                         const Eigen::Quaterniond& q_WS1, const double tol) {
  Eigen::Quaterniond dq = q_WS0.inverse() * q_WS1;
  EXPECT_LT(std::fabs(std::fabs(dq.w()) - 1), tol);
  EXPECT_LT(std::fabs(dq.x()), tol);
  EXPECT_LT(std::fabs(dq.y()), tol);
  EXPECT_LT(std::fabs(dq.z()), tol);
}

inline void check_v_near(const Eigen::Matrix<double, 3, 1>& v0,
                         const Eigen::Matrix<double, 3, 1>& v1,
                         const double tol) {
  EXPECT_LT(((v1 - v0).norm()), tol);
}

inline void check_p_near(const Eigen::Matrix<double, 3, 1>& p_WS_W0,
                         const Eigen::Matrix<double, 3, 1>& p_WS_W1,
                         const double tol) {
  EXPECT_LT((p_WS_W1 - p_WS_W0).norm(), tol);
}


inline void print_p_q_sb(const Eigen::Vector3d& p_WS_W,
                         const Eigen::Quaterniond& q_WS,
                         const Eigen::Matrix<double, 9, 1>& sb) {
  std::cout << "p:" << p_WS_W.transpose() << std::endl;
  std::cout << "q:" << q_WS.x() << " " << q_WS.y() << " " << q_WS.z() << " "
            << q_WS.w() << std::endl;
  std::cout << "v:" << sb.head<3>().transpose() << std::endl;
  std::cout << "bg ba:" << sb.tail<6>().transpose() << std::endl;
}

inline void print_p_q_v(const Eigen::Vector3d& p_WS_W,
                         const Eigen::Quaterniond& q_WS,
                         const Eigen::Matrix<double, 3, 1>& speed) {
  std::cout << "p:" << p_WS_W.transpose() << std::endl;
  std::cout << "q:" << q_WS.x() << " " << q_WS.y() << " " << q_WS.z() << " "
            << q_WS.w() << std::endl;
  std::cout << "v:" << speed.transpose() << std::endl;
}

inline void expectNearAbsRel(const Eigen::MatrixXd& ref,
                             const Eigen::MatrixXd& est,
                             double absTol,
                             double relTol) {
  ASSERT_EQ(ref.rows(), est.rows());
  ASSERT_EQ(ref.cols(), est.cols());

  for (int i = 0; i < ref.rows(); ++i) {
    for (int j = 0; j < ref.cols(); ++j) {
      const double a = ref(i,j);
      const double b = est(i,j);
      const double diff = std::abs(a - b);
      const double scale = std::max(std::abs(a), std::abs(b));
      const double tol = absTol + relTol * scale;

      EXPECT_LE(diff, tol)
        << "(" << i << "," << j << ") ref=" << a << " est=" << b
        << " diff=" << diff << " tol=" << tol
        << " absTol=" << absTol << " relTol=" << relTol;
    }
  }
}

template<typename ImuModelT>
struct CovPropConfig {
 private:
  const double g;
  const double sigma_g_c;
  const double sigma_a_c;
  const double sigma_gw_c;
  const double sigma_aw_c;
  const double dt;

  Eigen::Vector3d p_WS_W0;
  Eigen::Quaterniond q_WS0;
  Eigen::Matrix<double, 9, 1> sb0;
  Eigen::Matrix<double, 15, 15> cov0;
  okvis::ImuMeasurementDeque imuMeasurements;
  Eigen::Matrix<double, ImuModelT::kAugmentedMinDim, 1> imuAugmentedParams;

  okvis::ImuParameters imuParams;

  const bool nominalImuIntrinsics;  // use nominal or noisy values for IMU intrinsic parameters.
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  CovPropConfig(bool nominalPose, bool _nominalImuIntrinsics, unsigned int seed = time(0))
      : g(9.81),
        sigma_g_c(5e-2),
        sigma_a_c(3e-2),
        sigma_gw_c(7e-3),
        sigma_aw_c(2e-3),
        dt(0.005),
        nominalImuIntrinsics(_nominalImuIntrinsics) {
    srand(seed);

    imuParams.g_max = 7.8;
    imuParams.a_max = 176;
    imuParams.sigma_g_c = sigma_g_c;
    imuParams.sigma_a_c = sigma_a_c;
    imuParams.sigma_gw_c = sigma_gw_c;
    imuParams.sigma_aw_c = sigma_aw_c;
    imuParams.g = g;
    int freq = std::round(1.0 / dt);

    if (nominalPose) {
      p_WS_W0 = Eigen::Vector3d(0, 0, 0);
      q_WS0 = Eigen::Quaterniond(1, 0, 0, 0);
    } else {
      p_WS_W0 = Eigen::Vector3d(vio::gauss_rand(0, 1), vio::gauss_rand(0, 1),
                                vio::gauss_rand(0, 1));
      q_WS0 = Eigen::Quaterniond(vio::gauss_rand(0, 1), vio::gauss_rand(0, 1),
                                 vio::gauss_rand(0, 1), vio::gauss_rand(0, 1));
      q_WS0.normalize();
    }
    sb0 << vio::gauss_rand(0, 1), vio::gauss_rand(0, 1), vio::gauss_rand(0, 1), 0.1, 0.1, 0.1, 0.1, 0.1, 0.1;

    cov0.setIdentity();
    cov0.diagonal().head<3>().setConstant(vio::uniform_rand(0.1, 1) * 10);  // p
    cov0.diagonal().segment<3>(3).setConstant(vio::uniform_rand(0.1, 1) * 5);   // q
    cov0.diagonal().segment<3>(6).setConstant(vio::uniform_rand(0.1, 1) * 20);  // v
    cov0.diagonal().segment<3>(9).setConstant(vio::uniform_rand(0.1, 1) * 0.1); // bg
    cov0.diagonal().segment<3>(12).setConstant(vio::uniform_rand(0.1, 1) * 0.2); // ba
    for (int jack = 0; jack < freq * 10; ++jack) {
      Eigen::Vector3d gyr = Eigen::Vector3d::Random();  // range from [-1, 1]
      Eigen::Vector3d acc = Eigen::Vector3d::Random();
      acc[2] = g + vio::gauss_rand(0.0, 0.1);
      imuMeasurements.push_back(okvis::ImuMeasurement(
          okvis::Time(jack * dt), okvis::ImuSensorReadings(gyr, acc)));
    }
    // if to test Leutenegger's propagation method, set imuAugmentedParams as nominal
    // values
    imuAugmentedParams = ImuModelT::template getNominalAugmentedParams<double>();
    // otherwise, random initialization is OK
    if (!nominalImuIntrinsics)
      imuAugmentedParams += 5e-3 * (Eigen::Matrix<double, ImuModelT::kAugmentedMinDim, 1>::Random());
  }

  double get_g() const { return g; }
  double get_sigma_g_c() const { return sigma_g_c; }
  double get_sigma_a_c() const { return sigma_a_c; }
  double get_sigma_gw_c() const { return sigma_gw_c; }
  double get_sigma_aw_c() const { return sigma_aw_c; }
  double get_dt() const { return dt; }
  Eigen::Vector3d get_p_WS_W0() const { return p_WS_W0; }
  Eigen::Quaterniond get_q_WS0() const { return q_WS0; }
  Eigen::Matrix<double, 9, 1> get_sb0() const { return sb0; }
  Eigen::Matrix<double, 3, 1> get_v_WS0() const { return sb0.head<3>(); }
  Eigen::Matrix<double, 6, 1> get_bias0() const { return sb0.tail<6>(); }
  Eigen::Matrix<double, 15, 15> get_cov0() const { return cov0; }
  void zeroBias0() { sb0.tail<6>().setZero(); } 

  const okvis::ImuMeasurementDeque& get_imu_measurements() const {
    return imuMeasurements;
  }

  const Eigen::Matrix<double, ImuModelT::kAugmentedMinDim, 1>& getImuExtraParams() const { return imuAugmentedParams; }

  const double* getImuExtraParamPtr() const {
    return imuAugmentedParams.data();
  }

  const okvis::ImuParameters& get_imu_params() const { return imuParams; }
  okvis::Time get_meas_begin_time() const {
    return imuMeasurements.begin()->timeStamp;
  }
  okvis::Time get_meas_end_time() const {
    return imuMeasurements.rbegin()->timeStamp;
  }
  size_t get_meas_size() const { return imuMeasurements.size(); }

  Eigen::Matrix<double, 12, 1> get_q_n_aw_babw() const {
    Eigen::Matrix<double, 12, 1> q_n_aw_babw;
    q_n_aw_babw << pow(imuParams.sigma_a_c, 2), pow(imuParams.sigma_a_c, 2),
        pow(imuParams.sigma_a_c, 2), pow(imuParams.sigma_g_c, 2),
        pow(imuParams.sigma_g_c, 2), pow(imuParams.sigma_g_c, 2),
        pow(imuParams.sigma_aw_c, 2), pow(imuParams.sigma_aw_c, 2),
        pow(imuParams.sigma_aw_c, 2), pow(imuParams.sigma_gw_c, 2),
        pow(imuParams.sigma_gw_c, 2), pow(imuParams.sigma_gw_c, 2);
    return q_n_aw_babw;
  }
  std::vector<Eigen::Matrix<double, 7, 1>,
              Eigen::aligned_allocator<Eigen::Matrix<double, 7, 1>>>
  get_imu_measurement_vector() const {
    std::vector<Eigen::Matrix<double, 7, 1>,
                Eigen::aligned_allocator<Eigen::Matrix<double, 7, 1>>>
        measurements;

    Eigen::Matrix<double, 7, 1> meas;
    for (auto iter = imuMeasurements.begin(); iter != imuMeasurements.end();
         ++iter) {
      meas << iter->timeStamp.toSec(), iter->measurement.gyroscopes,
          iter->measurement.accelerometers;
      measurements.push_back(meas);
    }
    return measurements;
  }

  void check_q_near(const Eigen::Quaterniond& q_WS1, const double tol) const {
    ::check_q_near(q_WS0, q_WS1, tol);
  }

  void check_v_near(const Eigen::Matrix<double, 3, 1>& speed1,
                    const double tol) const {
    ::check_v_near(get_v_WS0(), speed1, tol);
  }

  void check_p_near(const Eigen::Matrix<double, 3, 1>& p_WS_W1,
                    const double tol) {
    ::check_p_near(p_WS_W0, p_WS_W1, tol);
  }
};

#endif // COVPROPCONFIG_HPP
