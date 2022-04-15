/**
 * @file sophus_operators.hpp
 * @brief File with operators copied from sophus for SO(3).
 * @author Jianzhu Huai
 */

#ifndef INCLUDE_OKVIS_KINEMATICS_SOPHUS_OPERATORS_HPP_
#define INCLUDE_OKVIS_KINEMATICS_SOPHUS_OPERATORS_HPP_

#include <stdint.h>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <okvis/kinematics/Transformation.hpp>
#include <glog/logging.h>

/// \brief okvis Main namespace of this package.
namespace okvis {
/// \brief kinematics Namespace for kinematics functionality, i.e. transformations and stuff.
namespace kinematics {

template <typename Scalar>
struct SophusConstants {
  EIGEN_ALWAYS_INLINE static Scalar epsilon() {
    return static_cast<Scalar>(1e-10);
  }

  EIGEN_ALWAYS_INLINE static Scalar pi() { return static_cast<Scalar>(M_PI); }
};

// from sophus/so3.hpp
template <typename Scalar>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE static Eigen::Matrix<Scalar, 3, 1> vee(
    const Eigen::Matrix<Scalar, 3, 3>& Omega) {
  return static_cast<Scalar>(0.5) *
         Eigen::Matrix<Scalar, 3, 1>(Omega(2, 1) - Omega(1, 2),
                                     Omega(0, 2) - Omega(2, 0),
                                     Omega(1, 0) - Omega(0, 1));
}

/// Warn: Do not use sinc or its templated version for autodiff involving quaternions
///  as its real part may be calculated without considering the infinisimal input.
/// Use expAndTheta borrowed from Sophus instead for this purpose
template <typename Scalar>
Eigen::Quaternion<Scalar> expAndTheta(const Eigen::Matrix<Scalar, 3, 1> & omega) {
    Scalar theta_sq = omega.squaredNorm();
    Scalar theta = sqrt(theta_sq);
    Scalar half_theta = static_cast<Scalar>(0.5)*(theta);

    Scalar imag_factor;
    Scalar real_factor;
    if(theta<SophusConstants<Scalar>::epsilon()) {
      Scalar theta_po4 = theta_sq*theta_sq;
      imag_factor = static_cast<Scalar>(0.5)
                    - static_cast<Scalar>(1.0/48.0)*theta_sq
                    + static_cast<Scalar>(1.0/3840.0)*theta_po4;
      real_factor = static_cast<Scalar>(1)
                    - static_cast<Scalar>(0.5)*theta_sq +
                    static_cast<Scalar>(1.0/384.0)*theta_po4;
    } else {
      Scalar sin_half_theta = sin(half_theta);
      imag_factor = sin_half_theta/theta;
      real_factor = cos(half_theta);
    }

    return Eigen::Quaternion<Scalar>(real_factor,
                                               imag_factor*omega.x(),
                                               imag_factor*omega.y(),
                                               imag_factor*omega.z());
}

// From sophus so3.hpp
template <typename Scalar>
Eigen::Matrix<Scalar, 3, 1> logAndTheta(const Eigen::Quaternion<Scalar> & other,
                          Scalar * theta) {
  Scalar squared_n
      = other.vec().squaredNorm();
  Scalar n = sqrt(squared_n);
  Scalar w = other.w();

  Scalar two_atan_nbyw_by_n;

  // Atan-based log thanks to
  //
  // C. Hertzberg et al.:
  // "Integrating Generic Sensor Fusion Algorithms with Sound State
  // Representation through Encapsulation of Manifolds"
  // Information Fusion, 2011

  if (n < SophusConstants<Scalar>::epsilon()) {
    // If quaternion is normalized and n=0, then w should be 1;
    // w=0 should never happen here!
    CHECK_GT(abs(w), SophusConstants<Scalar>::epsilon()) <<
                  "Quaternion should be normalized!";
    Scalar squared_w = w*w;
    two_atan_nbyw_by_n = static_cast<Scalar>(2) / w
                         - static_cast<Scalar>(2)*(squared_n)/(w*squared_w);
  } else {
    if (abs(w)<SophusConstants<Scalar>::epsilon()) {
      if (w > static_cast<Scalar>(0)) {
        two_atan_nbyw_by_n = M_PI/n;
      } else {
        two_atan_nbyw_by_n = -M_PI/n;
      }
    }else{
      two_atan_nbyw_by_n = static_cast<Scalar>(2) * atan(n/w) / n;
    }
  }

  *theta = two_atan_nbyw_by_n*n;

  return two_atan_nbyw_by_n * other.vec();
}
/**
 * @brief ominus The inverse of Tbar = T.oplus(delta).
 * @param Tbar
 * @param T
 * @return delta
 */
inline Eigen::Matrix<double, 6, 1> ominus(
    const okvis::kinematics::Transformation& Tbar,
    const okvis::kinematics::Transformation& T) {
  Eigen::Matrix<double, 3, 3> dR = Tbar.C() * T.C().transpose();
  Eigen::Matrix<double, 6, 1> delta;
  delta.head<3>() = Tbar.r() - T.r();
  delta.tail<3>() = vee(dR);
  return delta;
}

inline bool motionLessThan(const okvis::kinematics::Transformation& Tab,
                           double distanceThreshold, double angleThreshold) {
  return Tab.r().norm() < distanceThreshold &&
      std::fabs(Eigen::AngleAxisd(Tab.q()).angle()) < angleThreshold;
}

template <typename T>
T quaternionToYaw(const Eigen::Quaternion<T>& q) {
  T yaw;
  T sqw, sqx, sqy, sqz;
  sqx = q.x()*q.x();
  sqy = q.y()*q.y();
  sqz = q.z()*q.z();
  sqw = q.w()*q.w();

  T sarg = -T(2) * (q.x()*q.z() - q.w()*q.y()) / (sqx + sqy + sqz + sqw);
  if (sarg <= T(-0.99999)) {
    yaw = -T(2) * atan2(q.y(), q.x());
  } else if (sarg >= T(0.99999)) {
    yaw = T(2) * atan2(q.y(), q.x());
  } else {
    yaw = atan2(T(2) * (q.x()*q.y() + q.w()*q.z()), sqw + sqx - sqy - sqz);
  }

  return yaw;
}

// copied from vins mono
static Eigen::Vector3d R2ypr(const Eigen::Matrix3d &R) {
  Eigen::Vector3d n = R.col(0);
  Eigen::Vector3d o = R.col(1);
  Eigen::Vector3d a = R.col(2);

  Eigen::Vector3d ypr(3);
  double y = atan2(n(1), n(0));
  double p = atan2(-n(2), n(0) * cos(y) + n(1) * sin(y));
  double r =
      atan2(a(0) * sin(y) - a(1) * cos(y), -o(0) * sin(y) + o(1) * cos(y));
  ypr(0) = y;
  ypr(1) = p;
  ypr(2) = r;
  return ypr;
}

template <typename Derived>
static Eigen::Matrix<typename Derived::Scalar, 3, 3>
ypr2R(const Eigen::MatrixBase<Derived> &ypr) {
  typedef typename Derived::Scalar Scalar_t;
  Scalar_t y = ypr(0);
  Scalar_t p = ypr(1);
  Scalar_t r = ypr(2);

  Eigen::Matrix<Scalar_t, 3, 3> Rz;
  Rz << cos(y), -sin(y), 0, sin(y), cos(y), 0, 0, 0, 1;

  Eigen::Matrix<Scalar_t, 3, 3> Ry;
  Ry << cos(p), 0., sin(p), 0., 1., 0., -sin(p), 0., cos(p);

  Eigen::Matrix<Scalar_t, 3, 3> Rx;
  Rx << 1., 0., 0., 0., cos(r), -sin(r), 0., sin(r), cos(r);

  return Rz * Ry * Rx;
}

} // namespace kinematics
} // namespace okvis

#endif /* INCLUDE_OKVIS_KINEMATICS_SOPHUS_OPERATORS_HPP_ */
