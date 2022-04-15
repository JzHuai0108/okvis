#include <okvis/kinematics/sophus_operators.hpp>
#include <iostream>
#include "gtest/gtest.h"

TEST(Operators, R2ypr) {
  Eigen::Quaterniond q = Eigen::Quaterniond::UnitRandom();
  double y1 = okvis::kinematics::quaternionToYaw(q);
  Eigen::Vector3d ypr1 = q.toRotationMatrix().eulerAngles(0, 1, 2);
  Eigen::Vector3d ypr2 = okvis::kinematics::R2ypr(q.toRotationMatrix());
  // std::cout << "qxyzw " << q.coeffs().transpose() << "\n";
  // std::cout << " ypr1 " << ypr1.transpose() << "\n ypr2 " << ypr2.transpose()
  // << "\n";
  EXPECT_LT(std::abs(ypr2[0] - y1), 1e-7);
  EXPECT_GT((ypr1 - ypr2).lpNorm<Eigen::Infinity>(), 1.0);
}

TEST(Operators, ypr2R) {
  Eigen::Quaterniond q = Eigen::Quaterniond::UnitRandom();
  Eigen::Vector3d ypr = okvis::kinematics::R2ypr(q.toRotationMatrix());
  Eigen::Quaterniond q1(okvis::kinematics::ypr2R(ypr));
  EXPECT_LT((q1.conjugate() * q).coeffs().head<3>().lpNorm<Eigen::Infinity>(), 1e-7);
}
