/**
 * @file    TestPseudoInverse.cpp
 * @brief   Unit test for MatrixPseudoInverse
 * @author  Jianzhu Huai
 */

#include <gtest/gtest.h>
#include <chrono>
#include <swift_vio/VectorNormalizationJacobian.hpp>
#include <okvis/kinematics/MatrixPseudoInverse.hpp>

TEST(matrixTestSuite, pseudoInverseSymmSqrtFullRank) {
  double rootScale = 5;
  double scale = rootScale * rootScale;
  Eigen::Matrix3d lambda = Eigen::Matrix3d::Identity() * scale;
  Eigen::Matrix3d invLambda = Eigen::Matrix3d::Identity() / scale;
  Eigen::Matrix3d invLambdaSqrt = Eigen::Matrix3d::Identity() / rootScale;

  int rankExpected = 3;
  Eigen::Matrix3d result;
  int rank;
  okvis::MatrixPseudoInverse::pseudoInverseSymm(
      lambda, result, std::numeric_limits<double>::epsilon(), &rank);
  EXPECT_EQ(rank, rankExpected);
  const double eps = 1e-7;
  EXPECT_TRUE(invLambda.isApprox(result, eps));

  Eigen::Matrix3d resultSqrt;
  int rankSqrt;
  okvis::MatrixPseudoInverse::pseudoInverseSymmSqrt(
      lambda, resultSqrt, std::numeric_limits<double>::epsilon(), &rankSqrt);

  EXPECT_EQ(rankSqrt, rankExpected);
  EXPECT_TRUE(invLambdaSqrt.isApprox(resultSqrt, eps));
}

class SimpleProjection {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  SimpleProjection() {
    K << 600, 0, 320, 0, 600, 240, 0, 0, 1;
    Kinv << 1.0 / 600, 0, -320.0 / 600, 0, 1.0 / 600, -240.0 / 600, 0, 0, 1;
  }
  /**
   * @brief projectWithJacobian
   * @param xy1
   * @param uv
   * @param j d(u,v)/d(x,y)
   */
  void projectWithJacobian(const Eigen::Vector3d &xy1, Eigen::Vector2d &uv,
                           Eigen::Matrix<double, 2, 2> *j) const {
    uv = (K * xy1).head<2>();
    *j = K.topLeftCorner<2, 2>();
  }

  void backProjectWithJacobian(const Eigen::Vector2d &uv, Eigen::Vector3d &xy1,
                               Eigen::Matrix<double, 2, 2> *j) const {
    Eigen::Vector3d uv1;
    uv1 << uv, 1;
    xy1 = Kinv * uv1;
    *j = Kinv.topLeftCorner<2, 2>();
  }

  Eigen::Matrix3d K;
  Eigen::Matrix3d Kinv;
};

TEST(matrixTestSuite, pseudoInverseSymmSqrt) {
  Eigen::Vector2d uv = Eigen::Vector2d::Random();
  uv[0] *= 320;
  uv[1] *= 240;

  Eigen::Vector3d xy1;
  SimpleProjection camera;
  Eigen::Matrix2d dxy_duv;
  camera.backProjectWithJacobian(uv, xy1, &dxy_duv);
  Eigen::Matrix<double, 3, 2> dxy1_duv;
  dxy1_duv.setZero();
  dxy1_duv.topLeftCorner<2, 2>() = dxy_duv;

  swift_vio::VectorNormalizationJacobian normalJac(xy1);
  Eigen::Matrix3d dunit_dxy1;
  normalJac.dxi_dvec(&dunit_dxy1);
  Eigen::Vector2d covuvDiagonal = Eigen::Vector2d::Random().cwiseAbs();
  covuvDiagonal[0] += 0.5;
  covuvDiagonal[1] += 0.5;
  Eigen::Matrix2d covuv = covuvDiagonal.asDiagonal();
  Eigen::Matrix2d infouv = Eigen::Matrix2d::Identity();
  infouv(0, 0) /= covuvDiagonal[0];
  infouv(1, 1) /= covuvDiagonal[1];

  Eigen::Matrix<double, 3, 2> dunit_duv = dunit_dxy1 * dxy1_duv;
  Eigen::Matrix3d covxi = dunit_duv * covuv * dunit_duv.transpose();
  int rankExpected = 2;
  Eigen::Matrix3d blindMatrix = Eigen::Matrix3d::Identity();
  blindMatrix(2, 2) = 0;

  Eigen::Matrix3d result;
  int rank;
  okvis::MatrixPseudoInverse::pseudoInverseSymm(
      covxi, result, std::numeric_limits<double>::epsilon(), &rank);
  EXPECT_EQ(rank, rankExpected);
  Eigen::Matrix3d dev = result * covxi - blindMatrix;
  EXPECT_LT(dev.lpNorm<Eigen::Infinity>(), 1.0);
  EXPECT_LT(dev.lpNorm<Eigen::Infinity>(),
            1e-6 * result.lpNorm<Eigen::Infinity>())
      << "covxi\n"
      << covxi << "\nresult\n"
      << result;
  Eigen::Matrix3d resultSqrt;
  int rankSqrt;
  okvis::MatrixPseudoInverse::pseudoInverseSymmSqrt(
      covxi, resultSqrt, std::numeric_limits<double>::epsilon(), &rankSqrt);
  EXPECT_EQ(rankSqrt, rankExpected);
  dev = resultSqrt * resultSqrt.transpose() * covxi - blindMatrix;
  EXPECT_LT(dev.lpNorm<Eigen::Infinity>(), 1.0);
  EXPECT_LT(dev.lpNorm<Eigen::Infinity>(),
            1e-3 * resultSqrt.lpNorm<Eigen::Infinity>())
      << "covxi\n"
      << covxi << "\nresultSqrt\n"
      << resultSqrt;

  Eigen::MatrixXd pinv =
      covxi.completeOrthogonalDecomposition().pseudoInverse();
  EXPECT_LT((pinv - result).lpNorm<Eigen::Infinity>(), 1e-8)
      << "eigen pinv\n"
      << pinv << "\nokvis pinv\n"
      << result;
}

TEST(matrixTestSuite, pseudoInverseSymmSqrtRankDeficient) {
  Eigen::Matrix3d blindMatrix = Eigen::Matrix3d::Identity();
  blindMatrix(1, 1) = 0;

  Eigen::Matrix3d resultSqrt;
  int rankSqrt;
  okvis::MatrixPseudoInverse::pseudoSymmSqrt(
      blindMatrix, resultSqrt, std::numeric_limits<double>::epsilon(),
      &rankSqrt);

  EXPECT_EQ(rankSqrt, 2);
  EXPECT_LT((resultSqrt * resultSqrt.transpose() - blindMatrix)
                .lpNorm<Eigen::Infinity>(),
            1e-8)
      << "actual sqrt\n"
      << resultSqrt << "\nexpected sqrt\n"
      << blindMatrix;
}

TEST(matrixTestSuite, testMatrixSqrt) {
  for (int i = 0; i < 100; ++i) {
    Eigen::Matrix<double, 3, 3> U;
    U.setRandom();
    Eigen::Matrix3d R =
        U.transpose() * U + 5.0 * Eigen::Matrix<double, 3, 3>::Identity();
    Eigen::MatrixXd sqrtR;
    okvis::computeMatrixSqrt(R, sqrtR);
    Eigen::MatrixXd reconstructedR = sqrtR * sqrtR.transpose();
    EXPECT_LT((R - reconstructedR).lpNorm<Eigen::Infinity>(), 1e-9)
        << "Large error in reconstructing the matrix from the square root in "
           "iteration "
        << i;
  }
}

TEST(matrixTestSuite, testMatrixInverseSqrt) {
  using namespace Eigen;
  using namespace std::chrono;
  constexpr int rows = 9;
  int runs = 100;
  double time1(0);
  double time2(0);
  for (int i = 0; i < runs; ++i) {
    MatrixXd X = MatrixXd::Random(rows, rows);
    MatrixXd A = X * X.transpose();
    Eigen::MatrixXd L1(A.rows(), A.cols()), invA1(A.rows(), A.cols());

    Eigen::Matrix<double, rows, rows> A2 = A;
    Eigen::Matrix<double, rows, rows> L2, invA2;
    auto start = high_resolution_clock::now();
    okvis::MatrixPseudoInverse::pseudoInverseSymmSqrt(A, L1, invA1);
    auto stop = high_resolution_clock::now();
    auto duration1 = duration_cast<microseconds>(stop - start);
    okvis::MatrixPseudoInverse::inverseSymmSqrt(A2, L2, invA2);
    auto stop2 = high_resolution_clock::now();
    auto duration2 = duration_cast<microseconds>(stop2 - stop);
    time1 += duration1.count();
    time2 += duration2.count();
    EXPECT_LT((L1 * L1.transpose() * A - Matrix<double, rows, rows>::Identity())
                  .lpNorm<Eigen::Infinity>(),
              1e-7)
        << "Wrong square root from self adjoint eigen solver:\n"
        << L1;
    EXPECT_LT((L2 * L2.transpose() * A - Matrix<double, rows, rows>::Identity())
                  .lpNorm<Eigen::Infinity>(),
              1e-7)
        << "Wrong square root from llt:\n"
        << L1;
    EXPECT_LT((invA1 - invA2).lpNorm<Eigen::Infinity>(), 1e-3)
        << "Different invA " << i;
  }
  std::cout << "Average time: self adjoint eigen solver " << (time1 / runs)
            << " us, llt " << (time2 / runs) << " us.\n";
}

TEST(matrixTestSuite, matrixInverse) {
  using namespace Eigen;
  using namespace std::chrono;
  constexpr int rows = 9;
  int runs = 100;
  double time1(0);
  for (int i = 0; i < runs; ++i) {
    MatrixXd X = MatrixXd::Random(rows, rows);
    MatrixXd A = X * X.transpose();
    Eigen::MatrixXd invA(A.rows(), A.cols());

    auto start = high_resolution_clock::now();
    okvis::MatrixPseudoInverse::inverseSymm(A, invA);
    auto stop = high_resolution_clock::now();
    auto duration1 = duration_cast<microseconds>(stop - start);
    time1 += duration1.count();
    EXPECT_LT((invA * A - Matrix<double, rows, rows>::Identity())
                  .lpNorm<Eigen::Infinity>(),
              1e-7)
        << "Wrong square root from llt\n"
        << invA;
  }
  std::cout << "Average inversion time with llt " << (time1 / runs) << " us.\n";
}
