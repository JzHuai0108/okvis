/**
 * @file TestMatrixUtilities.cpp
 */
#include "swift_vio/matrixUtilities.h"
#include "gtest/gtest.h"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <okvis/assert_macros.hpp>
#include <chrono>

using namespace std::chrono;

TEST(MatrixUtilities, scaleBlockRows) {
    constexpr int covRows = 20;
    Eigen::Matrix<double, covRows, covRows> cov;
    Eigen::Matrix<double, covRows, covRows> gen = Eigen::Matrix<double, covRows, covRows>::Random();
    cov.noalias() = gen.transpose() * gen;

    Eigen::Quaterniond q = Eigen::Quaterniond::UnitRandom();
    Eigen::Matrix3d C = q.toRotationMatrix();
    Eigen::MatrixXd T = Eigen::MatrixXd::Identity(covRows, covRows);
    T.topLeftCorner<3, 3>() = C.transpose();
    T.block<3, 3>(3, 3) = C.transpose();
    T.block<3, 3>(6, 6) = C.transpose();

    auto start = high_resolution_clock::now();
    Eigen::MatrixXd P = T * cov * T.transpose();
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);

    Eigen::MatrixXd P2 = cov;
    auto start2 = std::chrono::high_resolution_clock::now();
    swift_vio::scaleBlockRows(C.transpose(), 3, &P2);
    swift_vio::scaleBlockCols(C, 3, &P2);
    auto stop2 = high_resolution_clock::now();
    auto duration2 = duration_cast<microseconds>(stop2 - start2);

    std::cout << "Product transform takes " << duration.count() << " usec. Scale transform takes "
              << duration2.count() << " usec for covariance matrix of rows " << covRows << ".\n";
    EXPECT_LT((P - P2).lpNorm<Eigen::Infinity>(), 1e-8) << "Scale block wrong!";
}

TEST(MatrixUtilities, upperTriangularBlocksToSymmMatrix) {
  std::vector<
      Eigen::Matrix<double, -1, -1, Eigen::RowMajor>,
      Eigen::aligned_allocator<Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>>
      covBlockList;
  std::vector<std::pair<int, int>> rows{
      {0, 2}, {2, 1}, {3, 3}}; // index in cov to param block size

  Eigen::Matrix<double, 6, 6> cov0;
  cov0.setRandom();
  cov0 = (cov0 * cov0.transpose()).eval();
  for (size_t i = 0; i < rows.size(); ++i) {
    for (size_t j = i; j < rows.size(); ++j) {
      covBlockList.emplace_back(cov0.block(rows[i].first, rows[j].first,
                                           rows[i].second, rows[j].second));
    }
  }
  Eigen::Matrix<double, -1, -1> cov;
  swift_vio::upperTriangularBlocksToSymmMatrix(covBlockList, &cov);
  EXPECT_LT((cov - cov0).lpNorm<Eigen::Infinity>(), 1e-7)
      << "cov0\n" << cov0 << "\ncov\n" << cov;
}
