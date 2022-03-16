
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
    OKVIS_ASSERT_LT(std::runtime_error, (P - P2).lpNorm<Eigen::Infinity>(), 1e-8, "Scale block wrong!");
}
