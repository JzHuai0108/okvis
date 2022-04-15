#ifndef CHECKIMUERROR_H
#define CHECKIMUERROR_H

#include <gtest/gtest.h>
#include <swift_vio/ceres/DynamicImuError.hpp>
#include <swift_vio/ceres/ImuErrorConstBias.hpp>

namespace okvis {
namespace ceres {

template <typename ImuModelT>
bool checkJacobians(const DynamicImuError<ImuModelT> &costFunction,
                    double *const *parameters) {
  typedef DynamicImuError<ImuModelT> ImuErrorT;
  double *jacobians[ImuErrorT::Index::extra + ImuModelT::kXBlockDims.size()];
  Eigen::Matrix<double, ImuErrorT::kNumResiduals, 7, Eigen::RowMajor> Jp0;
  Eigen::Matrix<double, ImuErrorT::kNumResiduals, 3, Eigen::RowMajor> Jv0;
  Eigen::Matrix<double, ImuErrorT::kNumResiduals, 6, Eigen::RowMajor> Jb0;
  Eigen::Matrix<double, ImuErrorT::kNumResiduals, 7, Eigen::RowMajor> Jp1;
  Eigen::Matrix<double, ImuErrorT::kNumResiduals, 3, Eigen::RowMajor> Jv1;
  Eigen::Matrix<double, ImuErrorT::kNumResiduals, 6, Eigen::RowMajor> Jb1;
  Eigen::Matrix<double, ImuErrorT::kNumResiduals, 3, Eigen::RowMajor> Jg;
  jacobians[ImuErrorT::Index::T_WB0] = Jp0.data();
  jacobians[ImuErrorT::Index::v_WB0] = Jv0.data();
  jacobians[ImuErrorT::Index::bgBa0] = Jb0.data();
  jacobians[ImuErrorT::Index::T_WB1] = Jp1.data();
  jacobians[ImuErrorT::Index::v_WB1] = Jv1.data();
  jacobians[ImuErrorT::Index::bgBa1] = Jb1.data();
  jacobians[ImuErrorT::Index::unitgW] = Jg.data();

  double *
      jacobiansMinimal[ImuErrorT::Index::extra + ImuModelT::kXBlockDims.size()];
  Eigen::Matrix<double, ImuErrorT::kNumResiduals, 6, Eigen::RowMajor> Jp0min;
  Eigen::Matrix<double, ImuErrorT::kNumResiduals, 3, Eigen::RowMajor> Jv0min;
  Eigen::Matrix<double, ImuErrorT::kNumResiduals, 6, Eigen::RowMajor> Jb0min;

  Eigen::Matrix<double, ImuErrorT::kNumResiduals, 6, Eigen::RowMajor> Jp1min;
  Eigen::Matrix<double, ImuErrorT::kNumResiduals, 3, Eigen::RowMajor> Jv1min;
  Eigen::Matrix<double, ImuErrorT::kNumResiduals, 6, Eigen::RowMajor> Jb1min;
  Eigen::Matrix<double, ImuErrorT::kNumResiduals, 2, Eigen::RowMajor> Jgmin;
  jacobiansMinimal[ImuErrorT::Index::T_WB0] = Jp0min.data();
  jacobiansMinimal[ImuErrorT::Index::v_WB0] = Jv0min.data();
  jacobiansMinimal[ImuErrorT::Index::bgBa0] = Jb0min.data();
  jacobiansMinimal[ImuErrorT::Index::T_WB1] = Jp1min.data();
  jacobiansMinimal[ImuErrorT::Index::v_WB1] = Jv1min.data();
  jacobiansMinimal[ImuErrorT::Index::bgBa1] = Jb1min.data();
  jacobiansMinimal[ImuErrorT::Index::unitgW] = Jgmin.data();

  std::vector<std::shared_ptr<Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>>
      jacPtrs;
  std::vector<std::shared_ptr<Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>>
      jacMinPtrs;
  for (size_t i = 0u; i < ImuModelT::kXBlockDims.size(); ++i) {
    std::shared_ptr<Eigen::Matrix<double, -1, -1, Eigen::RowMajor>> jacPtr(
        new Eigen::Matrix<double, -1, -1, Eigen::RowMajor>(
            ImuErrorT::kNumResiduals, ImuModelT::kXBlockDims[i]));
    jacobians[i + ImuErrorT::Index::extra] = jacPtr->data();

    std::shared_ptr<Eigen::Matrix<double, -1, -1, Eigen::RowMajor>> jacMinPtr(
        new Eigen::Matrix<double, -1, -1, Eigen::RowMajor>(
            ImuErrorT::kNumResiduals, ImuModelT::kXBlockMinDims[i]));
    jacobiansMinimal[i + ImuErrorT::Index::extra] = jacMinPtr->data();
    jacPtrs.push_back(jacPtr);
    jacMinPtrs.push_back(jacMinPtr);
  }

  Eigen::Matrix<double, ImuErrorT::kNumResiduals, 1> residuals;

  // evaluate twice to be sure that we will be using the linearisation of the
  // biases (i.e. no preintegrals redone)
  costFunction.Evaluate(parameters, residuals.data(), nullptr);
  costFunction.EvaluateWithMinimalJacobians(parameters, residuals.data(),
                                            jacobians, jacobiansMinimal);

  double *
      jacobiansNumeric[ImuErrorT::Index::extra + ImuModelT::kXBlockDims.size()];
  Eigen::Matrix<double, ImuErrorT::kNumResiduals, 7, Eigen::RowMajor>
      Jp0Numeric;
  Eigen::Matrix<double, ImuErrorT::kNumResiduals, 3, Eigen::RowMajor>
      Jv0Numeric;
  Eigen::Matrix<double, ImuErrorT::kNumResiduals, 6, Eigen::RowMajor>
      Jb0Numeric;
  Eigen::Matrix<double, ImuErrorT::kNumResiduals, 7, Eigen::RowMajor>
      Jp1Numeric;
  Eigen::Matrix<double, ImuErrorT::kNumResiduals, 3, Eigen::RowMajor>
      Jv1Numeric;
  Eigen::Matrix<double, ImuErrorT::kNumResiduals, 6, Eigen::RowMajor>
      Jb1Numeric;
  Eigen::Matrix<double, ImuErrorT::kNumResiduals, 3, Eigen::RowMajor> JgNumeric;
  jacobiansNumeric[ImuErrorT::Index::T_WB0] = Jp0Numeric.data();
  jacobiansNumeric[ImuErrorT::Index::v_WB0] = Jv0Numeric.data();
  jacobiansNumeric[ImuErrorT::Index::bgBa0] = Jb0Numeric.data();
  jacobiansNumeric[ImuErrorT::Index::T_WB1] = Jp1Numeric.data();
  jacobiansNumeric[ImuErrorT::Index::v_WB1] = Jv1Numeric.data();
  jacobiansNumeric[ImuErrorT::Index::bgBa1] = Jb1Numeric.data();
  jacobiansNumeric[ImuErrorT::Index::unitgW] = JgNumeric.data();

  double *jacobiansMinimalNumeric[ImuErrorT::Index::extra +
                                  ImuModelT::kXBlockDims.size()];
  Eigen::Matrix<double, ImuErrorT::kNumResiduals, 6, Eigen::RowMajor>
      Jp0minNumeric;
  Eigen::Matrix<double, ImuErrorT::kNumResiduals, 3, Eigen::RowMajor>
      Jv0minNumeric;
  Eigen::Matrix<double, ImuErrorT::kNumResiduals, 6, Eigen::RowMajor>
      Jb0minNumeric;

  Eigen::Matrix<double, ImuErrorT::kNumResiduals, 6, Eigen::RowMajor>
      Jp1minNumeric;
  Eigen::Matrix<double, ImuErrorT::kNumResiduals, 3, Eigen::RowMajor>
      Jv1minNumeric;
  Eigen::Matrix<double, ImuErrorT::kNumResiduals, 6, Eigen::RowMajor>
      Jb1minNumeric;

  Eigen::Matrix<double, ImuErrorT::kNumResiduals, 2, Eigen::RowMajor>
      JgminNumeric;
  jacobiansMinimalNumeric[ImuErrorT::Index::T_WB0] = Jp0minNumeric.data();
  jacobiansMinimalNumeric[ImuErrorT::Index::v_WB0] = Jv0minNumeric.data();
  jacobiansMinimalNumeric[ImuErrorT::Index::bgBa0] = Jb0minNumeric.data();
  jacobiansMinimalNumeric[ImuErrorT::Index::T_WB1] = Jp1minNumeric.data();
  jacobiansMinimalNumeric[ImuErrorT::Index::v_WB1] = Jv1minNumeric.data();
  jacobiansMinimalNumeric[ImuErrorT::Index::bgBa1] = Jb1minNumeric.data();
  jacobiansMinimalNumeric[ImuErrorT::Index::unitgW] = JgminNumeric.data();

  std::vector<std::shared_ptr<Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>>
      jacNumericPtrs;
  std::vector<std::shared_ptr<Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>>
      jacMinNumericPtrs;
  for (size_t i = 0u; i < ImuModelT::kXBlockDims.size(); ++i) {
    std::shared_ptr<Eigen::Matrix<double, -1, -1, Eigen::RowMajor>> jacPtr(
        new Eigen::Matrix<double, -1, -1, Eigen::RowMajor>(
            ImuErrorT::kNumResiduals, ImuModelT::kXBlockDims[i]));
    jacobiansNumeric[i + ImuErrorT::Index::extra] = jacPtr->data();

    std::shared_ptr<Eigen::Matrix<double, -1, -1, Eigen::RowMajor>> jacMinPtr(
        new Eigen::Matrix<double, -1, -1, Eigen::RowMajor>(
            ImuErrorT::kNumResiduals, ImuModelT::kXBlockMinDims[i]));
    jacobiansMinimalNumeric[i + ImuErrorT::Index::extra] = jacMinPtr->data();
    jacNumericPtrs.push_back(jacPtr);
    jacMinNumericPtrs.push_back(jacMinPtr);
  }

  costFunction.setReweight(false); // disable weighting update.
  Eigen::Matrix<double, ImuErrorT::kNumResiduals, 1> residualsNumeric;
  costFunction.EvaluateWithMinimalJacobiansNumeric(
      parameters, residualsNumeric.data(), jacobiansNumeric,
      jacobiansMinimalNumeric);

  // compare residuals evaluated with Imu param lin point and
  // residuals evaluated with actual Imu params.
//  const double eps = 1e-5;
//  for (size_t j = 0; j < ImuModelT::kXBlockDims.size(); ++j) {
//    for (int i = 0; i < ImuModelT::kXBlockDims[j]; ++i) {
//      std::cout << "xblock " << j << " dim " << i << "\n";
//      costFunction.setRedo(false);
//      Eigen::Matrix<double, 15, 1> errorLin;
//      parameters[ImuErrorT::Index::unitgW + j + 1][i] += eps;
//      costFunction.Evaluate(parameters, errorLin.data(), nullptr);
//      Eigen::Matrix<double, 15, 1> dlin = errorLin - residuals;
//      std::cout << "redo for " << j << " " << i << "\n";
//      costFunction.setRedo(true);
//      Eigen::Matrix<double, 15, 1> errorNonlin;
//      costFunction.Evaluate(parameters, errorNonlin.data(), nullptr);
//      Eigen::Matrix<double, 15, 1> dnonlin = errorNonlin - residuals;
//      parameters[ImuErrorT::Index::unitgW + j + 1][i] -= eps;
//      std::cout << "dlin " << dlin.transpose() << "\n";
//      std::cout << "dnonlin " << dnonlin.transpose() << "\n";
//      std::cout << "lin-nonlin " << (dlin - dnonlin).transpose() << "\n";
//      EXPECT_LT((dlin - dnonlin).lpNorm<Eigen::Infinity>(), eps * 0.5);
//    }
//  }

  constexpr double jacobianTolerance = ImuModelT::kJacobianTolerance;
  EXPECT_LT((Jp0min - Jp0minNumeric).norm(), jacobianTolerance) <<
      "diff " << (Jp0min - Jp0minNumeric).norm() << " >= tol "
              << jacobianTolerance << "\nwhere minimal Jacobian 0 = \n"
              << Jp0min << std::endl
              << "numDiff minimal Jacobian 0 = \n"
              << Jp0minNumeric;
  EXPECT_LT((Jp0 - Jp0Numeric).norm(), jacobianTolerance) <<
                    "Jacobian 0 = \n"
                        << Jp0 << std::endl
                        << "numDiff Jacobian 0 = \n"
                        << Jp0Numeric;

  double diffNorm = (Jv0min - Jv0minNumeric).template lpNorm<Eigen::Infinity>();
  EXPECT_LT(diffNorm, jacobianTolerance) <<
                    "minimal Jacobian v0 = \n"
                        << Jv0min << std::endl
                        << "numDiff minimal Jacobian v0 = \n"
                        << Jv0minNumeric << "\nDiff inf norm " << diffNorm;

  // std::cout << "minimal Jacobian v0 = \n"<<Jv0min<<std::endl;
  // std::cout << "numDiff minimal Jacobian v0 =\n"<<Jv0minNumeric<<std::endl;

  diffNorm = (Jb0min - Jb0minNumeric).template lpNorm<Eigen::Infinity>();
  EXPECT_LT(diffNorm, jacobianTolerance) <<
                    "diff " << diffNorm << ">= tol " << jacobianTolerance
                            << "\nminimal Jacobian bias0 = \n"
                            << Jb0min << "\nnumDiff minimal Jacobian bias0 = \n"
                            << Jb0minNumeric << "\n";

  EXPECT_LT((Jp1min - Jp1minNumeric).norm(), jacobianTolerance) <<
                    "minimal Jacobian p1 = \n"
                        << Jp1min << std::endl
                        << "numDiff minimal Jacobian p1 = \n"
                        << Jp1minNumeric;

  EXPECT_LT((Jp1 - Jp1Numeric).norm(), jacobianTolerance) <<
                    "Jacobian p1 = \n"
                        << Jp1 << std::endl
                        << "numDiff Jacobian p1 = \n"
                        << Jp1Numeric;

  diffNorm = (Jv1min - Jv1minNumeric).template lpNorm<Eigen::Infinity>();
  EXPECT_LT(diffNorm, jacobianTolerance) <<
                    "minimal Jacobian v1 = \n"
                        << Jv1min << std::endl
                        << "numDiff minimal Jacobian v1 = \n"
                        << Jv1minNumeric << "\nDiff inf norm " << diffNorm;

  diffNorm = (Jb1min - Jb1minNumeric).template lpNorm<Eigen::Infinity>();
  EXPECT_LT(diffNorm, jacobianTolerance) << "minimal Jacobian b1 = \n"
                        << Jb1min << std::endl
                        << "numDiff minimal Jacobian b1 = \n"
                        << Jb1minNumeric << "\nDiff inf norm " << diffNorm;

  EXPECT_LT((Jgmin - JgminNumeric).norm(), jacobianTolerance) <<
                    "minimal Jacobian g = \n"
                        << Jgmin << std::endl
                        << "numDiff minimal Jacobian g = \n"
                        << JgminNumeric;

  EXPECT_LT((Jg - JgNumeric).norm(), jacobianTolerance) <<
                    "Jacobian g = \n"
                        << Jg << std::endl
                        << "numDiff Jacobian g = \n"
                        << JgNumeric;

  for (size_t i = 0u; i < ImuModelT::kXBlockDims.size(); ++i) {
    double diffNorm =
        (*jacPtrs[i] - *jacNumericPtrs[i]).template lpNorm<Eigen::Infinity>();
    EXPECT_LT(diffNorm, 1e-2)
        << "For XParam " << i << ", numeric Jacobian differs by " << diffNorm
        << " from the analytic one."
        << "XParam " << i << " Jacobian =\n"
        << *jacPtrs[i] << "\nnumDiff Jacobian =\n"
        << *jacNumericPtrs[i] << "\n";

    diffNorm =
        (*jacMinPtrs[i] - *jacMinNumericPtrs[i]).template lpNorm<Eigen::Infinity>();
    EXPECT_LT(diffNorm, 1e-2)
        << "For XParam " << i << ", numeric Jacobian differs by " << diffNorm
        << " from the analytic one."
        << "Minimal XParam " << i << " Jacobian =\n"
        << *jacMinPtrs[i] << "\nnumDiff Jacobian =\n"
        << *jacMinNumericPtrs[i] << "\n";
  }
  return true;
}


template <typename ImuModelT>
bool checkJacobians(const ImuErrorConstBias<ImuModelT> &costFunction,
                    double *const *parameters) {
  typedef ImuErrorConstBias<ImuModelT> ImuErrorT;
  double *jacobians[ImuErrorT::Index::extra + ImuModelT::kXBlockDims.size()];
  Eigen::Matrix<double, ImuErrorT::kNumResiduals, 7, Eigen::RowMajor> Jp0;
  Eigen::Matrix<double, ImuErrorT::kNumResiduals, 3, Eigen::RowMajor> Jv0;
  Eigen::Matrix<double, ImuErrorT::kNumResiduals, 6, Eigen::RowMajor> Jb0;
  Eigen::Matrix<double, ImuErrorT::kNumResiduals, 7, Eigen::RowMajor> Jp1;
  Eigen::Matrix<double, ImuErrorT::kNumResiduals, 3, Eigen::RowMajor> Jv1;
  Eigen::Matrix<double, ImuErrorT::kNumResiduals, 3, Eigen::RowMajor> Jg;
  jacobians[ImuErrorT::Index::T_WB0] = Jp0.data();
  jacobians[ImuErrorT::Index::v_WB0] = Jv0.data();
  jacobians[ImuErrorT::Index::bgBa0] = Jb0.data();
  jacobians[ImuErrorT::Index::T_WB1] = Jp1.data();
  jacobians[ImuErrorT::Index::v_WB1] = Jv1.data();
  jacobians[ImuErrorT::Index::unitgW] = Jg.data();

  double *
      jacobiansMinimal[ImuErrorT::Index::extra + ImuModelT::kXBlockDims.size()];
  Eigen::Matrix<double, ImuErrorT::kNumResiduals, 6, Eigen::RowMajor> Jp0min;
  Eigen::Matrix<double, ImuErrorT::kNumResiduals, 3, Eigen::RowMajor> Jv0min;
  Eigen::Matrix<double, ImuErrorT::kNumResiduals, 6, Eigen::RowMajor> Jb0min;

  Eigen::Matrix<double, ImuErrorT::kNumResiduals, 6, Eigen::RowMajor> Jp1min;
  Eigen::Matrix<double, ImuErrorT::kNumResiduals, 3, Eigen::RowMajor> Jv1min;
  Eigen::Matrix<double, ImuErrorT::kNumResiduals, 2, Eigen::RowMajor> Jgmin;
  jacobiansMinimal[ImuErrorT::Index::T_WB0] = Jp0min.data();
  jacobiansMinimal[ImuErrorT::Index::v_WB0] = Jv0min.data();
  jacobiansMinimal[ImuErrorT::Index::bgBa0] = Jb0min.data();
  jacobiansMinimal[ImuErrorT::Index::T_WB1] = Jp1min.data();
  jacobiansMinimal[ImuErrorT::Index::v_WB1] = Jv1min.data();
  jacobiansMinimal[ImuErrorT::Index::unitgW] = Jgmin.data();

  std::vector<std::shared_ptr<Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>>
      jacPtrs;
  std::vector<std::shared_ptr<Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>>
      jacMinPtrs;
  for (size_t i = 0u; i < ImuModelT::kXBlockDims.size(); ++i) {
    std::shared_ptr<Eigen::Matrix<double, -1, -1, Eigen::RowMajor>> jacPtr(
        new Eigen::Matrix<double, -1, -1, Eigen::RowMajor>(
            ImuErrorT::kNumResiduals, ImuModelT::kXBlockDims[i]));
    jacobians[i + ImuErrorT::Index::extra] = jacPtr->data();

    std::shared_ptr<Eigen::Matrix<double, -1, -1, Eigen::RowMajor>> jacMinPtr(
        new Eigen::Matrix<double, -1, -1, Eigen::RowMajor>(
            ImuErrorT::kNumResiduals, ImuModelT::kXBlockMinDims[i]));
    jacobiansMinimal[i + ImuErrorT::Index::extra] = jacMinPtr->data();
    jacPtrs.push_back(jacPtr);
    jacMinPtrs.push_back(jacMinPtr);
  }

  Eigen::Matrix<double, ImuErrorT::kNumResiduals, 1> residuals;

  // evaluate twice to be sure that we will be using the linearisation of the
  // biases (i.e. no preintegrals redone)
  costFunction.Evaluate(parameters, residuals.data(), nullptr);
  costFunction.EvaluateWithMinimalJacobians(parameters, residuals.data(),
                                            jacobians, jacobiansMinimal);

  double *
      jacobiansNumeric[ImuErrorT::Index::extra + ImuModelT::kXBlockDims.size()];
  Eigen::Matrix<double, ImuErrorT::kNumResiduals, 7, Eigen::RowMajor>
      Jp0Numeric;
  Eigen::Matrix<double, ImuErrorT::kNumResiduals, 3, Eigen::RowMajor>
      Jv0Numeric;
  Eigen::Matrix<double, ImuErrorT::kNumResiduals, 6, Eigen::RowMajor>
      Jb0Numeric;
  Eigen::Matrix<double, ImuErrorT::kNumResiduals, 7, Eigen::RowMajor>
      Jp1Numeric;
  Eigen::Matrix<double, ImuErrorT::kNumResiduals, 3, Eigen::RowMajor>
      Jv1Numeric;
  Eigen::Matrix<double, ImuErrorT::kNumResiduals, 3, Eigen::RowMajor> JgNumeric;
  jacobiansNumeric[ImuErrorT::Index::T_WB0] = Jp0Numeric.data();
  jacobiansNumeric[ImuErrorT::Index::v_WB0] = Jv0Numeric.data();
  jacobiansNumeric[ImuErrorT::Index::bgBa0] = Jb0Numeric.data();
  jacobiansNumeric[ImuErrorT::Index::T_WB1] = Jp1Numeric.data();
  jacobiansNumeric[ImuErrorT::Index::v_WB1] = Jv1Numeric.data();
  jacobiansNumeric[ImuErrorT::Index::unitgW] = JgNumeric.data();

  double *jacobiansMinimalNumeric[ImuErrorT::Index::extra +
                                  ImuModelT::kXBlockDims.size()];
  Eigen::Matrix<double, ImuErrorT::kNumResiduals, 6, Eigen::RowMajor>
      Jp0minNumeric;
  Eigen::Matrix<double, ImuErrorT::kNumResiduals, 3, Eigen::RowMajor>
      Jv0minNumeric;
  Eigen::Matrix<double, ImuErrorT::kNumResiduals, 6, Eigen::RowMajor>
      Jb0minNumeric;

  Eigen::Matrix<double, ImuErrorT::kNumResiduals, 6, Eigen::RowMajor>
      Jp1minNumeric;
  Eigen::Matrix<double, ImuErrorT::kNumResiduals, 3, Eigen::RowMajor>
      Jv1minNumeric;

  Eigen::Matrix<double, ImuErrorT::kNumResiduals, 2, Eigen::RowMajor>
      JgminNumeric;
  jacobiansMinimalNumeric[ImuErrorT::Index::T_WB0] = Jp0minNumeric.data();
  jacobiansMinimalNumeric[ImuErrorT::Index::v_WB0] = Jv0minNumeric.data();
  jacobiansMinimalNumeric[ImuErrorT::Index::bgBa0] = Jb0minNumeric.data();
  jacobiansMinimalNumeric[ImuErrorT::Index::T_WB1] = Jp1minNumeric.data();
  jacobiansMinimalNumeric[ImuErrorT::Index::v_WB1] = Jv1minNumeric.data();
  jacobiansMinimalNumeric[ImuErrorT::Index::unitgW] = JgminNumeric.data();

  std::vector<std::shared_ptr<Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>>
      jacNumericPtrs;
  std::vector<std::shared_ptr<Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>>
      jacMinNumericPtrs;
  for (size_t i = 0u; i < ImuModelT::kXBlockDims.size(); ++i) {
    std::shared_ptr<Eigen::Matrix<double, -1, -1, Eigen::RowMajor>> jacPtr(
        new Eigen::Matrix<double, -1, -1, Eigen::RowMajor>(
            ImuErrorT::kNumResiduals, ImuModelT::kXBlockDims[i]));
    jacobiansNumeric[i + ImuErrorT::Index::extra] = jacPtr->data();

    std::shared_ptr<Eigen::Matrix<double, -1, -1, Eigen::RowMajor>> jacMinPtr(
        new Eigen::Matrix<double, -1, -1, Eigen::RowMajor>(
            ImuErrorT::kNumResiduals, ImuModelT::kXBlockMinDims[i]));
    jacobiansMinimalNumeric[i + ImuErrorT::Index::extra] = jacMinPtr->data();
    jacNumericPtrs.push_back(jacPtr);
    jacMinNumericPtrs.push_back(jacMinPtr);
  }

  // compare residuals evaluated with Imu param lin point and
  // residuals evaluated with actual Imu params.
//  const double eps = 1e-5;
//  for (size_t j = 0; j < ImuModelT::kXBlockDims.size(); ++j) {
//    for (int i = 0; i < ImuModelT::kXBlockDims[j]; ++i) {
//      std::cout << "xblock " << j << " dim " << i << "\n";
//      costFunction.setRedo(false);
//      Eigen::Matrix<double, 9, 1> errorLin;
//      parameters[ImuErrorT::Index::unitgW + j + 1][i] += eps;
//      costFunction.Evaluate(parameters, errorLin.data(), nullptr);
//      Eigen::Matrix<double, 9, 1> dlin = errorLin - residuals;
//      std::cout << "redo for " << j << " " << i << "\n";
//      costFunction.setRedo(true);
//      Eigen::Matrix<double, 9, 1> errorNonlin;
//      costFunction.Evaluate(parameters, errorNonlin.data(), nullptr);
//      Eigen::Matrix<double, 9, 1> dnonlin = errorNonlin - residuals;
//      parameters[ImuErrorT::Index::unitgW + j + 1][i] -= eps;
//      std::cout << "dlin " << dlin.transpose() << "\n";
//      std::cout << "dnonlin " << dnonlin.transpose() << "\n";
//      std::cout << "lin-nonlin " << (dlin - dnonlin).transpose() << "\n";
//      EXPECT_LT((dlin - dnonlin).lpNorm<Eigen::Infinity>(), eps * 0.5);
//    }
//  }

  costFunction.setReweight(false); // disable weighting update.
  Eigen::Matrix<double, ImuErrorT::kNumResiduals, 1> residualsNumeric;
  costFunction.EvaluateWithMinimalJacobiansNumeric(
      parameters, residualsNumeric.data(), jacobiansNumeric,
      jacobiansMinimalNumeric);

  constexpr double jacobianTolerance = ImuModelT::kJacobianTolerance;
  EXPECT_LT((Jp0min - Jp0minNumeric).norm(), jacobianTolerance) <<
      "diff " << (Jp0min - Jp0minNumeric).norm() << " >= tol "
              << jacobianTolerance << "\nwhere minimal Jacobian 0 = \n"
              << Jp0min << std::endl
              << "numDiff minimal Jacobian 0 = \n"
              << Jp0minNumeric;
  EXPECT_LT((Jp0 - Jp0Numeric).norm(), jacobianTolerance) <<
                    "Jacobian 0 = \n"
                        << Jp0 << std::endl
                        << "numDiff Jacobian 0 = \n"
                        << Jp0Numeric;

  double diffNorm = (Jv0min - Jv0minNumeric).template lpNorm<Eigen::Infinity>();
  EXPECT_LT(diffNorm, jacobianTolerance) <<
                    "minimal Jacobian v0 = \n"
                        << Jv0min << std::endl
                        << "numDiff minimal Jacobian v0 = \n"
                        << Jv0minNumeric << "\nDiff inf norm " << diffNorm;

  // std::cout << "minimal Jacobian v0 = \n"<<Jv0min<<std::endl;
  // std::cout << "numDiff minimal Jacobian v0 =\n"<<Jv0minNumeric<<std::endl;

  diffNorm = (Jb0min - Jb0minNumeric).template lpNorm<Eigen::Infinity>();
  EXPECT_LT(diffNorm, jacobianTolerance) <<
                    "diff " << diffNorm << ">= tol " << jacobianTolerance
                            << "\nminimal Jacobian bias0 = \n"
                            << Jb0min << "\nnumDiff minimal Jacobian bias0 = \n"
                            << Jb0minNumeric << "\n";

  EXPECT_LT((Jp1min - Jp1minNumeric).norm(), jacobianTolerance) <<
                    "minimal Jacobian p1 = \n"
                        << Jp1min << std::endl
                        << "numDiff minimal Jacobian p1 = \n"
                        << Jp1minNumeric;

  EXPECT_LT((Jp1 - Jp1Numeric).norm(), jacobianTolerance) <<
                    "Jacobian p1 = \n"
                        << Jp1 << std::endl
                        << "numDiff Jacobian p1 = \n"
                        << Jp1Numeric;

  diffNorm = (Jv1min - Jv1minNumeric).template lpNorm<Eigen::Infinity>();
  EXPECT_LT(diffNorm, jacobianTolerance) <<
                    "minimal Jacobian v1 = \n"
                        << Jv1min << std::endl
                        << "numDiff minimal Jacobian v1 = \n"
                        << Jv1minNumeric << "\nDiff inf norm " << diffNorm;

  EXPECT_LT((Jgmin - JgminNumeric).norm(), jacobianTolerance) <<
                    "minimal Jacobian g = \n"
                        << Jgmin << std::endl
                        << "numDiff minimal Jacobian g = \n"
                        << JgminNumeric;

  EXPECT_LT((Jg - JgNumeric).norm(), jacobianTolerance) <<
                    "Jacobian g = \n"
                        << Jg << std::endl
                        << "numDiff Jacobian g = \n"
                        << JgNumeric;

  for (size_t i = 0u; i < ImuModelT::kXBlockDims.size(); ++i) {
    double diffNorm =
        (*jacPtrs[i] - *jacNumericPtrs[i]).template lpNorm<Eigen::Infinity>();
    EXPECT_LT(diffNorm, 1e-2)
        << "For XParam " << i << ", numeric Jacobian differs by " << diffNorm
        << " from the analytic one."
        << "XParam " << i << " Jacobian =\n"
        << *jacPtrs[i] << "\nnumDiff Jacobian =\n"
        << *jacNumericPtrs[i] << "\n";

    diffNorm =
        (*jacMinPtrs[i] - *jacMinNumericPtrs[i]).template lpNorm<Eigen::Infinity>();
    EXPECT_LT(diffNorm, 1e-2)
        << "For XParam " << i << ", numeric Jacobian differs by " << diffNorm
        << " from the analytic one."
        << "Minimal XParam " << i << " Jacobian =\n"
        << *jacMinPtrs[i] << "\nnumDiff Jacobian =\n"
        << *jacMinNumericPtrs[i] << "\n";
  }
  return true;
}

} // namespace ceres
} // namespace okvis

#endif // CHECKIMUERROR_H
