/**
 * @file    MatrixPseudoInverse.cpp
 * @brief   Functions for matrix pseudo inverse, and matrix square root.
 * @author  Jianzhu Huai
 */

#ifndef INCLUDE_OKVIS_MATRIX_PSEUDO_INVERSE_HPP
#define INCLUDE_OKVIS_MATRIX_PSEUDO_INVERSE_HPP

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include <okvis/assert_macros.hpp>

namespace okvis {

/**
 * \brief Compute the square root of a matrix using the LDLt decomposition for
 * square, positive semidefinite matrices
 *
 * To reconstruct the input matrix, \f$ \mathbf A \f$, from the returned matrix,
 * \f$ \mathbf S \f$, use, \f$ \mathbf A = \mathbf S \mathbf S^T \f$.
 *
 *
 * @param inMatrix      The square matrix whose square root should be computed.
 * @param outMatrixSqrt The output square root.
 */
template <typename DERIVED1, typename DERIVED2>
/*Eigen::ComputationInfo*/ void
computeMatrixSqrt(const Eigen::MatrixBase<DERIVED1> &inMatrix,
                  const Eigen::MatrixBase<DERIVED2> &outMatrixSqrt) {
  OKVIS_ASSERT_EQ_DBG(std::runtime_error, inMatrix.rows(), inMatrix.cols(),
                      "This method is only valid for square input matrices");

  DERIVED2 &result = const_cast<DERIVED2 &>(outMatrixSqrt.derived());

  // This is tricky. Using the output matrix type causes the input matrix
  // type to be upgraded to a real numeric matrix. This is useful if,
  // for example, the inMatrix is something like Eigen::Matrix3d::Identity(),
  // which is not an actual matrix. Using DERIVED1 as the template argument
  // in that case will cause a firestorm of compiler errors.
  Eigen::LDLT<DERIVED2> ldlt(inMatrix.derived());
  result = ldlt.matrixL();
  result = ldlt.transpositionsP().transpose() * result;
  result *= ldlt.vectorD().array().sqrt().matrix().asDiagonal();

  // return ldlt.info();
}

class MatrixPseudoInverse
{
public:
  OKVIS_DEFINE_EXCEPTION(Exception,std::runtime_error)
  /**
   * @brief Pseudo inversion of a symmetric matrix.
   * @warning   This uses Eigen-decomposition, it assumes the input is symmetric positive semi-definite
   *            (negative Eigenvalues are set to zero).
   * @tparam Derived Matrix type (auto-deducible).
   * @param[in] a Input Matrix
   * @param[out] result Output, i.e. pseudo-inverse.
   * @param[in] epsilon The tolerance.
   * @param[out] rank Optional rank.
   * @return
   */
  template<typename Derived>
  static bool pseudoInverseSymm(
      const Eigen::MatrixBase<Derived>&a,
      const Eigen::MatrixBase<Derived>&result, double epsilon =
          std::numeric_limits<typename Derived::Scalar>::epsilon(), int * rank = 0);

  template<typename Derived1, typename Derived2>
  static bool inverseSymm(
      const Eigen::MatrixBase<Derived1> &A, const Eigen::MatrixBase<Derived2> &invA);

  /**
   * @brief Pseudo inversion and square root of a symmetric matrix.
   * @warning   This uses Eigen-decomposition, it assumes the input is symmetric positive semi-definite
   *            (negative Eigenvalues are set to zero). Also if the input is positive semi-definite,
   *            its zero eigenvalues are at the lower part of the diagonal.
   * @tparam Derived Matrix type (auto-deducible).
   * @param[in] a Input Matrix, \f$A\f$.
   * @param[out] result Output, \f$L\f$, i.e. \f$A^{-1} = L L^*\f$, note L is not a lower triangular matrix in general.
   * @param[in] epsilon The tolerance.
   * @param[out] rank The rank, if of interest.
   * @return
   */
  template<typename Derived>
  static bool pseudoInverseSymmSqrt(
      const Eigen::MatrixBase<Derived>&a,
      const Eigen::MatrixBase<Derived>&result, double epsilon =
          std::numeric_limits<typename Derived::Scalar>::epsilon(),
      int* rank = NULL);

  template <typename Derived>
  static bool pseudoInverseSymmSqrt(
      const Eigen::MatrixBase<Derived> &A, const Eigen::MatrixBase<Derived> &L,
      const Eigen::MatrixBase<Derived> &invA,
      double epsilon = std::numeric_limits<typename Derived::Scalar>::epsilon(),
      int *rank = NULL);

  /**
   * @brief take inverse of matrix A, invA, and compute the cholesky factor L of invA.
   * invA = L * L'.
   */
  template<typename Derived1, typename Derived2>
  static bool inverseSymmSqrt(
      const Eigen::MatrixBase<Derived1> &A, const Eigen::MatrixBase<Derived2> &L,
      const Eigen::MatrixBase<Derived2> &invA);

  /**
   * @brief Pseudo square root of a symmetric matrix.
   * @warning   This uses Eigen-decomposition, it assumes the input is symmetric positive semi-definite
   *            (negative Eigenvalues are set to zero). Also if the input is positive semi-definite,
   *            its zero eigenvalues are at the lower part of the diagonal.
   * @tparam Derived Matrix type (auto-deducible).
   * @param[in] a Input Matrix, \f$A\f$.
   * @param[out] result Output, \f$L\f$, i.e. \f$A = L L^*\f$, note L is not necessarily lower triangular.
   * @param[in] epsilon The tolerance.
   * @param[out] rank The rank, if of interest.
   * @return
   */
  template<typename Derived>
  static bool pseudoSymmSqrt(
      const Eigen::MatrixBase<Derived>&a,
      const Eigen::MatrixBase<Derived>&result, double epsilon =
          std::numeric_limits<typename Derived::Scalar>::epsilon(),
      int* rank = NULL);

  /**
   * @brief Block-wise pseudo inversion of a symmetric matrix with non-zero diagonal blocks.
   * @warning   This uses Eigen-decomposition, it assumes the input is symmetric positive semi-definite
   *            (negative Eigenvalues are set to zero).
   * @tparam Derived Matrix type (auto-deducible).
   * @tparam blockDim The block size of the diagonal blocks.
   * @param[in] M_in Input Matrix
   * @param[out] M_out Output, i.e. thepseudo-inverse.
   * @param[in] epsilon The tolerance.
   * @return
   */
  template<typename Derived, int blockDim>
  static void blockPinverse(
      const Eigen::MatrixBase<Derived>& M_in,
      const Eigen::MatrixBase<Derived>& M_out, double epsilon =
          std::numeric_limits<typename Derived::Scalar>::epsilon());


  /**
   * @brief Block-wise pseudo inversion and square root
   *        of a symmetric matrix with non-zero diagonal blocks.
   * @warning   This uses Eigen-decomposition, it assumes the input is symmetric positive semi-definite
   *            (negative Eigenvalues are set to zero).
   * @tparam Derived Matrix type (auto-deducible).
   * @tparam blockDim The block size of the diagonal blocks.
   * @param[in] M_in Input Matrix
   * @param[out] M_out Output, i.e. the SVD inverse root of a pseudo-inverse.
   * @param[in] epsilon The tolerance.
   * @return
   */
  template<typename Derived, int blockDim>
  static void blockPinverseSqrt(
      const Eigen::MatrixBase<Derived>& M_in,
      const Eigen::MatrixBase<Derived>& M_out, double epsilon =
          std::numeric_limits<typename Derived::Scalar>::epsilon());

};

// Pseudo inversion of a symmetric matrix.
// attention: this uses Eigen-decomposition, it assumes the input is symmetric positive semi-definite
// (negative Eigenvalues are set to zero)
template<typename Derived>
bool MatrixPseudoInverse::pseudoInverseSymm(
    const Eigen::MatrixBase<Derived>&a, const Eigen::MatrixBase<Derived>&result,
    double epsilon, int * rank) {

  OKVIS_ASSERT_TRUE_DBG(Exception, a.rows() == a.cols(),
                        "matrix supplied is not quadratic");

  Eigen::SelfAdjointEigenSolver<Derived> saes(a);

  typename Derived::Scalar tolerance = epsilon * a.cols()
      * saes.eigenvalues().array().maxCoeff();

  const_cast<Eigen::MatrixBase<Derived>&>(result) = (saes.eigenvectors())
      * Eigen::VectorXd(
          (saes.eigenvalues().array() > tolerance).select(
              saes.eigenvalues().array().inverse(), 0)).asDiagonal()
      * (saes.eigenvectors().transpose());

  if (rank) {
    *rank = 0;
    for (int i = 0; i < a.rows(); ++i) {
      if (saes.eigenvalues()[i] > tolerance)
        (*rank)++;
    }
  }

  return true;
}

template<typename Derived>
bool MatrixPseudoInverse::pseudoInverseSymmSqrt(
    const Eigen::MatrixBase<Derived>&a, const Eigen::MatrixBase<Derived>&result,
    double epsilon, int * rank) {

  OKVIS_ASSERT_TRUE_DBG(Exception, a.rows() == a.cols(),
                        "matrix supplied is not quadratic");

  Eigen::SelfAdjointEigenSolver<Derived> saes(a);

  typename Derived::Scalar tolerance = epsilon * a.cols()
      * saes.eigenvalues().array().maxCoeff();

  const_cast<Eigen::MatrixBase<Derived>&>(result) = (saes.eigenvectors())
      * Eigen::VectorXd(
          Eigen::VectorXd(
              (saes.eigenvalues().array() > tolerance).select(
                  saes.eigenvalues().array().inverse(), 0)).array().sqrt())
          .asDiagonal();

  if (rank) {
    *rank = 0;
    for (int i = 0; i < a.rows(); ++i) {
      if (saes.eigenvalues()[i] > tolerance)
        (*rank)++;
    }
  }

  return true;
}

template<typename Derived>
bool MatrixPseudoInverse::pseudoInverseSymmSqrt(
    const Eigen::MatrixBase<Derived>&A, const Eigen::MatrixBase<Derived> &L,
    const Eigen::MatrixBase<Derived> &invA,
    double epsilon, int * rank) {

  OKVIS_ASSERT_TRUE_DBG(Exception, A.rows() == A.cols(),
                        "matrix supplied is not quadratic");

  Eigen::SelfAdjointEigenSolver<Derived> saes(A);

  typename Derived::Scalar tolerance = epsilon * A.cols()
      * saes.eigenvalues().array().maxCoeff();

  const_cast<Eigen::MatrixBase<Derived>&>(L) = (saes.eigenvectors())
      * Eigen::VectorXd(
          Eigen::VectorXd(
              (saes.eigenvalues().array() > tolerance).select(
                  saes.eigenvalues().array().inverse(), 0)).array().sqrt())
          .asDiagonal();

  const_cast<Eigen::MatrixBase<Derived>&>(invA) = (saes.eigenvectors())
      * Eigen::VectorXd(
          (saes.eigenvalues().array() > tolerance).select(
              saes.eigenvalues().array().inverse(), 0)).asDiagonal()
      * (saes.eigenvectors().transpose());

  if (rank) {
    *rank = 0;
    for (int i = 0; i < A.rows(); ++i) {
      if (saes.eigenvalues()[i] > tolerance)
        (*rank)++;
    }
  }

  return true;
}

template<typename Derived1, typename Derived2>
bool MatrixPseudoInverse::inverseSymm(
    const Eigen::MatrixBase<Derived1> &A, const Eigen::MatrixBase<Derived2> &invA) {
  OKVIS_ASSERT_TRUE_DBG(Exception, A.rows() == A.cols(),
                        "matrix supplied is not quadratic");
  Derived2 &refInvA = const_cast<Derived2 &>(invA.derived());
  refInvA.setIdentity();
  A.llt().solveInPlace(refInvA);
  return true;
}

template<typename Derived1, typename Derived2>
bool MatrixPseudoInverse::inverseSymmSqrt(
    const Eigen::MatrixBase<Derived1> &A, const Eigen::MatrixBase<Derived2> &L,
    const Eigen::MatrixBase<Derived2> &invA) {
  OKVIS_ASSERT_TRUE_DBG(Exception, A.rows() == A.cols(),
                        "matrix supplied is not quadratic");
  Derived2 &refInvA = const_cast<Derived2 &>(invA.derived());
  refInvA.setIdentity();
  A.llt().solveInPlace(refInvA);

  Derived2 &refL = const_cast<Derived2 &>(L.derived());
  Eigen::LLT<Derived2> llt(refInvA);
  refL = llt.matrixL();
  return true;
}

template <typename Derived>
bool MatrixPseudoInverse::pseudoSymmSqrt(
    const Eigen::MatrixBase<Derived> &a,
    const Eigen::MatrixBase<Derived> &result, double epsilon, int *rank) {

  OKVIS_ASSERT_TRUE_DBG(Exception, a.rows() == a.cols(),
                        "matrix supplied is not quadratic");

  Eigen::SelfAdjointEigenSolver<Derived> saes(a);

  typename Derived::Scalar tolerance =
      epsilon * a.cols() * saes.eigenvalues().array().maxCoeff();

  const_cast<Eigen::MatrixBase<Derived> &>(result) =
      (saes.eigenvectors()) *
      Eigen::VectorXd(
          Eigen::VectorXd((saes.eigenvalues().array() > tolerance)
                              .select(saes.eigenvalues().array(), 0))
              .array()
              .sqrt())
          .asDiagonal();

  if (rank) {
    *rank = 0;
    for (int i = 0; i < a.rows(); ++i) {
      if (saes.eigenvalues()[i] > tolerance)
        (*rank)++;
    }
  }

  return true;
}

// Block-wise pseudo inversion of a symmetric matrix with non-zero diagonal blocks.
// attention: this uses Eigen-decomposition, it assumes the input is symmetric positive semi-definite
// (negative Eigenvalues are set to zero)
template<typename Derived, int blockDim>
void MatrixPseudoInverse::blockPinverse(
    const Eigen::MatrixBase<Derived>& M_in,
    const Eigen::MatrixBase<Derived>& M_out, double epsilon) {

  OKVIS_ASSERT_TRUE_DBG(Exception, M_in.rows() == M_in.cols(),
                        "matrix supplied is not quadratic");

  const_cast<Eigen::MatrixBase<Derived>&>(M_out).resize(M_in.rows(),
                                                        M_in.rows());
  const_cast<Eigen::MatrixBase<Derived>&>(M_out).setZero();
  for (int i = 0; i < M_in.cols(); i += blockDim) {
    Eigen::Matrix<double, blockDim, blockDim> inv;
    const Eigen::Matrix<double, blockDim, blockDim> in = M_in
        .template block<blockDim, blockDim>(i, i);
    //const Eigen::Matrix<double,blockDim,blockDim> in1=0.5*(in+in.transpose());
    pseudoInverseSymm(in, inv, epsilon);
    const_cast<Eigen::MatrixBase<Derived>&>(M_out)
        .template block<blockDim, blockDim>(i, i) = inv;
  }
}

template<typename Derived, int blockDim>
void MatrixPseudoInverse::blockPinverseSqrt(
    const Eigen::MatrixBase<Derived>& M_in,
    const Eigen::MatrixBase<Derived>& M_out, double epsilon) {

  OKVIS_ASSERT_TRUE_DBG(Exception, M_in.rows() == M_in.cols(),
                        "matrix supplied is not quadratic");

  const_cast<Eigen::MatrixBase<Derived>&>(M_out).resize(M_in.rows(),
                                                        M_in.rows());
  const_cast<Eigen::MatrixBase<Derived>&>(M_out).setZero();
  for (int i = 0; i < M_in.cols(); i += blockDim) {
    Eigen::Matrix<double, blockDim, blockDim> inv;
    const Eigen::Matrix<double, blockDim, blockDim> in = M_in
        .template block<blockDim, blockDim>(i, i);
    //const Eigen::Matrix<double,blockDim,blockDim> in1=0.5*(in+in.transpose());
    pseudoInverseSymmSqrt(in, inv, epsilon);
    const_cast<Eigen::MatrixBase<Derived>&>(M_out)
        .template block<blockDim, blockDim>(i, i) = inv;
  }
}

} // namespace okvis
#endif // INCLUDE_OKVIS_MATRIX_PSEUDO_INVERSE_HPP
