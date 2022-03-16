#ifndef MATRIXUTILITIES_H
#define MATRIXUTILITIES_H

#include <Eigen/Core>

namespace swift_vio {
template <typename T>
void vectorToLowerTriangularMatrix(const T* data, int startIndex, Eigen::Matrix<T, 3, 3>* mat33) {
  (*mat33)(0, 0) = data[startIndex];
  (*mat33)(0, 1) = 0;
  (*mat33)(0, 2) = 0;
  (*mat33)(1, 0) = data[startIndex + 1];
  (*mat33)(1, 1) = data[startIndex + 2];
  (*mat33)(1, 2) = 0;
  (*mat33)(2, 0) = data[startIndex + 3];
  (*mat33)(2, 1) = data[startIndex + 4];
  (*mat33)(2, 2) = data[startIndex + 5];
}

template <typename T>
void lowerTriangularMatrixToVector(const Eigen::Matrix<T, 3, 3> &mat33, T *data, int startIndex) {
  data[startIndex] = mat33(0, 0);
  data[startIndex + 1] = mat33(1, 0);
  data[startIndex + 2] = mat33(1, 1);
  data[startIndex + 3] = mat33(2, 0);
  data[startIndex + 4] = mat33(2, 1);
  data[startIndex + 5] = mat33(2, 2);
}

template <typename T>
void vectorToMatrix(const T* data, int startIndex, Eigen::Matrix<T, 3, 3>* mat33) {
  (*mat33)(0, 0) = data[startIndex];
  (*mat33)(0, 1) = data[startIndex + 1];
  (*mat33)(0, 2) = data[startIndex + 2];
  (*mat33)(1, 0) = data[startIndex + 3];
  (*mat33)(1, 1) = data[startIndex + 4];
  (*mat33)(1, 2) = data[startIndex + 5];
  (*mat33)(2, 0) = data[startIndex + 6];
  (*mat33)(2, 1) = data[startIndex + 7];
  (*mat33)(2, 2) = data[startIndex + 8];
}

template <typename T>
void matrixToVector(const Eigen::Matrix<T, 3, 3> &mat33, T* data, int startIndex) {
  data[startIndex] = (*mat33)(0, 0);
  data[startIndex + 1] = (*mat33)(0, 1);
  data[startIndex + 2] = (*mat33)(0, 2);
  data[startIndex + 3] = (*mat33)(1, 0);
  data[startIndex + 4] = (*mat33)(1, 1);
  data[startIndex + 5] = (*mat33)(1, 2);
  data[startIndex + 6] = (*mat33)(2, 0);
  data[startIndex + 7] = (*mat33)(2, 1);
  data[startIndex + 8] = (*mat33)(2, 2);
}

template <typename T>
void invertLowerTriangularMatrix(const T* data, int startIndex, Eigen::Matrix<T, 3, 3>* mat33) {
  //  syms a b c d e f positive
  //  g = [a, 0, 0, b, c, 0, d, e, f]
  //  [ a, 0, 0]
  //  [ b, c, 0]
  //  [ d, e, f]
  //  inv(g)
  //  [                 1/a,        0,   0]
  //  [            -b/(a*c),      1/c,   0]
  //  [ (b*e - c*d)/(a*c*f), -e/(c*f), 1/f]
  (*mat33)(0, 0) = 1 / data[startIndex];
  (*mat33)(0, 1) = 0;
  (*mat33)(0, 2) = 0;
  (*mat33)(1, 0) = - data[startIndex + 1] / (data[startIndex] * data[startIndex + 2]);
  (*mat33)(1, 1) = 1 / data[startIndex + 2];
  (*mat33)(1, 2) = 0;
  (*mat33)(2, 0) = (data[startIndex + 1] * data[startIndex + 4] -
      data[startIndex + 2] * data[startIndex + 3]) /
      (data[startIndex] * data[startIndex + 2] * data[startIndex + 5]);
  (*mat33)(2, 1) = - data[startIndex + 4] / (data[startIndex + 2] * data[startIndex + 5]);
  (*mat33)(2, 2) = 1 / data[startIndex + 5];
}

// \f$\frac{\partial{T_{3\times3} \vec{a}_{3}}}{\partial \vec{T}_9}\f$
template<typename Derived_T>
Eigen::Matrix<typename Eigen::internal::traits<Derived_T>::Scalar, 3, 9> dmatrix3_dvector9_multiply(
      Eigen::MatrixBase<Derived_T> const &rhs) {
    Eigen::Matrix<typename Eigen::internal::traits<Derived_T>::Scalar, 3, 9> m =
        Eigen::Matrix<typename Eigen::internal::traits<Derived_T>::Scalar, 3, 9>::Zero();
  m.template topLeftCorner<1, 3>() = rhs.transpose();
  m.template block<1, 3>(1, 3) = rhs.transpose();
  m.template block<1, 3>(2, 6) = rhs.transpose();
  return m;
}

/**
 * @brief Derivative of the product of a lower triangular matrix M and vector rhs
 * relative to the 6 parameters of M.
 * M = [a, 0, 0; b, c, 0; d, e, f];
 * matlab script
 * syms a b c d e f m n p
 * ltm = [a, 0, 0; b, c, 0; d, e, f];
 * v = [m; n; p];
 * r = ltm * v;
 * [diff(r, a), diff(r, b), diff(r, c), diff(r, d), diff(r, e), diff(r, f)]
 */
template<typename Derived_T>
Eigen::Matrix<typename Eigen::internal::traits<Derived_T>::Scalar, 3, 6> dltm3_dvector6_multiply(
    Eigen::MatrixBase<Derived_T> const &rhs) {
  Eigen::Matrix<typename Eigen::internal::traits<Derived_T>::Scalar, 3, 6> m =
      Eigen::Matrix<typename Eigen::internal::traits<Derived_T>::Scalar, 3, 6>::Zero();
  m(0, 0) = rhs[0];
  m(1, 1) = rhs[0];
  m(1, 2) = rhs[1];
  m(2, 3) = rhs[0];
  m(2, 4) = rhs[1];
  m(2, 5) = rhs[2];
  return m;
}

template <typename Derived_T>
void scaleBlockRows(const Eigen::Matrix<typename Eigen::internal::traits<Derived_T>::Scalar, 3, 3> &rotation,
                    int numBlockRows, Eigen::MatrixBase<Derived_T> *rhs) {
  for (int i = 0; i < numBlockRows; ++i) {
    rhs->template middleRows<3>(i * 3) = (rotation * rhs->template middleRows<3>(i * 3)).eval();
  }
}

template <typename Derived_T>
void scaleBlockCols(const Eigen::Matrix<typename Eigen::internal::traits<Derived_T>::Scalar, 3, 3> &rotation,
                    int numBlockCols, Eigen::MatrixBase<Derived_T> *rhs) {
  for (int i = 0; i < numBlockCols; ++i) {
    rhs->template middleCols<3>(i * 3) = (rhs->template middleCols<3>(i * 3) * rotation).eval();
  }
}

}  // namespace swift_vio

#endif // MATRIXUTILITIES_H
