/**
 * @file   MultipleTransformPointJacobian.h
 * @brief  Jacobians for T_1^{a_1} * ... * T_n^{a_2} * p.
 * where p is a 4D homogeneous point,
 * T = [Expmap(\alpha) * R   t + \delta t;
 *       0^T                 1].
 * @author Jianzhu Huai
 */

#ifndef INCLUDE_SWIFT_VIO_MULTIPLE_TRANSFORM_POINT_JACOBIAN_HPP
#define INCLUDE_SWIFT_VIO_MULTIPLE_TRANSFORM_POINT_JACOBIAN_HPP

#include <okvis/kinematics/Transformation.hpp>

#include <swift_vio/memory.h>
#include <swift_vio/InverseTransformPointJacobian.hpp>
#include <swift_vio/TransformPointJacobian.hpp>

namespace swift_vio {
struct TransformPointJacobianNode {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  TransformPointJacobianNode() {}
  okvis::kinematics::Transformation cumulativeLeftTransform_;
  Eigen::Vector4d point_;
  Eigen::Matrix<double, 4, 6> dpoint_dHeadTransform_;
};

/**
 * @brief The MultipleTransformPointJacobianNode struct
 * For q = T1^{a_1} * T2^{a_2} * ... * Tk^{a_k} * p, a_i \in {+1, -1},
 * p and q are 4D homogeneous points, Ti i = 1, 2, ..., k are Euclidean transforms.
 * first node contains
 * I=dq_dq, q, dq_d\delta T1
 * second node contains
 * T1^{a_1} = dq_dq1, q1 = T2^{a_2} * ... * Tk^{a_k} * p, dq1_d\delta T2
 * ...
 * second to last node contains
 * T1^{a_1} * T2^{a_2} * ... * T{k-1}^{a_{k-1}} = dq_dq_{k-1}, q{k-1} = Tk^{a_k} * p, dq_{k-1}_d\delta Tk.
 * last node contains T1^{a_1} * T2^{a_2} * ... * Tk^{a_k} = dq_dqk, qk = p, 0.
 *
 * Error in Ti, \delta Ti is defined by okvis::kinematics::minus and oplus.
 * Error in point is defined as p = \hat{p} + \delta p, \delta p is a 4D vector.
 */
class MultipleTransformPointJacobian {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  MultipleTransformPointJacobian() {}
  MultipleTransformPointJacobian(
      const Eigen::AlignedVector<okvis::kinematics::Transformation>& transformList,
      const std::vector<int>& exponentList, const Eigen::Vector4d& point)
      : transformList_(transformList),
        exponentList_(exponentList),
        point_(point),
        transformPointObject_(new TransformPointJacobian()),
        inverseTransformPointObject_(new InverseTransformPointJacobian()) {
//    computeJacobians();
  }

  MultipleTransformPointJacobian(
      const Eigen::AlignedVector<okvis::kinematics::Transformation>& transformList,
      const std::vector<int>& exponentList, const Eigen::Vector4d& point,
      std::shared_ptr<TransformPointJacobian> tpj,
      std::shared_ptr<InverseTransformPointJacobian> itpj)
      : transformList_(transformList),
        exponentList_(exponentList),
        point_(point),
        transformPointObject_(tpj),
        inverseTransformPointObject_(itpj) {
//    computeJacobians();
  }

  void initialize(
      const Eigen::AlignedVector<okvis::kinematics::Transformation>& transformList,
      const std::vector<int>& exponentList, const Eigen::Vector4d& point) {
    transformList_ = transformList;
    exponentList_ = exponentList;
    point_ = point;
    transformPointObject_.reset(new TransformPointJacobian());
    inverseTransformPointObject_.reset(new InverseTransformPointJacobian());
//    computeJacobians();
  }

  void initialize(
      const Eigen::AlignedVector<okvis::kinematics::Transformation>& transformList,
      const std::vector<int>& exponentList, const Eigen::Vector4d& point,
      std::shared_ptr<TransformPointJacobian> tpj,
      std::shared_ptr<InverseTransformPointJacobian> itpj) {
    initialize(transformList, exponentList, point);
    transformPointObject_ = tpj;
    inverseTransformPointObject_ = itpj;
//    computeJacobians();
  }

  Eigen::Vector4d evaluate() const;

  /**
   * @brief computeJacobians
   * The method to compute Jacobians is essentially the reverse mode autodiff algorithm.
   * In the forward pass(from input to output), intermediate values at
   * each node are computed, i.e., p, Tk^{a_k} * p, T(k-1)^{a_{k-1}} * p, ..., q.
   * In the reverse pass(from output to input), intermediate Jacobians
   * are computed as well as accumulated Jacobians by the chain rule.
   * Because there is only one output, intermediate Jacobians are computed in the forward pass.
   */
  void computeJacobians();

  /**
   * @brief dp_dT
   * @param transformIndex transform index in the transformList.
   * @return
   */
  Eigen::Matrix<double, 4, 6> dp_dT(size_t transformIndex) const {
    return transformJacobianList_[transformIndex].cumulativeLeftTransform_.T() *
           transformJacobianList_[transformIndex].dpoint_dHeadTransform_;
  }

  Eigen::Matrix<double, 4, 4> dp_dpoint() const {
    return transformJacobianList_.back().cumulativeLeftTransform_.T();
  }

 private:
  // input
  Eigen::AlignedVector<okvis::kinematics::Transformation>
      transformList_;              // T1, T2, ... Tk
  std::vector<int> exponentList_;  // a_1, a_2, ..., a_k
  Eigen::Vector4d point_;

  std::shared_ptr<TransformPointJacobian> transformPointObject_;
  std::shared_ptr<InverseTransformPointJacobian> inverseTransformPointObject_;

  // output
  // The cumulative transforms are: I, T1^{a_1}, T1^{a_1} * T2^{a_2}, ...,
  // T1^{a_1} * T2^{a_2} *... * Tk^{a_k}
  Eigen::AlignedVector<TransformPointJacobianNode> transformJacobianList_;
};
}  // namespace swift_vio
#endif // INCLUDE_SWIFT_VIO_MULTIPLE_TRANSFORM_POINT_JACOBIAN_HPP
