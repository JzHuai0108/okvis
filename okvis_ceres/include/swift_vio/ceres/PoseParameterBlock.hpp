
/**
 * @file PoseParameterBlock.hpp
 * @brief Header file for the PoseParameterBlock class.
 */

#ifndef INCLUDE_SWIFT_VIO_POSEPARAMETERBLOCK_HPP_
#define INCLUDE_SWIFT_VIO_POSEPARAMETERBLOCK_HPP_

#include <Eigen/Core>
#include <okvis/ceres/ParameterBlockSized.hpp>
#include <okvis/kinematics/Transformation.hpp>
#include <okvis/Time.hpp>

namespace swift_vio {

/// \brief Wraps the parameter block for a pose estimate
/// Compared to okvis::ceres::PoseParameterBlock, this PoseParameterBlock
/// 1) includes the linearization point,
/// 2) uses the trick in VINS Mono to set liftJacobian identity,
/// 3) discards plus(), minus(), liftJacobian().
class PoseParameterBlock : public okvis::ceres::ParameterBlockSized<7,6,okvis::kinematics::Transformation>{
public:

  /// \brief The estimate type (okvis::kinematics::Transformation ).
  typedef okvis::kinematics::Transformation estimate_t;

  /// \brief The base class type.
  typedef ParameterBlockSized<7,6,estimate_t> base_t;

  /// \brief Default constructor (assumes not fixed).
  PoseParameterBlock();

  /// \brief Constructor with estimate and time.
  /// @param[in] T_WS The pose estimate as T_WS.
  /// @param[in] id The (unique) ID of this block.
  /// @param[in] timestamp The timestamp of this state.
  PoseParameterBlock(const okvis::kinematics::Transformation& T_WS, uint64_t id, const okvis::Time& timestamp);

  /// \brief Trivial destructor.
  virtual ~PoseParameterBlock();

  // setters
  /// @brief Set estimate of this parameter block.
  /// @param[in] T_WS The estimate to set this to.
  virtual void setEstimate(const okvis::kinematics::Transformation& T_WS);

  void fixPositionLinPoint(const okvis::kinematics::Transformation &T_WS);

  void fixPositionLinPoint();

  // getters
  /// @brief Get estimate.
  /// \return The estimate.
  virtual okvis::kinematics::Transformation estimate() const;

  okvis::kinematics::Transformation linPoint() const;

  Eigen::Vector3d positionLinPoint() const {
    return pLinPoint_;
  }

  Eigen::Quaterniond orientationLinPoint() const {
    return Eigen::Quaterniond(parameters_[6], parameters_[3], parameters_[4],
                              parameters_[5]);
  }

  /// @brief Return parameter block type as string
  virtual std::string typeInfo() const {return "PoseParameterBlock";}

private:
  Eigen::Vector3d pLinPoint_;
  bool plinPointFixed_;
};

} // namespace swift_vio

#endif /* INCLUDE_SWIFT_VIO_POSEPARAMETERBLOCK_HPP_ */
