/*********************************************************************************
 *  OKVIS - Open Keyframe-based Visual-Inertial SLAM
 *  Copyright (c) 2015, Autonomous Systems Lab / ETH Zurich
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 * 
 *   * Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *   * Neither the name of Autonomous Systems Lab / ETH Zurich nor the names of
 *     its contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 *  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 *  Created on: Dec 30, 2014
 *      Author: Stefan Leutenegger (s.leutenegger@imperial.ac.uk)
 *    Modified: Andreas Forster (an.forster@gmail.com)
 *********************************************************************************/

/**
 * @file okvis/Estimator.hpp
 * @brief Header file for the Estimator class. This does all the backend work.
 * @author Stefan Leutenegger
 * @author Andreas Forster
 */

#ifndef INCLUDE_OKVIS_ESTIMATOR_HPP_
#define INCLUDE_OKVIS_ESTIMATOR_HPP_

#include <array>
#include <fstream>
#include <memory>
#include <mutex>

#include <ceres/ceres.h>
#include <okvis/kinematics/Transformation.hpp>

#include <okvis/assert_macros.hpp>
#include <loop_closure/KeyframeForLoopDetection.hpp>
#include <okvis/VioBackendInterface.hpp>
#include <okvis/MultiFrame.hpp>
#include <okvis/FrameTypedefs.hpp>
#include <okvis/Measurements.hpp>
#include <okvis/Variables.hpp>
#include <okvis/EstimatorBase.hpp>
#include <okvis/ceres/SpeedAndBiasParameterBlock.hpp>
#include <okvis/ceres/HomogeneousPointParameterBlock.hpp>
#include <okvis/ceres/Map.hpp>
#include <okvis/ceres/MarginalizationError.hpp>
#include <okvis/ceres/ReprojectionError.hpp>
#include <okvis/ceres/CeresIterationCallback.hpp>

#include <swift_vio/imu/BoundedImuDeque.hpp>
#include <swift_vio/CameraRig.hpp>
#include <swift_vio/imu/ImuRig.hpp>
#include <swift_vio/InitialNavState.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {

//! The estimator class
/*!
 The estimator class. This does all the backend work.
 Frames:
 W: World
 B: Body
 C: Camera
 S: Sensor (IMU)
 */
class Estimator : public EstimatorBase
{
 public:
  OKVIS_DEFINE_EXCEPTION(Exception, std::runtime_error)
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /**
   * @brief The default constructor.
   */
  Estimator();

  /**
   * @brief Constructor if a ceres map is already available.
   * @param mapPtr Shared pointer to ceres map.
   */
  Estimator(std::shared_ptr<okvis::ceres::Map> mapPtr);
  virtual ~Estimator();

  /**
   * @brief Add a pose to the state.
   * @param multiFrame Matched multiFrame.
   * @param imuMeasurements IMU measurements from last state to new one.
   * @param asKeyframe Is this new frame a keyframe?
   * @return True if successful.
   */
  bool addStates(okvis::MultiFramePtr multiFrame,
                 const okvis::ImuMeasurementDeque & imuMeasurements,
                 bool asKeyframe) override;

  /**
   * @brief Applies the dropping/marginalization strategy according to the RSS'13/IJRR'14 paper.
   *        The new number of frames in the window will be numKeyframes+numImuFrames.
   * @param removedLandmarks Get the landmarks that were removed by this operation.
   * @return True if successful.
   */
  bool applyMarginalizationStrategy(okvis::MapPointVector& removedLandmarks) override;

  /**
   * @brief Start ceres optimization.
   * @param[in] numIter Maximum number of iterations.
   * @param[in] numThreads Number of threads.
   * @param[in] verbose Print out optimization progress and result, if true.
   */
  void optimize(size_t numIter, size_t numThreads = 1, bool verbose = false) override;

  /**
   * @brief Set a time limit for the optimization process.
   * @param[in] timeLimit Time limit in seconds. If timeLimit < 0 the time limit is removed.
   * @param[in] minIterations minimum iterations the optimization process should do
   *            disregarding the time limit.
   * @return True if successful.
   */
  bool setOptimizationTimeLimit(double timeLimit, int minIterations) final;

  void setTimingLogfile(const std::string &logfile) final {
    timing_logfile_ = logfile;
    timing_log_ = std::ofstream(logfile, std::ios::out);
  }

  /**
   * @brief Prints state information to buffer.
   * @param poseId The pose Id for which to print.
   * @param buffer The puffer to print into.
   */
  void printStates(uint64_t poseId, std::ostream & buffer) const;

  /**
   * @brief computeCovariance compute covariance by okvis marginalization module
   * which handles rank deficiency caused by low-disparity landmarks.
   * @param cov covariance of p_WS, q_WS, v_WS, b_g, b_a.
   * @return true if covariance is computed successfully, false otherwise.
   */
  bool computeCovariance(Eigen::MatrixXd* cov) const override;

  /**
   * @brief computeCovarianceCeres compute covariance by ceres::Covariance which
   * can handle rank deficiency if DENSE_SVD is used.
   * @param[out] cov covariance of p_WS, q_WS, v_WS, b_g, b_a.
   * @param[in] covAlgorithm SPARSE_QR or DENSE_SVD. DENSE_SVD is slow but
   * handles rank deficiency.
   * @return true if covariance is computed successfully, false otherwise.
   */
  bool
  computeCovarianceCeres(Eigen::MatrixXd *cov,
                         ::ceres::CovarianceAlgorithmType covAlgorithm) const;


  std::vector<std::string> variableLabels() const override;

  /// @name Getters
  /// @{
  /**
   * @brief get std. dev. of state for nav state (p,q,v), imu(bg ba), and optionally
   * imu augmented intrinsic parameters, camera extrinsic, intrinsic, td, tr.
   * @param stateStd
   * @return true if std. dev. of states are computed successfully.
   */
  bool getStateStd(Eigen::Matrix<double, Eigen::Dynamic, 1>* stateStd) const override;
  ///@}

 private:
  /**
   * @brief addReprojectionFactors add reprojection factors for all observations
   * of landmarks whose residuals are NULL.
   *
   * OKVIS original frontend finds feature matches and immediately adds
   * reprojection factors to the ceres problem for all landmarks that can be
   * triangulated with a small chi2 cost, even when they are at infinity.
   * That is, every landmark in landmarksMap_ is accounted for in the optimizer.
   *
   * jhuai cuts the procedure into two steps, renewing the feature tracks with
   * feature matches done in the frontend, and adding reprojection factors to
   * the ceres problem done in Estimator::optimize(). This function carries out
   * the latter step.
   *
   * @attention This function considers implications from mergeTwoLandmarks
   * and replaceEpipolarWithReprojectionErrors, and addEpipolarConstraint.
   *
   * @warning But this function interferes with cases arising from
   * addObservation in using epipolar constraints, i.e., all
   * observations of a landmark are not asscoiated with any residual prior to
   * forming an epipolar factor for the landmark.
   *
   * @return
   */
  bool addReprojectionFactors();

  void updateSensorRigs();

 protected:

  void addPriorAndRelativeTerms(const okvis::ImuMeasurementDeque &imuMeasurements);

  template <class GEOMETRY_TYPE>
  ::ceres::ResidualBlockId addPointFrameResidual(uint64_t landmarkId,
                                                 const KeypointIdentifier& kpi);

  /**
   * @brief Remove an observation from a landmark.
   * @param residualBlockId Residual ID for this landmark.
   * @return True if successful.
   */
  bool removeObservationAndResidual(::ceres::ResidualBlockId residualBlockId);

  // loss function for reprojection errors
  std::shared_ptr< ::ceres::LossFunction> cauchyLossFunctionPtr_; ///< Cauchy loss.
  std::shared_ptr< ::ceres::LossFunction> huberLossFunctionPtr_; ///< Huber loss.

  // the marginalized error term
  std::shared_ptr<ceres::MarginalizationError> marginalizationErrorPtr_; ///< The marginalisation class
  ::ceres::ResidualBlockId marginalizationResidualId_; ///< Remembers the marginalisation object's Id

  // ceres iteration callback object
  std::unique_ptr<okvis::ceres::CeresIterationCallback> ceresCallback_; ///< Maybe there was a callback registered, store it here.

  std::string timing_logfile_;
  std::ofstream timing_log_;
};
}  // namespace okvis

#include "swift_vio/implementation/Estimator.hpp"

#endif /* INCLUDE_OKVIS_ESTIMATOR_HPP_ */
