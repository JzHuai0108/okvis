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
 *  Created on: Apr 22, 2012
 *      Author: Stefan Leutenegger (s.leutenegger@imperial.ac.uk)
 *    Modified: Andreas Forster (an.forster@gmail.com)
 *********************************************************************************/

/**
 * @file Parameters.hpp
 * @brief This file contains struct definitions that encapsulate parameters and settings.
 * @author Stefan Leutenegger
 * @author Andreas Forster
 */

#ifndef INCLUDE_OKVIS_PARAMETERS_HPP_
#define INCLUDE_OKVIS_PARAMETERS_HPP_

#include <deque>
#include <string>
#include <vector>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Woverloaded-virtual"
#include <opencv2/opencv.hpp>
#pragma GCC diagnostic pop
#include <Eigen/Dense>

#include <okvis/Time.hpp>
#include <okvis/MultiFrame.hpp>
#include <okvis/cameras/NCameraSystem.hpp>
#include <okvis/kinematics/Transformation.hpp>

#include <swift_vio/SwiftParameters.hpp>
#include <swift_vio/InitialNavState.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {

/// \brief Struct to define the behavior of the camera extrinsics.
struct CameraNoiseParameters {
  // set to 0 in order to turn off
  /// \brief Default Constructor -- fixed camera extrinsics.
  CameraNoiseParameters();

  /**
   * @brief Constructor.
   * @param sigma_absolute_translation Absolute translation stdev. [m]
   * @param sigma_absolute_orientation Absolute orientation stdev. [rad]
   * @param sigma_c_relative_translation Relative translation noise density.
   * [m/sqrt(Hz)]
   * @param sigma_c_relative_orientation Relative orientation noise density.
   * [rad/sqrt(Hz)]
   */
  CameraNoiseParameters(double sigma_absolute_translation,
                        double sigma_absolute_orientation,
                        double sigma_c_relative_translation,
                        double sigma_c_relative_orientation);

public:
  // absolute (prior) w.r.t frame S
  double sigma_absolute_translation; ///< Absolute translation stdev. [m]
  double sigma_absolute_orientation; ///< Absolute orientation stdev. [rad]

  // relative (temporal)
  double sigma_c_relative_translation; ///< Relative translation noise density.
                                       ///< [m/sqrt(Hz)]
  double sigma_c_relative_orientation; ///< Relative orientation noise density.
                                       ///< [rad/sqrt(Hz)]

  double sigma_focal_length; ///< stdev. of horizontal and vertical focal lengths.
  double sigma_principal_point; ///< stdev. of principal point coordinates.
  std::vector<double> sigma_distortion; ///< stdev. of camera distortion
  double sigma_td;                      ///< camera time delay stdev. [sec]
  double sigma_tr;                      ///< frame readout time stdev. [sec]
  double sigma_observation;             ///< camera observation noise std. dev. [px]
};

/*!
 * \brief IMU parameters.
 *
 * A simple struct to specify properties of an IMU.
 *
 */
struct ImuParameters{
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  okvis::kinematics::Transformation T_BS; ///< Transformation from Body frame to IMU (sensor frame S).
  double a_max;  ///< Accelerometer saturation. [m/s^2]
  double g_max;  ///< Gyroscope saturation. [rad/s]
  double sigma_g_c;  ///< Gyroscope noise density.
  double sigma_a_c;  ///< Accelerometer noise density.
  double sigma_bg;  ///< Initial gyroscope bias.
  double sigma_ba;  ///< Initial accelerometer bias.
  double sigma_gw_c; ///< Gyroscope drift noise density.
  double sigma_aw_c; ///< Accelerometer drift noise density.
  double tau;  ///< Reversion time constant of accerometer bias. [s]
  double g;  ///< Earth acceleration.
  Eigen::Vector3d g0;      ///< Mean of the prior gyroscope bias.
  Eigen::Vector3d a0;  ///< Mean of the prior accelerometer bias.
  int rate;  ///< IMU rate in Hz.

  double sigma_TGElement;  /// std for every element in shape matrix T_g
  double sigma_TSElement;
  double sigma_TAElement;
  Eigen::Matrix<double, 9, 1> Tg0;  // initial Tg estimate
  Eigen::Matrix<double, 9, 1> Ts0;
  Eigen::Matrix<double, 9, 1> Ta0;
  std::string model_type; // 0 bg_ba, 1 bg_ba_Tg_Ts_Ta, 2, scaledmisaligned
  bool estimateGravityDirection;
  double sigmaGravityDirection; // The uncertainty in both roll and pitch of the gravity direction.

  ImuParameters();

  const Eigen::Vector3d &gravityDirection() const;

  Eigen::Vector3d gravity() const;

  void setGravityDirection(const Eigen::Vector3d &gravityDirection);

private:
  Eigen::Vector3d normalGravity;
};

/*!
 * \brief Magnetometer parameters.
 *
 * A simple struct to specify properties of a magnetometer.
 *
 */
struct MagnetometerParameters{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  double stdev; ///< Measurement (white noise part) standard deviation. [uT]
  double priorStdev; ///< Prior. [uT]
  double tau;        ///< Reversion time constant of bias [s]
  double sigma_c;    ///< Bias noise density [uT/sqrt(Hz)]
  double updateFrequency; ///< Related state estimates are inserted at this frequency. [Hz]
};

/*!
 * \brief GPS parameters
 *
 * A simple struct to specify properties of a GPS receiver.
 *
 */
struct GpsParameters{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Eigen::Vector3d antennaOffset; ///< The position offset of the antenna in body (B) coordinates.
};

/*!
 * \brief Position sensor parameters.
 *
 * A simple struct to specify properties of a position sensor.
 *
 */
struct PositionSensorParameters{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Eigen::Vector3d positionSensorOffset; ///< The position offset of the position sensor in body (B) coordinates.
  bool isLeveled; ///< If true, the position sensor measurements are assumed to be world z up (exactly, i.e. only yaw gets estimated).
};

/*!
 * \brief Magnetic ENU z bias
 *
 * A simple struct to specify the dynamics of magnetic ENU z component variation.
 *
 */
struct MagneticEnuZParameters{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  double priorStdev;  ///< ENU z-component bias prior. [uT]
  double sigma_c;     ///< ENU z-component noise density. [uT/s/sqrt(Hz)]
  double tau;         ///< Reversion time constant of ENU z-component. [s]
  double updateFrequency; ///< Related state estimates are inserted at this frequency. [Hz]
};

/*!
 * \brief Barometer parameters.
 *
 * A simple struct to specify properties of a barometer.
 *
 */
struct BarometerParameters{
  double staticPressureStdev; ///< Measurement (white noise part) standard deviation. [kPa]
  double temperatureStdev; ///< Measurement (white noise part) standard deviation. [kPa]
};

/*!
 * \brief QFF parameters.
 *
 * A simple struct to specify the QFF state dynamics.
 *
 */
struct QffParameters{
  double priorStdev;  ///< Prior of QFF [kPa]
  double sigma_c;     ///< Drift noise density. [kPa/sqrt(s)]
  double tau;         ///< Reversion time constant. [s]
  double updateFrequency; ///< Related state estimates are inserted at this frequency. [Hz]
};

/*!
 * \brief Differential pressure sensor parameters.
 *
 * A simple struct to specify properties of a differential pressure sensor.
 *
 */
struct DifferentialPressureSensorParameters{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  double differentialPressureStdev; ///< Measurement (white noise part) standard deviation. [Pa]
  /// Measurement (white noise part) standard deviation [m/s^2]
  /// Be conservative, this has to account for model uncertainties, too...
  double accelerationStdev;
  /// Polynomial for -z direction (dimensionless) force, alpha deg.
  /// polynomial vector of lenght n as (Matlab polyval style):
  /// c[0]*x^(n-1)+c[1]*x^(n-2)+...+c[n-2]*x+c[n-1] .
  Eigen::VectorXd c_minus_z_of_alpha;
  double c_y_of_beta; ///< Polynomial (only proportionality) for y direction (dimensionless) force, beta deg.
  double m; ///< Mass. [kg]
  double A; ///< Reference wing area. [m^2]
};

/*!
 * \brief Wind parameters.
 *
 * A simple struct to specify the wind state dynamics.
 *
 */
struct WindParameters{
  double priorStdev;  ///< Prior of wind. [m/s]
  double sigma_c;     ///< Drift noise density. [m/s/sqrt(s)]
  double tau;         ///< Reversion time constant. [s];
  double updateFrequency; ///< Related state estimates are inserted at this frequency. [Hz]
};

/**
 * @brief Parameters for optimization and related things (detection).
 */
struct Optimization{
  int max_iterations; ///< Maximum iterations the optimization should perform.
  int min_iterations; ///< Minimum iterations the optimization should perform.
  double timeLimitForMatchingAndOptimization; ///< The time limit for both matching and optimization. [s]
  okvis::Duration timeReserve; ///< Store a little more on the beginning and end of the IMU buffer. [s]

  int numKeyframes; ///< Number of keyframes.
  int numImuFrames; ///< Number of IMU frames.

  swift_vio::EstimatorAlgorithm algorithm;

  bool useEpipolarConstraint;
  int cameraObservationModelId;

  // set true to analyze the consistency of Estimator,
  // set false to analyze the accuracy of Estimator.
  bool computeOkvisNees;

  // Malalanobis gating test or a simple reprojection error threshold to remove outliers in filtering.
  bool useMahalanobisGating;
  // Max allowed projection error for an image feature to check outliers in the reprojection error scheme.
  double maxProjectionErrorTol;

  // The number of frames to delay the nonlinear filter initialization since motion is detected?
  int delayFilterInitByFrames;

  Optimization(int _max_iterations = 10, int _min_iterations = 3,
               double _timeLimitForMatchingAndOptimization = 0.035,
               okvis::Duration _timeReserve = okvis::Duration(0.005),
               int _numKeyframes = 5, int _numImuFrames = 3,
               swift_vio::EstimatorAlgorithm _algorithm =
                   swift_vio::EstimatorAlgorithm::OKVIS,
               bool _useEpipolarConstraint = false,
               int _cameraObservationModelId = 0, bool _computeOkvisNees = false,
               bool _useMahalanobisGating = true,
               double _maxProjectionErrorTol = 7, int _delayFilterInitByFrames = 3);

  std::string toString(std::string lead) const;
};

/**
 * @brief Information on camera and IMU setup.
 */
struct SensorsInformation {
  int cameraRate;     ///< Camera rate in Hz.
  double imageDelay;  ///< Camera image delay. [s]
  int imuIdx;         ///< IMU index. Anything other than 0 will probably not work.
  double frameTimestampTolerance; ///< Time tolerance between frames to accept them as stereo frames. [s]
};

/// @brief Some visualization settings.
struct Visualization {
  bool displayImages; ///< Display images?
};

enum class FrameName { B, S, W, Wc };

/// @brief Some publishing parameters.
struct PublishingParameters {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  int publishRate = 200;  ///< Maximum publishing rate. [Hz]
  bool publishLandmarks = true; ///< Select, if you want to publish landmarks at all.
  float landmarkQualityThreshold = 1.0e-5; ///< Quality threshold under which landmarks are not published. Between 0 and 1.
  float maxLandmarkQuality = 0.05; ///< Quality above which landmarks are assumed to be of the best quality. Between 0 and 1.
  size_t maxPathLength = 100 ; ///< Maximum length of ros::nav_mgsgs::Path to be published.
  bool publishImuPropagatedState = true; ///< Should the state that is propagated with IMU messages be published? Or just the optimized ones?
  okvis::kinematics::Transformation T_Wc_W = okvis::kinematics::Transformation::Identity(); ///< Provide custom World frame Wc
  FrameName trackedBodyFrame = FrameName::B; ///< B or S, the frame of reference that will be expressed relative to the selected worldFrame Wc
  FrameName velocitiesFrame = FrameName::B; ///< B or S,  the frames in which the velocities of the selected trackedBodyFrame will be expressed in
};

/// @brief Struct to combine all parameters and settings.
struct VioParameters {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Optimization optimization;    ///< Optimization parameters.
  Visualization visualization;  ///< Visualization parameters.
  SensorsInformation sensors_information; ///< Information on camera and IMU setup.
  CameraNoiseParameters camera_extrinsics; ///< Camera extrinsic estimation parameters.
  okvis::cameras::NCameraSystem nCameraSystem;  ///< Camera configuration.
  ImuParameters imu;  ///< IMU parameters
  MagnetometerParameters magnetometer;  ///< Magnetometer parameters.
  PositionSensorParameters position;  ///< Position sensor parameters.
  GpsParameters gps; ///< GPS parameters
  MagneticEnuZParameters magnetic_enu_z;  ///< Dynamics of magnetic ENU z component variation.
  BarometerParameters barometer;  ///< Barometer parameters.
  QffParameters qff;  ///< QFF parameters.
  DifferentialPressureSensorParameters differential; ///< Differential pressure sensor parameters.
  WindParameters wind;  ///< Wind parameters.
  PublishingParameters publishing; ///< Publishing parameters.
  swift_vio::InputData input;
  swift_vio::InitialNavState initialState;
  swift_vio::FrontendOptions frontendOptions;
  swift_vio::PointLandmarkOptions pointLandmarkOptions;
  swift_vio::PoseGraphOptions poseGraphOptions;
};

} // namespace okvis

#endif // INCLUDE_OKVIS_PARAMETERS_HPP_
