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
 *  Created on: Jun 17, 2013
 *      Author: Stefan Leutenegger (s.leutenegger@imperial.ac.uk)
 *    Modified: Andreas Forster (an.forster@gmail.com)
 *********************************************************************************/

/**
 * @file VioParametersReader.cpp
 * @brief Source file for the VioParametersReader class.
 * @author Stefan Leutenegger
 * @author Andreas Forster
 */

#include <algorithm>

#include <glog/logging.h>

#include <okvis/cameras/NCameraSystem.hpp>
#include <okvis/cameras/PinholeCamera.hpp>
#include <okvis/cameras/EquidistantDistortion.hpp>
#include <okvis/cameras/RadialTangentialDistortion.hpp>
#include <okvis/cameras/RadialTangentialDistortion8.hpp>
#include <okvis/cameras/FovDistortion.hpp>
#include <okvis/cameras/EUCM.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>

#include <okvis/VioParametersReader.hpp>

#ifdef HAVE_LIBVISENSOR
  #include <visensor/visensor_api.hpp>
#endif

/// \brief okvis Main namespace of this package.
namespace okvis {

// The default constructor.
VioParametersReader::VioParametersReader()
    : useDriver(false),
      readConfigFile_(false) {
  vioParameters_.publishing.publishRate = 0;
}

// The constructor. This calls readConfigFile().
VioParametersReader::VioParametersReader(const std::string& filename) {
  // reads
  readConfigFile(filename);
}

void parseCameraNoises(cv::FileNode cameraParamNode,
                       CameraNoiseParameters *camera_noise) {
  if (cameraParamNode["sigma_absolute_translation"].isReal()) {
    cameraParamNode["sigma_absolute_translation"]
        >> camera_noise->sigma_absolute_translation;
  }
  if (cameraParamNode["sigma_absolute_orientation"].isReal()) {
    cameraParamNode["sigma_absolute_orientation"]
        >> camera_noise->sigma_absolute_orientation;
  }
  if (cameraParamNode["sigma_c_relative_translation"].isReal()) {
    cameraParamNode["sigma_c_relative_translation"]
        >> camera_noise->sigma_c_relative_translation;
  }
  if (cameraParamNode["sigma_c_relative_orientation"].isReal()) {
    cameraParamNode["sigma_c_relative_orientation"]
        >> camera_noise->sigma_c_relative_orientation;
  }
  if (cameraParamNode["sigma_focal_length"].isReal()) {
    cameraParamNode["sigma_focal_length"] >>
        camera_noise->sigma_focal_length;
  }
  if (cameraParamNode["sigma_principal_point"].isReal()) {
    cameraParamNode["sigma_principal_point"] >>
        camera_noise->sigma_principal_point;
  }
  cv::FileNode distortionNode = cameraParamNode["sigma_distortion"];
  camera_noise->sigma_distortion.clear();
  camera_noise->sigma_distortion.reserve(5);
  if (distortionNode.isSeq()) {
    for (size_t jack = 0; jack < distortionNode.size(); ++jack)
      camera_noise->sigma_distortion.push_back(
          static_cast<double>(distortionNode[jack]));
  }
  if (cameraParamNode["sigma_td"].isReal()) {
    cameraParamNode["sigma_td"] >> camera_noise->sigma_td;
  }
  if (cameraParamNode["sigma_tr"].isReal()) {
    cameraParamNode["sigma_tr"] >> camera_noise->sigma_tr;
  }
  camera_noise->updateParameterStatus();
}

void parsePublishOptions(cv::FileNode publishOptionNode,
                         PublishingParameters *publishing) {
  if (publishOptionNode["publish_rate"].isInt()) {
    publishOptionNode["publish_rate"] >> publishing->publishRate;
  }

  if (publishOptionNode["landmarkQualityThreshold"].isReal()) {
    publishOptionNode["landmarkQualityThreshold"] >>
        publishing->landmarkQualityThreshold;
  }

  if (publishOptionNode["maximumLandmarkQuality"].isReal()) {
    publishOptionNode["maximumLandmarkQuality"] >>
        publishing->maxLandmarkQuality;
  }

  if (publishOptionNode["maxPathLength"].isInt()) {
    publishing->maxPathLength = (int)(publishOptionNode["maxPathLength"]);
  }

  parseBoolean(publishOptionNode["publishImuPropagatedState"],
               publishing->publishImuPropagatedState);

  parseBoolean(publishOptionNode["publishLandmarks"],
               publishing->publishLandmarks);

  cv::FileNode T_Wc_W_ = publishOptionNode["T_Wc_W"];
  if (T_Wc_W_.isSeq()) {
    Eigen::Matrix4d T_Wc_W_e;
    T_Wc_W_e << T_Wc_W_[0], T_Wc_W_[1], T_Wc_W_[2], T_Wc_W_[3], T_Wc_W_[4],
        T_Wc_W_[5], T_Wc_W_[6], T_Wc_W_[7], T_Wc_W_[8], T_Wc_W_[9], T_Wc_W_[10],
        T_Wc_W_[11], T_Wc_W_[12], T_Wc_W_[13], T_Wc_W_[14], T_Wc_W_[15];

    publishing->T_Wc_W = okvis::kinematics::Transformation(T_Wc_W_e);
    std::stringstream s;
    s << publishing->T_Wc_W.T();
    VLOG(2) << "Custom World frame provided T_Wc_W=\n" << s.str();
  }

  if (publishOptionNode["trackedBodyFrame"].isString()) {
    std::string frame = (std::string)publishOptionNode["trackedBodyFrame"];
    // cut out first word. str currently contains everything including comments
    frame = frame.substr(0, frame.find(" "));
    if (frame.compare("B") == 0)
      publishing->trackedBodyFrame = FrameName::B;
    else if (frame.compare("S") == 0)
      publishing->trackedBodyFrame = FrameName::S;
    else {
      LOG(WARNING)
          << frame
          << " unknown/invalid frame for trackedBodyFrame, setting to B";
      publishing->trackedBodyFrame = FrameName::B;
    }
  }

  if (publishOptionNode["velocitiesFrame"].isString()) {
    std::string frame = (std::string)publishOptionNode["velocitiesFrame"];
    // cut out first word. str currently contains everything including comments
    frame = frame.substr(0, frame.find(" "));
    if (frame.compare("B") == 0)
      publishing->velocitiesFrame = FrameName::B;
    else if (frame.compare("S") == 0)
      publishing->velocitiesFrame = FrameName::S;
    else if (frame.compare("Wc") == 0)
      publishing->velocitiesFrame = FrameName::Wc;
    else {
      LOG(WARNING)
          << frame
          << " unknown/invalid frame for velocitiesFrame, setting to Wc";
      publishing->velocitiesFrame = FrameName::Wc;
    }
  }
}

void parseInitialState(cv::FileNode initialStateNode,
                       swift_vio::InitialNavState* initialState) {
  bool initializeToCustomPose = true;
  cv::FileNode timeNode = initialStateNode["state_time"];
  if (timeNode.isReal()) {
    double time;
    timeNode >> time;
    initialState->stateTime = okvis::Time(time);
  } else {
    initializeToCustomPose = false;
  }

  cv::FileNode vsNode = initialStateNode["v_WS"];
  if (vsNode.isSeq()) {
    Eigen::Vector3d vs;
    vs << vsNode[0], vsNode[1], vsNode[2];
    initialState->v_WS = vs;
  } else {
    initializeToCustomPose = false;
  }

  cv::FileNode stdvsNode = initialStateNode["sigma_v_WS"];
  if (stdvsNode.isSeq()) {
    Eigen::Vector3d stdvs;
    stdvs << stdvsNode[0], stdvsNode[1], stdvsNode[2];
    initialState->sigma_v_WS = stdvs;
  }

  cv::FileNode stdpsNode = initialStateNode["sigma_p_WS"];
  if (stdpsNode.isSeq()) {
    Eigen::Vector3d stdps;
    stdps << stdpsNode[0], stdpsNode[1], stdpsNode[2];
    initialState->sigma_p_WS = stdps;
  }

  cv::FileNode qsNode = initialStateNode["q_WS"];
  if (qsNode.isSeq()) {
    Eigen::Vector4d qs;
    qs << qsNode[0], qsNode[1], qsNode[2], qsNode[3];
    initialState->q_WS = Eigen::Quaterniond(qs[3], qs[0], qs[1], qs[2]);
  }

  cv::FileNode stdqsNode = initialStateNode["sigma_q_WS"];
  if (stdqsNode.isSeq()) {
    Eigen::Vector3d stdqs;
    stdqs << stdqsNode[0], stdqsNode[1], stdqsNode[2];
    initialState->sigma_q_WS = stdqs;
  }

  initialState->initializeToCustomPose = initializeToCustomPose;
  VLOG(2) << initialState->toString();
}

void parseEstimatorOptions(cv::FileNode optNode, EstimatorOptions *optParams) {
  if (optNode["algorithm"].isString()) {
    std::string description = (std::string)optNode["algorithm"];
    optParams->algorithm = swift_vio::EstimatorAlgorithmNameToId(description);
  } else {
    optParams->algorithm = swift_vio::EstimatorAlgorithm::SlidingWindowSmoother;
  }
  parseBoolean(optNode["constantBias"], optParams->constantBias);

  parseBoolean(optNode["useEpipolarConstraint"], optParams->useEpipolarConstraint);

  if (optNode["cameraObservationModelId"].isInt()) {
    optParams->cameraObservationModelId =
        static_cast<int>(optNode["cameraObservationModelId"]);
  }
  parseBoolean(optNode["computeOkvisNees"], optParams->computeOkvisNees);
  parseBoolean(optNode["useMahalanobisGating"],
               optParams->useMahalanobisGating);
  if (optNode["maxProjectionErrorTol"].isInt()) {
    optNode["maxProjectionErrorTol"] >> optParams->maxProjectionErrorTol;
  }
  if (optNode["delayFilterInitByFrames"].isInt()) {
    optNode["delayFilterInitByFrames"] >> optParams->delayFilterInitByFrames;
  }
  LOG(INFO) << optParams->toString("Estimator options: ");
}

void parseFrontendOptions(cv::FileNode frontendNode,
                          swift_vio::FrontendOptions* frontendOptions) {
  parseBoolean(frontendNode["useMedianFilter"],
               frontendOptions->useMedianFilter);
  if (frontendNode["keyframeInsertionOverlapThreshold"].isReal()) {
    frontendNode["keyframeInsertionOverlapThreshold"] >>
        frontendOptions->keyframeInsertionOverlapThreshold;
  }
  if (frontendNode["keyframeInsertionMatchingRatioThreshold"].isReal()) {
    frontendNode["keyframeInsertionMatchingRatioThreshold"] >>
        frontendOptions->keyframeInsertionMatchingRatioThreshold;
  }
  parseBoolean(frontendNode["stereoMatchWithEpipolarCheck"],
               frontendOptions->stereoMatchWithEpipolarCheck);
  LOG(INFO) << "Stereo match with epipolar check? "
            << frontendOptions->stereoMatchWithEpipolarCheck;
  if (frontendNode["epipolarDistanceThreshold"].isReal()) {
    frontendNode["epipolarDistanceThreshold"] >>
        frontendOptions->epipolarDistanceThreshold;
  }
  LOG(INFO) << "Epipolar distance threshold in stereo matching: "
            << frontendOptions->epipolarDistanceThreshold;
  if (frontendNode["featureTrackingMethod"].isInt()) {
    int trackingMethod;
    frontendNode["featureTrackingMethod"] >> trackingMethod;
    frontendOptions->featureTrackingMethod =
        static_cast<swift_vio::FeatureTrackingScheme>(trackingMethod);
  }
  if (frontendNode["numThreads"].isInt()) {
    frontendNode["numThreads"] >> frontendOptions->numThreads;
  }
  LOG(INFO) << "Feature tracking method in frontend: "
            << frontendOptions->featureTrackingMethod;
}

void parseDetectionOptions(cv::FileNode detectionNode,
                           swift_vio::FrontendOptions* frontendOptions) {
  // detection threshold
  bool success = detectionNode["threshold"].isReal();
  OKVIS_ASSERT_TRUE(
      std::runtime_error, success,
      "'detection threshold' parameter missing in configuration file.");
  detectionNode["threshold"] >> frontendOptions->detectionThreshold;

  // detection octaves
  success = detectionNode["octaves"].isInt();
  OKVIS_ASSERT_TRUE(
      std::runtime_error, success,
      "'detection octaves' parameter missing in configuration file.");
  detectionNode["octaves"] >> frontendOptions->detectionOctaves;
  OKVIS_ASSERT_TRUE(std::runtime_error,
                    frontendOptions->detectionOctaves >= 0,
                    "Invalid parameter value.");

  // maximum detections
  success = detectionNode["maxNoKeypoints"].isInt();
  OKVIS_ASSERT_TRUE(
      std::runtime_error, success,
      "'detection maxNoKeypoints' parameter missing in configuration file.");
  detectionNode["maxNoKeypoints"] >> frontendOptions->maxNoKeypoints;
  OKVIS_ASSERT_TRUE(std::runtime_error,
                    frontendOptions->maxNoKeypoints >= 0,
                    "Invalid parameter value.");
}

void parsePointLandmarkOptions(cv::FileNode plNode,
                               swift_vio::PointLandmarkOptions* plOptions) {
  if (plNode["landmarkModelId"].isInt()) {
    plOptions->landmarkModelId = static_cast<int>(plNode["landmarkModelId"]);
  }

  if (plNode["minTrackLength"].isInt()) {
    plOptions->minTrackLengthForMsckf = static_cast<size_t>(
        std::max(static_cast<int>(plNode["minTrackLength"]), 3));
  }

  if (plNode["maxHibernationFrames"].isInt()) {
    plOptions->maxHibernationFrames = static_cast<size_t>(
        std::max(static_cast<int>(plNode["maxHibernationFrames"]), 1));
  }

  if (plNode["minTrackLengthForSlam"].isInt()) {
    plOptions->minTrackLengthForSlam = static_cast<size_t>(
        std::max(static_cast<int>(plNode["minTrackLengthForSlam"]), 3));
  }

  if (plNode["maxInStateLandmarks"].isInt()) {
    plOptions->maxInStateLandmarks =
        std::max(static_cast<int>(plNode["maxInStateLandmarks"]), 0);
  }

  if (plNode["maxMarginalizedLandmarks"].isInt()) {
    plOptions->maxMarginalizedLandmarks = static_cast<int>(plNode["maxMarginalizedLandmarks"]);
  }

  if (plNode["triangulationMaxDepth"].isReal()) {
    plNode["triangulationMaxDepth"] >>
        plOptions->triangulationMaxDepth;
  }
  LOG(INFO) << plOptions->toString("Point landmark options: ");
}

void parsePoseGraphOptions(cv::FileNode pgNode, swift_vio::PoseGraphOptions* pgOptions) {
  if (pgNode["maxOdometryConstraintForAKeyframe"].isInt()) {
    pgNode["maxOdometryConstraintForAKeyframe"] >>
        pgOptions->maxOdometryConstraintForAKeyframe;
  }
  if (pgNode["minDistance"].isReal()) {
    pgNode["minDistance"] >> pgOptions->minDistance;
  }
  if (pgNode["minAngle"].isReal()) {
    pgNode["minAngle"] >> pgOptions->minAngle;
  }
  LOG(INFO) << "Max #odometry constraint for a keyframe "
            << pgOptions->maxOdometryConstraintForAKeyframe << ", Min distance "
            << pgOptions->minDistance << ", Min angle " << pgOptions->minAngle;
}

// Read and parse a config file.
void VioParametersReader::readConfigFile(const std::string& filename) {
  vioParameters_.optimization.timeReserve.fromSec(0.005);

  // reads
  cv::FileStorage file(filename, cv::FileStorage::READ);

  OKVIS_ASSERT_TRUE(Exception, file.isOpened(),
                    "Could not open config file: " << filename);
  LOG(INFO) << "Opened configuration file: " << filename;

  // number of keyframes
  if (file["numKeyframes"].isInt()) {
    file["numKeyframes"] >> vioParameters_.optimization.numKeyframes;
  } else {
    LOG(WARNING)
        << "numKeyframes parameter not provided. Setting to default numKeyframes=5.";
    vioParameters_.optimization.numKeyframes = 5;
  }
  // number of IMU frames
  if (file["numImuFrames"].isInt()) {
    file["numImuFrames"] >> vioParameters_.optimization.numImuFrames;
  } else {
    LOG(WARNING)
        << "numImuFrames parameter not provided. Setting to default numImuFrames=2.";
    vioParameters_.optimization.numImuFrames = 2;
  }

  parseEstimatorOptions(
      file["optimization"], &vioParameters_.optimization);

  parseFrontendOptions(file["frontend"], &vioParameters_.frontendOptions);

  parseDetectionOptions(file["detection_options"], &vioParameters_.frontendOptions);

  parsePointLandmarkOptions(file["point_landmark"], &vioParameters_.pointLandmarkOptions);

  parsePoseGraphOptions(file["pose_graph"], &vioParameters_.poseGraphOptions);

  // minimum ceres iterations
  if (file["ceres_options"]["minIterations"].isInt()) {
    file["ceres_options"]["minIterations"]
        >> vioParameters_.optimization.min_iterations;
  } else {
    LOG(WARNING)
        << "ceres_options: minIterations parameter not provided. Setting to default minIterations=1";
    vioParameters_.optimization.min_iterations = 1;
  }
  // maximum ceres iterations
  if (file["ceres_options"]["maxIterations"].isInt()) {
    file["ceres_options"]["maxIterations"]
        >> vioParameters_.optimization.max_iterations;
  } else {
    LOG(WARNING)
        << "ceres_options: maxIterations parameter not provided. Setting to default maxIterations=10.";
    vioParameters_.optimization.max_iterations = 10;
  }
  // ceres time limit
  if (file["ceres_options"]["timeLimit"].isReal()) {
    file["ceres_options"]["timeLimit"] >> vioParameters_.optimization.timeLimitForMatchingAndOptimization;
  } else {
    LOG(WARNING)
        << "ceres_options: timeLimit parameter not provided. Setting no time limit.";
    vioParameters_.optimization.timeLimitForMatchingAndOptimization = -1.0;
  }

  // do we use the direct driver?
  bool success = parseBoolean(file["useDriver"], useDriver);
  OKVIS_ASSERT_TRUE(Exception, success,
                    "'useDriver' parameter missing in configuration file.");

  // display images?
  success = parseBoolean(file["displayImages"],
                         vioParameters_.visualization.displayImages);
  OKVIS_ASSERT_TRUE(Exception, success,
                    "'displayImages' parameter missing in configuration file.");

  // image delay
//  success = file["imageDelay"].isReal();
//  OKVIS_ASSERT_TRUE(Exception, success,
//                    "'imageDelay' parameter missing in configuration file.");
//  file["imageDelay"] >> vioParameters_.sensors_information.imageDelay;
//  VLOG(2) << "imageDelay = " << std::setprecision(15)
//            << vioParameters_.sensors_information.imageDelay;

  // camera rate
  success = file["camera_params"]["camera_rate"].isInt();
  OKVIS_ASSERT_TRUE(
      Exception, success,
      "'camera_params: camera_rate' parameter missing in configuration file.");
  file["camera_params"]["camera_rate"]
      >> vioParameters_.sensors_information.cameraRate;

  // timestamp tolerance
  if (file["camera_params"]["timestamp_tolerance"].isReal()) {
    file["camera_params"]["timestamp_tolerance"]
        >> vioParameters_.sensors_information.frameTimestampTolerance;
    OKVIS_ASSERT_TRUE(
        Exception,
        vioParameters_.sensors_information.frameTimestampTolerance
            < 0.5 / vioParameters_.sensors_information.cameraRate,
        "Timestamp tolerance for stereo frames is larger than half the time between frames.");
    OKVIS_ASSERT_TRUE(
        Exception,
        vioParameters_.sensors_information.frameTimestampTolerance >= 0.0,
        "Timestamp tolerance is smaller than 0");
  } else {
    vioParameters_.sensors_information.frameTimestampTolerance = 0.2
        / vioParameters_.sensors_information.cameraRate;
    LOG(WARNING)
        << "No timestamp tolerance for stereo frames specified. Setting to "
        << vioParameters_.sensors_information.frameTimestampTolerance;
  }

  parseCameraNoises(file["camera_params"], &vioParameters_.camera_noise);

  parsePublishOptions(file["publishing_options"], &vioParameters_.publishing);

  parseInitialState(file["initial_state"], &vioParameters_.initialState);

  std::vector<CameraCalibration,Eigen::aligned_allocator<CameraCalibration>> calibrations;
  if(!getCameraCalibration(calibrations, file))
    LOG(FATAL) << "Did not find any calibration!";

  buildCameraSystem(calibrations, &vioParameters_.nCameraSystem);

  cv::FileNode imu_params = file["imu_params"];
  parseImuParameters(imu_params, &vioParameters_.imu);
  readConfigFile_ = true;
}

void VioParametersReader::buildCameraSystem(
    const std::vector<CameraCalibration,
                      Eigen::aligned_allocator<CameraCalibration>>
        &calibrations,
    okvis::cameras::NCameraSystem *nCameraSystem) {
  bool computeOverlaps = false;
  size_t camIdx = 0;
  for (size_t i = 0; i < calibrations.size(); ++i) {
    std::shared_ptr<okvis::kinematics::Transformation> T_SC_okvis_ptr(
        new okvis::kinematics::Transformation(
            calibrations[i].T_SC.r(), calibrations[i].T_SC.q().normalized()));
    std::string distortionType = calibrations[i].distortionType;
    std::transform(distortionType.begin(), distortionType.end(),
                   distortionType.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    if (strcmp(distortionType.c_str(), "equidistant") == 0) {
      nCameraSystem->addCamera(
          T_SC_okvis_ptr,
          std::shared_ptr<okvis::cameras::CameraBase>(
              new okvis::cameras::PinholeCamera<
                  okvis::cameras::EquidistantDistortion>(
                  calibrations[i].imageDimension[0],
                  calibrations[i].imageDimension[1],
                  calibrations[i].focalLength[0],
                  calibrations[i].focalLength[1],
                  calibrations[i].principalPoint[0],
                  calibrations[i].principalPoint[1],
                  okvis::cameras::EquidistantDistortion(
                      calibrations[i].distortionCoefficients[0],
                      calibrations[i].distortionCoefficients[1],
                      calibrations[i].distortionCoefficients[2],
                      calibrations[i].distortionCoefficients[3]),
                  calibrations[i].imageDelaySecs,
                  calibrations[i].readoutTimeSecs
                  /*, id ?*/)),
          okvis::cameras::DistortionType::Equidistant,
          calibrations[i].projectionIntrinsicRepName,
          calibrations[i].extrinsicRepName, computeOverlaps);
      std::stringstream s;
      s << calibrations[i].T_SC.T();
      LOG(INFO) << "Equidistant pinhole camera " << camIdx << " with T_SC=\n"
                << s.str();
    } else if (strcmp(distortionType.c_str(), "radialtangential") == 0 ||
               strcmp(distortionType.c_str(), "plumb_bob") == 0) {
      nCameraSystem->addCamera(
          T_SC_okvis_ptr,
          std::shared_ptr<okvis::cameras::CameraBase>(
              new okvis::cameras::PinholeCamera<
                  okvis::cameras::RadialTangentialDistortion>(
                  calibrations[i].imageDimension[0],
                  calibrations[i].imageDimension[1],
                  calibrations[i].focalLength[0],
                  calibrations[i].focalLength[1],
                  calibrations[i].principalPoint[0],
                  calibrations[i].principalPoint[1],
                  okvis::cameras::RadialTangentialDistortion(
                      calibrations[i].distortionCoefficients[0],
                      calibrations[i].distortionCoefficients[1],
                      calibrations[i].distortionCoefficients[2],
                      calibrations[i].distortionCoefficients[3]),
                  calibrations[i].imageDelaySecs,
                  calibrations[i].readoutTimeSecs
                  /*, id ?*/)),
          okvis::cameras::DistortionType::RadialTangential,
          calibrations[i].projectionIntrinsicRepName,
          calibrations[i].extrinsicRepName, computeOverlaps);
      std::stringstream s;
      s << calibrations[i].T_SC.T();
      LOG(INFO) << "Radial tangential pinhole camera " << camIdx
                << " with T_SC=\n"
                << s.str();
    } else if (strcmp(distortionType.c_str(), "radialtangential8") == 0 ||
               strcmp(distortionType.c_str(), "plumb_bob8") == 0) {
      nCameraSystem->addCamera(
          T_SC_okvis_ptr,
          std::shared_ptr<okvis::cameras::CameraBase>(
              new okvis::cameras::PinholeCamera<
                  okvis::cameras::RadialTangentialDistortion8>(
                  calibrations[i].imageDimension[0],
                  calibrations[i].imageDimension[1],
                  calibrations[i].focalLength[0],
                  calibrations[i].focalLength[1],
                  calibrations[i].principalPoint[0],
                  calibrations[i].principalPoint[1],
                  okvis::cameras::RadialTangentialDistortion8(
                      calibrations[i].distortionCoefficients[0],
                      calibrations[i].distortionCoefficients[1],
                      calibrations[i].distortionCoefficients[2],
                      calibrations[i].distortionCoefficients[3],
                      calibrations[i].distortionCoefficients[4],
                      calibrations[i].distortionCoefficients[5],
                      calibrations[i].distortionCoefficients[6],
                      calibrations[i].distortionCoefficients[7]),
                  calibrations[i].imageDelaySecs,
                  calibrations[i].readoutTimeSecs
                  /*, id ?*/)),
          okvis::cameras::DistortionType::RadialTangential8,
          calibrations[i].projectionIntrinsicRepName,
          calibrations[i].extrinsicRepName, computeOverlaps);
      std::stringstream s;
      s << calibrations[i].T_SC.T();
      LOG(INFO) << "Radial tangential 8 pinhole camera " << camIdx
                << " with T_SC=\n"
                << s.str();
    } else if (strcmp(distortionType.c_str(), "fov") == 0) {
      std::shared_ptr<okvis::cameras::CameraBase> camPtr(
          new okvis::cameras::PinholeCamera<okvis::cameras::FovDistortion>(
              calibrations[i].imageDimension[0],
              calibrations[i].imageDimension[1], calibrations[i].focalLength[0],
              calibrations[i].focalLength[1], calibrations[i].principalPoint[0],
              calibrations[i].principalPoint[1],
              okvis::cameras::FovDistortion(
                  calibrations[i].distortionCoefficients[0]),
              calibrations[i].imageDelaySecs, calibrations[i].readoutTimeSecs
              /*, id ?*/));
      Eigen::VectorXd intrin(5);
      intrin[0] = calibrations[i].focalLength[0];
      intrin[1] = calibrations[i].focalLength[1];
      intrin[2] = calibrations[i].principalPoint[0];
      intrin[3] = calibrations[i].principalPoint[1];
      intrin[4] = calibrations[i].distortionCoefficients[0];
      camPtr->setIntrinsics(intrin);
      nCameraSystem->addCamera(
          T_SC_okvis_ptr, camPtr, okvis::cameras::DistortionType::Fov,
          calibrations[i].projectionIntrinsicRepName,
          calibrations[i].extrinsicRepName, computeOverlaps);
      std::stringstream s;
      s << calibrations[i].T_SC.T();
      LOG(INFO) << "FOV pinhole camera " << camIdx << " with Omega "
                << calibrations[i].distortionCoefficients[0] << " with T_SC=\n"
                << s.str();
    } else if (strcmp(distortionType.c_str(), "EUCM") == 0 ||
               strcmp(distortionType.c_str(), "eucm") == 0) {
      std::shared_ptr<okvis::cameras::CameraBase> camPtr(
          new okvis::cameras::EUCM(
              calibrations[i].imageDimension[0],
              calibrations[i].imageDimension[1], calibrations[i].focalLength[0],
              calibrations[i].focalLength[1], calibrations[i].principalPoint[0],
              calibrations[i].principalPoint[1],
              calibrations[i].distortionCoefficients[0],
              calibrations[i].distortionCoefficients[1],
              calibrations[i].imageDelaySecs, calibrations[i].readoutTimeSecs
              /*, id ?*/));
      nCameraSystem->addCamera(
          T_SC_okvis_ptr, camPtr, okvis::cameras::DistortionType::Eucm,
          calibrations[i].projectionIntrinsicRepName,
          calibrations[i].extrinsicRepName, computeOverlaps);
      std::stringstream s;
      s << calibrations[i].T_SC.T();
      LOG(INFO) << "Extended Unified camera model " << camIdx << " with alpha "
                << calibrations[i].distortionCoefficients[0] << " beta "
                << calibrations[i].distortionCoefficients[1] << " with T_SC=\n"
                << s.str();
    } else {
      LOG(ERROR) << "unrecognized distortion type "
                 << calibrations[i].distortionType;
    }
    ++camIdx;
  }

  size_t numCameras = calibrations.size();
  bool isOverlapGiven = true;
  for (size_t i = 0; i < numCameras; ++i) {
    if (calibrations[i].getOverlapCameraIds().size() == 0) {
      isOverlapGiven = false;
      break;
    }
  }
  if (isOverlapGiven) {
    std::vector<std::vector<bool>> overlap;
    overlap.reserve(numCameras);
    for (size_t i = 0; i < numCameras; ++i) {
      std::vector<bool> overlapCamIds =
          calibrations[i].getOverlapCameraIds(i, numCameras);
      overlap.emplace_back(overlapCamIds);
    }
    nCameraSystem->setOverlaps(overlap);
  } else {
    nCameraSystem->computeOverlaps();
  }
}

void parseImuParameters(cv::FileNode node, ImuParameters *imuParams) {
  cv::FileNode T_BS_ = node["T_BS"];
  if (T_BS_.isSeq()) {
    Eigen::Matrix4d T_BS_e;
    T_BS_e << T_BS_[0], T_BS_[1], T_BS_[2], T_BS_[3], T_BS_[4], T_BS_[5],
        T_BS_[6], T_BS_[7], T_BS_[8], T_BS_[9], T_BS_[10], T_BS_[11], T_BS_[12],
        T_BS_[13], T_BS_[14], T_BS_[15];
    imuParams->T_BS = okvis::kinematics::Transformation(T_BS_e);
  }
  std::stringstream s;
  s << imuParams->T_BS.T();
  VLOG(2) << "IMU with transformation T_BS = \n" << s.str();

  if (node["a_max"].isReal())
    node["a_max"] >> imuParams->a_max;
  if (node["g_max"].isReal())
    node["g_max"] >> imuParams->g_max;
  if (node["sigma_g_c"].isReal())
    node["sigma_g_c"] >> imuParams->sigma_g_c;
  if (node["sigma_a_c"].isReal())
    node["sigma_a_c"] >> imuParams->sigma_a_c;
  if (node["sigma_bg"].isReal())
    node["sigma_bg"] >> imuParams->sigma_bg;
  if (node["sigma_ba"].isReal())
    node["sigma_ba"] >> imuParams->sigma_ba;
  if (node["sigma_gw_c"].isReal())
    node["sigma_gw_c"] >> imuParams->sigma_gw_c;
  if (node["sigma_aw_c"].isReal())
    node["sigma_aw_c"] >> imuParams->sigma_aw_c;

  if (node["imu_rate"].isInt())
    node["imu_rate"] >> imuParams->rate;
  if (node["imu_id"].isInt())
    node["imu_id"] >> imuParams->imuIdx;
  if (node["tau"].isReal())
    node["tau"] >> imuParams->tau;
  if (node["g"].isReal())
    node["g"] >> imuParams->g;

  cv::FileNode gravityDirection = node["gravityDirection"];
  if (gravityDirection.isSeq()) {
    Eigen::Vector3d direction;
    direction << gravityDirection[0], gravityDirection[1], gravityDirection[2];
    imuParams->setGravityDirection(direction.normalized());
  }
  if (node["sigma_gravity_direction"].isReal()) {
    node["sigma_gravity_direction"] >> imuParams->sigma_gravity_direction;
  }
  if (node["a0"].isSeq())
    imuParams->setInitialAccelBias(Eigen::Vector3d((double)(node["a0"][0]),
                                                   (double)(node["a0"][1]),
                                                   (double)(node["a0"][2])));
  cv::FileNode initGyroBias = node["g0"];
  if (initGyroBias.isSeq()) {
    Eigen::Vector3d g0;
    g0 << initGyroBias[0], initGyroBias[1], initGyroBias[2];
    imuParams->setInitialGyroBias(g0);
  }

  if (node["model_name"].isString()) {
    node["model_name"] >> imuParams->model_name;
  }
  if (node["sigma_Mg_element"].isReal()) {
    node["sigma_Mg_element"] >> imuParams->sigma_Mg_element;
  }
  if (node["sigma_Ts_element"].isReal()) {
    node["sigma_Ts_element"] >> imuParams->sigma_Ts_element;
  }
  if (node["sigma_Ma_element"].isReal()) {
    node["sigma_Ma_element"] >> imuParams->sigma_Ma_element;
  }

  cv::FileNode initMg = node["Mg0"];
  if (initMg.isSeq()) {
    Eigen::Matrix<double, 9, 1> Mg;
    Mg << initMg[0], initMg[1], initMg[2], initMg[3], initMg[4], initMg[5],
        initMg[6], initMg[7], initMg[8];
    imuParams->setGyroCorrectionMatrix(Mg);
  }

  cv::FileNode initTs = node["Ts0"];
  if (initTs.isSeq()) {
    Eigen::Matrix<double, 9, 1> Ts;
    Ts << initTs[0], initTs[1], initTs[2], initTs[3], initTs[4], initTs[5],
        initTs[6], initTs[7], initTs[8];
    imuParams->setGyroGSensitivity(Ts);
  }

  cv::FileNode initMa = node["Ma0"];
  if (initMa.isSeq()) {
    Eigen::Matrix<double, 6, 1> Ma;
    Ma << initMa[0], initMa[1], initMa[2], initMa[3], initMa[4], initMa[5];
    imuParams->setAccelCorrectionMatrix(Ma);
  }
}

// Parses booleans from a cv::FileNode. OpenCV sadly has no implementation like this.
bool parseBoolean(cv::FileNode node, bool& val) {
  if (node.isInt()) {
    val = (int) (node) != 0;
    return true;
  }
  if (node.isString()) {
    std::string str = (std::string) (node);
    // cut out first word. str currently contains everything including comments
    str = str.substr(0,str.find(" "));
    // transform it to all lowercase
    std::transform(str.begin(), str.end(), str.begin(), ::tolower);
    /* from yaml.org/type/bool.html:
     * Booleans are formatted as English words
     * (“true”/“false”, “yes”/“no” or “on”/“off”)
     * for readability and may be abbreviated as
     * a single character “y”/“n” or “Y”/“N”. */
    if (str.compare("false")  == 0
        || str.compare("no")  == 0
        || str.compare("n")   == 0
        || str.compare("off") == 0) {
      val = false;
      return true;
    }
    if (str.compare("true")   == 0
        || str.compare("yes") == 0
        || str.compare("y")   == 0
        || str.compare("on")  == 0) {
      val = true;
      return true;
    }
  }
  return false;
}

bool parseMatrixInYaml(cv::FileNode matNode, Eigen::MatrixXd *res, int rows,
                       int cols) {
  if (matNode.empty()) {
    return false;
  } else if (matNode.isSeq()) {
    res->resize(rows, cols);
    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        (*res)(r, c) = matNode[r * cols + c];
      }
    }
    return true;
  } else {
    cv::Mat mat;
    matNode >> mat;
    cv::cv2eigen(mat, *res);
    return true;
  }
}

bool VioParametersReader::getCameraCalibration(
    std::vector<CameraCalibration,Eigen::aligned_allocator<CameraCalibration>> & calibrations,
    cv::FileStorage& configurationFile) {
  bool success = getCalibrationViaConfig(calibrations, configurationFile["cameras"]);
  bool monocularInput = false;
  bool parseOk = parseBoolean(configurationFile["monocular_input"], monocularInput);
  if (parseOk && monocularInput) {
    calibrations.resize(1);
  }
  LOG(INFO) << "Images from " << calibrations.size() << " cameras will be used.";
#ifdef HAVE_LIBVISENSOR
  if (useDriver && !success) {
    // start up sensor
    viSensor = std::shared_ptr<void>(
          new visensor::ViSensorDriver());
    try {
      // use autodiscovery to find sensor. TODO: specify IP in config?
      std::static_pointer_cast<visensor::ViSensorDriver>(viSensor)->init();
    } catch (Exception const &ex) {
      LOG(ERROR) << ex.what();
      exit(1);
    }

    success = getCalibrationViaVisensorAPI(calibrations);
  }
#endif

  return success;
}

// Get the camera calibration via the configuration file.
bool VioParametersReader::getCalibrationViaConfig(
    std::vector<CameraCalibration,Eigen::aligned_allocator<CameraCalibration>> & calibrations,
    cv::FileNode cameraNode) const {

  calibrations.clear();
  bool gotCalibration = false;
  // first check if calibration is available in config file
  if (cameraNode.isSeq()
     && cameraNode.size() > 0) {
    size_t camIdx = 0;
    for (cv::FileNodeIterator it = cameraNode.begin();
        it != cameraNode.end(); ++it) {
      if ((*it).isMap()
          && (*it)["T_SC"].isSeq()
          && (*it)["image_dimension"].isSeq()
          && (*it)["image_dimension"].size() == 2
          && (*it)["distortion_coefficients"].isSeq()
          && (*it)["distortion_coefficients"].size() >= 1
          && (*it)["distortion_type"].isString()
          && (*it)["focal_length"].isSeq()
          && (*it)["focal_length"].size() == 2
          && (*it)["principal_point"].isSeq()
          && (*it)["principal_point"].size() == 2) {
        LOG(INFO) << "Found calibration in configuration file for camera " << camIdx;
        gotCalibration = true;
      } else {
        LOG(WARNING) << "Found incomplete calibration in configuration file for camera " << camIdx
                     << ". Will not use the calibration from the configuration file.";
        return false;
      }
      ++camIdx;
    }
  }
  else
    LOG(INFO) << "Did not find a calibration in the configuration file.";

  if (gotCalibration) {
    size_t camIdx = 0u;
    for (cv::FileNodeIterator it = cameraNode.begin();
        it != cameraNode.end(); ++it, ++camIdx) {
      CameraCalibration calib;

      cv::FileNode T_SC_node = (*it)["T_SC"];
      int downScale = 1;
      if ((*it)["down_scale"].isInt()) downScale = (*it)["down_scale"];
      cv::FileNode imageDimensionNode = (*it)["image_dimension"];
      cv::FileNode distortionCoefficientNode = (*it)["distortion_coefficients"];
      cv::FileNode focalLengthNode = (*it)["focal_length"];
      cv::FileNode principalPointNode = (*it)["principal_point"];

      // extrinsics
      Eigen::Matrix4d T_SC;
      T_SC << T_SC_node[0], T_SC_node[1], T_SC_node[2], T_SC_node[3], T_SC_node[4], T_SC_node[5], T_SC_node[6], T_SC_node[7], T_SC_node[8], T_SC_node[9], T_SC_node[10], T_SC_node[11], T_SC_node[12], T_SC_node[13], T_SC_node[14], T_SC_node[15];
      calib.T_SC = okvis::kinematics::Transformation(T_SC);

      calib.imageDimension << static_cast<double>(imageDimensionNode[0]) /
                                  downScale,
          static_cast<double>(imageDimensionNode[1]) / downScale;
      calib.distortionCoefficients.resize(distortionCoefficientNode.size());
      for(size_t i=0; i<distortionCoefficientNode.size(); ++i) {
        calib.distortionCoefficients[i] = distortionCoefficientNode[i];
      }
      calib.focalLength << static_cast<double>(focalLengthNode[0]) / downScale,
          static_cast<double>(focalLengthNode[1]) / downScale;
      calib.principalPoint << static_cast<double>(principalPointNode[0]) /
                                  downScale,
          static_cast<double>(principalPointNode[1]) / downScale;
      calib.distortionType = (std::string)((*it)["distortion_type"]);

      if ((*it)["image_delay"].isReal()) {
        (*it)["image_delay"] >> calib.imageDelaySecs;
      } else {
        calib.imageDelaySecs = 0.0;
        LOG(WARNING) << "'image_delay' parameter for camera " << camIdx
                     << " missing in configuration file. Setting to "
                     << calib.imageDelaySecs;
      }
      if ((*it)["image_readout_time"].isReal()) {
        (*it)["image_readout_time"] >> calib.readoutTimeSecs;
          const double upper = 1.0;
          const double lower = 0.0;
          OKVIS_ASSERT_LE(Exception,
                          calib.readoutTimeSecs, upper,
                          "image_readout_time should be no more than " + std::to_string(upper) + " sec.");
          OKVIS_ASSERT_GE(Exception,
                          calib.readoutTimeSecs, lower,
                          "image_readout_time should be no less than " + std::to_string(lower) + " sec.");
      } else {
        calib.readoutTimeSecs = 0.0;
        LOG(WARNING) << "'image_readout_time' parameter for camera " << camIdx <<
                        " missing in configuration file. Setting to "
                     << calib.readoutTimeSecs;
      }

      if ((*it)["extrinsic_rep"].isString()) {
        calib.extrinsicRepName =
            static_cast<std::string>((*it)["extrinsic_rep"]);
      }
      if ((*it)["projection_intrinsic_rep"].isString()) {
        calib.projectionIntrinsicRepName =
            static_cast<std::string>((*it)["projection_intrinsic_rep"]);
      }
      if ((*it)["cam_overlaps"].isInt()) {
        int overlapCamId;
        (*it)["cam_overlaps"] >> overlapCamId;
        calib.overlapCameraIds = {overlapCamId};
      }
      if ((*it)["cam_overlaps"].isSeq()) {
        size_t numOverlapCams = (*it)["cam_overlaps"].size();
        calib.overlapCameraIds.reserve(numOverlapCams);
        for (size_t i = 0 ; i < numOverlapCams; ++i) {
          calib.overlapCameraIds.emplace_back((*it)["cam_overlaps"][i]);
        }
      }
      calibrations.push_back(calib);
    }
  }
  return gotCalibration;
}

// Get the camera calibrations via the visensor API.
bool VioParametersReader::getCalibrationViaVisensorAPI(
    std::vector<CameraCalibration,Eigen::aligned_allocator<CameraCalibration>> & calibrations) const{
#ifdef HAVE_LIBVISENSOR
  if (viSensor == nullptr) {
    LOG(ERROR) << "Tried to get calibration from the sensor. But the sensor is not set up.";
    return false;
  }

  calibrations.clear();

  std::vector<visensor::SensorId::SensorId> listOfCameraIds =
      std::static_pointer_cast<visensor::ViSensorDriver>(viSensor)->getListOfCameraIDs();

  for (auto it = listOfCameraIds.begin(); it != listOfCameraIds.end(); ++it) {
    visensor::ViCameraCalibration calibrationFromAPI;
    okvis::VioParametersReader::CameraCalibration calibration;
    if(!std::static_pointer_cast<visensor::ViSensorDriver>(viSensor)->getCameraCalibration(*it,calibrationFromAPI)) {
      LOG(ERROR) << "Reading the calibration via the sensor API failed.";
      calibrations.clear();
      return false;
    }
    LOG(INFO) << "Reading the calbration for camera " << size_t(*it) << " via API successful";
    double* R = calibrationFromAPI.R;
    double* t = calibrationFromAPI.t;
    // getCameraCalibration apparently gives T_CI back.
    //(Confirmed by comparing it to output of service)
    Eigen::Matrix4d T_CI;
    T_CI << R[0], R[1], R[2], t[0],
            R[3], R[4], R[5], t[1],
            R[6], R[7], R[8], t[2],
            0,    0,    0,    1;
    okvis::kinematics::Transformation T_CI_okvis(T_CI);
    calibration.T_SC = T_CI_okvis.inverse();

    calibration.focalLength << calibrationFromAPI.focal_point[0],
                               calibrationFromAPI.focal_point[1];
    calibration.principalPoint << calibrationFromAPI.principal_point[0],
                                  calibrationFromAPI.principal_point[1];
    calibration.distortionCoefficients.resize(4); // FIXME: 8 coeff support?
    calibration.distortionCoefficients << calibrationFromAPI.dist_coeff[0],
                                          calibrationFromAPI.dist_coeff[1],
                                          calibrationFromAPI.dist_coeff[2],
                                          calibrationFromAPI.dist_coeff[3];
    calibration.imageDimension << 752, 480;
    calibration.distortionType = "plumb_bob";
    calibrations.push_back(calibration);
  }

  return calibrations.empty() == false;
#else
  static_cast<void>(calibrations); // unused
  LOG(ERROR) << "Tried to get calibration directly from the sensor. However libvisensor was not found.";
  return false;
#endif
}


}  // namespace okvis
