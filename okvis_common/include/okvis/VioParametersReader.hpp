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
 * @file VioParametersReader.hpp
 * @brief Header file for the VioParametersReader class.
 * @author Stefan Leutenegger
 * @author Andreas Forster
 */

#ifndef INCLUDE_OKVIS_VIOPARAMETERSREADER_HPP_
#define INCLUDE_OKVIS_VIOPARAMETERSREADER_HPP_

#include <string>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#include <opencv2/core/core.hpp>
#pragma GCC diagnostic pop

#include <okvis/assert_macros.hpp>
#include <okvis/Parameters.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {

/**
 * @brief This class reads and parses config file.
 */
class VioParametersReader{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  OKVIS_DEFINE_EXCEPTION(Exception,std::runtime_error)

  /// \brief The default constructor.
  VioParametersReader();

  /**
   * @brief The constructor. This calls readConfigFile().
   * @param filename Configuration filename.
   */
  VioParametersReader(const std::string& filename);

  /// @brief Trivial destructor.
  virtual ~VioParametersReader() {}

  /**
   * @brief Read and parse a config file.
   *        To get the result call getParameters().
   * @param filename Configuration filename.
   */
  void readConfigFile(const std::string& filename);

  /**
   * @brief Get parameters.
   * @param[out] parameters A copy of the parameters.
   * @return True if parameters have been read from a configuration file. If it
   *         returns false then the variable \e parameters has not been changed.
   */
  bool getParameters(okvis::VioParameters& parameters) const {
    if(readConfigFile_)
      parameters = vioParameters_;
    return readConfigFile_;
  }

  /// Directly interface with driver without ROS message passing.
  bool useDriver;

  std::shared_ptr<void> viSensor;

 protected:

  /// @brief Struct that contains all the camera calibration information.
  struct CameraCalibration {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    okvis::kinematics::Transformation T_SC;   ///< Transformation from camera to sensor (IMU) frame.
    Eigen::Vector2d imageDimension;           ///< Image dimension. [pixels]
    std::vector<double> distortionCoefficients;   ///< Distortion Coefficients.
    Eigen::Vector2d focalLength;              ///< Focal length.
    Eigen::Vector2d principalPoint;           ///< Principal point.
    std::string distortionType;               ///< Distortion type. ('radialtangential' 'plumb_bob' 'equdistant')
    double imageDelaySecs;                    ///< Remaining image delay after compensating for SensorsInformation.imageDelay
    double readoutTimeSecs;
    std::string projectionIntrinsicRepName;
    std::string extrinsicRepName;
    std::vector<int> overlapCameraIds;
    CameraCalibration() : projectionIntrinsicRepName("FXY_CXY"), extrinsicRepName("P_BC_Q_BC") {}
    std::string toString() const {
      std::stringstream ss;
      ss << "T_SC\n"
         << T_SC.T3x4() << "\nimage dimension " << imageDimension.transpose()
         << ", distortion type " << distortionType.c_str() << ", ["
         << Eigen::Map<const Eigen::VectorXd>(distortionCoefficients.data(),
                                              distortionCoefficients.size()).transpose() 
        << "]\n(fx, fy) "
         << focalLength.transpose() << ", (cx, cy) "
         << principalPoint.transpose() << ", imageDelay " << imageDelaySecs
         << " secs, readoutTime " << readoutTimeSecs << " secs\n"
         << "projectionIntrinsicRep " << projectionIntrinsicRepName
         << ", extrinsicRep " << extrinsicRepName;
      if (overlapCameraIds.size()) {
        ss << ", overlap cameras:";
        for (auto id : overlapCameraIds) {
          ss << " " << id;
        }
        ss << "\n";
      }
      return ss.str();
    }
    /**
     * @brief getOverlapCameraIds of this camera
     * @param id index of this camera
     * @param numCameras number of cameras in the NCameraSystem
     * @return ids of cameras of views overlapping this camera.
     */
    std::vector<bool> getOverlapCameraIds(size_t id, size_t numCameras) const {
      std::vector<bool> overlapStatus(numCameras, false);
      overlapStatus[id] = true; // self
      for (auto camId : overlapCameraIds)
        overlapStatus[camId] = true;
      return overlapStatus;
    }

    const std::vector<int> &getOverlapCameraIds() const {
      return overlapCameraIds;
    }
  };

  /// If readConfigFile() has been called at least once this is true
  bool readConfigFile_;
  /// The parameters.
  okvis::VioParameters vioParameters_;

  /**
   * @brief Get the camera calibration. This looks for the calibration in the
   *        configuration file first. If this fails it will directly get the calibration
   *        from the sensor, if useDriver is set to true.
   * @remark Overload this function if you want to get the calibrations via ROS for example.
   * @param calibrations The calibrations.
   * @param configurationFile The config file.
   * @return True if reading of the calibration was successful.
   */
  virtual bool getCameraCalibration(
      std::vector<CameraCalibration,Eigen::aligned_allocator<CameraCalibration>> & calibrations,
      cv::FileStorage& configurationFile);

  /**
   * @brief Get the camera calibration via the configuration file.
   * @param[out] calibrations Read calibration.
   * @param[in] cameraNode File node pointing to the cameras sequence.
   * @return True if reading and parsing of calibration was successful.
   */
  bool getCalibrationViaConfig(
      std::vector<CameraCalibration,Eigen::aligned_allocator<CameraCalibration>> & calibrations,
      cv::FileNode cameraNode) const;

  /**
   * @brief Get the camera calibrations via the visensor API.
   * @param[out] calibrations The calibrations.
   * @return True if successful.
   */
  bool getCalibrationViaVisensorAPI(
      std::vector<CameraCalibration,Eigen::aligned_allocator<CameraCalibration>> & calibrations) const;

  /**
   * @brief build CameraSystem from CameraCalibration.
   * @param calibrations
   * @param nCameraSystem
   */
  static void buildCameraSystem(
      const std::vector<CameraCalibration,
                        Eigen::aligned_allocator<CameraCalibration>>
          &calibrations,
      okvis::cameras::NCameraSystem *nCameraSystem);
};

/**
 * @brief Parses booleans from a cv::FileNode. OpenCV sadly has no implementation like this.
 * @param[in] node The file node.
 * @param[out] val The parsed value.
 * @return True if parsing was successful. If it returns false then the variable \e val has not
 *         been changed.
 */
bool parseBoolean(cv::FileNode node, bool& val);

template <typename E> bool parseEnum(cv::FileNode n, E *e) {
  if (n.isInt()) {
    int v;
    n >> v;
    *e = static_cast<E>(v);
  } else if (n.isString()) {
    std::string s;
    n >> s;
    EnumFromString(s, e);
  } else {
    return false;
  }
  return true;
}

void parseImuParameters(cv::FileNode node, ImuParameters *imuParams);

void parseCameraNoises(cv::FileNode cameraParamNode,
                       CameraNoiseParameters *camera_noise);
/**
 * @brief parseMatrixInYaml
 * For a matrix attribute say T_cam_imu, two forms are supported.
 * 1. seq form example
 *   T_cam_imu:
 *     [-0.9995110484978581, 0.030299116376600627, -0.0077218830287333565, -0.053697434688869734,
 *      0.008104079263822521, 0.012511643720192351, -0.9998888851620987, -0.046131737923635924,
 *      -0.030199136245891378, -0.9994625667418545, -0.012751072573940885, -0.07149261284195751,
 *      0.0, 0.0, 0.0, 1.0]
 * 2. mat form example
 *   T_cam_imu: !!opencv-matrix
 *     rows: 4
 *     cols: 4
 *     dt: d
 *     data: [-0.9995110484978581, 0.030299116376600627, -0.0077218830287333565, -0.053697434688869734,
 *            0.008104079263822521, 0.012511643720192351, -0.9998888851620987, -0.046131737923635924,
 *            -0.030199136245891378, -0.9994625667418545, -0.012751072573940885, -0.07149261284195751,
 *            0.0, 0.0, 0.0, 1.0]
 * @param matNode
 * @param res
 * @param rows
 * @param cols
 * @return
 */
bool parseMatrixInYaml(cv::FileNode matNode, Eigen::MatrixXd *res, int rows,
                       int cols);
}

#endif /* INCLUDE_OKVIS_VIOPARAMETERSREADER_HPP_ */
