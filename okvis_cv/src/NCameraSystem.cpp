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
 *  Created on: Apr 1, 2015
 *      Author: Stefan Leutenegger (s.leutenegger@imperial.ac.uk)
 *    Modified: Andreas Forster (an.forster@gmail.com)
 *********************************************************************************/

/**
 * @file NCameraSystem.cpp
 * @brief Sourc file for the NCameraSystem.cpp class.
 * @author Stefan Leutenegger
 * @author Andreas Forster
 */

#include "okvis/cameras/NCameraSystem.hpp"

#include <glog/logging.h>

#include <okvis/cameras/EquidistantDistortion.hpp>
#include <okvis/cameras/EUCM.hpp>
#include <okvis/cameras/NoDistortion.hpp>
#include <okvis/cameras/PinholeCamera.hpp>
#include <okvis/cameras/RadialTangentialDistortion.hpp>
#include <okvis/cameras/RadialTangentialDistortion8.hpp>
#include <okvis/cameras/FovDistortion.hpp>

#include <opencv2/highgui/highgui.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {
/// \brief cameras Namespace for camera-related functionality.
namespace cameras {
std::string NCameraSystem::DistortionTypeToKalibrModel(DistortionType dt) {
  switch (dt) {
  case Equidistant:
    return "equidistant";
  case RadialTangential:
    return "radtan";
  case RadialTangential8:
    return "radtan8";
  case FOV:
    return "fov";
  case EUCM:
    return "eucm";
  case NoDistortion:
    return "no";
  default:
    return "no";
  }
}

/// \brief compute all the overlaps of fields of view. Attention: can be expensive.
void NCameraSystem::computeOverlaps()
{
  LOG(INFO) << "Computing overlaps between " << T_SC_.size() << " cameras... It may take a few seconds!";
  OKVIS_ASSERT_TRUE_DBG(
      Exception, T_SC_.size() == cameraGeometries_.size(),
      "Number of extrinsics must match number of camera models!");

  overlapMats_.resize(cameraGeometries_.size());
  overlaps_.resize(cameraGeometries_.size());
  for (size_t cameraIndexSeenBy = 0; cameraIndexSeenBy < overlapMats_.size();
      ++cameraIndexSeenBy) {
    overlapMats_[cameraIndexSeenBy].resize(cameraGeometries_.size());
    overlaps_[cameraIndexSeenBy].resize(cameraGeometries_.size());
    for (size_t cameraIndex = 0; cameraIndex < overlapMats_.size();
        ++cameraIndex) {

      std::shared_ptr<const CameraBase> camera = cameraGeometries_[cameraIndex];

      // self-visibility is trivial:
      if (cameraIndex == cameraIndexSeenBy) {
        // sizing the overlap map:
        overlapMats_[cameraIndexSeenBy][cameraIndex] = cv::Mat::ones(
            camera->imageHeight(), camera->imageWidth(), CV_8UC1);
        overlaps_[cameraIndexSeenBy][cameraIndex] = true;
      } else {
        // sizing the overlap map:
        const size_t height = camera->imageHeight();
        const size_t width = camera->imageWidth();
        cv::Mat& overlapMat = overlapMats_[cameraIndexSeenBy][cameraIndex];
        overlapMat = cv::Mat::zeros(height, width, CV_8UC1);
        // go through all the pixels:
        std::shared_ptr<const CameraBase> otherCamera =
            cameraGeometries_[cameraIndexSeenBy];
        const okvis::kinematics::Transformation T_Cother_C =
            T_SC_[cameraIndexSeenBy]->inverse() * (*T_SC_[cameraIndex]);
        bool hasOverlap = false;
        for (size_t u = 0; u < width; ++u) {
          for (size_t v = 0; v < height; ++v) {
            // backproject
            Eigen::Vector3d ray_C;
            camera->backProject(Eigen::Vector2d(double(u), double(v)), &ray_C);
            // project into other camera
            Eigen::Vector3d ray_Cother = T_Cother_C.C() * ray_C;  // points at infinity, i.e. we only do rotation
            Eigen::Vector2d imagePointInOtherCamera;
            CameraBase::ProjectionStatus status = otherCamera->project(
                ray_Cother, &imagePointInOtherCamera);

            // check the result
            if (status == CameraBase::ProjectionStatus::Successful) {

              Eigen::Vector3d verificationRay;
              otherCamera->backProject(imagePointInOtherCamera,&verificationRay);

              // to avoid an artefact of some distortion models, check again
              // note: (this should be fixed in the distortion implementation)
              if(fabs(ray_Cother.normalized().transpose()*verificationRay.normalized()-1.0)<1.0e-10) {
                // fill in the matrix:
                overlapMat.at<uchar>(v,u) = 1;
                // and remember there is some overlap at all.
                if (!hasOverlap) {
                  overlaps_[cameraIndexSeenBy][cameraIndex] = true;
                }
                hasOverlap = true;
              }
            }
          }
        }
      }
      //std::stringstream name;
      //name << (cameraIndexSeenBy)<<"+"<<(cameraIndex);
      //cv::imshow(name.str().c_str(),255*overlapMats_[cameraIndexSeenBy][cameraIndex]);
    }
  }
  //cv::waitKey();
}

std::shared_ptr<NCameraSystem> NCameraSystem::deepCopy() const {
  std::shared_ptr<NCameraSystem> rig(new NCameraSystem());
  for (size_t i = 0u; i < T_SC_.size(); ++i) {
    std::shared_ptr<okvis::kinematics::Transformation> T_SC(
        new okvis::kinematics::Transformation(*T_SC_[i]));
    auto geometry = cloneCameraGeometry(cameraGeometries_[i]);
    rig->addCamera(T_SC, geometry, distortionTypes_[i], proj_opt_rep_[i],
                  extrinsic_opt_rep_[i]);
  }
  return rig;
}

std::shared_ptr<okvis::cameras::CameraBase> cloneCameraGeometry(
    std::shared_ptr<const okvis::cameras::CameraBase> cameraGeometry) {
  std::string geometryType = cameraGeometry->type();
  std::string distortionType = cameraGeometry->distortionType();
  Eigen::VectorXd intrinsic_vec;
  cameraGeometry->getIntrinsics(intrinsic_vec);
  uint64_t id = cameraGeometry->id();
  if (geometryType.find("PinholeCamera<") == 0) {
    const int distortion_start_index = 4;
    if (strcmp(distortionType.c_str(), "EquidistantDistortion") == 0) {
      return std::shared_ptr<okvis::cameras::CameraBase>(
          new okvis::cameras::PinholeCamera<
              okvis::cameras::EquidistantDistortion>(
              cameraGeometry->imageWidth(), cameraGeometry->imageHeight(),
              intrinsic_vec[0], intrinsic_vec[1], intrinsic_vec[2],
              intrinsic_vec[3],
              okvis::cameras::EquidistantDistortion(
                  intrinsic_vec[distortion_start_index],
                  intrinsic_vec[distortion_start_index + 1],
                  intrinsic_vec[distortion_start_index + 2],
                  intrinsic_vec[distortion_start_index + 3]),
              cameraGeometry->imageDelay(), cameraGeometry->readoutTime(),
              id));

    } else if (strcmp(distortionType.c_str(), "RadialTangentialDistortion") ==
               0) {
      return std::shared_ptr<okvis::cameras::CameraBase>(
          new okvis::cameras::PinholeCamera<
              okvis::cameras::RadialTangentialDistortion>(
              cameraGeometry->imageWidth(), cameraGeometry->imageHeight(),
              intrinsic_vec[0], intrinsic_vec[1], intrinsic_vec[2],
              intrinsic_vec[3],
              okvis::cameras::RadialTangentialDistortion(
                  intrinsic_vec[distortion_start_index],
                  intrinsic_vec[distortion_start_index + 1],
                  intrinsic_vec[distortion_start_index + 2],
                  intrinsic_vec[distortion_start_index + 3]),
              cameraGeometry->imageDelay(), cameraGeometry->readoutTime(),
              id));

    } else if (strcmp(distortionType.c_str(), "RadialTangentialDistortion8") ==
               0) {
      return std::shared_ptr<okvis::cameras::CameraBase>(
          new okvis::cameras::PinholeCamera<
              okvis::cameras::RadialTangentialDistortion8>(
              cameraGeometry->imageWidth(), cameraGeometry->imageHeight(),
              intrinsic_vec[0], intrinsic_vec[1], intrinsic_vec[2],
              intrinsic_vec[3],
              okvis::cameras::RadialTangentialDistortion8(
                  intrinsic_vec[distortion_start_index],
                  intrinsic_vec[distortion_start_index + 1],
                  intrinsic_vec[distortion_start_index + 2],
                  intrinsic_vec[distortion_start_index + 3],
                  intrinsic_vec[distortion_start_index + 4],
                  intrinsic_vec[distortion_start_index + 5],
                  intrinsic_vec[distortion_start_index + 6],
                  intrinsic_vec[distortion_start_index + 7]),
              cameraGeometry->imageDelay(), cameraGeometry->readoutTime(),
              id));
    } else if (strcmp(distortionType.c_str(), "NoDistortion") == 0) {
      return std::shared_ptr<okvis::cameras::CameraBase>(
          new okvis::cameras::PinholeCamera<okvis::cameras::NoDistortion>(
              cameraGeometry->imageWidth(), cameraGeometry->imageHeight(),
              intrinsic_vec[0], intrinsic_vec[1], intrinsic_vec[2],
              intrinsic_vec[3], okvis::cameras::NoDistortion(),
              cameraGeometry->imageDelay(), cameraGeometry->readoutTime(),
              id));
    } else if (strcmp(distortionType.c_str(), "FovDistortion") == 0) {
      return std::shared_ptr<okvis::cameras::CameraBase>(
          new okvis::cameras::PinholeCamera<
              okvis::cameras::FovDistortion>(
              cameraGeometry->imageWidth(), cameraGeometry->imageHeight(),
              intrinsic_vec[0], intrinsic_vec[1], intrinsic_vec[2],
              intrinsic_vec[3],
              okvis::cameras::FovDistortion(
                  intrinsic_vec[distortion_start_index]),
              cameraGeometry->imageDelay(), cameraGeometry->readoutTime(),
              id));
    } else {
      std::cerr << "unrecognized distortion type " << distortionType << ".\n";
    }
  } else if (strcmp(geometryType.c_str(), "EUCM") == 0 || strcmp(geometryType.c_str(), "eucm") == 0) {
    return std::shared_ptr<okvis::cameras::CameraBase>(new okvis::cameras::EUCM(
        cameraGeometry->imageWidth(), cameraGeometry->imageHeight(),
        intrinsic_vec[0], intrinsic_vec[1], intrinsic_vec[2], intrinsic_vec[3],
        intrinsic_vec[4], intrinsic_vec[5], cameraGeometry->imageDelay(),
        cameraGeometry->readoutTime(), id));
  }else {
    std::cerr << "unrecognized camera geometry type " << cameraGeometry->type() << ".\n";
  }
  return std::shared_ptr<okvis::cameras::CameraBase>();
}

}  // namespace cameras
}  // namespace okvis

