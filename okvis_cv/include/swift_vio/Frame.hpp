
/**
 * @file okvis/Frame.hpp
 * @brief Header file for the Frame class.
 */

#ifndef INCLUDE_SWIFT_VIO_FRAME_HPP_
#define INCLUDE_SWIFT_VIO_FRAME_HPP_

#include <Eigen/StdVector>
#include <Eigen/Core>
#include <memory>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#include <opencv2/core/core.hpp> // Code that causes warning goes here
#pragma GCC diagnostic pop
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Woverloaded-virtual"
#include <opencv2/features2d/features2d.hpp> // Code that causes warning goes here
#pragma GCC diagnostic pop
#include <okvis/Time.hpp>
#include <okvis/assert_macros.hpp>
#include "okvis/cameras/CameraBase.hpp"

namespace swift_vio {

typedef Eigen::Vector3f KeypointReduced; // x, y, size

/// \class Frame
/// \brief A single camera frame equipped with keypoint detector / extractor.
class Frame
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  OKVIS_DEFINE_EXCEPTION(Exception, std::runtime_error)

  /// \brief a default constructor
  inline Frame()
  {
  }

  /// \brief A constructor that uses the image, specified geometry,
  /// detector and extractor
  /// @param[in] image The image.
  inline Frame(const cv::Mat & image);

  /// \brief A simple destructor
  inline virtual ~Frame()
  {
  }

  /// \brief Set the frame image;
  /// @param[in] image The image.
  inline void setImage(const cv::Mat & image);

  /// \brief Obtain the image
  /// \return The image.
  inline const cv::Mat & image() const;

  /// \brief Detect keypoints. This uses virtual function calls.
  ///        That's a negligibly small overhead for many detections.
  /// \return The number of detected points.
//  inline int detect();

  /// \brief Describe keypoints. This uses virtual function calls.
  ///        That's a negligibly small overhead for many detections.
  /// \param extractionDirection The extraction direction in camera frame
  /// \return The number of detected points.
//  inline int describe(
//      const Eigen::Vector3d & extractionDirection = Eigen::Vector3d(0, 0, 1));

  /// \brief Describe keypoints. This uses virtual function calls.
  ///        That's a negligibly small overhead for many detections.
  /// \tparam GEOMETRY_T The type for the camera geometry requested.
  /// \param extractionDirection the extraction direction in camera frame
  /// \return The number of detected points.
//  template<class GEOMETRY_T>
//  inline int describeAs(
//      const Eigen::Vector3d & extractionDirection = Eigen::Vector3d(0, 0, 1));

  /// \brief Access a specific keypoint in OpenCV format
  /// @param[in] keypointIdx The requested keypoint's index.
  /// @param[out] keypoint The requested keypoint.
  /// \return whether or not the operation was successful.
  inline bool getCvKeypoint(size_t keypointIdx, cv::KeyPoint & keypoint) const;

  /// \brief Get a specific keypoint
  /// @param[in] keypointIdx The requested keypoint's index.
  /// @param[out] keypoint The requested keypoint.
  /// \return whether or not the operation was successful.
  inline bool getKeypoint(size_t keypointIdx, Eigen::Vector2d & keypoint) const;

  /// \brief Get the size of a specific keypoint
  /// @param[in] keypointIdx The requested keypoint's index.
  /// @param[out] keypointSize The requested keypoint's size.
  /// \return whether or not the operation was successful.
  inline bool getKeypointSize(size_t keypointIdx, double & keypointSize) const;

  inline bool getKeypointSize(size_t keypointIdx, float &keypointSize) const;

  /// \brief Access the descriptor -- CAUTION: high-speed version.
  /// @param[in] keypointIdx The requested keypoint's index.
  /// \return The descriptor data pointer; NULL if out of bounds.
  inline const unsigned char * keypointDescriptor(size_t keypointIdx) const;

  /// \brief Provide keypoints externally.
  /// @param[in] keypoints A vector of keyoints.
  /// \return whether or not the operation was successful.
  inline bool resetKeypoints(const std::vector<cv::KeyPoint> & keypoints);

  /// \brief provide descriptors externally
  /// @param[in] descriptors A vector of descriptors.
  /// \return whether or not the operation was successful.
  inline bool resetDescriptors(const cv::Mat & descriptors);

  /// \brief Get the number of keypoints.
  /// \return The number of keypoints.
  inline size_t numKeypoints() const;

  cv::Mat getDescriptors() const {
    return descriptors_;
  }

  const std::vector<cv::KeyPoint>& getKeypoints() const {
    return keypoints_;
  }

  cv::Mat &descriptors() {
    return descriptors_;
  }

  std::vector<cv::KeyPoint> &keypoints() {
    return keypoints_;
  }

  inline void setTimestamp(const okvis::Time& stamp) {
    stamp_ = stamp;
  }

  inline okvis::Time timestamp() const {
    return stamp_;
  }

  inline cv::Mat copyDescriptorsAt(
      const std::vector<int>& descriptorIndices) const;

  inline std::vector<KeypointReduced, Eigen::aligned_allocator<KeypointReduced>>
  copyKeypointsAt(const std::vector<int>& keypointIndices) const;

  inline std::vector<KeypointReduced, Eigen::aligned_allocator<KeypointReduced>>
  copyKeypoints() const;

  void createTestImage(int rows, int cols, int type) {
    image_ = cv::Mat(rows, cols, type, cv::Scalar(168));
  }

 protected:
  cv::Mat image_;  ///< the image as OpenCV's matrix
  std::vector<cv::KeyPoint> keypoints_;  ///< we store keypoints using OpenCV's struct
  cv::Mat descriptors_;  ///< we store the descriptors using OpenCV's matrices
  okvis::Time stamp_;
};

struct BareFrame {
  okvis::Time stamp_;
  std::vector<uint64_t> landmarkIds_;

  explicit BareFrame(okvis::Time time) : stamp_(time) {}
};

inline cv::Mat selectDescriptors(const cv::Mat descriptors,
                                 const std::vector<int>& descriptorIndices);

}  // namespace swift_vio

#include "implementation/Frame.hpp"

#endif /* INCLUDE_SWIFT_VIO_FRAME_HPP_ */
