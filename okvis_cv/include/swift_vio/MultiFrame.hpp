/**
 * @file swift_vio/MultiFrame.hpp
 * @brief Header file for the MultiFrame class.
 */

#ifndef INCLUDE_SWIFT_VIO_MULTIFRAME_HPP_
#define INCLUDE_SWIFT_VIO_MULTIFRAME_HPP_

#include <memory>
#include <unordered_map>

#include <okvis/assert_macros.hpp>
#include <swift_vio/Frame.hpp>

#include <okvis/MultiFrame.hpp>  // backward compatibility

namespace swift_vio {

/// \class MultiFrame
/// \brief A multi camera frame that uses okvis::Frame underneath.
class MultiFrame
{
 public:

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  OKVIS_DEFINE_EXCEPTION(Exception,std::runtime_error)

  /// \brief Default constructor
  inline MultiFrame();

  /// \brief Construct from NCameraSystem
  /// @param[in] cameraSystem The camera system for which this is a multi-frame.
  /// @param[in] timestamp The time this frame was recorded.
  /// @param[in] id A unique frame Id.
  inline MultiFrame(int numCameras, const okvis::Time & timestamp, uint64_t id = 0);

  /// \brief Destructor...
  inline virtual ~MultiFrame();

  /// \brief (Re)set the NCameraSystem -- which clears the frames as well.
  /// @param[in] cameraSystem The camera system for which this is a multi-frame.
  inline void resetFrames(int numCameras);

  /// \brief (Re)set the timestamp
  /// @param[in] timestamp The time this frame was recorded.
  inline void setTimestamp(const okvis::Time & timestamp);

  /// \brief (Re)set the timestamp for frame at cameraIdx.
  inline void setTimestamp(size_t cameraIdx, const okvis::Time& stamp);

  /// \brief (Re)set the id
  /// @param[in] id A unique frame Id.
  inline void setId(uint64_t id);

  /// \brief Obtain the frame timestamp
  /// \return The time this frame was recorded.
  inline const okvis::Time & timestamp() const;

  inline okvis::Time timestamp(size_t cameraIdx) const;

  /// \brief Obtain the frame id
  /// \return The unique frame Id.
  inline uint64_t id() const;

  /// \brief The number of frames/cameras
  /// \return How many individual frames/cameras there are.
  inline size_t numFrames() const;

  /// \brief Get the extrinsics of a camera
  /// @param[in] cameraIdx The camera index for which the extrinsics are queried.
  /// \return The extrinsics as T_SC.
//  inline std::shared_ptr<const okvis::kinematics::Transformation> T_SC(
//      size_t cameraIdx) const;

  //////////////////////////////////////////////////////////////
  /// \name The following mirror the Frame functionality.
  /// @{

  /// \brief Set the frame image;
  /// @param[in] cameraIdx The camera index that took the image.
  /// @param[in] image The image.
  inline void setImage(size_t cameraIdx, const cv::Mat & image);

  /// \brief Set the geometry
  /// @param[in] cameraIdx The camera index.
  /// @param[in] cameraGeometry The camera geometry.
//  inline void setGeometry(
//      size_t cameraIdx,
//      std::shared_ptr<const okvis::cameras::CameraBase> cameraGeometry);

  /// \brief Set the detector
  /// @param[in] cameraIdx The camera index.
  /// @param[in] detector The detector to be used.
//  inline void setDetector(size_t cameraIdx,
//                          std::shared_ptr<cv::FeatureDetector> detector);

  /// \brief Set the extractor
  /// @param[in] cameraIdx The camera index.
  /// @param[in] extractor The extractor to be used.
//  inline void setExtractor(
//      size_t cameraIdx,
//      std::shared_ptr<cv::DescriptorExtractor> extractor);

  /// \brief Obtain the image
  /// @param[in] cameraIdx The camera index.
  /// \return The image.
  inline const cv::Mat & image(size_t cameraIdx) const;

  /// \brief get the base class geometry (will be slow to use)
  /// @param[in] cameraIdx The camera index.
  /// \return The camera geometry.
//  inline std::shared_ptr<const okvis::cameras::CameraBase> geometry(
//      size_t cameraIdx) const;

  /// \brief Get the specific geometry (will be fast to use)
  /// \tparam GEOMETRY_T The type for the camera geometry requested.
  /// @param[in] cameraIdx The camera index.
  /// \return The camera geometry.
//  template<class GEOMETRY_T>
//  inline std::shared_ptr<const GEOMETRY_T> geometryAs(size_t cameraIdx) const;

  /// \brief Detect keypoints. This uses virtual function calls.
  ///        That's a negligibly small overhead for many detections.
  /// \return The number of detected points.
//  inline int detect(size_t cameraIdx);

  /// \brief Describe keypoints. This uses virtual function calls.
  ///        That's a negligibly small overhead for many detections.
  /// @param[in] cameraIdx The camera index.
  /// @param[in] extractionDirection The extraction direction in camera frame
  /// \return the number of detected points.
//  inline int describe(size_t cameraIdx,
//                      const Eigen::Vector3d & extractionDirection =
//                          Eigen::Vector3d(0, 0, 1));
  /// \brief Describe keypoints. This uses virtual function calls.
  ///        That's a negligibly small overhead for many detections.
  /// \tparam GEOMETRY_T The type for the camera geometry requested.
  /// @param[in] cameraIdx The camera index.
  /// @param[in] extractionDirection The extraction direction in camera frame
  /// \return the number of detected points.
//  template<class GEOMETRY_T>
//  inline int describeAs(size_t cameraIdx,
//                        const Eigen::Vector3d & extractionDirection =
//                            Eigen::Vector3d(0, 0, 1));

  /// \brief Access a specific keypoint in OpenCV format
  /// @param[in] cameraIdx The camera index.
  /// @param[in] keypointIdx The requested keypoint's index.
  /// @param[out] keypoint The requested keypoint.
  /// \return whether or not the operation was successful.
  inline bool getCvKeypoint(size_t cameraIdx, size_t keypointIdx,
                            cv::KeyPoint & keypoint) const;

  /// \brief Get a specific keypoint
  /// @param[in] cameraIdx The camera index.
  /// @param[in] keypointIdx The requested keypoint's index.
  /// @param[out] keypoint The requested keypoint.
  /// \return whether or not the operation was successful.
  inline bool getKeypoint(size_t cameraIdx, size_t keypointIdx,
                          Eigen::Vector2d & keypoint) const;

  /// \brief Get the size of a specific keypoint
  /// @param[in] cameraIdx The camera index.
  /// @param[in] keypointIdx The requested keypoint's index.
  /// @param[out] keypointSize The requested keypoint's size.
  /// \return whether or not the operation was successful.
  inline bool getKeypointSize(size_t cameraIdx, size_t keypointIdx,
                              double & keypointSize) const;

  inline bool getKeypointSize(size_t cameraIdx, size_t keypointIdx,
                              float &keypointSize) const;

  /// \brief Access the descriptor -- CAUTION: high-speed version.
  /// @param[in] cameraIdx The camera index.
  /// @param[in] keypointIdx The requested keypoint's index.
  /// \return The descriptor data pointer; NULL if out of bounds.
  inline const unsigned char * keypointDescriptor(size_t cameraIdx,
                                                  size_t keypointIdx) const;


  /// \brief provide keypoints externally
  /// @param[in] cameraIdx The camera index.
  /// @param[in] keypoints A vector of keyoints.
  /// \return whether or not the operation was successful.
  inline bool resetKeypoints(size_t cameraIdx,
                             const std::vector<cv::KeyPoint> & keypoints);

  /// \brief provide descriptors externally
  /// @param[in] cameraIdx The camera index.
  /// @param[in] descriptors A vector of descriptors.
  /// \return whether or not the operation was successful.
  inline bool resetDescriptors(size_t cameraIdx, const cv::Mat & descriptors);

  /// \brief the number of keypoints
  /// @param[in] cameraIdx The camera index.
  /// \return The number of keypoints.
  inline size_t numKeypoints(size_t cameraIdx) const;
  /// @}

  /// \brief Get the total number of keypoints in all frames.
  /// \return The total number of keypoints.
  inline size_t numKeypoints() const;

  /// \brief Get the overlap mask. Sorry for the weird syntax, but remember that
  /// cv::Mat is essentially a shared pointer.
  /// @param[in] cameraIndexSeenBy The camera index for one camera.
  /// @param[in] cameraIndex The camera index for the other camera.
  /// @return The overlap mask image.
//  inline const cv::Mat overlap(size_t cameraIndexSeenBy,
//                               size_t cameraIndex) const
//  {
//    return cameraSystem_.overlap(cameraIndexSeenBy, cameraIndex);
//  }

  /// \brief Can the first camera see parts of the FOV of the second camera?
  /// @param[in] cameraIndexSeenBy The camera index for one camera.
  /// @param[in] cameraIndex The camera index for the other camera.
  /// @return True, if there is at least one pixel of overlap.
//  inline bool hasOverlap(size_t cameraIndexSeenBy, size_t cameraIndex) const
//  {
//    return cameraSystem_.hasOverlap(cameraIndexSeenBy, cameraIndex);
//  }

  inline cv::Mat drawStereoMatches(const std::vector<cv::DMatch>& matches) const;

  inline cv::Mat computeIntraMatches(
      cv::Ptr<cv::DescriptorMatcher> feature_matcher, std::vector<int>* i_query,
      std::vector<int>* i_match, double lowe_ratio, bool draw_matches) const;

  /// \brief copy descriptors at specified indices.
  inline cv::Mat copyDescriptorsAt(
      int cameraIdx, const std::vector<int>& descriptorIndices) const;

  /// \brief copy reduced keypoints at specified indices.
  inline std::vector<KeypointReduced, Eigen::aligned_allocator<KeypointReduced>>
  copyKeypointsAt(int cameraIdx, const std::vector<int>& keypointIndices) const;

  inline std::vector<KeypointReduced, Eigen::aligned_allocator<KeypointReduced>>
  copyKeypoints(int cameraIdx) const;


  cv::Mat getDescriptors(size_t cameraIdx) const {
    return frames_[cameraIdx].getDescriptors();
  }

  const std::vector<cv::KeyPoint>& getKeypoints(size_t cameraIdx) const {
    return frames_[cameraIdx].getKeypoints();
  }

  cv::Mat &descriptors(size_t cameraIdx) {
    return frames_[cameraIdx].descriptors();
  }

  std::vector<cv::KeyPoint> &keypoints(size_t cameraIdx) {
    return frames_[cameraIdx].keypoints();
  }

  bool isKeyframe() const {
    return isKeyframe_;
  }

  void setKeyframe(bool asKeyframe) {
    isKeyframe_ = asKeyframe;
  }

//  const cameras::CameraRig& cameraSystem() const {
//    return cameraSystem_;
//  }

//  okvis::cameras::DistortionType distortionType(size_t cameraIdx) const {
//    return cameraSystem_.distortionType(cameraIdx);
//  }

//  void setCameraSystem(const cameras::CameraRig& cameraSystem) {
//    cameraSystem_ = cameraSystem;
//  }

  void createTestImages(int rows, int cols, int type = CV_8UC1) {
    for (std::vector<Frame, Eigen::aligned_allocator<Frame>>::iterator it =
             frames_.begin();
         it != frames_.end(); ++it) {
      it->createTestImage(rows, cols, type);
    }
  }

 protected:
  okvis::Time timestamp_;  ///< the frame timestamp
  uint64_t id_;  ///< the frame id
  std::vector<Frame, Eigen::aligned_allocator<Frame>> frames_;  ///< the individual frames
//  cameras::CameraRig cameraSystem_;  ///< the camera system
  bool isKeyframe_;
};

struct BareMultiFrame {
  okvis::Time timestamp_;
  uint64_t id_;

  BareMultiFrame(okvis::Time time, uint64_t id, size_t numCams) : timestamp_(time), id_(id) {
    frames_.resize(numCams, BareFrame(okvis::Time(0)));
  }

  size_t numFrames() const {
    return frames_.size();
  }

  const std::vector<uint64_t> &landmarkIds(size_t camId) const {
    return frames_.at(camId).landmarkIds_;
  }

  std::vector<uint64_t> &landmarkIdsMutable (size_t camId) {
    return frames_.at(camId).landmarkIds_;
  }

  okvis::Time timestamp(size_t camId) const {
    return frames_.at(camId).stamp_;
  }

  void setTimestamp(size_t camId, okvis::Time time) {
    frames_.at(camId).stamp_ = time;
  }

  void setLandmarkId(size_t camId, size_t keypointIdx, uint64_t id) {
    frames_.at(camId).landmarkIds_.at(keypointIdx) = id;
  }

  uint64_t landmarkId(size_t camId, size_t keypointIdx) const {
    return frames_.at(camId).landmarkIds_.at(keypointIdx);
  }
private:
  std::vector<BareFrame> frames_;

};

std::shared_ptr<MultiFrame> adaptMultiFramePtr(std::shared_ptr<const okvis::MultiFrame> mf);

inline std::shared_ptr<const MultiFrame> adaptMultiFramePtr(std::shared_ptr<const MultiFrame> mf) {
  return mf;
}

enum class HistogramMethod {NONE, HISTOGRAM, CLAHE};

bool EnumFromString(std::string name, HistogramMethod *m);

std::string EnumToString(HistogramMethod m);

inline std::ostream &operator<<(std::ostream &s, HistogramMethod m) {
  return s << EnumToString(m);
}

cv::Mat enhanceImage(cv::Mat frame, bool useMedianFilter,
                     HistogramMethod histogramMethod, bool clone);

typedef std::shared_ptr<MultiFrame> MultiFramePtr;  ///< For convenience.
typedef std::shared_ptr<const MultiFrame> ConstMultiFramePtr;
typedef std::unordered_map<uint64_t, BareMultiFrame> BareMultiFrameMap;

}  // namespace swift_vio

#include "implementation/MultiFrame.hpp"

#endif /* INCLUDE_OKVIS_MULTIFRAME_HPP_ */
