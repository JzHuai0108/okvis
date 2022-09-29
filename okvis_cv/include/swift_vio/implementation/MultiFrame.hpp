
/**
 * @file implementation/MultiFrame.hpp
 * @brief Header implementation file for the MultiFrame class.
 */

namespace swift_vio {

// Default constructor
MultiFrame::MultiFrame() : id_(0), isKeyframe_(false) {}

// Construct from NCameraSystem
MultiFrame::MultiFrame(int numCameras, const okvis::Time & timestamp, uint64_t id)
    : timestamp_(timestamp),
      id_(id), isKeyframe_(false)
{
  resetFrames(numCameras);
}

MultiFrame::~MultiFrame()
{

}

// (Re)set the NCameraSystem -- which clears the frames as well.
void MultiFrame::resetFrames(
    int numCameras)
{
  frames_.clear();  // erase -- for safety
  frames_.resize(numCameras);
}

// (Re)set the timestamp
void MultiFrame::setTimestamp(const okvis::Time & timestamp)
{
  timestamp_ = timestamp;
}

void MultiFrame::setTimestamp(size_t cameraIdx, const okvis::Time& timestamp)
{
  frames_[cameraIdx].setTimestamp(timestamp);
}

// (Re)set the id
void MultiFrame::setId(uint64_t id)
{
  id_ = id;
}

// Obtain the frame timestamp
const okvis::Time & MultiFrame::timestamp() const
{
  return timestamp_;
}

okvis::Time MultiFrame::timestamp(size_t cameraIdx) const
{
  return frames_[cameraIdx].timestamp();
}

// Obtain the frame id
uint64_t MultiFrame::id() const
{
  return id_;
}

// The number of frames/cameras
size_t MultiFrame::numFrames() const
{
  return frames_.size();
}

//std::shared_ptr<const okvis::kinematics::Transformation> MultiFrame::T_SC(size_t cameraIdx) const {
//  return cameraSystem_.getCameraExtrinsicPtr(cameraIdx);
//}


//////////////////////////////////////////////////////////////
// The following mirror the Frame functionality.
//

// Set the frame image;
void MultiFrame::setImage(size_t cameraIdx, const cv::Mat & image)
{
  OKVIS_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range");
  frames_[cameraIdx].setImage(image);
}

// Set the geometry
//void MultiFrame::setGeometry(
//    size_t cameraIdx, std::shared_ptr<const okvis::cameras::CameraBase> cameraGeometry)
//{
//  OKVIS_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range");
//  frames_[cameraIdx].setGeometry(cameraGeometry);
//}

// Set the detector
//void MultiFrame::setDetector(size_t cameraIdx,
//                             std::shared_ptr<cv::FeatureDetector> detector)
//{
//  OKVIS_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range");
//  frames_[cameraIdx].setDetector(detector);
//}

// Set the extractor
//void MultiFrame::setExtractor(
//    size_t cameraIdx, std::shared_ptr<cv::DescriptorExtractor> extractor)
//{
//  OKVIS_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range");
//  frames_[cameraIdx].setExtractor(extractor);
//}

// Obtain the image
const cv::Mat & MultiFrame::image(size_t cameraIdx) const
{
  OKVIS_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range");
  return frames_[cameraIdx].image();
}

// get the base class geometry (will be slow to use)
//std::shared_ptr<const okvis::cameras::CameraBase> MultiFrame::geometry(
//    size_t cameraIdx) const
//{
//  OKVIS_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range");
//  return frames_[cameraIdx].geometry();
//}

// Get the specific geometry (will be fast to use)
//template<class GEOMETRY_T>
//std::shared_ptr<const GEOMETRY_T> MultiFrame::geometryAs(size_t cameraIdx) const
//{
//  OKVIS_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range");
//  return frames_[cameraIdx].geometryAs<GEOMETRY_T>();
//}

// Detect keypoints. This uses virtual function calls.
///        That's a negligibly small overhead for many detections.
///        returns the number of detected points.
//int MultiFrame::detect(size_t cameraIdx)
//{
//  OKVIS_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range");
//  return frames_[cameraIdx].detect();
//}

// Describe keypoints. This uses virtual function calls.
///        That's a negligibly small overhead for many detections.
///        \param extractionDirection the extraction direction in camera frame
///        returns the number of detected points.
//int MultiFrame::describe(size_t cameraIdx,
//                         const Eigen::Vector3d & extractionDirection)
//{
//  OKVIS_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range");
//  return frames_[cameraIdx].describe(extractionDirection);
//}
//template<class GEOMETRY_T>
//int MultiFrame::describeAs(size_t cameraIdx,
//                           const Eigen::Vector3d & extractionDirection)
//{
//  OKVIS_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range");
//  return frames_[cameraIdx].template describeAs <GEOMETRY_T> (extractionDirection);
//}

// Access a specific keypoint in OpenCV format
bool MultiFrame::getCvKeypoint(size_t cameraIdx, size_t keypointIdx,
                               cv::KeyPoint & keypoint) const
{
  OKVIS_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range");
  return frames_[cameraIdx].getCvKeypoint(keypointIdx, keypoint);
}

// Get a specific keypoint
bool MultiFrame::getKeypoint(size_t cameraIdx, size_t keypointIdx,
                             Eigen::Vector2d & keypoint) const
{
  OKVIS_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range");
  return frames_[cameraIdx].getKeypoint(keypointIdx, keypoint);
}

// Get the size of a specific keypoint
bool MultiFrame::getKeypointSize(size_t cameraIdx, size_t keypointIdx,
                                 double & keypointSize) const
{
  OKVIS_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range");
  return frames_[cameraIdx].getKeypointSize(keypointIdx, keypointSize);
}

bool MultiFrame::getKeypointSize(size_t cameraIdx, size_t keypointIdx,
                                 float &keypointSize) const
{
  OKVIS_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range");
  return frames_[cameraIdx].getKeypointSize(keypointIdx, keypointSize);
}

// Access the descriptor -- CAUTION: high-speed version.
///        returns NULL if out of bounds.
const unsigned char * MultiFrame::keypointDescriptor(size_t cameraIdx,
                                                     size_t keypointIdx) const
{
  OKVIS_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range");
  return frames_[cameraIdx].keypointDescriptor(keypointIdx);
}

// number of keypoints
size_t MultiFrame::numKeypoints(size_t cameraIdx) const
{
  OKVIS_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range");
  return frames_[cameraIdx].numKeypoints();
}

// provide keypoints externally
bool MultiFrame::resetKeypoints(size_t cameraIdx, const std::vector<cv::KeyPoint> & keypoints){
  OKVIS_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range");
  return frames_[cameraIdx].resetKeypoints(keypoints);
}

// provide descriptors externally
bool MultiFrame::resetDescriptors(size_t cameraIdx, const cv::Mat & descriptors) {
  OKVIS_ASSERT_TRUE_DBG(Exception, cameraIdx < frames_.size(), "Out of range");
  return frames_[cameraIdx].resetDescriptors(descriptors);
}

//

// get the total number of keypoints in all frames.
size_t MultiFrame::numKeypoints() const
{
  size_t numKeypoints = 0;
  for (size_t i = 0; i < frames_.size(); ++i) {
    numKeypoints += frames_[i].numKeypoints();
  }
  return numKeypoints;
}

cv::Mat MultiFrame::computeIntraMatches(
    cv::Ptr<cv::DescriptorMatcher> feature_matcher, std::vector<int>* i_query,
    std::vector<int>* i_match, double lowe_ratio, bool draw_matches) const {
  // Get two best matches between frame descriptors.
  std::vector<std::vector<cv::DMatch>> matches;
  feature_matcher->knnMatch(frames_[0].getDescriptors(),
                            frames_[1].getDescriptors(), matches, 2u);

  const size_t n_matches = matches.size();
  i_query->reserve(n_matches);
  i_match->reserve(n_matches);
  std::vector<cv::DMatch> good_matches;
  for (size_t i = 0; i < n_matches; i++) {
    const std::vector<cv::DMatch>& match = matches[i];
    if (match[0].distance < lowe_ratio * match[1].distance) {
      i_query->push_back(match[0].queryIdx);
      i_match->push_back(match[0].trainIdx);
      good_matches.push_back(match[0]);
    }
  }

  cv::Mat img_matches;
  if (draw_matches) {
    img_matches = drawStereoMatches(good_matches);
  }
  return img_matches;
}

cv::Mat MultiFrame::drawStereoMatches(const std::vector<cv::DMatch>& matches) const {
  cv::Mat img_matches;
  cv::drawMatches(frames_[0].image(), frames_[0].getKeypoints(), frames_[1].image(),
                  frames_[1].getKeypoints(), matches, img_matches,
                  cv::Scalar(255, 0, 0), cv::Scalar(255, 0, 0));
  return img_matches;
}

cv::Mat MultiFrame::copyDescriptorsAt(
    int cameraIdx, const std::vector<int>& descriptorIndices) const {
  return frames_[cameraIdx].copyDescriptorsAt(descriptorIndices);
}

inline std::vector<KeypointReduced, Eigen::aligned_allocator<KeypointReduced>>
MultiFrame::copyKeypointsAt(int cameraIdx,
                            const std::vector<int>& keypointIndices) const {
  return frames_[cameraIdx].copyKeypointsAt(keypointIndices);
}

inline std::vector<KeypointReduced, Eigen::aligned_allocator<KeypointReduced>>
MultiFrame::copyKeypoints(int cameraIdx) const {
  return frames_[cameraIdx].copyKeypoints();
}

}// namespace swift_vio
