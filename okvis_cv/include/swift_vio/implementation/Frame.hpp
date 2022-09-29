
/**
 * @file implementation/Frame.hpp
 * @brief Header implementation file for the Frame class.
 */

namespace swift_vio {

// a constructor that uses the specified geometry,
/// detector and extractor
Frame::Frame(const cv::Mat & image)
    : image_(image) {
}

// set the frame image;
void Frame::setImage(const cv::Mat & image)
{
  image_ = image;
}

// set the detector
//void Frame::setDetector(std::shared_ptr<cv::FeatureDetector> detector)
//{
//  detector_ = detector;
//}

// set the extractor
//void Frame::setExtractor(std::shared_ptr<cv::DescriptorExtractor> extractor)
//{
//  extractor_ = extractor;
//}

// obtain the image
const cv::Mat & Frame::image() const
{
  return image_;
}

// detect keypoints. This uses virtual function calls.
///        That's a negligibly small overhead for many detections.
///        returns the number of detected points.
//int Frame::detect()
//{
//  // make sure things are set to zero for safety
//  keypoints_.clear();
//  descriptors_.resize(0);

//  // run the detector
//  OKVIS_ASSERT_TRUE_DBG(Exception, detector_ != NULL,
//                        "Detector not initialised!");
//  detector_->detect(image_, keypoints_);
//  return keypoints_.size();
//}

// describe keypoints. This uses virtual function calls.
///        That's a negligibly small overhead for many detections.
///        \param extractionDirection the extraction direction in camera frame
///        returns the number of detected points.
//int Frame::describe(const Eigen::Vector3d & extractionDirection)
//{
//  // check initialisation
//  OKVIS_ASSERT_TRUE_DBG(Exception, extractor_ != NULL,
//                        "Detector not initialised!");

//  // orient the keypoints according to the extraction direction:
//  Eigen::Vector3d ep;
//  Eigen::Vector2d reprojection;
//  Eigen::Matrix<double, 2, 3> Jacobian;
//  Eigen::Vector2d eg_projected;
//  for (size_t k = 0; k < keypoints_.size(); ++k) {
//    cv::KeyPoint& ckp = keypoints_[k];
//    // project ray
//    cameraGeometry_->backProject(Eigen::Vector2d(ckp.pt.x, ckp.pt.y), &ep);
//    // obtain image Jacobian
//    cameraGeometry_->project(ep, &reprojection, &Jacobian);
//    // multiply with gravity direction
//    eg_projected = Jacobian * extractionDirection;
//    double angle = atan2(eg_projected[1], eg_projected[0]);
//    // set
//    ckp.angle = angle / M_PI * 180.0;
//  }

//  // extraction
//  extractor_->compute(image_, keypoints_, descriptors_);
//  return keypoints_.size();
//}
// describe keypoints. This uses virtual function calls.
///        That's a negligibly small overhead for many detections.
///        \param extractionDirection the extraction direction in camera frame
///        returns the number of detected points.
//template<class GEOMETRY_T>
//int Frame::describeAs(const Eigen::Vector3d & extractionDirection)
//{
//  // check initialisation
//  OKVIS_ASSERT_TRUE_DBG(Exception, extractor_ != NULL,
//                        "Detector not initialised!");

//  // orient the keypoints according to the extraction direction:
//  Eigen::Vector3d ep;
//  Eigen::Vector2d reprojection;
//  Eigen::Matrix<double, 2, 3> Jacobian;
//  Eigen::Vector2d eg_projected;
//  for (size_t k = 0; k < keypoints_.size(); ++k) {
//    cv::KeyPoint& ckp = keypoints_[k];
//    // project ray
//    geometryAs<GEOMETRY_T>()->backProject(Eigen::Vector2d(ckp.pt.x, ckp.pt.y),
//                                          &ep);
//    // obtain image Jacobian
//    geometryAs<GEOMETRY_T>()->project(ep, &reprojection, &Jacobian);
//    // multiply with gravity direction
//    eg_projected = Jacobian * extractionDirection;
//    double angle = atan2(eg_projected[1], eg_projected[0]);
//    // set
//    ckp.angle = angle / M_PI * 180.0;
//  }

//  // extraction
//  extractor_->compute(image_, keypoints_, descriptors_);
//  return keypoints_.size();
//}

// access a specific keypoint in OpenCV format
bool Frame::getCvKeypoint(size_t keypointIdx, cv::KeyPoint & keypoint) const
{
#ifndef NDEBUG
  OKVIS_ASSERT_TRUE(
      Exception,
      keypointIdx < keypoints_.size(),
      "keypointIdx " << keypointIdx << "out of range: keypoints has size "
          << keypoints_.size());
  keypoint = keypoints_[keypointIdx];
  return keypointIdx < keypoints_.size();
#else
  keypoint = keypoints_[keypointIdx];
  return true;
#endif
}

// get a specific keypoint
bool Frame::getKeypoint(size_t keypointIdx, Eigen::Vector2d & keypoint) const
{
#ifndef NDEBUG
  OKVIS_ASSERT_TRUE(
      Exception,
      keypointIdx < keypoints_.size(),
      "keypointIdx " << keypointIdx << "out of range: keypoints has size "
          << keypoints_.size());
  keypoint = Eigen::Vector2d(keypoints_[keypointIdx].pt.x,
                             keypoints_[keypointIdx].pt.y);
  return keypointIdx < keypoints_.size();
#else
  keypoint = Eigen::Vector2d(keypoints_[keypointIdx].pt.x, keypoints_[keypointIdx].pt.y);
  return true;
#endif
}

// get the size of a specific keypoint
bool Frame::getKeypointSize(size_t keypointIdx, double & keypointSize) const
{
#ifndef NDEBUG
  OKVIS_ASSERT_TRUE(
      Exception,
      keypointIdx < keypoints_.size(),
      "keypointIdx " << keypointIdx << "out of range: keypoints has size "
          << keypoints_.size());
  keypointSize = keypoints_[keypointIdx].size;
  return keypointIdx < keypoints_.size();
#else
  keypointSize = keypoints_[keypointIdx].size;
  return true;
#endif
}

bool Frame::getKeypointSize(size_t keypointIdx, float& keypointSize) const
{
#ifndef NDEBUG
  OKVIS_ASSERT_TRUE(
      Exception,
      keypointIdx < keypoints_.size(),
      "keypointIdx " << keypointIdx << "out of range: keypoints has size "
          << keypoints_.size());
  keypointSize = keypoints_[keypointIdx].size;
  return keypointIdx < keypoints_.size();
#else
  keypointSize = keypoints_[keypointIdx].size;
  return true;
#endif
}

// access the descriptor -- CAUTION: high-speed version.
///        returns NULL if out of bounds.
const unsigned char * Frame::keypointDescriptor(size_t keypointIdx) const
{
#ifndef NDEBUG
  OKVIS_ASSERT_TRUE(
      Exception,
      keypointIdx < keypoints_.size(),
      "keypointIdx " << keypointIdx << "out of range: keypoints has size "
          << keypoints_.size());
  return descriptors_.data + descriptors_.cols * keypointIdx;
#else
  return descriptors_.data + descriptors_.cols * keypointIdx;
#endif
}

// provide keypoints externally
inline bool Frame::resetKeypoints(const std::vector<cv::KeyPoint> & keypoints) {
  keypoints_ = keypoints;
  return true;
}

// provide descriptors externally
inline bool Frame::resetDescriptors(const cv::Mat & descriptors) {
  descriptors_ = descriptors;
  return true;
}

size_t Frame::numKeypoints() const {
  return keypoints_.size();
}

cv::Mat selectDescriptors(
    const cv::Mat descriptors,
    const std::vector<int>& descriptorIndices) {
  cv::Mat result(descriptorIndices.size(), descriptors.cols,
                 descriptors.type());
  int j = 0;
  for (auto index : descriptorIndices) {
    descriptors.row(index).copyTo(result.row(j));
    ++j;
  }
  return result;
}

cv::Mat Frame::copyDescriptorsAt(
    const std::vector<int>& descriptorIndices) const {
  return selectDescriptors(descriptors_, descriptorIndices);
}

std::vector<KeypointReduced, Eigen::aligned_allocator<KeypointReduced>>
Frame::copyKeypointsAt(const std::vector<int>& keypointIndices) const {
  std::vector<KeypointReduced, Eigen::aligned_allocator<KeypointReduced>>
      result(keypointIndices.size());
  int j = 0;
  for (auto index : keypointIndices) {
    result[j][0] = keypoints_[index].pt.x;
    result[j][1] = keypoints_[index].pt.y;
    result[j][2] = keypoints_[index].size;
    ++j;
  }
  return result;
}

std::vector<KeypointReduced, Eigen::aligned_allocator<KeypointReduced>>
Frame::copyKeypoints() const {
  std::vector<KeypointReduced, Eigen::aligned_allocator<KeypointReduced>>
      result(keypoints_.size());
  int j = 0;
  for (const cv::KeyPoint& keypoint : keypoints_) {
    result[j][0] = keypoint.pt.x;
    result[j][1] = keypoint.pt.y;
    result[j][2] = keypoint.size;
    ++j;
  }
  return result;
}

}  // namespace swift_vio
