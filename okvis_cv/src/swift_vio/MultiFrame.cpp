#include <swift_vio/MultiFrame.hpp>
#include <opencv2/imgproc.hpp>


namespace swift_vio {
std::shared_ptr<MultiFrame>
adaptMultiFramePtr(std::shared_ptr<const okvis::MultiFrame> mf) {
  std::shared_ptr<MultiFrame> mfptr(
      new MultiFrame(mf->numFrames(), mf->timestamp(), mf->id()));
  for (size_t i = 0; i < mf->numFrames(); ++i) {
    mfptr->setTimestamp(i, mf->timestamp(i));
    mfptr->resetKeypoints(i, mf->getKeypoints(i));
    mfptr->resetDescriptors(i, mf->getDescriptors(i));  // shallow copy
    mfptr->setImage(i, mf->image(i));  // shallow copy
  }
  return mfptr;
}

std::string EnumToString(HistogramMethod m) {
  const std::string names[] = {"NONE", "HISTOGRAM", "CLAHE"};
  return names[static_cast<int>(m)];
}

bool EnumFromString(std::string name, HistogramMethod *m) {
  std::transform(name.begin(), name.end(), name.begin(), ::toupper);
  std::unordered_map<std::string, HistogramMethod> descriptionToId{
      {"NONE", HistogramMethod::NONE},
      {"HISTOGRAM", HistogramMethod::HISTOGRAM},
      {"CLAHE", HistogramMethod::CLAHE}};

  auto iter = descriptionToId.find(name);
  if (iter == descriptionToId.end()) {
    *m = HistogramMethod::NONE;
    return false;
  } else {
    *m = iter->second;
  }
  return true;
}

cv::Mat enhanceImage(cv::Mat frame, bool useMedianFilter,
                     HistogramMethod histogramMethod, bool clone) {
  cv::Mat filtered;
  if (useMedianFilter) {
    cv::medianBlur(frame, filtered, 3);
  } else {
    filtered = clone ? frame.clone() : frame;
  }

  // Histogram equalize
  cv::Mat img;
  if (histogramMethod == HistogramMethod::HISTOGRAM) {
    cv::equalizeHist(filtered, img);
  } else if (histogramMethod == HistogramMethod::CLAHE) {
    double eq_clip_limit = 10.0;
    cv::Size eq_win_size = cv::Size(8, 8);
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(eq_clip_limit, eq_win_size);
    clahe->apply(filtered, img);
  } else {
    img = filtered;
  }
  return img;
}

} // namespace swift_vio
