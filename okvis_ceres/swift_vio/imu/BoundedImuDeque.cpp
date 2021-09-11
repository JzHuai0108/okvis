#include "swift_vio/imu/BoundedImuDeque.hpp"
#include <glog/logging.h>

namespace swift_vio {

bool cmp(okvis::ImuMeasurement lhs, okvis::ImuMeasurement rhs) {
  return lhs.timeStamp < rhs.timeStamp;
}

BoundedImuDeque::BoundedImuDeque() {}

BoundedImuDeque::~BoundedImuDeque() {}

int BoundedImuDeque::push_back(const okvis::ImuMeasurementDeque& imu_segment) {
  // find the insertion point
  auto iter = std::lower_bound(imu_meas_.begin(), imu_meas_.end(),
                               imu_segment.front(), cmp);
  if (iter == imu_meas_.end()) {
    imu_meas_.insert(iter, imu_segment.begin(), imu_segment.end());
    return imu_segment.size();
  } else {
    CHECK_EQ(iter->timeStamp, imu_segment.front().timeStamp);
    if (imu_meas_.back().timeStamp < imu_segment.back().timeStamp) {
      size_t erased = imu_meas_.end() - iter;
      imu_meas_.erase(iter, imu_meas_.end());
      imu_meas_.insert(imu_meas_.end(), imu_segment.begin(), imu_segment.end());
      return (int)(imu_segment.size() - erased);
    } else {
      return 0;
    }
  }
}

int BoundedImuDeque::pop_front(const okvis::Time& eraseUntil) {
  return deleteImuMeasurements(eraseUntil, this->imu_meas_, nullptr);
}

const okvis::ImuMeasurementDeque BoundedImuDeque::find(
    const okvis::Time& begin_time, const okvis::Time& end_time) const {
  return getImuMeasurements(begin_time, end_time, this->imu_meas_, nullptr);
}

const okvis::ImuMeasurementDeque BoundedImuDeque::findWindow(
    const okvis::Time& center_time, const okvis::Duration& half_window) const {
  okvis::ImuMeasurementDeque raw_meas =
          getImuMeasurements(center_time - half_window, center_time + half_window,
                            this->imu_meas_, nullptr);
  if (raw_meas.size()) {
    if (raw_meas.front().timeStamp > center_time - half_window) {
      // Pad at the left. This can happen at the begnning of data streams.
      raw_meas.push_front(raw_meas.front());
      raw_meas.front().timeStamp = center_time - half_window;
    }
    if (raw_meas.back().timeStamp < center_time + half_window) {
      // Pad at the right. This always happen for the latest NFrame when half_window is big say >0.2 s.
      raw_meas.push_back(raw_meas.back());
      raw_meas.back().timeStamp = center_time + half_window;
    }
  }
  return raw_meas;
}

const okvis::ImuMeasurementDeque& BoundedImuDeque::getAllImuMeasurements()
    const {
  return imu_meas_;
}

okvis::ImuMeasurementDeque getImuMeasurements(
    const okvis::Time& imuDataBeginTime, const okvis::Time& imuDataEndTime,
    const okvis::ImuMeasurementDeque& imuMeasurements_,
    std::mutex* imuMeasurements_mutex_) {
  // sanity checks:
  // if end time is smaller than begin time, return empty queue.
  // if begin time is larger than newest imu time, return empty queue.
  if (imuDataEndTime < imuDataBeginTime ||
      imuDataBeginTime > imuMeasurements_.back().timeStamp)
    return okvis::ImuMeasurementDeque();

  std::unique_lock<std::mutex> lock =
      imuMeasurements_mutex_ == nullptr
          ? std::unique_lock<std::mutex>()
          : std::unique_lock<std::mutex>(*imuMeasurements_mutex_);

  auto first_imu_package = std::lower_bound(
      imuMeasurements_.begin(), imuMeasurements_.end(),
      okvis::ImuMeasurement(imuDataBeginTime, okvis::ImuSensorReadings()), cmp);
  if (first_imu_package != imuMeasurements_.begin() &&
      first_imu_package->timeStamp > imuDataBeginTime) {
    --first_imu_package;
  }
  auto last_imu_package = std::lower_bound(
      imuMeasurements_.begin(), imuMeasurements_.end(),
      okvis::ImuMeasurement(imuDataEndTime, okvis::ImuSensorReadings()), cmp);
  if (last_imu_package != imuMeasurements_.end()) {
    ++last_imu_package;
  }

  // get iterator to imu data before previous frame
  //  okvis::ImuMeasurementDeque::const_iterator first_imu_package =
  //      imuMeasurements_.begin();
  //  okvis::ImuMeasurementDeque::const_iterator last_imu_package =
  //      imuMeasurements_.end();
  //  // TODO go backwards through queue. Is probably faster.
  //  for (auto iter = imuMeasurements_.begin(); iter != imuMeasurements_.end();
  //       ++iter) {
  //    // move first_imu_package iterator back until iter->timeStamp is higher
  //    than
  //    // requested begintime
  //    if (iter->timeStamp <= imuDataBeginTime) first_imu_package = iter;

  //    // set last_imu_package iterator as soon as we hit first timeStamp
  //    higher
  //    // than requested endtime & break
  //    if (iter->timeStamp >= imuDataEndTime) {
  //      last_imu_package = iter;
  //      // since we want to include this last imu measurement in returned
  //      Deque we
  //      // increase last_imu_package iterator once.
  //      ++last_imu_package;
  //      break;
  //    }
  //  }

  // create copy of imu buffer
  return okvis::ImuMeasurementDeque(first_imu_package, last_imu_package);
}

// Remove IMU measurements from the internal buffer.
int deleteImuMeasurements(const okvis::Time& eraseUntil,
                          okvis::ImuMeasurementDeque& imuMeasurements_,
                          std::mutex* imuMeasurements_mutex_) {
  std::unique_lock<std::mutex> lock =
      imuMeasurements_mutex_ == nullptr
          ? std::unique_lock<std::mutex>()
          : std::unique_lock<std::mutex>(*imuMeasurements_mutex_);
  if (imuMeasurements_.front().timeStamp > eraseUntil) return 0;

  auto eraseEnd =
      std::lower_bound(imuMeasurements_.begin(), imuMeasurements_.end(),
                       okvis::ImuMeasurement(eraseUntil, okvis::ImuSensorReadings()), cmp);
  int removed = eraseEnd - imuMeasurements_.begin();

  //  okvis::ImuMeasurementDeque::iterator eraseEnd;
  //  int removed = 0;
  //  for (auto it = imuMeasurements_.begin(); it != imuMeasurements_.end();
  //  ++it) {
  //    eraseEnd = it;
  //    if (it->timeStamp >= eraseUntil) break;
  //    ++removed;
  //  }

  imuMeasurements_.erase(imuMeasurements_.begin(), eraseEnd);

  return removed;
}

void padImuToLeft(okvis::Time leftTimeLimit, okvis::Duration delta,
                  okvis::ImuMeasurementDeque *imuMeasurements) {
  while (imuMeasurements->front().timeStamp > leftTimeLimit) {
    okvis::ImuMeasurement meas = imuMeasurements->front();
    meas.timeStamp -= delta;
    imuMeasurements->push_front(meas);
  }
}

void padImuToRight(okvis::Time rightTimeLimit, okvis::Duration delta,
                   okvis::ImuMeasurementDeque *imuMeasurements) {
  while (imuMeasurements->back().timeStamp < rightTimeLimit) {
    okvis::ImuMeasurement meas = imuMeasurements->back();
    meas.timeStamp += delta;
    imuMeasurements->push_back(meas);
  }
}

}  // namespace swift_vio
