
/**
 * @file FrameTypedefs.hpp
 * @brief This file contains useful typedefs and structs related to frames.
 * @author Jianzhu Huai
 */

#ifndef INCLUDE_SWIFT_VIO_FRAMETYPEDEFS_HPP_
#define INCLUDE_SWIFT_VIO_FRAMETYPEDEFS_HPP_

#include <cstddef>
#include <cstdint>

namespace swift_vio {
struct FeatureTrackStatus {
  enum MeasurementType {
    kPremature = 0,
    kMsckfTrack,
    kSlamObservation,
    kSlamInitialization
  };

  enum MeasurementFate {
    kUndetermined = 0,
    kSuccessful,
    kComputingJacobiansFailed,
    kPotentialOutlier,
    kLeftOver,
  };

  FeatureTrackStatus()
      : inState(false), numMissFrames(0u), measurementType(kPremature),
        measurementFate(kUndetermined) {}

  void updateTrackStat(bool observedInCurrentFrame) {
    if (observedInCurrentFrame) {
      numMissFrames = 0u;
    } else {
      ++numMissFrames;
    }
  }

  bool inState;
  size_t numMissFrames; // number of successive miss frames since its last
                     // observing frame. For instance, if it is not observed in
                     // the last and current frame, it would be 2.

  MeasurementType measurementType;
  MeasurementFate measurementFate;
};

struct AnchorFrameIdentifier {
  uint64_t frameId_;
  size_t cameraIndex_; // which camera?
  size_t observationIndex_; // index in the observation sequence.
  AnchorFrameIdentifier(uint64_t frameId, size_t cameraIndex,
                        size_t observationIndex)
      : frameId_(frameId),
        cameraIndex_(cameraIndex),
        observationIndex_(observationIndex) {}
};

static const size_t kMainCameraIndex = 0u;
}  // namespace swift_vio

#endif /* INCLUDE_SWIFT_VIO_FRAMETYPEDEFS_HPP_ */
