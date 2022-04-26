#ifndef MAPPOINT_H
#define MAPPOINT_H

#include <Eigen/Core>
#include <okvis/FrameTypedefs.hpp>
#include <swift_vio/CameraIdentifier.h>
#include <swift_vio/memory.h>

namespace swift_vio {
struct FluidObservation {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Eigen::Vector2f uv;
  float size;
  Eigen::Vector2f velocity_uv;
  FluidObservation(const Eigen::Vector2f &_uv, float _size,
                   const Eigen::Vector2f &vel)
      : uv(_uv), size(_size), velocity_uv(vel) {}

  FluidObservation(const Eigen::Vector2d &_uv, float _size,
                   const Eigen::Vector2d &vel)
      : uv(_uv.cast<float>()), size(_size), velocity_uv(vel.cast<float>()) {}
};

struct KeypointObservation {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Eigen::Vector2f uv;
    float size;
    uint64_t residualId;
    KeypointObservation(const Eigen::Vector2f &_uv, float _size, uint64_t _residualId = 0) :
        uv(_uv), size(_size), residualId(_residualId) {
    }

    KeypointObservation(const FluidObservation & fluidObs) :
      uv(fluidObs.uv), size(fluidObs.size), residualId(0) {}
};

struct IsObservedInNFrame {
  IsObservedInNFrame(uint64_t x) : frameId(x) {}
  bool operator()(
      const std::pair<okvis::KeypointIdentifier, KeypointObservation> &v) const {
    return v.first.frameId == frameId;
  }

 private:
  uint64_t frameId;  ///< Multiframe ID.
};

struct IsObservedInFrame {
  IsObservedInFrame(uint64_t _frameId, size_t _camIdx)
      : frameId(_frameId), cameraIndex(_camIdx) {}
  bool
  operator()(const std::pair<okvis::KeypointIdentifier, KeypointObservation> &v) const {
    return v.first.frameId == frameId && v.first.cameraIndex == cameraIndex;
  }

private:
  uint64_t frameId; ///< Multiframe ID.
  size_t cameraIndex;
};

/**
 * @brief A type to store information about a point in the world map.
 */
struct MapPoint
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /**
   * @brief Constructor.
   * @param id        ID of the point. E.g. landmark ID.
   * @param point     Homogeneous coordinate of the point.
   * @param quality   Quality of the point. Usually between 0 and 1.
   * @param distance  Distance to origin of the frame the coordinates are given in.
   */
  MapPoint(uint64_t id = 0, const Eigen::Vector4d & point = Eigen::Vector4d::Zero(),
           double quality = 0.0, double distance = 0.0, uint64_t anchorState = 0u,
           uint64_t anchorCamera = 0u, bool inited = false)
      : id(id),
        pointHomog(point),
        quality(quality),
        distance(distance),
        anchorStateId(anchorState),
        anchorCameraId(anchorCamera),
        initialized(inited), timesObserved(0u)
  {
  }

  explicit MapPoint(const okvis::MapPoint &mp)
      : id(mp.id), pointHomog(mp.pointHomog), quality(mp.quality),
        distance(mp.distance), anchorStateId(mp.anchorStateId),
        anchorCameraId(mp.anchorCameraId), status(mp.getStatus()),
        initialized(mp.isInitialized()) {
    for (const auto &p : mp.observations) {
      observations.emplace(
          p.first, KeypointObservation(Eigen::Vector2f::Zero(), -1, p.second));
    }
    timesObserved = observations.size();
  }

  bool trackedInCurrentFrame(uint64_t currentFrameId) const {
    return observations.rbegin()->first.frameId == currentFrameId;
  }

  CameraIdentifier anchorCamera() const {
    return CameraIdentifier(anchorStateId, anchorCameraId);
  }

  bool shouldRemove(size_t maxHibernationFrames) const {
    bool toRemove(false);
    if (status.measurementType == swift_vio::FeatureTrackStatus::kMsckfTrack &&
        status.measurementFate == swift_vio::FeatureTrackStatus::kSuccessful) {
      toRemove = true;
    }
    if (status.numMissFrames >= maxHibernationFrames) {
      toRemove = true;
    }
    return toRemove;
  }

  /**
   * @brief shouldChangeAnchor should change the anchor frame of a landmark?
   * @param sortedToRemoveStateIds
   * @return 0 if the landmark is not parameterized with an anchor frame or the
   * anchor frame is NOT one of the state variables to remove, otherwise, the
   * candidate anchor frame id is returned.
   */
  uint64_t
  shouldChangeAnchor(const std::vector<uint64_t> &sortedToRemoveStateIds) const;

  /**
   * @brief goodForMarginalization Is this map point good to be used in the frame marginalization step?
   * This checks if this map point's observations have been used earlier as measurements.
   * @param minCulledFrames minimum frames to be culled in one frame marginalization step.
   * @return true if the map point is good for marginalization.
   */
  bool goodForMarginalization(size_t minCulledFrames) const;

  void updateStatus(uint64_t currentFrameId, size_t minTrackLengthForMsckf,
                    size_t minTrackLengthForSlam);

  void setMeasurementType(swift_vio::FeatureTrackStatus::MeasurementType measurementType) {
    status.measurementType = measurementType;
  }

  void setMeasurementFate(swift_vio::FeatureTrackStatus::MeasurementFate measurementFate) {
    status.measurementFate = measurementFate;
  }

  void setInState(bool instate) {
    status.inState = instate;
    if (instate)
      quality = 1.0;
  }

  void setInitialized(bool inited) {
    initialized = inited;
  }

  swift_vio::FeatureTrackStatus::MeasurementType measurementType() const {
    return status.measurementType;
  }

  swift_vio::FeatureTrackStatus::MeasurementFate measurementFate() const {
    return status.measurementFate;
  }

  bool inState() const {
    return status.inState;
  }

  bool isInitialized() const {
    return initialized;
  }

  size_t numTimesObserved() const {
    return timesObserved;
  }

  bool hasObservationInImage(uint64_t frameId, size_t cameraId) const {
    auto iter = observations.lower_bound(okvis::KeypointIdentifier(frameId, cameraId, 0u));
    if (iter != observations.end() && iter->first.frameId == frameId && iter->first.cameraIndex == cameraId) {
      return true;
    }
    return false;
  }

  void addObservations(const Eigen::AlignedMap<okvis::KeypointIdentifier, FluidObservation> &newObservations);

  inline void addObservation(const okvis::KeypointIdentifier &kid, const KeypointObservation &kp) {
    observations.emplace(kid, kp);
    ++timesObserved;
  }

  uint64_t id;            ///< ID of the point. E.g. landmark ID.
  Eigen::Vector4d pointHomog;  ///< Homogeneous coordinate of the point in the World frame.

  double quality;         ///< Quality of the point. Usually between 0 and 1.
  double distance;        ///< Distance to origin of the frame the coordinates are given in.
  Eigen::AlignedMap<okvis::KeypointIdentifier, KeypointObservation> observations;   ///< Observations of this point.

  uint64_t anchorStateId;
  size_t anchorCameraId;

private:
  mutable FeatureTrackStatus status;
  mutable bool initialized; // is this landmark initialized in position?
  size_t timesObserved;  // how many times this map point has been observed?
};

struct BareMapPoint {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  uint64_t id;
  Eigen::Vector4d pointHomog;
  double quality;

  explicit BareMapPoint(const MapPoint &mp)
      : id(mp.id), pointHomog(mp.pointHomog), quality(mp.quality) {}

  explicit BareMapPoint(const okvis::MapPoint &mp)
      : id(mp.id), pointHomog(mp.pointHomog), quality(mp.quality) {}

  BareMapPoint(uint64_t _id, const Eigen::Vector4d & _point, double _quality = 0) :
    id(_id), pointHomog(_point), quality(_quality) {
  }
};

struct FeatureTrack {
  uint64_t id;
  Eigen::AlignedMap<okvis::KeypointIdentifier, FluidObservation> observations;

  FeatureTrack(uint64_t _id) : id(_id) {}
};

struct PointAndVariance {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Eigen::Vector4d point;
  Eigen::Vector3d variance;

  PointAndVariance() {}
};

typedef Eigen::AlignedUnorderedMap<uint64_t, FeatureTrack> FeatureTrackMap;

typedef Eigen::AlignedVector<BareMapPoint> MapPointVector;

typedef Eigen::AlignedMap<uint64_t, MapPoint> PointMap;

typedef Eigen::AlignedMap<okvis::KeypointIdentifier, KeypointObservation> ObservationMap;

typedef Eigen::AlignedMap<okvis::KeypointIdentifier, FluidObservation> FluidObservationMap;

}  // namespace swift_vio

#endif // MAPPOINT_H
