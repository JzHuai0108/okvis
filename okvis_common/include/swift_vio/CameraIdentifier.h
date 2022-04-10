#ifndef CAMERAIDENTIFIER_H
#define CAMERAIDENTIFIER_H

#include <unordered_map>

namespace swift_vio {

struct CameraIdentifier {
  uint64_t frameId;
  size_t cameraIndex;

  CameraIdentifier(uint64_t fi = 0, size_t ci = 0)
      : frameId(fi), cameraIndex(ci) {}

  bool isBinaryEqual(const CameraIdentifier &rhs) const {
    return frameId == rhs.frameId && cameraIndex == rhs.cameraIndex;
  }

  bool operator==(const CameraIdentifier &rhs) const {
    return isBinaryEqual(rhs);
  }

  bool operator<(const CameraIdentifier &rhs) const {
    if (frameId == rhs.frameId) {
      return cameraIndex < rhs.cameraIndex;
    }
    return frameId < rhs.frameId;
  }
};
}  // namespace swift_vio

template<>
struct std::hash<swift_vio::CameraIdentifier>
{
    std::size_t operator()(swift_vio::CameraIdentifier const& s) const noexcept
    {
        return std::hash<uint64_t>{}(s.frameId + s.cameraIndex);
    }
};

#endif // CAMERAIDENTIFIER_H
