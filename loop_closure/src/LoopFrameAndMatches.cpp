#include <loop_closure/LoopFrameAndMatches.hpp>
namespace swift_vio {
LoopFrameAndMatches::LoopFrameAndMatches() {}

LoopFrameAndMatches::LoopFrameAndMatches(
    uint64_t id, okvis::Time stamp, uint64_t queryKeyframeId,
    okvis::Time queryKeyframeStamp,
    const okvis::kinematics::Transformation& T_BlBq)
    : id_(id),
      stamp_(stamp),
      queryKeyframeId_(queryKeyframeId),
      queryKeyframeStamp_(queryKeyframeStamp),
      T_BlBq_(T_BlBq) {}

LoopFrameAndMatches::~LoopFrameAndMatches() {}
}  // namespace swift_vio
