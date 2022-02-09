#ifndef INCLUDE_OKVIS_LOOP_CLOSURE_MODULE_HPP_
#define INCLUDE_OKVIS_LOOP_CLOSURE_MODULE_HPP_

#include <memory>
#include <thread>

#include <loop_closure/KeyframeForLoopDetection.hpp>
#include <loop_closure/LoopClosureMethod.hpp>

#include <okvis/threadsafe/ThreadsafeQueue.hpp>
#include <okvis/timing/Timer.hpp>

namespace swift_vio {
typedef std::function<void(std::shared_ptr<LoopFrameAndMatches>)>
    OutputLoopFrameCallback;

typedef std::function<
  void(const okvis::Time &, const okvis::kinematics::Transformation &)> StateCallback;
class LoopClosureModule {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

#ifdef DEACTIVATE_TIMERS
  typedef okvis::timing::DummyTimer TimerSwitchable;
#else
  typedef okvis::timing::Timer TimerSwitchable;
#endif
  LoopClosureModule();

  explicit LoopClosureModule(
      std::shared_ptr<LoopClosureMethod> loopClosureMethod);

  ~LoopClosureModule();

  void setBlocking(bool blocking);

  bool push(std::shared_ptr<LoopQueryKeyframeMessage> queryKeyframe);

  void setOutputLoopFrameCallback(
      const OutputLoopFrameCallback& outputCallback);

  /// \brief Append the stateCallback to be called every time a new keyframe is processed.
  ///        When an implementing class has an estimate, they can call:
  ///        stateCallback_( stamp, T_w_vk );
  ///        where stamp is the timestamp
  ///        and T_w_vk is the transformation (and uncertainty) that
  ///        transforms points from the vehicle frame to the world frame
  void appendStateCallback(const StateCallback & stateCallback);

  void startThreads();

  void loopClosureLoop();

  /// \brief Loop that publishes the newest state.
  void publisherLoop();

  void shutdown();

 private:
  bool blocking_;  ///< Blocking option. Whether the addMeasurement() functions
                   ///< should wait until proccessing is complete.
                   ///< Default to block because the PGO module requires all keyframes are connected.

  okvis::threadsafe::ThreadSafeQueue<
      std::shared_ptr<LoopQueryKeyframeMessage>>
      queryKeyframeList_;

  /// The queue containing the results of the PGO ready for publishing.
  okvis::threadsafe::ThreadSafeQueue<PgoResult> pgoResults_;

  OutputLoopFrameCallback
      outputLoopFrameCallback_;  ///< output loop frame and matches callback
                                 ///< function.

  std::shared_ptr<LoopClosureMethod> loopClosureMethod_;

  std::thread loopClosureThread_;   ///< Thread running loopClosureLoop().
  std::thread publisherThread_;     ///< Thread running publisherLoop().
  std::vector<StateCallback> stateCallbackList_; ///< State callback functions.
  const size_t maxQueryKeyframeQueueSize_ = 5;
};
}  // namespace swift_vio
#endif  // INCLUDE_OKVIS_LOOP_CLOSURE_MODULE_HPP_
