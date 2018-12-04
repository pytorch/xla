#include "tensorflow/compiler/xla/xla_client/triggered_task.h"

namespace xla {
namespace xla_util {

TriggeredTask::TriggeredTask(std::function<void()> function)
    : function_(std::move(function)),
      thread_(new std::thread([this]() { Runner(); })) {}

void TriggeredTask::Stop() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stopped_ = true;
  }
  cv_.notify_one();
  thread_->join();
}

void TriggeredTask::Activate() {
  bool notify = false;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    notify = !activated_;
    activated_ = true;
  }
  if (notify) {
    cv_.notify_one();
  }
}

void TriggeredTask::Runner() {
  while (true) {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait(lock, [this] { return activated_ || stopped_; });
      if (stopped_) {
        break;
      }
      activated_ = false;
    }
    function_();
  }
}

}  // namespace xla_util
}  // namespace xla
