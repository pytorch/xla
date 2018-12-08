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
  run_cv_.notify_all();
  cv_.notify_one();
  thread_->join();
}

size_t TriggeredTask::Activate() {
  bool notify = false;
  size_t run_id;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    notify = !activated_;
    activated_ = true;
    run_id = run_id_;
  }
  if (notify) {
    cv_.notify_one();
  }
  return run_id;
}

bool TriggeredTask::WaitForRun(size_t run_id) {
  std::unique_lock<std::mutex> lock(mutex_);
  ++run_waiters_;
  run_cv_.wait(lock, [this, run_id] { return run_id_ > run_id || stopped_; });
  --run_waiters_;
  return run_id_ > run_id;
}

void TriggeredTask::Runner() {
  while (true) {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      if (run_waiters_ > 0) {
        run_cv_.notify_all();
      }
      cv_.wait(lock, [this] { return activated_ || stopped_; });
      if (stopped_) {
        break;
      }
      ++run_id_;
      activated_ = false;
    }
    function_();
  }
}

}  // namespace xla_util
}  // namespace xla
