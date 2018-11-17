#include "tensorflow/compiler/xla/xla_client/multi_wait.h"

namespace xla {
namespace xla_util {

void MultiWait::Done() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    completed_count_ += 1;
  }
  cv_.notify_all();
}

void MultiWait::Wait(size_t count) {
  std::unique_lock<std::mutex> lock(mutex_);
  cv_.wait(lock, [this, count] { return completed_count_ >= count; });
}

}  // namespace xla_util
}  // namespace xla
