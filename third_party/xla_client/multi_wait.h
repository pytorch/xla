#ifndef TENSORFLOW_COMPILER_XLA_XLA_CLIENT_MULTI_WAIT_H_
#define TENSORFLOW_COMPILER_XLA_XLA_CLIENT_MULTI_WAIT_H_

#include <condition_variable>
#include <mutex>

namespace xla {
namespace xla_util {

// Support waiting for a number of tasks to complete.
class MultiWait {
 public:
  // Signal the completion of a single task.
  void Done();

  // Waits until at leas count tasks completed.
  void Wait(size_t count);

 private:
  std::mutex mutex_;
  std::condition_variable cv_;
  size_t completed_count_ = 0;
};

}  // namespace xla_util
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_XLA_CLIENT_MULTI_WAIT_H_
