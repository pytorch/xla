#ifndef TENSORFLOW_COMPILER_XLA_XLA_CLIENT_MULTI_WAIT_H_
#define TENSORFLOW_COMPILER_XLA_XLA_CLIENT_MULTI_WAIT_H_

#include <condition_variable>
#include <functional>
#include <mutex>

#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/types.h"

namespace xla {
namespace xla_util {

// Support waiting for a number of tasks to complete.
class MultiWait {
 public:
  explicit MultiWait(size_t count) : count_(count) {}

  // Signal the completion of a single task.
  void Done() { Done(Status::OK()); }
  void Done(Status status);

  // Waits until at least count (passed as constructor value) completions
  // happened.
  Status Wait();

  // Resets the threshold counter for the MultiWait object. The completed count
  // is also reset to zero.
  void Reset(size_t count);

  // Creates a completer functor which signals the mult wait object once func
  // has completed. Handles exceptions by signaling the multi wait with the
  // proper status value.
  std::function<void()> Completer(std::function<void()> func);

 private:
  std::mutex mutex_;
  std::condition_variable cv_;
  size_t count_ = 0;
  size_t completed_count_ = 0;
  Status status_;
};

}  // namespace xla_util
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_XLA_CLIENT_MULTI_WAIT_H_
