#ifndef TENSORFLOW_COMPILER_XLA_XLA_CLIENT_TRIGGERED_TASK_H_
#define TENSORFLOW_COMPILER_XLA_XLA_CLIENT_TRIGGERED_TASK_H_

#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>

namespace xla {
namespace xla_util {

// Wraps a function which should be run many times upon user activations.
class TriggeredTask {
 public:
  explicit TriggeredTask(std::function<void()> function);

  // Stops the background thread and waits for it to complete.
  void Stop();

  // Triggers a function run. If the function is already running, it will run
  // again immediately after it completes.
  size_t Activate();

  // Wait until a run-ID returned by the Activate() API completed. Returns true
  // if the run was successfully completed, or false if Stop() was called
  // before.
  bool WaitForRun(size_t run_id);

 private:
  // Function implementing the main thread loop running the user function.
  void Runner();

  std::function<void()> function_;
  std::mutex mutex_;
  std::condition_variable cv_;
  std::condition_variable run_cv_;
  size_t run_id_ = 0;
  size_t run_waiters_ = 0;
  bool activated_ = false;
  bool stopped_ = false;
  std::unique_ptr<std::thread> thread_;
};

}  // namespace xla_util
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_XLA_CLIENT_TRIGGERED_TASK_H_
