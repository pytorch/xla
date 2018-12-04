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
  void Activate();

 private:
  // Function implementing the main thread loop running the user function.
  void Runner();

  std::function<void()> function_;
  std::mutex mutex_;
  std::condition_variable cv_;
  bool activated_ = false;
  bool stopped_ = false;
  std::unique_ptr<std::thread> thread_;
};

}  // namespace xla_util
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_XLA_CLIENT_TRIGGERED_TASK_H_
