#ifndef XLA_CLIENT_OPERATION_TRACKER_H_
#define XLA_CLIENT_OPERATION_TRACKER_H_

#include <atomic>
#include <condition_variable>
#include <memory>
#include <shared_mutex>
#include <mutex>

#include "absl/types/span.h"
// #include "absl/synchronization/notification.h"

namespace torch_xla {
namespace runtime {

class OperationTracker {
 public:
  OperationTracker() = default;
  OperationTracker(absl::Span<const std::string>);

  OperationTracker(const OperationTracker&) = delete;
  OperationTracker& operator=(const OperationTracker&) = delete;

  OperationTracker(OperationTracker&&) = default;
  OperationTracker& operator=(OperationTracker&&) = default;

  class Counter {
   public:
    void Increment();

    void Decrement();

    void Wait();

    std::unique_lock<std::shared_mutex> BlockNewOperations();

   private:
    std::shared_mutex pending_operations_mu_;
    std::atomic<int64_t> count_;

    std::mutex cv_mu_;
    std::condition_variable cv_;
  };

  class Operation {
   public:
    Operation(Counter* counter);
    ~Operation();

    Operation(const Operation&) = delete;
    Operation& operator=(const Operation&) = delete;

   private:
    std::string device_;
    Counter* counter_;
  };

  // TODO: should this be a shared_ptr?
  std::shared_ptr<Operation> StartOperation(std::string device);

  void WaitForDevices(absl::Span<const std::string> devices);

 private:
  // TODO: figure out how to safely construct counters in map
  std::unordered_map<std::string, std::unique_ptr<Counter>> op_counters_;
};

}  // namespace runtime
}  // namespace torch_xla

#endif
