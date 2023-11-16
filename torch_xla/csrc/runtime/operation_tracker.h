#ifndef XLA_CLIENT_OPERATION_TRACKER_H_
#define XLA_CLIENT_OPERATION_TRACKER_H_

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <shared_mutex>

#include "absl/types/span.h"

namespace torch_xla {
namespace runtime {

// Track inflight operations for each device.
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
    Counter(const std::string& device) : device_(device){};

    Counter(const Counter&) = delete;
    Counter& operator=(const Counter&) = delete;

    // Register a new operation. Blocks if `BlockNewOperations` has been called.
    void Increment();

    // Mark an inflight task completed.
    void Decrement();

    // Wait until all operations are complete. Does not block new operations
    // (see BlockNewOperations).
    void Wait();

    // Returns a lock that prevents new operations on the device.
    std::unique_lock<std::shared_mutex> BlockNewOperations();

   private:
    std::string device_;

    std::shared_mutex pending_operations_mu_;
    std::atomic<int64_t> count_{0};

    std::mutex cv_mu_;
    std::condition_variable cv_;
  };

  class Operation {
   public:
    // Register an operation in the `counter_`.
    Operation(Counter* counter);

    // Mark an operation complete in `counter_`.
    ~Operation();

    Operation(const Operation&) = delete;
    Operation& operator=(const Operation&) = delete;

   private:
    std::string device_;
    Counter* counter_;
  };

  // Register a new operation for `device`.
  std::unique_ptr<Operation> StartOperation(std::string device);

  // Wait for all device execution to complete on devices.
  void WaitForDevices(absl::Span<const std::string> devices);

 private:
  std::unordered_map<std::string, Counter> op_counters_;
};

}  // namespace runtime
}  // namespace torch_xla

#endif
