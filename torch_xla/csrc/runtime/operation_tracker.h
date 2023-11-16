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
    Counter(const std::string& device) : device_(device) {};

    Counter(const Counter&) = delete;
    Counter& operator=(const Counter&) = delete;

    void Increment();

    void Decrement();

    void Wait();

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
    Operation(Counter* counter);
    ~Operation();

    Operation(const Operation&) = delete;
    Operation& operator=(const Operation&) = delete;

   private:
    std::string device_;
    Counter* counter_;
  };

  std::unique_ptr<Operation> StartOperation(std::string device);

  void WaitForDevices(absl::Span<const std::string> devices);

 private:
  std::unordered_map<std::string, Counter> op_counters_;
};

}  // namespace runtime
}  // namespace torch_xla

#endif
