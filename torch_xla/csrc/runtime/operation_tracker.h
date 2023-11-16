#ifndef XLA_CLIENT_OPERATION_TRACKER_H_
#define XLA_CLIENT_OPERATION_TRACKER_H_

#include <condition_variable>
#include <memory>
#include <mutex>

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
    void Increment();

    void Decrement();

    void Wait();

   private:
    std::mutex mu_;
    std::condition_variable cv_;
    int64_t count_;
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
