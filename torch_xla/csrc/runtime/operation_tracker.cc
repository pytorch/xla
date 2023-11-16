#include "torch_xla/csrc/runtime/operation_tracker.h"

#include <mutex>

#include "absl/types/span.h"
#include "torch_xla/csrc/runtime/debug_macros.h"
#include "torch_xla/csrc/runtime/tf_logging.h"

namespace torch_xla {
namespace runtime {

OperationTracker::OperationTracker(absl::Span<const std::string> devices) {
  for (auto& device : devices) {
    op_counters_.emplace(device, std::make_unique<Counter>());
  }
}

OperationTracker::Operation::Operation(Counter* counter) : counter_(counter) {
  XLA_CHECK(counter_);
  counter_->Increment();
}

OperationTracker::Operation::~Operation() { counter_->Decrement(); }

std::shared_ptr<OperationTracker::Operation> OperationTracker::StartOperation(
    std::string device) {
  return std::make_shared<Operation>(op_counters_.at(device).get());
}

void OperationTracker::WaitForDevices(absl::Span<const std::string> devices) {
  for (const std::string& device_str : devices) {
    TF_VLOG(3) << "Waiting for device execution for " << device_str
               << " to finish";
    op_counters_.at(device_str)->Wait();
    TF_VLOG(3) << "Waiting for device execution for " << device_str
               << " to finish.. Done";
  }
}

void OperationTracker::Counter::Increment() {
  std::unique_lock<std::mutex> lock(mu_);
  count_++;
  TF_VLOG(3) << "Increment.... " << count_;
}

void OperationTracker::Counter::Decrement() {
  std::unique_lock<std::mutex> lock(mu_);
  count_--;
  TF_VLOG(3) << "Decrement.... " << count_;

  if (count_ == 0) {
    TF_VLOG(3) << "notify";
    cv_.notify_all();
  }
}

void OperationTracker::Counter::Wait() {
  std::unique_lock<std::mutex> lock(mu_);
  TF_VLOG(3) << "Waiting.... " << count_;
  cv_.wait(lock, [this] { return count_ == 0; });
  TF_VLOG(3) << "Done waiting.";
}

}  // namespace runtime
}  // namespace torch_xla
