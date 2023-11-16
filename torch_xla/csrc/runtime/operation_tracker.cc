#include "torch_xla/csrc/runtime/operation_tracker.h"

#include <shared_mutex>

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

std::unique_ptr<OperationTracker::Operation> OperationTracker::StartOperation(
    std::string device) {
  return std::make_unique<Operation>(op_counters_.at(device).get());
}

void OperationTracker::WaitForDevices(absl::Span<const std::string> devices) {
  std::vector<std::unique_lock<std::shared_mutex>> locks;
  locks.reserve(devices.size());

  for (const std::string& device_str : devices) {
    TF_VLOG(5) << "Blocking new operations on " << device_str;
    auto lock = op_counters_.at(device_str)->BlockNewOperations();
    locks.emplace_back(std::move(lock));

    TF_VLOG(3) << "Waiting for device execution for " << device_str
               << " to finish";
    op_counters_.at(device_str)->Wait();
    TF_VLOG(3) << "Finished operations on device " << device_str;
  }
}

void OperationTracker::Counter::Increment() {
  // Block new operations after Wait() is called. count_ is already atomic, so
  // atomic so we don't need an exclusive lock to prevent data races.
  std::shared_lock lock(pending_operations_mu_);
  auto current = count_.fetch_add(1, std::memory_order_acq_rel) + 1;
  TF_VLOG(3) << "Increment.... " << current;
}

void OperationTracker::Counter::Decrement() {
  auto current = count_.fetch_sub(1, std::memory_order_acq_rel) - 1;
  TF_VLOG(3) << "Decrement.... " << current;

  if (current == 0) {
    std::unique_lock cv_lock(cv_mu_);
    TF_VLOG(3) << "notify";
    cv_.notify_all();
  }
}

std::unique_lock<std::shared_mutex>
OperationTracker::Counter::BlockNewOperations() {
  return std::unique_lock(pending_operations_mu_);
}

void OperationTracker::Counter::Wait() {
  TF_VLOG(3) << "Waiting.... " << count_;
  std::unique_lock cv_lock(cv_mu_);
  cv_.wait(cv_lock,
           [this] { return count_.load(std::memory_order_acquire) == 0; });
  TF_VLOG(3) << "Done waiting.";
}

}  // namespace runtime
}  // namespace torch_xla
