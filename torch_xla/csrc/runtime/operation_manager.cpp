#include "torch_xla/csrc/runtime/operation_manager.h"

#include <shared_mutex>

#include "absl/types/span.h"
#include "torch_xla/csrc/runtime/debug_macros.h"
#include "torch_xla/csrc/runtime/tf_logging.h"

namespace torch_xla {
namespace runtime {

OperationManager::OperationManager(absl::Span<const std::string> devices) {
  for (auto& device : devices) {
    op_counters_.try_emplace(device, device);
  }
}

OperationManager::OperationTracker::OperationTracker(Counter* counter)
    : counter_(counter) {
  XLA_CHECK(counter_);
  counter_->Increment();
}

OperationManager::OperationTracker::~OperationTracker() {
  counter_->Decrement();
}

std::unique_ptr<OperationManager::OperationTracker>
OperationManager::StartOperation(std::string device) {
  return std::make_unique<OperationTracker>(&op_counters_.at(device));
}

void OperationManager::WaitForDevices(absl::Span<const std::string> devices) {
  std::vector<std::unique_lock<std::shared_mutex>> locks;
  locks.reserve(devices.size());

  for (const std::string& device_str : devices) {
    TF_VLOG(5) << "Blocking new operations on " << device_str;
    auto lock = op_counters_.at(device_str).BlockNewOperations();
    locks.emplace_back(std::move(lock));

    TF_VLOG(3) << "Waiting for device execution for " << device_str
               << " to finish";
    op_counters_.at(device_str).Wait();
    TF_VLOG(3) << "Finished operations on device " << device_str;
  }
}

void OperationManager::Counter::Increment() {
  // Block new operations after BlockNewOperations() is called. count_ is
  // already atomic, so atomic so we don't need an exclusive lock to prevent
  // data races.
  std::shared_lock lock(pending_operations_mu_);
  auto current = count_.fetch_add(1, std::memory_order_acq_rel) + 1;
  TF_VLOG(5) << "Incremented operations for " << device_ << " to " << current;
}

void OperationManager::Counter::Decrement() {
  auto current = count_.fetch_sub(1, std::memory_order_acq_rel) - 1;
  TF_VLOG(5) << "Decremented operations for " << device_ << " to " << current;

  if (current == 0) {
    std::unique_lock cv_lock(cv_mu_);
    TF_VLOG(3) << "All operations complete for " << device_;
    cv_.notify_all();
  }
}

std::unique_lock<std::shared_mutex>
OperationManager::Counter::BlockNewOperations() {
  return std::unique_lock(pending_operations_mu_);
}

void OperationManager::Counter::Wait() {
  TF_VLOG(5) << "Waiting for " << count_ << " operations on " << device_;
  std::unique_lock cv_lock(cv_mu_);
  cv_.wait(cv_lock,
           [this] { return count_.load(std::memory_order_acquire) == 0; });
  TF_VLOG(5) << "Done waiting for " << device_;
}

}  // namespace runtime
}  // namespace torch_xla
