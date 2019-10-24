#include "torch_xla_test.h"

#include <ATen/ATen.h>

#include "absl/memory/memory.h"
#include "torch_xla/csrc/aten_xla_type.h"
#include "torch_xla/csrc/helpers.h"

namespace torch_xla {
namespace cpp_test {

void XlaTest::SetUp() {
  at::manual_seed(42);
  start_msnap_ = absl::make_unique<MetricsSnapshot>();
}

void XlaTest::TearDown() {}

bool XlaTest::CounterChanged(
    const std::string& counter_regex,
    const std::unordered_set<std::string>* ignore_set) {
  MakeEndSnapshot();
  return start_msnap_->CounterChanged(counter_regex, *end_msnap_, ignore_set);
}

void XlaTest::MakeEndSnapshot() {
  if (end_msnap_ == nullptr) {
    end_msnap_ = absl::make_unique<MetricsSnapshot>();
  }
}

void XlaTest::CommonSetup() {
  XlaHelpers::set_mat_mul_precision(xla::PrecisionConfig::HIGHEST);
}

void TorchXlaTest::SetUpTestCase() { CommonSetup(); }

void AtenXlaTensorTestBase::SetUpTestCase() {
  CommonSetup();
  AtenXlaType::InitializeAtenBindings();
}

}  // namespace cpp_test
}  // namespace torch_xla
