#include "torch_xla_test.h"

#include <ATen/ATen.h>

#include "absl/memory/memory.h"
#include "tensorflow/compiler/xla/xla_client/tf_logging.h"
#include "torch_xla/csrc/aten_xla_type.h"
#include "torch_xla/csrc/helpers.h"

namespace torch_xla {
namespace cpp_test {

void XlaTest::SetUp() {
  at::manual_seed(42);
  start_msnap_ = absl::make_unique<MetricsSnapshot>();
}

void XlaTest::TearDown() {}

void XlaTest::ExpectCounterNotChanged(
    const std::string& counter_regex,
    const std::unordered_set<std::string>* ignore_set) {
  MakeEndSnapshot();
  auto changed =
      start_msnap_->CounterChanged(counter_regex, *end_msnap_, ignore_set);
  for (auto& change_counter : changed) {
    TF_LOG(INFO) << "Counter '" << change_counter.name
                 << "' changed: " << change_counter.before << " -> "
                 << change_counter.after;
  }
  EXPECT_TRUE(changed.empty());
}

void XlaTest::ExpectCounterChanged(
    const std::string& counter_regex,
    const std::unordered_set<std::string>* ignore_set) {
  MakeEndSnapshot();
  auto changed =
      start_msnap_->CounterChanged(counter_regex, *end_msnap_, ignore_set);
  EXPECT_TRUE(!changed.empty());
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
