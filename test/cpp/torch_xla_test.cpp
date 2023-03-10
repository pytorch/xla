#include "test/cpp/torch_xla_test.h"

#include <ATen/ATen.h>

#include "absl/memory/memory.h"
#include "third_party/xla_client/sys_util.h"
#include "third_party/xla_client/tf_logging.h"
#include "torch_xla/csrc/device.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/tensor.h"
#include "torch_xla/csrc/xla_backend_impl.h"
#include "torch_xla/csrc/xla_graph_executor.h"

namespace torch_xla {
namespace cpp_test {

static bool xla_backend_inited = InitXlaBackend();

void XlaTest::SetUp() {
  at::manual_seed(42);
  XLAGraphExecutor::Get()->SetRngSeed(GetCurrentDevice(), 42);
  start_msnap_ = absl::make_unique<MetricsSnapshot>();
}

void XlaTest::TearDown() {
  static bool dump_metrics =
      xla::sys_util::GetEnvBool("XLA_TEST_DUMP_METRICS", false);
  if (dump_metrics) {
    MakeEndSnapshot();

    std::string diffs = start_msnap_->DumpDifferences(*end_msnap_,
                                                      /*ignore_se=*/nullptr);
    if (!diffs.empty()) {
      TF_LOG(INFO)
          << ::testing::UnitTest::GetInstance()->current_test_info()->name()
          << " Metrics Differences:\n"
          << diffs;
    }
  }
}

static void ExpectCounterNotChanged_(
    const std::vector<MetricsSnapshot::ChangedCounter>& changed) {
  for (auto& change_counter : changed) {
    TF_LOG(INFO) << "Counter '" << change_counter.name
                 << "' changed: " << change_counter.before << " -> "
                 << change_counter.after;
  }
  EXPECT_TRUE(changed.empty());
}

void XlaTest::ExpectCounterNotChanged(
    const std::string& counter_regex,
    const std::unordered_set<std::string>* ignore_set) {
  MakeEndSnapshot();
  auto changed =
      start_msnap_->CounterChanged(counter_regex, *end_msnap_, ignore_set);

  ExpectCounterNotChanged_(changed);

  // Some operators could've been renamed to `opName_symint`, yet the tests are
  // using the old names. We modify `ExpectCounterNotChanged` to also check
  // `opName_symint` counters. When we finish migrating the ops to symints, we
  // would remove this logic and fix all the tests
  auto changed_symint = start_msnap_->CounterChanged(counter_regex + "_symint",
                                                     *end_msnap_, ignore_set);

  ExpectCounterNotChanged_(changed_symint);
}

void XlaTest::ExpectCounterChanged(
    const std::string& counter_regex,
    const std::unordered_set<std::string>* ignore_set) {
  MakeEndSnapshot();
  auto changed =
      start_msnap_->CounterChanged(counter_regex, *end_msnap_, ignore_set);

  // Some operators could've been renamed to `opName_symint`, yet the tests are
  // using the old names. We modify `ExpectCounterChanged` to also check
  // `opName_symint` counters. When we finish migrating the ops to symints, we
  // would remove this logic and fix all the tests
  auto changed_symint = start_msnap_->CounterChanged(counter_regex + "_symint",
                                                     *end_msnap_, ignore_set);
  EXPECT_TRUE(!changed.empty() || !changed_symint.empty());
  // We expect *either* changed or changed_symint to contain changed counters
  // but not *both*. Likewise, if both are empty, the assertion above should
  // fail
  EXPECT_TRUE(changed.empty() != changed_symint.empty());
}

void XlaTest::ResetCounters() {
  start_msnap_ = std::move(end_msnap_);
  end_msnap_ = nullptr;
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

void AtenXlaTensorTestBase::SetUpTestCase() { CommonSetup(); }

}  // namespace cpp_test
}  // namespace torch_xla
