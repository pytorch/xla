#ifndef XLA_TEST_CPP_METRICS_SNAPSHOT_H_
#define XLA_TEST_CPP_METRICS_SNAPSHOT_H_

#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "tensorflow/compiler/xla/types.h"
#include "third_party/xla_client/metrics.h"

namespace torch_xla {
namespace cpp_test {

class MetricsSnapshot {
 public:
  struct ChangedCounter {
    std::string name;
    int64_t before = 0;
    int64_t after = 0;
  };

  MetricsSnapshot();

  std::vector<ChangedCounter> CounterChanged(
      const std::string& counter_regex, const MetricsSnapshot& after,
      const std::unordered_set<std::string>* ignore_set) const;

  std::string DumpDifferences(
      const MetricsSnapshot& after,
      const std::unordered_set<std::string>* ignore_set) const;

 private:
  struct MetricSamples {
    std::vector<xla::metrics::Sample> samples;
    double accumulator = 0.0;
    size_t total_samples = 0;
  };

  static void DumpMetricDifference(const std::string& name,
                                   const MetricSamples& before,
                                   const MetricSamples& after,
                                   std::stringstream* ss);

  std::unordered_map<std::string, MetricSamples> metrics_map_;
  std::unordered_map<std::string, int64_t> counters_map_;
};

}  // namespace cpp_test
}  // namespace torch_xla

#endif  // XLA_TEST_CPP_METRICS_SNAPSHOT_H_