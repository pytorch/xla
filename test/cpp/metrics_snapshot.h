#pragma once

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_client/metrics.h"

namespace torch_xla {
namespace cpp_test {

class MetricsSnapshot {
 public:
  MetricsSnapshot();

  bool CounterChanged(const std::string& counter_regex,
                      const MetricsSnapshot& after,
                      const std::unordered_set<std::string>* ignore_set) const;

 private:
  struct MetricSamples {
    std::vector<xla::metrics::Sample> samples;
    double accumulator = 0.0;
    size_t total_samples = 0;
  };

  std::unordered_map<std::string, MetricSamples> metrics_map_;
  std::unordered_map<std::string, xla::int64> counters_map_;
};

}  // namespace cpp_test
}  // namespace torch_xla
