#include "metrics_snapshot.h"

#include <regex>

#include "tensorflow/compiler/xla/xla_client/tf_logging.h"
#include "tensorflow/compiler/xla/xla_client/util.h"

namespace torch_xla {
namespace cpp_test {

MetricsSnapshot::MetricsSnapshot() {
  for (auto& name : xla::metrics::GetMetricNames()) {
    xla::metrics::MetricData* metric = xla::metrics::GetMetric(name);
    MetricSamples msamples;
    msamples.samples =
        metric->Samples(&msamples.accumulator, &msamples.total_samples);
    metrics_map_.emplace(name, std::move(msamples));
  }
  for (auto& name : xla::metrics::GetCounterNames()) {
    xla::metrics::CounterData* counter = xla::metrics::GetCounter(name);
    counters_map_.emplace(name, counter->Value());
  }
}

bool MetricsSnapshot::CounterChanged(
    const std::string& counter_regex, const MetricsSnapshot& after,
    const std::unordered_set<std::string>* ignore_set) const {
  std::regex cregex(counter_regex);
  for (auto& name_counter : after.counters_map_) {
    std::smatch match;
    if ((ignore_set == nullptr || ignore_set->count(name_counter.first) == 0) &&
        std::regex_match(name_counter.first, match, cregex)) {
      xla::int64 start_value =
          xla::util::FindOr(counters_map_, name_counter.first, 0);
      if (name_counter.second != start_value) {
        TF_LOG(INFO) << "Counter '" << name_counter.first
                     << "' changed: " << start_value << " -> "
                     << name_counter.second;
        return true;
      }
    }
  }
  return false;
}

}  // namespace cpp_test
}  // namespace torch_xla
