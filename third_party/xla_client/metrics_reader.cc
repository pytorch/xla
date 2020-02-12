#include "tensorflow/compiler/xla/xla_client/metrics_reader.h"

#include <sstream>

#include "tensorflow/compiler/xla/xla_client/computation_client.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/metrics.h"
#include "tensorflow/compiler/xla/xla_client/util.h"

namespace xla {
namespace metrics_reader {
namespace {

std::string CreateXrtMetricReport() {
  // The metrics are in milliseconds while metrics::MetricFnTime takes
  // nanoseconds.
  // TODO: Read the unit of measure and use the proper formatting function.
  const double scale = 1e6;
  metrics::MetricReprFn repr_fn = metrics::MetricFnTime;
  auto xrt_metrics = ComputationClient::Get()->GetMetrics();
  std::stringstream ss;
  for (auto& name_metric : xrt_metrics) {
    if (name_metric.second.percentile) {
      const Percentile& percentile = *name_metric.second.percentile;
      ss << "Metric: " << name_metric.first << std::endl;
      ss << "  TotalSamples: " << percentile.total_samples << std::endl;
      ss << "  Accumulator: " << repr_fn(percentile.accumulator * scale)
         << std::endl;
      ss << "  Mean: " << repr_fn(percentile.mean * scale) << std::endl;
      ss << "  StdDev: " << repr_fn(percentile.stddev * scale) << std::endl;

      uint64 delta_time = percentile.end_nstime - percentile.start_nstime;
      if (delta_time > 0) {
        double count_sec = 1e6 * (static_cast<double>(percentile.num_samples) /
                                  (delta_time / 1000.0));
        ss << "  Rate: " << count_sec << " / second" << std::endl;
      }

      ss << "  Percentiles: ";
      for (size_t i = 0; i < percentile.points.size(); ++i) {
        if (i > 0) {
          ss << "; ";
        }
        ss << percentile.points[i].percentile
           << "%=" << repr_fn(percentile.points[i].value * scale);
      }
      ss << std::endl;
    } else if (name_metric.second.int64_value) {
      ss << "Counter: " << name_metric.first << std::endl;
      ss << "  Value: " << *name_metric.second.int64_value << std::endl;
    }
  }
  return ss.str();
}

}  // namespace

std::string CreateMetricReport() {
  return metrics::CreateMetricReport() + CreateXrtMetricReport();
}

}  // namespace metrics_reader
}  // namespace xla
