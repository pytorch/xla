#include "torch_xla/csrc/runtime/metrics_reader.h"

#include <sstream>

#include "torch_xla/csrc/runtime/debug_macros.h"
#include "torch_xla/csrc/runtime/metrics.h"
#include "torch_xla/csrc/runtime/util.h"

namespace torch_xla {
namespace runtime {
namespace metrics_reader {
namespace {

struct MetricFnInfo {
  metrics::MetricReprFn repr_fn;
  double scale;
};

MetricFnInfo GetMetricRenderInfo(const Percentile& percentile) {
  switch (percentile.unit_of_measure) {
    default:
    case Percentile::UnitOfMeaure::kNumber:
      return {metrics::MetricFnValue, 1.0};
    case Percentile::UnitOfMeaure::kTime:
      return {metrics::MetricFnTime, 1e6};
    case Percentile::UnitOfMeaure::kBytes:
      return {metrics::MetricFnBytes, 1.0};
  }
}

std::string CreateXrtMetricReport(
    const std::map<std::string, Metric>& xrt_metrics) {
  std::stringstream ss;
  for (const auto& name_metric : xrt_metrics) {
    if (name_metric.second.percentile) {
      const Percentile& percentile = *name_metric.second.percentile;
      MetricFnInfo minfo = GetMetricRenderInfo(percentile);
      ss << "Metric: " << name_metric.first << std::endl;
      ss << "  TotalSamples: " << percentile.total_samples << std::endl;
      ss << "  Accumulator: "
         << minfo.repr_fn(percentile.accumulator * minfo.scale) << std::endl;
      ss << "  Mean: " << minfo.repr_fn(percentile.mean * minfo.scale)
         << std::endl;
      ss << "  StdDev: " << minfo.repr_fn(percentile.stddev * minfo.scale)
         << std::endl;

      uint64_t delta_time = percentile.end_nstime - percentile.start_nstime;
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
           << "%=" << minfo.repr_fn(percentile.points[i].value * minfo.scale);
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

std::string CreateMetricReport(
    const std::map<std::string, Metric>& xrt_metrics) {
  return metrics::CreateMetricReport() + CreateXrtMetricReport(xrt_metrics);
}

std::string CreateMetricReport(const std::vector<std::string>& counter_names,
                               const std::vector<std::string>& metric_names) {
  return metrics::CreateMetricReport(counter_names, metric_names);
}

}  // namespace metrics_reader
}  // namespace runtime
}  // namespace torch_xla
