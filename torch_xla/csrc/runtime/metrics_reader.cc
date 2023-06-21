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

}  // namespace

std::string CreateMetricReport(
    const std::map<std::string, Metric>& metrics) {
  return metrics::CreateMetricReport();
}

std::string CreateMetricReport(const std::vector<std::string>& counter_names,
                               const std::vector<std::string>& metric_names) {
  return metrics::CreateMetricReport(counter_names, metric_names);
}

}  // namespace metrics_reader
}  // namespace runtime
}  // namespace torch_xla
