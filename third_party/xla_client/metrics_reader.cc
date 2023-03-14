#include "third_party/xla_client/metrics_reader.h"

#include <sstream>

#include "third_party/xla_client/debug_macros.h"
#include "third_party/xla_client/metrics.h"
#include "third_party/xla_client/util.h"

namespace xla {
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
    const std::map<std::string, Metric>& xrt_metrics) {
  return metrics::CreateMetricReport();
}

std::string CreateMetricReport(const std::vector<std::string>& counter_names,
                               const std::vector<std::string>& metric_names) {
  return metrics::CreateMetricReport(counter_names, metric_names);
}

}  // namespace metrics_reader
}  // namespace xla
