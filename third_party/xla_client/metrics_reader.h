#ifndef XLA_CLIENT_METRICS_READER_H_
#define XLA_CLIENT_METRICS_READER_H_

#include <map>
#include <string>
#include <vector>

#include "third_party/xla_client/metrics.h"

namespace xla {
namespace metrics_reader {

// Creates a report with the current metrics statistics.
std::string CreateMetricReport(
    const std::map<std::string, xla::Metric>& xrt_metrics);

// Creates a report with the selected metrics statistics.
std::string CreateMetricReport(const std::vector<std::string>& counter_names,
                               const std::vector<std::string>& metric_names);

}  // namespace metrics_reader
}  // namespace xla

#endif  // XLA_CLIENT_METRICS_READER_H_
