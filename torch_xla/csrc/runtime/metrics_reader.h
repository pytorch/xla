#ifndef XLA_CLIENT_METRICS_READER_H_
#define XLA_CLIENT_METRICS_READER_H_

#include <map>
#include <string>
#include <vector>

#include "torch_xla/csrc/runtime/metrics.h"
#include "torch_xla/csrc/runtime/types.h"

namespace torch_xla {
namespace runtime {
namespace metrics_reader {

// Creates a report with the current metrics statistics.
std::string CreateMetricReport(
    const std::map<std::string, torch_xla::runtime::Metric>& xrt_metrics);

// Creates a report with the selected metrics statistics.
std::string CreateMetricReport(const std::vector<std::string>& counter_names,
                               const std::vector<std::string>& metric_names);

}  // namespace metrics_reader
}  // namespace runtime
}  // namespace torch_xla

#endif  // XLA_CLIENT_METRICS_READER_H_
