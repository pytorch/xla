#ifndef XLA_CLIENT_METRICS_READER_H_
#define XLA_CLIENT_METRICS_READER_H_

#include <string>
#include <vector>

namespace xla {
namespace metrics_reader {

// Creates a report with the current metrics statistics.
std::string CreateMetricReport();

// Creates a report with the selected metrics statistics.
std::string CreateMetricReport(const std::vector<std::string>& counter_names,
                               const std::vector<std::string>& metric_names);

}  // namespace metrics_reader
}  // namespace xla

#endif  // XLA_CLIENT_METRICS_READER_H_
