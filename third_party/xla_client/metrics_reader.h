#ifndef XLA_CLIENT_METRICS_READER_H_
#define XLA_CLIENT_METRICS_READER_H_

#include <string>

namespace xla {
namespace metrics_reader {

// Creates a report with the current metrics statistics.
std::string CreateMetricReport();

}  // namespace metrics_reader
}  // namespace xla

#endif  // XLA_CLIENT_METRICS_READER_H_
