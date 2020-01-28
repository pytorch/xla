#ifndef TENSORFLOW_COMPILER_XLA_RPC_METRICS_READER_H_
#define TENSORFLOW_COMPILER_XLA_RPC_METRICS_READER_H_

#include <string>

namespace xla {
namespace metrics_reader {

// Creates a report with the current metrics statistics.
std::string CreateMetricReport();

}  // namespace metrics_reader
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_RPC_METRICS_READER_H_
