#ifndef XLA_CLIENT_METRICS_ANALYSIS_H_
#define XLA_CLIENT_METRICS_ANALYSIS_H_

#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "torch_xla/csrc/runtime/types.h"

namespace torch_xla {
namespace runtime {
namespace metrics {

// Performance degradation symptoms detected:
// - Dynamic graphs
// - Very slow graph compilation
// - Very slow graph execution
// - Frequent XLA->CPU transfers
// - Device HBM to host RAM swapping and HBM defragmentation
// - Unlowered aten:: ops

struct Analysis {
  enum class Symptom {
    kNormal,
    kMetricTooFrequent,
    kMetricTooSlow,
    kUnloweredOp,
  };

  Analysis() = default;
  Analysis(Symptom symptom) : symptom(symptom) {}
  Analysis(Symptom symptom, std::string repr) : symptom(symptom), repr(repr) {}

  Symptom symptom;
  std::string repr;
};

class Analyzer {
 public:
  virtual ~Analyzer() = default;

  virtual Analysis Run() = 0;
  virtual Analysis Run(
      const std::map<std::string, torch_xla::runtime::Metric>& metrics) {
    return Run();
  }
};

std::string CreatePerformanceReport(
    const std::map<std::string, torch_xla::runtime::Metric>& metrics);

}  // namespace metrics
}  // namespace runtime
}  // namespace torch_xla

#endif  // XLA_CLIENT_METRICS_ANALYSIS_H_
