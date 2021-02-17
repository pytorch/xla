#ifndef XLA_CLIENT_METRICS_ANALYSIS_H_
#define XLA_CLIENT_METRICS_ANALYSIS_H_

#include <iostream>
#include <memory>
#include <vector>

namespace xla {
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
  virtual Analysis Run() = 0;
};

std::string CreatePerformanceReport();

}  // namespace metrics
}  // namespace xla

#endif  // XLA_CLIENT_METRICS_ANALYSIS_H_
