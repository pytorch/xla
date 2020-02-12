#ifndef XLA_CLIENT_TYPES_H_
#define XLA_CLIENT_TYPES_H_

#include <cmath>
#include <vector>

#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/types.h"

namespace xla {

struct Percentile {
  enum class UnitOfMeaure {
    kNumber,
    kTime,
    kBytes,
  };
  struct Point {
    double percentile = 0.0;
    double value = 0.0;
  };

  UnitOfMeaure unit_of_measure = UnitOfMeaure::kNumber;
  uint64 start_nstime = 0;
  uint64 end_nstime = 0;
  double min_value = NAN;
  double max_value = NAN;
  double mean = NAN;
  double stddev = NAN;
  size_t num_samples = 0;
  size_t total_samples = 0;
  double accumulator = NAN;
  std::vector<Point> points;
};

struct Metric {
  absl::optional<Percentile> percentile;
  absl::optional<int64> int64_value;
};

}  // namespace xla

#endif  // XLA_CLIENT_TYPES_H_
