#ifndef XLA_CLIENT_TYPES_H_
#define XLA_CLIENT_TYPES_H_

#include <cmath>
#include <vector>

#include "absl/numeric/int128.h"
#include "absl/types/optional.h"
#include "xla/types.h"

namespace xla {

using hash_t = absl::uint128;

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
  uint64_t start_nstime = 0;
  uint64_t end_nstime = 0;
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
  absl::optional<int64_t> int64_value;
};

}  // namespace xla

#endif  // XLA_CLIENT_TYPES_H_
