#pragma once

#include <vector>

#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/types.h"

namespace xla {

std::vector<int64> InversePermutation(
    absl::Span<const int64> input_permutation);

bool IsPermutation(absl::Span<const int64> permutation);

bool IsIdentityPermutation(absl::Span<const int64> permutation);

template <typename Container>
inline std::vector<typename Container::value_type> PermuteInverse(
    const Container& input, absl::Span<const int64> permutation) {
  using T = typename Container::value_type;
  absl::Span<const T> data(input);
  CHECK(IsPermutation(permutation));
  std::vector<T> output(data.size());
  for (size_t i = 0; i < permutation.size(); ++i) {
    output[permutation[i]] = data[i];
  }
  return output;
}

}  // namespace xla
