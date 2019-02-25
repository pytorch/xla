#include "torch_xla/csrc/index_op_util.h"
#include <ATen/ExpandUtils.h>
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"

namespace torch_xla {
namespace {

// Transposes the tensor and indices together so that all the non-null indices
// index the first k dimensions of the tensor. Returns the transposed tensor and
// the reordered indices. For example:
//  TransposeToFront(tensor, {nullptr, a, nullptr, b})
// returns
//  tensor.permute([1, 3, 0, 2]), {a, b}
//
// This is a simplified version of at::native::transposeToFront which better
// fits our requirements.
CanonicalIndexInfo TransposeToFront(at::Tensor base, at::TensorList indices) {
  std::vector<int64_t> dims;
  std::vector<at::Tensor> transposed_indices;
  size_t base_rank = base.dim();
  dims.reserve(base_rank);
  XLA_CHECK_LE(indices.size(), base_rank);
  for (size_t i = 0; i < indices.size(); i++) {
    if (indices[i].defined()) {
      dims.push_back(i);
      transposed_indices.emplace_back(indices[i]);
    }
  }
  for (size_t i = 0; i < indices.size(); i++) {
    if (!indices[i].defined()) {
      dims.push_back(i);
    }
  }
  for (size_t i = indices.size(); i < base_rank; ++i) {
    dims.push_back(i);
  }
  return {base.permute(dims), std::move(transposed_indices)};
}

}  // namespace

CanonicalIndexInfo GetCanonicalIndexInfo(const at::Tensor& base,
                                         at::TensorList indices) {
  const auto expanded_indices = at::expand_outplace(indices);
  // If the non-null indices are not all adjacent, transpose base and indices
  // together so that they're adjacent at the front.
  return TransposeToFront(base, expanded_indices);
}

}  // namespace torch_xla
