#include "torch_xla/csrc/index_op_util.h"
#include <ATen/ExpandUtils.h>
#include <ATen/Functions.h>
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "torch_xla/csrc/aten_xla_bridge.h"

namespace torch_xla {
namespace {

void CheckIndexTensorTypes(at::TensorList indices) {
  for (auto& tensor : indices) {
    if (tensor.defined()) {
      auto& type = tensor.type();
      auto scalarType = type.scalarType();
      if (scalarType != at::kLong && scalarType != at::kByte) {
        XLA_ERROR() << "tensors used as indices must be long or byte tensors";
      }
    }
  }
}

// Expands byte tensors (masks) into the equivalent indexing by LongTensors.
// This is a version of at::native::expandByteTensors with style adjustments.
std::vector<at::Tensor> ExpandByteTensors(const at::Tensor& self,
                                          at::TensorList indices) {
  std::vector<at::Tensor> result;
  for (auto& index : indices) {
    if (index.type().scalarType() == at::kByte) {
      // The sizes of the ByteTensor mask must match the sizes of the
      // corresponding dimensions in self.
      for (int64_t j = 0; j < index.dim(); j++) {
        int64_t src_idx = result.size() + j;
        XLA_CHECK_EQ(index.size(j), self.size(src_idx))
            << "The shape of the mask " << index.sizes() << " at index " << j
            << " does not match the shape of the indexed tensor "
            << self.sizes() << " at index " << src_idx;
      }
      // Replace with nonzeros.
      auto nonzero = index.nonzero();
      for (int64_t j = 0; j < index.dim(); j++) {
        result.emplace_back(nonzero.select(1, j));
      }
    } else {
      result.emplace_back(index);
    }
  }
  return result;
}

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
                                         at::TensorList orig_indices) {
  CheckIndexTensorTypes(orig_indices);
  // First expand ByteTensor (boolean masks) into 1 or more LongTensors, then
  // broadcast all index tensors together.
  auto indices = at::expand_outplace(ExpandByteTensors(base, orig_indices));
  // If the non-null indices are not all adjacent, transpose base and indices
  // together so that they're adjacent at the front.
  CanonicalIndexInfo canonical_index_info = TransposeToFront(base, indices);
  // Ensure indices are on the same device as the base.
  for (size_t i = 0; i < canonical_index_info.indices.size(); i++) {
    if (canonical_index_info.indices[i].device() != base.device()) {
      canonical_index_info.indices[i] =
          canonical_index_info.indices[i].to(base.device());
    }
  }
  return canonical_index_info;
}

}  // namespace torch_xla
