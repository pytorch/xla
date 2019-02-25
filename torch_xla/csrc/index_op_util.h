#pragma once

#include <ATen/core/Tensor.h>

namespace torch_xla {

struct CanonicalIndexInfo {
  at::Tensor base;
  std::vector<at::Tensor> indices;
};

// Transform the given base and indices to a form supported by the XLATensor
// index implementation. Input indices are reordered so that non-null indices
// are first and the tail of null indices is dropped. The dimensions of the base
// are reordered to be consistent with this reordering.
CanonicalIndexInfo GetCanonicalIndexInfo(const at::Tensor& base,
                                         at::TensorList indices);

}  // namespace torch_xla
