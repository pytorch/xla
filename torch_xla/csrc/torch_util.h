#pragma once

#include <ATen/ATen.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Optional.h>

namespace torch_xla {

// Makes a deep copy of an ATEN tensor.
at::Tensor CopyTensor(const at::Tensor& ref);

// Same as above, with an additional cast.
at::Tensor CopyTensor(const at::Tensor& ref, at::ScalarType dest_type);

// Return at::ScalarType from at::Scalar
at::ScalarType GetScalarType(at::Scalar scalar);

template <typename T, typename S>
T OptionalOr(const c10::optional<S>& value, T defval) {
  return value ? static_cast<T>(*value) : defval;
}

}  // namespace torch_xla
