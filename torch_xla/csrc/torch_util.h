#pragma once

#include <ATen/ATen.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Optional.h>

#include "tensorflow/compiler/xla/shape.h"
#include "torch/csrc/lazy/core/hash.h"
namespace torch_xla {

// Return at::ScalarType from at::Scalar
at::ScalarType GetScalarType(const at::Scalar& scalar);

template <typename T>
at::Scalar MakeIntScalar(T value) {
  return at::Scalar(static_cast<int64_t>(value));
}

template <typename T>
at::Scalar MakeFloatScalar(T value) {
  return at::Scalar(static_cast<double>(value));
}

// Unwraps tensor to target dtype if it's a wrapped number.
at::Tensor UnwrapNumber(const at::Tensor& tensor, at::ScalarType dtype);

// Checks whether a c10::optional<Tensor> is defined.
inline bool IsDefined(const c10::optional<at::Tensor>& tensor) {
  return tensor.has_value() && tensor.value().defined();
}

}  // namespace torch_xla

namespace torch {
namespace lazy {
// Adapters that provide torch::lazy Hash functions for xla types
torch::lazy::hash_t Hash(const xla::Shape& shape);

template <typename T>
torch::lazy::hash_t Hash(absl::Span<const T> values) {
  return torch::lazy::ContainerHash(values);
}

// When specializing Hash(T) also specialize MHash(T, ...) since
// torch::lazy::MHash template won't be aware of the Hash(T) here
template <typename T, typename... Targs>
hash_t MHash(absl::Span<const T> value, Targs... Fargs) {
  return HashCombine(Hash(value), MHash(Fargs...));
}

}  // namespace lazy
}  // namespace torch
