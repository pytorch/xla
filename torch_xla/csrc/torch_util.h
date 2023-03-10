#ifndef XLA_TORCH_XLA_CSRC_TORCH_UTIL_H_
#define XLA_TORCH_XLA_CSRC_TORCH_UTIL_H_

#include <ATen/ATen.h>
#include <ATen/FunctionalTensorWrapper.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Optional.h>
#include <torch/csrc/lazy/core/dynamic_ir.h>
#include <torch/csrc/lazy/core/hash.h>
#include <torch/csrc/lazy/core/tensor.h>
#include <torch/csrc/lazy/core/util.h>

#include "tensorflow/compiler/xla/shape.h"
#include "third_party/xla_client/debug_macros.h"

namespace torch_xla {

// Unpack SymInt objects into their building blocks
struct SymIntElements {
 public:
  SymIntElements(c10::SymInt& size) { AddSymIntNodeElements(size); }
  SymIntElements(c10::SymIntArrayRef& size) {
    std::vector<c10::SymInt> _sizes = torch::lazy::ToVector<c10::SymInt>(size);
    for (auto& _size : _sizes) {
      AddSymIntNodeElements(_size);
    }
  }
  std::vector<torch::lazy::NodePtr> GetSizeNodes() const { return size_nodes_; }
  std::vector<int64_t> GetUpperBounds() const { return upper_bounds_; }
  std::vector<bool> GetDynamicDims() const { return dynamic_dims_; }
  torch::lazy::NodePtr GetSizeNode(size_t index) const {
    return size_nodes_[index];
  }
  void SetUpperBound(int64_t index, int64_t upper_bound) {
    XLA_CHECK_GT(upper_bounds_.size(), index);
    upper_bounds_[index] = upper_bound;
  }

 private:
  void AddSymIntNodeElements(c10::SymInt& size);
  // Only the symbolic symint will have a size_nodes, static symint
  // will have a nullptr in this vector.
  std::vector<torch::lazy::NodePtr> size_nodes_;
  std::vector<int64_t> upper_bounds_;
  std::vector<bool> dynamic_dims_;
};

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

// Wraps tensor to functional tensor if XLA_DISABLE_FUNCTIONALIZATION is false
// or not set. For unwrapping, `torch::lazy::maybe_unwrap_functional()` will
// only unwrap tensors that are functional. So, nothing needs to be done there.
at::Tensor MaybeWrapTensorToFunctional(const at::Tensor& tensor);

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

#endif  // XLA_TORCH_XLA_CSRC_TORCH_UTIL_H_