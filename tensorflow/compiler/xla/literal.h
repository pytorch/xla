#pragma once

#include <ATen/TensorIndexing.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/TensorFactories.h>

#include <string>

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace xla {

class Literal {
 public:
  Literal() { TF_LOG(FATAL) << "Not implemented yet."; }

  explicit Literal(const Shape& shape);

  const Shape& shape() const;

  template <typename NativeT>
  absl::Span<const NativeT> data(const ShapeIndex& shape_index = {}) const {
    LTC_CHECK(shape_index.empty()) << "Sub-literals not supported yet";
    return absl::MakeConstSpan(static_cast<const NativeT*>(value_.data_ptr()),
                               value_.numel());
  }

  void* untyped_data(const ShapeIndex& shape_index = {}) {
    TF_LOG(FATAL) << "Not implemented yet.";
  }
  int64 size_bytes(const ShapeIndex& shape_index = {}) const {
    TF_LOG(FATAL) << "Not implemented yet.";
  }

  std::string ToStringWithoutShape() const {
    TF_LOG(FATAL) << "Not implemented yet.";
  }

  size_t Hash() const { TF_LOG(FATAL) << "Not implemented yet."; }

  Literal Clone() const { TF_LOG(FATAL) << "Not implemented yet."; }

  template <typename NativeT>
  void Set(absl::Span<const int64> multi_index, NativeT value) {
    if (multi_index.empty()) {
      value_.fill_(value);
      return;
    }
    auto options = at::TensorOptions().device(at::kCPU).dtype(at::kLong);
    const auto index_tensor = at::tensor(
        std::vector<int64_t>(multi_index.begin(), multi_index.end()), options);
    value_.index_put_({at::indexing::TensorIndex(index_tensor)}, value);
  }

 private:
  at::Tensor value_;
  Shape shape_;
};

template <>
inline void Literal::Set<xla::uint32>(absl::Span<const int64> multi_index,
                                      xla::uint32 value) {
  Set<int64_t>(multi_index, static_cast<int64_t>(value));
}

template <>
inline void Literal::Set<xla::uint64>(absl::Span<const int64> multi_index,
                                      xla::uint64 value) {
  TF_LOG(FATAL) << "Not implemented yet.";
}

template <>
inline void Literal::Set<xla::bfloat16>(absl::Span<const int64> multi_index,
                                        xla::bfloat16 value) {
  TF_LOG(FATAL) << "Not implemented yet.";
}

template <>
inline void Literal::Set<xla::half>(absl::Span<const int64> multi_index,
                                    xla::half value) {
  TF_LOG(FATAL) << "Not implemented yet.";
}

template <>
inline void Literal::Set<xla::complex64>(absl::Span<const int64> multi_index,
                                         xla::complex64 value) {
  TF_LOG(FATAL) << "Not implemented yet.";
}

template <>
inline void Literal::Set<xla::complex128>(absl::Span<const int64> multi_index,
                                          xla::complex128 value) {
  TF_LOG(FATAL) << "Not implemented yet.";
}

class LiteralSlice {
 public:
  LiteralSlice(const Literal& literal) : literal_(&literal) {}

  const Literal* literal() const { return literal_; }

 private:
  const Literal* literal_;
};

}  // namespace xla
