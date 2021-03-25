#pragma once

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"

namespace xla {

class TensorFormat {
 public:
  TensorFormat() = default;
  TensorFormat(int batch_dimension, int feature_dimension,
               absl::Span<const int64> spatial_dimensions) {
    TF_LOG(FATAL) << "Not implemented yet.";
  }
};

// Computes the max pool of 'operand'.
inline XlaOp MaxPool(XlaOp operand, absl::Span<const int64> kernel_size,
                     absl::Span<const int64> stride, Padding padding,
                     const TensorFormat& data_format) {
  TF_LOG(FATAL) << "Not implemented yet.";
}

// Computes the average pool of 'operand'.
inline XlaOp AvgPool(XlaOp operand, absl::Span<const int64> kernel_size,
                     absl::Span<const int64> stride,
                     absl::Span<const std::pair<int64, int64>> padding,
                     const TensorFormat& data_format,
                     const bool counts_include_padding) {
  TF_LOG(FATAL) << "Not implemented yet.";
}

// Computes the average pool gradient.
inline XlaOp AvgPoolGrad(
    XlaOp out_backprop, absl::Span<const int64> gradients_size,
    absl::Span<const int64> kernel_size, absl::Span<const int64> stride,
    absl::Span<const std::pair<int64, int64>> spatial_padding,
    const TensorFormat& data_format, const bool counts_include_padding) {
  TF_LOG(FATAL) << "Not implemented yet.";
}

}  // namespace xla
