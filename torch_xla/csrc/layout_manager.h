#ifndef XLA_TORCH_XLA_CSRC_LAYOUT_MANAGER_H_
#define XLA_TORCH_XLA_CSRC_LAYOUT_MANAGER_H_

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/types.h"
#include "torch_xla/csrc/device.h"

namespace torch_xla {

// Creates a minor-to-major layout from given dimensions. The dynamic_dimensions
// slice should be either empty, or of the same size as dimensions.
xla::Shape MakeTorchTensorLayout(absl::Span<const int64_t> dimensions,
                                 absl::Span<const bool> dynamic_dimensions,
                                 xla::PrimitiveType type);

// Create an XLA shape with the given dimensions and type, suitable to be used
// in the specified device type. The type of device can affect the choice of the
// XLA layout. The dynamic_dimensions slice should be either empty, or of the
// same size as dimensions.
xla::Shape MakeArrayShapeFromDimensions(
    absl::Span<const int64_t> dimensions,
    absl::Span<const bool> dynamic_dimensions, xla::PrimitiveType type,
    XlaDeviceType hw_type);

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_LAYOUT_MANAGER_H_