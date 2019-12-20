#pragma once

#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "torch_xla/csrc/device.h"

namespace torch_xla {

// Creates a minor-to-major layout from given dimensions. The dynamic_dimensions
// slice should be either empty, or of the same size as dimensions.
xla::Shape MakeTorchTensorLayout(
    tensorflow::gtl::ArraySlice<const xla::int64> dimensions,
    tensorflow::gtl::ArraySlice<const bool> dynamic_dimensions,
    xla::PrimitiveType type);

// Create an XLA shape with the given dimensions and type, suitable to be used
// in the specified device type. The type of device can affect the choice of the
// XLA layout. The dynamic_dimensions slice should be either empty, or of the
// same size as dimensions.
xla::Shape MakeArrayShapeFromDimensions(
    tensorflow::gtl::ArraySlice<const xla::int64> dimensions,
    tensorflow::gtl::ArraySlice<const bool> dynamic_dimensions,
    xla::PrimitiveType type, DeviceType device_type);

}  // namespace torch_xla
