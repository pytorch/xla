#ifndef XLA_TORCH_XLA_CSRC_DTYPE_H_
#define XLA_TORCH_XLA_CSRC_DTYPE_H_

#include "torch_xla/csrc/device.h"
#include "xla/shape.h"

namespace torch_xla {

at::ScalarType TorchTypeFromXlaType(xla::PrimitiveType xla_type);

xla::PrimitiveType XlaTypeFromTorchType(at::ScalarType scalar_type);

// TODO better names
xla::PrimitiveType GetDevicePrimitiveType(
    xla::PrimitiveType type, const torch::lazy::BackendDevice& device);

at::ScalarType GetHostScalarType(xla::PrimitiveType xla_type);

xla::PrimitiveType GetXlaTypeFromTensorType(
    at::ScalarType scalar_type, const torch::lazy::BackendDevice& device);

}

#endif  // XLA_TORCH_XLA_CSRC_DTYPE_H_
