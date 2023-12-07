#ifndef XLA_TORCH_XLA_CSRC_DTYPE_H_
#define XLA_TORCH_XLA_CSRC_DTYPE_H_

#include "torch_xla/csrc/device.h"
#include "xla/shape.h"

namespace torch_xla {

at::ScalarType TorchTypeFromXlaType(xla::PrimitiveType xla_type);

xla::PrimitiveType XlaTypeFromTorchType(at::ScalarType scalar_type);

// Downcast type to be compatible with device if necessary.
xla::PrimitiveType MaybeDowncastToXlaDeviceType(
    xla::PrimitiveType type, const torch::lazy::BackendDevice& device);

xla::PrimitiveType MaybeDowncastToXlaDeviceType(
    at::ScalarType scalar_type, const torch::lazy::BackendDevice& device);

// Upcast type to original PyTorch type.
at::ScalarType MaybeUpcastToHostTorchType(xla::PrimitiveType xla_type);

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_DTYPE_H_
