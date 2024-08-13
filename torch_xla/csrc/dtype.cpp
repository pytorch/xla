#include "torch_xla/csrc/device.h"
#include "torch_xla/csrc/runtime/debug_macros.h"
#include "torch_xla/csrc/runtime/sys_util.h"
#include "xla/shape.h"

namespace torch_xla {

at::ScalarType TorchTypeFromXlaType(xla::PrimitiveType xla_type) {
  switch (xla_type) {
    case xla::PrimitiveType::BF16:
      return at::ScalarType::BFloat16;
    case xla::PrimitiveType::F8E4M3FN:
      return at::ScalarType::Float8_e4m3fn;
    case xla::PrimitiveType::F8E5M2:
      return at::ScalarType::Float8_e5m2;
    case xla::PrimitiveType::F16:
      return at::ScalarType::Half;
    case xla::PrimitiveType::F32:
      return at::ScalarType::Float;
    case xla::PrimitiveType::F64:
      return at::ScalarType::Double;
    case xla::PrimitiveType::PRED:
      return at::ScalarType::Bool;
    case xla::PrimitiveType::U8:
      return at::ScalarType::Byte;
    case xla::PrimitiveType::S8:
      return at::ScalarType::Char;
    case xla::PrimitiveType::S16:
    case xla::PrimitiveType::U16:
      return at::ScalarType::Short;
    case xla::PrimitiveType::S32:
    case xla::PrimitiveType::U32:
      return at::ScalarType::Int;
    case xla::PrimitiveType::S64:
    case xla::PrimitiveType::U64:
      return at::ScalarType::Long;
    case xla::PrimitiveType::C64:
      return at::ScalarType::ComplexFloat;
    case xla::PrimitiveType::C128:
      return at::ScalarType::ComplexDouble;
    default:
      XLA_ERROR() << "XLA type not supported: " << xla_type;
  }
}

xla::PrimitiveType XlaTypeFromTorchType(at::ScalarType scalar_type) {
  switch (scalar_type) {
    case at::ScalarType::Double:
      return xla::PrimitiveType::F64;
    case at::ScalarType::Float:
      return xla::PrimitiveType::F32;
    case at::ScalarType::BFloat16:
      return xla::PrimitiveType::BF16;
    case at::ScalarType::Half:
      return xla::PrimitiveType::F16;
    case at::ScalarType::Float8_e4m3fn:
      return xla::PrimitiveType::F8E4M3FN;
    case at::ScalarType::Float8_e5m2:
      return xla::PrimitiveType::F8E5M2;
    case at::ScalarType::Bool:
      return xla::PrimitiveType::PRED;
    case at::ScalarType::Byte:
      return xla::PrimitiveType::U8;
    case at::ScalarType::Char:
      return xla::PrimitiveType::S8;
    case at::ScalarType::UInt16:
      return xla::PrimitiveType::U16;
    case at::ScalarType::Short:
      return xla::PrimitiveType::S16;
    case at::ScalarType::UInt32:
      return xla::PrimitiveType::U32;
    case at::ScalarType::Int:
      return xla::PrimitiveType::S32;
    case at::ScalarType::UInt64:
      return xla::PrimitiveType::U64;
    case at::ScalarType::Long:
      return xla::PrimitiveType::S64;
    case at::ScalarType::ComplexFloat:
      return xla::PrimitiveType::C64;
    case at::ScalarType::ComplexDouble:
      return xla::PrimitiveType::C128;
    default:
      XLA_ERROR() << "Type not supported: " << scalar_type;
  }
}

xla::PrimitiveType MaybeDowncastToXlaDeviceType(
    xla::PrimitiveType type, const torch::lazy::BackendDevice& device) {
  XlaDeviceType hw_type = static_cast<XlaDeviceType>(device.type());
  switch (type) {
    case xla::PrimitiveType::F64:
      if (hw_type == XlaDeviceType::NEURON) {
        return xla::PrimitiveType::F32;
      }
      return xla::PrimitiveType::F64;
    case xla::PrimitiveType::F32:
      return xla::PrimitiveType::F32;
    case xla::PrimitiveType::U16:
      return hw_type != XlaDeviceType::NEURON ? xla::PrimitiveType::U16
                                              : xla::PrimitiveType::U32;
    case xla::PrimitiveType::S16:
      return hw_type != XlaDeviceType::NEURON ? xla::PrimitiveType::S16
                                              : xla::PrimitiveType::S32;
    case xla::PrimitiveType::S64:
      return xla::PrimitiveType::S64;
    case xla::PrimitiveType::U64:
      return xla::PrimitiveType::U64;
    case xla::PrimitiveType::C128:
      return xla::PrimitiveType::C128;
    default:
      return type;
  }
}

xla::PrimitiveType MaybeDowncastToXlaDeviceType(
    at::ScalarType scalar_type, const torch::lazy::BackendDevice& device) {
  xla::PrimitiveType xla_type = XlaTypeFromTorchType(scalar_type);
  return MaybeDowncastToXlaDeviceType(xla_type, device);
}

at::ScalarType MaybeUpcastToHostTorchType(xla::PrimitiveType xla_type) {
  at::ScalarType scalar_type = TorchTypeFromXlaType(xla_type);
  switch (scalar_type) {
    case at::ScalarType::BFloat16:
      return at::ScalarType::BFloat16;
    case at::ScalarType::Half:
      return at::ScalarType::Half;
    case at::ScalarType::Float:
      return at::ScalarType::Float;
    default:
      return scalar_type;
  }
}

}  // namespace torch_xla
