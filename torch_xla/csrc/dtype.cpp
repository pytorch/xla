#include "torch_xla/csrc/device.h"
#include "torch_xla/csrc/runtime/debug_macros.h"
#include "torch_xla/csrc/runtime/sys_util.h"
#include "xla/shape.h"

namespace torch_xla {

namespace {

bool ShouldUseBF16() {
  bool use_bf16 = runtime::sys_util::GetEnvBool("XLA_USE_BF16", false);
  if (use_bf16) {
    std::cout
        << "XLA_USE_BF16 will be deprecated after the 2.5 release, please "
           "convert your model to bf16 directly\n";
    TF_LOG(INFO) << "Using BF16 data type for floating point values";
  }
  return use_bf16;
}

bool ShouldDowncastToBF16() {
  bool downcast_bf16 =
      runtime::sys_util::GetEnvBool("XLA_DOWNCAST_BF16", false);
  if (downcast_bf16) {
    std::cout
        << "XLA_DOWNCAST_BF16 will be deprecated after the 2.5 release, please "
           "downcast your model directly\n";
    TF_LOG(INFO) << "Downcasting floating point values, F64->F32, F32->BF16";
  }
  return downcast_bf16;
}

bool UseBF16() {
  static bool use_bf16 = ShouldUseBF16();
  return use_bf16;
}

bool DowncastBF16() {
  static bool downcast_bf16 = ShouldDowncastToBF16();
  return downcast_bf16;
}

}  // namespace

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
      if (UseBF16()) {
        return xla::PrimitiveType::BF16;
      }
      if (DowncastBF16() || CheckNeuronDevice(hw_type)) {
        return xla::PrimitiveType::F32;
      }
      return xla::PrimitiveType::F64;
    case xla::PrimitiveType::F32:
      return UseBF16() || DowncastBF16() ? xla::PrimitiveType::BF16
                                         : xla::PrimitiveType::F32;
    case xla::PrimitiveType::U16:
      return CheckNeuronDevice(hw_type) ? xla::PrimitiveType::U32
                                        : xla::PrimitiveType::U16;
    case xla::PrimitiveType::S16:
      return CheckNeuronDevice(hw_type) ? xla::PrimitiveType::S32
                                        : xla::PrimitiveType::S16;
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
      return UseBF16() || DowncastBF16() ? at::ScalarType::Float
                                         : at::ScalarType::BFloat16;
    case at::ScalarType::Half:
      return at::ScalarType::Half;
    case at::ScalarType::Float:
      return DowncastBF16() ? at::ScalarType::Double : at::ScalarType::Float;
    default:
      return scalar_type;
  }
}

}  // namespace torch_xla
