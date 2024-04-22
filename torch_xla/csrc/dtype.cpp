#include "torch_xla/csrc/device.h"
#include "torch_xla/csrc/runtime/debug_macros.h"
#include "torch_xla/csrc/runtime/sys_util.h"
#include "xla/shape.h"

namespace torch_xla {

namespace {

bool ShouldUseBF16() {
  bool use_bf16 = runtime::sys_util::GetEnvBool("XLA_USE_BF16", false);
  if (use_bf16) {
    TF_LOG(INFO) << "Using BF16 data type for floating point values";
  }
  return use_bf16;
}

bool ShouldUseF16() {
  bool use_fp16 = runtime::sys_util::GetEnvBool("XLA_USE_FP16", false);
  if (use_fp16) {
    TF_LOG(INFO) << "Using F16 data type for floating point values";
  }
  return use_fp16;
}

bool ShouldDowncastToBF16() {
  bool downcast_bf16 =
      runtime::sys_util::GetEnvBool("XLA_DOWNCAST_BF16", false);
  if (downcast_bf16) {
    TF_LOG(INFO) << "Downcasting floating point values, F64->F32, F32->BF16";
  }
  return downcast_bf16;
}

bool ShouldDowncastToF16() {
  bool downcast_fp16 =
      runtime::sys_util::GetEnvBool("XLA_DOWNCAST_FP16", false);
  if (downcast_fp16) {
    TF_LOG(INFO) << "Downcasting floating point values, F64->F32, F32->FP16";
  }
  return downcast_fp16;
}

bool ShouldUse32BitLong() {
  bool use_32bit_long =
      runtime::sys_util::GetEnvBool("XLA_USE_32BIT_LONG", false);
  if (use_32bit_long) {
    TF_LOG(INFO) << "Using 32bit integers for kLong values";
  }
  return use_32bit_long;
}

bool UseBF16() {
  static bool use_bf16 = ShouldUseBF16();
  return use_bf16;
}

bool UseF16() {
  static bool use_fp16 = ShouldUseF16();
  return use_fp16;
}

bool DowncastBF16() {
  static bool downcast_bf16 = ShouldDowncastToBF16();
  return downcast_bf16;
}

bool DowncastF16() {
  static bool downcast_fp16 = ShouldDowncastToF16();
  return downcast_fp16;
}

bool Use32BitLong() {
  static bool use_32bit_long = ShouldUse32BitLong();
  return use_32bit_long;
}

bool IsTpuDevice(XlaDeviceType hw_type) {
  static bool spmd_device_is_tpu =
      (hw_type == XlaDeviceType::SPMD) &&
      // HACK: find a better way to decide if SPMD is actually a TPU without
      // accessing the runtime.
      runtime::sys_util::GetEnvString("PJRT_DEVICE", "") == "TPU";
  return (hw_type == XlaDeviceType::TPU) || spmd_device_is_tpu;
}

}  // namespace

at::ScalarType TorchTypeFromXlaType(xla::PrimitiveType xla_type) {
  switch (xla_type) {
    case xla::PrimitiveType::BF16:
      return at::ScalarType::BFloat16;
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
    case at::ScalarType::Bool:
      return xla::PrimitiveType::PRED;
    case at::ScalarType::Byte:
      return xla::PrimitiveType::U8;
    case at::ScalarType::Char:
      return xla::PrimitiveType::S8;
    case at::ScalarType::Short:
      return xla::PrimitiveType::S16;
    case at::ScalarType::Int:
      return xla::PrimitiveType::S32;
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
      if (UseF16()) {
        return xla::PrimitiveType::F16;
      }
      if (UseBF16()) {
        return xla::PrimitiveType::BF16;
      }
      if (DowncastBF16() || DowncastF16() || IsTpuDevice(hw_type) ||
          hw_type == XlaDeviceType::NEURON) {
        return xla::PrimitiveType::F32;
      }
      return xla::PrimitiveType::F64;
    case xla::PrimitiveType::F32:
      if (UseF16() || DowncastF16()) {
        return xla::PrimitiveType::F16;
      }
      return UseBF16() || DowncastBF16() ? xla::PrimitiveType::BF16
                                         : xla::PrimitiveType::F32;
    case xla::PrimitiveType::U16:
      return !IsTpuDevice(hw_type) && hw_type != XlaDeviceType::NEURON
                 ? xla::PrimitiveType::U16
                 : xla::PrimitiveType::U32;
    case xla::PrimitiveType::S16:
      return !IsTpuDevice(hw_type) && hw_type != XlaDeviceType::NEURON
                 ? xla::PrimitiveType::S16
                 : xla::PrimitiveType::S32;
    case xla::PrimitiveType::S64:
      return Use32BitLong() ? xla::PrimitiveType::S32 : xla::PrimitiveType::S64;
    case xla::PrimitiveType::U64:
      return Use32BitLong() ? xla::PrimitiveType::U32 : xla::PrimitiveType::U64;
    case xla::PrimitiveType::C128:
      return !IsTpuDevice(hw_type) ? xla::PrimitiveType::C128
                                   : xla::PrimitiveType::C64;
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
      return UseF16() || DowncastF16() ? at::ScalarType::Float
                                       : at::ScalarType::Half;
    case at::ScalarType::Float:
      return DowncastBF16() || DowncastF16() ? at::ScalarType::Double
                                             : at::ScalarType::Float;
    default:
      return scalar_type;
  }
}

}  // namespace torch_xla
