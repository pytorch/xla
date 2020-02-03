#include "torch_xla/csrc/convert_ops.h"

#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/tensor_util.h"

namespace torch_xla {
namespace {

xla::XlaOp ExplicitBooleanConvert(xla::XlaOp op, xla::PrimitiveType from) {
  xla::XlaOp zero = xla::Zero(op.builder(), from);
  return xla::Ne(op, zero);
}

}  // namespace

xla::XlaOp ConvertTo(xla::XlaOp op, xla::PrimitiveType from,
                     xla::PrimitiveType to, const Device* device) {
  if (from == to) {
    return op;
  }
  if (GetDeviceOrCurrent(device).hw_type != DeviceType::TPU) {
    return xla::ConvertElementType(op, to);
  }
  switch (from) {
    case xla::PrimitiveType::PRED:
    case xla::PrimitiveType::S8:
    case xla::PrimitiveType::U8:
    case xla::PrimitiveType::S16:
    case xla::PrimitiveType::U16:
    case xla::PrimitiveType::S32:
    case xla::PrimitiveType::U32:
    case xla::PrimitiveType::BF16:
    case xla::PrimitiveType::F32:
      return xla::ConvertElementType(op, to);
    case xla::PrimitiveType::S64:
    case xla::PrimitiveType::U64: {
      switch (to) {
        case xla::PrimitiveType::PRED:
          return ExplicitBooleanConvert(op, from);
        default:
          return xla::ConvertElementType(op, to);
      }
      break;
    }
    default:
      XLA_ERROR() << "Unsupported XLA type " << from;
  }
}

xla::XlaOp ConvertToNumeric(xla::XlaOp op, xla::PrimitiveType from) {
  Device xla_device = GetCurrentDevice();
  return from != xla::PrimitiveType::PRED
             ? op
             : ConvertTo(
                   op, from,
                   GetDevicePrimitiveType(xla::PrimitiveType::U8, &xla_device),
                   &xla_device);
}

xla::XlaOp ConvertToNumeric(xla::XlaOp op) {
  return ConvertToNumeric(op, XlaHelpers::TypeOfXlaOp(op));
}

xla::XlaOp CastToScalarType(xla::XlaOp input,
                            c10::optional<at::ScalarType> dtype) {
  if (dtype) {
    return ConvertTo(input, XlaHelpers::TypeOfXlaOp(input),
                     MakeXlaPrimitiveType(*dtype, /*device=*/nullptr),
                     /*device=*/nullptr);
  } else {
    return ConvertToNumeric(input, XlaHelpers::TypeOfXlaOp(input));
  }
}

}  // namespace torch_xla
