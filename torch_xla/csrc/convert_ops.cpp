#include "torch_xla/csrc/convert_ops.h"

#include <climits>

#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/tensor_util.h"

namespace torch_xla {
namespace {

xla::XlaOp ExplicitBooleanConvert(xla::XlaOp op, xla::PrimitiveType from) {
  xla::XlaOp zero = xla::Zero(op.builder(), from);
  return xla::Ne(op, zero);
}

xla::XlaOp CreateRawMask(xla::XlaOp op, xla::PrimitiveType type, int64_t size,
                         int64_t narrow_size) {
  uint64_t mask_value =
      (static_cast<uint64_t>(1) << narrow_size * CHAR_BIT) - 1;
  xla::XlaOp mask = XlaHelpers::ScalarValue(mask_value, type, op.builder());
  if (xla::primitive_util::IsSignedIntegralType(type)) {
    // Sign extend the truncation mask.
    xla::XlaOp shift = XlaHelpers::ScalarValue<int32_t>(
        (size - narrow_size) * CHAR_BIT, op.builder());
    mask = (mask << shift) >> shift;
  }
  return mask;
}

xla::XlaOp ConvertData(xla::XlaOp op, xla::PrimitiveType type,
                       xla::PrimitiveType narrow_type) {
  if (!xla::primitive_util::IsIntegralType(type) ||
      !xla::primitive_util::IsIntegralType(narrow_type)) {
    return op;
  }
  int64_t size = xla::ShapeUtil::ByteSizeOfPrimitiveType(type);
  int64_t narrow_size = xla::ShapeUtil::ByteSizeOfPrimitiveType(narrow_type);
  XLA_CHECK_GE(size, narrow_size);
  if (size == narrow_size) {
    return op;
  }
  xla::XlaOp mask = CreateRawMask(op, type, size, narrow_size);
  return op & mask;
}

}  // namespace

xla::XlaOp ConvertTo(xla::XlaOp op, xla::PrimitiveType from,
                     xla::PrimitiveType to,
                     const torch::lazy::BackendDevice* device) {
  if (from == to) {
    return op;
  }
  XlaDeviceType hw_type =
      static_cast<XlaDeviceType>(GetDeviceOrCurrent(device).type());
  if (hw_type != XlaDeviceType::TPU) {
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

xla::XlaOp ConvertToRaw(xla::XlaOp op, xla::PrimitiveType from,
                        xla::PrimitiveType raw_from, xla::PrimitiveType to,
                        xla::PrimitiveType raw_to,
                        const torch::lazy::BackendDevice* device) {
  if (from != raw_from) {
    op = ConvertData(op, from, raw_from);
  }
  xla::XlaOp result = ConvertTo(op, from, to, device);
  return to == raw_to ? result : ConvertData(result, to, raw_to);
}

xla::XlaOp ConvertToNumeric(xla::XlaOp op, xla::PrimitiveType from) {
  if (from == xla::PrimitiveType::PRED) {
    torch::lazy::BackendDevice xla_device = GetCurrentDevice();
    op = ConvertTo(op, from,
                   GetDevicePrimitiveType(xla::PrimitiveType::U8, &xla_device),
                   &xla_device);
  }
  return op;
}

xla::XlaOp ConvertToNumeric(xla::XlaOp op) {
  return ConvertToNumeric(op, XlaHelpers::TypeOfXlaOp(op));
}

xla::XlaOp CastToScalarType(xla::XlaOp input,
                            c10::optional<at::ScalarType> dtype) {
  if (dtype) {
    torch::lazy::BackendDevice xla_device = GetCurrentDevice();
    return ConvertTo(input, XlaHelpers::TypeOfXlaOp(input),
                     MakeXlaPrimitiveType(*dtype, &xla_device), &xla_device);
  }
  return ConvertToNumeric(input, XlaHelpers::TypeOfXlaOp(input));
}

xla::XlaOp MaybeConvertTo(xla::XlaOp input, xla::PrimitiveType type) {
  return XlaHelpers::TypeOfXlaOp(input) != type
             ? xla::ConvertElementType(input, type)
             : input;
}

}  // namespace torch_xla
