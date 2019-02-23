#include "torch_xla/csrc/convert_ops.h"

#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "torch_xla/csrc/helpers.h"

namespace torch_xla {
namespace {

xla::XlaOp ExplicitBooleanConvert(const xla::XlaOp& op,
                                  xla::PrimitiveType from) {
  xla::XlaOp zero =
      xla::ConstantLiteral(op.builder(), xla::LiteralUtil::Zero(from));
  return xla::Ne(op, xla::Broadcast(zero, XlaHelpers::SizesOfXlaOp(op)));
}

}  // namespace

xla::XlaOp ConvertTo(const xla::XlaOp& op, xla::PrimitiveType from,
                     xla::PrimitiveType to, const Device* device) {
  if (device == nullptr) {
    device = GetDefaultDevice();
  }
  if (device->hw_type != DeviceType::TPU) {
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

xla::XlaOp ConvertToNumeric(const xla::XlaOp& op, xla::PrimitiveType from) {
  return from != xla::PrimitiveType::PRED
             ? op
             : ConvertTo(op, from, xla::PrimitiveType::U8, /*device=*/nullptr);
}

}  // namespace torch_xla
