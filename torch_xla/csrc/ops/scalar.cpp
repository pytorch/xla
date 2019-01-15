#include "ops/scalar.h"

#include <sstream>

#include "helpers.h"
#include "lowering_context.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"

namespace torch_xla {
namespace ir {
namespace ops {

Scalar::Scalar(double value, xla::Shape shape)
    : Node(OpKind(at::prim::Constant), {}, shape),
      value_(value) {}

Scalar::Scalar(double value, xla::PrimitiveType type)
    : Node(OpKind(at::prim::Constant), {}, xla::ShapeUtil::MakeShape(type, {})),
      value_(value) {}

std::string Scalar::ToString() const {
  std::stringstream ss;
  ss << value_;
  return ss.str();
}

XlaOpVector Scalar::Lower(LoweringContext* loctx) const {
  xla::Literal literal(xla::ShapeUtil::MakeShape(shape().element_type(), {}));
  switch (shape().element_type()) {
    case xla::S8:
      literal.Set<xla::int8>({}, static_cast<xla::int8>(value_));
      break;
    case xla::U8:
      literal.Set<xla::uint8>({}, static_cast<xla::uint8>(value_));
      break;
    case xla::S16:
      literal.Set<xla::int16>({}, static_cast<xla::int16>(value_));
      break;
    case xla::U16:
      literal.Set<xla::uint16>({}, static_cast<xla::uint16>(value_));
      break;
    case xla::S32:
      literal.Set<xla::int32>({}, static_cast<xla::int32>(value_));
      break;
    case xla::U32:
      literal.Set<xla::uint32>({}, static_cast<xla::uint32>(value_));
      break;
    case xla::S64:
      literal.Set<xla::int64>({}, static_cast<xla::int64>(value_));
      break;
    case xla::U64:
      literal.Set<xla::uint64>({}, static_cast<xla::uint64>(value_));
      break;
    case xla::F32:
      literal.Set<float>({}, static_cast<float>(value_));
      break;
    case xla::BF16:
      literal.Set<xla::bfloat16>({}, static_cast<xla::bfloat16>(value_));
      break;
    default:
      XLA_ERROR() << "Unable to lower scalar " << value_ << " of shape "
                  << shape();
  }

  xla::XlaOp op = xla::ConstantLiteral(loctx->builder(), literal);
  if (shape().rank() > 0) {
    op = xla::Broadcast(op, XlaHelpers::ShapeSizes(shape()));
  }
  return ReturnOp(op, loctx);
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
