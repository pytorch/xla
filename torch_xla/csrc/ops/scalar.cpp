#include "torch_xla/csrc/ops/scalar.h"

#include <functional>
#include <sstream>

#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"

namespace torch_xla {
namespace ir {
namespace ops {

Scalar::Scalar(at::Scalar value, xla::Shape shape)
    : Node(OpKind(at::prim::Constant), std::move(shape), /*num_outputs=*/1,
           ScalarHash(value)),
      value_(std::move(value)) {}

Scalar::Scalar(at::Scalar value, xla::PrimitiveType type)
    : Node(OpKind(at::prim::Constant), xla::ShapeUtil::MakeShape(type, {}),
           /*num_outputs=*/1, ScalarHash(value)),
      value_(std::move(value)) {}

std::string Scalar::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", value=" << value_;
  return ss.str();
}

NodePtr Scalar::Clone(OpList operands) const {
  return MakeNode<Scalar>(value_, shape());
}

XlaOpVector Scalar::Lower(LoweringContext* loctx) const {
  xla::Literal literal(xla::ShapeUtil::MakeShape(shape().element_type(), {}));
  switch (shape().element_type()) {
    case xla::PrimitiveType::PRED:
      literal.Set<bool>({}, static_cast<bool>(value_.toInt()));
      break;
    case xla::PrimitiveType::S8:
      literal.Set<xla::int8>({}, static_cast<xla::int8>(value_.toChar()));
      break;
    case xla::PrimitiveType::U8:
      literal.Set<xla::uint8>({}, static_cast<xla::uint8>(value_.toByte()));
      break;
    case xla::PrimitiveType::S16:
      literal.Set<xla::int16>({}, static_cast<xla::int16>(value_.toShort()));
      break;
    case xla::PrimitiveType::U16:
      literal.Set<xla::uint16>({}, static_cast<xla::uint16>(value_.toShort()));
      break;
    case xla::PrimitiveType::S32:
      literal.Set<xla::int32>({}, static_cast<xla::int32>(value_.toInt()));
      break;
    case xla::PrimitiveType::U32:
      literal.Set<xla::uint32>({}, static_cast<xla::uint32>(value_.toInt()));
      break;
    case xla::PrimitiveType::S64:
      literal.Set<xla::int64>({}, static_cast<xla::int64>(value_.toLong()));
      break;
    case xla::PrimitiveType::U64:
      literal.Set<xla::uint64>({}, static_cast<xla::uint64>(value_.toLong()));
      break;
    case xla::PrimitiveType::F32:
      literal.Set<float>({}, static_cast<float>(value_.toDouble()));
      break;
    case xla::PrimitiveType::F64:
      literal.Set<double>({}, value_.toDouble());
      break;
    case xla::PrimitiveType::BF16:
      literal.Set<xla::bfloat16>({},
                                 static_cast<xla::bfloat16>(value_.toDouble()));
      break;
    case xla::PrimitiveType::C64:
      literal.Set<xla::complex64>({}, xla::complex64(value_.toComplexFloat()));
      break;
    case xla::PrimitiveType::C128:
      literal.Set<xla::complex128>({},
                                   xla::complex128(value_.toComplexDouble()));
      break;
    default:
      XLA_ERROR() << "Unable to lower scalar " << value_ << " of shape "
                  << shape();
  }

  xla::XlaOp op = xla::ConstantLiteral(loctx->builder(), literal);
  if (shape().rank() > 0) {
    op = xla::Broadcast(op, shape().dimensions());
  }
  return ReturnOp(op, loctx);
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
