#include "torch_xla/csrc/ops/scalar.h"

#include <functional>
#include <sstream>

#include "xla/shape_util.h"
#include "xla/xla_client/debug_macros.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"

namespace torch_xla {

Scalar::Scalar(const at::Scalar& value, xla::Shape shape)
    : XlaNode(torch::lazy::OpKind(at::prim::Constant), std::move(shape),
              /*num_outputs=*/1, ScalarHash(value)),
      value_(std::move(value)) {}

Scalar::Scalar(const at::Scalar& value, xla::PrimitiveType type)
    : XlaNode(torch::lazy::OpKind(at::prim::Constant),
              xla::ShapeUtil::MakeShape(type, {}),
              /*num_outputs=*/1, ScalarHash(value)),
      value_(std::move(value)) {}

std::string Scalar::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", value=" << value_;
  return ss.str();
}

torch::lazy::NodePtr Scalar::Clone(torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<Scalar>(value_, xla_shape());
}

XlaOpVector Scalar::Lower(LoweringContext* loctx) const {
  xla::Literal literal(
      xla::ShapeUtil::MakeShape(xla_shape().element_type(), {}));
  switch (xla_shape().element_type()) {
    case xla::PrimitiveType::PRED:
      literal.Set<bool>({}, static_cast<bool>(value_.toInt()));
      break;
    case xla::PrimitiveType::S8:
      literal.Set<int8_t>({}, static_cast<int8_t>(value_.toChar()));
      break;
    case xla::PrimitiveType::U8:
      literal.Set<uint8_t>({}, static_cast<uint8_t>(value_.toByte()));
      break;
    case xla::PrimitiveType::S16:
      literal.Set<int16_t>({}, static_cast<int16_t>(value_.toShort()));
      break;
    case xla::PrimitiveType::U16:
      literal.Set<uint16_t>({}, static_cast<uint16_t>(value_.toShort()));
      break;
    case xla::PrimitiveType::S32:
      literal.Set<int32_t>({}, static_cast<int32_t>(value_.toInt()));
      break;
    case xla::PrimitiveType::U32:
      literal.Set<uint32_t>({}, static_cast<uint32_t>(value_.toInt()));
      break;
    case xla::PrimitiveType::S64:
      literal.Set<int64_t>({}, static_cast<int64_t>(value_.toLong()));
      break;
    case xla::PrimitiveType::U64:
      literal.Set<uint64_t>({}, static_cast<uint64_t>(value_.toLong()));
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
    case xla::PrimitiveType::F16:
      literal.Set<xla::half>({}, static_cast<xla::half>(value_.toDouble()));
      break;
    case xla::PrimitiveType::C64:
      literal.Set<xla::complex64>({}, xla::complex64(value_.toComplexFloat()));
      break;
    case xla::PrimitiveType::C128:
      literal.Set<xla::complex128>({},
                                   xla::complex128(value_.toComplexDouble()));
      break;
    default:
      XLA_ERROR() << "Unable to lower scalar " << &value_ << " of shape "
                  << xla_shape();
  }

  xla::XlaOp op = xla::ConstantLiteral(loctx->builder(), literal);
  if (xla_shape().rank() > 0) {
    op = xla::Broadcast(op, xla_shape().dimensions());
  }
  return ReturnOp(op, loctx);
}

torch::lazy::hash_t ScalarHash(const at::Scalar& s) {
  return s.isFloatingPoint() ? torch::lazy::Hash(s.toDouble())
                             : torch::lazy::Hash(s.toLong());
}

}  // namespace torch_xla
