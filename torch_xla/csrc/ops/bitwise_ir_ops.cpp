#include "torch_xla/csrc/ops/bitwise_ir_ops.h"

#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"
#include "torch_xla/csrc/ops/ops.h"

namespace torch_xla {
namespace ir {
namespace ops {

Value BitwiseAnd(const Value& node1, const Value& node2) {
  auto lower_fn = [](const Node& node, LoweringContext* loctx) -> XlaOpVector {
    xla::XlaOp op0 = loctx->GetOutputOp(node.operand(0));
    xla::XlaOp op1 = loctx->GetOutputOp(node.operand(1));
    return node.ReturnOp(op0 & op1, loctx);
  };
  auto shape_fn = [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return operands[0] & operands[1];
  };
  XLA_CHECK_EQ(node1.shape().element_type(), node2.shape().element_type());
  return GenericOp(
      OpKind(at::aten::__and__), OpList{node1, node2},
      [&]() {
        return InferOutputShape({node1.shape(), node2.shape()}, shape_fn);
      },
      std::move(lower_fn));
}

Value BitwiseOr(const Value& node1, const Value& node2) {
  auto lower_fn = [](const Node& node, LoweringContext* loctx) -> XlaOpVector {
    xla::XlaOp op0 = loctx->GetOutputOp(node.operand(0));
    xla::XlaOp op1 = loctx->GetOutputOp(node.operand(1));
    return node.ReturnOp(op0 | op1, loctx);
  };
  auto shape_fn = [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return operands[0] | operands[1];
  };
  XLA_CHECK_EQ(node1.shape().element_type(), node2.shape().element_type());
  return GenericOp(
      OpKind(at::aten::__or__), OpList{node1, node2},
      [&]() {
        return InferOutputShape({node1.shape(), node2.shape()}, shape_fn);
      },
      std::move(lower_fn));
}

Value BitwiseXor(const Value& node1, const Value& node2) {
  auto lower_fn = [](const Node& node, LoweringContext* loctx) -> XlaOpVector {
    xla::XlaOp op0 = loctx->GetOutputOp(node.operand(0));
    xla::XlaOp op1 = loctx->GetOutputOp(node.operand(1));
    return node.ReturnOp(op0 ^ op1, loctx);
  };
  auto shape_fn = [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return operands[0] ^ operands[1];
  };
  XLA_CHECK_EQ(node1.shape().element_type(), node2.shape().element_type());
  return GenericOp(
      OpKind(at::aten::__xor__), OpList{node1, node2},
      [&]() {
        return InferOutputShape({node1.shape(), node2.shape()}, shape_fn);
      },
      std::move(lower_fn));
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
