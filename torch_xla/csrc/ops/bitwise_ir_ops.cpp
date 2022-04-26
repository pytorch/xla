#include "torch_xla/csrc/ops/bitwise_ir_ops.h"

#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"
#include "torch_xla/csrc/ops/ops.h"

namespace torch_xla {
namespace ir {
namespace ops {

XlaValue BitwiseAnd(const XlaValue& node1, const XlaValue& node2) {
  auto lower_fn = [](const XlaNode& node,
                     LoweringContext* loctx) -> XlaOpVector {
    xla::XlaOp op0 = loctx->GetOutputOp(node.operand(0));
    xla::XlaOp op1 = loctx->GetOutputOp(node.operand(1));
    xla::XlaOp result = XlaHelpers::PromotedBinaryOp(
        op0, op1, [](xla::XlaOp lhs, xla::XlaOp rhs) { return lhs & rhs; });
    return node.ReturnOp(result, loctx);
  };
  auto shape_fn = [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return XlaHelpers::PromotedBinaryOp(
        operands[0], operands[1],
        [](xla::XlaOp lhs, xla::XlaOp rhs) { return lhs & rhs; });
  };
  return GenericOp(torch::lazy::OpKind(at::aten::__and__), {node1, node2},
                   [&]() {
                     return InferOutputShape(
                         {node1.xla_shape(), node2.xla_shape()}, shape_fn);
                   },
                   std::move(lower_fn));
}

XlaValue BitwiseOr(const XlaValue& node1, const XlaValue& node2) {
  auto lower_fn = [](const XlaNode& node,
                     LoweringContext* loctx) -> XlaOpVector {
    xla::XlaOp op0 = loctx->GetOutputOp(node.operand(0));
    xla::XlaOp op1 = loctx->GetOutputOp(node.operand(1));
    xla::XlaOp result = XlaHelpers::PromotedBinaryOp(
        op0, op1, [](xla::XlaOp lhs, xla::XlaOp rhs) { return lhs | rhs; });
    return node.ReturnOp(result, loctx);
  };
  auto shape_fn = [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return XlaHelpers::PromotedBinaryOp(
        operands[0], operands[1],
        [](xla::XlaOp lhs, xla::XlaOp rhs) { return lhs | rhs; });
  };
  return GenericOp(torch::lazy::OpKind(at::aten::__or__), {node1, node2},
                   [&]() {
                     return InferOutputShape(
                         {node1.xla_shape(), node2.xla_shape()}, shape_fn);
                   },
                   std::move(lower_fn));
}

XlaValue BitwiseXor(const XlaValue& node1, const XlaValue& node2) {
  auto lower_fn = [](const XlaNode& node,
                     LoweringContext* loctx) -> XlaOpVector {
    xla::XlaOp op0 = loctx->GetOutputOp(node.operand(0));
    xla::XlaOp op1 = loctx->GetOutputOp(node.operand(1));
    xla::XlaOp result = XlaHelpers::PromotedBinaryOp(
        op0, op1, [](xla::XlaOp lhs, xla::XlaOp rhs) { return lhs ^ rhs; });
    return node.ReturnOp(result, loctx);
  };
  auto shape_fn = [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return XlaHelpers::PromotedBinaryOp(
        operands[0], operands[1],
        [](xla::XlaOp lhs, xla::XlaOp rhs) { return lhs ^ rhs; });
  };
  return GenericOp(torch::lazy::OpKind(at::aten::__xor__), {node1, node2},
                   [&]() {
                     return InferOutputShape(
                         {node1.xla_shape(), node2.xla_shape()}, shape_fn);
                   },
                   std::move(lower_fn));
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
