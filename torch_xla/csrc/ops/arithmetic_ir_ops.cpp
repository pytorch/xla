#include "arithmetic_ir_ops.h"

#include <memory>

#include "helpers.h"
#include "lowering_context.h"
#include "ops/ops.h"

namespace torch_xla {
namespace ir {

NodeOperand operator+(const NodeOperand& node1, const NodeOperand& node2) {
  auto lower_fn = [](const Node& node, LoweringContext* loctx) -> XlaOpVector {
    xla::XlaOp op0 = loctx->GetOutputOp(node.operand(0));
    xla::XlaOp op1 = loctx->GetOutputOp(node.operand(1));
    return node.ReturnOp(XlaHelpers::PromotedAdd(op0, op1), loctx);
  };
  return ops::GenericOp(
      OpKind(at::aten::add), OpList{node1, node2},
      XlaHelpers::GetPromotedShape(node1.shape(), node2.shape()),
      std::move(lower_fn));
}

NodeOperand operator-(const NodeOperand& node1, const NodeOperand& node2) {
  auto lower_fn = [](const Node& node, LoweringContext* loctx) -> XlaOpVector {
    xla::XlaOp op0 = loctx->GetOutputOp(node.operand(0));
    xla::XlaOp op1 = loctx->GetOutputOp(node.operand(1));
    return node.ReturnOp(XlaHelpers::PromotedSub(op0, op1), loctx);
  };
  return ops::GenericOp(
      OpKind(at::aten::sub), OpList{node1, node2},
      XlaHelpers::GetPromotedShape(node1.shape(), node2.shape()),
      std::move(lower_fn));
}

NodeOperand operator*(const NodeOperand& node1, const NodeOperand& node2) {
  auto lower_fn = [](const Node& node, LoweringContext* loctx) -> XlaOpVector {
    xla::XlaOp op0 = loctx->GetOutputOp(node.operand(0));
    xla::XlaOp op1 = loctx->GetOutputOp(node.operand(1));
    return node.ReturnOp(XlaHelpers::PromotedMul(op0, op1), loctx);
  };
  return ops::GenericOp(
      OpKind(at::aten::mul), OpList{node1, node2},
      XlaHelpers::GetPromotedShape(node1.shape(), node2.shape()),
      std::move(lower_fn));
}

NodeOperand operator/(const NodeOperand& node1, const NodeOperand& node2) {
  auto lower_fn = [](const Node& node, LoweringContext* loctx) -> XlaOpVector {
    xla::XlaOp op0 = loctx->GetOutputOp(node.operand(0));
    xla::XlaOp op1 = loctx->GetOutputOp(node.operand(1));
    return node.ReturnOp(XlaHelpers::PromotedDiv(op0, op1), loctx);
  };
  return ops::GenericOp(
      OpKind(at::aten::div), OpList{node1, node2},
      XlaHelpers::GetPromotedShape(node1.shape(), node2.shape()),
      std::move(lower_fn));
}

}  // namespace ir
}  // namespace torch_xla
