#include "arithmetic_ir_ops.h"

#include <memory>

#include "helpers.h"
#include "lowering_context.h"
#include "ops/ops.h"

namespace torch_xla {
namespace ir {

NodePtr operator+(const NodePtr& node1, const NodePtr& node2) {
  auto lower_fn = [](const Node& node, LoweringContext* loctx) -> XlaOpVector {
    xla::XlaOp op0 = loctx->GetOutputOp(node.operand(0));
    xla::XlaOp op1 = loctx->GetOutputOp(node.operand(1));
    return node.ReturnOp(XlaHelpers::PromotedAdd(op0, op1), loctx);
  };
  return ops::GenericOp(
      OpKind(at::aten::mul), OpList{NodeOperand(node1), NodeOperand(node2)},
      XlaHelpers::GetPromotedShape(node1->shape(), node2->shape()),
      std::move(lower_fn));
}

NodePtr operator-(const NodePtr& node1, const NodePtr& node2) {
  auto lower_fn = [](const Node& node, LoweringContext* loctx) -> XlaOpVector {
    xla::XlaOp op0 = loctx->GetOutputOp(node.operand(0));
    xla::XlaOp op1 = loctx->GetOutputOp(node.operand(1));
    return node.ReturnOp(XlaHelpers::PromotedSub(op0, op1), loctx);
  };
  return ops::GenericOp(
      OpKind(at::aten::sub), OpList{NodeOperand(node1), NodeOperand(node2)},
      XlaHelpers::GetPromotedShape(node1->shape(), node2->shape()),
      std::move(lower_fn));
}

NodePtr operator*(const NodePtr& node1, const NodePtr& node2) {
  auto lower_fn = [](const Node& node, LoweringContext* loctx) -> XlaOpVector {
    xla::XlaOp op0 = loctx->GetOutputOp(node.operand(0));
    xla::XlaOp op1 = loctx->GetOutputOp(node.operand(1));
    return node.ReturnOp(XlaHelpers::PromotedMul(op0, op1), loctx);
  };
  return ops::GenericOp(
      OpKind(at::aten::mul), OpList{NodeOperand(node1), NodeOperand(node2)},
      XlaHelpers::GetPromotedShape(node1->shape(), node2->shape()),
      std::move(lower_fn));
}

NodePtr operator/(const NodePtr& node1, const NodePtr& node2) {
  auto lower_fn = [](const Node& node, LoweringContext* loctx) -> XlaOpVector {
    xla::XlaOp op0 = loctx->GetOutputOp(node.operand(0));
    xla::XlaOp op1 = loctx->GetOutputOp(node.operand(1));
    return node.ReturnOp(XlaHelpers::PromotedDiv(op0, op1), loctx);
  };
  return ops::GenericOp(
      OpKind(at::aten::div), OpList{NodeOperand(node1), NodeOperand(node2)},
      XlaHelpers::GetPromotedShape(node1->shape(), node2->shape()),
      std::move(lower_fn));
}

}  // namespace ir
}  // namespace torch_xla
