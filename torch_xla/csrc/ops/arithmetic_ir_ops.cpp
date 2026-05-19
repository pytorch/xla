#include "torch_xla/csrc/ops/arithmetic_ir_ops.h"

#include <memory>

#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/ops.h"

namespace torch_xla {

torch::lazy::NodePtr operator+(const torch::lazy::Value& node1,
                               const torch::lazy::Value& node2) {
  auto lower_fn = [](const XlaNode& node,
                     LoweringContext* loctx) -> XlaOpVector {
    xla::XlaOp op0 = loctx->GetOutputOp(node.operand(0));
    xla::XlaOp op1 = loctx->GetOutputOp(node.operand(1));
    return node.ReturnOp(XlaHelpers::PromotedAdd(op0, op1), loctx);
  };
  return GenericOp(torch::lazy::OpKind(at::aten::add), {node1, node2},
                   XlaHelpers::GetPromotedBinaryOpShape(GetXlaShape(node1),
                                                        GetXlaShape(node2)),
                   std::move(lower_fn));
}

torch::lazy::NodePtr operator-(const torch::lazy::Value& node1,
                               const torch::lazy::Value& node2) {
  auto lower_fn = [](const XlaNode& node,
                     LoweringContext* loctx) -> XlaOpVector {
    xla::XlaOp op0 = loctx->GetOutputOp(node.operand(0));
    xla::XlaOp op1 = loctx->GetOutputOp(node.operand(1));
    return node.ReturnOp(XlaHelpers::PromotedSub(op0, op1), loctx);
  };
  return GenericOp(torch::lazy::OpKind(at::aten::sub), {node1, node2},
                   XlaHelpers::GetPromotedBinaryOpShape(GetXlaShape(node1),
                                                        GetXlaShape(node2)),
                   std::move(lower_fn));
}

torch::lazy::NodePtr operator*(const torch::lazy::Value& node1,
                               const torch::lazy::Value& node2) {
  auto lower_fn = [](const XlaNode& node,
                     LoweringContext* loctx) -> XlaOpVector {
    xla::XlaOp op0 = loctx->GetOutputOp(node.operand(0));
    xla::XlaOp op1 = loctx->GetOutputOp(node.operand(1));
    return node.ReturnOp(XlaHelpers::PromotedMul(op0, op1), loctx);
  };
  return GenericOp(torch::lazy::OpKind(at::aten::mul), {node1, node2},
                   XlaHelpers::GetPromotedBinaryOpShape(GetXlaShape(node1),
                                                        GetXlaShape(node2)),
                   std::move(lower_fn));
}

torch::lazy::NodePtr operator/(const torch::lazy::Value& node1,
                               const torch::lazy::Value& node2) {
  auto lower_fn = [](const XlaNode& node,
                     LoweringContext* loctx) -> XlaOpVector {
    xla::XlaOp op0 = loctx->GetOutputOp(node.operand(0));
    xla::XlaOp op1 = loctx->GetOutputOp(node.operand(1));
    return node.ReturnOp(XlaHelpers::PromotedDiv(op0, op1), loctx);
  };
  return GenericOp(torch::lazy::OpKind(at::aten::div), {node1, node2},
                   XlaHelpers::GetPromotedBinaryOpShape(GetXlaShape(node1),
                                                        GetXlaShape(node2)),
                   std::move(lower_fn));
}

}  // namespace torch_xla
