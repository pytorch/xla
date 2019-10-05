#include "torch_xla/csrc/ops/arithmetic_ir_ops.h"

#include <memory>

#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"
#include "torch_xla/csrc/ops/ops.h"

namespace torch_xla {
namespace ir {

NodePtr operator+(const Value& node1, const Value& node2) {
  auto shape_fn = [&](tensorflow::gtl::ArraySlice<const xla::XlaOp> operands)
      -> xla::XlaOp {
    return XlaHelpers::PromotedAdd(operands[0], operands[1]);
  };

  auto lower_fn = [](const Node& node, LoweringContext* loctx) -> XlaOpVector {
    xla::XlaOp op0 = loctx->GetOutputOp(node.operand(0));
    xla::XlaOp op1 = loctx->GetOutputOp(node.operand(1));
    return node.ReturnOp(XlaHelpers::PromotedAdd(op0, op1), loctx);
  };
  return ops::GenericOp(
      OpKind(at::aten::add), OpList{node1, node2},
      [&]() {
        return ops::InferOutputShape({node1.shape(), node2.shape()}, shape_fn);
      },
      std::move(lower_fn));
}

NodePtr operator-(const Value& node1, const Value& node2) {
  auto shape_fn = [&](tensorflow::gtl::ArraySlice<const xla::XlaOp> operands)
      -> xla::XlaOp {
    return XlaHelpers::PromotedSub(operands[0], operands[1]);
  };

  auto lower_fn = [](const Node& node, LoweringContext* loctx) -> XlaOpVector {
    xla::XlaOp op0 = loctx->GetOutputOp(node.operand(0));
    xla::XlaOp op1 = loctx->GetOutputOp(node.operand(1));
    return node.ReturnOp(XlaHelpers::PromotedSub(op0, op1), loctx);
  };
  return ops::GenericOp(
      OpKind(at::aten::sub), OpList{node1, node2},
      [&]() {
        return ops::InferOutputShape({node1.shape(), node2.shape()}, shape_fn);
      },
      std::move(lower_fn));
}

NodePtr operator*(const Value& node1, const Value& node2) {
  auto shape_fn = [&](tensorflow::gtl::ArraySlice<const xla::XlaOp> operands)
      -> xla::XlaOp {
    return XlaHelpers::PromotedMul(operands[0], operands[1]);
  };

  auto lower_fn = [](const Node& node, LoweringContext* loctx) -> XlaOpVector {
    xla::XlaOp op0 = loctx->GetOutputOp(node.operand(0));
    xla::XlaOp op1 = loctx->GetOutputOp(node.operand(1));
    return node.ReturnOp(XlaHelpers::PromotedMul(op0, op1), loctx);
  };
  return ops::GenericOp(
      OpKind(at::aten::mul), OpList{node1, node2},
      [&]() {
        return ops::InferOutputShape({node1.shape(), node2.shape()}, shape_fn);
      },
      std::move(lower_fn));
}

NodePtr operator/(const Value& node1, const Value& node2) {
  auto shape_fn = [&](tensorflow::gtl::ArraySlice<const xla::XlaOp> operands)
      -> xla::XlaOp {
    return XlaHelpers::PromotedDiv(operands[0], operands[1]);
  };

  auto lower_fn = [](const Node& node, LoweringContext* loctx) -> XlaOpVector {
    xla::XlaOp op0 = loctx->GetOutputOp(node.operand(0));
    xla::XlaOp op1 = loctx->GetOutputOp(node.operand(1));
    return node.ReturnOp(XlaHelpers::PromotedDiv(op0, op1), loctx);
  };
  return ops::GenericOp(
      OpKind(at::aten::div), OpList{node1, node2},
      [&]() {
        return ops::InferOutputShape({node1.shape(), node2.shape()}, shape_fn);
      },
      std::move(lower_fn));
}
}  // namespace ir
}  // namespace torch_xla
