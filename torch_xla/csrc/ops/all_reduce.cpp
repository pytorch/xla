#include "torch_xla/csrc/ops/all_reduce.h"

#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch/csrc/lazy/core/util.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/xla_ops.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(absl::Span<const XlaValue> operands,
                           const XlaValue& token) {
  std::vector<xla::Shape> tuple_shapes;
  tuple_shapes.reserve(operands.size() + 1);
  for (auto& operand : operands) {
    tuple_shapes.push_back(operand.xla_shape());
  }
  tuple_shapes.push_back(token.xla_shape());
  return xla::ShapeUtil::MakeTupleShape(tuple_shapes);
}

std::vector<XlaValue> GetOperandList(absl::Span<const XlaValue> operands,
                                     const XlaValue& token) {
  std::vector<XlaValue> operand_list(operands.begin(), operands.end());
  operand_list.push_back(token);
  return operand_list;
}

}  // namespace

AllReduce::AllReduce(AllReduceType reduce_type,
                     absl::Span<const XlaValue> operands, const XlaValue& token,
                     double scale, std::vector<std::vector<int64_t>> groups,
                     bool pin_layout)
    : XlaNode(xla_cross_replica_sum, GetOperandList(operands, token),
              [&]() { return NodeOutputShape(operands, token); },
              /*num_outputs=*/operands.size() + 1,
              torch::lazy::MHash(torch::lazy::GetEnumValue(reduce_type), scale,
                                 groups, pin_layout)),
      reduce_type_(reduce_type),
      scale_(scale),
      groups_(std::move(groups)),
      pin_layout_(pin_layout) {}

torch::lazy::NodePtr AllReduce::Clone(OpList operands) const {
  std::vector<XlaValue> operand_list(operands.begin(), operands.end() - 1);
  return ir::MakeNode<AllReduce>(reduce_type_, operand_list, operands.back(),
                                 scale_, groups_, pin_layout_);
}

XlaOpVector AllReduce::Lower(LoweringContext* loctx) const {
  auto& operand_list = operands();
  std::vector<xla::XlaOp> inputs;
  inputs.reserve(operand_list.size());
  for (size_t i = 0; i + 1 < operand_list.size(); ++i) {
    inputs.push_back(loctx->GetOutputOp(operand_list[i]));
  }
  xla::XlaOp token = loctx->GetOutputOp(operand_list.back());
  return ReturnOps(
      BuildAllReduce(reduce_type_, inputs, token, scale_, groups_, pin_layout_),
      loctx);
}

std::string AllReduce::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString()
     << ", reduce_type=" << torch::lazy::GetEnumValue(reduce_type_)
     << ", scale=" << scale_ << ", pin_layout=" << pin_layout_ << ", groups=(";
  for (size_t i = 0; i < groups_.size(); ++i) {
    ss << (i == 0 ? "(" : ",(");
    ss << absl::StrJoin(groups_[i], ", ") << ")";
  }
  ss << ")";
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
