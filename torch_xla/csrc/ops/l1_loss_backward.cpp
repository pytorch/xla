#include "torch_xla/csrc/ops/l1_loss_backward.h"

#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch/csrc/lazy/core/util.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"

namespace torch_xla {
namespace {

xla::Shape NodeOutputShape(const XlaValue& grad_output, const XlaValue& input,
                           const XlaValue& target, ReductionMode reduction) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildL1LossBackward(operands[0], operands[1], operands[2],
                               reduction);
  };
  return InferOutputShape(
      {grad_output.xla_shape(), input.xla_shape(), target.xla_shape()},
      lower_for_shape_fn);
}

}  // namespace

L1LossBackward::L1LossBackward(const XlaValue& grad_output,
                               const XlaValue& input, const XlaValue& target,
                               ReductionMode reduction)
    : XlaNode(torch::lazy::OpKind(at::aten::l1_loss_backward),
              {grad_output, input, target},
              [&]() {
                return NodeOutputShape(grad_output, input, target, reduction);
              },
              /*num_outputs=*/1,
              torch::lazy::MHash(torch::lazy::GetEnumValue(reduction))),
      reduction_(reduction) {}

torch::lazy::NodePtr L1LossBackward::Clone(OpList operands) const {
  return torch::lazy::MakeNode<L1LossBackward>(operands.at(0), operands.at(1),
                                               operands.at(2), reduction_);
}

XlaOpVector L1LossBackward::Lower(LoweringContext* loctx) const {
  xla::XlaOp grad_output = loctx->GetOutputOp(operand(0));
  xla::XlaOp input = loctx->GetOutputOp(operand(1));
  xla::XlaOp target = loctx->GetOutputOp(operand(2));
  return ReturnOp(BuildL1LossBackward(grad_output, input, target, reduction_),
                  loctx);
}

std::string L1LossBackward::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString()
     << ", reduction=" << torch::lazy::GetEnumValue(reduction_);
  return ss.str();
}

}  // namespace torch_xla
