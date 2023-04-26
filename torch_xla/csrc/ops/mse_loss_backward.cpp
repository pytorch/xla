#include "torch_xla/csrc/ops/mse_loss_backward.h"

#include <torch/csrc/lazy/core/util.h>

#include "third_party/xla_client/util.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"
#include "torch_xla/csrc/ops/mse_loss.h"
#include "torch_xla/csrc/reduction.h"

namespace torch_xla {
namespace {

xla::Shape NodeOutputShape(const torch::lazy::Value& grad_output,
                           const torch::lazy::Value& input,
                           const torch::lazy::Value& target,
                           ReductionMode reduction) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildMseLossBackward(operands[0], operands[1], operands[2],
                                reduction);
  };
  return InferOutputShape(
      {GetXlaShape(grad_output), GetXlaShape(input), GetXlaShape(target)},
      lower_for_shape_fn);
}

}  // namespace

MseLossBackward::MseLossBackward(const torch::lazy::Value& grad_output,
                                 const torch::lazy::Value& input,
                                 const torch::lazy::Value& target,
                                 ReductionMode reduction)
    : XlaNode(torch::lazy::OpKind(at::aten::mse_loss_backward),
              {grad_output, input, target},
              [&]() {
                return NodeOutputShape(grad_output, input, target, reduction);
              },
              /*num_outputs=*/1,
              torch::lazy::MHash(torch::lazy::GetEnumValue(reduction))),
      reduction_(reduction) {}

torch::lazy::NodePtr MseLossBackward::Clone(
    torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<MseLossBackward>(operands.at(0), operands.at(1),
                                                operands.at(2), reduction_);
}

XlaOpVector MseLossBackward::Lower(LoweringContext* loctx) const {
  xla::XlaOp grad_output = loctx->GetOutputOp(operand(0));
  xla::XlaOp input = loctx->GetOutputOp(operand(1));
  xla::XlaOp target = loctx->GetOutputOp(operand(2));
  return ReturnOp(BuildMseLossBackward(grad_output, input, target, reduction_),
                  loctx);
}

std::string MseLossBackward::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString()
     << ", reduction=" << torch::lazy::GetEnumValue(reduction_);
  return ss.str();
}

}  // namespace torch_xla
