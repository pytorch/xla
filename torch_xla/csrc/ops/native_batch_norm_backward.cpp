#include "torch_xla/csrc/ops/native_batch_norm_backward.h"

#include "third_party/xla_client/debug_macros.h"
#include "torch_xla/csrc/batch_norm.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"

namespace torch_xla {
namespace {

xla::Shape NodeOutputShape(const torch::lazy::Value& grad_out,
                           const torch::lazy::Value& input,
                           const torch::lazy::Value& weight,
                           const torch::lazy::Value& save_mean,
                           const torch::lazy::Value& save_invstd,
                           bool training) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    BatchNormGrads xla_outputs =
        BuildBatchNormBackward(operands[0], operands[1], operands[2],
                               operands[3], operands[4], training, 0.5);
    return xla::Tuple(operands[0].builder(),
                      {xla_outputs.grad_input, xla_outputs.grad_weight,
                       xla_outputs.grad_bias});
  };
  return InferOutputShape(
      {GetXlaShape(grad_out), GetXlaShape(input), GetXlaShape(weight),
       GetXlaShape(save_mean), GetXlaShape(save_invstd)},
      lower_for_shape_fn);
}

}  // namespace

NativeBatchNormBackward::NativeBatchNormBackward(
    const torch::lazy::Value& grad_out, const torch::lazy::Value& input,
    const torch::lazy::Value& weight, const torch::lazy::Value& save_mean,
    const torch::lazy::Value& save_invstd, bool training, double eps)
    : XlaNode(torch::lazy::OpKind(at::aten::native_batch_norm_backward),
              {grad_out, input, weight, save_mean, save_invstd},
              [&]() {
                return NodeOutputShape(grad_out, input, weight, save_mean,
                                       save_invstd, training);
              },
              /*num_outputs=*/3, torch::lazy::MHash(training, eps)),
      training_(training),
      eps_(eps) {}

torch::lazy::NodePtr NativeBatchNormBackward::Clone(
    torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<NativeBatchNormBackward>(
      operands.at(0), operands.at(1), operands.at(2), operands.at(3),
      operands.at(4), training_, eps_);
}

XlaOpVector NativeBatchNormBackward::Lower(LoweringContext* loctx) const {
  xla::XlaOp grad_out = loctx->GetOutputOp(operand(0));
  xla::XlaOp input = loctx->GetOutputOp(operand(1));
  xla::XlaOp weight = loctx->GetOutputOp(operand(2));
  xla::XlaOp save_mean = loctx->GetOutputOp(operand(3));
  xla::XlaOp save_invstd = loctx->GetOutputOp(operand(4));
  BatchNormGrads grads = BuildBatchNormBackward(
      grad_out, input, weight, save_mean, save_invstd, training_, eps_);
  return ReturnOps({std::move(grads.grad_input), std::move(grads.grad_weight),
                    std::move(grads.grad_bias)},
                   loctx);
}

std::string NativeBatchNormBackward::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", training=" << training_ << ", eps=" << eps_;
  return ss.str();
}

}  // namespace torch_xla
