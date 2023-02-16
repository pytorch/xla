#include "torch_xla/csrc/ops/native_batch_norm_forward.h"

#include "third_party/xla_client/debug_macros.h"
#include "torch_xla/csrc/batch_norm.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"

namespace torch_xla {
namespace {

std::vector<xla::XlaOp> LowerBatchNorm(xla::XlaOp input, xla::XlaOp weight,
                                       xla::XlaOp bias, xla::XlaOp running_mean,
                                       xla::XlaOp running_var, bool training,
                                       double eps) {
  std::vector<xla::XlaOp> values;
  if (training) {
    BatchNormOutput batch_norm_output =
        BuildBatchNormTraining(input, weight, bias, eps);
    values.push_back(std::move(batch_norm_output.output));
    values.push_back(std::move(batch_norm_output.batch_mean));
    values.push_back(batch_norm_output.batch_variance);
    values.push_back(
        BatchNormVarianceInvert(batch_norm_output.batch_variance, eps));
  } else {
    values.push_back(BuildBatchNormInference(input, weight, bias, running_mean,
                                             running_var, eps));
    values.push_back(running_mean);
    values.push_back(running_var);
    values.push_back(BatchNormVarianceInvert(running_var, eps));
  }
  return values;
}

xla::Shape NodeOutputShape(const torch::lazy::Value& input,
                           const torch::lazy::Value& weight,
                           const torch::lazy::Value& bias,
                           const torch::lazy::Value& running_mean,
                           const torch::lazy::Value& running_var,
                           bool training) {
  auto lower_for_shape_fn =
      [training](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    std::vector<xla::XlaOp> values =
        LowerBatchNorm(operands[0], operands[1], operands[2], operands[3],
                       operands[4], training, 0.5);
    return xla::Tuple(operands[0].builder(), values);
  };
  return InferOutputShape(
      {GetXlaShape(input), GetXlaShape(weight), GetXlaShape(bias),
       GetXlaShape(running_mean), GetXlaShape(running_var)},
      lower_for_shape_fn);
}

}  // namespace

NativeBatchNormForward::NativeBatchNormForward(
    const torch::lazy::Value& input, const torch::lazy::Value& weight,
    const torch::lazy::Value& bias, const torch::lazy::Value& running_mean,
    const torch::lazy::Value& running_var, bool training, double eps)
    : XlaNode(torch::lazy::OpKind(at::aten::native_batch_norm),
              {input, weight, bias, running_mean, running_var},
              [&]() {
                return NodeOutputShape(input, weight, bias, running_mean,
                                       running_var, training);
              },
              /*num_outputs=*/4, torch::lazy::MHash(training, eps)),
      training_(training),
      eps_(eps) {}

torch::lazy::NodePtr NativeBatchNormForward::Clone(
    torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<NativeBatchNormForward>(
      operands.at(0), operands.at(1), operands.at(2), operands.at(3),
      operands.at(4), training_, eps_);
}

XlaOpVector NativeBatchNormForward::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp weight = loctx->GetOutputOp(operand(1));
  xla::XlaOp bias = loctx->GetOutputOp(operand(2));
  xla::XlaOp running_mean = loctx->GetOutputOp(operand(3));
  xla::XlaOp running_var = loctx->GetOutputOp(operand(4));

  return ReturnOps(LowerBatchNorm(input, weight, bias, running_mean,
                                  running_var, training_, eps_),
                   loctx);
}

std::string NativeBatchNormForward::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", training=" << training_ << ", eps=" << eps_;
  return ss.str();
}

}  // namespace torch_xla
