#include "torch_xla/csrc/ops/native_batch_norm_forward.h"

#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/batch_norm.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

std::vector<xla::XlaOp> LowerBatchNorm(const xla::XlaOp& input,
                                       const xla::XlaOp& weight,
                                       const xla::XlaOp& bias,
                                       const xla::XlaOp& running_mean,
                                       const xla::XlaOp& running_var,
                                       bool training, double eps) {
  std::vector<xla::XlaOp> values;
  if (training) {
    BatchNormOutput batch_norm_output =
        BuildBatchNormTraining(input, weight, bias, eps);
    values.push_back(std::move(batch_norm_output.output));
    values.push_back(std::move(batch_norm_output.batch_mean));
    values.push_back(std::move(batch_norm_output.batch_variance));
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

xla::Shape NodeOutputShape(const Value& input, const Value& weight,
                           const Value& bias, const Value& running_mean,
                           const Value& running_var, bool training) {
  auto lower_for_shape_fn =
      [training](tensorflow::gtl::ArraySlice<const xla::XlaOp> operands)
      -> xla::XlaOp {
    std::vector<xla::XlaOp> values =
        LowerBatchNorm(operands[0], operands[1], operands[2], operands[3],
                       operands[4], training, 0.5);
    return xla::Tuple(operands[0].builder(), values);
  };
  return InferOutputShape({input->shape(), weight->shape(), bias->shape(),
                           running_mean->shape(), running_var->shape()},
                          lower_for_shape_fn);
}

}  // namespace

NativeBatchNormForward::NativeBatchNormForward(const Value& input,
                                               const Value& weight,
                                               const Value& bias,
                                               const Value& running_mean,
                                               const Value& running_var,
                                               bool training, double eps)
    : Node(ir::OpKind(at::aten::native_batch_norm),
           {input, weight, bias, running_mean, running_var},
           [&]() {
             return NodeOutputShape(input, weight, bias, running_mean,
                                    running_var, training);
           },
           /*num_outputs=*/4, xla::util::MHash(training, eps)),
      training_(training),
      eps_(eps) {}

NodePtr NativeBatchNormForward::Clone(OpList operands) const {
  return MakeNode<NativeBatchNormForward>(operands.at(0), operands.at(1),
                                          operands.at(2), operands.at(3),
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
  ss << Node::ToString() << ", training=" << training_ << ", eps=" << eps_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
