#include "ops/batch_norm_forward.h"
#include "batch_norm.h"
#include "lowering_context.h"
#include "ops/infer_output_shape.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/util.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(const Value& input, const Value& weight,
                           const Value& bias) {
  auto lower_for_shape_fn =
      [](tensorflow::gtl::ArraySlice<const xla::XlaOp> operands) -> xla::XlaOp {
    XLA_CHECK_EQ(operands.size(), 3)
        << "Unexpected number of operands: " << operands.size();
    BatchNormOutput xla_outputs =
        BuildBatchNorm(operands[0], operands[1], operands[2], 0);
    return xla_outputs.output;
  };
  return InferOutputShape({input->shape(), weight->shape(), bias->shape()},
                          lower_for_shape_fn);
}

}  // namespace

BatchNormForward::BatchNormForward(const Value& input, const Value& weight,
                                   const Value& bias, const Value& running_mean,
                                   const Value& running_var, double momentum,
                                   double eps)
    : Node(ir::OpKind(at::aten::batch_norm),
           {input, weight, bias, running_mean, running_var},
           NodeOutputShape(input, weight, bias),
           /*num_outputs=*/1, xla::util::MHash(momentum, eps)),
      momentum_(momentum),
      eps_(eps) {}

XlaOpVector BatchNormForward::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp weight = loctx->GetOutputOp(operand(1));
  xla::XlaOp bias = loctx->GetOutputOp(operand(2));
  BatchNormOutput batch_norm_output = BuildBatchNorm(input, weight, bias, eps_);
  return ReturnOp(batch_norm_output.output, loctx);
}

std::string BatchNormForward::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", momentum=" << momentum_ << ", eps=" << eps_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
