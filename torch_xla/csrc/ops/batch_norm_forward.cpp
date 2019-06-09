#include "torch_xla/csrc/ops/batch_norm_forward.h"

#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/batch_norm.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"

namespace torch_xla {
namespace ir {
namespace ops {

BatchNormForward::BatchNormForward(const Value& input, const Value& weight,
                                   const Value& bias, double momentum,
                                   double eps)
    : Node(ir::OpKind(at::aten::batch_norm), {input, weight, bias},
           input.shape(),
           /*num_outputs=*/1, xla::util::MHash(momentum, eps)),
      momentum_(momentum),
      eps_(eps) {}

NodePtr BatchNormForward::Clone(OpList operands) const {
  return MakeNode<BatchNormForward>(operands.at(0), operands.at(1),
                                    operands.at(2), momentum_, eps_);
}

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
