#include "torch_xla/csrc/ops/native_batch_norm_backward.h"

#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/batch_norm.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(const Value& grad_out, const Value& input,
                           const Value& weight, const Value& save_mean,
                           const Value& save_invstd) {
  auto lower_for_shape_fn =
      [](tensorflow::gtl::ArraySlice<const xla::XlaOp> operands) -> xla::XlaOp {
    XLA_CHECK_EQ(operands.size(), 5);
    BatchNormGrads xla_outputs = BuildBatchNormBackward(
        operands[0], operands[1], operands[2], operands[3], operands[4], 0);
    return xla::Tuple(operands[0].builder(),
                      {xla_outputs.grad_input, xla_outputs.grad_weight,
                       xla_outputs.grad_bias});
  };
  return InferOutputShape({grad_out.shape(), input.shape(), weight.shape(),
                           save_mean.shape(), save_invstd.shape()},
                          lower_for_shape_fn);
}

}  // namespace

NativeBatchNormBackward::NativeBatchNormBackward(
    const Value& grad_out, const Value& input, const Value& weight,
    const Value& save_mean, const Value& save_invstd, double eps)
    : Node(ir::OpKind(at::aten::native_batch_norm_backward),
           {grad_out, input, weight, save_mean, save_invstd},
           [&]() {
             return NodeOutputShape(grad_out, input, weight, save_mean,
                                    save_invstd);
           },
           /*num_outputs=*/3, xla::util::MHash(eps)),
      eps_(eps) {}

NodePtr NativeBatchNormBackward::Clone(OpList operands) const {
  return MakeNode<NativeBatchNormBackward>(operands.at(0), operands.at(1),
                                           operands.at(2), operands.at(3),
                                           operands.at(4), eps_);
}

XlaOpVector NativeBatchNormBackward::Lower(LoweringContext* loctx) const {
  xla::XlaOp grad_out = loctx->GetOutputOp(operand(0));
  xla::XlaOp input = loctx->GetOutputOp(operand(1));
  xla::XlaOp weight = loctx->GetOutputOp(operand(2));
  xla::XlaOp save_mean = loctx->GetOutputOp(operand(3));
  xla::XlaOp save_invstd = loctx->GetOutputOp(operand(4));
  BatchNormGrads grads = BuildBatchNormBackward(grad_out, input, weight,
                                                save_mean, save_invstd, eps_);
  return ReturnOps({std::move(grads.grad_input), std::move(grads.grad_weight),
                    std::move(grads.grad_bias)},
                   loctx);
}

std::string NativeBatchNormBackward::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", eps=" << eps_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
