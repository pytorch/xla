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

xla::Shape NodeOutputShape(const Value& input, const Value& weight,
                           const Value& bias) {
  auto lower_for_shape_fn =
      [](tensorflow::gtl::ArraySlice<const xla::XlaOp> operands) -> xla::XlaOp {
    XLA_CHECK_EQ(operands.size(), 3);
    BatchNormOutput xla_outputs =
        BuildBatchNorm(operands[0], operands[1], operands[2], 0);
    return xla::Tuple(operands[0].builder(),
                      {xla_outputs.output, xla_outputs.save_mean,
                       xla_outputs.save_invstd_eps});
  };
  return InferOutputShape({input->shape(), weight->shape(), bias->shape()},
                          lower_for_shape_fn);
}

}  // namespace

NativeBatchNormForward::NativeBatchNormForward(const Value& input,
                                               const Value& weight,
                                               const Value& bias,
                                               double momentum, double eps)
    : Node(
          ir::OpKind(at::aten::native_batch_norm), {input, weight, bias},
          [&]() { return NodeOutputShape(input, weight, bias); },
          /*num_outputs=*/3, xla::util::MHash(momentum, eps)),
      momentum_(momentum),
      eps_(eps) {}

XlaOpVector NativeBatchNormForward::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp weight = loctx->GetOutputOp(operand(1));
  xla::XlaOp bias = loctx->GetOutputOp(operand(2));
  BatchNormOutput batch_norm_output = BuildBatchNorm(input, weight, bias, eps_);
  return ReturnOps({std::move(batch_norm_output.output),
                    std::move(batch_norm_output.save_mean),
                    std::move(batch_norm_output.save_invstd_eps)},
                   loctx);
}

std::string NativeBatchNormForward::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", momentum=" << momentum_ << ", eps=" << eps_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
