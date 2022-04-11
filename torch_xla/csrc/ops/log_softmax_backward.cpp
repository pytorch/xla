#include "torch_xla/csrc/ops/log_softmax_backward.h"

#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"
#include "torch_xla/csrc/softmax_builder.h"

namespace torch_xla {
namespace ir {
namespace ops {

LogSoftmaxBackward::LogSoftmaxBackward(const Value& grad_output,
                                       const Value& output, int64_t dim)
    : Node(torch::lazy::OpKind(at::aten::_log_softmax_backward_data),
           {grad_output, output}, grad_output.xla_shape(),
           /*num_outputs=*/1, torch::lazy::MHash(dim)),
      dim_(dim) {}

torch::lazy::NodePtr LogSoftmaxBackward::Clone(OpList operands) const {
  return ir::MakeNode<LogSoftmaxBackward>(operands.at(0), operands.at(1), dim_);
}

XlaOpVector LogSoftmaxBackward::Lower(LoweringContext* loctx) const {
  xla::XlaOp grad_output = loctx->GetOutputOp(operand(0));
  xla::XlaOp output = loctx->GetOutputOp(operand(1));
  xla::XlaOp grad_input =
      BuildLogSoftmaxGrad(/*grad_output=*/grad_output, /*output=*/output, dim_);
  return ReturnOp(grad_input, loctx);
}

std::string LogSoftmaxBackward::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", dim=" << dim_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
