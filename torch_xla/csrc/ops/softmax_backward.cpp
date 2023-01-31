#include "torch_xla/csrc/ops/softmax_backward.h"

#include "xla/xla_client/debug_macros.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"
#include "torch_xla/csrc/softmax_builder.h"

namespace torch_xla {

SoftmaxBackward::SoftmaxBackward(const torch::lazy::Value& grad_output,
                                 const torch::lazy::Value& output, int64_t dim)
    : XlaNode(torch::lazy::OpKind(at::aten::_softmax_backward_data),
              {grad_output, output}, GetXlaShape(grad_output),
              /*num_outputs=*/1, torch::lazy::MHash(dim)),
      dim_(dim) {}

torch::lazy::NodePtr SoftmaxBackward::Clone(
    torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<SoftmaxBackward>(operands.at(0), operands.at(1),
                                                dim_);
}

XlaOpVector SoftmaxBackward::Lower(LoweringContext* loctx) const {
  xla::XlaOp grad_output = loctx->GetOutputOp(operand(0));
  xla::XlaOp output = loctx->GetOutputOp(operand(1));
  xla::XlaOp grad_input =
      BuildSoftmaxGrad(/*grad_output=*/grad_output, /*output=*/output, dim_);
  return ReturnOp(grad_input, loctx);
}

std::string SoftmaxBackward::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", dim=" << dim_;
  return ss.str();
}

}  // namespace torch_xla
