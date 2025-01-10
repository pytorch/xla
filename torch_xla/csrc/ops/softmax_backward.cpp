#include "torch_xla/csrc/ops/softmax_backward.h"

#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"
#include "torch_xla/csrc/runtime/debug_macros.h"
#include "torch_xla/csrc/shape_helper.h"
#include "torch_xla/csrc/softmax_builder.h"
#include "torch_xla/csrc/torch_util.h"

namespace torch_xla {

SoftmaxBackward::SoftmaxBackward(const torch::lazy::Value& grad_output,
                                 const torch::lazy::Value& output, int64_t dim)
    : XlaNode(torch::lazy::OpKind(at::aten::_softmax_backward_data),
              {grad_output, output}, GetXlaShape(grad_output),
              /*num_outputs=*/1, torch::lazy::MHash(dim)),
      dim_(dim) {}

torch::lazy::NodePtr SoftmaxBackward::Clone(
    torch::lazy::OpList operands) const {
  return torch_xla::MakeNode<SoftmaxBackward>(operands.at(0), operands.at(1),
                                              dim_);
}

XlaOpVector SoftmaxBackward::Lower(LoweringContext* loctx) const {
  xla::XlaOp grad_output = loctx->GetOutputOp(operand(0));
  xla::XlaOp output = loctx->GetOutputOp(operand(1));

  // Build computation.
  const std::string name =
      std::string(GetCompositeNamespace()) + "softmax_backward";
  const std::string attr = "{dim = " + std::to_string(dim_) + " : i64}";
  xla::XlaBuilder builder(name);
  xla::XlaOp arg_grad_output = xla::Parameter(
      &builder, 0, ShapeHelper::ShapeOfXlaOp(grad_output), "arg_grad_output");
  xla::XlaOp arg_output = xla::Parameter(
      &builder, 1, ShapeHelper::ShapeOfXlaOp(grad_output), "arg_output");
  xla::XlaOp ret = BuildSoftmaxGrad(/*grad_output=*/arg_grad_output,
                                    /*output=*/arg_output, dim_);
  xla::XlaComputation computation = ConsumeValue(builder.Build(ret));

  // Build call to computation.
  std::vector<xla::XlaOp> inputs{grad_output, output};
  xla::XlaOp grad_input =
      xla::CompositeCall(loctx->builder(), computation, inputs, name, attr);

  return ReturnOp(grad_input, loctx);
}

std::string SoftmaxBackward::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", dim=" << dim_;
  return ss.str();
}

}  // namespace torch_xla
