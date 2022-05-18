#include "torch_xla/csrc/ops/leaky_relu.h"

#include "torch_xla/csrc/elementwise.h"
#include "torch_xla/csrc/lowering_context.h"

namespace torch_xla {

LeakyRelu::LeakyRelu(const torch::lazy::Value& input, double negative_slope)
    : XlaNode(torch::lazy::OpKind(at::aten::leaky_relu), {input},
              GetXlaShape(input),
              /*num_outputs=*/1, torch::lazy::MHash(negative_slope)),
      negative_slope_(negative_slope) {}

torch::lazy::NodePtr LeakyRelu::Clone(torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<LeakyRelu>(operands.at(0), negative_slope_);
}

XlaOpVector LeakyRelu::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp output = BuildLeakyRelu(input, negative_slope_);
  return ReturnOp(output, loctx);
}

std::string LeakyRelu::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", negative_slope=" << negative_slope_;
  return ss.str();
}

}  // namespace torch_xla
