#include "torch_xla/csrc/ops/leaky_relu.h"

#include "torch_xla/csrc/elementwise.h"
#include "torch_xla/csrc/lowering_context.h"

namespace torch_xla {
namespace ir {
namespace ops {

LeakyRelu::LeakyRelu(const XlaValue& input, double negative_slope)
    : XlaNode(torch::lazy::OpKind(at::aten::leaky_relu), {input},
              input.xla_shape(),
              /*num_outputs=*/1, torch::lazy::MHash(negative_slope)),
      negative_slope_(negative_slope) {}

torch::lazy::NodePtr LeakyRelu::Clone(OpList operands) const {
  return ir::MakeNode<LeakyRelu>(operands.at(0), negative_slope_);
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

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
