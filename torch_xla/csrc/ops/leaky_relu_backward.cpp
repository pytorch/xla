#include "torch_xla/csrc/ops/leaky_relu_backward.h"

#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/elementwise.h"
#include "torch_xla/csrc/lowering_context.h"

namespace torch_xla {
namespace ir {
namespace ops {

LeakyReluBackward::LeakyReluBackward(const Value& grad_output,
                                     const Value& input, double negative_slope)
    : Node(ir::OpKind(at::aten::leaky_relu_backward), {grad_output, input},
           input.shape(),
           /*num_outputs=*/1, xla::util::MHash(negative_slope)),
      negative_slope_(negative_slope) {}

NodePtr LeakyReluBackward::Clone(OpList operands) const {
  return MakeNode<LeakyReluBackward>(operands.at(0), operands.at(1),
                                     negative_slope_);
}

XlaOpVector LeakyReluBackward::Lower(LoweringContext* loctx) const {
  xla::XlaOp grad_output = loctx->GetOutputOp(operand(0));
  xla::XlaOp input = loctx->GetOutputOp(operand(1));
  xla::XlaOp output =
      BuildLeakyReluBackward(grad_output, input, negative_slope_);
  return ReturnOp(output, loctx);
}

std::string LeakyReluBackward::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", negative_slope=" << negative_slope_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
