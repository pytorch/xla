#include "torch_xla/csrc/ops/linear_interpolation.h"

#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/xla_ops.h"

namespace torch_xla {

LinearInterpolation::LinearInterpolation(const torch::lazy::Value& value,
                                         const torch::lazy::Value& new_value,
                                         double alpha)
    : XlaNode(xla_moving_average, {value, new_value}, GetXlaShape(value),
              /*num_outputs=*/1, torch::lazy::MHash(alpha)),
      alpha_(alpha) {}

torch::lazy::NodePtr LinearInterpolation::Clone(
    torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<LinearInterpolation>(operands.at(0),
                                                    operands.at(1), alpha_);
}

XlaOpVector LinearInterpolation::Lower(LoweringContext* loctx) const {
  xla::XlaOp value = loctx->GetOutputOp(operand(0));
  xla::XlaOp new_value = loctx->GetOutputOp(operand(1));
  return ReturnOp(XlaHelpers::LinearInterpolation(value, new_value, alpha_),
                  loctx);
}

std::string LinearInterpolation::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", alpha=" << alpha_;
  return ss.str();
}

}  // namespace torch_xla
