#include "torch_xla/csrc/ops/linear_interpolation.h"

#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/xla_ops.h"

namespace torch_xla {
namespace ir {
namespace ops {

LinearInterpolation::LinearInterpolation(const Value& value,
                                         const Value& new_value, double alpha)
    : Node(xla_moving_average, {value, new_value}, value.shape(),
           /*num_outputs=*/1, xla::util::MHash(alpha)),
      alpha_(alpha) {}

NodePtr LinearInterpolation::Clone(OpList operands) const {
  return MakeNode<LinearInterpolation>(operands.at(0), operands.at(1), alpha_);
}

XlaOpVector LinearInterpolation::Lower(LoweringContext* loctx) const {
  xla::XlaOp value = loctx->GetOutputOp(operand(0));
  xla::XlaOp new_value = loctx->GetOutputOp(operand(1));
  return ReturnOp(XlaHelpers::LinearInterpolation(value, new_value, alpha_),
                  loctx);
}

std::string LinearInterpolation::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", alpha=" << alpha_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
