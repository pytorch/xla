#include "torch_xla/csrc/ops/linspace.h"

#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/xla_lower_util.h"

namespace torch_xla {
namespace ir {
namespace ops {

Linspace::Linspace(const Value& start, const Value& end, int64_t steps)
    : Node(torch::lazy::OpKind(at::aten::linspace), {start, end},
           [&]() {
             xla::PrimitiveType dtype = XlaHelpers::PromoteType(
                 start.shape().element_type(), end.shape().element_type());
             return xla::ShapeUtil::MakeShape(dtype, {steps});
           },
           /*num_outputs=*/1, torch::lazy::MHash(steps)),
      steps_(steps) {}

NodePtr Linspace::Clone(OpList operands) const {
  return MakeNode<Linspace>(operands.at(0), operands.at(1), steps_);
}

XlaOpVector Linspace::Lower(LoweringContext* loctx) const {
  xla::XlaOp start = loctx->GetOutputOp(operand(0));
  xla::XlaOp end = loctx->GetOutputOp(operand(1));
  return ReturnOp(BuildLinspace(loctx->device(), start, end, steps_), loctx);
}

std::string Linspace::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", steps=" << steps_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
