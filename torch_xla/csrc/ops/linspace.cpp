#include "torch_xla/csrc/ops/linspace.h"

#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/xla_lower_util.h"

namespace torch_xla {

Linspace::Linspace(const torch::lazy::Value& start,
                   const torch::lazy::Value& end, int64_t steps)
    : XlaNode(
          torch::lazy::OpKind(at::aten::linspace), {start, end},
          [&]() {
            xla::PrimitiveType dtype =
                XlaHelpers::PromoteType(GetXlaShape(start).element_type(),
                                        GetXlaShape(end).element_type());
            return xla::ShapeUtil::MakeShape(dtype, {steps});
          },
          /*num_outputs=*/1, torch::lazy::MHash(steps)),
      steps_(steps) {}

torch::lazy::NodePtr Linspace::Clone(torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<Linspace>(operands.at(0), operands.at(1),
                                         steps_);
}

XlaOpVector Linspace::Lower(LoweringContext* loctx) const {
  xla::XlaOp start = loctx->GetOutputOp(operand(0));
  xla::XlaOp end = loctx->GetOutputOp(operand(1));
  return ReturnOp(BuildLinspace(loctx->device(), start, end, steps_), loctx);
}

std::string Linspace::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", steps=" << steps_;
  return ss.str();
}

}  // namespace torch_xla
