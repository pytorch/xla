#include "torch_xla/csrc/ops/roll.h"

#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/xla_lower_util.h"
namespace torch_xla {
namespace ir {
namespace ops {

Roll::Roll(const XlaValue& input, std::vector<int64_t> shifts,
           std::vector<int64_t> dims)
    : XlaNode(torch::lazy::OpKind(at::aten::roll), {input}, input.xla_shape(),
              /*num_outputs=*/1, torch::lazy::MHash(shifts, dims)),
      shifts_(std::move(shifts)),
      dims_(std::move(dims)) {}

torch::lazy::NodePtr Roll::Clone(OpList operands) const {
  return ir::MakeNode<Roll>(operands.at(0), shifts_, dims_);
}

XlaOpVector Roll::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  return ReturnOp(BuildRoll(input, shifts_, dims_), loctx);
}

std::string Roll::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", shifts=(" << absl::StrJoin(shifts_, ", ")
     << ")"
     << ", dims=(" << absl::StrJoin(dims_, ", ") << ")";
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
