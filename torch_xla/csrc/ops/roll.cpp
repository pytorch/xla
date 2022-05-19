#include "torch_xla/csrc/ops/roll.h"

#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/xla_lower_util.h"
namespace torch_xla {

Roll::Roll(const torch::lazy::Value& input, std::vector<int64_t> shifts,
           std::vector<int64_t> dims)
    : XlaNode(torch::lazy::OpKind(at::aten::roll), {input}, GetXlaShape(input),
              /*num_outputs=*/1, torch::lazy::MHash(shifts, dims)),
      shifts_(std::move(shifts)),
      dims_(std::move(dims)) {}

torch::lazy::NodePtr Roll::Clone(torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<Roll>(operands.at(0), shifts_, dims_);
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

}  // namespace torch_xla
