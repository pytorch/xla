#include "torch_xla/csrc/ops/count_nonzero.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/xla_lower_util.h"

namespace torch_xla {
namespace {

xla::Shape NodeOutputShape(const torch::lazy::Value& input,
                           c10::optional<int64_t> dim) {
  std::vector<int64_t> dimensions;

  xla::Shape input_shape = GetXlaShape(input);
  return xla::ShapeUtil::MakeShape(input_shape.element_type(), dimensions);
}

}  // namespace

CountNonzero::CountNonzero(const torch::lazy::Value& input,
                 c10::optional<int64_t> dim)
    : XlaNode(torch::lazy::OpKind(at::aten::count_nonzero), {input},
              [&]() { return NodeOutputShape(input, dim); },
              /*num_outputs=*/1,
              torch::lazy::MHash(dim)),
      dim_(dim){}

torch::lazy::NodePtr CountNonzero::Clone(torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<CountNonzero>(operands.at(0), dim_);
}

XlaOpVector CountNonzero::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  return ReturnOp(BuildCountNonzero(input, dim_), loctx);
}

std::string CountNonzero::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString();
  if (dim_) {
    ss << ", dim=" << *dim_;
  }
  return ss.str();
}

}  // namespace torch_xla
