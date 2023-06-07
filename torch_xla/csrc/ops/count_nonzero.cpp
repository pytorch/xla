#include "torch_xla/csrc/ops/count_nonzero.h"

#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/xla_lower_util.h"

namespace torch_xla {
namespace {

xla::Shape NodeOutputShape(const torch::lazy::Value& input,
                           std::vector<int64_t> dims) {
  xla::Shape input_shape = GetXlaShape(input);
  std::vector<int64_t> dimensions;
  if (dims.empty()) {
    return xla::ShapeUtil::MakeShape(input_shape.element_type(), dimensions);
  }

  std::unordered_set<int64_t> dims_set(dims.begin(), dims.end());
  for (int64_t i = 0; i < input_shape.rank(); i++) {
    if (dims_set.find(i) != dims_set.end()) {
      dimensions.push_back(i);
    }
  }
  return xla::ShapeUtil::MakeShape(input_shape.element_type(), dimensions);
}

}  // namespace

CountNonzero::CountNonzero(const torch::lazy::Value& input,
                           std::vector<int64_t> dims)
    : XlaNode(torch::lazy::OpKind(at::aten::count_nonzero), {input},
              [&]() { return NodeOutputShape(input, dims); },
              /*num_outputs=*/1, torch::lazy::MHash(dims)),
      dims_(dims) {}

torch::lazy::NodePtr CountNonzero::Clone(torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<CountNonzero>(operands.at(0), dims_);
}

XlaOpVector CountNonzero::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  return ReturnOp(BuildCountNonzero(input, dims_), loctx);
}

std::string CountNonzero::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", dims=" << dims_;
  return ss.str();
}

}  // namespace torch_xla
